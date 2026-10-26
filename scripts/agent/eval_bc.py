#!/usr/bin/env python3
"""Independent BC control and optional train-only teacher queries; no optimizer."""
import argparse
from dataclasses import dataclass,field,asdict
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from ct_mjlab.bc import ActionDataset,load_actor,LEGACY_SAMPLING
from ct_mjlab.envs import task_registry
from ct_mjlab import wheel_angles
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose,inverted
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.teacher.standup import FAILED
from ct_mjlab import reference_contract
from check_lqr_student_closed_loop import BridgeCfg,ClosedLoopAction,digest
from check_smp_policy_source import balance_substep
from collect_lqr_student import collect_quality,QUALITY


class QueryAction(ClosedLoopAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.balance_synchronized=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)

    def process_actions(self,actions):
        if hasattr(self._env,'_bc_reference_observer'):
            super().process_actions(actions)
            self.label=self.raw_action.clone()
            HuananhuPdAction.process_actions(self,actions)
            return
        self.balance_synchronized[self._env.episode_length_buf==0]=False
        probe=self._env._source_probe
        ready=probe['live'] & (probe['hold']>=500) & ~self.balance_synchronized
        if ready.any():
            self.synchronize_balance(ready)
            self.balance_synchronized[ready]=True
        super().process_actions(actions)
        self.label=self.raw_action.clone()
        # 教师只查询标签，真实机器人仍完全由学生动作经过原PD/FIFO驱动。
        HuananhuPdAction.process_actions(self,actions)


@dataclass(kw_only=True)
class QueryCfg(BridgeCfg):
    class_type:type=field(default_factory=lambda:QueryAction)


@torch.inference_mode()
def run(args):
    if args.collect_corrections and args.split!='train':
        raise ValueError('Do not turn heldout evaluation poses into training corrections')
    actor,saved=load_actor(args.checkpoint,args.device)
    data=ActionDataset([args.dataset],args.corrections,action_sampling=saved['dataset'].get('action_sampling',LEGACY_SAMPLING))
    assert saved['dataset']['fingerprint']==data.description['fingerprint']
    pose_manifest=getattr(args,'pose_dataset',None)
    pose_data=ActionDataset([pose_manifest]) if pose_manifest is not None else data
    if pose_manifest is not None:
        declared=data.description['group_split']
        assert pose_data.motion.xml_sha256==data.motion.xml_sha256
        assert all({e['group_id'] for e in pose_data.motion.splits[s]}==set(declared[s]) for s in ('train','validation'))
    chosen={}
    for e in pose_data.motion.splits[args.split]:chosen.setdefault(e['group_id'],e)
    episodes=[chosen[k] for k in sorted(chosen)]
    pairs=[(a,b) for a in (.002,.004) for b in (.002,.004)]
    count=len(episodes)*len(pairs);records=[]
    for e in episodes:
        with np.load(e['path']) as d:records.append((d['initial_qpos'].copy(),d['initial_qvel'].copy()))
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    angle_profile=data.description['observation_contract'] in wheel_angles.PROFILES
    handover_reset=data.description['observation_contract']==wheel_angles.HANDOVER_OBSERVATION_CONTRACT
    if angle_profile:
        multimode_observations.configure_wheel_angle_observations(cfg,handover_reset=handover_reset)
    elif data.observation_dim == 41:
        multimode_observations.configure_reference_observations(cfg)
    elif data.observation_dim == 40:
        multimode_observations.configure_wheel_odometry(cfg)
    cfg.auto_reset=False;cfg.seed=42
    cfg.commands['standup']=SmpCommandCfg()
    cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.terminations['inverted']=TerminationTermCfg(func=inverted)
    cfg.rewards={}
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.metrics['quality']=MetricsTermCfg(func=collect_quality,per_substep=True,reduce='last')
    if args.collect_corrections:
        cfg.actions['pd']=QueryCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'})
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False
    cfg.actions['pd'].action_delay_choices_s=(.002,.004)
    args.output.mkdir(parents=True,exist_ok=False)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=count,device=args.device)
    if args.full_precision:
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=False)
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        qpos=torch.tensor(np.asarray([q for pair in pairs for q,v in records]),device=args.device)
        qvel=torch.tensor(np.asarray([v for pair in pairs for q,v in records]),device=args.device)
        qpos[:,:2]=raw.scene.env_origins[:,:2]
        raw.sim.data.qpos.copy_(qpos);raw.sim.data.qvel.copy_(qvel);raw.sim.forward()
        action.reset()
        leg=torch.tensor([round(a/.002) for a,b in pairs],device=args.device).repeat_interleave(len(records))
        wheel=torch.tensor([round(b/.002) for a,b in pairs],device=args.device).repeat_interleave(len(records))
        action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
        raw.command_manager.get_term('standup').ground_start[:]=True
        raw.command_manager.compute(dt=0.)
        multimode_observations.reset_observations(raw,torch.arange(count,device=args.device))
        obs=raw.observation_manager.compute(update_history=True)
        if angle_profile:
            assert not hasattr(raw,'_bc_reference_observer')
            motor_actuators=torch.tensor([raw.sim.mj_model.actuator('robot/'+n+'_motor').id for n in ACTUATED_JOINTS],device=raw.device)
        raw._source_weight=float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe=dict(live=torch.ones(count,dtype=torch.bool,device=args.device),
            hold=torch.zeros(count,dtype=torch.long,device=args.device),strict_run=torch.zeros(count,dtype=torch.long,device=args.device),
            strict_ever=torch.zeros(count,dtype=torch.bool,device=args.device))
        raw._collection_probe=dict(contact=torch.zeros(count,dtype=torch.bool,device=args.device),tilt=torch.zeros(count,device=args.device))
        reader=MotionStateReader(raw)
        buffers={k:[] for k in ('qpos','qvel','motor_dq','features','observation','action','alive','done')}
        if args.collect_corrections:buffers.update(label=[],label_valid=[])
        if data.observation_dim==41:
            buffers.update(reference_error=[],reference_state=[],reference_events=[],source_q=[],source_dq=[],actuator_torque=[])
        if angle_profile:buffers.update(source_q=[],source_dq=[],actuator_torque=[])
        if handover_reset:buffers.update(wheel_angle_origin=[],wheel_angle_handover_latched=[],wheel_angle_recovered=[])
        before={k:v.clone() for k,v in actor.state_dict().items()}
        for tick in range(2000):
            live=raw._source_probe['live'].clone();inputs=obs['actor'].clone()
            if angle_profile:
                angle_values=dict(source_q=raw.sim.data.qpos[:,action._motor_q_adr].clone(),
                                  source_dq=raw.sim.data.qvel[:,action._motor_v_adr].clone())
                if handover_reset:
                    angle_state=multimode_observations.get_state(raw)
                    angle_values.update(wheel_angle_origin=angle_state.wheel_angle_origin.clone(),
                        wheel_angle_handover_latched=angle_state.wheel_angle_handover_latched.clone(),
                        wheel_angle_recovered=raw.command_manager.get_term('standup').recovered.clone())
            if data.observation_dim==41:
                reference=raw._bc_reference_observer.update()
                torch.testing.assert_close(inputs[:,40:41],reference.error,atol=0,rtol=0)
                ref_values=dict(reference_error=reference.error.clone(),reference_state=reference.state.clone(),
                    reference_events=reference.events.clone(),source_q=raw.sim.data.qpos[:,action._motor_q_adr].clone(),
                    source_dq=raw.sim.data.qvel[:,action._motor_v_adr].clone())
            outputs=actor.mlp(inputs)
            obs,_,done,_=env.step(outputs)
            state=reader.read()
            values=dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,motor_dq=state['motor_dq'],
                features=reader.features(),observation=inputs,action=outputs,alive=live,done=done.bool())
            if args.collect_corrections:
                phase=action.teacher_phase_used if data.observation_dim==41 else action.teacher.controller.standup.phase
                values.update(label=action.label,label_valid=live & (phase!=FAILED))
            if data.observation_dim==41:
                values.update(ref_values)
                values['actuator_torque']=raw.sim.data.actuator_force[:,raw._bc_reference_observer.teacher.motor_actuators]
            if angle_profile:
                values.update(angle_values)
                values['actuator_torque']=raw.sim.data.actuator_force[:,motor_actuators]
            for k,v in values.items():buffers[k].append(v.cpu().numpy().copy())
            if tick<20:
                assert not action.torques.any()
                if data.observation_dim==41 or angle_profile:assert not values['actuator_torque'].any()
            raw._source_probe['live']&=~done.bool()
            if tick%500==0:print(f't={(tick+1)*.01:.2f}s live={int(raw._source_probe["live"].sum())}/{count}',flush=True)
            if not raw._source_probe['live'].any():break
            if done.any():
                raw.reset(env_ids=done.nonzero().flatten());action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
                if data.observation_dim==41 or angle_profile:obs=raw.observation_manager.compute(update_history=False)
        arrays={k:np.asarray(v) for k,v in buffers.items()};time=(np.arange(len(arrays['alive']))+1)*.01
        rows=[]
        for index in range(count):
            length=int(arrays['alive'][:,index].sum());late=(time[:length]>15.)&(time[:length]<=20.)
            full=length==2000;root=arrays['qpos'][:length,index,:3]
            dq=arrays['motor_dq'][:length,index][:,[0,1,3,4]]
            rms=float(np.sqrt(np.square(dq[late]).mean())) if late.any() else None
            # 速度使用记录的机体系线速度，水平范数不受航向旋转影响。
            speed=float(np.linalg.norm(arrays['features'][:length,index,4:6][late],axis=-1).mean()) if late.any() else None
            yaw=float(np.abs(arrays['features'][:length,index,9][late]).mean()) if late.any() else None
            drift=float(np.linalg.norm(root[late,:2]-root[late,:2][0],axis=-1).max()) if late.any() else None
            finite=bool(np.isfinite(arrays['qpos'][:length,index]).all() and np.isfinite(arrays['action'][:length,index]).all())
            balance=bool(full and raw._source_probe['hold'][index]>=500)
            strict=bool(raw._source_probe['strict_ever'][index])
            parked=bool(full and rms<=QUALITY['last5s_leg_velocity_rms_rad_s'] and speed<=QUALITY['last5s_mean_speed_m_s'] and drift<=QUALITY['last5s_xy_excursion_m'] and yaw<=.02)
            rows.append(dict(env=index,group_id=episodes[index%len(episodes)]['group_id'],episode_id=episodes[index%len(episodes)]['episode_id'],
                leg_delay_s=pairs[index//len(episodes)][0],wheel_delay_s=pairs[index//len(episodes)][1],
                observed_s=length*.01,full_20s=full,strict_rise=strict,final_balance_v31=balance,
                last5s_leg_velocity_rms_rad_s=rms,last5s_mean_speed_m_s=speed,last5s_xy_excursion_m=drift,last5s_mean_abs_yaw_rate_rad_s=yaw,
                finite=finite,parking_and_smoothness=parked,qualified=bool(strict and balance and parked and finite)))
        for k,v in before.items():torch.testing.assert_close(actor.state_dict()[k],v,atol=0,rtol=0)
        np.savez_compressed(args.output/'trace.npz',**arrays,time=time,initial_qpos=qpos.cpu().numpy(),initial_qvel=qvel.cpu().numpy())
        report=dict(completed=True,optimizer_updates=0,split=args.split,num_envs=count,rows=rows,
            actor_updates=saved.get('total_actor_updates',saved['iteration']),
            independent_student_control=True,teacher_queries_only=args.collect_corrections,
            strict_rise=sum(r['strict_rise'] for r in rows),final_balance=sum(r['final_balance_v31'] for r in rows),
            qualified=sum(r['qualified'] for r in rows),actor_unchanged=True,
            checkpoint_sha256=digest(args.checkpoint),dataset_fingerprint=data.description['fingerprint'],
            base_motion_fingerprint=data.motion.fingerprint,
            full_precision=args.full_precision,
            xml_sha256=data.motion.xml_sha256,trace_sha256=digest(args.output/'trace.npz'),
            observation_contract=data.description['observation_contract'],
            script_sha256=digest(__file__),scope='Nominal independent actor, 2/4ms delays and original startup hold; no PPO, noise/randomization or hardware.')
        if args.collect_corrections:
            report['correction_schema']='ct_bc_corrections_v1'
            report['valid_correction_frames']=int(arrays['label_valid'].sum())
        if data.observation_dim==41:
            report.update(label_reference=reference_contract.REFERENCE,wheel_odometry=data.description['wheel_odometry'])
        if angle_profile:
            report.update(wheel_angles=data.description['wheel_angles'],label_reference=data.description['label_reference'],
                          no_LQR_reference_observer_created=not hasattr(raw,'_bc_reference_observer'))
        if pose_manifest is not None:
            report.update(evaluation_pose_manifest_sha256=digest(pose_manifest),
                          evaluation_pose_dataset_fingerprint=pose_data.motion.fingerprint,
                          evaluation_pose_manifest_scope='Initial qpos/qvel and heldout membership only; no training labels imported')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg))
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--pose-dataset',type=Path,help='Immutable original pose cohort, including poses lacking qualified teacher demonstrations')
    p.add_argument('--split',choices=('train','validation'),default='validation')
    p.add_argument('--collect-corrections',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--corrections',type=Path,action='append',default=[])
    p.add_argument('--full-precision',action='store_true',help='Diagnostic IEEE FP32 inference instead of the usual TF32 backend')
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
