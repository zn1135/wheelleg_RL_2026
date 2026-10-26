#!/usr/bin/env python3
"""Collect train-pose, zero-command teacher braking through the original student PD."""
import argparse,json,shutil
from dataclasses import asdict,dataclass,field
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from ct_mjlab.bc import ActionDataset,load_actor,LEGACY_SAMPLING,LABEL_REFERENCE
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose,inverted
from ct_mjlab.teacher.standup import DONE
from check_lqr_student_closed_loop import BridgeCfg,ClosedLoopAction,digest
from check_smp_policy_source import balance_substep
from collect_lqr_student import collect_quality,QUALITY,ROOT

TAKEOVER=10.
DURATION=30.


class BrakingAction(ClosedLoopAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.selected=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.entry_hold=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.entry_phase=torch.full_like(self.entry_hold,-1)

    def process_actions(self,actions):
        self.selected[self._env.episode_length_buf==0]=False
        switch=self._env.episode_length_buf==round(TAKEOVER/self._env.step_dt)
        probe=self._env._source_probe
        ready=switch & probe['live'] & (probe['hold']>=500)
        if ready.any():
            history=self._action_history.clone();head=self._history_head
            self.entry_hold[ready]=probe['hold'][ready]
            self.entry_phase[ready]=self.teacher.controller.standup.phase[ready]
            self.synchronize_balance(ready)
            self.selected[ready]=True
            assert torch.equal(history,self._action_history) and head==self._history_head
        before=self.target_clipped.clone()
        super().process_actions(actions)
        self.label=self.raw_action.clone()
        self.label_eligible=self.selected & (self.teacher.controller.standup.phase==DONE) & (self.target_clipped==before)
        # 前10秒由冻结学生到达实际漂移状态；此段绝不作为教师标签。
        HuananhuPdAction.process_actions(self,torch.where(self.selected[:,None],self.label,actions))


@dataclass(kw_only=True)
class BrakingCfg(BridgeCfg):
    class_type:type=field(default_factory=lambda:BrakingAction)


@torch.inference_mode()
def run(args):
    actor,saved=load_actor(args.checkpoint,args.device)
    data=ActionDataset([args.dataset],args.corrections,action_sampling=saved['dataset'].get('action_sampling',LEGACY_SAMPLING))
    assert saved['dataset']['fingerprint']==data.description['fingerprint']
    chosen={}
    for e in data.motion.splits['train']:chosen.setdefault(e['group_id'],e)
    episodes=[chosen[k] for k in sorted(chosen)]
    pairs=[(a,b) for a in (.002,.004) for b in (.002,.004)]
    initial=[]
    for e in episodes:
        with np.load(e['path']) as d:initial.append((d['initial_qpos'].copy(),d['initial_qvel'].copy()))
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(split='train',group_ids=[e['group_id'] for e in episodes],delay_pairs_s=pairs,
        takeover_s=TAKEOVER,duration_s=DURATION,quality_thresholds={**QUALITY,'last5s_mean_abs_yaw_rate_rad_s':.02},
        commands=[0.,0.,.30],checkpoint_sha256=digest(args.checkpoint),xml_sha256=digest(MODEL_XML),
        label_scope='Only teacher-controlled, unclipped DONE-stage frames of fully qualified episodes; no learner prefix or heldout pose.')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    sources={}
    paths=list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[Path(__file__),
        ROOT/'scripts/agent/check_lqr_student_closed_loop.py',ROOT/'scripts/agent/check_smp_policy_source.py',
        ROOT/'scripts/agent/collect_lqr_student.py',ROOT/'scripts/agent/eval_bc.py']
    for path in paths:
        name=str(path.relative_to(ROOT));sources[name]=digest(path)
        target=args.output/'snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    if data.observation_dim == 40:
        multimode_observations.configure_wheel_odometry(cfg)
    cfg.auto_reset=False;cfg.seed=42;cfg.episode_length_s=DURATION+1.
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.terminations['inverted']=TerminationTermCfg(func=inverted);cfg.rewards={}
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.metrics['quality']=MetricsTermCfg(func=collect_quality,per_substep=True,reduce='last')
    cfg.actions['pd']=BrakingCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'})
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False
    cfg.actions['pd'].action_delay_choices_s=(.002,.004)
    count=len(episodes)*len(pairs)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=count,device=args.device)
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        assert raw.physics_dt==.002 and raw.step_dt==.01
        qpos=torch.tensor(np.asarray([q for pair in pairs for q,v in initial]),device=args.device)
        qvel=torch.tensor(np.asarray([v for pair in pairs for q,v in initial]),device=args.device)
        qpos[:,:2]=raw.scene.env_origins[:,:2]
        raw.sim.data.qpos.copy_(qpos);raw.sim.data.qvel.copy_(qvel);raw.sim.forward();action.reset()
        leg=torch.tensor([round(a/.002) for a,b in pairs],device=args.device).repeat_interleave(len(initial))
        wheel=torch.tensor([round(b/.002) for a,b in pairs],device=args.device).repeat_interleave(len(initial))
        action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
        raw.command_manager.get_term('standup').ground_start[:]=True;raw.command_manager.compute(dt=0.)
        multimode_observations.reset_observations(raw,torch.arange(count,device=args.device))
        obs=raw.observation_manager.compute(update_history=True)
        raw._source_weight=float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe=dict(live=torch.ones(count,dtype=torch.bool,device=args.device),hold=torch.zeros(count,dtype=torch.long,device=args.device),
            strict_run=torch.zeros(count,dtype=torch.long,device=args.device),strict_ever=torch.zeros(count,dtype=torch.bool,device=args.device))
        raw._collection_probe=dict(contact=torch.zeros(count,dtype=torch.bool,device=args.device),tilt=torch.zeros(count,device=args.device))
        names=('qpos','qvel','motor_dq','features','observation','action','label','applied_action','label_eligible',
            'teacher_phase','teacher_fault','teacher_selected','teacher_command','source_q','source_dq',
            'actual_torque','balance_hold','alive','done')
        buffers={k:[] for k in names};reader=MotionStateReader(raw)
        original={k:v.clone() for k,v in actor.state_dict().items()}
        for tick in range(round(DURATION/raw.step_dt)):
            live=raw._source_probe['live'].clone();inputs=obs['actor'].clone();outputs=actor.mlp(inputs)
            obs,_,done,_=env.step(outputs);state=reader.read();c=action.teacher.controller
            values=dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,motor_dq=state['motor_dq'],features=reader.features(),
                observation=inputs,action=outputs,label=action.label,applied_action=action.raw_action,label_eligible=action.label_eligible,
                teacher_phase=c.standup.phase,teacher_fault=c.standup.fault,teacher_selected=action.selected,
                teacher_command=action.teacher.command,source_q=action._source_q,source_dq=action._source_dq,
                actual_torque=raw.sim.data.actuator_force[:,action.teacher.motor_actuators],balance_hold=raw._source_probe['hold'],
                alive=live,done=done.bool())
            for k,v in values.items():buffers[k].append(v.cpu().numpy().copy())
            if tick<20:assert not action.torques.any() and not values['actual_torque'].any()
            raw._source_probe['live']&=~done.bool()
            if tick%500==0:print(f't={(tick+1)*.01:.2f}s live={int(raw._source_probe["live"].sum())}/{count} teacher={int(action.selected.sum())}',flush=True)
        arrays={k:np.asarray(v) for k,v in buffers.items()};time=(np.arange(len(arrays['alive']))+1)*.01;rows=[]
        late=time>DURATION-5.;post=time>TAKEOVER;entry=round(TAKEOVER/.01)
        for i in range(count):
            full=bool(arrays['alive'][:,i].all() and not arrays['done'][:,i].any())
            f=arrays['features'][:,i];q=arrays['qpos'][:,i]
            rms=float(np.sqrt(np.square(arrays['motor_dq'][late,i][:,[0,1,3,4]]).mean()))
            speed=float(np.linalg.norm(f[late,4:6],axis=-1).mean());yaw=float(np.abs(f[late,9]).mean())
            drift=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max())
            checks=dict(full_30s=full,entered_on_support=bool(action.entry_hold[i]>=500),
                strict_rise=bool(raw._source_probe['strict_ever'][i]),final_balance=bool(raw._source_probe['hold'][i]>=500),
                teacher_done_after_entry=bool((arrays['teacher_phase'][post,i]==DONE).all()),
                teacher_targets_unclipped=bool(arrays['label_eligible'][post,i].all()),
                no_body_contact_after_settle=not bool(raw._collection_probe['contact'][i]),
                tilt=bool(raw._collection_probe['tilt'][i]<=QUALITY['post_settle_tilt_rad']),
                height=bool(np.abs(f[post,0]-.33648).mean()<=QUALITY['mean_height_error_m']),
                smooth=rms<=.02,stopped=speed<=.01,yaw_stopped=yaw<=.02,position_stopped=drift<=.02,
                finite=all(np.isfinite(arrays[k][:,i]).all() for k in names))
            rows.append(dict(env=i,group_id=episodes[i%len(episodes)]['group_id'],pose_row=episodes[i%len(episodes)]['pose_row'],
                leg_delay_s=pairs[i//len(episodes)][0],wheel_delay_s=pairs[i//len(episodes)][1],
                entry_hold_substeps=int(action.entry_hold[i]),old_teacher_phase=int(action.entry_phase[i]),
                entry_speed_m_s=float(np.linalg.norm(f[entry-1,4:6])),entry_yaw_rate_rad_s=float(f[entry-1,9]),
                last5s_leg_velocity_rms_rad_s=rms,last5s_mean_speed_m_s=speed,last5s_mean_abs_yaw_rate_rad_s=yaw,
                last5s_xy_excursion_m=drift,checks=checks,rejection_reasons=[k for k,v in checks.items() if not v],qualified=all(checks.values())))
        qualified=np.array([r['qualified'] for r in rows])
        arrays['label_valid']=arrays['label_eligible'] & arrays['alive'] & qualified[None,:]
        for k,v in original.items():torch.testing.assert_close(actor.state_dict()[k],v,atol=0,rtol=0)
        np.savez_compressed(args.output/'trace.npz',**arrays,time=time,observation_time=time-.01,action_time=time-.01,
            initial_qpos=qpos.cpu().numpy(),initial_qvel=qvel.cpu().numpy(),
            nominal_motor_positions=action.nominal_positions[0].cpu().numpy(),reference_motor_positions=action.reference_positions[0].cpu().numpy())
        report=dict(completed=True,correction_schema='ct_bc_corrections_v1',split='train',optimizer_updates=0,
            num_envs=count,training_pose_groups=len(episodes),takeover_s=TAKEOVER,duration_s=DURATION,rows=rows,
            qualified=int(qualified.sum()),valid_correction_frames=int(arrays['label_valid'].sum()),
            teacher_queries_only=False,independent_student_control=False,actor_unchanged=True,
            label_reference=LABEL_REFERENCE,
            observation_contract=data.description['observation_contract'],
            **({'wheel_odometry':data.description['wheel_odometry']} if data.observation_dim == 40 else {}),
            source_actor_scope='Frozen learner0..10s, actual closed-loop teacher10..30s; learner prefix never labeled.',
            base_motion_fingerprint=data.motion.fingerprint,dataset_fingerprint=data.description['fingerprint'],
            checkpoint_sha256=digest(args.checkpoint),xml_sha256=digest(MODEL_XML),source_sha256=sources,
            trace_sha256=digest(args.output/'trace.npz'),quality_thresholds=plan['quality_thresholds'],
            fifo_preserved_at_entry=True,heldout_used=False,numeric_snapshot_restoration=False)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg));print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256')},indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');run(p.parse_args())
