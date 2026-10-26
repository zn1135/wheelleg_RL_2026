#!/usr/bin/env python3
"""Recollect all BC source classes with the same pre-step shared reference input."""
import argparse
from dataclasses import asdict,dataclass,field
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from ct_mjlab.bc import ActionDataset,load_actor,LABEL_REFERENCE,ALIGNED_LABEL_REFERENCE
from ct_mjlab.bc_preservation import migrate_wheel_actor
from ct_mjlab import reference_contract as contract
from ct_mjlab import wheel_angles
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations as observations
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS,MODEL_XML
from ct_mjlab.smp.features import FEATURE_SCHEMA,FEATURE_DIM,FEATURE_GROUPS,MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose,inverted
from ct_mjlab.teacher.standup import DONE,FAILED
from ct_mjlab.wheel_odometry import WHEEL_ODOMETRY
from check_lqr_student_closed_loop import BridgeCfg,ClosedLoopAction,digest
from check_smp_policy_source import balance_substep
from collect_lqr_student import collect_quality,QUALITY,ROOT


class AlignedAction(ClosedLoopAction):
    def process_actions(self,actions):
        reference=getattr(self._env,'_bc_reference_observer',None)
        angle_takeover=reference is None and self.cfg.source_mode=='takeover'
        if angle_takeover:
            if not hasattr(self,'balance_synchronized'):
                self.balance_synchronized=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.balance_synchronized[self._env.episode_length_buf==0]=False
            probe=self._env._source_probe
            ready=probe['live']&(probe['hold']>=500)&~self.balance_synchronized
            if ready.any():
                self.synchronize_balance(ready);self.balance_synchronized[ready]=True
        before=self.target_clipped.clone()
        super().process_actions(actions)
        self.label=self.raw_action.clone()
        self.selected=(torch.ones(self.num_envs,dtype=torch.bool,device=self.device) if self.cfg.source_mode=='teacher' else
            self._substep_ticks*self._env.physics_dt>=self.cfg.takeover_s-1e-7 if angle_takeover else
            reference.takeover_done.clone() if self.cfg.source_mode=='takeover' else torch.zeros_like(reference.takeover_done))
        self.label_eligible=(self.teacher_phase_used!=FAILED)&(self.target_clipped==before)
        if self.cfg.source_mode=='takeover':
            self.label_eligible&=self.selected
            if not angle_takeover:self.label_eligible&=self.teacher_phase_used==DONE
        HuananhuPdAction.process_actions(self,torch.where(self.selected[:,None],self.label,actions))

@dataclass(kw_only=True)
class AlignedCfg(BridgeCfg):
    class_type:type=field(default_factory=lambda:AlignedAction)
    source_mode:str='teacher'
    takeover_s:float=10.


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


@torch.inference_mode()
def collect(args):
    torch.set_num_threads(1)
    base=ActionDataset([args.base])
    angle_profile=getattr(args,'wheel_angle_slots',False)
    handover_reset=getattr(args,'wheel_angle_handover_reset',False)
    if handover_reset and not angle_profile:
        raise ValueError('Handover reset requires wheel-angle slots')
    angle_contract=wheel_angles.HANDOVER_OBSERVATION_CONTRACT if handover_reset else wheel_angles.OBSERVATION_CONTRACT
    angle_definition=wheel_angles.PROFILES[angle_contract]
    if angle_profile and args.mode not in ('teacher','takeover'):
        raise ValueError('Wheel-angle corrections require actual teacher takeover, not shadow queries')
    group_split=base.description.get('group_split') or {s:sorted({e['group_id'] for e in base.motion.splits[s]}) for s in ('train','validation')}
    pose_manifest=getattr(args,'pose_dataset',None)
    pose_data=ActionDataset([pose_manifest]) if pose_manifest is not None else base
    if pose_manifest is not None:
        assert pose_data.motion.xml_sha256==base.motion.xml_sha256
        assert all({e['group_id'] for e in pose_data.motion.splits[s]}==set(group_split[s]) for s in group_split)
    manifest=json.loads(args.base.read_text())
    records=base.motion.episodes if args.mode=='teacher' else []
    fill=getattr(args,'fill_missing_groups',None)
    if args.mode!='teacher' or fill is not None or getattr(args,'all_delay_pairs',False):
        chosen={}
        pool=pose_data.motion.splits['train'] if args.mode!='teacher' else pose_data.motion.episodes
        for e in pool:chosen.setdefault(e['group_id'],e)
        if fill is not None:
            present={e['group_id'] for e in json.loads(fill.read_text())['episodes']}
            chosen={group:e for group,e in chosen.items() if group not in present}
        records=[dict(chosen[group],leg_delay_s=leg,wheel_delay_s=wheel) for leg in (.002,.004)
            for wheel in (.002,.004) for group in sorted(chosen)]
    initial=[]
    for row in records:
        with np.load(row['path']) as d:initial.append((d['initial_qpos'].copy(),d['initial_qvel'].copy()))
    actor=None;saved=None
    if args.mode!='teacher':
        if args.checkpoint is None:raise ValueError('Query and takeover sources require a frozen learner')
        actor,saved=load_actor(args.checkpoint,args.device)
        if actor.mlp[0].in_features!=39:raise ValueError('Recollection requires an actor39 learner')
        if getattr(args,'legacy_zero_wheel',False):
            if not angle_profile or args.mode!='takeover':raise ValueError('Legacy wheel migration is only for actual wheel39 takeover')
            actor=migrate_wheel_actor(actor,saved,base.description)
        elif angle_profile and saved['dataset']['observation_contract']!=angle_contract:
            raise ValueError('Wheel-angle takeover requires the same learner observation contract')
    duration=30. if args.mode=='takeover' and not angle_profile else 20.
    if getattr(args,'duration_s',None) is not None:duration=args.duration_s
    if not np.isfinite(duration) or duration<20.:raise ValueError('Collection duration must be finite and at least20s')
    takeover_s=getattr(args,'takeover_s',10.)
    if args.mode=='takeover' and not .2<=takeover_s<duration-5.:
        raise ValueError('Takeover must follow startup and precede the final five-second assessment')
    args.output.mkdir(parents=True,exist_ok=False)
    sources={}
    files=list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[
        Path(__file__),ROOT/'scripts/agent/check_lqr_student_closed_loop.py',ROOT/'scripts/agent/check_smp_policy_source.py',
        ROOT/'scripts/agent/collect_lqr_student.py']
    for file in files:
        name=file.relative_to(ROOT);target=args.output/'snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target);sources[str(name)]=digest(file)
    metadata=dict(schema='chuanliantui_smp_episodes_v1',feature_schema=FEATURE_SCHEMA,feature_dim=FEATURE_DIM,
        feature_groups=FEATURE_GROUPS,sample_dt=.01,physics_dt=.002,action_dt=.01,motor_order=ACTUATED_JOINTS,
        xml_sha256=digest(MODEL_XML),source_sha256=sources,source_id=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        initial='ground',ground_start_hold_s=.2,schedule=[dict(time=0.,vx=0.,yaw=0.,leg_length=.30)],
        source_controller='live_lqr_vmc_through_student_pd',velocity_integral=True,inverted_recovery=False,
        action_pair_schema=contract.PAIR_SCHEMA,observation_contract=contract.OBSERVATION_CONTRACT,
        observation_noise=False,observation_delay=False,wheel_odometry=WHEEL_ODOMETRY,label_reference=contract.REFERENCE,
        group_split=group_split,observation_contract_legacy=manifest['observation_contract'],
        observation_pair_timing='Shared reference computed once before observation and reused for teacher target at the same physical state')
    if angle_profile:
        metadata.update(action_pair_schema=angle_definition['pair_schema'],observation_contract=angle_contract,
            wheel_angles=angle_definition['wheel_angles'],label_reference=ALIGNED_LABEL_REFERENCE if handover_reset else LABEL_REFERENCE,
            observation_pair_timing='Pre-step actor39 and live teacher PD target from the same physical state; no LQR observation sidecar')
        metadata.pop('wheel_odometry')
    write(args.output/'plan.json',dict(mode=args.mode,num_envs=len(records),duration_s=duration,
        pose_dataset=str(pose_manifest) if pose_manifest else None,
        pose_dataset_sha256=digest(pose_manifest) if pose_manifest else None,
        group_split=group_split,quality_thresholds={**QUALITY,'last5s_mean_abs_yaw_rate_rad_s':.02},
        checkpoint_sha256=digest(args.checkpoint) if args.checkpoint else None,takeover_s=takeover_s,
        legacy_zero_wheel=getattr(args,'legacy_zero_wheel',False),source_sha256=sources))
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    cfg.auto_reset=False;cfg.seed=42;cfg.episode_length_s=duration+1.;cfg.rewards={}
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.terminations['inverted']=TerminationTermCfg(func=inverted)
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.metrics['quality']=MetricsTermCfg(func=collect_quality,per_substep=True,reduce='last')
    cfg.actions['pd']=AlignedCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'},source_mode=args.mode,takeover_s=takeover_s)
    cfg.actions['pd'].shared_wheel_position_reference=handover_reset
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False
    if angle_profile:observations.configure_wheel_angle_observations(cfg,handover_reset=handover_reset)
    else:observations.configure_reference_observations(cfg,takeover_s=takeover_s if args.mode=='takeover' else None)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=len(records),device=args.device)
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        qpos=torch.tensor(np.array([q for q,v in initial]),device=raw.device)
        qvel=torch.tensor(np.array([v for q,v in initial]),device=raw.device)
        qpos[:,:2]=raw.scene.env_origins[:,:2]
        raw.sim.data.qpos.copy_(qpos);raw.sim.data.qvel.copy_(qvel);raw.sim.forward();action.reset()
        leg=torch.tensor([round(e['leg_delay_s']/.002) for e in records],device=raw.device)
        wheel=torch.tensor([round(e['wheel_delay_s']/.002) for e in records],device=raw.device)
        action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
        raw.command_manager.get_term('standup').ground_start[:]=True;raw.command_manager.compute(dt=0.)
        ids=torch.arange(len(records),device=raw.device);observations.reset_observations(raw,ids)
        obs=raw.observation_manager.compute(update_history=True)
        if angle_profile:
            assert obs['actor'].shape[1]==39 and obs['critic'].shape[1]==82
            assert not hasattr(raw,'_bc_reference_observer')
        raw._source_weight=float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe=dict(live=torch.ones(len(records),dtype=torch.bool,device=raw.device),
            hold=torch.zeros(len(records),dtype=torch.long,device=raw.device),
            strict_run=torch.zeros(len(records),dtype=torch.long,device=raw.device),strict_ever=torch.zeros(len(records),dtype=torch.bool,device=raw.device))
        raw._collection_probe=dict(contact=torch.zeros(len(records),dtype=torch.bool,device=raw.device),tilt=torch.zeros(len(records),device=raw.device))
        reader=MotionStateReader(raw);buffers={}
        for tick in range(round(duration/.01)):
            live=raw._source_probe['live'].clone();inputs=obs['actor'].clone()
            if handover_reset:
                angle_state=observations.get_state(raw)
                angle_values=dict(wheel_angle_origin=angle_state.wheel_angle_origin.clone(),
                    wheel_angle_handover_latched=angle_state.wheel_angle_handover_latched.clone(),
                    wheel_angle_recovered=raw.command_manager.get_term('standup').recovered.clone())
            outputs=torch.zeros(len(records),6,device=raw.device) if actor is None else actor.mlp(inputs[:,:39].contiguous())
            clip_before=action.target_clipped.clone()
            obs,_,done,_=env.step(outputs)
            clipped=action.target_clipped!=clip_before
            if angle_profile:
                origin=angle_values['wheel_angle_origin'] if handover_reset else qpos[:,action._motor_q_adr][:,[2,5]]
                expected=(action._source_q[:,[2,5]]-origin).clamp(-100,100)
                torch.testing.assert_close(inputs[:,14:16],expected,atol=2e-6,rtol=2e-6)
            else:torch.testing.assert_close(inputs[:,40:41],action.reference_error_used,atol=0,rtol=0)
            state=reader.read()
            values=dict(**state,features=reader.features(),qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,
                observation=inputs,action=action.raw_action,
                student_action=outputs,label=action.label,applied_action=action.raw_action,source_q=action._source_q,source_dq=action._source_dq,
                source_motor_q=action._source_q,source_motor_dq=action._source_dq,
                torque=action.torques,actuator_torque=raw.sim.data.actuator_force[:,action.teacher.motor_actuators],
                command=action.teacher.command,control_active=action.teacher_active_used,
                phase=action.teacher_phase_used,fault=action.teacher_fault_used,
                teacher_phase=action.teacher_phase_used,teacher_fault=action.teacher_fault_used,
                teacher_selected=action.selected,
                label_eligible=action.label_eligible&~clipped,target_clipped=clipped,
                balance_hold=raw._source_probe['hold'],alive=live,done=done.bool())
            if not angle_profile:
                values.update(reference_error=action.reference_error_used,reference_state=action.reference_state_used,
                              reference_events=action.reference_events_used)
            if handover_reset:
                values.update(angle_values)
                controller=action.teacher.controller
                values.update(lqr_position_reference_event=controller.position_reference_event,
                    lqr_position_error_used=controller.x[:,0],lqr_position_integrating=controller.position_integrating,
                    lqr_velocity_error_used=controller.x[:,1]-action.teacher.command[:,0],
                    lqr_position_contribution=controller.contributions[:,:,0])
            for key,value in values.items():buffers.setdefault(key,[]).append(value.cpu().numpy().copy())
            if tick<20:assert not values['actuator_torque'].any()
            raw._source_probe['live']&=~done.bool()
            if tick%500==0:print(f'{args.mode}: {tick+1}/{round(duration/.01)} live={int(raw._source_probe["live"].sum())}',flush=True)
            if done.any():
                raw.reset(env_ids=done.nonzero().flatten());action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
                obs=raw.observation_manager.compute(update_history=False)
        arrays={key:np.array(value) for key,value in buffers.items()};time=(np.arange(len(arrays['alive']))+1)*.01
        late=time>duration-5.;settled=time>=2.;rows=[]
        for i,e in enumerate(records):
            f=arrays['features'][:,i];q=arrays['qpos'][:,i]
            quality=dict(full=bool(arrays['alive'][:,i].all()),strict_rise=bool(raw._source_probe['strict_ever'][i]),
                final_balance=bool(raw._source_probe['hold'][i]>=500),
                leg_rms=float(np.sqrt(np.square(arrays['motor_dq'][late,i][:,[0,1,3,4]]).mean())),
                speed=float(np.linalg.norm(f[late,4:6],axis=-1).mean()),yaw=float(np.abs(f[late,9]).mean()),
                drift=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max()),
                height=float(np.abs(f[settled,0]-.33648).mean()),tilt=float(raw._collection_probe['tilt'][i]),
                body_contact=bool(raw._collection_probe['contact'][i]),finite=bool(np.isfinite(q).all()))
            reasons=[k for k in ('full','strict_rise','final_balance','finite') if not quality[k]]
            reasons += [name for name,limit in [('leg_rms',.02),('speed',.01),('yaw',.02),('drift',.02),
                ('height',.03),('tilt',float(np.deg2rad(25)))] if quality[name]>limit]
            if quality['body_contact']:reasons.append('body_contact')
            post=time>takeover_s if args.mode=='takeover' else np.ones(len(time),dtype=bool)
            if args.mode!='query' and ((arrays['teacher_phase'][post,i]==FAILED).any() or (arrays['fault'][post,i]!=0).any()):reasons.append('teacher_fault')
            if args.mode!='query' and arrays['target_clipped'][post,i].any():reasons.append('target_clipped')
            if args.mode=='takeover' and not arrays['teacher_selected'][post,i].all():reasons.append('no_supported_takeover')
            rows.append(dict(env=i,group_id=e['group_id'],pose_row=e['pose_row'],leg_delay_s=e['leg_delay_s'],wheel_delay_s=e['wheel_delay_s'],
                qualified=not reasons,rejection_reasons=reasons,quality=quality))
        qualified=np.array([row['qualified'] for row in rows])
        arrays['label_valid']=arrays['alive']&arrays['label_eligible']
        if args.mode=='takeover':arrays['label_valid']&=qualified[None,:]
        if args.mode=='teacher':
            accepted=[];rejected=[]
            for i,(e,row) in enumerate(zip(records,rows)):
                identity=hashlib.sha256(json.dumps(dict(source_episode=e['episode_id'],source_id=metadata['source_id'],
                    observation_contract=metadata['observation_contract'],leg_delay_s=e['leg_delay_s'],
                    wheel_delay_s=e['wheel_delay_s']),sort_keys=True).encode()).hexdigest()
                record=dict(row,episode_id=identity,seed=42,behavior='standup',accepted=row['qualified'],
                    schedule=metadata['schedule'],path=f'{"accepted" if row["qualified"] else "rejected"}/{identity}.npz')
                target=args.output/record['path'];target.parent.mkdir(exist_ok=True)
                np.savez_compressed(target,**{k:v[:,i] for k,v in arrays.items()},time=time,observation_time=time-.01,action_time=time-.01,
                    nominal_motor_positions=action.nominal_positions[i].cpu().numpy(),initial_qpos=qpos[i].cpu().numpy(),initial_qvel=qvel[i].cpu().numpy(),
                    metadata=np.asarray(json.dumps(dict(metadata,**record))))
                record['sha256']=digest(target);(accepted if row['qualified'] else rejected).append(record)
            write(args.output/'dataset.json',dict(metadata,episodes=accepted))
            write(args.output/'report.json',dict(complete=True,accepted=len(accepted),rejected_episodes=rejected,optimizer_updates=0))
        else:
            np.savez_compressed(args.output/'trace.npz',**arrays,time=time,observation_time=time-.01,action_time=time-.01,
                nominal_motor_positions=action.nominal_positions[0].cpu().numpy(),reference_motor_positions=action.reference_positions[0].cpu().numpy(),
                initial_qpos=qpos.cpu().numpy(),initial_qvel=qvel.cpu().numpy())
            result=dict(completed=True,correction_schema='ct_bc_corrections_v1',split='train',optimizer_updates=0,
                rows=rows,num_envs=len(rows),qualified=int(qualified.sum()),valid_correction_frames=int(arrays['label_valid'].sum()),
                teacher_queries_only=args.mode=='query',independent_student_control=args.mode=='query',heldout_used=False,
                checkpoint_sha256=digest(args.checkpoint),xml_sha256=metadata['xml_sha256'],source_sha256=sources,
                base_motion_fingerprint=None,source_base_motion_fingerprint=base.motion.fingerprint,
                observation_contract=contract.OBSERVATION_CONTRACT,wheel_odometry=WHEEL_ODOMETRY,label_reference=contract.REFERENCE,
                takeover_s=takeover_s,duration_s=duration,trace_sha256=digest(args.output/'trace.npz'),
                note='base_motion_fingerprint assigned only by a later checked assembly against recollected teacher41 manifest')
            if angle_profile:
                result.update(base_motion_fingerprint=base.motion.fingerprint,
                    observation_contract=angle_contract,wheel_angles=angle_definition['wheel_angles'],
                    label_reference=ALIGNED_LABEL_REFERENCE if handover_reset else LABEL_REFERENCE,
                    legacy_zero_wheel=getattr(args,'legacy_zero_wheel',False),
                    note='Actual teacher recovery after a frozen-student prefix; qualified training poses only; no observer or shadow labels')
                result.pop('wheel_odometry')
            write(args.output/'result.json',result)
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg))
        print(f'completed {args.mode}: accepted={int(qualified.sum())}/{len(rows)}',flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--pose-dataset',type=Path,help='Original pose manifest only; preserves declared split without admitting rejected labels')
    p.add_argument('--mode',choices=('teacher','query','takeover'),required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fill-missing-groups',type=Path,help='Teacher only: try all four nominal delay pairs for absent pose groups')
    p.add_argument('--all-delay-pairs',action='store_true',help='Teacher: sample each original pose with all four 2/4ms pairs')
    p.add_argument('--takeover-s',type=float,default=10.,help='Time of actual teacher intervention in takeover mode')
    p.add_argument('--wheel-angle-slots',action='store_true',help='Teacher or actual takeover actor39 pairs with reset-relative wheel-angle slots')
    p.add_argument('--wheel-angle-handover-reset',action='store_true',help='Clear wheel-angle slots once at the first ground-start recovery; requires --wheel-angle-slots')
    p.add_argument('--legacy-zero-wheel',action='store_true',help='Explicitly retain legacy stable actor while ignoring wheel-angle columns')
    p.add_argument('--duration-s',type=float,help='Recorded collection duration; default follows the source mode')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');collect(p.parse_args())
