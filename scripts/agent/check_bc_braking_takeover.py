#!/usr/bin/env python3
"""Train-pose student/teacher braking diagnosis with live duplicate-world prefixes."""
import argparse,json,shutil
from dataclasses import asdict,dataclass,field
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from ct_mjlab.bc import ActionDataset,load_actor,LEGACY_SAMPLING
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

MODES=('student','keep-reference','relock')
TAKEOVER=10.
DURATION=30.
WHEEL_FIELDS=('pre_dq','delayed_velocity_target','nominal_pd_torque','actual_motor_torque','post_dq')


class TakeoverAction(ClosedLoopAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert env.num_envs%len(MODES)==0
        self.mode_ids=torch.arange(len(MODES),device=env.device).repeat_interleave(env.num_envs//len(MODES))
        self.selected=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.entry_hold=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.entry_phase=torch.full_like(self.entry_hold,-1)
        self.entry_before=torch.zeros(env.num_envs,2,device=env.device)
        self.entry_after=torch.zeros_like(self.entry_before)
        self.wheel_steps=torch.empty(round(DURATION/env.physics_dt),env.num_envs,len(WHEEL_FIELDS),2,device=env.device)
        self.wheel_index=0

    def process_actions(self,actions):
        probe=self._env._source_probe;c=self.teacher.controller
        switch=self._env.episode_length_buf==round(TAKEOVER/self._env.step_dt)
        ready=switch & probe['live'] & (probe['hold']>=500)
        if ready.any():
            history=self._action_history.clone();head=self._history_head
            self.entry_hold[ready]=probe['hold'][ready];self.entry_phase[ready]=c.standup.phase[ready]
            self.entry_before[ready]=torch.stack((c.pos,c.yaw_target),-1)[ready]
            # 所有分支的教师阶段都同步；仅relock改停车参考，学生分支只查询。
            keep=ready & (self.mode_ids!=2);relock=ready & (self.mode_ids==2)
            c.standup.reset(keep,standing=True)
            self.synchronize_balance(relock)
            self.entry_after[ready]=torch.stack((c.pos,c.yaw_target),-1)[ready]
            self.selected[ready]=self.mode_ids[ready]!=0
            assert torch.equal(history,self._action_history) and head==self._history_head
        before=self.target_clipped.clone()
        super().process_actions(actions)
        self.label=self.raw_action.clone();self.clipped=self.target_clipped!=before
        q=self._source_q;dq=self._source_dq;legs=self._leg_action_ids;wheels=self._wheel_action_ids
        implied=torch.zeros_like(actions)
        target=(self.nominal_positions[:,legs]+self.cfg.leg_position_scale*actions[:,legs]).clamp(
            self.reference_positions[:,legs]-3.14,self.reference_positions[:,legs]+3.14)
        implied[:,legs]=self.cfg.leg_stiffness*(target-q[:,legs])-self.cfg.leg_damping*dq[:,legs]
        implied[:,wheels]=self.cfg.wheel_stiffness*((actions[:,wheels]*self.cfg.wheel_velocity_scale).clamp(
            -self.cfg.max_wheel_velocity,self.cfg.max_wheel_velocity)-dq[:,wheels])
        self.student_requested=implied.clamp(-self._effort_limits,self._effort_limits)
        HuananhuPdAction.process_actions(self,torch.where(self.selected[:,None],self.label,actions))

    def apply_actions(self):
        super().apply_actions()
        k=self.wheel_index;w=self._wheel_action_ids
        head=(self._history_head-self.wheel_action_delay_steps)%len(self._action_history)
        delayed=self._action_history[head,self._env_ids][:,w]
        self.wheel_steps[k,:,0]=self._env.sim.data.qvel[:,self._motor_v_adr][:,w]
        self.wheel_steps[k,:,1]=(delayed*self.cfg.wheel_velocity_scale).clamp(-self.cfg.max_wheel_velocity,self.cfg.max_wheel_velocity)
        self.wheel_steps[k,:,2]=self.torques[:,w]
        self.wheel_index+=1


def capture_wheels(env):
    action=env.action_manager.get_term('pd');k=action.wheel_index-1
    action.wheel_steps[k,:,3]=env.sim.data.actuator_force[:,action.teacher.motor_actuators][:,action._wheel_action_ids]
    action.wheel_steps[k,:,4]=env.sim.data.qvel[:,action._motor_v_adr][:,action._wheel_action_ids]
    return torch.zeros(env.num_envs,device=env.device)


@dataclass(kw_only=True)
class TakeoverCfg(BridgeCfg):
    class_type:type=field(default_factory=lambda:TakeoverAction)


@torch.inference_mode()
def run(args):
    actor,saved=load_actor(args.checkpoint,args.device)
    data=ActionDataset([args.dataset],args.corrections,action_sampling=saved['dataset'].get('action_sampling',LEGACY_SAMPLING))
    assert data.description['fingerprint']==saved['dataset']['fingerprint']
    chosen={}
    for e in data.motion.splits['train']:chosen.setdefault(e['group_id'],e)
    episodes=[chosen[k] for k in sorted(chosen)];pairs=[(a,b) for a in (.002,.004) for b in (.002,.004)]
    base_count=len(episodes)*len(pairs);count=base_count*len(MODES)
    initial=[]
    for e in episodes:
        with np.load(e['path']) as d:initial.append((d['initial_qpos'].copy(),d['initial_qvel'].copy()))
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(split='train',modes=MODES,training_pose_groups=len(episodes),num_cases=base_count,num_envs=count,
        group_ids=[e['group_id'] for e in episodes],delay_pairs_s=pairs,takeover_s=TAKEOVER,duration_s=DURATION,
        commands=[0.,0.,.30],quality_thresholds={**QUALITY,'last5s_mean_abs_yaw_rate_rad_s':.02},
        prefix='Duplicate independent physics worlds start with identical diagnostic origins, XYZ and joint states; prefix numerical differences are measured, not assumed zero. No state restoration.',
        checkpoint_sha256=digest(args.checkpoint),xml_sha256=digest(MODEL_XML),optimizer_updates=0)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    sources={}
    for path in list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[Path(__file__),
        ROOT/'scripts/agent/check_lqr_student_closed_loop.py',ROOT/'scripts/agent/check_smp_policy_source.py',ROOT/'scripts/agent/collect_lqr_student.py']:
        name=str(path.relative_to(ROOT));sources[name]=digest(path);target=args.output/'snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    if data.observation_dim == 40:
        multimode_observations.configure_wheel_odometry(cfg)
    cfg.auto_reset=False;cfg.seed=42;cfg.episode_length_s=DURATION+1.;cfg.rewards={}
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.terminations['inverted']=TerminationTermCfg(func=inverted)
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.metrics['quality']=MetricsTermCfg(func=collect_quality,per_substep=True,reduce='last')
    cfg.metrics['wheel_capture']=MetricsTermCfg(func=capture_wheels,per_substep=True,reduce='last')
    cfg.actions['pd']=TakeoverCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'})
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False;cfg.actions['pd'].action_delay_choices_s=(.002,.004)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=count,device=args.device)
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        assert raw.physics_dt==.002 and raw.step_dt==.01 and not action.cfg.motor_randomization
        qpos=torch.tensor(np.asarray([q for mode in MODES for pair in pairs for q,v in initial]),device=args.device)
        qvel=torch.tensor(np.asarray([v for mode in MODES for pair in pairs for q,v in initial]),device=args.device)
        # 独立world可使用相同XYZ；避免显示平移引入分支前浮点差异，不修改生产复位。
        diagnostic_origins=raw.scene.env_origins[:base_count].clone().repeat(len(MODES),1)
        raw.scene.env_origins.copy_(diagnostic_origins)
        qpos[:,:2]=diagnostic_origins[:,:2]
        raw.sim.data.qpos.copy_(qpos);raw.sim.data.qvel.copy_(qvel);raw.sim.forward();action.reset()
        leg=torch.tensor([round(a/.002) for mode in MODES for a,b in pairs],device=args.device).repeat_interleave(len(initial))
        wheel=torch.tensor([round(b/.002) for mode in MODES for a,b in pairs],device=args.device).repeat_interleave(len(initial))
        action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
        raw.command_manager.get_term('standup').ground_start[:]=True;raw.command_manager.compute(dt=0.)
        multimode_observations.reset_observations(raw,torch.arange(count,device=args.device));obs=raw.observation_manager.compute(update_history=True)
        raw._source_weight=float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe=dict(live=torch.ones(count,dtype=torch.bool,device=args.device),hold=torch.zeros(count,dtype=torch.long,device=args.device),
            strict_run=torch.zeros(count,dtype=torch.long,device=args.device),strict_ever=torch.zeros(count,dtype=torch.bool,device=args.device))
        raw._collection_probe=dict(contact=torch.zeros(count,dtype=torch.bool,device=args.device),tilt=torch.zeros(count,device=args.device))
        names=('qpos','qvel','features','motor_dq','observation','student_action','teacher_action','applied_action',
            'teacher_requested','student_requested','teacher_x','teacher_target','wheel_contributions','wheel_unsaturated','teacher_phase',
            'teacher_fault','teacher_air','teacher_clipped','teacher_selected','source_q','source_dq','actual_torque','balance_hold','alive','done')
        buffers={k:[] for k in names};reader=MotionStateReader(raw)
        original={k:v.clone() for k,v in actor.state_dict().items()};prefix_max=0.;switch_file=None
        for tick in range(round(DURATION/.01)):
            if tick==round(TAKEOVER/.01):
                switch_file=args.output/'switch.npz'
                np.savez_compressed(switch_file,qpos=raw.sim.data.qpos.clone().cpu().numpy(),qvel=raw.sim.data.qvel.clone().cpu().numpy(),
                    observation=obs['actor'].cpu().numpy(),fifo=action._action_history.cpu().numpy(),
                    teacher_phase=action.teacher.controller.standup.phase.cpu().numpy(),balance_hold=raw._source_probe['hold'].cpu().numpy())
            live=raw._source_probe['live'].clone();inputs=obs['actor'].clone();outputs=actor.mlp(inputs)
            obs,_,done,_=env.step(outputs);state=reader.read();c=action.teacher.controller
            values=dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,features=reader.features(),motor_dq=state['motor_dq'],
                observation=inputs,student_action=outputs,teacher_action=action.label,applied_action=action.raw_action,
                teacher_requested=action.teacher.torques,student_requested=action.student_requested,teacher_x=c.x,teacher_target=c.target,
                wheel_contributions=c.contributions[:,:2,:4],wheel_unsaturated=c.contributions[:,:2].sum(-1),teacher_phase=c.standup.phase,teacher_fault=c.standup.fault,
                teacher_air=c.air.flight,teacher_clipped=action.clipped,teacher_selected=action.selected,source_q=action._source_q,source_dq=action._source_dq,
                actual_torque=raw.sim.data.actuator_force[:,action.teacher.motor_actuators],balance_hold=raw._source_probe['hold'],alive=live,done=done.bool())
            for k,v in values.items():buffers[k].append(v.cpu().numpy().copy())
            assert torch.equal(action.raw_action,torch.where(action.selected[:,None],action.label,outputs))
            if tick<20:assert not action.torques.any() and not values['actual_torque'].any()
            if tick<round(TAKEOVER/.01):
                for k in ('qpos','qvel','observation','student_action'):
                    v=values[k].reshape(len(MODES),base_count,-1)
                    prefix_max=max(prefix_max,float((v[1:]-v[0]).abs().max()))
            if tick==round(TAKEOVER/.01):switch_k=c.K.cpu().numpy().copy()
            raw._source_probe['live']&=~done.bool()
            if tick%500==0:print(f't={(tick+1)*.01:.2f}s live={int(raw._source_probe["live"].sum())}/{count} teacher={int(action.selected.sum())} prefix_max={prefix_max}',flush=True)
            if done.any():
                raw.reset(env_ids=done.nonzero().flatten())
                action.selected[done.bool()]=False
                action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
        arrays={k:np.asarray(v) for k,v in buffers.items()};time=(np.arange(len(arrays['alive']))+1)*.01;rows=[]
        late=time>25.;post=time>10.
        for i in range(count):
            full=bool(arrays['alive'][:,i].all() and not arrays['done'][:,i].any());f=arrays['features'][:,i];q=arrays['qpos'][:,i]
            rms=float(np.sqrt(np.square(arrays['motor_dq'][late,i][:,[0,1,3,4]]).mean()))
            speed=float(np.linalg.norm(f[late,4:6],axis=-1).mean());yaw=float(np.abs(f[late,9]).mean())
            drift=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max())
            balance=bool(full and raw._source_probe['hold'][i]>=500)
            checks=dict(full_30s=full,entered_on_support=bool(action.entry_hold[i]>=500),strict_rise=bool(raw._source_probe['strict_ever'][i]),
                final_balance=balance,no_body_contact=not bool(raw._collection_probe['contact'][i]),
                tilt=bool(raw._collection_probe['tilt'][i]<=QUALITY['post_settle_tilt_rad']),
                height=bool(np.abs(f[post,0]-.33648).mean()<=.03),
                smooth=rms<=.02,stopped=speed<=.01,yaw_stopped=yaw<=.02,position_stopped=drift<=.02,
                finite=all(np.isfinite(arrays[k][:,i]).all() for k in names))
            if i//base_count:
                checks['teacher_done_after_entry']=bool((arrays['teacher_phase'][post,i]==DONE).all())
                checks['teacher_targets_unclipped']=not bool(arrays['teacher_clipped'][post,i].any())
            case=i%base_count
            rows.append(dict(env=i,case=case,mode=MODES[i//base_count],group_id=episodes[case%len(episodes)]['group_id'],
                pose_row=episodes[case%len(episodes)]['pose_row'],leg_delay_s=pairs[case//len(episodes)][0],wheel_delay_s=pairs[case//len(episodes)][1],
                entry_hold_substeps=int(action.entry_hold[i]),old_teacher_phase=int(action.entry_phase[i]),
                entry_speed_m_s=float(np.linalg.norm(f[999,4:6])),entry_yaw_rate_rad_s=float(f[999,9]),
                last5s_leg_velocity_rms_rad_s=rms,last5s_mean_speed_m_s=speed,last5s_mean_abs_yaw_rate_rad_s=yaw,
                last5s_xy_excursion_m=drift,checks=checks,qualified=all(checks.values()),rejection_reasons=[k for k,v in checks.items() if not v]))
        for k,v in original.items():torch.testing.assert_close(actor.state_dict()[k],v,atol=0,rtol=0)
        assert action.wheel_index==round(DURATION/.002)
        np.savez_compressed(args.output/'trace.npz',**arrays,time=time,observation_time=time-.01,action_time=time-.01,
            initial_qpos=qpos.cpu().numpy(),initial_qvel=qvel.cpu().numpy(),entry_reference_before=action.entry_before.cpu().numpy(),
            entry_reference_after=action.entry_after.cpu().numpy(),switch_K=switch_k,
            nominal_motor_positions=action.nominal_positions[0].cpu().numpy(),reference_motor_positions=action.reference_positions[0].cpu().numpy())
        np.savez_compressed(args.output/'wheel-500hz.npz',values=action.wheel_steps.cpu().numpy(),time=(np.arange(action.wheel_index)+1)*.002,
            field_names=np.asarray(WHEEL_FIELDS),response_parameters=action._wheel_response.cpu().numpy(),
            nominal_stiffness=action.cfg.wheel_stiffness,nominal_damping=action.cfg.wheel_damping)
        summary={mode:dict(cases=base_count,full_30s=sum(r['checks']['full_30s'] for r in rows if r['mode']==mode),
            final_balance=sum(r['checks']['final_balance'] for r in rows if r['mode']==mode),qualified=sum(r['qualified'] for r in rows if r['mode']==mode)) for mode in MODES}
        report=dict(completed=True,diagnostic_only=True,training_labels_exported=False,optimizer_updates=0,split='train',heldout_used=False,
            checkpoint_sha256=digest(args.checkpoint),xml_sha256=digest(MODEL_XML),dataset_fingerprint=data.description['fingerprint'],
            modes=MODES,base_cases=base_count,num_envs=count,summary=summary,rows=rows,prefix_max_difference=prefix_max,
            duration_s=DURATION,takeover_s=TAKEOVER,actor_unchanged=True,fifo_preserved_at_entry=True,numeric_snapshot_restoration=False,
            source_sha256=sources,trace_sha256=digest(args.output/'trace.npz'),wheel_trace_sha256=digest(args.output/'wheel-500hz.npz'),
            switch_sha256=digest(switch_file))
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg));print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256')},indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');run(p.parse_args())
