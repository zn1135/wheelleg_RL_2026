#!/usr/bin/env python3
"""Zero-update parking diagnosis: teacher queries and delayed PD takeover at 10s."""
import argparse,json
from dataclasses import asdict,dataclass,field
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from ct_mjlab.bc import ActionDataset,load_actor
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose,inverted
from ct_mjlab.teacher.env import body_angles
from ct_mjlab.teacher.standup import DONE,FAILED
from check_lqr_student_closed_loop import BridgeCfg,ClosedLoopAction,digest
from check_smp_policy_source import balance_substep

TAKEOVER=10.
DURATION=30.
PHYSICS_FIELDS=('qpos','qvel','qacc_warmstart','ctrl','act','time','qfrc_applied','xfrc_applied',
    'actuator_force','qfrc_actuator','xpos','xquat','cvel','subtree_com','sensordata')


def object_state(obj):
    result={}
    for k,v in vars(obj).items():
        if k in ('env','_env','robot','action','cfg','_entity','_data','_model'):continue
        if isinstance(v,torch.Tensor):result[k]=('tensor',v.detach().cpu().clone())
        elif isinstance(v,(bool,int,float,str)):result[k]=('scalar',v)
        elif isinstance(v,dict) and all(isinstance(x,torch.Tensor) for x in v.values()):
            result[k]=('dict',{n:x.detach().cpu().clone() for n,x in v.items()})
        elif hasattr(v,'__dict__') and type(v).__module__.startswith(('ct_mjlab','mjlab.sensor')):
            result[k]=('object',object_state(v))
    return result


def restore_object(obj,state):
    for k,(kind,value) in state.items():
        if kind=='tensor':getattr(obj,k).copy_(value.to(getattr(obj,k).device))
        elif kind=='scalar':setattr(obj,k,value)
        elif kind=='dict':
            for n,x in value.items():getattr(obj,k)[n].copy_(x.to(getattr(obj,k)[n].device))
        else:restore_object(getattr(obj,k),value)


def switch_state(raw,action,obs):
    torch.cuda.synchronize()
    return dict(physics={n:getattr(raw.sim.data,n).clone().cpu() for n in PHYSICS_FIELDS if hasattr(raw.sim.data,n)},
        action=object_state(action),command=object_state(raw.command_manager.get_term('standup')),
        observation=object_state(multimode_observations.get_state(raw)),
        obs={k:v.clone().cpu() for k,v in obs.items()},episode=raw.episode_length_buf.clone().cpu(),
        counters={k:getattr(raw,k) for k in ('common_step_counter','_sim_step_counter') if hasattr(raw,k)},
        probe={k:v.clone().cpu() for k,v in raw._source_probe.items()},
        sensors={n:object_state(raw.scene[n]) for n in ('wheel_contact','base_contact','undesired_contact')},
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(raw.device))


def restore_switch(raw,action,obs,state):
    torch.cuda.synchronize()
    for n,v in state['physics'].items():getattr(raw.sim.data,n).copy_(v.to(raw.device))
    restore_object(action,state['action']);restore_object(raw.command_manager.get_term('standup'),state['command'])
    restore_object(multimode_observations.get_state(raw),state['observation'])
    for n,v in state['obs'].items():obs[n].copy_(v.to(raw.device))
    raw.episode_length_buf.copy_(state['episode'].to(raw.device))
    for n,v in state['counters'].items():setattr(raw,n,v)
    for n,v in state['probe'].items():raw._source_probe[n].copy_(v.to(raw.device))
    for n,v in state['sensors'].items():restore_object(raw.scene[n],v)
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],raw.device)
    torch.cuda.synchronize()
    # 真实状态、缓存观测和旧动作队列都必须逐项相同，随后才允许分支选动作。
    for n,v in state['physics'].items():torch.testing.assert_close(getattr(raw.sim.data,n).clone().cpu(),v,atol=0,rtol=0)
    for n,v in state['obs'].items():torch.testing.assert_close(obs[n].cpu(),v,atol=0,rtol=0)
    torch.testing.assert_close(action._action_history.cpu(),state['action']['_action_history'][1],atol=0,rtol=0)


class ParkingAction(ClosedLoopAction):
    def process_actions(self,actions):
        switch=self._env.episode_length_buf==round(TAKEOVER/self._env.step_dt)
        if switch.any():
            history=self._action_history.clone();head=self._history_head
            # 只改停车参考，不重置物理、原FIFO、估计器或腿部控制历史。
            if self.cfg.parking_mode in ('relock','balance'):
                self.teacher.controller.pos[switch]=0.
                self.teacher.controller.yaw_target[switch]=body_angles(self._entity.data.root_link_quat_w)[2][switch]
            if self.cfg.parking_mode=='balance':
                assert (self._env._source_probe['hold'][switch]>=500).all(),'Only enter balance on confirmed wheel support'
                self.teacher.controller.standup.reset(switch,standing=True)
        super().process_actions(actions)
        self.teacher_target=self.raw_action.clone()
        self.student_target=actions.clone()
        q=self._env.sim.data.qpos[:,self._motor_q_adr]
        dq=self._env.sim.data.qvel[:,self._motor_v_adr]
        legs,wheels=self._leg_action_ids,self._wheel_action_ids
        implied=torch.zeros_like(actions)
        target=self.nominal_positions[:,legs]+self.cfg.leg_position_scale*actions[:,legs]
        target=target.clamp(self.reference_positions[:,legs]-3.14,self.reference_positions[:,legs]+3.14)
        implied[:,legs]=self.cfg.leg_stiffness*(target-q[:,legs])-self.cfg.leg_damping*dq[:,legs]
        implied[:,wheels]=self.cfg.wheel_stiffness*((actions[:,wheels]*self.cfg.wheel_velocity_scale).clamp(
            -self.cfg.max_wheel_velocity,self.cfg.max_wheel_velocity)-dq[:,wheels])
        self.student_requested=implied.clamp(-self._effort_limits,self._effort_limits)
        active=(self._env.episode_length_buf>=round(TAKEOVER/self._env.step_dt)) & (self.cfg.parking_mode!='student')
        self.teacher_selected=active
        selected=torch.where(active[:,None],self.teacher_target,actions)
        HuananhuPdAction.process_actions(self,selected)
        if switch.any():
            assert torch.equal(self._action_history,history) and self._history_head==head


@dataclass(kw_only=True)
class ParkingCfg(BridgeCfg):
    class_type:type=field(default_factory=lambda:ParkingAction)
    parking_mode:str='student'


@torch.inference_mode()
def run(args):
    data=ActionDataset([args.dataset],args.corrections);actor,saved=load_actor(args.checkpoint,args.device)
    assert saved['dataset']['fingerprint']==data.description['fingerprint']
    chosen={}
    for row in data.motion.splits['validation']:chosen.setdefault(row['group_id'],row)
    episodes=[chosen[k] for k in sorted(chosen)];pairs=[(a,b) for a in (.002,.004) for b in (.002,.004)]
    initial=[]
    for e in episodes:
        with np.load(e['path']) as d:initial.append((d['initial_qpos'].copy(),d['initial_qvel'].copy()))
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    cfg.auto_reset=False;cfg.seed=42;cfg.episode_length_s=DURATION+1.
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.terminations['inverted']=TerminationTermCfg(func=inverted);cfg.rewards={}
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.actions['pd']=ParkingCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'},parking_mode=args.mode)
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False
    cfg.actions['pd'].action_delay_choices_s=(.002,.004)
    count=len(episodes)*len(pairs);args.output.mkdir(parents=True,exist_ok=False)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=count,device=args.device)
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
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
        names=('qpos','qvel','motor_dq','features','observation','student_action','teacher_action','applied_action',
            'teacher_requested','student_requested','teacher_x','teacher_target_state','teacher_contributions','teacher_phase',
            'teacher_fault','teacher_air','teacher_selected','actual_torque','alive','done')
        buffers={k:[] for k in names};reader=MotionStateReader(raw)
        original={k:v.clone() for k,v in actor.state_dict().items()}
        for tick in range(round(DURATION/raw.step_dt)):
            if tick==round(TAKEOVER/raw.step_dt):
                if args.switch_state is not None:
                    frozen=torch.load(args.switch_state,map_location='cpu',weights_only=True)
                    restore_switch(raw,action,obs,frozen)
                else:
                    frozen=switch_state(raw,action,obs)
                    torch.save(frozen,args.output/'switch-state.pt')
                np.savez_compressed(args.output/'switch-check.npz',qpos=raw.sim.data.qpos.clone().cpu().numpy(),
                    qvel=raw.sim.data.qvel.clone().cpu().numpy(),observation=obs['actor'].cpu().numpy(),
                    fifo=action._action_history.cpu().numpy(),teacher_phase=action.teacher.controller.standup.phase.cpu().numpy())
            live=raw._source_probe['live'].clone();inputs=obs['actor'].clone();outputs=actor.mlp(inputs)
            obs,_,done,_=env.step(outputs);state=reader.read();c=action.teacher.controller
            values=dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,motor_dq=state['motor_dq'],features=reader.features(),
                observation=inputs,student_action=outputs,teacher_action=action.teacher_target,applied_action=action.raw_action,
                teacher_requested=action.teacher.torques,student_requested=action.student_requested,teacher_x=c.x,
                teacher_target_state=c.target,teacher_contributions=c.contributions,teacher_phase=c.standup.phase,
                teacher_fault=c.standup.fault,teacher_air=c.air.flight,teacher_selected=action.teacher_selected,
                actual_torque=raw.sim.data.actuator_force[:,action.teacher.motor_actuators],alive=live,done=done.bool())
            for k,v in values.items():buffers[k].append(v.cpu().numpy().copy())
            assert torch.equal(action.raw_action,torch.where(action.teacher_selected[:,None],action.teacher_target,outputs))
            if tick<20:assert not action.torques.any()
            raw._source_probe['live']&=~done.bool()
            if tick%500==0:print(f'{args.mode} t={(tick+1)*.01:.2f}s live={int(raw._source_probe["live"].sum())}/{count} teacher_done={int((c.standup.phase==DONE).sum())}',flush=True)
            if not raw._source_probe['live'].any():break
            if done.any():
                raw.reset(env_ids=done.nonzero().flatten());action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
        arrays={k:np.asarray(v) for k,v in buffers.items()};time=(np.arange(len(arrays['alive']))+1)*.01;rows=[]
        for index in range(count):
            length=int(arrays['alive'][:,index].sum());late=(time[:length]>DURATION-5.)
            full=length==round(DURATION/.01);f=arrays['features'][:length,index];q=arrays['qpos'][:length,index]
            rms=float(np.sqrt(np.square(arrays['motor_dq'][:length,index][:,[0,1,3,4]][late]).mean())) if full else None
            speed=float(np.linalg.norm(f[late,4:6],axis=-1).mean()) if full else None
            yaw=float(np.abs(f[late,9]).mean()) if full else None
            drift=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max()) if full else None
            balance=bool(full and raw._source_probe['hold'][index]>=500)
            stopped=bool(full and speed<=.01 and yaw<=.02 and drift<=.02)
            parked=bool(stopped and balance)  # 趴地不动不能计为双轮停车。
            rows.append(dict(env=index,group_id=episodes[index%len(episodes)]['group_id'],leg_delay_s=pairs[index//len(episodes)][0],
                wheel_delay_s=pairs[index//len(episodes)][1],observed_s=length*.01,full_30s=full,final_balance=balance,
                last5s_leg_velocity_rms_rad_s=rms,last5s_mean_speed_m_s=speed,last5s_mean_abs_yaw_rate_rad_s=yaw,
                last5s_xy_excursion_m=drift,kinematic_stop=stopped,parking=parked,smooth_parking=bool(parked and rms<=.02)))
        for k,v in original.items():torch.testing.assert_close(actor.state_dict()[k],v,atol=0,rtol=0)
        np.savez_compressed(args.output/'trace.npz',**arrays,time=time,initial_qpos=qpos.cpu().numpy(),initial_qvel=qvel.cpu().numpy())
        report=dict(completed=True,mode=args.mode,num_envs=count,takeover_s=TAKEOVER,duration_s=DURATION,
            optimizer_updates=0,diagnostic_only=True,heldout_labels_for_training=False,rows=rows,
            full_30s=sum(r['full_30s'] for r in rows),final_balance=sum(r['final_balance'] for r in rows),
            kinematic_stop=sum(r['kinematic_stop'] for r in rows),
            parking=sum(r['parking'] for r in rows),smooth_parking=sum(r['smooth_parking'] for r in rows),
            checkpoint_sha256=digest(args.checkpoint),xml_sha256=digest(MODEL_XML),actor_unchanged=True,
            dataset_fingerprint=data.description['fingerprint'],trace_sha256=digest(args.output/'trace.npz'),
            script_sha256=digest(__file__),switch_state_sha256=digest(args.switch_state or args.output/'switch-state.pt'),
            teacher_history='Frozen canonical10s physics/warmstart/feedback, FIFO, teacher/observation/command/sensor histories and RNG. relock changes pos/yaw references only; balance also resets the standup sub-state to DONE on confirmed wheel support.',
            scope='Nominal zero-command 2/4ms student-PD parking takeover; no training data export or production control changes.')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mode',choices=('student','warm','relock','balance'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--switch-state',type=Path)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
