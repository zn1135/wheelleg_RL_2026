#!/usr/bin/env python3
"""Compare one-step teacher/actor feedback at identical live standing states."""
import argparse,copy,json,math,shutil
from dataclasses import dataclass,field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.bc import load_actor
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.teacher.env import body_angles
from ct_mjlab.teacher.standup import DONE
from check_lqr_student_closed_loop import BridgeCfg,ClosedLoopAction,digest
import eval_bc


def frozen_state(obj,prefix=''):
    result={}
    for name,value in vars(obj).items():
        key=prefix+name
        if isinstance(value,torch.Tensor):result[key]=value.cpu().numpy().copy()
        elif isinstance(value,(bool,int,float)):result[key]=np.asarray(value)
        elif hasattr(value,'__dict__'):result.update(frozen_state(value,key+'.'))
    return result


def pd_request(action,target,q,dq):
    legs,wheels=action._leg_action_ids,action._wheel_action_ids
    result=torch.zeros_like(target)
    position=(action.nominal_positions[:,legs]+action.cfg.leg_position_scale*target[:,legs]).clamp(
        action.reference_positions[:,legs]-3.14,action.reference_positions[:,legs]+3.14)
    result[:,legs]=action.cfg.leg_stiffness*(position-q[:,legs])-action.cfg.leg_damping*dq[:,legs]
    velocity=(action.cfg.wheel_velocity_scale*target[:,wheels]).clamp(-action.cfg.max_wheel_velocity,action.cfg.max_wheel_velocity)
    result[:,wheels]=action.cfg.wheel_stiffness*(velocity-dq[:,wheels])-action.cfg.wheel_damping*dq[:,wheels]
    return result.clamp(-action._effort_limits,action._effort_limits)


def read_inputs(action):
    teacher=action.teacher;robot=action._entity.data
    q=action._env.sim.data.qpos[:,teacher.qadr][:,teacher.order]*teacher.sign
    dq=action._env.sim.data.qvel[:,teacher.vadr][:,teacher.order]*teacher.sign
    q=q[:,:4].reshape(-1,2,2).clone();q[:,:,0]+=math.pi
    pitch,roll,yaw,upright=body_angles(robot.root_link_quat_w)
    velocity=robot.root_link_lin_vel_w
    acceleration=torch.where(teacher.velocity_ready[:,None],(velocity-teacher.previous_velocity)/action.control_dt,0.)
    feedback=action._env.sim.data.actuator_force[:,teacher.motor_actuators][:,teacher.order]*teacher.sign
    return dict(q=q,dq=dq[:,:4].reshape(-1,2,2),wheel_speed=dq[:,4:],pitch=pitch,roll=roll,yaw=yaw,
        gyro_pitch=robot.root_link_ang_vel_b[:,1],gyro_yaw=robot.root_link_ang_vel_b[:,2],upright=upright,
        acceleration=acceleration[:,0]*yaw.cos()+acceleration[:,1]*yaw.sin(),vertical_acceleration=acceleration[:,2],
        feedback_torque=feedback[:,:4].reshape(-1,2,2),command=teacher.command)


def teacher_target(action,controller,inputs,q,dq):
    output=controller.step(**inputs)
    desired=torch.zeros_like(action.raw_action)
    desired[:,action.teacher.order]=output*action.teacher.sign
    desired=desired.clamp(-action.teacher.limits,action.teacher.limits)
    target=torch.zeros_like(desired);legs,wheels=action._leg_action_ids,action._wheel_action_ids
    target[:,legs]=(q[:,legs]+(desired[:,legs]+action.cfg.leg_damping*dq[:,legs])/action.cfg.leg_stiffness-
        action.nominal_positions[:,legs])/action.cfg.leg_position_scale
    target[:,wheels]=(dq[:,wheels]+desired[:,wheels]/action.cfg.wheel_stiffness)/action.cfg.wheel_velocity_scale
    return target,desired


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1)
    models={};saved_models={};before={}
    for name,path in (('stable_legacy39',args.stable),('wheel39_5000',args.first),('wheel39_10000',args.latest)):
        actor,saved=load_actor(path,'cuda:0');models[name]=actor;saved_models[name]=saved
        before[name]={k:v.clone() for k,v in actor.state_dict().items()}
    assert saved_models['stable_legacy39']['format']=='ct_bc_actor39_pd6_v1'
    assert all(saved_models[name].get('dataset',{}).get('wheel_angles') for name in ('wheel39_5000','wheel39_10000'))
    samples=[];applied=[];baseline_differences=[]
    args.output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[2];sources={}
    for file in list((source_root/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[
        Path(__file__),source_root/'scripts/agent/eval_bc.py',source_root/'scripts/agent/check_lqr_student_closed_loop.py',
        source_root/'scripts/agent/check_smp_policy_source.py',source_root/'scripts/agent/collect_lqr_student.py']:
        name=file.relative_to(source_root);target=args.output/'snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target);sources[str(name)]=digest(file)
    class FeedbackAction(ClosedLoopAction):
        def process_actions(self,actions):
            tick=int(self._substep_ticks[0])
            probe=None
            if tick in (2500,5000,7500):
                assert (self.kp_scale==1).all() and (self.kd_scale==1).all() and (self.motor_strength==1).all()
                history=copy.deepcopy(self.teacher.controller)
                obs=multimode_observations.get_state(self._env).actor().clone()
                q=self._env.sim.data.qpos[:,self._motor_q_adr].clone()
                dq=self._env.sim.data.qvel[:,self._motor_v_adr].clone()
                inputs={k:v.clone() for k,v in read_inputs(self).items()}
                valid=(self._env._source_probe['live']&(self._env._source_probe['hold']>=500)&
                    (history.standup.phase==DONE)&~history.air.flight)
                assert valid.any()
                base_target,base_desired=teacher_target(self,copy.deepcopy(history),inputs,q,dq)
                np.savez_compressed(args.output/f'controller-{tick}.npz',**frozen_state(history),
                    **{'input.'+k:v.cpu().numpy() for k,v in inputs.items()},observation=obs.cpu().numpy(),
                    physical_q=q.cpu().numpy(),physical_dq=dq.cpu().numpy(),valid=valid.cpu().numpy(),
                    nominal_positions=self.nominal_positions.cpu().numpy(),reference_positions=self.reference_positions.cpu().numpy())
                responses={}
                baseline_targets={'teacher':base_target}
                for name,actor in models.items():
                    model_obs=obs.clone()
                    if name=='stable_legacy39':model_obs[:,14:16]=0.
                    baseline_targets[name]=actor.mlp(model_obs)
                for variable,amplitudes in (('pitch_rad',[math.radians(x) for x in (.25,.5,1.)]),
                                            ('wheel_forward_rad_s',[.1,.25,.5])):
                    for amplitude in amplitudes:
                        for sign in (-1,1):
                            delta=sign*amplitude
                            altered=obs.clone();physical_dq=dq.clone()
                            measurement={k:v.clone() for k,v in inputs.items()}
                            if variable=='pitch_rad':
                                measurement['pitch']+=delta
                                measurement['upright']=measurement['roll'].cos()*measurement['pitch'].cos()
                                angle=torch.asin(obs[:,7].clamp(-1,1))+delta
                                roll=torch.atan2(-obs[:,8],-obs[:,9])
                                altered[:,7]=angle.sin();altered[:,8]=-roll.sin()*angle.cos();altered[:,9]=-roll.cos()*angle.cos()
                            else:
                                measurement['wheel_speed']+=delta
                                physical_dq[:,2]+=delta;physical_dq[:,5]-=delta
                                altered[:,20]+=delta*.1;altered[:,21]-=delta*.1
                            target,_=teacher_target(self,copy.deepcopy(history),measurement,q,physical_dq)
                            targets={'teacher':target}
                            for name,actor in models.items():
                                model_obs=altered.clone()
                                if name=='stable_legacy39':model_obs[:,14:16]=0.
                                targets[name]=actor.mlp(model_obs)
                            key=f'{variable}:{amplitude}:{sign}'
                            responses[key]={name:dict(target=target.cpu().numpy(),
                                torque=pd_request(self,target,q,physical_dq).cpu().numpy()) for name,target in targets.items()}
                probe=dict(tick=tick,time_s=tick*self._env.physics_dt,valid=valid.cpu().numpy(),
                    observation=obs.cpu().numpy(),base_target=base_target,base_desired=base_desired,
                    baseline={name:dict(target=target.cpu().numpy(),torque=pd_request(self,target,q,dq).cpu().numpy())
                        for name,target in baseline_targets.items()},responses=responses)
            super().process_actions(actions)  # 真实轨迹始终由教师经过原PD/FIFO驱动。
            applied.append(self.raw_action.cpu().numpy().copy())
            if probe is not None:
                torch.testing.assert_close(probe.pop('base_target'),self.raw_action,atol=2e-6,rtol=2e-6)
                desired=probe.pop('base_desired')
                torch.testing.assert_close(desired,self.teacher.torques,atol=2e-6,rtol=2e-6)
                baseline_differences.append(float((desired-self.teacher.torques).abs().max()))
                samples.append(probe)
    @dataclass(kw_only=True)
    class FeedbackCfg(BridgeCfg):
        class_type:type=field(default_factory=lambda:FeedbackAction)
    get_cfgs=eval_bc.task_registry.get_cfgs
    def configuration(*a,**k):
        cfg,training=get_cfgs(*a,**k)
        cfg.actions['pd']=FeedbackCfg(**{name:value for name,value in vars(cfg.actions['pd']).items() if name!='class_type'})
        return cfg,training
    call=SimpleNamespace(dataset=args.dataset,pose_dataset=args.pose_dataset,corrections=[],checkpoint=args.first,
        device='cuda:0',split='validation',collect_corrections=False,full_precision=False,output=args.output/'teacher-rollout')
    with patch.object(eval_bc.task_registry,'get_cfgs',side_effect=configuration):eval_bc.run(call)
    report_path=args.output/'teacher-rollout/result.json';report=json.loads(report_path.read_text())
    with np.load(args.output/'teacher-rollout/trace.npz') as d:arrays={k:d[k] for k in d.files}
    arrays['applied_teacher_action']=np.asarray(applied)
    np.savez_compressed(args.output/'teacher-rollout/trace.npz',**arrays)
    report.update(diagnostic_only=True,independent_student_control=False,teacher_controls_entire_rollout=True,
        trace_sha256=digest(args.output/'teacher-rollout/trace.npz'),
        scope='Live teacher anchors only. Recorded action is the unused actor request; applied_teacher_action drives physics. No teacher labels enter training.')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    assert len(samples)==3
    summary=[];archive={}
    for i,sample in enumerate(samples):
        valid=sample['valid'];archive[f'{i}.valid']=valid;archive[f'{i}.observation']=sample['observation']
        for name,values in sample['baseline'].items():
            for key,value in values.items():archive[f'{i}.base.{name}.{key}']=value
        for key,values in sample['responses'].items():
            for name,response in values.items():
                for kind,value in response.items():archive[f'{i}.{key}.{name}.{kind}']=value
    np.savez_compressed(args.output/'probes.npz',**archive)
    for variable,amplitudes in (('pitch_rad',[math.radians(x) for x in (.25,.5,1.)]),('wheel_forward_rad_s',[.1,.25,.5])):
        for amplitude in amplitudes:
            derivatives={name:[] for name in ('teacher',*models)}
            for sample in samples:
                for name in derivatives:
                    plus=sample['responses'][f'{variable}:{amplitude}:1'][name]['torque']
                    minus=sample['responses'][f'{variable}:{amplitude}:-1'][name]['torque']
                    derivatives[name].append(((plus-minus)/(2*amplitude))[sample['valid']])
            derivatives={k:np.concatenate(v) for k,v in derivatives.items()}
            teacher=derivatives['teacher'];wheel_teacher=(teacher[:,2]-teacher[:,5])*.5
            active=abs(wheel_teacher)>1e-3
            entry=dict(variable=variable,amplitude=amplitude,standing_anchors=len(teacher),models={})
            for name,values in derivatives.items():
                wheel=(values[:,2]-values[:,5])*.5
                cosine=np.sum(values*teacher,axis=-1)/(np.linalg.norm(values,axis=-1)*np.linalg.norm(teacher,axis=-1)).clip(1e-9)
                entry['models'][name]=dict(wheel_forward_torque_derivative_median=float(np.median(wheel)),
                    wheel_forward_torque_derivative_min=float(wheel.min()),wheel_forward_torque_derivative_max=float(wheel.max()),
                    wheel_direction_agreement=float((wheel[active]*wheel_teacher[active]>0).mean()) if active.any() else None,
                    wheel_signed_gain_ratio_median=float(np.median(wheel[active]/wheel_teacher[active])) if active.any() else None,
                    six_motor_derivative_median=np.median(values,axis=0).tolist(),six_motor_direction_cosine_median=float(np.median(cosine)))
            summary.append(entry)
    for name,actor in models.items():
        for key,value in before[name].items():torch.testing.assert_close(value,actor.state_dict()[key],atol=0,rtol=0)
    result=dict(completed=True,optimizer_updates=0,anchor_times_s=[x['time_s'] for x in samples],
        valid_anchor_counts=[int(x['valid'].sum()) for x in samples],
        max_teacher_branch_vs_live_request_difference=max(baseline_differences),feedback=summary,
        checkpoint_sha256={name:digest(path) for name,path in (('stable_legacy39',args.stable),('wheel39_5000',args.first),('wheel39_10000',args.latest))},
        source_sha256=sources,probes_sha256=digest(args.output/'probes.npz'),models_unchanged=True,
        scope='Centered finite differences of requested nominal motor torques at fixed live teacher states/history. Inputs perturbed consistently; no physics perturbation, FIFO rollout, motor-response gain, training or stability proof.')
    (args.output/'check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('feedback','source_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pose-dataset','stable','first','latest','output'):p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
