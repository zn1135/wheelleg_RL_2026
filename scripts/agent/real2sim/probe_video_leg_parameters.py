"""Frozen-policy GPU parameter sensitivity; independent of training/shared XML."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.fudan_ppo_walk.protocol import samples
from mjlab.managers.event_manager import RecomputeLevel
from scripts.agent.fudan_ppo_walk import evaluate


def candidates():
    rows=[dict(name='baseline')]
    rows += [dict(name=f'joint_damping_{v:g}',damping=v) for v in (.1,.5,1.)]
    rows += [dict(name=f'joint_friction_{v:g}',frictionloss=v) for v in (.1,.3)]
    rows += [dict(name=f'armature_x{v:g}',armature_scale=v) for v in (.5,2.,4.)]
    rows += [dict(name=f'spring_damping_{v:g}',spring_damping=v) for v in (0.,25.,100.)]
    return rows


def items(*_,profiles=None):
    original=[s for s in samples()[::21] if s['case']['category']=='stop_forward']
    return [dict(s,id=f'{c["name"]}_h{int(s["case"]["segments"][0]["command"][2]*100)}',
        seed=30003+3*i,candidate=c) for c in (candidates() if profiles is None else profiles) for i,s in enumerate(original)]


def report(folder):
    import mujoco
    m=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));d=mujoco.MjData(m)
    mapping=json.loads((folder/'fixtures/mapping.json').read_text())
    qi=np.asarray(mapping['motor_q'])[[0,1,3,4]]
    hip=m.joint('robot/rf0').id;wheel=m.body('robot/rfwheel').id
    rows=[]
    for s in json.loads((folder/'protocol.json').read_text())['samples']:
        with np.load(folder/'traces'/(s['id']+'.npz')) as tr:
            end=s['case']['duration'];mask=(tr['qpos_time']>=end-5)&(tr['qpos_time']<end);q=tr['qpos'][mask]
            angles=[];lengths=[]
            # 只做保存姿态的运动学，得到侧视髋轮连线；不积分CPU物理。
            for state in q:
                d.qpos[:]=state;mujoco.mj_kinematics(m,d)
                vector=d.xpos[wheel]-d.xanchor[hip]
                w,x,y,z=state[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                horizontal=vector[0]*np.cos(yaw)+vector[1]*np.sin(yaw)
                angles.append(np.rad2deg(np.arctan2(horizontal,-vector[2])))
                lengths.append(np.hypot(horizontal,vector[2]))
            joints=np.rad2deg(q[:,qi]);joint_pp=np.ptp(joints,axis=0)
            mm=(tr['time']>=end-5)&(tr['time']<end)
            rows.append(dict(sample_id=s['id'],candidate=s['candidate'],height=s['case']['segments'][0]['command'][2],
                window_s=[end-5,end],complete=bool(len(q)==500 and tr['time'][-1]>=end-.002),finite=bool(np.isfinite(tr['qpos']).all()),
                joint_pp_deg=joint_pp.tolist(),joint_max_pp_deg=float(joint_pp.max()),
                last_second_joint_max_pp_deg=float(np.ptp(joints[-100:],axis=0).max()),
                hip_wheel_side_angle_pp_deg=float(np.ptp(angles)),hip_wheel_length_pp_m=float(np.ptp(lengths)),
                body_height_pp_m=float(np.ptp(tr['position'][mm,2])),body_height_mean_m=float(tr['position'][mm,2].mean()),
                horizontal_speed_peak_m_s=float(np.linalg.norm(tr['linear_world'][mm,:2],axis=1).max()),
                pitch_abs_peak_deg=float(np.abs(tr['roll_pitch_deg'][mm,1]).max()),
                any_tilt_over60=bool((np.abs(tr['roll_pitch_deg'])>60).any())))
    result=dict(scope='Single-parameter sensitivity; side-view proxies cannot be equated to uncalibrated video angles',rows=rows)
    (folder/'parameter-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--leg-p',type=float,choices=(30.,60.),default=30.)
    p.add_argument('--leg-d',type=float,choices=(2.,10.),default=2.)
    p.add_argument('--wheel-p',type=float,choices=(.1,.045),default=.1,help='Explicit wheel PD; historical probes retain0.1')
    p.add_argument('--standstill',action='store_true',help='Zero forward/yaw command throughout, rather than a drive-stop transient')
    p.add_argument('--armature-only',action='store_true',help='Compare baseline with the user-selected x2 armature only')
    p.add_argument('--shared-delay-scan',action='store_true',help='Common x2 inertia with0/2/4/6/8ms motor velocity feedback delay; record actual500Hz PD signals')
    p.add_argument('--shared-noise-scan',action='store_true',help='Common x2 inertia with0/.05/.1/.2/.5rad/s Gaussian legPDvelocity noise,500Hz')
    p.add_argument('--profiles-json',type=Path,help='Explicit independent diagnostic candidates; armature_scale is relative to0.015795')
    p.add_argument('--duration-seconds',type=float,default=15.,help='Keep the7s drive phase and extend only zero-command hold')
    p.add_argument('--report-only',action='store_true');args=p.parse_args()
    if args.report_only:report(args.output);return
    assert (FudanWalkCfg.control.leg_stiffness,FudanWalkCfg.control.leg_damping)==(30.,2.)
    profiles=[c for c in candidates() if not args.armature_only or c['name'] in ('baseline','armature_x2')]
    if args.shared_delay_scan:
        profiles=[dict(name='baseline'),dict(name='armature_x2',armature_scale=2.)]
        profiles += [dict(name=f'armature_x2_veldelay_{ms}ms',armature_scale=2.,velocity_delay_ms=ms) for ms in (2,4,6,8)]
    if args.shared_noise_scan:
        assert not args.shared_delay_scan
        profiles=[dict(name='baseline'),dict(name='armature_x2',armature_scale=2.)]
        profiles += [dict(name=f'armature_x2_velnoise_{sigma:g}',armature_scale=2.,velocity_noise_std_rad_s=sigma) for sigma in (.05,.1,.2,.5)]
        profiles.append(dict(name='armature_x2_veldelay_6ms',armature_scale=2.,velocity_delay_ms=6))
    if args.profiles_json:
        assert not (args.armature_only or args.shared_delay_scan or args.shared_noise_scan)
        profiles=json.loads(args.profiles_json.read_text())
    pd_experiment=any('velocity_delay_ms' in c or 'velocity_noise_std_rad_s' in c for c in profiles)
    noise_experiment=any(c.get('velocity_noise_std_rad_s',0.)>0 for c in profiles)
    selected=items(profiles=profiles)
    if args.duration_seconds!=15.:
        assert args.duration_seconds>=15.
        selected=[dict(s,case=dict(s['case'],duration=args.duration_seconds,segments=[
            dict(s['case']['segments'][0]),dict(s['case']['segments'][1],duration=args.duration_seconds-7.)])) for s in selected]
    if args.standstill:
        selected=[dict(s,case=dict(id=f'zero_hold_{s["case"]["segments"][0]["command"][2]:g}',category='zero',duration=args.duration_seconds,
            segments=[dict(start=0.,duration=args.duration_seconds,command=[0.,0.,s['case']['segments'][0]['command'][2]])])) for s in selected]
    torch.set_num_threads(1);start=time.time();deadline=start+args.seconds
    configured=replace(from_sim2sim_config(),pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
    nominal_action_ms=configured.action_delay_ms[0];nominal_sensor_ms=configured.sensor_delay_ms[0]
    if any('action_delay_ms' in c or 'sensor_delay_ms' in c for c in profiles):
        # 先留足每环境队列容量，配置夹具后再写各候选的固定数据龄。
        configured=replace(configured,action_delay_ms=(nominal_action_ms,max(c.get('action_delay_ms',nominal_action_ms) for c in profiles)),
            sensor_delay_ms=(nominal_sensor_ms,max(c.get('sensor_delay_ms',nominal_sensor_ms) for c in profiles)))
    wheel_experiment=any(any(k.startswith('wheel_motor_') or k=='wheel_output_limit_ratio' for k in c) for c in profiles)
    mechanical_experiment=any(c.get('wheel_mechanical',False) for c in profiles)
    wheel_experiment=wheel_experiment or mechanical_experiment
    original=evaluate.configure_samples;records={};controls=[];pd_controls=[];velocity_records=[];applied_torques=[];wheel_records=[];identity_error=[0.]
    def configure(wrapper,batch,folder):
        obs=original(wrapper,batch,folder);sim=wrapper.env.sim;action=wrapper.action
        assert (action.cfg.leg_stiffness,action.cfg.leg_damping)==(args.leg_p,args.leg_d)
        (folder/'actual-control.json').write_text(json.dumps(dict(leg_p=action.cfg.leg_stiffness,leg_d=action.cfg.leg_damping,
            wheel_p=action.cfg.wheel_stiffness,wheel_d=action.cfg.wheel_damping),indent=2)+'\n')
        sim.expand_model_fields(('dof_damping','dof_frictionloss','dof_armature','body_inertia','body_ipos'))
        legs=action._motor_v_adr[[0,1,3,4]]
        wheels=action._wheel_action_ids
        wheel_dofs=action._motor_v_adr[wheels]
        wheel_actuators=[sim.mj_model.actuator('robot/'+n+'_motor').id for n in ('lfwheel','rfwheel')]
        mechanical=torch.tensor([s['candidate'].get('wheel_mechanical',False) for s in batch],device=wrapper.device)
        if mechanical_experiment:sim.expand_model_fields(('actuator_gainprm',))
        base=sim.mj_model.body('robot/base_link').id
        # 探针倍率固定相对历史1倍值；共享XML变为2倍后，避免候选再次翻倍。
        sim.model.dof_armature[:,legs]=.015795
        limits=torch.full((len(batch),2),float('inf'),device=wrapper.device)
        for i,s in enumerate(batch):
            if 'wheel_output_limit_ratio' in s['candidate']:
                limits[i]=action._effort_limits[action._wheel_action_ids]*action._wheel_response[:,0]*s['candidate']['wheel_output_limit_ratio']
        action.wheel_motor.configure([s['candidate'].get('wheel_motor_gain',1.) for s in batch],
            [s['candidate'].get('wheel_motor_delay_ms',0.) for s in batch],
            [s['candidate'].get('wheel_motor_time_constant_ms',0.) for s in batch],limits)
        wheel_fixture={k:v.cpu().numpy() for k,v in action.wheel_motor.fixture().items()}
        for i,s in enumerate(batch):
            c=s['candidate']
            if 'damping' in c:sim.model.dof_damping[i,legs]=c['damping']
            if 'frictionloss' in c:sim.model.dof_frictionloss[i,legs]=c['frictionloss']
            if 'armature_scale' in c:sim.model.dof_armature[i,legs]*=c['armature_scale']
            if 'spring_damping' in c:action.spring_damping[i]=c['spring_damping']
            if 'base_inertia_scale' in c:sim.model.body_inertia[i,base]*=c['base_inertia_scale']
            if 'base_com_offset_m' in c:sim.model.body_ipos[i,base]+=torch.tensor(c['base_com_offset_m'],device=wrapper.device)
            if 'wheel_strength_scale' in c:action.motor_strength[i,action._wheel_action_ids]*=c['wheel_strength_scale']
            if c.get('wheel_mechanical',False):
                assert c.get('wheel_motor_gain',1.)==1. and c.get('wheel_motor_delay_ms',0.)==0. and c.get('wheel_motor_time_constant_ms',0.)==0.
                assert 'wheel_output_limit_ratio' not in c and 'wheel_strength_scale' not in c
                # 机械候选使用单位执行器增益，轮速衰减在实际提交处旁路。
                sim.model.actuator_gainprm[i,wheel_actuators,:]=0.
                sim.model.actuator_gainprm[i,wheel_actuators,0]=1.
                for field in ('armature','damping','frictionloss'):
                    value=torch.as_tensor(c['wheel_'+field],device=wrapper.device)
                    assert torch.isfinite(value).all() and (value>=0).all()
                    getattr(sim.model,'dof_'+field)[i,wheel_dofs]=value
            state=action.disturbances
            state.action_steps[i]=round(c.get('action_delay_ms',nominal_action_ms)/2)
            state.sensor_steps[i]=round(c.get('sensor_delay_ms',nominal_sensor_ms)/2)
            initial=folder/(s['id']+'_initial.npz')
            with np.load(initial) as saved:fixture={k:saved[k].copy() for k in saved.files}
            fixture['dof_armature']=sim.model.dof_armature[i].cpu().numpy().copy()
            fixture['body_inertia']=sim.model.body_inertia[i].cpu().numpy().copy()
            fixture['body_ipos']=sim.model.body_ipos[i].cpu().numpy().copy()
            fixture['strength']=action.motor_strength[i].cpu().numpy().copy()
            fixture['action_delay_steps']=state.action_steps[i].cpu().numpy().copy()
            fixture['sensor_delay_steps']=state.sensor_steps[i].cpu().numpy().copy()
            fixture.update({k:v[i] for k,v in wheel_fixture.items()})
            if mechanical_experiment:
                fixture['wheel_mechanical_model']=mechanical[i].cpu().numpy()
                fixture['actuator_gainprm']=sim.model.actuator_gainprm[i].cpu().numpy().copy()
                fixture['dof_damping']=sim.model.dof_damping[i].cpu().numpy().copy()
                fixture['dof_frictionloss']=sim.model.dof_frictionloss[i].cpu().numpy().copy()
            np.savez_compressed(initial,**fixture)
            records[s['id']]=dict(candidate=c,dof_damping=sim.model.dof_damping[i,legs].cpu().tolist(),
                dof_frictionloss=sim.model.dof_frictionloss[i,legs].cpu().tolist(),
                dof_armature=sim.model.dof_armature[i,legs].cpu().tolist(),spring_damping=action.spring_damping[i].cpu().tolist(),
                velocity_feedback_delay_ms=c.get('velocity_delay_ms',0),velocity_feedback_noise_std_rad_s=c.get('velocity_noise_std_rad_s',0.),
                leg_torque_time_constant_ms=c.get('torque_time_constant_ms',0.))
            records[s['id']].update(base_inertia=sim.model.body_inertia[i,base].cpu().tolist(),
                base_com_m=sim.model.body_ipos[i,base].cpu().tolist(),motor_strength=action.motor_strength[i].cpu().tolist(),
                action_delay_ms=2*int(state.action_steps[i]),sensor_delay_ms=2*int(state.sensor_steps[i]))
            records[s['id']].update({k:v[i].tolist() for k,v in wheel_fixture.items()})
            if mechanical_experiment:
                records[s['id']].update(wheel_mechanical=bool(mechanical[i]),
                    wheel_armature=sim.model.dof_armature[i,wheel_dofs].cpu().tolist(),
                    wheel_damping=sim.model.dof_damping[i,wheel_dofs].cpu().tolist(),
                    wheel_frictionloss=sim.model.dof_frictionloss[i,wheel_dofs].cpu().tolist(),
                    wheel_actuator_gain=sim.model.actuator_gainprm[i,wheel_actuators,0].cpu().tolist())
        sim.recompute_constants(RecomputeLevel.set_const)
        (folder/'parameter-overrides.json').write_text(json.dumps(records,indent=2)+'\n')
        old_apply=action.apply_actions;last={}
        delay_steps=torch.tensor([s['candidate'].get('velocity_delay_ms',0)//2 for s in batch],device=wrapper.device)
        noise_std=torch.tensor([s['candidate'].get('velocity_noise_std_rad_s',0.) for s in batch],device=wrapper.device)
        noise_rng=torch.Generator(device=wrapper.device).manual_seed(73011)
        history=sim.data.qvel[:,legs].clone().unsqueeze(0).repeat(int(delay_steps.max())+1,1,1)
        worlds=torch.arange(len(batch),device=wrapper.device);head=[0]
        physical_legs=[action._physical_joint_ids[int(j)] for j in action._leg_action_ids]
        lag_ms=torch.tensor([s['candidate'].get('torque_time_constant_ms',0.) for s in batch],device=wrapper.device)
        torque_lag=bool((lag_ms>0).any())
        alpha=-torch.expm1(-.002/(lag_ms.clamp(min=1e-9)*.001))
        lagged_torque=torch.zeros((len(batch),4),device=wrapper.device)
        def apply():
            old_apply();ids=action._leg_action_ids
            if mechanical_experiment:
                motor=action.wheel_motor;limited=action._motor_response_torque[:,wheels]
                if not motor.enabled:
                    motor.apply(limited,sim.data.qvel[:,wheel_dofs]);motor.state.copy_(limited)
                ctrl=torch.where(mechanical[:,None],limited,motor.last_ctrl)
                action._entity.set_joint_effort_target(ctrl,joint_ids=action._physical_joint_ids[wheels])
                motor.last_ctrl.copy_(ctrl)
                motor.last_shaft_torque.copy_(ctrl*sim.model.actuator_gainprm[:,wheel_actuators,0])
            target=(action._nominal_pos[:,ids]+action.cfg.leg_position_scale*action._control_actions[:,ids]).clamp(
                action._reference_pos[:,ids]-3.14,action._reference_pos[:,ids]+3.14)
            if pd_experiment:
                history[head[0]].copy_(sim.data.qvel[:,legs])
                velocity=history[(head[0]-delay_steps)%len(history),worlds]
                head[0]=(head[0]+1)%len(history)
                noise=torch.zeros_like(velocity)
                if noise_experiment:
                    # 三模型、各档幅度复用相同基础噪声序列；不消费训练/策略的RNG。
                    standard=torch.randn((3,4),device=wrapper.device,generator=noise_rng)
                    noise=standard[worlds%3]*noise_std[:,None]
                    velocity=velocity+noise
                # 仅延迟腿PD的速度反馈；位置反馈、策略观测延迟、轮控制保持。
                torque=(action.cfg.leg_stiffness*action.kp_scale[:,ids]*(target-sim.data.qpos[:,action._motor_q_adr[ids]])
                    -action.cfg.leg_damping*action.kd_scale[:,ids]*velocity)*action.motor_strength[:,ids]
                torque=torque.clamp(-action._effort_limits[ids],action._effort_limits[ids])
                zero=(delay_steps==0)&(noise_std==0)
                identity_error[0]=max(identity_error[0],float((torque[zero]-action._torques[zero][:,ids]).abs().max()))
                if identity_error[0]>1e-5:raise ValueError('Zero-delay leg PD no longer matches baseline')
                action._torques[:,ids]=torque;action._motor_response_torque[:,ids]=torque
                action._entity.set_joint_effort_target(torque,joint_ids=physical_legs)
            if torque_lag:
                # 只让四腿提交力矩渐近跟随请求值；零滞后行逐值旁路，轮控制保持。
                requested=action._motor_response_torque[:,ids]
                lagged_torque.add_(alpha[:,None]*(requested-lagged_torque))
                lagged_torque[lag_ms==0]=requested[lag_ms==0]
                action._motor_response_torque[:,ids]=lagged_torque
                action._entity.set_joint_effort_target(lagged_torque,joint_ids=physical_legs)
                applied_torques.append(lagged_torque.clone())
            last['sample']=torch.stack((target,sim.data.qpos[:,action._motor_q_adr[ids]],action._torques[:,ids]),-1).clone()
            if pd_experiment or torque_lag:
                pd_controls.append(last['sample'].clone())
            if pd_experiment:
                velocity_records.append(torch.stack((sim.data.qvel[:,legs],velocity,noise),-1).clone())
        action.apply_actions=apply
        if wheel_experiment:
            old_write=wrapper.env.scene.write_data_to_sim
            wheels=action._wheel_action_ids
            motors=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ('lfwheel','rfwheel')]
            def write_controls():
                old_write()
                # 必须在scene写入后读ctrl；set_joint_effort_target本身只写目标缓存。
                wheel_records.append(torch.stack((action._torques[:,wheels],action.wheel_motor.last_delayed,
                    action.wheel_motor.state,sim.data.qvel[:,action._motor_v_adr[wheels]],
                    action.wheel_motor.last_ctrl,sim.data.ctrl[:,motors]),-1).clone())
            wrapper.env.scene.write_data_to_sim=write_controls
        old_step=wrapper.step
        def step(*a,**kw):
            result=old_step(*a,**kw);controls.append(last['sample'].cpu().numpy());return result
        wrapper.step=step
        return obs
    request=SimpleNamespace(checkpoint=args.checkpoint,output=args.output,deadline=deadline,diagnostic=True,
        diagnostic_height=None,batch_size=len(selected),selection=None,leg_load_profile='legacy',response_parameters=None)
    try:
        # 显式固定本次轮P；默认0.1保留历史探针语义，不以倍率偷偷改变控制器。
        with patch.object(evaluate,'evaluation_items',return_value=selected),patch.object(evaluate,'configure_samples',configure),patch.object(evaluate,'from_sim2sim_config',return_value=configured),patch.object(FudanWalkCfg.control,'leg_stiffness',args.leg_p),patch.object(FudanWalkCfg.control,'leg_damping',args.leg_d),patch.object(FudanWalkCfg.control,'wheel_stiffness',args.wheel_p),patch.object(FudanWalkCfg.sim,'wheel_model','legacy_curve'):evaluate.gpu(request)
    finally:
        if args.output.exists():
            (args.output/'probe-run.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                started_unix=start,finished_unix=time.time(),deadline_unix=deadline,leg_p=args.leg_p,leg_d=args.leg_d,wheel_p=args.wheel_p,standstill=args.standstill,
                parameter_candidates=profiles,policy_updates=0,zero_delay_PD_max_error_nm=identity_error[0],
                velocity_delay_scope='LegPDvelocity only; diagnostic unmeasured actuator/feedback lag' if any(c.get('velocity_delay_ms',0)>0 for c in profiles) else None,
                velocity_noise_scope='LegPDvelocity only; independent500Hz zero-mean Gaussian,seed73011,shared normalized draws across profiles/models; unmeasured diagnostic amplitude' if noise_experiment else None,
                torque_lag_scope='Four leg torques only, first-order response after PD and limits; wheels unchanged, unmeasured diagnostic time constant' if applied_torques else None,
                control_sample='LastPDsubstep before integration, t=policy_index*.01+.008; columns target rad, actual rad, requested torque Nm'),indent=2)+'\n')
            if controls:np.savez_compressed(args.output/'control-traces.npz',values=np.asarray(controls),time=np.arange(len(controls))*.01+.008,sample_ids=[s['id'] for s in selected])
            if pd_controls:np.savez_compressed(args.output/'pd-500hz.npz',values=torch.stack(pd_controls).cpu().numpy(),time=np.arange(len(pd_controls))*.002,sample_ids=[s['id'] for s in selected])
            if velocity_records:np.savez_compressed(args.output/'velocity-feedback-500hz.npz',values=torch.stack(velocity_records).cpu().numpy(),time=np.arange(len(velocity_records))*.002,sample_ids=[s['id'] for s in selected])
            if applied_torques:np.savez_compressed(args.output/'applied-torque-500hz.npz',values=torch.stack(applied_torques).cpu().numpy(),time=np.arange(len(applied_torques))*.002,sample_ids=[s['id'] for s in selected])
            if wheel_records:np.savez_compressed(args.output/'wheel-response-500hz.npz',values=torch.stack(wheel_records).cpu().numpy(),
                time=np.arange(len(wheel_records))*.002,sample_ids=[s['id'] for s in selected],
                columns=['limited_PD_request','delayed_request','lag_state','wheel_velocity','response_ctrl','submitted_ctrl'])
    report(args.output)


if __name__=='__main__':main()
