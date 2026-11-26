"""从实机测量重建初态，进行第二跳的GPU闭环交接对照。"""
import argparse
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation, Slerp

from .analyze_real022 import read_csv, matrix
from .compare_landing_lead import ALIGN, ARCHIVE
from .evaluate import policies, WALK
from .probe_takeoff_work import CHECKPOINT, CHECKPOINT_SHA, ROOT, write_json, sha
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import jump_config
from ct_mjlab.envs.chuanliantui.model import load_model
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.reset_geometry import mesh_vertices
from .render_real_log import MirrorPose, MODEL, H7_PROJECT_CONFIGS

ARMS=('walk','wait','recorded')
# 同工况三组只改变交接规则；第三组用日志动作驱动仿真PD，诊断动力学误差。
SCENARIOS=('imu','vx_minus1','vx_plus1','vertical_zero','stationary_xy')
STARTS=(-50,0)
STEPS=60
WHEEL_ARMS = ('walk', 'onset_input', 'right_wheel_log', 'both_wheels_log',
              'right_wheel_log_wait', 'both_wheels_log_wait')
SWING_ARMS = ('right_wheel_log', 'logged_swing', 'center_swing', 'logged_then_center')
ONSET_COLUMNS = [0, 1, 2, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24]
ATTITUDE_SCENARIOS = ('imu','vx_minus03','vx_plus03','early_touch','late_touch')
HEIGHT_ESTIMATE = ROOT/'artifacts/fudan_ppo_jump/20261125-imu-height-replay-01/height.npz'
CONTACT_PROFILES = {
    # 滑动摩擦系数、接触响应时间常数、双轮附加惯量倍率。
    'imu': (.5, .020, 1.),
    'mu025': (.25, .020, 1.),
    'mu100': (1., .020, 1.),
    'mu150': (1.5, .020, 1.),
    'contact10': (.5, .010, 1.),
    'contact05': (.5, .005, 1.),
    'inertia_half': (.5, .020, .5),
    'mu100_contact05': (1., .005, 1.),
    'mu100_contact05_inertia_half': (1., .005, .5),
}
CONTACT_REFINEMENT = {
    'imu': CONTACT_PROFILES['imu'],
    'mu025': CONTACT_PROFILES['mu025'],
    'mu025_contact05': (.25, .005, 1.),
    'mu025_contact05_inertia_half': (.25, .005, .5),
    'mu035_contact05_inertia_half': (.35, .005, .5),
}


def sample(t, x, at):
    return np.array([np.interp(at,t,x[:,j]) for j in range(x.shape[1])])


def seed(t0, s, c, solver, archived, original_yaw):
    t=s['rel_s']; model=solver.model; d=solver.data
    rot=Rotation.from_quat(matrix(s,[f'quat_{j}' for j in (1,2,3,0)]))
    orient=Slerp(t,rot)(t0); aligned=Rotation.from_euler('z',-original_yaw)*orient
    d.qpos[:]=archived
    pos=sample(t,matrix(s,[f'hip_pos_rad_{j}' for j in range(4)]),t0)*solver.sign[:4]
    for side in (0,1):assert solver._solve_leg(side,pos[side*2:side*2+2])
    d.qpos[:7]=[0,0,0,*aligned.as_quat()[[3,0,1,2]]]
    mujoco.mj_kinematics(model,d)
    wheel_ids=[model.body(n).id for n in ('lfwheel','rfwheel')]
    wheel_bottom=min(mesh_vertices(model,d,g)[:,2].min() for g in range(model.ngeom) if model.geom_bodyid[g] in wheel_ids and model.geom_contype[g])
    d.qpos[2]=-wheel_bottom-.0002  # 假设至少较低轮承重；近接触微小压入仅初始化一次。
    model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    mujoco.mj_forward(model,d)
    v=np.zeros(model.nv);motor_names=('lf0','lf00','rf0','rf00','lfwheel','rfwheel')
    vi=[model.joint(name).dofadr[0] for name in motor_names]
    v[vi]=sample(c['rel_s'],matrix(c,[f'velocity_{j}' for j in range(6)]),t0)
    v[3:6]=sample(t,matrix(s,[f'gyro_rad_s_{j}' for j in range(3)]),t0)
    # 用闭链六组等式的速度约束解被动轴速度。
    J=d.efc_J.reshape(-1,model.nv)[d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    passive=np.setdiff1d(np.arange(6,model.nv),vi);v[passive]=np.linalg.lstsq(J[:,passive],-J@v,rcond=None)[0]
    residual=float(abs(J@v).max())
    accel=rot.apply(matrix(s,[f'acc_g_{j}' for j in range(3)])*[1,-1,-1])*9.81-[0,0,9.81]
    bias=accel[(t>=-.9)&(t<-.2)].mean(0);integ=cumulative_trapezoid(accel-bias,t,initial=0,axis=0);integ-=integ[np.searchsorted(t,-.2)].copy()
    v[:3]=Rotation.from_euler('z',-original_yaw).apply(sample(t,integ,t0))
    return d.qpos.copy(),v,residual


@torch.inference_mode()
def run(out, sensor_age_ms=3, *, full_jump=False, wheel_study=False, swing_study=False, attitude_study=False, contact_study=False, contact_refinement=False):
    out.mkdir(parents=True,exist_ok=False); assert sha(CHECKPOINT)==CHECKPOINT_SHA
    with zipfile.ZipFile(ARCHIVE) as z:n,s,c=[read_csv(z,'试次2-'+name+'.csv') for name in ('模型输入动作','H7反馈','H7控制')]
    obs_log=matrix(n,[f'obs_{j}' for j in range(25)]); hlog=matrix(n,[f'history_{j}' for j in range(125)]);actions=matrix(n,[f'action_{j}' for j in range(6)])
    jump_ticks=np.flatnonzero(n['model']=='5200')
    switch=int(jump_ticks[-1])+1; t_walk=n['rel_s'][switch]; print('Walk',t_walk,flush=True)
    full_jump = full_jump or wheel_study or swing_study
    start_offsets=(10*(int(jump_ticks[0])-switch),) if full_jump else STARTS
    contact_study = contact_study or contact_refinement
    attitude_study = attitude_study or contact_study
    contact_profiles = CONTACT_REFINEMENT if contact_refinement else CONTACT_PROFILES
    if attitude_study:
        start_ms = (520,550,610) if contact_refinement else ((520,) if contact_study else (320,460,520))
        start_offsets = tuple(int(tick+10*(int(jump_ticks[0])-switch)) for tick in start_ms)
    steps=STEPS-start_offsets[0]//10 if full_jump or attitude_study else STEPS
    arms = ('walk','recorded_torque') if contact_study else (('walk','recorded','recorded_torque') if attitude_study else (SWING_ARMS if swing_study else (WHEEL_ARMS if wheel_study else ARMS)))
    airborne_study = wheel_study or swing_study
    speed_delta = .1 if airborne_study else 1.
    archive_dir=ARCHIVE.parent; assert json.loads((archive_dir/'report.json').read_text())['archive_sha256']==sha(ARCHIVE)
    ar=np.load(archive_dir/'displayed-poses.npz');mask=ar['case']==1;poses=ar['qpos'][mask];board=ar['board_us'][mask]
    ref_rot=Rotation.from_quat(matrix(s,[f'quat_{j}' for j in (1,2,3,0)]));yaw0=ref_rot[0].as_euler('xyz')[2]
    solver=MirrorPose(MODEL,H7_PROJECT_CONFIGS['H7_ros']);seeds=[]; cases=[]; starts=[]
    height_anchors={}
    if attitude_study:
        alignment=json.loads(ALIGN.read_text());trial=next(t for t in alignment['trials'] if t['trial']==2)
        bounds=np.asarray(trial['touchdown_bracket_frames'])/alignment['fps']-trial['video_offset_s']
        with np.load(HEIGHT_ESTIMATE) as data:
            assert str(data['archive_sha256'].item())==alignment['archive_sha256']==sha(ARCHIVE)
            height_t, height_z=data['trial2_relative_s'],data['trial2_height_m']
        for name,t_anchor in zip(('early_touch','imu','late_touch'),(bounds[0],bounds.mean(),bounds[1])):
            anchor_frame=np.argmin(abs(board-(trial['first_jump_board_us']+t_anchor*1e6)))
            q_anchor,_,_=seed(t_anchor,s,c,solver,poses[anchor_frame],yaw0)
            height_anchors[name]=dict(time_s=float(t_anchor),offset_m=float(q_anchor[2]-np.interp(t_anchor,height_t,height_z)))
    for offset in start_offsets:
        i=switch+offset//10;t=n['rel_s'][i];q,v,res=seed(t,s,c,solver,poses[np.argmin(abs(board-n['board_us'][i]))],yaw0)
        seeds.append(dict(offset_ms=offset,root_vel=v[:6].tolist(),closure_velocity_residual=res,z=float(q[2])))
        scenarios = contact_profiles if contact_study else (ATTITUDE_SCENARIOS if attitude_study else SCENARIOS)
        for scenario in scenarios:
            if contact_refinement and t > .53 and scenario not in ('imu','mu025'):
                continue
            for arm in arms:
                vel=v.copy();pose=q.copy();heading=Rotation.from_quat(q[[4,5,6,3]]).as_euler('xyz')[2]
                if scenario in ('vx_minus1','vx_plus1'):vel[:2]+=np.array([np.cos(heading),np.sin(heading)])*speed_delta*(1 if scenario=='vx_plus1' else -1)
                if scenario=='vertical_zero':vel[2]=0
                if scenario=='stationary_xy':vel[:2]=0
                if attitude_study:
                    anchor=height_anchors.get(scenario,height_anchors['imu'])
                    if not (contact_refinement and t > .53):
                        pose[2]=np.interp(t,height_t,height_z)+anchor['offset_m']
                    if scenario in ('vx_minus03','vx_plus03'):
                        vel[:2]+=np.array([np.cos(heading),np.sin(heading)])*(.3 if scenario=='vx_plus03' else -.3)
                starts.append((i,pose,vel));cases.append(dict(offset_ms=offset,scenario=scenario,arm=arm))
    N=len(cases);walk,jump,_=policies(CHECKPOINT,'cuda:0');env=JumpEnv(N,823,'cuda:0',False,cfg=jump_config(),record_flat=True)
    def tensor(x):return torch.as_tensor(x,device=env.device,dtype=torch.float32)
    try:
        env.reset_idx(torch.arange(N,device=env.device));sim=env.env.sim
        assert sim.mj_model.nq==solver.model.nq and sim.mj_model.nv==solver.model.nv
        for j in range(1,solver.model.njnt):assert sim.mj_model.joint('robot/'+solver.model.joint(j).name).qposadr[0]==solver.model.jnt_qposadr[j]
        if contact_study:
            from mjlab.managers.event_manager import RecomputeLevel
            model = sim.mj_model
            wheel_names = ('robot/lfwheel', 'robot/rfwheel')
            wheel_bodies = [model.body(name).id for name in wheel_names]
            wheel_dofs = [int(model.joint(name).dofadr[0]) for name in wheel_names]
            wheel_geoms = np.flatnonzero(np.isin(model.geom_bodyid, wheel_bodies) & (model.geom_contype != 0))
            assert len(wheel_geoms) == 2 and (model.geom_priority[wheel_geoms] > model.geom_priority[model.geom('terrain').id]).all()
            sim.expand_model_fields(('geom_friction', 'geom_solref', 'dof_armature'))
            profiles = tensor([contact_profiles[case['scenario']] for case in cases])
            torch.testing.assert_close(sim.model.geom_friction[:, wheel_geoms, 0], torch.full_like(sim.model.geom_friction[:, wheel_geoms, 0], .5))
            torch.testing.assert_close(sim.model.geom_solref[:, wheel_geoms, 0], torch.full_like(sim.model.geom_solref[:, wheel_geoms, 0], .02))
            armature = sim.model.dof_armature[:, wheel_dofs].clone()
            sim.model.geom_friction[:, wheel_geoms, 0] = profiles[:, 0:1]
            sim.model.geom_solref[:, wheel_geoms, 0] = profiles[:, 1:2]
            sim.model.dof_armature[:, wheel_dofs] = armature * profiles[:, 2:3]
            sim.recompute_constants(RecomputeLevel.set_const)
        q0=np.stack([b for _,b,_ in starts]);v0=np.stack([b for _,_,b in starts]);sim.data.qpos[:]=tensor(q0);sim.data.qvel[:]=tensor(v0);sim.data.ctrl.zero_();sim.forward();env.env.scene.update(dt=.001)
        mujoco.mj_saveModel(sim.mj_model,str(out/'model.mjb'))
        ids=np.array([i for i,_,_ in starts]);obs=tensor(obs_log[ids]);history=tensor(hlog[ids]);env.history.copy_(history);env._history_needs_fill.zero_()
        env.action.sample_velocity.copy_(sim.data.qvel[:,env.action._motor_v_adr]);env.action.last_sample_q.copy_(sim.data.qpos[:,env.action._motor_q_adr]);env.action._raw_actions.copy_(obs[:,19:25]);env.previous[:,:,0]=obs[:,19:25];env.previous_position.copy_(env.robot.data.root_link_pos_w)
        case_arms=np.array([case['arm'] for case in cases])
        replay_torques = None
        torque_cases = case_arms=='recorded_torque'
        torque_error = torch.zeros((),device=env.device)
        substep = 0
        if attitude_study:
            pd_time = n['rel_s'][ids[torque_cases]][None,:]+np.arange(steps*2)[:,None]*.005
            # 日志控制记录与策略拍错相约 2 ms，取各 5 ms 网格最近的请求。
            right = np.searchsorted(c['rel_s'],pd_time).clip(1,len(c['rel_s'])-1)
            left = right-1
            pick = np.where(abs(c['rel_s'][left]-pd_time)<=abs(c['rel_s'][right]-pd_time),left,right)
            assert abs(c['rel_s'][pick]-pd_time).max()<.003 and not c['fault'][pick].any()
            assert ((c['flags'][pick].astype(int)>>7)&1).all()
            replay_torques = tensor(matrix(c,[f'pd_Nm_{i}' for i in (0,1,4,2,3,5)])[pick])
            assert (abs(replay_torques)<=env.action._effort_limits+1e-6).all()
            drive=env.action
            apply=drive.apply_actions
            def apply_logged_torque():
                apply()
                drive._torques[torque_cases]=replay_torques[substep//5]
                drive._entity.set_joint_effort_target(drive.torques,joint_ids=drive._physical_joint_ids)
            drive.apply_actions=apply_logged_torque
            actuator_ids=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        # 从首拍后转入仿真反馈，使用实际部署观测的−2°重力投影偏置。
        angle=np.deg2rad(-2.);co,si=np.cos(angle),np.sin(angle)
        lagged=[]
        fine_states = []
        fine_forces = []
        fine_wheel_forces = []
        frame=env.trace_frame
        def sample_sensor():
            nonlocal substep
            trace=frame()
            if replay_torques is not None:
                applied=sim.data.actuator_force[torque_cases][:,actuator_ids]
                torque_error.copy_(torch.maximum(torque_error,abs(applied-replay_torques[substep//5]).max()))
                substep+=1
            if contact_study:
                fine_states.append(torch.cat((sim.data.qpos.clone(), sim.data.qvel.clone()), dim=1))
                fine_forces.append(sim.data.actuator_force[:, actuator_ids].clone())
                fine_wheel_forces.append(env.env.scene['jump_wheel'].data.force.clone())
            if sensor_age_ms and len(env.substeps)==9-sensor_age_ms:
                # qpos/qvel已更新；根据此拍四元数计算重力，避免几何缓存的一拍差异。
                q=sim.data.qpos;v=sim.data.qvel;w,x,y,z=q[:,3:7].unbind(-1)
                gravity=torch.stack((2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1),dim=1)
                motors=env.action._motor_q_adr;legs=env.action._leg_action_ids
                lagged.append(torch.cat((v[:,3:6]*.25,gravity,(q[:,motors]-env.action.nominal_positions)[:,legs],v[:,env.action._motor_v_adr]*.05),dim=1))
            return trace
        env.trace_frame=sample_sensor
        frames=[];qs=[];vs=[];modes=[];acts=[];forces=[];support=torch.zeros(N,device=env.device,dtype=torch.bool);streak=np.zeros(N,int);switched=np.zeros(N,bool);switch_time=np.full(N,np.nan);done_at=np.full(N,np.nan)
        record=np.isin(case_arms,('recorded','recorded_torque'))
        wait=np.isin(case_arms, ('wait','right_wheel_log_wait','both_wheels_log_wait'))
        onset_inputs=case_arms=='onset_input'
        log_both=np.isin(case_arms, ('both_wheels_log','both_wheels_log_wait'))
        log_right=log_both | np.isin(case_arms, ('right_wheel_log','right_wheel_log_wait',*SWING_ARMS[1:]))
        onset_cols=torch.as_tensor(ONSET_COLUMNS,device=env.device)
        hip_columns = [0,1,3,4]
        hips = env.action._leg_action_ids
        nominal_hip = env.action._nominal_pos[:,hips]
        lower_hip = env.action._reference_pos[:,hips]-3.14
        upper_hip = env.action._reference_pos[:,hips]+3.14
        hip_scale = env.action.cfg.leg_position_scale
        # 一次腾空状态后若接触地面，结束该例的空中干预。
        in_flight=np.zeros(N,bool); returned_to_ground=np.zeros(N,bool)
        intervention=np.zeros((steps,N),bool)
        swing_intervention=np.zeros((steps,N),bool)
        swing_targets=[]; swing_targets_before=[]
        obs0=obs.cpu().numpy().copy();first_err=[]
        for tick in range(steps):
            indices=ids+tick;dt_ms=(indices-switch)*10;rootq=sim.data.qpos[:,3:7].cpu().numpy();rot=Rotation.from_quat(rootq[:,[1,2,3,0]]);rpy=rot.as_euler('xyz');omega=sim.data.qvel[:,3:6].cpu().numpy();pitch_rate=omega[:,1]*np.cos(rpy[:,0])-omega[:,2]*np.sin(rpy[:,0]);
            quiet=(abs(pitch_rate)<.5)&(abs(rpy[:,1])<np.deg2rad(10))&support.cpu().numpy()&(dt_ms>=0);streak=np.where(quiet,streak+1,0)
            switch_now=(dt_ms>=0)&(~wait|(streak>=3)|(dt_ms>=200));switch_time[~switched&switch_now]=dt_ms[~switched&switch_now];switched|=switch_now
            # 两网使用相同的物理命令和已执行动作历史；本段日志前进/转向/高度取对应策略帧。
            scales=np.where(n['model'][indices]=='5200',3.,2.);cmd=obs_log[indices,6:9]/np.stack([scales,np.full(N,.25),np.full(N,5.)],axis=1);env.commands[:]=tensor(cmd)
            obs[:,6:9]=env.commands*tensor([3.,.25,5.]);history[:,-19:-16]=obs[:,6:9]
            wo=obs.clone();wh=history.reshape(N,5,25).clone();wo[:,6]*=2/3;wh[:,:,6]*=2/3
            jump_obs, jump_history = obs, history
            if wheel_study and 16 <= tick <= 20:
                active = onset_inputs & in_flight & ~returned_to_ground
                jump_obs, jump_history = obs.clone(), history.clone()
                # 只替换此次推理的输入，随后历史继续记录实际执行的动作。
                oi = tensor(obs_log[indices][active])
                hi = tensor(hlog[indices][active]).reshape(-1,5,25)
                jump_obs[active] = jump_obs[active].index_copy(1, onset_cols, oi[:,ONSET_COLUMNS])
                jump_history[active] = jump_history[active].reshape(-1,5,25).index_copy(2, onset_cols, hi[:,:,ONSET_COLUMNS]).flatten(1)
                intervention[tick,active] = True
            wa,_=walk.act_inference(wo,wh.flatten(1));ja,_=jump.act_inference(jump_obs,jump_history);action=torch.where(torch.tensor(switched,device=env.device)[:,None],wa,ja)
            if tick==0:
                compare=~wait if np.any(wait) else np.ones(N,bool);first_err=float(abs(action.cpu().numpy()[compare]-actions[indices][compare]).max());assert first_err<1e-4,first_err
            action[record]=tensor(actions[indices[record]])
            if airborne_study and 17 <= tick <= 52:
                airborne = in_flight & ~returned_to_ground
                both = log_both & airborne
                right = log_right & airborne
                action[right,5] = tensor(actions[indices[right],5])
                action[both,2] = tensor(actions[indices[both],2])
                intervention[tick,right] = True
            if swing_study:
                hip_target=(nominal_hip+hip_scale*action[:,hip_columns]).clamp(lower_hip,upper_hip)
                swing_targets_before.append(hip_target.cpu().numpy().copy())
                if tick>=25:
                    active=(case_arms!='right_wheel_log') & in_flight & ~returned_to_ground & ~switched
                    use_logged=(case_arms=='logged_swing') | ((case_arms=='logged_then_center') & (tick<40))
                    # 520 ms 是实拍触地区间前最后一整拍，后续只保持该参考值。
                    logged_indices=ids+min(tick,52)
                    logged=(nominal_hip+hip_scale*tensor(actions[logged_indices][:,hip_columns])).clamp(lower_hip,upper_hip)
                    mean_target=hip_target.reshape(N,2,2).mean(2)
                    desired_mean=torch.where(torch.as_tensor(use_logged,device=env.device)[:,None],logged.reshape(N,2,2).mean(2),0.)
                    # 前后髋施加相同角偏置，保留每条腿的伸缩目标角差。
                    minimum=(lower_hip-hip_target).reshape(N,2,2).amax(2)
                    maximum=(upper_hip-hip_target).reshape(N,2,2).amin(2)
                    shift=(desired_mean-mean_target).clamp(minimum,maximum).repeat_interleave(2,dim=1)
                    hip_target[active] += shift[active]
                    updated=action[active].clone()
                    updated[:,hip_columns]=((hip_target-nominal_hip)/hip_scale)[active]
                    action[active]=updated
                    swing_intervention[tick,active]=True
                swing_targets.append(hip_target.cpu().numpy().copy())
            qs.append(sim.data.qpos.cpu().numpy().copy());vs.append(sim.data.qvel.cpu().numpy().copy());modes.append(switched.copy());acts.append(action.cpu().numpy().copy());env.substeps.clear()
            obs,_,_,done,_,_=env.step(action,auto_reset=False,resample=False);assert len(env.substeps)==10 and not env.env.sim.wp_data.overflow.numpy().any()
            newdone=done.cpu().numpy()&np.isnan(done_at);done_at[newdone]=dt_ms[newdone]+10
            trace=torch.stack(env.substeps);frames.append(trace.cpu().numpy());support=(trace[:,:,8:10]>.5).all(2).all(0)
            if airborne_study:
                contact = (trace[:,:,8:10]>.5).any(dim=(0,2)).cpu().numpy()
                returned_to_ground |= in_flight & contact
                in_flight |= (~contact & (trace[-1,:,6:8].min(1).values>.02).cpu().numpy())
            # 维持历史的真实前缀；随后只混入已模拟的观测。横滚/俯仰速度由当前仿真产生。
            if sensor_age_ms:
                assert len(lagged)==1
                sensor=lagged.pop();obs[:,:6]=sensor[:,:6];obs[:,9:19]=sensor[:,6:]
            gx=obs[:,3].clone();gz=obs[:,5].clone();obs[:,3]=co*gx+si*gz;obs[:,5]=co*gz-si*gx
            history=torch.cat((history[:,25:],obs),dim=1);forces.append(env.action.torques.cpu().numpy().copy())
        assert torque_error.item()<1e-4
        np.savez_compressed(out/'gpu.npz',q=np.array(qs),v=np.array(vs),trace=np.concatenate(frames),mode=modes,action=acts,torque=forces,initial_obs=obs0,q0=q0,v0=v0,switch_ms=switch_time,done_ms=done_at,intervention=intervention)
        if contact_study:
            state = torch.stack(fine_states).cpu().numpy()
            np.savez_compressed(out/'first-contact.npz', q=state[:, :, :sim.mj_model.nq],
                                v=state[:, :, sim.mj_model.nq:], torque=torch.stack(fine_forces).cpu().numpy(),
                                wheel_force=torch.stack(fine_wheel_forces).cpu().numpy())
        if swing_study:
            np.savez_compressed(out/'swing-targets.npz',active=swing_intervention,before=swing_targets_before,applied=swing_targets)
        write_json(out/'capture.json',dict(cases=cases,seeds=seeds,real_walk_s=float(t_walk),dt_s=.001,pd_ms=5,policy_ms=10,sensor_age_ms=sensor_age_ms,states_injected_once=True,initial_logged_action_error=first_err,logged_torque_max_error_Nm=torque_error.item() if attitude_study else None,checkpoint=CHECKPOINT_SHA,walk_sha=sha(WALK),steps=steps,optimizer_updates=0,archive_sha=sha(ARCHIVE),full_jump=full_jump,wheel_study=wheel_study,swing_study=swing_study,attitude_study=attitude_study,height_anchors=height_anchors,forward_speed_delta_m_s=.3 if attitude_study else speed_delta,contact_profiles=contact_profiles if contact_study else None,contact_refinement=contact_refinement))
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--sensor-age-ms',type=int,choices=(0,3),default=3);studies=p.add_mutually_exclusive_group();studies.add_argument('--full-jump',action='store_true');studies.add_argument('--wheel-study',action='store_true');studies.add_argument('--swing-study',action='store_true');studies.add_argument('--attitude-study',action='store_true');studies.add_argument('--contact-study',action='store_true');studies.add_argument('--contact-refinement',action='store_true');args=p.parse_args();run(args.output,args.sensor_age_ms,full_jump=args.full_jump,wheel_study=args.wheel_study,swing_study=args.swing_study,attitude_study=args.attitude_study,contact_study=args.contact_study,contact_refinement=args.contact_refinement)
