#!/usr/bin/env python3
"""Frozen walk-v7 diagnosis: controlled ramps, training-noise rewards and checkpoint comparison.

No production edits or learning. Each invocation requires a new output directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.utils.lab_api.math import quat_apply_inverse
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split.runner import validate
from ct_mjlab.envs.chuanliantui_split.mdp import WALK_INITIAL_HEIGHT

JOINTS = ('lf0','lf00','rf0','rf00','lf1','rf1','lfwheel','rfwheel')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def stats(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return None if not x.size else dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
        min=float(x.min()), max=float(x.max()), p05=float(np.quantile(x,.05)), p95=float(np.quantile(x,.95)))


def get(data, meta, key):
    a,b = meta['layout'][key]
    return data[...,a:b]


def summarize(data, meta):
    times = np.arange(1,len(data)+1)*meta['dt']
    rows=[]; dt=meta['dt']
    for case in meta['cases']:
        ids=np.asarray(case['env_ids']); target=case['speed']; accel=case['acceleration']
        endpoint=meta['warmup']+abs(target)/accel+meta['hold']
        full=meta['warmup']+abs(target)/accel
        end=np.asarray(meta['episode_end'])[ids]
        survived=end >= endpoint-1e-6
        valid=get(data,meta,'valid')[:,ids,0]>.5
        steady=valid & (times[:,None]>=full+.5) & (times[:,None]<=endpoint)
        last=steady & (times[:,None]>=endpoint-2.) & survived[None,:]
        vx=get(data,meta,'base')[:,ids,3]
        rewards=get(data,meta,'reward_terms')[:,ids]
        total=rewards.sum(-1)
        ramp=valid & (times[:,None]>=meta['warmup']) & (times[:,None]<full)
        terminal=np.zeros_like(valid)
        for local,e in enumerate(ids):
            if end[local]<endpoint:
                terminal[:,local]=(times<=end[local]+1e-5)&(times>end[local]-.1)&valid[:,local]
        row=dict(case=case['name'],speed=target,acceleration=accel,height=case.get('height'),
            envs=len(ids),survived=int(survived.sum()),endpoint_s=endpoint,
            survived_ramp_plus2s=int((end>=full+2.-1e-6).sum()),
            failures_before_ramp=int((end<meta['warmup']).sum()),
            vx_last2_survivors=stats(vx[last]),mae_last2_survivors=stats(abs(vx-target)[last]),
            vx_full_command_all_live=stats(vx[steady]),
            reward_rate_ramp=stats(total[ramp]),reward_rate_steady=stats(total[steady]),
            reward_rate_terminal100ms=stats(total[terminal]),
            reward_terms_ramp={k:stats(rewards[...,j][ramp]) for j,k in enumerate(meta['reward_names'])},
            reward_terms_terminal100ms={k:stats(rewards[...,j][terminal]) for j,k in enumerate(meta['reward_names'])},
            reward_terms_last2_survivors={k:stats(rewards[...,j][last]) for j,k in enumerate(meta['reward_names'])},
            roll_last2=stats(get(data,meta,'base')[:,ids,2][last]),
            height_last2=stats(get(data,meta,'base')[:,ids,0][last]))
        if 'td_delta' in meta['layout']:
            td=get(data,meta,'td_delta')[:,ids,0]
            row['td_terminal100ms']=stats(td[terminal])
            final_mask=terminal & (get(data,meta,'done')[:,ids,0]>.5)
            row['td_at_failure']=stats(td[final_mask])
            row['critic_value_at_failure']=stats(get(data,meta,'value_prediction')[:,ids,0][final_mask])
        rows.append(row)
    return rows


@torch.inference_mode()
def run_batch(args, cases, checkpoint, filename, training=False):
    count=sum(len(c['env_ids']) for c in cases)
    cfg,train=task_registry.get_cfgs('chuanliantui_walk',play=not training)
    cfg.auto_reset=False; cfg.seed=args.seed
    wrapped,_=task_registry.make_env('chuanliantui_walk',env_cfg=cfg,train_cfg=train,
        num_envs=count,device=args.device,seed=args.seed)
    raw=wrapped.unwrapped; frames=[]; layout={}
    try:
        built,_=task_registry.make_alg_runner(wrapped,'chuanliantui_walk',train_cfg=train,device=args.device)
        saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
        state=raw.command_manager.get_term('standup'); state.speed_curriculum_frozen=True
        validate(saved['infos'],state.cfg)
        # 只载入Actor及编码器权重，不恢复优化器、不调用learn。
        built.alg.actor.load_state_dict(saved['actor_state_dict'],strict=True)
        built.alg.critic.load_state_dict(saved['critic_state_dict'],strict=True)
        built.alg.critic.eval()
        policy=built.get_inference_policy(device=args.device)
        before={k:v.clone() for k,v in built.alg.actor.state_dict().items()}
        obs,_=raw.reset(seed=args.seed)
        pd=raw.action_manager.get_term('pd'); robot=raw.scene['robot']
        joint_ids=robot.find_joints(JOINTS,preserve_order=True)[0]
        qadr=robot.indexing.joint_q_adr[joint_ids]; vadr=robot.indexing.joint_v_adr[joint_ids]
        hips=robot.find_bodies(('lf0','rf0'),preserve_order=True)[0]
        wheels=robot.find_bodies(('lfwheel','rfwheel'),preserve_order=True)[0]
        targets=torch.zeros(count,device=args.device); accels=torch.ones_like(targets)
        for c in cases:
            ids=c['env_ids']; targets[ids]=c['speed']; accels[ids]=c['acceleration']
            if c.get('height') is not None:state.episode_height[ids]=c['height']
        state._apply_height(None)
        # 复位事件/PD/观测仍使用正式实现；只固定诊断请求，不让随机段覆盖它。
        def commands(t):
            state.tracking_episode.fill_(True); state.zero_hold.zero_()
            state.tracking_velocity.copy_(targets if t>=args.warmup else torch.zeros_like(targets))
            state.tracking_yaw_rate.zero_(); state.tracking_interval_steps.fill_(1000000)
            state.tracking_acceleration.copy_(accels)
            for c in cases:
                if c.get('height') is not None:state.episode_height[c['env_ids']]=c['height']
            state._apply_height(None)
        commands(0.)
        raw.observation_manager.reset(); raw.obs_buf=raw.observation_manager.compute(update_history=True)
        obs=raw.get_observations()
        dt=raw.step_dt
        duration=args.warmup+max(abs(c['speed'])/c['acceleration'] for c in cases)+args.hold
        live=torch.ones(count,dtype=torch.bool,device=args.device)
        end=np.full(count,duration+dt); reward_names=list(raw.reward_manager._term_names)
        model=raw.sim.mj_model
        efc_type=wp.to_torch(raw.sim.wp_data.efc.type)
        efc_id=wp.to_torch(raw.sim.wp_data.efc.id)
        efc_force=wp.to_torch(raw.sim.wp_data.efc.force)
        constraint_rows=torch.arange(efc_type.shape[1],device=args.device)[None,:]
        knee_ids=[model.joint('robot/'+name).id for name in ('lf1','rf1')]
        mujoco.mj_saveModel(model,str(args.output/(filename+'.mjb')),None)
        meta=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            actor_load='strict state_dict only',parameter_updates=0,seed=args.seed,cases=cases,
            training_config=training,noise_std=args.noise if training else 0.,
            warmup=args.warmup,hold=args.hold,dt=dt,physics_dt=raw.physics_dt,
            initial_nominal_height=WALK_INITIAL_HEIGHT,duration=duration,layout=layout,
            joint_names=JOINTS,reward_names=reward_names,reward_weights={k:v.weight for k,v in raw.cfg.rewards.items()},
            joint_limits=[[float(v) if math.isfinite(float(v)) else None for v in row]
                for row in robot.data.default_joint_pos_limits[0,joint_ids].cpu().numpy()],
            nominal_motor_pos=pd._nominal_pos[0].cpu().numpy().tolist(),
            pd=dict(leg_stiffness=pd.cfg.leg_stiffness,leg_damping=pd.cfg.leg_damping,
                wheel_stiffness=pd.cfg.wheel_stiffness,wheel_damping=pd.cfg.wheel_damping,
                effort_limits=pd._effort_limits.cpu().numpy().tolist(),max_wheel_velocity=pd.cfg.max_wheel_velocity,
                leg_position_scale=pd.cfg.leg_position_scale,wheel_velocity_scale=pd.cfg.wheel_velocity_scale),
            wheel_response=pd._wheel_response.cpu().numpy().tolist(),
            total_mass=float(model.body_mass.sum()),
            initial_target_heights=state.command[:,2].cpu().numpy().tolist(),
            timing='post-step qpos/vel and contact samples; reward uses applied_command before command update; derived physical signals may lag one 2ms substep')
        torch.manual_seed(2026)
        for tick in range(round(duration/dt)):
            t=tick*dt; commands(t)
            applied=state.command.clone(); action=policy(obs)
            estimated_velocity=built.alg.actor.encoder(obs['history'])/2.
            true_velocity_before=robot.data.root_com_lin_vel_b.clone()
            value=built.alg.critic(obs).flatten()
            actual=action+args.noise*torch.randn_like(action) if training else action
            if train.clip_actions is not None:actual=actual.clamp(-train.clip_actions,train.clip_actions)
            valid=live.clone()
            obs,reward,terminated,timeout,_=raw.step(actual)
            next_value=built.alg.critic(obs).flatten()
            td_delta=reward+built.alg.gamma*next_value*(~(terminated|timeout))-value
            assert torch.isfinite(reward).all() and torch.isfinite(raw.sim.data.qpos).all()
            data=robot.data; gravity=data.projected_gravity_b
            relative_world=data.body_link_pos_w[:,wheels]-data.body_link_pos_w[:,hips]
            quat=data.root_link_quat_w[:,None,:].expand(-1,2,-1)
            relative=quat_apply_inverse(quat.reshape(-1,4),relative_world.reshape(-1,3)).reshape(-1,2,3)
            delayed=pd._action_history[(pd._history_head-pd.action_delay_steps)%len(pd._action_history),pd._env_ids]
            wheel_delayed=pd._action_history[(pd._history_head-pd.wheel_action_delay_steps)%len(pd._action_history),pd._env_ids]
            legs,wh=pd._leg_action_ids,pd._wheel_action_ids
            desired=pd._nominal_pos[:,legs]+pd.cfg.leg_position_scale*delayed[:,legs]
            target=pd._leg_position_target(delayed[:,legs])
            motorq=raw.sim.data.qpos[:,pd._motor_q_adr]; motorv=raw.sim.data.qvel[:,pd._motor_v_adr]
            wheel_desired=pd.cfg.wheel_velocity_scale*wheel_delayed[:,wh]
            wheel_target=wheel_desired.clamp(-pd.cfg.max_wheel_velocity,pd.cfg.max_wheel_velocity)
            pre_leg=(pd.cfg.leg_stiffness*pd.kp_scale[:,legs]*(target-motorq[:,legs])
                     -pd.cfg.leg_damping*pd.kd_scale[:,legs]*motorv[:,legs])*pd.motor_strength[:,legs]
            pre_wheel=pd.kd_scale[:,wh]*(pd.cfg.wheel_stiffness*(wheel_target-motorv[:,wh])
                       -pd.cfg.wheel_damping*motorv[:,wh])*pd.motor_strength[:,wh]
            base=torch.cat((state.height_above_ground[:,None],
                torch.asin(gravity[:,:1].clamp(-1,1)).rad2deg(),
                torch.atan2(-gravity[:,1:2],-gravity[:,2:3]).rad2deg(),
                data.root_link_lin_vel_b,data.root_link_ang_vel_b),-1)
            active_rows=constraint_rows<raw.sim.data.nefc[:,None]
            knee_force=torch.stack([torch.where(active_rows
                & (efc_type==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)) & (efc_id==jid),
                efc_force.abs(),0.).sum(-1) for jid in knee_ids],-1)
            fields=dict(estimated_velocity=estimated_velocity,true_velocity_before=true_velocity_before,
                valid=valid[:,None].float(),applied_command=applied,next_command=state.command,
                raw_action=action,executed_action=actual,joint_q=raw.sim.data.qpos[:,qadr],joint_dq=raw.sim.data.qvel[:,vadr],
                pd_torque=pd._torques,actuator_torque=raw.sim.data.qfrc_actuator[:,pd._motor_v_adr],
                leg_target=target,leg_target_unclipped=desired,leg_torque_reconstructed=pre_leg,
                wheel_target=wheel_target,wheel_target_unclipped=wheel_desired,wheel_torque_reconstructed=pre_wheel,
                force=raw.scene['wheel_contact'].data.force.flatten(1),base=base,
                leg_length=relative.norm(dim=-1),swing=torch.atan2(-relative[...,0],-relative[...,2]).rad2deg(),
                wheel_z=data.body_link_pos_w[:,wheels,2]-raw.scene.env_origins[:,None,2],
                hip_z=data.body_link_pos_w[:,hips,2]-raw.scene.env_origins[:,None,2],relative_world=relative_world.flatten(1),
                qpos=raw.sim.data.qpos.clone(),qvel=raw.sim.data.qvel.clone(),knee_limit_force=knee_force,
                joint_constraint=raw.sim.data.qfrc_constraint[:,vadr],
                down_ticks=state.walk_down_ticks[:,None].float(),
                tracking_active=state.tracking_active[:,None].float(),
                reward_terms=raw.reward_manager._step_reward,actual_reward=reward[:,None],
                value_prediction=value[:,None],next_value_prediction=next_value[:,None],td_delta=td_delta[:,None],
                done=(terminated|timeout)[:,None].float())
            if not layout:
                start=0
                for key,value in fields.items():layout[key]=[start,start+value.shape[1]]; start+=value.shape[1]
            frames.append(torch.cat(tuple(fields.values()),-1).cpu().numpy())
            done=(terminated|timeout)&live
            end[done.cpu().numpy()]=(tick+1)*dt; live &= ~done
            reset=(terminated|timeout).nonzero(as_tuple=False).flatten()
            if len(reset):raw.reset(env_ids=reset); obs=raw.get_observations()
            if (tick+1)%200==0:print(filename,round((tick+1)*dt,2),'s live',int(live.sum()),flush=True)
        if filename=='static':
            response={}
            for h in (.222,.26,.29):
                alternative={key:value.clone() for key,value in obs.items()}
                alternative['actor'][:,8]=5.*h
                alternative['history'].reshape(count,5,25)[:,:,8]=5.*h
                response[str(h)]=policy(alternative).mean(0).cpu().numpy().tolist()
            meta['same_history_height_action_means']=response
        assert all(torch.equal(value,built.alg.actor.state_dict()[key]) for key,value in before.items())
        assert all(torch.equal(v,built.alg.critic.state_dict()[k]) for k,v in saved['critic_state_dict'].items())
        meta['critic_bitwise_unchanged']=True; meta['gamma']=built.alg.gamma
        meta['actor_bitwise_unchanged']=True; meta['episode_end']=end.tolist()
        array=np.asarray(frames); meta['summary']=summarize(array,meta)
        np.savez_compressed(args.output/(filename+'.npz'),records=array)
        write_json(args.output/(filename+'.json'),meta)
        return meta
    finally:raw.close()


def cases_for(speeds, accelerations, n, heights):
    result=[];start=0
    for height in heights:
        for accel in accelerations:
            for speed in speeds:
                result.append(dict(name=f'v{speed:g}-a{accel:g}-h{height}',speed=speed,acceleration=accel,
                    height=height,env_ids=list(range(start,start+n))));start+=n
    return result


def archived_forward(path):
    archive=np.load(path/'trajectory.npz');model=mujoco.MjModel.from_binary_path(str(path/'scene.mjb'))
    q=archive['qpos'];valid=archive['valid'];times=archive['time_s'];metrics=archive['metrics']
    names=archive['metric_names'].tolist();force=metrics[...,[names.index('left_support'),names.index('right_support')]]
    request=metrics[...,names.index('request_vx')];rows=[]
    knee_adr=model.jnt_qposadr[model.joint('robot/lf1').id]
    for e in range(valid.shape[1]):
        if valid[-1,e]:continue
        f=np.flatnonzero(valid[:,e])[-1];start=max(0,f-100)
        if request[start:f+1,e].max()<.5:continue
        lo=max(start,f-50)
        def first(mask):
            ids=np.flatnonzero(mask[lo:f+1]);return None if not len(ids) else int(ids[0]+lo)
        support=first((force[:,e]<1.).all(-1))
        upper=first(q[:,e,knee_adr]>=.12)
        near=first(q[:,e,knee_adr]>=.12-.03*(.12+.77))
        rows.append(dict(env=e,failure_s=float(times[f]),support_loss_s=None if support is None else float(times[support]),
            left_upper_cross_s=None if upper is None else float(times[upper]),
            left_near_upper_s=None if near is None else float(times[near]),
            upper_minus_support_ms=None if upper is None or support is None else float((times[upper]-times[support])*1000)))
    delta=[r['upper_minus_support_ms'] for r in rows if r['upper_minus_support_ms'] is not None]
    return dict(source=str(path),rows=rows,forward_failures=len(rows),q_upper_minus_both_support_ms=stats(delta),
        upper_before_support=sum(d<-.1 for d in delta),same_tick=sum(abs(d)<.1 for d in delta),
        upper_after_support=sum(d>.1 for d in delta),no_upper_cross=sum(r['left_upper_cross_s'] is None for r in rows),
        caveat='Archived poses/contact samples only; a qpos crossing is not a measured limit constraint force')


def analyze_runs(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ct_mjlab.envs.chuanliantui_stand.mdp import standing_pose
    report=dict(parameter_updates=0,tables=[],events=[],static={},kinematics={})
    all_inputs=[]
    for folder in args.inputs:
        for file in sorted(folder.glob('*.npz')):
            meta=json.loads(file.with_suffix('.json').read_text())
            data=np.load(file)['records']; times=np.arange(1,len(data)+1)*meta['dt']
            report['tables'].append(dict(input=str(file),summary=meta['summary']))
            model=mujoco.MjModel.from_binary_path(str(file.with_suffix('.mjb')))
            md=mujoco.MjData(model);maxlen=[];ranges=[]
            for knee,hip,wheel in zip(('lf1','rf1'),('lf0','rf0'),('lfwheel','rfwheel')):
                kid=model.joint('robot/'+knee).id; adr=model.jnt_qposadr[kid]
                limits=model.jnt_range[kid];lengths=[]
                for value in np.linspace(*limits,401):
                    md.qpos[:]=model.qpos0;md.qpos[adr]=value;mujoco.mj_kinematics(model,md)
                    lengths.append(np.linalg.norm(md.xpos[model.body('robot/'+wheel).id]-md.xpos[model.body('robot/'+hip).id]))
                maxlen.append(float(max(lengths)));ranges.append(limits.tolist())
            report['kinematics']['serial_length_envelope']=dict(max_lengths=maxlen,knee_ranges=ranges,
                note='Necessary hip-to-wheel serial-chain bound, not a proof that every point closes the four-bar loop')
            force=get(data,meta,'force').reshape(len(data),-1,2,3);support=-force[...,2]
            joint=get(data,meta,'joint_q');limitforce=get(data,meta,'knee_limit_force')
            pdtorque=get(data,meta,'pd_torque');acttorque=get(data,meta,'actuator_torque')
            target=get(data,meta,'leg_target');uncapped=get(data,meta,'leg_target_unclipped')
            end=np.asarray(meta['episode_end'])
            for case in meta['cases']:
                case_end=meta['warmup']+abs(case['speed'])/case['acceleration']+meta['hold']
                for e in case['env_ids']:
                    if not case['speed']>0 or not end[e]<case_end-1e-5:continue
                    f=min(len(data)-1,int(round(end[e]/meta['dt']))-1)
                    start=max(0,f-round(1./meta['dt']))
                    def onset(mask):
                        idx=np.flatnonzero(mask[start:f+1]);return None if not len(idx) else int(idx[0]+start)
                    low=onset((support[:,e]<1.).all(-1))
                    if low is None:low=onset(support[:,e].sum(-1)<.8*meta['total_mass']*9.81)
                    left=onset(limitforce[:,e,0]>.05);right=onset(limitforce[:,e,1]>.05)
                    saturation=onset((abs(pdtorque[:,e,[0,1,3,4]])>=39.6).any(-1))
                    wsat=onset((abs(pdtorque[:,e,[2,5]])>=3.861).any(-1))
                    tsat=onset((abs(target[:,e]-uncapped[:,e])>1e-6).any(-1))
                    velcap=onset((abs(get(data,meta,'wheel_target_unclipped')[:,e])>=150).any(-1))
                    row=dict(input=str(file),case=case['name'],env=e,fail_s=end[e],
                        both_support_loss_s=None if low is None else float(times[low]),
                        left_knee_limit_force_s=None if left is None else float(times[left]),
                        right_knee_limit_force_s=None if right is None else float(times[right]),
                        left_limit_minus_support_ms=None if low is None or left is None else (left-low)*meta['dt']*1000,
                        leg_torque_saturation_s=None if saturation is None else float(times[saturation]),
                        wheel_torque_saturation_s=None if wsat is None else float(times[wsat]),
                        leg_target_saturation_s=None if tsat is None else float(times[tsat]),
                        wheel_velocity_cap_s=None if velcap is None else float(times[velcap]),samples={})
                    reference=low if low is not None else f
                    for offset in (-.25,-.10,-.05,0.,.05):
                        k=max(0,min(f,reference+round(offset/meta['dt'])))
                        rel=get(data,meta,'relative_world')[k,e].reshape(2,3)
                        length=get(data,meta,'leg_length')[k,e]
                        unit=rel/length[:,None]
                        needed=(get(data,meta,'hip_z')[k,e]-.06)/(-unit[:,2])
                        md.qpos[:]=get(data,meta,'qpos')[k,e];md.qvel[:]=get(data,meta,'qvel')[k,e]
                        mujoco.mj_forward(model,md)
                        rotation=md.xmat[model.body('robot/base_link').id].reshape(3,3)
                        velocities=[]
                        for wheel in ('lfwheel','rfwheel'):
                            jp=np.zeros((3,model.nv));jr=jp.copy()
                            mujoco.mj_jacBody(model,md,jp,jr,model.body('robot/'+wheel).id)
                            linear=jp@md.qvel;angular=jr@md.qvel
                            forward=float(linear@rotation[:,0]);rolling=float(angular@rotation[:,1])
                            velocities.append(dict(vertical_m_s=float(linear[2]),forward_m_s=forward,
                                absolute_rolling_rad_s=rolling,contact_slip_m_s=forward-.06*rolling))
                        contact_body=(-force[k,e])@rotation
                        ratio=np.linalg.norm(force[k,e,:,:2],axis=-1)/np.maximum(support[k,e],1.)
                        row['samples'][str(offset)]=dict(contact_force_body=contact_body.tolist(),friction_ratio=ratio.tolist(),
                            wheel_velocity_fk=velocities,time_s=float(times[k]),
                                q=joint[k,e].tolist(),dq=get(data,meta,'joint_dq')[k,e].tolist(),
                                leg_target=target[k,e].tolist(),action=get(data,meta,'raw_action')[k,e].tolist(),
                                pd_torque=pdtorque[k,e].tolist(),actuator_torque=acttorque[k,e].tolist(),
                                knee_limit_force=limitforce[k,e].tolist(),support=support[k,e].tolist(),
                                base=get(data,meta,'base')[k,e].tolist(),
                                length=length.tolist(),swing=get(data,meta,'swing')[k,e].tolist(),
                                wheel_z=get(data,meta,'wheel_z')[k,e].tolist(),
                                required_length_at_chosen_direction=needed.tolist(),
                                beyond_serial_envelope=(needed>np.asarray(maxlen)).tolist())
                    report['events'].append(row)
            # 按预设规则选第一个+1失败样本，绘制失载前后1秒，不挑最好看的案例。
            forward=[r for r in report['events'] if r['input']==str(file) and r['case'].startswith('v1-')]
            if forward:
                row=forward[0];e=row['env'];f=min(len(data)-1,int(round(row['fail_s']/meta['dt']))-1)
                loss=row['both_support_loss_s'] or row['fail_s'];sel=(times>=loss-1.)&(times<=row['fail_s'])
                tt=times[sel]-loss
                fig,ax=plt.subplots(4,2,figsize=(14,12),sharex=True)
                series=[('Leg length (m)',get(data,meta,'leg_length'),['L','R']),
                    ('Swing (deg)',get(data,meta,'swing'),['L','R']),
                    ('Knee q (rad)',joint[...,4:6],['lf1','rf1']),
                    ('Joint-limit constraint torque (Nm)',limitforce,['L','R']),
                    ('Actual wheel torque (Nm)',acttorque[...,[2,5]],['L','R']),
                    ('Wheel dq (rad/s)',get(data,meta,'joint_dq')[...,6:8],['L','R']),
                    ('Support (N)',support,['L','R']),
                    ('Base height / vx',get(data,meta,'base')[...,[0,3]],['height m','vx m/s'])]
                for axis,(title,values,labels) in zip(ax.flat,series):
                    axis.plot(tt,values[sel,e]);axis.set_title(title);axis.legend(labels);axis.grid(alpha=.3);axis.axvline(0,c='k',ls='--')
                ax[1,0].axhline(ranges[0][1],c='C0',ls=':');ax[1,0].axhline(ranges[1][0],c='C1',ls=':')
                for axis in ax[-1]:axis.set_xlabel('Seconds relative to first both-support loss (<1 N each)')
                fig.suptitle(f'{file.stem}, env {e}; physical inputs unchanged');fig.tight_layout()
                fig.savefig(args.output/(file.stem+'-failure.png'),dpi=160);plt.close(fig)
                fig,ax=plt.subplots(3,2,figsize=(14,10),sharex=True)
                for j,label in enumerate(('lf0','rf0')):
                    qi=0 if j==0 else 2
                    ax[0,0].plot(tt,joint[sel,e,qi],label=label+' q')
                    ax[0,0].plot(tt,target[sel,e,qi],ls='--',label=label+' target')
                for j,label in enumerate(('lf00','rf00')):
                    qi=1 if j==0 else 3
                    ax[0,1].plot(tt,joint[sel,e,qi],label=label+' q')
                    ax[0,1].plot(tt,target[sel,e,qi],ls='--',label=label+' target')
                ax[1,0].plot(tt,get(data,meta,'joint_dq')[sel,e,:4]);ax[1,0].legend(JOINTS[:4])
                ax[1,1].plot(tt,pdtorque[sel,e][:,[0,1,3,4]]);ax[1,1].legend(JOINTS[:4])
                ax[1,1].axhline(40,c='k',ls=':');ax[1,1].axhline(-40,c='k',ls=':')
                ax[2,0].plot(tt,get(data,meta,'raw_action')[sel,e]);ax[2,0].legend(('lf0','lf00','Lwheel','rf0','rf00','Rwheel'))
                ax[2,1].plot(tt,get(data,meta,'wheel_target')[sel,e],ls='--')
                ax[2,1].plot(tt,get(data,meta,'joint_dq')[sel,e,6:8]);ax[2,1].legend(('L target','R target','L dq','R dq'))
                titles=('Hip q / requested target (rad)','Parallel motor q / target (rad)',
                    'Leg joint velocity (rad/s)','Nominal leg PD torque (Nm, cap +/-40)',
                    'Raw policy outputs','Wheel relative dq / target (rad/s, cap +/-150)')
                for axis,title in zip(ax.flat,titles):
                    axis.set_title(title);axis.axvline(0,c='k',ls='--');axis.grid(alpha=.3)
                for axis in ax[-1]:axis.set_xlabel('Seconds relative to both-support loss')
                fig.suptitle(f'{file.stem}, env {e}');fig.tight_layout()
                fig.savefig(args.output/(file.stem+'-actions.png'),dpi=160);plt.close(fig)
            # 末两秒的正反向姿态对照，避免只在失败瞬间比较。
            pose_rows=[]
            for case in meta['cases']:
                ids=case['env_ids'];endpoint=meta['warmup']+abs(case['speed'])/case['acceleration']+meta['hold']
                valid=get(data,meta,'valid')[:,ids,0]>.5
                sel=valid&(times[:,None]>=meta['warmup']+abs(case['speed'])/case['acceleration']+.5)&(times[:,None]<=endpoint)
                pose_rows.append(dict(case=case['name'],
                    length=[stats(get(data,meta,'leg_length')[:,ids,j][sel]) for j in range(2)],
                    swing=[stats(get(data,meta,'swing')[:,ids,j][sel]) for j in range(2)],
                    knees=[stats(joint[:,ids,4+j][sel]) for j in range(2)]))
            report.setdefault('posture',[]).append(dict(input=str(file),cases=pose_rows))
            if 'estimated_velocity' in meta['layout']:
                estimate=get(data,meta,'estimated_velocity')[...,0]
                truth=get(data,meta,'true_velocity_before')[...,0]
                feedback=[]
                for case in meta['cases']:
                    ids=case['env_ids'];valid=get(data,meta,'valid')[:,ids,0]>.5
                    mask=valid & (times[:,None]>=meta['warmup']+.3)
                    feedback.append(dict(case=case['name'],estimated_vx=stats(estimate[:,ids][mask]),
                        true_vx=stats(truth[:,ids][mask]),estimation_error=stats((estimate-truth)[:,ids][mask])))
                report['feedback']=feedback
                fig,axes=plt.subplots(1,3,figsize=(15,4))
                selected=[next(c for c in meta['cases'] if c['speed']==v and c['acceleration']==.5) for v in (1.,-1.,.3)]
                for axis,case in zip(axes,selected):
                    e=case['env_ids'][0];mask=get(data,meta,'valid')[:,e,0]>.5
                    axis.plot(times[mask]-meta['dt'],truth[mask,e],label='true body COM vx')
                    axis.plot(times[mask]-meta['dt'],estimate[mask,e],label='encoder vx')
                    axis.plot(times[mask]-meta['dt'],get(data,meta,'applied_command')[mask,e,0],label='applied command')
                    axis.set_title(case['name']);axis.set_xlabel('Time (s)');axis.grid(alpha=.3);axis.legend()
                fig.tight_layout();fig.savefig(args.output/'velocity-estimator.png',dpi=160);plt.close(fig)
            if file.stem=='static':
                report['static']['policy_rollouts']=meta['summary']
                # 仅评估必要几何/膝余量和高度核，不把这些IK姿态冒充动态稳站。
                poses=[]
                for h in (.222,.26,.29):
                    pose=standing_pose(h);md.qpos[:]=model.qpos0
                    for name,value in pose['joint_positions'].items():md.qpos[model.jnt_qposadr[model.joint('robot/'+name).id]]=value
                    md.qpos[2]=pose['height'];mujoco.mj_kinematics(model,md)
                    kq=np.array([md.qpos[model.jnt_qposadr[model.joint('robot/'+n).id]] for n in ('lf1','rf1')])
                    lim=np.asarray(ranges);penetration=np.maximum(lim[:,0]+.05-kq,kq-lim[:,1]+.05).clip(0)
                    cost=max(-1.,-.3*np.mean((penetration/.05)**2))
                    poses.append(dict(height=h,knees=kq.tolist(),knee_margin_weighted=float(cost),
                        height_kernel_if_at_target=1.,height_kernel_if_stay222=float(np.exp(-(h-.222)**2/.0025))))
                report['static']['nominal_closed_pose_geometry']=poses
        all_inputs.append(str(folder))
    if args.archive is not None:report['archived_forward']=archived_forward(args.archive)
    report['inputs']=all_inputs
    delays=[r['left_limit_minus_support_ms'] for r in report['events'] if r['left_limit_minus_support_ms'] is not None]
    report['event_summary']=dict(failed_forward_samples=len(report['events']),left_limit_minus_support_ms=stats(delays),
        left_limit_precedes_support=sum(v<0 for v in delays),ties=sum(v==0 for v in delays),
        left_limit_follows_support=sum(v>0 for v in delays),no_observed_left_limit=sum(r['left_knee_limit_force_s'] is None for r in report['events']),
        leg_torque_saturated=sum(r['leg_torque_saturation_s'] is not None for r in report['events']),
        wheel_torque_saturated=sum(r['wheel_torque_saturation_s'] is not None for r in report['events']),
        leg_target_clipped=sum(r['leg_target_saturation_s'] is not None for r in report['events']),
        wheel_velocity_capped=sum(r['wheel_velocity_cap_s'] is not None for r in report['events']))
    return report


def main(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    report=dict(verified=False,parameter_updates=0,phase=args.phase,runs=[],scope='冻结策略诊断；非训练、非正式行为验收')
    try:
        if args.phase=='analyze':
            report['analysis']=analyze_runs(args)
        elif args.phase=='speeds':
            for acceleration in args.accelerations:
                cases=cases_for([v*s for v in args.speeds for s in (1,-1)],[acceleration],args.num_per_case,[args.height])
                report['runs'].append(run_batch(args,cases,args.checkpoint,f'model999-a{acceleration:g}'))
        elif args.phase=='checkpoints':
            for iteration in args.iterations:
                checkpoint=args.checkpoint.with_name(f'model_{iteration}.pt')
                cases=cases_for([1.,-1.],args.accelerations,args.num_per_case,[args.height])
                report['runs'].append(run_batch(args,cases,checkpoint,f'model{iteration}'))
        elif args.phase=='feedback':
            cases=cases_for([1.,-1.,.3,.4],args.accelerations,args.num_per_case,[args.height])
            report['runs'].append(run_batch(args,cases,args.checkpoint,'feedback'))
        elif args.phase=='noise':
            cases=cases_for([1.,-1.,0.],[1.],args.num_per_case,[None])
            report['runs'].append(run_batch(args,cases,args.checkpoint,'noise',training=True))
        elif args.phase=='static':
            cases=cases_for([0.],[1.],args.num_per_case,[.222,.26,.29])
            report['runs'].append(run_batch(args,cases,args.checkpoint,'static'))
        baseline_path=args.output.parent/'production-before.json'
        if baseline_path.exists():
            baseline=json.loads(baseline_path.read_text())
            changed=[path for path,sha in baseline.items() if digest(path)!=sha]
            report['observed_document_changes']=[path for path in changed if path.startswith('docs/')]
            report['production_changed']=[path for path in changed if not path.startswith('docs/')]
            assert not report['production_changed'],report['production_changed']
        report['verified']=True
    finally:
        report['source_sha256']=digest(args.output/'source.py')
        write_json(args.output/'result.json',report)
    print(json.dumps([r['summary'] for r in report['runs']],ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('speeds','checkpoints','noise','static','feedback','analyze'),default='speeds')
    parser.add_argument('--archive',type=Path,help='Optional existing natural-command evaluation directory, read only')
    parser.add_argument('--inputs',type=Path,nargs='+',default=[],help='Recorded phase directories for offline analyze')
    parser.add_argument('--num-per-case',type=int,default=32)
    parser.add_argument('--speeds',type=float,nargs='+',default=[.3,.5,.7,.85,1.])
    parser.add_argument('--accelerations',type=float,nargs='+',default=[.5,2.])
    parser.add_argument('--iterations',type=int,nargs='+',default=[200,400,600,800])
    parser.add_argument('--height',type=float,default=.26)
    parser.add_argument('--warmup',type=float,default=2.)
    parser.add_argument('--hold',type=float,default=5.)
    parser.add_argument('--noise',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Diagnosis cannot train')):
        main(parser.parse_args())
