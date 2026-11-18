"""GPU free closed-loop takeover from measured6838 states/history; no future measured-state forcing."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.signal import periodogram

from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen
from scripts.agent.fudan_ppo_walk import inspect_slope5to20_mu08 as study

ROOT=frozen.ROOT
REAL=ROOT/'artifacts/real2sim/20261117-high-length6838-01'
SHA='8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
STATES=('initial_history_ready','stable','height_return','early_growth')
ARMS=(('raw_short',0.,0),('sham_short',0.,0),('bias_short',-2.,0),('raw_delay15',0.,15),('bias_delay15',-2.,15))


def rotate_gravity(value,degrees):
    co,si=np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))
    result=value.copy();result[...,3]=co*value[...,3]+si*value[...,5]
    result[...,5]=co*value[...,5]-si*value[...,3]
    return result


def band(y,fs=100.):
    f,p=periodogram(y,fs=fs,axis=0);mask=(f>=8)&(f<=14)
    return np.sqrt(np.trapezoid(p[mask],f[mask],axis=0)),f[mask][np.argmax(p[mask],axis=0)]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actuator-study',action='store_true');p.add_argument('--imu-study',action='store_true')
    p.add_argument('--targets-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py')
    assert frozen.sha(study.RUN/'model_final.pt')==SHA
    for name,digest in json.loads((study.RUN/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert frozen.sha(study.RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest
        elif name.startswith('resources/'):assert frozen.sha(ROOT/name)==digest
    geometry=json.loads((REAL/'real-initial-geometry.json').read_text())
    state_rows={r['label']:r for r in geometry['rows']}
    assert all(r['closure_error_m']<1e-8 and not r['joint_limit_violations'] for r in state_rows.values())
    with (REAL/'raw/高腿长抖腿/6838高腿长对照_162432/full-pd.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['rl_engaged']=='1' and r['warmup']=='0' and r['pd_valid']=='1']
    unique=[r for i,r in enumerate(rows) if i==0 or r['obs_sequence']!=rows[i-1]['obs_sequence']]
    real_t=np.array([float(r['h7_us'])/1e6 for r in unique])
    commands=np.array([[float(r['obs_6'])/2,float(r['obs_7'])/.25,float(r['obs_8'])/5] for r in unique])
    real_observations=np.array([[float(r[f'obs_{j}']) for j in range(25)] for r in unique])
    real_targets=np.array([[float(r['target_'+j]) for j in ('lf','lr','lw','rf','rr','rw')] for r in unique])
    arms=[(*arm,[0,0,0,0],0) for arm in ARMS]
    if a.actuator_study:
        arms=[('raw_short',0.,0,[0,0,0,0],0),('bias_short',-2.,0,[0,0,0,0],0),
              ('bias_actuator17',-2.,0,[17,17,17,17],0),('bias_actuator_axes',-2.,0,[16,20,18,18],0),
              ('raw_actuator_axes',0.,0,[16,20,18,18],0),('bias_training15',-2.,15,[0,0,0,0],0)]
    if a.imu_study:
        arms=[('bias_ideal',-2.,0,[0,0,0,0],0),('bias_imu20',-2.,0,[0,0,0,0],20),
              ('bias_motor17',-2.,0,[17,17,17,17],0),('bias_combined',-2.,0,[17,17,17,17],20),
              ('raw_combined',0.,0,[17,17,17,17],20),('sham_combined',-2.,0,[17,17,17,17],20)]
    if a.targets_only:
        arms=[('ideal_target',0.,0,[0,0,0,0],0),('delay17_target',0.,0,[17,17,17,17],0),
              ('axes_target',0.,0,[16,20,18,18],0)]
    samples=[dict(id=arm+'__'+state,state=state,projection=angle,delay_steps=delay,torque_delay_ms=torque_delay,imu_delay_ms=imu_delay,
        start_h7_s=state_rows[state]['h7_us']/1e6,
        duration=round((real_t[-1]-state_rows[state]['h7_us']/1e6)/.01)*.01)
        for arm,angle,delay,torque_delay,imu_delay in arms for state in STATES]
    frozen.write(out/'plan.json',dict(samples=samples,checkpoint_sha256=SHA,forced_target_replay=a.targets_only,
        measured_initial_geometry=geometry,intervention='0/-2deg gravity projection and zero/15ms additional FIFO hypotheses; original control and physical model',
        limits=['Root height/passive joints inferred from geometry; root linear velocity assumed0',
                'Measured history seeds controller once; no measured gyro/pose is injected afterwards',
                '0ms means zero extra simulated delay, not measured zero hardware latency',
                'All future motion/height commands replayed; no periodic11Hz force or noise added'],
        actuator_hypothesis=('Pure leg-output delay17ms or16/20/18/18ms from request-feedback phase; measurement filtering not identified separately, so not a calibrated actuator model' if a.actuator_study or a.imu_study else None),
        imu_hypothesis=('Only IMU gyro/gravity packet delayed20ms, joint feedback stays fresh; motivated by leg-height/IMU phase under assumed two-wheel support, not a proven transport delay' if a.imu_study else None),
        reproduction='8-14Hz gyroX/left-rear target/request RMS within factor2 of same real window, finite and no60deg fall; already-growing starts are not proof of spontaneous onset',
        optimizer_updates=0,wall_limit_s=600))
    import torch
    frozen.RUN=study.RUN
    assert not any(n=='ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    sys.meta_path.insert(0,frozen.FrozenTrainingSource())
    from mjlab.managers.event_manager import RecomputeLevel
    from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record,SENSOR_SLOTS
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from ct_mjlab.fudan_ppo_walk.measurement import configure_samples,update_evaluation_commands,gpu_row,FIELDS
    torch.set_num_threads(1);policy,meta=load_policy(study.RUN/'model_final.pt','cuda:0')
    before={k:v.cpu().clone() for k,v in policy.state_dict().items()}
    cfg=FudanWalkCfg()
    for section,fields in json.loads((study.RUN/'effective-training-config.json').read_text()).items():
        for key,value in fields.items():setattr(getattr(cfg,section),key,value)
    cfg.terrain.mode='plane';cfg.observations.critic_height_scan=False
    conditions=replace(from_record(meta['disturbances']),pushes=False,external_force_range=(0.,0.),
        external_torque_range=(0.,0.),action_delay_ms=(0.,20.),sensor_delay_ms=(0.,20.))
    old_cfg=physical.make_env_cfg
    def make_cfg(*args,**kwargs):
        c=old_cfg(*args,**kwargs);c.sim.nconmax,c.sim.njmax=64,256;return c
    with patch.object(physical,'make_env_cfg',make_cfg):
        env=physical.FudanEnv(len(samples),seed=50001,randomized=False,cfg=cfg,disturbances=conditions,reset_contract=meta['reset_contract'])
    sim=env.env.sim;act=env.action;device=env.device
    sim.expand_model_fields(FIELDS)
    items=[dict(id=s['id'],seed=50001,randomized=False,case=dict(id=s['state'],category='zero',duration=s['duration'],segments=[dict(start=0.,duration=s['duration'],command=commands[0].tolist())])) for s in samples]
    configure_samples(env,items,out/'fixtures')
    geoms=np.flatnonzero(np.isin(sim.mj_model.geom_bodyid,env.robot.indexing.body_ids.cpu().numpy())|
                         (sim.mj_model.geom_bodyid==sim.mj_model.body('terrain').id)).tolist()
    sim.model.geom_friction[:,geoms,0]=.8;sim.recompute_constants(RecomputeLevel.set_const)
    env.env.scene.env_origins.zero_()
    history=[];qpos=[];qvel=[]
    with np.load(REAL/'real-initial-geometry.npz') as data:
        for s in samples:
            qpos.append(data[s['state']+'_qpos'].copy());qvel.append(data[s['state']+'_qvel'].copy())
            history.append(rotate_gravity(data[s['state']+'_history'].copy(),2.))
    history=torch.as_tensor(np.asarray(history),dtype=torch.float32,device=device)
    sim.data.qpos[:]=torch.as_tensor(np.asarray(qpos),device=device,dtype=torch.float32)
    sim.data.qvel[:]=torch.as_tensor(np.asarray(qvel),device=device,dtype=torch.float32)
    sim.data.ctrl.zero_();sim.data.qacc_warmstart.zero_();sim.forward();env.env.scene.update(dt=0.);sim.sense()
    previous=history[:,-1,19:25];act._raw_actions[:]=previous;act._control_actions[:]=previous
    act.last_sample_q.copy_(sim.data.qpos[:,act._motor_q_adr]);act.sample_velocity.copy_(sim.data.qvel[:,act._motor_v_adr])
    env.previous[:,:,0]=previous;env.previous[:,:,1]=history[:,-2,19:25]
    env.previous_vel.copy_(act.sample_velocity);env.previous_position.copy_(env.robot.data.root_link_pos_w)
    env._update_state();env.noise_mask.zero_();env.history=history.reshape(len(samples),125).clone()
    env._history_frame=history[:,-1].clone();env._history_needs_fill.zero_()
    state=act.disturbances;state.reset(slice(None),act.sensor_frame())
    state.action_steps[:]=torch.tensor([s['delay_steps'] for s in samples],device=device)
    state.sensor_steps[:]=state.action_steps
    for slot in range(len(state.actions)):
        age=(len(state.actions)-slot)%len(state.actions);back=min(4,max(0,age//10))
        state.actions[slot]=history[:,4-back,19:25]
    for slot in range(len(state.sensors)):
        age=(len(state.sensors)-1-slot)%len(state.sensors);back=min(4,max(0,age//10))
        state.sensors[slot]=history[:,4-back,list(SENSOR_SLOTS)]
    obs=history[:,-1].clone();initial={k:getattr(sim.data,k).cpu().numpy().copy() for k in ('qpos','qvel','ctrl','qacc_warmstart')}
    initial.update(history=env.history.cpu().numpy(),actions=state.actions.cpu().numpy(),sensors=state.sensors.cpu().numpy(),
        action_steps=state.action_steps.cpu().numpy(),sensor_steps=state.sensor_steps.cpu().numpy())
    for i in range(len(STATES)):
        for key in ('qpos','qvel','ctrl','qacc_warmstart','history'):
            assert all(np.array_equal(initial[key][i],initial[key][i+4*j]) for j in range(len(arms))),key
    np.savez_compressed(out/'initial.npz',**initial)
    np.savez_compressed(out/'initial-assumptions.npz',world_linear_velocity=np.zeros((len(samples),3)))
    recorded=[];pd=[];steps=[0];errors=[0.];old_apply=act.apply_actions
    output_rows=[]
    if a.actuator_study or a.imu_study or a.targets_only:
        motor_ids=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        leg_ids=[motor_ids[i] for i in (0,1,3,4)]
        length=21;buffer=torch.zeros(length,len(samples),4,device=device)
        for j,s in enumerate(samples):
            for slot in range(length):
                age=(length-slot)%length or length
                stamp=s['start_h7_s']-age*.001
                prior=max((r for r in rows if float(r['h7_us'])/1e6<=stamp),key=lambda r:float(r['h7_us']),default=rows[0])
                buffer[slot,j]=torch.tensor([float(prior['pd_model_nm_'+leg]) for leg in ('lf','lr','rf','rr')],device=device)
        delays=torch.tensor([s['torque_delay_ms'] for s in samples],device=device,dtype=torch.long)
        world=torch.arange(len(samples),device=device)[:,None];motor=torch.arange(4,device=device)[None,:]
        head=[0];old_write=env.env.scene.write_data_to_sim
        def write_output():
            old_write();requested=sim.data.ctrl[:,leg_ids].clone();buffer[head[0]]=requested
            delivered=buffer[(head[0]-delays)%length,world,motor]
            sim.data.ctrl[:,leg_ids]=delivered
            output_rows.append(torch.stack((requested,delivered),1).cpu().numpy())
            head[0]=(head[0]+1)%length
        env.env.scene.write_data_to_sim=write_output
    def apply():
        update=act._pd_tick==0;old_apply();steps[0]+=1
        if update:
            q=sim.data.qpos[:,act._motor_q_adr];v=sim.data.qvel[:,act._motor_v_adr]
            target=act._nominal_pos+.5*act._control_actions;legs=act._leg_action_ids;wheels=act._wheel_action_ids
            target[:,legs]=torch.maximum(torch.minimum(target[:,legs],act._reference_pos[:,legs]+3.14),act._reference_pos[:,legs]-3.14)
            target[:,wheels]=(10*act._control_actions[:,wheels]).clamp(-150.,150.)
            expected=torch.zeros_like(q);expected[:,legs]=20*(target[:,legs]-q[:,legs])-v[:,legs];expected[:,wheels]=.1*(target[:,wheels]-v[:,wheels]);expected=expected.clamp(-act._effort_limits,act._effort_limits)
            errors[0]=max(errors[0],float((expected-act.torques).abs().max()))
            pd.append(torch.stack((target,q,v,act.torques),1).cpu().numpy())
    act.apply_actions=apply
    imu_ring=[history[:,-3,:6].clone(),history[:,-2,:6].clone()]
    imu_history=history[:,:,:6].clone()
    for i,s in enumerate(samples):
        back=s['imu_delay_ms']//10
        if back:imu_history[i]=history[i,torch.arange(5,device=device).sub(back).clamp_min(0),:6]
    count=round(max(s['duration'] for s in samples)/.01);start=time.time();gyro_inputs=[]
    try:
        with torch.no_grad(),patch('torch.optim.Adam.step',side_effect=AssertionError('No learning')):
            for tick in range(count):
                assert time.time()-start<600
                t=tick*.01;cmd=[commands[np.clip(np.searchsorted(real_t,s['start_h7_s']+t+1e-8,side='right')-1,0,len(real_t)-1)].tolist() for s in samples]
                obs=update_evaluation_commands(env,cmd,obs);current=obs.clone();hist=env.history.clone()
                imu_ring.append(obs[:,:6].clone())
                held=obs[:,:6].clone()
                for i,s in enumerate(samples):
                    if s['imu_delay_ms']:held[i]=imu_ring[-1-s['imu_delay_ms']//10][i]
                if tick==0:imu_history[:,-1]=held
                else:imu_history=torch.cat((imu_history[:,1:],held[:,None,:]),1)
                current[:,:6]=held;hist.view(-1,5,25)[:,:,:6]=imu_history
                imu_ring=imu_ring[-3:]
                for i,s in enumerate(samples):
                    if s['projection']:
                        co,si=np.cos(np.deg2rad(s['projection'])),np.sin(np.deg2rad(s['projection']))
                        g=current[i,3:6].clone();h=hist.view(-1,5,25)[i,:,3:6].clone()
                        current[i,3]=co*g[0]+si*g[2];current[i,5]=co*g[2]-si*g[0]
                        hist.view(-1,5,25)[i,:,3]=co*h[:,0]+si*h[:,2];hist.view(-1,5,25)[i,:,5]=co*h[:,2]-si*h[:,0]
                if a.targets_only:
                    # 只验证机械/伺服通路；真实未来目标作输入，不能称为自主起振复现。
                    indices=[int(np.argmin(abs(real_t-(s['start_h7_s']+t)))) for s in samples]
                    target=real_targets[indices].copy()
                    target[:,[0,1,3,4]]=(target[:,[0,1,3,4]]-np.asarray(meta['control']['nominal_leg_positions']))/.5
                    target[:,[2,5]]/=10.
                    action=torch.as_tensor(target,device=device,dtype=torch.float32)
                else:action,_=policy.act_inference(current,hist)
                gyro_inputs.append(torch.stack((obs[:,0]/.25,current[:,0]/.25),-1).cpu().numpy())
                obs,_,_,_,_,_=env.step(action,auto_reset=False,resample=False)
                recorded.append(gpu_row(env.env).cpu().numpy());
                if tick%100==0:
                    assert not sim.wp_data.overflow.numpy().any()
                    assert float(sim.data.xfrc_applied.abs().max())==0.
                    print(json.dumps(dict(step=tick,total=count,worlds=len(samples))),flush=True)
    finally:env.close()
    assert errors[0]<2e-5 and frozen.sha(study.RUN/'model_final.pt')==SHA
    assert all(torch.equal(v.cpu(),before[k]) for k,v in policy.state_dict().items())
    recorded=np.asarray(recorded);pd=np.asarray(pd);gyro_inputs=np.asarray(gyro_inputs)
    np.savez_compressed(out/'trace.npz',policy_time=np.arange(count)*.01,body_time=np.arange(1,count+1)*.01,
        pd_time=np.arange(len(pd))*.005,body=recorded,motor=pd,gyro_input=gyro_inputs)
    if output_rows:np.savez_compressed(out/'actuator-output.npz',values=np.asarray(output_rows))
    results=[]
    for i,s in enumerate(samples):
        t=s['start_h7_s']+np.arange(1,count+1)*.01;tp=s['start_h7_s']+np.arange(len(pd))*.005
        lo=max(380.773242,s['start_h7_s']);hi=385.393247
        take=(t>=lo-1e-8)&(t<hi-1e-8)&(t<s['start_h7_s']+s['duration']-1e-8)
        takep=(tp>=lo-1e-8)&(tp<hi-1e-8)&(tp<s['start_h7_s']+s['duration']-1e-8)
        real=(real_t>=lo)&(real_t<hi)
        real_g=real_observations[real,0]/.25
        real_lr=.6083241726212976+.5*real_observations[real,20]
        real_g_amp,_=band(real_g);sim_g_amp,peak=band(recorded[take,i,9]);target_amp,_=band(pd[takep,i,0,1],200.)
        height=recorded[take,i,2];max_tilt=float(abs(recorded[take,i,12:14]).max())
        results.append(dict(**s,window_h7_s=[lo,hi],gyroX_8to14_rms=float(sim_g_amp),gyro_peak_Hz=float(peak),
            real_gyroX_8to14_rms=float(real_g_amp),LR_target_8to14_rms=float(target_amp),
            LR_torque_8to14_rms=float(band(pd[takep,i,3,1],200.)[0]),max_tilt_deg=max_tilt,
            body_height_mean=float(height.mean()),finite=bool(np.isfinite(recorded[take,i]).all()),
            reached_real_gyro_amplitude=bool(.5*real_g_amp<=sim_g_amp<=2*real_g_amp),
            spontaneous_onset_eligible=(not a.targets_only and
                s['state'] in ('initial_history_ready','stable','height_return'))))
    frozen.write(out/'result.json',dict(rows=results,physics_steps=steps[0],PD_error_nm=errors[0],weights_unchanged=True,
        optimizer_updates=0,forced_target_replay=a.targets_only,
        limitations=json.loads((out/'plan.json').read_text())['limits'],frozen_modules=frozen.LOADED))
    print(json.dumps([{k:r[k] for k in ('id','gyroX_8to14_rms','real_gyroX_8to14_rms','LR_target_8to14_rms','max_tilt_deg')} for r in results]),flush=True)


if __name__=='__main__':main()
