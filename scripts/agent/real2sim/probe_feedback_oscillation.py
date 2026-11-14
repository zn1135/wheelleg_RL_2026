"""Isolated GPU sensitivity probe: feedback delay and actuator weakness, not calibration."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from scipy.signal import detrend,find_peaks,welch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config,SENSOR_SLOTS
from ct_mjlab.envs.fudan_ppo_walk import height_reward,posture_reward
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.fudan_ppo_walk.measurement import gpu_row,arrays,update_evaluation_commands
from .replay_usb_pd_targets import sha,write


def profiles():
    rows=[dict(name='baseline'),dict(name='baseline_repeat')]
    rows += [dict(name=f'policy_sensor_{v}ms',sensor_ms=v) for v in (8,12,20,30)]
    rows += [dict(name=f'action_{v}ms',action_ms=v) for v in (20,30,40)]
    rows += [dict(name=f'wheel_gain_{v:g}',wheel_gain=v) for v in (.7,.45,.25)]
    rows += [dict(name=f'wheel_lag_{v}ms',wheel_lag_ms=v) for v in (10,20,40)]
    rows += [dict(name=f'leg_gain_{v:g}',leg_gain=v) for v in (.7,.45)]
    rows += [dict(name=f'leg_PD_velocity_{v}ms',pd_velocity_ms=v) for v in (4,8)]
    rows += [dict(name='wheel045_sensor12',wheel_gain=.45,sensor_ms=12),
             dict(name='wheel045_action20',wheel_gain=.45,action_ms=20),
             dict(name='wheel045_lag20',wheel_gain=.45,wheel_lag_ms=20)]
    return rows


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    candidates=json.loads(a.profiles.read_text()) if a.profiles else profiles();assert candidates[0]['name']=='baseline' and len(candidates)<=30
    source=a.experiment/'closed-loop-from-log-01';initial=np.load(source/'zero/fixtures/baseline_initial.npz')
    policy,metadata=load_policy(a.checkpoint,'cuda:0');cfg=FudanWalkCfg()
    cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    height_reward.configure(cfg,height_reward.from_record(metadata.get('height_reward')))
    posture_reward.configure(cfg,posture_reward.from_record(metadata.get('posture_reward')))
    original=from_sim2sim_config();assert original.action_delay_ms==(10.,10.) and original.sensor_delay_ms==(4.,4.)
    disturbances=replace(original,action_delay_ms=(10.,max(10.,max(c.get('action_ms',10.) for c in candidates))),
        sensor_delay_ms=(4.,max(4.,max(c.get('sensor_ms',4.) for c in candidates))))
    assert not disturbances.pushes and disturbances.external_force_range==(0.,0.) and disturbances.external_torque_range==(0.,0.)
    count=len(candidates);duration=30.;steps=round(duration/.01)
    commands=np.asarray([[0.,0.,c.get('height',a.height)] for c in candidates])
    plan=dict(profiles=candidates,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
        initial_fixture=str((source/'zero/fixtures/baseline_initial.npz').resolve()),initial_fixture_sha256=sha(source/'zero/fixtures/baseline_initial.npz'),
        command=[0.,0.,a.height],commands=commands.tolist(),duration_s=duration,physics_dt_s=.002,policy_dt_s=.01,PD=dict(leg_p=60.,leg_d=2.,wheel_p=.1),
        scope='Unmeasured counterfactual response hypotheses. Output gain/lag applied only in diagnostic hook after nominal PD limits; production mechanical-wheel guard and defaults remain intact.',
        baseline_timing=dict(action_ms=10,sensor_ms=4,pd_velocity_ms=0),
        reproduction_rule='Both20-25s and25-30s max joint peak-to-peak>=5deg, >=3 prominent peaks during20-30s, late body contact mean<=1N, both wheels>10%weight for>=95%late samples, mean height>=0.18m, whole tilt<60deg and late tilt<30deg.',
        fast_motion_rule='Additionally report3-10Hz angle RMS in20-25s/25-30s; >=0.5deg on same axis in both windows with supported upright stance flags a fast-motion candidate, not exact real waveform matching.',
        motor_feedback_semantics='Where shared_motor_feedback=true, delayed motor position/velocity and declared velocity noise feed both PD and policy motor channels. Policy sensor queue remains additional. Baseline uses original finite-difference policy velocity; shared_feedback_baseline isolates this source difference.',
        training_updates=0,maximum_wall_s=600)
    write(out/'plan.json',plan);shutil.copy2(__file__,out/'run-source.py');(out/'fixtures').mkdir();(out/'traces').mkdir()
    env=FudanEnv(count,seed=30001,randomized=False,reset_contract=metadata['reset_contract'],disturbances=disturbances,cfg=cfg)
    sim=env.env.sim;action=env.action;device=env.device;began=time.time()
    try:
        # 每组同一实测初态和历史，只改变声明的延迟/出力假设。
        q=torch.as_tensor(initial['qpos'],device=device,dtype=torch.float32).expand(count,-1).clone();q[:,:3]+=env.env.scene.env_origins
        v=torch.as_tensor(initial['qvel'],device=device,dtype=torch.float32).expand(count,-1).clone()
        sim.data.qpos.copy_(q);sim.data.qvel.copy_(v);sim.data.ctrl.zero_();sim.forward();env.env.scene.update(dt=0.);sim.sense()
        first_obs=torch.as_tensor(initial['observation'],device=device,dtype=torch.float32);previous=first_obs[19:25]
        action._raw_actions[:]=previous;action._control_actions[:]=previous
        action.last_sample_q.copy_(sim.data.qpos[:,action._motor_q_adr]);action.sample_velocity.copy_(sim.data.qvel[:,action._motor_v_adr])
        env.previous[:,:,0]=previous;env.previous_vel.copy_(action.sample_velocity);env.previous_position.copy_(env.robot.data.root_link_pos_w);env._update_state()
        env.commands[:]=torch.as_tensor(commands,device=device);state=action.disturbances
        state.reset(slice(None),action.sensor_frame());state.actions[:]=previous;state.sensors[:]=first_obs[list(SENSOR_SLOTS)]
        state.action_steps[:]=torch.tensor([round(c.get('action_ms',10.)/2) for c in candidates],device=device)
        state.sensor_steps[:]=torch.tensor([round(c.get('sensor_ms',4.)/2) for c in candidates],device=device)
        obs=first_obs.expand(count,-1).clone();env.history=torch.as_tensor(initial['history'],device=device,dtype=torch.float32).expand(count,-1).clone()
        env._history_frame=obs.clone();env._history_needs_fill.zero_();obs=update_evaluation_commands(env,commands,obs)
        legs=[0,1,3,4];wheels=[2,5];qd_ids=action._motor_v_adr;q_ids=action._motor_q_adr
        motor_ids=[sim.mj_model.actuator('robot/'+n+'_motor').id for n in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        assert not action.wheel_motor.enabled and not action._wheel_speed_shaping
        assert action.kp_scale.eq(1).all() and action.kd_scale.eq(1).all() and action.motor_strength.eq(1).all()
        assert np.all(sim.mj_model.actuator_gainprm[motor_ids,0]==1)
        gains=torch.ones(count,6,device=device)
        gains[:,legs]=torch.tensor([c.get('leg_gain',1.) for c in candidates],device=device)[:,None]
        gains[:,wheels]=torch.tensor([c.get('wheel_gain',1.) for c in candidates],device=device)[:,None]
        pd_steps=torch.tensor([round(c.get('pd_velocity_ms',0)/2) for c in candidates],device=device)
        position_steps=torch.tensor([round(c.get('pd_position_ms',0)/2) for c in candidates],device=device)
        history_length=max(int(pd_steps.max()),int(position_steps.max()))+1;head=0
        velocity_history=sim.data.qvel[:,qd_ids[legs]].clone()[None].repeat(history_length,1,1)
        position_history=sim.data.qpos[:,q_ids[legs]].clone()[None].repeat(history_length,1,1)
        noise_std=torch.tensor([c.get('velocity_noise_std',0.) for c in candidates],device=device)
        noise_rng=torch.Generator(device=device).manual_seed(73011);current_noise=torch.zeros(count,4,device=device)
        shared=torch.tensor([c.get('shared_motor_feedback',False) for c in candidates],device=device)
        worlds=torch.arange(count,device=device)
        lag=torch.tensor([c.get('wheel_lag_ms',0.)*.001 for c in candidates],device=device)
        alpha=-torch.expm1(-.002/lag.clamp(min=1e-12));lag_state=torch.zeros(count,2,device=device)
        leg_lag=torch.tensor([c.get('leg_lag_ms',0.)*.001 for c in candidates],device=device)
        leg_alpha=-torch.expm1(-.002/leg_lag.clamp(min=1e-12));leg_lag_state=torch.zeros(count,4,device=device)
        request_rows=[];output_rows=[];target_rows=[];velocity_rows=[];position_rows=[];noise_rows=[];qrows=[sim.data.qpos[:,q_ids].cpu().numpy().copy()]
        old_write=env.env.scene.write_data_to_sim;old_step=sim.step;identity_error=torch.zeros((),device=device)
        def telemetry():
            return position_history[(head-position_steps)%history_length,worlds],velocity_history[(head-pd_steps)%history_length,worlds]+current_noise
        old_sensor=action.sensor_frame
        def sensor_frame():
            frame=old_sensor()
            if shared.any():
                measured_q,measured_v=telemetry()
                frame[shared,6:10]=(measured_q-action.nominal_positions[:,legs])[shared]
                for j,motor in enumerate(legs):frame[shared,10+motor]=measured_v[shared,j]*cfg.normalization.joint_velocity
            return frame
        action.sensor_frame=sensor_frame
        def write_controls():
            nonlocal identity_error
            old_write();raw=action._control_actions;now_v=sim.data.qvel[:,qd_ids]
            feedback_q,feedback_v=telemetry()
            target=torch.empty_like(raw);target[:,legs]=(action.nominal_positions[:,legs]+.5*raw[:,legs]).clamp(action.reference_positions[:,legs]-3.14,action.reference_positions[:,legs]+3.14)
            target[:,wheels]=(10*raw[:,wheels]).clamp(-150,150)
            request=action._torques.clone();new_leg=(60*(target[:,legs]-feedback_q)-2*feedback_v).clamp(-40,40)
            unchanged=(pd_steps==0)&(position_steps==0)&(noise_std==0)
            identity_error=torch.maximum(identity_error,abs(new_leg[unchanged]-request[unchanged][:,legs]).max())
            request[:,legs]=new_leg
            if not request_rows:lag_state.copy_(request[:,wheels]);leg_lag_state.copy_(request[:,legs])
            lag_state.copy_(torch.where(lag[:,None]==0,request[:,wheels],lag_state+alpha[:,None]*(request[:,wheels]-lag_state)))
            leg_lag_state.copy_(torch.where(leg_lag[:,None]==0,request[:,legs],leg_lag_state+leg_alpha[:,None]*(request[:,legs]-leg_lag_state)))
            output=request*gains;output[:,wheels]=lag_state*gains[:,wheels]
            output[:,legs]=leg_lag_state*gains[:,legs]
            sim.data.ctrl[:,motor_ids]=output
            request_rows.append(request.cpu().numpy().copy());output_rows.append(sim.data.ctrl[:,motor_ids].cpu().numpy().copy())
            target_rows.append(target.cpu().numpy().copy());velocity_rows.append(torch.stack((now_v[:,legs],feedback_v),-1).cpu().numpy().copy())
            position_rows.append(torch.stack((sim.data.qpos[:,q_ids[legs]],feedback_q),-1).cpu().numpy().copy());noise_rows.append(current_noise.cpu().numpy().copy())
        def physics_step():
            nonlocal head
            old_step();qrows.append(sim.data.qpos[:,q_ids].cpu().numpy().copy())
            head=(head+1)%history_length;velocity_history[head]=sim.data.qvel[:,qd_ids[legs]];position_history[head]=sim.data.qpos[:,q_ids[legs]]
            if noise_std.any():current_noise.copy_(torch.randn((1,4),generator=noise_rng,device=device)*noise_std[:,None])
        env.env.scene.write_data_to_sim=write_controls;sim.step=physics_step
        mujoco.mj_saveModel(sim.mj_model,str(out/'fixtures/model.mjb'))
        np.savez_compressed(out/'fixtures/initial.npz',qpos=initial['qpos'],qvel=initial['qvel'],history=initial['history'],
            observation=initial['observation'],action_steps=state.action_steps.cpu().numpy(),sensor_steps=state.sensor_steps.cpu().numpy(),
            output_gains=gains.cpu().numpy(),wheel_lag_s=lag.cpu().numpy(),pd_velocity_steps=pd_steps.cpu().numpy(),
            leg_lag_s=leg_lag.cpu().numpy(),pd_position_steps=position_steps.cpu().numpy(),velocity_noise_std=noise_std.cpu().numpy(),shared_motor_feedback=shared.cpu().numpy(),
            dof_armature=sim.model.dof_armature.cpu().numpy(),dof_damping=sim.model.dof_damping.cpu().numpy(),dof_frictionloss=sim.model.dof_frictionloss.cpu().numpy())
        write(out/'fixtures/mapping.json',dict(motor_q=q_ids.cpu().tolist(),motor_v=qd_ids.cpu().tolist(),motor_ids=motor_ids))
        action.record_substeps=True;rows=[gpu_row(env.env).cpu().numpy()];poses=[q.cpu().numpy().copy()];terms=[np.zeros(count,dtype=bool)];actions=[];observations=[]
        with torch.inference_mode():
            for step in range(steps):
                if time.time()-began>600:raise TimeoutError('Oscillation probe exceeded600s')
                observations.append(obs.cpu().numpy().copy());output,_=policy.act_inference(obs,env.history);actions.append(output.cpu().numpy().copy())
                try:obs,_,_,_,_,_=env.step(output,auto_reset=False,resample=False)
                except FloatingPointError:
                    bad=~torch.isfinite(sim.data.qpos).all(dim=1)|~torch.isfinite(sim.data.qvel).all(dim=1)
                    failed=bad.nonzero().flatten().cpu().tolist()
                    write(out/'failure.json',dict(complete=False,policy_step=step,physical_time_s=(step+1)*.01,
                        nonfinite_candidates=[candidates[i] for i in failed],reason='Non-finite physical state; whole batch stopped, no reset or passing score',training_updates=0))
                    np.savez_compressed(out/'failure-snapshot.npz',last_valid_policy_qpos=poses[-1],
                        current_qpos=sim.data.qpos.cpu().numpy(),current_qvel=sim.data.qvel.cpu().numpy(),
                        recent_motor_q=np.asarray(qrows[-101:]),recent_requested=np.asarray(request_rows[-100:]),recent_output=np.asarray(output_rows[-100:]))
                    raise
                rows.extend(torch.stack(action.substep_rows).cpu().numpy());poses.append(sim.data.qpos.cpu().numpy().copy())
                terms.extend([(env.fail_ticks>cfg.env.fail_ticks).cpu().numpy().copy()]*5)
                if step%200==0:print(json.dumps(dict(step=step,total=steps,profiles=count)),flush=True)
        assert float(identity_error)<1e-5 and len(request_rows)==steps*5 and len(qrows)==steps*5+1
        rows=np.asarray(rows);poses=np.asarray(poses);terms=np.asarray(terms);actions=np.asarray(actions)
        observations=np.asarray(observations)
        signals=dict(motor_q=np.asarray(qrows),target=np.asarray(target_rows),PD_request=np.asarray(request_rows),actual_ctrl=np.asarray(output_rows),velocity_feedback=np.asarray(velocity_rows),position_feedback=np.asarray(position_rows),velocity_noise=np.asarray(noise_rows))
        samples=[]
        for i,c in enumerate(candidates):
            pose=poses[:,i].copy();pose[:,:3]-=env.env.scene.env_origins[i].cpu().numpy()
            trace=arrays(rows[:,i],pose,terminated=terms[:,i]);trace.update(actions=actions[:,i],observations=observations[:,i],**{k:v[:,i] for k,v in signals.items()})
            np.savez_compressed(out/'traces'/(c['name']+'.npz'),**trace)
            samples.append(dict(id=c['name'],seed=30001,randomized=False,candidate=c,case=dict(id='zero_h'+str(commands[i,2]),category='zero',duration=duration,segments=[dict(start=0.,duration=duration,command=commands[i].tolist())])) )
        write(out/'protocol.json',dict(mode='diagnostic_not_formal_acceptance',samples=samples,validation_backend='mujoco_warp_gpu',physical_scope=plan['scope']))
        write(out/'execution.json',dict(complete=True,engine='mjlab',validation_backend='mujoco_warp_gpu',worlds=count,physics_steps_per_world=steps*5,
            policy_batch_forwards=steps,wall_s=time.time()-began,zero_PD_delay_identity_error_Nm=float(identity_error),training_updates=0,checkpoint_unchanged=sha(a.checkpoint)==plan['checkpoint_sha256']))
    finally:env.close()


def report(out):
    plan=json.loads((out/'plan.json').read_text());rows=[]
    for c in plan['profiles']:
        with np.load(out/'traces'/(c['name']+'.npz')) as tr:
            t=tr['time'];q=np.rad2deg(tr['motor_q'][:,[0,1,3,4]]);last=(t>=20)&(t<30)
            pp=np.array([np.ptp(q[(t>=lo)&(t<lo+5)],axis=0) for lo in (10,15,20,25)])
            selected=int(np.argmax(np.minimum(pp[-2],pp[-1])));signal=detrend(q[last,selected])
            frequency=np.fft.rfftfreq(len(signal),.002);power=abs(np.fft.rfft(signal))**2;band=(frequency>=.3)&(frequency<=100)
            peak=float(frequency[band][np.argmax(power[band])]);peaks,_=find_peaks(signal,prominence=1.,distance=25)
            body=tr['base_contact'][last];wheel=tr['wheel_support'][last];weight=tr['weight'][last]
            supported=bool(body.mean()<=1 and np.mean(np.all(wheel>.1*weight[:,None],axis=1))>=.95 and tr['position'][last,2].mean()>=.18)
            upright=bool(abs(tr['roll_pitch_deg']).max()<60 and abs(tr['roll_pitch_deg'][last]).max()<30)
            sustained=bool(np.minimum(pp[-2],pp[-1]).max()>=5 and len(peaks)>=3)
            fast=[]
            for lo in (20,25):
                f,power_fast=welch(q[(t>=lo)&(t<lo+5)],fs=500,nperseg=2000,axis=0,detrend='linear')
                fast.append(np.sqrt(power_fast[(f>=3)&(f<10)].sum(axis=0)*(f[1]-f[0])))
            fast=np.asarray(fast);fast_index=int(np.argmax(np.minimum(fast[0],fast[1])))
            f,power_fast=welch(q[last,fast_index],fs=500,nperseg=2000,detrend='linear');band_fast=(f>=3)&(f<50)
            fast_peak=float(f[band_fast][np.argmax(power_fast[band_fast])])
            all_band=(f>=.5)&(f<50);fast_joint_peak=float(f[all_band][np.argmax(power_fast[all_band])])
            fast_fraction=float(power_fast[(f>=3)&(f<10)].sum()/max(power_fast[all_band].sum(),1e-30))
            has_fast=bool(np.minimum(fast[0],fast[1]).max()>=.5 and supported and upright)
            rows.append(dict(profile=c,leg_pp_by5s_deg=pp.tolist(),selected_joint=['lf','lr','rf','rr'][selected],
                last10s_dominant_Hz=peak,last10s_prominent_peaks=len(peaks),sustained_large_swing=sustained,
                fast3to10Hz_RMS_by5s_deg=fast.tolist(),fast_selected_joint=['lf','lr','rf','rr'][fast_index],fast_band_peak_Hz=fast_peak,
                fast_reproduced_supported=has_fast,fast_joint_dominant_Hz=fast_joint_peak,fast_band_power_fraction=fast_fraction,
                fast_dominant_supported=bool(has_fast and 3<=fast_joint_peak<50),
                supported_standing=supported,upright=upright,reproduced_supported_swing=bool(sustained and supported and upright),
                last10s_body_contact_mean_N=float(body.mean()),last10s_both_wheels_loaded_fraction=float(np.mean(np.all(wheel>.1*weight[:,None],axis=1))),
                height_command_m=c.get('height',plan['command'][2]),last10s_height_mean_m=float(tr['position'][last,2].mean()),maximum_tilt_deg=float(abs(tr['roll_pitch_deg']).max()),
                last10s_speed_peak_m_s=float(np.linalg.norm(tr['linear_world'][last,:2],axis=1).max()),
                last10s_target_pp_deg=np.ptp(np.rad2deg(tr['target'][10000:15000][:,[0,1,3,4]]),axis=0).tolist()))
    ranked=sorted([r for r in rows if r['reproduced_supported_swing']],key=lambda r:max(np.minimum(r['leg_pp_by5s_deg'][-2],r['leg_pp_by5s_deg'][-1])),reverse=True)
    write(out/'summary.json',dict(rows=rows,reproduced=[r['profile']['name'] for r in ranked],fast_candidates=[r['profile']['name'] for r in rows if r['fast_dominant_supported']],fast_component_candidates=[r['profile']['name'] for r in rows if r['fast_reproduced_supported']],scope='Synthetic sensitivity, not measured real delay/gain identification or hardware validation',adopted=False))
    print(json.dumps(dict(reproduced=[r['profile']['name'] for r in ranked],rows=[dict(name=r['profile']['name'],last5s_pp=max(r['leg_pp_by5s_deg'][-1]),frequency=r['last10s_dominant_Hz'],standing=r['supported_standing'],upright=r['upright']) for r in rows]),indent=2))


def audit(out):
    plan=json.loads((out/'plan.json').read_text());fixture=np.load(out/'fixtures/initial.npz');checks=[]
    model=mujoco.MjModel.from_binary_path(str(out/'fixtures/model.mjb'))
    motor_names=('lf0','lf00','lfwheel','rf0','rf00','rfwheel');qi=[int(model.joint('robot/'+x).qposadr[0]) for x in motor_names]
    reference=model.qpos0[qi];legs=[0,1,3,4];wheels=[2,5]
    nominal=np.asarray(FudanWalkCfg.control.nominal_leg_positions);previous=fixture['observation'][19:25]
    for i,c in enumerate(plan['profiles']):
        with np.load(out/'traces'/(c['name']+'.npz')) as tr:
            assert len(tr['time'])==15001 and len(tr['target'])==15000 and np.isfinite(tr['motor_q']).all()
            raw=np.repeat(np.clip(tr['actions'],-100,100),5,axis=0);d=int(fixture['action_steps'][i])
            delayed=np.concatenate((np.repeat(previous[None],d,axis=0),raw),axis=0)[:len(raw)]
            target=np.empty_like(raw);target[:,legs]=np.clip(nominal+.5*delayed[:,legs],reference[legs]-3.14,reference[legs]+3.14)
            target[:,wheels]=np.clip(10*delayed[:,wheels],-150,150)
            target_error=float(abs(target-tr['target']).max())
            velocity=tr['velocity_feedback'];dv=int(fixture['pd_velocity_steps'][i]);noise=tr['velocity_noise'] if 'velocity_noise' in tr else 0.
            expected_v=velocity[np.maximum(np.arange(len(raw))-dv,0),:,0]+noise
            delay_error=float(abs(expected_v-velocity[:,:,1]).max())
            position=tr['position_feedback'] if 'position_feedback' in tr else np.stack((tr['motor_q'][:-1][:,legs],tr['motor_q'][:-1][:,legs]),axis=-1)
            dp=int(fixture['pd_position_steps'][i]) if 'pd_position_steps' in fixture else 0
            position_error=float(abs(position[np.maximum(np.arange(len(raw))-dp,0),:,0]-position[:,:,1]).max())
            expected_leg=np.clip(60*(tr['target'][:,legs]-position[:,:,1])-2*velocity[:,:,1],-40,40)
            pd_error=float(abs(expected_leg-tr['PD_request'][:,legs]).max())
            request=tr['PD_request'];g=fixture['output_gains'][i];tau=float(fixture['wheel_lag_s'][i]);expected=request*g
            lag_state=request[0,wheels].astype('float64').copy();alpha=-np.expm1(-.002/max(tau,1e-12))
            for k in range(len(raw)):
                lag_state=request[k,wheels] if tau==0 else lag_state+alpha*(request[k,wheels]-lag_state)
                expected[k,wheels]=lag_state*g[wheels]
            tau_leg=float(fixture['leg_lag_s'][i]) if 'leg_lag_s' in fixture else 0.
            if tau_leg:
                lag_state=request[0,legs].astype('float64').copy();alpha=-np.expm1(-.002/tau_leg)
                for k in range(len(raw)):
                    lag_state+=alpha*(request[k,legs]-lag_state);expected[k,legs]=lag_state*g[legs]
            response_error=float(abs(expected-tr['actual_ctrl']).max())
            shared_error=0.
            if c.get('shared_motor_feedback',False):
                k=np.arange(10,len(tr['actions']))*5-int(fixture['sensor_steps'][i]);observed=tr['observations'][10:]
                expected_q=np.clip(position[k,:,1]-nominal,-100,100)
                expected_v=np.clip(velocity[k,:,1]*FudanWalkCfg.normalization.joint_velocity,-100,100)
                shared_error=max(float(abs(observed[:,9:13]-expected_q).max()),float(abs(observed[:,np.asarray(legs)+13]-expected_v).max()))
            assert target_error<2e-5 and delay_error<1e-6 and position_error<1e-6 and pd_error<2e-5 and response_error<2e-5 and shared_error<2e-5,(c,target_error,delay_error,position_error,pd_error,response_error,shared_error)
            assert abs(tr['actual_ctrl'][:,legs]).max()<=40.0001 and abs(tr['actual_ctrl'][:,wheels]).max()<=3.9001
            checks.append(dict(name=c['name'],target_queue_error=target_error,PD_velocity_queue_error=delay_error,PD_position_queue_error=position_error,PD_formula_error_Nm=pd_error,output_response_error_Nm=response_error,shared_policy_feedback_error=shared_error))
    write(out/'audit.json',dict(passed=True,checks=checks,source_sha256=sha(__file__),training_updates=0,
        scope='Independent saved500Hz request/response and queue arithmetic; not real-robot parameter identification'))


def render(out,samples):
    from contextlib import ExitStack
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    plan=json.loads((out/'plan.json').read_text());video=out/'video';video.mkdir(exist_ok=False)
    assert 1<=len(samples)<=3
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    with ExitStack() as stack:
        sides=[]
        for sid in samples:
            c=next(c for c in plan['profiles'] if c['name']==sid);path=out/'traces'/(sid+'.npz')
            with np.load(path) as f:tr={k:f[k].copy() for k in f.files}
            model=mujoco.MjModel.from_binary_path(str(out/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth=480;model.vis.global_.offheight=460;data=mujoco.MjData(model)
            renderer=stack.enter_context(mujoco.Renderer(model,width=480,height=460));cam=mujoco.MjvCamera();cam.distance=1.45;cam.elevation=-15.
            opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0];sides.append((c,tr,model,data,renderer,cam,opt,path))
        output=video/'comparison.mp4';frames=0
        with imageio.get_writer(output,fps=50,codec='libx264',quality=8) as writer:
            for k in range(0,3000,2):
                canvas=Image.new('RGB',(480*len(samples),528),'#18212b');draw=ImageDraw.Draw(canvas)
                draw.text((10,6),'1x NORMAL | model3000 | ZERO vx/yaw, height per panel | P60 D2 / wheel P0.1',font=font,fill='white')
                draw.text((10,31),'UNMEASURED hypotheses; simulated feedback loop, not confirmed hardware parameters.',font=font,fill='#ffca75')
                for col,(c,tr,model,data,renderer,cam,opt,_) in enumerate(sides):
                    q=tr['qpos'][k];data.qpos[:]=q;mujoco.mj_forward(model,data);tick=k*5
                    w,x,y,z=q[3:7];cam.azimuth=90+np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)));cam.lookat[:]=[q[0],q[1],.22]
                    renderer.update_scene(data,camera=cam,scene_option=opt);panel=Image.fromarray(renderer.render().copy());text=ImageDraw.Draw(panel)
                    text.rectangle((0,0,480,158),fill='#18212b');joint=np.rad2deg(tr['motor_q'][max(0,tick-500):tick+1][:,[0,1,3,4]])
                    pp=np.ptp(joint,axis=0).max();support=tr['wheel_support'][tick]
                    lines=[f'{c["name"]} | target h={c.get("height",plan["command"][2]):.2f}',f'Action/Obs={c.get("action_ms",10)}/{c.get("sensor_ms",4)}ms PD q/v={c.get("pd_position_ms",0)}/{c.get("pd_velocity_ms",0)}ms',
                        f'Wheel G={c.get("wheel_gain",1):g} tau={c.get("wheel_lag_ms",0)}ms Leg G={c.get("leg_gain",1):g} tau={c.get("leg_lag_ms",0)}ms',
                        f'Velocity noise={c.get("velocity_noise_std",0):g}rad/s shared={c.get("shared_motor_feedback",False)}',
                        f't={k*.01:.2f}s roll/pitch={tr["roll_pitch_deg"][tick,0]:+.1f}/{tr["roll_pitch_deg"][tick,1]:+.1f}deg',
                        f'Leg last1s p-p={pp:.1f}deg',f'Wheel L/R={support[0]:.1f}/{support[1]:.1f}N Body={tr["base_contact"][tick]:.1f}N']
                    for line,value in enumerate(lines):text.text((8,4+22*line),value,font=font,fill='white')
                    canvas.paste(panel,(480*col,68))
                frame=np.asarray(canvas);writer.append_data(frame);frames+=1
                if k in (200,2000,2800):imageio.imwrite(video/f'frame-{k:04d}.png',frame)
        write(video/'index.json',dict(path=str(output.resolve()),samples=samples,frames=frames,fps=50,duration_s=30,sha256=sha(output),
            trace_sha256={s[-1].name:sha(s[-1]) for s in sides},physics_steps=0,policy_forwards=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((out/'summary.json').read_text());fig,axes=plt.subplots(len(samples),2,figsize=(13,3.2*len(samples)),squeeze=False)
    for row,sid in enumerate(samples):
        result=next(r for r in summary['rows'] if r['profile']['name']==sid)
        selected=result['fast_selected_joint'] if result.get('fast_reproduced_supported',False) else result['selected_joint'];j=['lf','lr','rf','rr'].index(selected);motor=[0,1,3,4][j]
        with np.load(out/'traces'/(sid+'.npz')) as tr:
            t=tr['time'];end=22 if result.get('fast_reproduced_supported',False) else 30;mask=(t>=20)&(t<end);axes[row,0].plot(t[mask],np.rad2deg(tr['motor_q'][mask,motor]),label='Actual joint',lw=1)
            axes[row,0].plot(t[:-1][mask[:-1]],np.rad2deg(tr['target'][mask[:-1],motor]),label='PD target',lw=.8,alpha=.7)
            axes[row,1].plot(t[mask],tr['roll_pitch_deg'][mask,1],label='Body pitch',lw=.8)
            axes[row,1].plot(t[mask],tr['roll_pitch_deg'][mask,0],label='Body roll',lw=.8)
        displayed_frequency=result['fast_joint_dominant_Hz'] if result.get('fast_reproduced_supported',False) else result['last10s_dominant_Hz']
        axes[row,0].set_title(sid+' / '+selected);axes[row,1].set_title(f'Body motion / dominant leg frequency {displayed_frequency:.2f}Hz')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Time(s)');ax.set_ylabel('deg')
    fig.tight_layout();fig.savefig(video/'motion-curves.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','report','audit','render'))
    p.add_argument('--experiment',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--profiles',type=Path)
    p.add_argument('--height',type=float,default=.22);p.add_argument('--samples',nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='run':run(a)
    elif a.mode=='report':report(a.output)
    elif a.mode=='audit':audit(a.output)
    else:render(a.output,a.samples)


if __name__=='__main__':main()
