"""Frozen3000 closed-loop GPU replay from a recorded leg pose; no training."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config,SENSOR_SLOTS
from ct_mjlab.envs.fudan_ppo_walk import height_reward,posture_reward
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.fudan_ppo_walk.measurement import gpu_row,arrays,update_evaluation_commands
from .replay_usb_pd_targets import PoseSolver,sha,write
from .validate_usb_physics import FIELDS


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    with np.load(args.experiment/'mujoco-target-replay-01/inputs.npz') as f:inputs={k:f[k].copy() for k in f.files}
    with (args.experiment/'rl_valid.csv').open() as stream:
        reader=csv.reader(stream);names=next(reader);values=np.asarray(list(reader),dtype=float)
    observed=values[:,[names.index(f'obs_{i}') for i in range(25)]].astype('float32')
    # 40ms起点已有五帧真实历史；以后只使用仿真反馈，不再注入实测关节或姿态。
    start=20;history=observed[np.arange(0,start+1,5)].reshape(-1)
    q0=inputs['reference_qpos'][start//10].copy();solver=PoseSolver();motors=inputs['current'][start].copy();motors[[2,5]]=0
    qa,_=solver.solve(motors);qb,_=solver.solve(motors+1e-4*inputs['velocity'][start])
    v0=np.zeros(solver.model.nv);mujoco.mj_differentiatePos(solver.model,v0,1e-4,qa,qb)
    v0[:3]=0;v0[3:6]=observed[start,:3]*4
    policy,metadata=load_policy(args.checkpoint,'cuda:0')
    cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    height_reward.configure(cfg,height_reward.from_record(metadata.get('height_reward')))
    posture_reward.configure(cfg,posture_reward.from_record(metadata.get('posture_reward')))
    disturbances=from_sim2sim_config()
    assert not disturbances.pushes and disturbances.external_force_range==(0.,0.) and disturbances.external_torque_range==(0.,0.)
    profiles=['baseline','candidate','gas_off'];log_steps=(len(inputs['time_s'])-start)//5
    conditions=[dict(id=p,mode=mode,physical=p,duration=30. if mode=='zero' else log_steps*.01)
                for mode in ('zero','logged') for p in profiles]
    plan=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),conditions=conditions,
        source_start_s=start*.002,source_inputs_sha256=sha(args.experiment/'mujoco-target-replay-01/inputs.npz'),
        controller='100Hz fresh inference from simulated25/125 observations;500Hz motor PD',
        PD=dict(leg_p=60.,leg_d=2.,wheel_p=.1),disturbances=disturbances.record(),
        initial_height_m=float(q0[2]),initial_state='Recorded motor angles/velocities and published IMU; passive joints/height from geometry, rootXY/yaw0 and linear velocity0',
        initialization='Past five real observations seed history once. Previous action holds initial delay queue. No real state is injected after t0.',
        limitation='Near-standing logged pose, not a fallen get-up test. Actual body height/translation/contact load and exact end-to-end latency are unmeasured.',training_updates=0)
    write(out/'plan.json',plan);shutil.copy2(__file__,out/'run-source.py')
    parameters=json.loads((args.experiment/'physics-fit-04/actual-parameters.json').read_text())
    env=FudanEnv(6,seed=30001,randomized=False,reset_contract=metadata['reset_contract'],disturbances=disturbances,cfg=cfg)
    sim=env.env.sim;action=env.action;device=env.device;began=time.time()
    try:
        sim.expand_model_fields(FIELDS)
        for i,c in enumerate(conditions):
            selected=1 if c['physical']=='candidate' else 0
            for field in FIELDS:getattr(sim.model,field)[i]=torch.as_tensor(parameters[field][selected],device=device)
            action._spring_force_bias[i]=torch.as_tensor(parameters['spring_bias_n'][selected],device=device)
            action.spring_damping[i]=torch.as_tensor(parameters['spring_damping'][selected],device=device)
        sim.recompute_constants(RecomputeLevel.set_const)
        q=torch.as_tensor(q0,device=device,dtype=torch.float32).expand(6,-1).clone();q[:,:3]+=env.env.scene.env_origins
        v=torch.as_tensor(v0,device=device,dtype=torch.float32).expand(6,-1).clone()
        sim.data.qpos.copy_(q);sim.data.qvel.copy_(v);sim.data.ctrl.zero_();sim.forward();env.env.scene.update(dt=0.);sim.sense()
        previous=torch.as_tensor(observed[start,19:25],device=device)
        action._raw_actions[:]=previous;action._control_actions[:]=previous
        action.last_sample_q.copy_(sim.data.qpos[:,action._motor_q_adr]);action.sample_velocity.copy_(sim.data.qvel[:,action._motor_v_adr])
        env.previous[:,:,0]=previous;env.previous[:,:,1]=torch.as_tensor(observed[start-5,19:25],device=device)
        env.previous_vel.copy_(action.sample_velocity);env.previous_position.copy_(env.robot.data.root_link_pos_w)
        env._update_state();env.commands[:]=torch.as_tensor(inputs['commands'][start],device=device)
        state=action.disturbances;state.reset(slice(None),action.sensor_frame());state.actions[:]=previous
        state.sensors[:]=torch.as_tensor(observed[start,list(SENSOR_SLOTS)],device=device)
        obs=torch.as_tensor(observed[start],device=device).expand(6,-1).clone()
        env.history=torch.as_tensor(history,device=device).expand(6,-1).clone();env._history_frame=obs.clone();env._history_needs_fill.zero_()
        enabled=torch.tensor([c['physical']!='gas_off' for c in conditions],device=device)
        old_write=env.env.scene.write_data_to_sim;motor_ids=[sim.mj_model.actuator('robot/'+n+'_motor').id for n in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        pd_error=torch.zeros((),device=device);spring_off_max=torch.zeros((),device=device)
        def write_controls():
            nonlocal pd_error,spring_off_max
            old_write();spring=sim.data.ctrl[:,action._spring_ids];spring[~enabled]=0.;sim.data.ctrl[:,action._spring_ids]=spring
            spring_off_max=torch.maximum(spring_off_max,abs(sim.data.ctrl[~enabled][:,action._spring_ids]).max())
            pd_error=torch.maximum(pd_error,abs(sim.data.ctrl[:,motor_ids]-action._torques).max())
        env.env.scene.write_data_to_sim=write_controls
        assert not action.wheel_motor.enabled and not action._wheel_speed_shaping
        assert action.kp_scale.eq(1).all() and action.kd_scale.eq(1).all() and action.motor_strength.eq(1).all()
        for i,c in enumerate(conditions):
            group=out/c['mode'];fixture=group/'fixtures';fixture.mkdir(parents=True,exist_ok=True);(group/'traces').mkdir(exist_ok=True)
            np.savez_compressed(fixture/(c['id']+'_initial.npz'),qpos=q0,qvel=v0,history=history,observation=observed[start],
                **{k:getattr(sim.model,k)[i].cpu().numpy() for k in FIELDS},spring_bias=action._spring_force_bias[i].cpu().numpy(),
                spring_damping=action.spring_damping[i].cpu().numpy(),gas_spring_enabled=bool(enabled[i]),
                action_delay_steps=int(state.action_steps[i]),sensor_delay_steps=int(state.sensor_steps[i]))
            model=mujoco.MjModel.from_binary_path(str(args.experiment/'physics-fit-04/template-model.mjb'))
            for field in FIELDS:getattr(model,field)[:]=getattr(sim.model,field)[i].cpu().numpy()
            mujoco.mj_saveModel(model,str(fixture/(c['id']+'.mjb')))
        action.record_substeps=True;rows=[gpu_row(env.env).cpu().numpy()];poses=[q.cpu().numpy().copy()]
        terms=[np.zeros(6,dtype=bool)];actions=[];commands=[];first_error=None
        with torch.inference_mode():
            for step in range(3000):
                if time.time()-began>420:raise TimeoutError('Closed-loop diagnostic exceeded420s')
                command=np.tile([0.,0.,.22],(6,1));command[3:]=inputs['commands'][min(start+5*step,len(inputs['commands'])-1)]
                obs=update_evaluation_commands(env,command,obs)
                output,_=policy.act_inference(obs,env.history)
                if step==0:
                    expected=torch.as_tensor(observed[start+5,19:25],device=device)
                    first_error=float(abs(output-expected).max());assert first_error<1e-4
                actions.append(output.cpu().numpy().copy());commands.append(command)
                obs,_,_,_,_,_=env.step(output,auto_reset=False,resample=False)
                rows.extend(torch.stack(action.substep_rows).cpu().numpy());poses.append(sim.data.qpos.cpu().numpy().copy())
                terms.extend([(env.fail_ticks>cfg.env.fail_ticks).cpu().numpy().copy()]*5)
                if step%200==0:print(json.dumps(dict(step=step,total=3000)),flush=True)
        assert float(pd_error)<1e-5 and float(spring_off_max)==0
        rows=np.asarray(rows);poses=np.asarray(poses);terms=np.asarray(terms);actions=np.asarray(actions);commands=np.asarray(commands)
        for i,c in enumerate(conditions):
            count=round(c['duration']/.01);end=count*5+1;qtrace=poses[:count+1,i].copy();qtrace[:,:3]-=env.env.scene.env_origins[i].cpu().numpy()
            trace=arrays(rows[:end,i],qtrace,terminated=terms[:end,i]);trace.update(policy_time=np.arange(count)*.01,actions=actions[:count,i],commands=commands[:count,i])
            np.savez_compressed(out/c['mode']/'traces'/(c['id']+'.npz'),**trace)
        for mode in ('zero','logged'):
            samples=[]
            for i,c in enumerate(conditions):
                if c['mode']!=mode:continue
                count=round(c['duration']/.01);cmd=commands[:count,i]
                changes=np.r_[0,np.flatnonzero(np.any(np.diff(cmd,axis=0)!=0,axis=1))+1,count]
                segments=[dict(start=float(lo*.01),duration=float((hi-lo)*.01),command=cmd[lo].tolist()) for lo,hi in zip(changes[:-1],changes[1:])]
                samples.append(dict(id=c['id'],seed=30001,randomized=False,physical_profile=c['physical'],
                    case=dict(id=mode+'_from_recorded_pose',category='zero' if mode=='zero' else 'recorded_commands',duration=c['duration'],
                              segments=segments)))
            write(out/mode/'protocol.json',dict(mode='diagnostic_not_formal_acceptance',samples=samples,
                validation_backend='mujoco_warp_gpu',command_semantics='zero throughout' if mode=='zero' else 'Actual varying logged commands; not a standard fixed-command acceptance matrix',
                disturbances=disturbances.record(),checkpoint_sha256=sha(args.checkpoint)))
            write(out/mode/'execution.json',dict(engine='mjlab',validation_backend='mujoco_warp_gpu',complete_traces=3,planned=3))
        write(out/'execution.json',dict(complete=True,wall_s=time.time()-began,physics_steps_per_world=15000,worlds=6,
            policy_batch_forwards=3000,first_policy_vs_real_action_max_error=first_error,PD_write_max_error_Nm=float(pd_error),
            disabled_spring_force_max_N=float(spring_off_max),training_updates=0,checkpoint_unchanged=sha(args.checkpoint)==plan['checkpoint_sha256']))
    finally:env.close()


def report(out,experiment=None):
    plan=json.loads((out/'plan.json').read_text());rows=[]
    for c in plan['conditions']:
        with np.load(out/c['mode']/'traces'/(c['id']+'.npz')) as tr:
            t=tr['time'];last=(t>=c['duration']-5)&(t<c['duration'])
            wheel=tr['wheel_support'];weight=tr['weight'];body=tr['base_contact'];height=tr['position'][:,2]
            rows.append(dict(**c,initial_height_m=float(height[0]),height_first2s_max_m=float(height[t<=2].max()),
                height_last5s_mean_m=float(height[last].mean()),wheel_support_last5s_mean_N=wheel[last].mean(0).tolist(),
                weight_N=float(weight.mean()),body_contact_last5s_mean_N=float(body[last].mean()),
                body_contact_first2s_max_N=float(body[t<=2].max()),body_supported_fraction_after2s=float(np.mean(body[t>=2]>.2*weight[t>=2])),
                wheel_loaded_fraction_last5s=float(np.mean(np.all(wheel[last]>.1*weight[last,None],axis=1))),
                speed_last5s_peak_m_s=float(np.linalg.norm(tr['linear_world'][last,:2],axis=1).max()),
                pitch_last5s_peak_deg=float(abs(tr['roll_pitch_deg'][last,1]).max()),maximum_abs_tilt_deg=float(abs(tr['roll_pitch_deg']).max()),
                finite=bool(np.isfinite(tr['qpos']).all()),complete=bool(t[-1]>=c['duration']-1e-6)))
    write(out/'summary.json',dict(rows=rows,scope='Support and height diagnosis from near-standing measured joint pose; not fallen get-up or complete sim-real identification'))
    if experiment is not None:
        with np.load(experiment/'mujoco-target-replay-01/inputs.npz') as f:inputs={k:f[k].copy() for k in f.files}
        def gravity(q):
            w,x,y,z=q.T
            return np.stack((2*(w*y-x*z),-2*(w*x+y*z),-(1-2*(x*x+y*y))),axis=-1)
        comparison=[];start=round(plan['source_start_s']/.002)
        for sid in ('baseline','candidate','gas_off'):
            model=mujoco.MjModel.from_binary_path(str(out/'logged/fixtures'/(sid+'.mjb')))
            qi=[int(model.joint('robot/'+n).qposadr[0]) for n in ('lf0','lf00','rf0','rf00')]
            with np.load(out/'logged/traces'/(sid+'.npz')) as tr:
                count=len(tr['actions']);source=start+5*np.arange(count)
                error=np.rad2deg(tr['qpos'][:count,qi]-inputs['current'][source][:,[0,1,3,4]])
                k=np.arange(0,count,2);reference=inputs['reference_qpos'][source[k]//10,3:7]
                cosine=np.sum(gravity(reference)*gravity(tr['qpos'][k,3:7]),axis=1)
                angle=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
                comparison.append(dict(profile=sid,leg_joint_RMS_deg=float(np.sqrt(np.mean(error**2))),
                    gravity_direction_RMS_deg=float(np.sqrt(np.mean(angle**2))),
                    scope='Same commands and frozen policy; time-aligned closed-loop diagnostic, not exact physical-state identification or actual-height comparison'))
        write(out/'logged-comparison.json',comparison)
    print(json.dumps(rows,indent=2))


def render(out):
    from contextlib import ExitStack
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    video=out/'video';video.mkdir(exist_ok=False);index=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for mode in ('zero','logged'):
        with ExitStack() as stack:
            sides=[]
            for sid,label in (('baseline','BASELINE'),('candidate','CANDIDATE: leg J x0.5 / spring +25N'),('gas_off','GAS SPRING OFF')):
                source=out/mode/'traces'/(sid+'.npz')
                with np.load(source) as f:trace={k:f[k].copy() for k in f.files}
                model=mujoco.MjModel.from_binary_path(str(out/mode/'fixtures'/(sid+'.mjb')))
                disable_model_sameframe_shortcuts(model);model.vis.global_.offwidth=480;model.vis.global_.offheight=440
                data=mujoco.MjData(model);renderer=stack.enter_context(mujoco.Renderer(model,width=480,height=440))
                camera=mujoco.MjvCamera();camera.distance=1.5;camera.elevation=-16.;camera.azimuth=90.
                option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
                sides.append((label,trace,model,data,renderer,camera,option,source))
            path=video/(mode+'.mp4');frames=0
            with imageio.get_writer(path,fps=50,codec='libx264',quality=8) as writer:
                for i in range(0,len(sides[0][1]['qpos'])-1,2):
                    canvas=Image.new('RGB',(1440,512),'#18212b');draw=ImageDraw.Draw(canvas)
                    draw.text((12,7),f'1x NORMAL | CLOSED-LOOP model3000 | {mode} commands | t={i*.01:.2f}s | P60/D2 / wheel P0.1',font=font,fill='white')
                    draw.text((12,32),'All panels are physics simulations; camera follows XY/heading, height is NOT forced.',font=font,fill='#ffca75')
                    for col,(label,tr,model,data,renderer,cam,option,_) in enumerate(sides):
                        q=tr['qpos'][i];tick=i*5;data.qpos[:]=q;mujoco.mj_forward(model,data)
                        w,x,y,z=q[3:7];cam.azimuth=90+np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                        cam.lookat[:]=[q[0],q[1],.22];renderer.update_scene(data,camera=cam,scene_option=option)
                        panel=Image.fromarray(renderer.render().copy());text=ImageDraw.Draw(panel);text.rectangle((0,0,480,147),fill='#18212b')
                        cmd=tr['commands'][i];support=tr['wheel_support'][tick];speed=np.linalg.norm(tr['linear_world'][tick,:2])
                        lines=[label,f'cmd vx={cmd[0]:+.2f} yaw={cmd[1]:+.2f} h={cmd[2]:.2f}',
                            f'Actual height={q[2]:.3f}m pitch={tr["roll_pitch_deg"][tick,1]:+.1f}deg',
                            f'Wheel load L/R={support[0]:.1f}/{support[1]:.1f}N',
                            f'Body-ground force={tr["base_contact"][tick]:.1f}N',
                            f'Speed={speed:.3f}m/s XY drift={np.linalg.norm(q[:2]-tr["qpos"][0,:2]):.3f}m']
                        for line,text_line in enumerate(lines):text.text((8,3+23*line),text_line,font=font,fill='white')
                        canvas.paste(panel,(480*col,72))
                    frame=np.asarray(canvas);writer.append_data(frame);frames+=1
                    if i in (0,200,1200):imageio.imwrite(video/f'{mode}-{i:04d}.png',frame)
            index.append(dict(path=str(path.resolve()),frames=frames,fps=50,duration_s=frames/50,sha256=sha(path),
                sources=[dict(path=str(x[-1].resolve()),sha256=sha(x[-1])) for x in sides]))
    write(video/'index.json',dict(videos=index,physics_steps=0,policy_forwards=0,training_updates=0))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('run','report','render'))
    p.add_argument('--experiment',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='run':run(a)
    elif a.mode=='report':report(a.output,a.experiment)
    else:render(a.output)


if __name__=='__main__':main()
