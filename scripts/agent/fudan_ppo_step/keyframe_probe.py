"""Bounded 20mm scripted leg guidance on frozen3000; GPU physics, no learning."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg, SOURCE, SOURCE_SHA, recipe_record
from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
from ct_mjlab.envs.fudan_ppo_step.runner import validate_source
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.envs.chuanliantui.model import load_model, solve_standing_joint_positions, closed_chain_residual
from .diagnose_reverse import summarize, sha, write
from .evaluate import longest, score

ROOT=Path(__file__).resolve().parents[3]
MOTORS=('lf0','lf00','lfwheel','rf0','rf00','rfwheel')


def plan(args):
    model=load_model();data=mujoco.MjData(model);baseline=solve_standing_joint_positions(model)
    previous={side:np.array([baseline[hip],baseline[knee]]) for side,hip,knee in
              [('left','lf0','lf1'),('right','rf0','rf1')]}
    rows=[];poses=[]
    for height in np.arange(.22,.3851,.005):
        front={};errors=[]
        for side,hip,knee,wheel in [('left','lf0','lf1','lfwheel'),('right','rf0','rf1','rfwheel')]:
            data.qpos[:]=model.qpos0
            for name,value in baseline.items():data.qpos[model.joint(name).qposadr[0]]=value
            data.qpos[:7]=[0.,0.,height,1.,0.,0.,0.]
            addresses=[int(model.joint(x).qposadr[0]) for x in (hip,knee)]
            lower=[float(model.joint(x).range[0]) if model.joint(x).limited[0] else model.qpos0[a]-math.pi
                   for x,a in zip((hip,knee),addresses)]
            upper=[float(model.joint(x).range[1]) if model.joint(x).limited[0] else model.qpos0[a]+math.pi
                   for x,a in zip((hip,knee),addresses)]
            def residual(values):
                data.qpos[addresses]=values;mujoco.mj_kinematics(model,data)
                return data.body(wheel).xpos[[0,2]]-[0.,.06]
            fit=least_squares(residual,np.clip(previous[side],lower,upper),bounds=(lower,upper),
                xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=150)
            previous[side]=fit.x;errors.append(float(np.linalg.norm(residual(fit.x))))
            front[side]=tuple(fit.x-model.qpos0[addresses])
        joints=solve_standing_joint_positions(model,front_joint_targets=front)
        for name,value in joints.items():data.qpos[model.joint(name).qposadr[0]]=value
        data.qpos[:7]=[0.,0.,height,1.,0.,0.,0.];mujoco.mj_forward(model,data)
        violation=max([max(float(model.joint(name).range[0]-value),float(value-model.joint(name).range[1]),0.)
                       for name,value in joints.items() if model.joint(name).limited[0]] or [0.])
        error=closed_chain_residual(model,data)
        rows.append(dict(height_m=float(height),motor_positions=[joints[x] for x in MOTORS],
            wheel_target_error_m=max(errors),closed_residual_m=error,limit_violation_rad=violation,
            valid=max(errors)<.0005 and error<1e-7 and violation<1e-7))
        poses.append(data.qpos.copy())
    valid=[r for r in rows if r['valid']]
    assert valid and min(r['height_m'] for r in valid)<=.26 and max(r['height_m'] for r in valid)>=.37
    write(args.output/'plan.json',dict(rows=rows,physics_steps=0,optimizer_updates=0,
        scope='Closed-chain kinematic target table only; not dynamic or collision-free proof'))
    np.savez_compressed(args.output/'poses.npz',qpos=np.array(poses))
    print(json.dumps(dict(valid=len(valid),height_range=[valid[0]['height_m'],valid[-1]['height_m']])))


def cases():
    rows=[]
    for speed in (.5,1.):
        rows.append(dict(id=f'control_v{speed:.2f}',kind='control',speed=speed,side='both',lift=0.,rise=.1))
    for side,lift,rise,speed in itertools.product(('left','right','both'),(.03,.06),(.08,.16),(.5,1.)):
        rows.append(dict(id=f'lift_{side}_z{lift:.2f}_t{rise:.2f}_v{speed:.2f}',kind='lift',
            side=side,lift=lift,rise=rise,speed=speed))
    for crouch,push,flight,speed in itertools.product((.05,.08),(.08,.14),(.10,.18),(.5,1.)):
        rows.append(dict(id=f'hop_c{crouch:.2f}_p{push:.2f}_f{flight:.2f}_v{speed:.2f}',kind='hop',
            side='both',crouch=crouch,push=push,flight=flight,speed=speed))
    return [dict(r,height=.02,angle=0.,direction='forward',friction=.8) for r in rows]


def staged_cases():
    rows=[dict(id=f'control_v{v:.2f}',kind='control',speed=v,side='both',lift=0.,rise=.1) for v in (.8,1.2)]
    for crouch,up_velocity,speed,retract in itertools.product((.08,.10),(.15,.30),(.8,1.2),(.22,.26)):
        rows.append(dict(id=f'staged_c{crouch:.2f}_vz{up_velocity:.2f}_v{speed:.2f}_h{retract:.2f}',
            kind='staged',side='both',crouch=crouch,up_velocity=up_velocity,speed=speed,retract=retract))
    return [dict(r,height=.02,angle=0.,direction='forward',friction=.8) for r in rows]


def smooth(value):
    value=np.clip(value,0.,1.)
    return value*value*(3.-2.*value)


def height_target(case,elapsed):
    if elapsed<0 or case['kind']=='control':return .35
    if case['kind']=='lift':
        rise=case['rise'];fall=rise+.10
        amount=smooth(elapsed/rise)*(1.-smooth((elapsed-fall)/.20))
        return .35-case['lift']*amount
    times=[0.,.20,.20+case['push'],.26+case['push'],.26+case['push']+case['flight'],.46+case['push']+case['flight']]
    heights=[.35,.35-case['crouch'],.37,.26,.26,.35]
    for i in range(len(times)-1):
        if elapsed<=times[i+1]:
            ratio=smooth((elapsed-times[i])/(times[i+1]-times[i]))
            return heights[i]+ratio*(heights[i+1]-heights[i])
    return .35


def measure(case,trace):
    row=summarize(case,trace);n=int(trace['valid'].sum());dt=.001
    w=trace['wheel_sub'][:n].reshape(-1,2,3);supported=trace['wheel_support_sub'][:n].reshape(-1,2).astype(bool)
    edge=float(trace['origin'][0]+1.2)
    # 必须在接触台阶之前真实离地，不能把沿面滚动当作主动抬轮。
    lifted=(w[:,:,2]>.085)&(w[:,:,0]<edge-.06)&~supported
    active_lift_s=[longest(lifted[:,j])*dt for j in range(2)]
    clearance=[float(w[:,j,2][w[:,j,0]<edge-.06].max()-.06)
               if (w[:,j,0]<edge-.06).any() else 0. for j in range(2)]
    transfer=[]
    for j in range(2):
        starts=np.flatnonzero(np.convolve(lifted[:,j].astype(int),np.ones(20,dtype=int),mode='valid')==20) if len(w)>=20 else []
        clean=False
        for start in starts:
            hits=np.flatnonzero(supported[start+20:,j])
            if len(hits):
                landing=int(start+20+hits[0])
                if w[landing,j,0]>edge-.045 and w[landing,j,2]>.075:clean=True;break
        transfer.append(clean)
    limit=float(trace['joint_limit_violation'][:n].max())
    stroke=float(trace['slide_limit_violation'][:n].max())
    nonwheel=float(trace['force'][:n].max())
    row.update(active_preedge_lift_s=active_lift_s,active_lift=max(active_lift_s)>=.02,
        preedge_wheel_clearance_max_m=clearance,clean_air_transfer_to_step=transfer,
        nonwheel_free=nonwheel<=1.,joint_limit_violation_rad=limit,slide_limit_violation_m=stroke,
        guidance_completed=bool(trace['guide_elapsed'][:n].max()>.8),
        valid_reference_ascent=bool(any(transfer) and row['landed'] and nonwheel<=1. and limit<=.02 and stroke<=.001),
        learned_skill=False)
    row['task_score']=score(dict(id=case['id'],mode='up',height=.02),trace,duration=float(trace['requested_seconds']),max_approach_speed=3.)
    return row


def run(args):
    table=json.loads((args.plan/'plan.json').read_text());legal=[r for r in table['rows'] if r['valid']]
    grid=np.array([r['height_m'] for r in legal]);q=np.array([r['motor_positions'] for r in legal])
    qzero=np.array([np.interp(.35,grid,q[:,j]) for j in range(6)])
    selected=staged_cases() if args.guidance=='staged' else cases()
    if args.kind!='all':selected=[r for r in selected if r['kind'] in (args.kind,'control')]
    if args.case:selected=[r for r in selected if r['id'] in args.case]
    assert len(selected)<=48 and args.seconds<=12 and args.seconds>0
    cfg=FudanStepCfg();cfg.step.curriculum=False;cfg.step.side_obstacle_level=99
    env=None;torch.set_num_threads(1);checkpoint=ROOT/SOURCE
    report=dict(kind='20mm_reference_probe',backend='mujoco_warp_gpu',cases=selected,
        seconds=args.seconds,high_speed=True,max_approach_speed=3.,command_height_m=.35,
        source_checkpoint=str(checkpoint),source_sha256=sha(checkpoint),recipe=recipe_record(cfg),
        policy_label='REFERENCE + FROZEN3000',optimizer_updates=0,trace_sha256={},
        root_state_forcing=False,external_wrench=False,learned_skill=False,guidance_mode=args.guidance,
        initial_spawn_offset_x_m=.75 if args.guidance=='staged' else 0.,
        minimum_release_body_height_m=args.launch_height,
        target_table=str((args.plan/'plan.json').resolve()),target_table_sha256=sha(args.plan/'plan.json'),
        scope='Scripted state-aware leg trajectory plus frozen balance policy; not a learned stair skill')
    (args.output/'source-keyframe_probe.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'traces').mkdir();records=[]
    write(args.output/'protocol.json',report)
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Reference probe cannot train')):
            policy,saved=load_policy(checkpoint,device='cuda:0');validate_source(saved,cfg)
            env=StepEnv(len(selected),cfg=cfg,seed=36116,randomized=False)
            n=env.num_envs;ids=torch.arange(n,device=env.device);legs=env.action._leg_action_ids
            env.terrain.terrain_types[:]=1;env.terrain.terrain_levels[:]=1
            target=torch.tensor([r['speed'] for r in selected],device=env.device)
            initial_v=torch.zeros_like(target) if args.guidance=='staged' else target
            env.fixed_commands=torch.stack((initial_v,torch.zeros_like(target),torch.full_like(target,.35)),-1)
            env.env.sim.model.geom_friction[:,:,0]=.8;env.env._fudan_parameters['friction'].fill_(.8)
            env.action.disturbances.action_steps.fill_(15);env.action.disturbances.sensor_steps.fill_(15)
            obs,_,history=env.reset();env.contacts.reset(ids);env.contacts.record_substeps=True
            if args.guidance=='staged':
                # 只在t=0前设置出生点；执行过程中不改活跃机器人的根状态。
                root=torch.cat((env.env.sim.data.qpos[:,:7].clone(),torch.zeros(n,6,device=env.device)),-1)
                root[:,0]+=.75;env.robot.write_root_state_to_sim(root,env_ids=ids)
                env.env.sim.forward();env.env.scene.update(dt=0.);env.env.sim.sense();env._update_state()
                env.previous_position.copy_(env.position);env.previous_vel.copy_(env.velocity)
                env.action.disturbances.sensors[:,ids]=env.action.sensor_frame()[ids]
                obs,_=env.observations();env.history[:]=env._history_frame.repeat(1,5);history=env.history
            origin=env.env.scene.env_origins.cpu().numpy().copy();initial=env.env.sim.data.qpos.cpu().numpy().copy()
            weights=(env.env.sim.model.body_mass[:,env.robot.indexing.body_ids].sum(-1).expand(n)*9.81).cpu().numpy()
            model=env.env.sim.mj_model;mujoco.mj_saveModel(model,str(args.output/'model.mjb'),None)
            motor_ids=[model.actuator('robot/'+x+'_motor').id for x in MOTORS]
            caps=torch.tensor([40.,40.,3.9,40.,40.,3.9],device=env.device)
            limit_ids=[j for j in range(model.njnt) if model.jnt_limited[j] and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
            addresses=torch.tensor([model.jnt_qposadr[j] for j in limit_ids],device=env.device,dtype=torch.long)
            limits=torch.tensor(model.jnt_range[limit_ids],device=env.device)
            slide_ids=[j for j in range(model.njnt) if model.jnt_limited[j] and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE]
            slide_addresses=torch.tensor([model.jnt_qposadr[j] for j in slide_ids],device=env.device,dtype=torch.long)
            slide_limits=torch.tensor(model.jnt_range[slide_ids],device=env.device)
            alive=torch.ones(n,dtype=torch.bool,device=env.device);reasons=['duration']*n
            started=np.full(n,np.nan);held=torch.zeros(n,4,device=env.device);sub=[]
            phase=np.zeros(n,dtype=int);phase_start=np.zeros(n);push_height=np.zeros(n)
            original_update=env.contacts.update
            def update():
                out=original_update()
                pos=env.env.sim.data.qpos[:,addresses]
                violation=torch.maximum((limits[:,0]-pos).clamp_min(0),(pos-limits[:,1]).clamp_min(0)).amax(-1)
                slide_pos=env.env.sim.data.qpos[:,slide_addresses]
                stroke=torch.maximum((slide_limits[:,0]-slide_pos).clamp_min(0),
                    (slide_pos-slide_limits[:,1]).clamp_min(0)).amax(-1) if len(slide_ids) else torch.zeros(n,device=env.device)
                sub.append(torch.cat((env.env.sim.data.xpos[:,env.wheel_ids].flatten(1),
                    env.env.sim.data.ctrl[:,motor_ids],env.contacts.supported.float(),
                    env.contacts.side_contact[:,None].float(),violation[:,None],stroke[:,None],
                    env.env.sim.data.xfrc_applied.abs().flatten(1).amax(-1)[:,None]),-1).detach().clone())
                return out
            env.contacts.update=update
            with torch.inference_mode():
                for step in range(round(args.seconds/env.dt)):
                    if time.time()>=args.deadline:raise TimeoutError('Reference probe budget expired')
                    env.commands[:,0]=target;env.commands[(~alive)|env.crossed,:2]=0.
                    if args.guidance=='staged':
                        waiting=torch.tensor([step*env.dt<1.2 if c['kind']=='control' else phase[i]<2
                                              for i,c in enumerate(selected)],device=env.device)
                        env.commands[waiting,:2]=0.
                    env.fixed_commands.copy_(env.commands)
                    obs,_=env.observations();env.history[:,-25:]=env._history_frame;history=env.history
                    action,_=policy.act_inference(obs,history);baseline_action=action.clone()
                    dist=(env.env.sim.data.xpos[:,env.wheel_ids,0]-env.env.scene.env_origins[:,0,None]-1.2).cpu().numpy()
                    heights=[]
                    for i,case in enumerate(selected):
                        if args.guidance=='staged':
                            now=step*env.dt
                            if case['kind']=='control':heights.append(.35);continue
                            if phase[i]==0 and now>=1.2:
                                phase[i]=1;phase_start[i]=now;started[i]=now;held[i]=baseline_action[i,legs]
                            elapsed=now-phase_start[i]
                            if phase[i]==1:
                                vz=float(env.robot.data.root_link_lin_vel_w[i,2])
                                if elapsed>=.6 and (abs(vz)<.08 or elapsed>=1.):
                                    phase[i]=2;phase_start[i]=now;push_height[i]=float(env.position[i,2]);elapsed=0.
                            if phase[i]==2:
                                vz=float(env.robot.data.root_link_lin_vel_w[i,2]);z=float(env.position[i,2])
                                release_height=max(push_height[i]+.015,args.launch_height or 0.)
                                push_timeout=.7 if args.launch_height is not None else .5
                                if elapsed>=.08 and ((vz>case['up_velocity'] and z>release_height) or elapsed>=push_timeout):
                                    phase[i]=3;phase_start[i]=now;elapsed=0.
                            if phase[i]==3 and elapsed>=.20:
                                phase[i]=4;phase_start[i]=now;elapsed=0.
                            if phase[i]==0:h=.35
                            elif phase[i]==1:h=.35-case['crouch']*smooth(elapsed/.25)
                            elif phase[i]==2:h=(.35-case['crouch'])+(.375-(.35-case['crouch']))*smooth(elapsed/.06)
                            elif phase[i]==3:h=.375+(case['retract']-.375)*smooth(elapsed/.06)
                            else:h=case['retract']+(.35-case['retract'])*smooth(elapsed/.2)
                            heights.append(h);continue
                        threshold=(.38 if case['speed']==.5 else .50) if case['kind']=='hop' else (.12 if case['speed']==.5 else .18)
                        wheel=0 if case['side']=='left' else 1
                        near=dist[i].max() if case['side']=='both' else dist[i,wheel]
                        if math.isnan(started[i]) and near>-threshold and alive[i]:
                            started[i]=step*env.dt;held[i]=baseline_action[i,legs]
                        elapsed=step*env.dt-started[i] if not math.isnan(started[i]) else -1.
                        heights.append(height_target(case,elapsed))
                    offsets=np.array([[np.interp(h,grid,q[:,j])-qzero[j] for j in range(6)] for h in heights])
                    delta=torch.tensor(offsets[:,[0,1,3,4]]/.5,dtype=action.dtype,device=env.device)
                    for i,case in enumerate(selected):
                        if case['side']=='left':delta[i,2:]=0
                        elif case['side']=='right':delta[i,:2]=0
                    if args.guidance in ('hold','staged'):
                        mask=torch.tensor([math.isfinite(started[i]) and c['kind']!='control' for i,c in enumerate(selected)],device=env.device)
                        action[:,legs]=torch.where(mask[:,None],held,baseline_action[:,legs])+delta
                    else:action[:,legs]+=delta
                    action[~alive]=0.;command=env.commands.clone();valid=alive.clone();sub.clear()
                    obs,_,_,done,info,history=env.step(action,auto_reset=False,resample=False)
                    ticks=torch.stack(sub,1)
                    assert ticks.shape[1]==10 and (ticks[:,:,6:12].abs()<=caps+.0001).all()
                    assert ticks[:,:,-1].max()==0 and torch.isfinite(env.env.sim.data.qpos).all()
                    row=dict(qpos=env.env.sim.data.qpos,qvel=env.env.sim.data.qvel,command=command,valid=valid,
                        position=env.position,wheel_pos=env.env.sim.data.xpos[:,env.wheel_ids],
                        support=env.contacts.supported,support_height=env.support_height,normal_support=env.normal_support,
                        terrain_valid=info['terrain_valid'],linear_body=env.linear,angular_body=env.angular,
                        linear_world=env.robot.data.root_link_lin_vel_w,gravity=env.gravity,speed_peak=env.contacts.step_speed_peak,
                        wheel_sub=ticks[:,:,:6].reshape(n,10,2,3),torque_sub=ticks[:,:,6:12],
                        wheel_support_sub=ticks[:,:,12:14],side_sub=ticks[:,:,14],joint_limit_violation=ticks[:,:,15],slide_limit_violation=ticks[:,:,16],
                        external_wrench_peak=ticks[:,:,17],
                        force=torch.stack(env.contacts.step_forces,1),contact_position=torch.stack(env.contacts.step_positions,1),
                        action=action,baseline_action=baseline_action,reference_delta=delta,
                        reference_phase=torch.tensor(phase,device=env.device),
                        guide_elapsed=torch.tensor([step*env.dt-s if math.isfinite(s) else -1 for s in started],device=env.device))
                    records.append({k:v.cpu().numpy().copy() for k,v in row.items()})
                    for i in (alive&done).nonzero().flatten().tolist():
                        reasons[i]='missing_terrain' if not info['terrain_valid'][i] else ('parked' if env.success[i] else
                            ('nonwheel_contact' if env._contact_failure()[i] else ('timeout' if info['time_outs'][i] else 'fall_or_boundary')))
                    alive &= ~done
                    if not alive.any():break
                    dead=(~alive).nonzero().flatten()
                    if len(dead):env.reset_idx(dead)
            frozen=torch.load(checkpoint,map_location='cuda:0',weights_only=False)['model_state_dict']
            assert all(torch.equal(v,frozen[k]) for k,v in policy.state_dict().items())
            rows=[]
            for i,case in enumerate(selected):
                trace={k:np.stack([r[k][i] for r in records]) for k in records[0]}
                trace.update(origin=origin[i],initial=initial[i],weight=weights[i],stop_reason=np.array(reasons[i]),requested_seconds=np.array(args.seconds))
                path=args.output/'traces'/(case['id']+'.npz');np.savez_compressed(path,**trace)
                report['trace_sha256'][case['id']]=sha(path);rows.append(measure(case,trace))
            report.update(all_networks_unchanged=True,all_finite=True,physics_steps=env.contacts.total_ticks,
                constraint_peak=int(env.contacts.constraint_peak),contact_peak=int(env.contacts.global_contact_peak),
                capacity_overflow=bool(env.contacts.capacity_overflow.any()))
            write(args.output/'metrics.json',rows)
            print(json.dumps(dict(cases=n,active_lift=sum(r['active_lift'] for r in rows),landed=sum(r['landed'] for r in rows),
                valid_reference_ascent=sum(r['valid_reference_ascent'] for r in rows),optimizer_updates=0)),flush=True)
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        write(args.output/'protocol.json',report)


def audit(args):
    protocol=json.loads((args.recording/'protocol.json').read_text());assert 'error' not in protocol
    assert protocol['all_networks_unchanged'] and protocol['optimizer_updates']==0 and not protocol['capacity_overflow']
    assert sha(Path(protocol['source_checkpoint']))==SOURCE_SHA
    expected=json.loads((args.recording/'metrics.json').read_text());actual=[]
    for case in protocol['cases']:
        path=args.recording/'traces'/(case['id']+'.npz');assert sha(path)==protocol['trace_sha256'][case['id']]
        with np.load(path) as saved:trace={k:saved[k] for k in saved.files}
        n=int(trace['valid'].sum());assert trace['valid'][:n].all() and not trace['valid'][n:].any()
        assert (abs(trace['torque_sub'][:n])<=np.array([40.,40.,3.9,40.,40.,3.9])+.0001).all()
        if 'external_wrench_peak' in trace:assert not trace['external_wrench_peak'][:n].any()
        assert trace['terrain_valid'][:n].all()
        assert all(np.isfinite(trace[key][:n]).all() for key in ('qpos','qvel','torque_sub','force'))
        actual.append(measure(case,trace))
    # 旧记录缺少后来新增的净空/连续转移字段；原字段仍须逐值一致。
    assert len(actual)==len(expected)
    assert all({key:row[key] for key in old}==old for row,old in zip(actual,expected))
    write(args.output/'recomputed-metrics.json',actual)
    write(args.output/'audit.json',dict(passed=True,cases=len(actual),trace_hashes_and_metrics_recomputed=True,
        actual_motor_caps_verified=True,optimizer_updates=0,learned_skill=False,
        recorded_source_sha256=sha(args.recording/'source-keyframe_probe.py'),
        valid_reference_ascents=[r['id'] for r in actual if r['valid_reference_ascent']]))


def render(args):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    protocol=json.loads((args.recording/'protocol.json').read_text())
    with np.load(args.recording/'traces'/(args.case+'.npz')) as data:t={k:data[k] for k in data.files}
    n=int(t['valid'].sum());model=mujoco.MjModel.from_binary_path(str(args.recording/'model.mjb'))
    state=mujoco.MjData(model);model.vis.global_.offwidth=800;model.vis.global_.offheight=480
    camera=mujoco.MjvCamera();camera.azimuth=90.;camera.elevation=-12.;camera.distance=1.7
    renderer=mujoco.Renderer(model,height=480,width=800);fps=25;frames=math.ceil(n*.01*fps)
    path=args.output/'replay-1x.mp4'
    try:
        with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=None) as writer:
            for f in range(frames):
                i=min(n-1,round(f/fps/.01));state.qpos[:]=t['qpos'][i];mujoco.mj_kinematics(model,state)
                camera.lookat[:]=state.qpos[:3]+[.12,0.,-.1];renderer.update_scene(state,camera=camera)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                force=t['force'][i].max(0);color='#872126' if force.max()>1 else '#172a35'
                draw.rectangle((0,0,800,68),fill=color)
                draw.text((8,4),f'SCRIPTED GUIDANCE + FROZEN3000 | GPU 1x | NOT A LEARNED SKILL | t={(i+1)*.01:.2f}s',fill='white')
                draw.text((8,24),f'{args.case} | body/leg={force.round(1)} N | end={t["stop_reason"].item()}',fill='white')
                draw.text((8,44),f'wheel center above lower-floor radius={(1000*(t["wheel_pos"][i,:,2]-.06)).round(1)} mm',fill='white')
                writer.append_data(np.asarray(frame))
                if f in (0,frames//2,frames-1):frame.save(args.output/f'key-{f:03d}.png')
    finally:renderer.close()
    reader=imageio.get_reader(path);count=sum(1 for _ in reader);reader.close();assert count==frames
    write(args.output/'media.json',dict(case=args.case,frames=frames,fps=fps,duration_s=frames/fps,
        all_frames_decoded=True,video_sha256=sha(path),trace_sha256=sha(args.recording/'traces'/(args.case+'.npz')),
        optimizer_updates=0,new_physics_steps=0,learned_skill=False))


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('plan');a.add_argument('--output',type=Path,required=True)
    r=s.add_parser('run');r.add_argument('--plan',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--deadline',type=float,required=True);r.add_argument('--seconds',type=float,default=8.)
    r.add_argument('--guidance',choices=('additive','hold','staged'),default='additive')
    r.add_argument('--kind',choices=('all','lift','hop'),default='all')
    r.add_argument('--case',nargs='+',help='Replay explicit cases without expanding the experiment')
    r.add_argument('--launch-height',type=float,help='Absolute body-height condition before retraction; no body-state forcing')
    a=s.add_parser('audit');a.add_argument('--recording',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('render');a.add_argument('--recording',type=Path,required=True);a.add_argument('--case',required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    {'plan':plan,'run':run,'audit':audit,'render':render}[args.mode](args)


if __name__=='__main__':main()
