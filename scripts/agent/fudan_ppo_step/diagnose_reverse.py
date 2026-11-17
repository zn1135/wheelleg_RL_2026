"""Local reverse-step diagnostics: bounded GPU replay and static trajectory IK; no learning."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .evaluate import longest, score
from .check import source_fingerprint
from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg, SOURCE, SOURCE_SHA, recipe_record

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'artifacts/fudan_reference_step/20261115-reverse-01/runs/matched200'


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(args):
    from scipy.optimize import least_squares
    from ct_mjlab.envs.chuanliantui.model import load_spec, solve_standing_joint_positions, closed_chain_residual
    source=REFERENCE/'reverse_v0.50_a-10.npz'
    original=mujoco.MjModel.from_binary_path(str(REFERENCE/'model.mjb'));od=mujoco.MjData(original)
    spec=load_spec()
    floor=spec.worldbody.add_geom(name='probe_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(5.,5.,.1))
    platform=spec.worldbody.add_geom(name='probe_platform',type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(2.09/2,1.148/2,.10),pos=(2.09/2,0.,.10))
    model=spec.compile();data=mujoco.MjData(model);baseline=solve_standing_joint_positions(model)
    base=model.body('base_link').id;wids=[model.body(x).id for x in ('lfwheel','rfwheel')]
    terrain={model.geom('probe_floor').id,model.geom('probe_platform').id}
    rows=[];poses=[];previous=np.array([[baseline['lf0'],baseline['lf1']],[baseline['rf0'],baseline['rf1']]])
    with np.load(source) as t:
        # 保留碰沿前站姿及整个抬轮窗口；逐帧IK不推进CPU物理。
        indices=[int(np.argmin(abs(t['time']-.8)))]+list(range(1749,2800,25))
        for index in indices:
            od.qpos[:]=t['qpos'][index];mujoco.mj_kinematics(original,od)
            rotation=od.xmat[original.body('base_Link_del').id].reshape(3,3)
            targets=[];errors=[];front={}
            for j,(side,hip,knee,wheel,oh,ow) in enumerate([
                ('left','lf0','lf1','lfwheel','lf0_Joint','l_wheel_Link'),
                ('right','rf0','rf1','rfwheel','rf0_Joint','r_wheel_Link')]):
                target=(od.body(ow).xpos-od.xanchor[original.joint(oh).id])@rotation
                targets.append(target.tolist())
                addresses=[int(model.joint(x).qposadr[0]) for x in (hip,knee)]
                for name,value in baseline.items():data.qpos[model.joint(name).qposadr[0]]=value
                data.qpos[:7]=[0.,0.,0.,1.,0.,0.,0.]
                lo=[baseline[hip]-math.pi,float(model.joint(knee).range[0])]
                hi=[baseline[hip]+math.pi,float(model.joint(knee).range[1])]
                def residual(values):
                    data.qpos[addresses]=values;mujoco.mj_kinematics(model,data)
                    return (data.body(wheel).xpos-data.xanchor[model.joint(hip).id]-target)[[0,2]]
                fit=least_squares(residual,np.clip(previous[j],lo,hi),bounds=(lo,hi),
                    xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=120)
                previous[j]=fit.x;errors.append(float(np.linalg.norm(residual(fit.x))))
                front[side]=tuple(fit.x-model.qpos0[addresses])
            row=dict(time_s=float(t['time'][index]),target_hip_wheel_base_m=targets,
                front_target_error_m=errors,closed=False,limits_valid=False,
                base_penetration_m=None,leg_penetration_m=None)
            try:
                joints=solve_standing_joint_positions(model,front_joint_targets=front)
                data.qpos[:]=model.qpos0
                for name,value in joints.items():data.qpos[model.joint(name).qposadr[0]]=value
                data.qpos[:7]=od.qpos[:7]
                mujoco.mj_kinematics(model,data)
                # 对齐两轮平均位置，避免把两机root原点差异误作净空收益。
                source_mid=np.mean([od.body(x).xpos for x in ('l_wheel_Link','r_wheel_Link')],axis=0)
                data.qpos[:3]+=source_mid-data.xpos[wids].mean(0)
                mujoco.mj_forward(model,data)
                violations={name:float(value) for name,value in joints.items()
                    if model.joint(name).limited[0] and not model.joint(name).range[0]-1e-7<=value<=model.joint(name).range[1]+1e-7}
                penetration=[0.,0.]
                for contact in data.contact:
                    a,b=map(int,contact.geom)
                    if (a in terrain)==(b in terrain):continue
                    robot_geom=b if a in terrain else a
                    body=int(model.geom_bodyid[robot_geom])
                    if body in wids:continue
                    k=0 if body==base else 1
                    penetration[k]=max(penetration[k],float(-contact.dist))
                row.update(closed=closed_chain_residual(model,data)<1e-7,limits_valid=not violations,
                    limit_violations=violations,base_penetration_m=penetration[0],leg_penetration_m=penetration[1],
                    root_height_m=float(data.qpos[2]),wheel_xyz=data.xpos[wids].tolist(),
                    wheel_fit_world_error_m=np.linalg.norm(data.xpos[wids]-t['wheel_pos'][index],axis=-1).tolist())
                poses.append(data.qpos.copy())
            except RuntimeError as exc:
                row['error']=str(exc);poses.append(np.full(model.nq,np.nan))
            row['template_pose_feasible']=bool(max(errors)<.001 and row['closed'] and row['limits_valid'] and
                row['base_penetration_m']<.001 and row['leg_penetration_m']<.001)
            rows.append(row)
    mujoco.mj_saveModel(model,str(args.output/'model.mjb'),None)
    np.savez_compressed(args.output/'poses.npz',qpos=np.array(poses),time=np.array([r['time_s'] for r in rows]))
    write(args.output/'geometry.json',dict(source=str(source),source_sha256=sha(source),
        source_model_sha256=sha(REFERENCE/'model.mjb'),local_xml_sha256=sha(ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'),
        physics_steps=0,optimizer_updates=0,rows=rows,
        note='Retargeted hip-relative XZ vectors, source body orientation, mean wheel position aligned. '
             'An infeasible template does not exclude another pose/path; no balance, torque or time-feasibility proof.'))
    print(json.dumps(dict(poses=len(rows),template_feasible=sum(r['template_pose_feasible'] for r in rows),
        max_target_error_m=max(max(r['front_target_error_m']) for r in rows)),ensure_ascii=False),flush=True)


def summarize(case, trace):
    n=int(trace['valid'].sum());dt=.001
    if n==0:return dict(id=case['id'],reason='empty',landed=False,parked=False)
    wheel=trace['wheel_sub'][:n].reshape(-1,2,3);support=trace['wheel_support_sub'][:n].reshape(-1,2).astype(bool)
    edge=trace['origin'][0]+1.2;dist=wheel[:,:,0]-edge
    lane=(abs(wheel[:,:,1]-trace['origin'][1])<1.148/2-.04).all(-1)
    top=(abs(wheel[:,:,2]-(case['height']+.06))<.02)
    arrived=(dist>.06)&top&support
    both=arrived.all(-1)&lane
    if case['height']==0:both[:]=False
    force=trace['force'][:n].reshape(-1,2)
    side=trace['side_sub'][:n].reshape(-1).astype(bool)
    tt=np.arange(len(force))*.001+.001
    event=np.flatnonzero((tt>1)&((force>1).any(-1)|side|both))
    first=float(tt[event[0]]) if len(event) else None
    pre=(tt>=first-.2)&(tt<=first-.05) if first is not None else np.zeros(len(tt),bool)
    v=np.repeat(trace['linear_world'][:n],10,axis=0)
    active=(tt>1)&(dist.min(-1)>-.5)&(dist.min(-1)<.8)
    torque=trace['torque_sub'][:n].reshape(-1,6)
    caps=np.array([40.,40.,3.9,40.,40.,3.9])
    speed=np.linalg.norm(trace['linear_world'][:n,:2],axis=-1)
    gravity=trace['gravity'][:n]
    pitch=np.degrees(np.arcsin(np.clip(gravity[:,0],-1,1)))
    roll=np.degrees(np.arctan2(-gravity[:,1],-gravity[:,2]))
    stable=both.reshape(n,10).all(-1)&(speed<=.05)&(abs(pitch)<=5)&(abs(roll)<=5)&(force.reshape(n,10,2)<=1).all(axis=(1,2))
    stable &= (abs(trace['command'][:n,:2]).max(-1)<1e-6)
    first_wheels=[]
    for j in range(2):
        spans=np.convolve(arrived[:,j].astype(int),np.ones(20,dtype=int),mode='valid') if len(arrived)>=20 else []
        hits=np.flatnonzero(np.asarray(spans)==20)
        first_wheels.append(float(tt[hits[0]]) if len(hits) else None)
    return dict(id=case['id'],**{k:v for k,v in case.items() if k!='id'},valid_duration_s=n*.01,
        landed=bool(longest(both)*dt>=.1),parked=bool(longest(stable)*.01>=2),
        top_support_longest_s=longest(both)*dt,parking_longest_s=longest(stable)*.01,
        first_wheel_top_support_s=first_wheels,wheel_x_max_relative_edge_m=dist.max(0).tolist(),
        wheel_height_max_m=wheel[:,:,2].max(0).tolist(),
        actual_approach_speed_m_s=float(np.linalg.norm(v[pre,:2],axis=-1).mean()) if pre.any() else None,
        actual_approach_normal_speed_m_s=float(v[pre,0].mean()) if pre.any() else None,
        contact_event_s=first,nonwheel_peak_N=force.max(0).tolist(),
        nonwheel_contact_longest_s=[longest(force[:,j]>1)*dt for j in range(2)],
        torque_peak_Nm=np.abs(torque).max(0).tolist(),
        torque_saturation_fraction=(np.abs(torque)>=caps-.001).mean(0).tolist(),
        edge_torque_saturation_fraction=(np.abs(torque[active])>=caps-.001).mean(0).tolist() if active.any() else None,
        pitch_range_deg=[float(pitch.min()),float(pitch.max())],
        end_root_relative_xyz=(trace['qpos'][n-1,:3]-trace['origin']).tolist(),
        stop_reason=trace['stop_reason'].item(),
        flat_final_speed_peak_m_s=float(speed[-200:].max()) if case['height']==0 else None)


def stance(args):
    from scipy.optimize import least_squares
    from ct_mjlab.envs.chuanliantui.model import load_spec,solve_standing_joint_positions,closed_chain_residual
    spec=load_spec()
    spec.worldbody.add_geom(name='probe_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(5.,5.,.1))
    spec.worldbody.add_geom(name='probe_platform',type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(2.09/2,1.148/2,.1),pos=(2.09/2,0.,.1))
    model=spec.compile();data=mujoco.MjData(model);baseline=solve_standing_joint_positions(model)
    wids=[model.body(x).id for x in ('lfwheel','rfwheel')];base=model.body('base_link').id
    terrain={model.geom(x).id for x in ('probe_floor','probe_platform')}
    rows=[];poses=[]
    for height,pitch in itertools.product((.30,.32,.345,.36,.38),(0.,-10.,10.)):
        theta=math.radians(pitch)/2;front={};errors=[]
        for side,hip,knee,wheel in [('left','lf0','lf1','lfwheel'),('right','rf0','rf1','rfwheel')]:
            for name,value in baseline.items():data.qpos[model.joint(name).qposadr[0]]=value
            data.qpos[:7]=[0.,0.,height,math.cos(theta),0.,math.sin(theta),0.]
            addresses=[int(model.joint(x).qposadr[0]) for x in (hip,knee)]
            lo=[baseline[hip]-math.pi,float(model.joint(knee).range[0])]
            hi=[baseline[hip]+math.pi,float(model.joint(knee).range[1])]
            def residual(values):
                data.qpos[addresses]=values;mujoco.mj_kinematics(model,data)
                return data.body(wheel).xpos[[0,2]]-[0.,.06]
            fit=least_squares(residual,np.clip([baseline[hip],baseline[knee]],lo,hi),bounds=(lo,hi),
                xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=120)
            errors.append(float(np.linalg.norm(residual(fit.x))))
            front[side]=tuple(fit.x-model.qpos0[addresses])
        joints=solve_standing_joint_positions(model,front_joint_targets=front)
        for name,value in joints.items():data.qpos[model.joint(name).qposadr[0]]=value
        violations=[name for name,value in joints.items() if model.joint(name).limited[0]
            and not model.joint(name).range[0]-1e-7<=value<=model.joint(name).range[1]+1e-7]
        approach=[]
        for x in (-.35,-.25,-.15,-.08):
            data.qpos[:7]=[x,0.,height,0.,-math.sin(theta),0.,math.cos(theta)]
            mujoco.mj_forward(model,data);depth=[0.,0.]
            for c in data.contact:
                a,b=map(int,c.geom)
                if (a in terrain)==(b in terrain):continue
                body=int(model.geom_bodyid[b if a in terrain else a])
                if body in wids:continue
                k=0 if body==base else 1;depth[k]=max(depth[k],float(-c.dist))
            approach.append(dict(root_x_m=x,base_penetration_m=depth[0],leg_penetration_m=depth[1]))
        closed=closed_chain_residual(model,data)<1e-7
        row=dict(root_height_m=height,pitch_deg=pitch,target_error_m=errors,limits_valid=not violations,
            closed=closed,approach=approach,knee_rad=[joints[x] for x in ('lf1','rf1')],
            feasible_grounded_approach=bool(max(errors)<.001 and not violations and closed and
                all(max(c['base_penetration_m'],c['leg_penetration_m'])<.001 for c in approach)))
        rows.append(row);poses.append(data.qpos.copy())
    mujoco.mj_saveModel(model,str(args.output/'model.mjb'),None)
    np.savez_compressed(args.output/'poses.npz',qpos=np.array(poses))
    write(args.output/'stance.json',dict(physics_steps=0,optimizer_updates=0,rows=rows,
        note='Grounded symmetric poses at four pre-edge distances; not a full ascent, balance or motor feasibility proof. '
             'Root height is an imposed geometric coordinate, not a policy height command.'))
    print(json.dumps([r for r in rows if r['feasible_grounded_approach']],ensure_ascii=False),flush=True)


def run(args):
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.fudan_ppo_step.runner import validate_source
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    cases=[]
    for height,speed,angle,direction in itertools.product(args.heights,args.speeds,args.angles,args.directions):
        cases.append(dict(id=f'{direction}_h{height:.3f}_v{speed:.2f}_a{angle:+.0f}',
            height=height,speed=speed,angle=angle,direction=direction))
    for speed,direction in itertools.product(args.speeds,args.directions):
        cases.append(dict(id=f'flat_{direction}_v{speed:.2f}',height=0.,speed=speed,angle=0.,direction=direction))
    if args.high_speed:
        cases=[dict(case,id=case['id']+f'_mu{mu:.2f}',friction=mu)
               for case,mu in itertools.product(cases,args.frictions or [args.friction])]
    if len(cases)>128:raise ValueError('Diagnostic capped at128 worlds')
    cfg=FudanStepCfg();cfg.step.curriculum=False
    # 匹配复旦低速诊断，先隔离方向；名义质量/增益、固定摩擦，不进行域随机化扫描。
    if not args.high_speed:cfg.step.side_obstacle_level=len(cfg.step.heights)
    speed_limit=cfg.step.max_approach_speed if args.high_speed else .8
    checkpoint=args.checkpoint.resolve()
    report=dict(kind='local_reverse_diagnostic',backend='mujoco_warp_gpu',cases=cases,
        seconds=args.seconds,settle_s=0. if args.high_speed else 1.,ramp_s=0. if args.high_speed else .5,
        flat_stop_s=7.,friction=args.frictions or args.friction,
        high_speed=args.high_speed,max_approach_speed=speed_limit,
        diagnostic_action_delay_ms=15.,diagnostic_sensor_delay_ms=15.,
        source_checkpoint=str(checkpoint),source_sha256=sha(checkpoint),recipe=recipe_record(cfg),
        command_height_m=args.command_height,policy_label=args.label,
        source_fingerprint=source_fingerprint(),script_sha256=sha(Path(__file__)),
        optimizer_updates=0,physics_steps=0,policy_calls=0,trace_sha256={},
        scope=('Frozen high-speed diagnostic; direct reset commands, configured side obstacles; not low-speed acceptance'
               if args.high_speed else 'Frozen policy, nominal parameters, fixed delays/friction, no side obstacles; diagnostic only'))
    (args.output/'source_diagnose_reverse.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'traces').mkdir();write(args.output/'protocol.json',report)
    env=None;records=[];torch.set_num_threads(1)
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Diagnostic cannot learn')):
            policy,saved=load_policy(checkpoint,device='cuda:0');validate_source(saved,cfg)
            n=len(cases);env=StepEnv(n,seed=35115,randomized=False,cfg=cfg)
            ids=torch.arange(n,device=env.device)
            env.terrain.terrain_types[:]=torch.tensor([0 if c['height']==0 else 1 for c in cases],device=env.device)
            env.terrain.terrain_levels[:]=torch.tensor([cfg.step.heights.index(c['height']) for c in cases],device=env.device)
            env.env._step_fixed_yaw=torch.tensor([math.radians(c['angle'])+(math.pi if c['direction']=='reverse' else 0.) for c in cases],device=env.device)
            env.fixed_commands=torch.tensor([[c['speed']*(-1 if c['direction']=='reverse' else 1) if args.high_speed else 0.,0.,args.command_height] for c in cases],device=env.device)
            mu=torch.tensor([c.get('friction',args.friction) for c in cases],device=env.device)
            env.env.sim.model.geom_friction[:,:,0]=mu[:,None]
            env.env._fudan_parameters['friction'].copy_(mu)
            # 两方向使用相同延迟，避免把逐世界随机延迟的差别归因于朝向。
            env.action.disturbances.action_steps.fill_(15)
            env.action.disturbances.sensor_steps.fill_(15)
            obs,_,history=env.reset();env.contacts.reset(ids)
            env.contacts.record_substeps=True
            origin=env.env.scene.env_origins.cpu().numpy().copy()
            initial=env.env.sim.data.qpos.cpu().numpy().copy()
            weights=(env.env.sim.model.body_mass[:,env.robot.indexing.body_ids].sum(-1).expand(n)*9.81).cpu().numpy()
            model=env.env.sim.mj_model
            mujoco.mj_saveModel(model,str(args.output/'model.mjb'),None)
            motor_ids=[model.actuator('robot/'+name+'_motor').id for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
            caps=torch.tensor([40.,40.,3.9,40.,40.,3.9],device=env.device)
            alive=torch.ones(n,dtype=torch.bool,device=env.device);reasons=['duration']*n
            target=torch.tensor([c['speed']*(-1 if c['direction']=='reverse' else 1) for c in cases],device=env.device)
            flat=env.modes==0;sub=[];original_update=env.contacts.update
            def update():
                value=original_update()
                # 记录实际提交的实体ctrl，包含PD保持及限幅，不拿策略动作当力矩。
                sub.append(torch.cat((env.env.sim.data.xpos[:,env.wheel_ids].flatten(1),
                    env.env.sim.data.ctrl[:,motor_ids],env.contacts.supported.float(),
                    env.contacts.side_contact[:,None].float()),dim=1).detach().clone())
                return value
            env.contacts.update=update
            with torch.inference_mode():
                for step in range(round(args.seconds/env.dt)):
                    if time.time()>=args.deadline:raise TimeoutError('Reverse diagnostic deadline expired')
                    t=step*env.dt;ramp=1. if args.high_speed else min(1.,max(0.,(t-1.)/.5))
                    env.commands[:,0]=target*ramp
                    env.commands[(~alive)|env.crossed|(flat&(t>=7.)),:2]=0.
                    env.fixed_commands.copy_(env.commands)
                    obs,_=env.observations();env.history[:,-25:]=env._history_frame;history=env.history
                    action,_=policy.act_inference(obs,history);report['policy_calls']+=1
                    action[~alive]=0.;command=env.commands.clone();valid=alive.clone();sub.clear()
                    obs,_,_,done,info,history=env.step(action,auto_reset=False,resample=False)
                    assert torch.isfinite(env.env.sim.data.qpos).all() and torch.isfinite(env.env.sim.data.qvel).all()
                    substep=torch.stack(sub,dim=1)
                    assert substep.shape[1]==10 and (substep[:,:,6:12].abs()<=caps+.0001).all()
                    row=dict(qpos=env.env.sim.data.qpos,qvel=env.env.sim.data.qvel,command=command,valid=valid,
                        position=env.position,wheel_pos=env.env.sim.data.xpos[:,env.wheel_ids],
                        support=env.contacts.supported,support_height=env.support_height,
                        normal_support=env.normal_support,terrain_valid=info['terrain_valid'],
                        linear_body=env.linear,angular_body=env.angular,speed_peak=env.contacts.step_speed_peak,
                        wheel_sub=substep[:,:,:6].reshape(n,10,2,3),torque_sub=substep[:,:,6:12],
                        wheel_support_sub=substep[:,:,12:14],side_sub=substep[:,:,14],
                        linear_world=env.robot.data.root_link_lin_vel_w,gravity=env.gravity,
                        force=torch.stack(env.contacts.step_forces,dim=1),
                        contact_position=torch.stack(env.contacts.step_positions,dim=1))
                    records.append({k:v.cpu().numpy().copy() for k,v in row.items()})
                    for i in (alive&done).nonzero().flatten().tolist():
                        reasons[i]='missing_terrain' if not info['terrain_valid'][i] else ('parked' if env.success[i] else ('nonwheel_contact' if env._contact_failure()[i] else
                            ('approach_speed' if env.max_approach_speed[i]>cfg.step.max_approach_speed else
                             ('timeout' if info['time_outs'][i] else 'fall_or_boundary'))))
                    alive &= ~done
                    if not alive.any():break
                    dead=(~alive).nonzero().flatten()
                    if len(dead):env.reset_idx(dead)
            report.update(physics_steps=env.contacts.total_ticks,world_physics_transitions=env.contacts.total_ticks*n,
                action_delay_steps=env.action.disturbances.action_steps.cpu().tolist(),
                sensor_delay_steps=env.action.disturbances.sensor_steps.cpu().tolist(),
                all_finite=True,all_substep_controls_within_caps=True)
            rows=[]
            for i,case in enumerate(cases):
                trace={k:np.stack([r[k][i] for r in records]) for k in records[0]}
                trace.update(origin=origin[i],initial=initial[i],weight=weights[i],stop_reason=np.array(reasons[i]))
                path=args.output/'traces'/(case['id']+'.npz');np.savez_compressed(path,**trace)
                report['trace_sha256'][case['id']]=sha(path)
                row=summarize(case,trace)
                if case['height']>0:
                    row['task_score']=score(dict(id=case['id'],mode='up',height=case['height']),trace,duration=args.seconds,max_approach_speed=speed_limit,
                        allow_temporary_support=getattr(cfg.step,'allow_temporary_support',False),
                        landing_clearance_m=getattr(cfg.step,'landing_clearance_m',.45))
                rows.append(row)
            frozen=torch.load(checkpoint,map_location='cuda:0',weights_only=False)['model_state_dict']
            assert all(torch.equal(v,frozen[k]) for k,v in policy.state_dict().items())
            report['all_networks_unchanged']=True
            write(args.output/'metrics.json',rows)
            print(json.dumps(dict(cases=n,landed=sum(r['landed'] for r in rows),parked=sum(r['parked'] for r in rows),
                physics_steps=report['physics_steps'],optimizer_updates=0)),flush=True)
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        write(args.output/'protocol.json',report)


def render(args):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol=json.loads((args.recording/'protocol.json').read_text())
    case=next(c for c in protocol['cases'] if c['id']==args.case)
    with np.load(args.recording/'traces'/(args.case+'.npz')) as data:t={k:data[k] for k in data.files}
    n=int(t['valid'].sum());model=mujoco.MjModel.from_binary_path(str(args.recording/'model.mjb'))
    d=mujoco.MjData(model);model.vis.global_.offwidth=800;model.vis.global_.offheight=480
    renderer=mujoco.Renderer(model,height=480,width=800)
    camera=mujoco.MjvCamera();camera.azimuth=90.;camera.elevation=-15.;camera.distance=2.7
    if protocol.get('high_speed'):
        # 从接近方向观察，避免平台两侧障碍遮住轮子和机身。
        camera.azimuth=180.;camera.elevation=-20.
    path=args.output/'replay-1x.mp4';fps=25;frames=max(1,math.ceil(n*.01*fps))
    try:
        with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=None) as writer:
            for f in range(frames):
                i=min(n-1,round(f/fps/.01));d.qpos[:]=t['qpos'][i];mujoco.mj_kinematics(model,d)
                camera.lookat[:]=d.qpos[:3]+[.15,0.,.1];renderer.update_scene(d,camera=camera)
                panel=Image.fromarray(renderer.render());pen=ImageDraw.Draw(panel)
                pen.rectangle((0,0,800,62),fill='#18212b')
                pen.text((8,5),f'{protocol.get("policy_label","LOCAL3000")} | GPU | 1x | {args.case} | t={(i+1)*.01:.2f}s',fill='white')
                pen.text((8,24),f'body/leg peak={t["force"][i].max(0).round(1)} N | end={t["stop_reason"].item()}',fill='white')
                writer.append_data(np.asarray(panel))
                if f in (0,frames//2,frames-1):panel.save(args.output/f'key-{f:03d}.png')
    finally:renderer.close()
    reader=imageio.get_reader(path);count=reader.count_frames();reader.get_data(count-1);reader.close();assert count==frames
    tt=np.arange(1,n+1)*.01;fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    wheel=t['wheel_sub'][:n,-1];edge=t['origin'][0]+1.2
    axes[0].plot(tt,wheel[:,:,0]-edge);axes[0].axhline(0,c='k',ls='--');axes[0].set_ylabel('wheel x - edge(m)')
    axes[1].plot(tt,wheel[:,:,2]);axes[1].axhline(case['height']+.06,c='k',ls='--');axes[1].set_ylabel('wheel z(m)')
    axes[2].plot(tt,t['force'][:n].max(1));axes[2].legend(['body','legs']);axes[2].set_ylabel('contact(N)')
    axes[3].plot(tt,t['torque_sub'][:n,-1]);axes[3].set_ylabel('motor ctrl(Nm)');axes[3].set_xlabel('time(s)')
    for ax in axes:ax.grid(alpha=.3)
    fig.suptitle(args.case);fig.tight_layout();fig.savefig(args.output/'mechanism.png',dpi=130);plt.close(fig)
    write(args.output/'media.json',dict(case=args.case,frames=frames,fps=fps,duration_s=frames/fps,
        video_sha256=sha(path),trace_sha256=sha(args.recording/'traces'/(args.case+'.npz'))))


def render_geometry(args):
    from PIL import Image,ImageDraw
    model=mujoco.MjModel.from_binary_path(str(args.recording/'model.mjb'));data=mujoco.MjData(model)
    with np.load(args.recording/'poses.npz') as t:data.qpos[:]=t['qpos'][args.index]
    if not np.isfinite(data.qpos).all():raise ValueError('No closed pose at selected index')
    mujoco.mj_kinematics(model,data)
    model.vis.global_.offwidth=800;model.vis.global_.offheight=480
    renderer=mujoco.Renderer(model,height=480,width=800)
    camera=mujoco.MjvCamera();camera.azimuth=90.;camera.elevation=-12.;camera.distance=2.
    camera.lookat[:]=data.qpos[:3]+[.2,0.,.1]
    try:
        renderer.update_scene(data,camera=camera);panel=Image.fromarray(renderer.render())
        pen=ImageDraw.Draw(panel);pen.rectangle((0,0,800,40),fill='#18212b')
        pen.text((8,5),f'STATIC GEOMETRY ONLY | pose {args.index} | no dynamics / no balance proof',fill='white')
        panel.save(args.output/'pose.png')
    finally:renderer.close()


def audit(args):
    protocol=json.loads((args.recording/'protocol.json').read_text())
    assert 'error' not in protocol and protocol['backend']=='mujoco_warp_gpu'
    assert protocol['optimizer_updates']==0 and protocol['all_networks_unchanged']
    assert sha(args.recording/'source_diagnose_reverse.py')==protocol['script_sha256']
    assert sha(Path(protocol['source_checkpoint']))==protocol['source_sha256']
    recorded=json.loads((args.recording/'metrics.json').read_text());rows=[];flat=[]
    for case in protocol['cases']:
        path=args.recording/'traces'/(case['id']+'.npz')
        assert sha(path)==protocol['trace_sha256'][case['id']]
        with np.load(path) as data:t={k:data[k] for k in data.files}
        n=int(t['valid'].sum());assert n>0 and t['valid'][:n].all() and not t['valid'][n:].any()
        assert all(np.isfinite(t[k]).all() for k in ('qpos','qvel','wheel_sub','torque_sub','force'))
        assert (abs(t['torque_sub'])<=np.array([40.,40.,3.9,40.,40.,3.9])+.0001).all()
        row=summarize(case,t)
        if 'command_height_m' in protocol and case['height']>0:
            step=protocol['recipe']['step']
            row['task_score']=score(dict(id=case['id'],mode='up',height=case['height']),t,duration=protocol['seconds'],max_approach_speed=protocol.get('max_approach_speed',.8),
                allow_temporary_support=step.get('allow_temporary_support',False),landing_clearance_m=step.get('landing_clearance_m',.45))
        rows.append(row)
        if case['height']==0:
            tail=slice(max(0,n-200),n);speed=np.linalg.norm(t['linear_world'][tail,:2],axis=-1)
            g=t['gravity'][tail];pitch=np.degrees(np.arcsin(np.clip(g[:,0],-1,1)))
            roll=np.degrees(np.arctan2(-g[:,1],-g[:,2]))
            flat.append(dict(id=case['id'],speed_peak_m_s=float(speed.max()),
                observed_tail_s=min(n,200)*.01,last2s_complete=n>=200,
                pitch_abs_peak_deg=float(abs(pitch).max()),roll_abs_peak_deg=float(abs(roll).max()),
                nonwheel_peak_N=float(t['force'][tail].max())))
    assert rows==recorded
    groups=[]
    for h,direction in itertools.product(sorted(set(c['height'] for c in protocol['cases'])),('forward','reverse')):
        selected=[r for r in rows if r['height']==h and r['direction']==direction]
        if selected:groups.append(dict(height_m=h,direction=direction,count=len(selected),
            landed=sum(r['landed'] for r in selected),strict_parked=sum(r['parked'] for r in selected),
            online_parked=sum(r['stop_reason']=='parked' for r in selected)))
    write(args.output/'audit.json',dict(passed=True,cases=len(rows),trace_hashes_and_metrics_recomputed=True,
        actual_motor_caps_verified=True,all_finite=True,source_model_unchanged=True,
        optimizer_updates=0,groups=groups,flat_last2s=flat,
        note='Strict parking additionally requires both wheels loaded throughout every1ms sample. '
             'Online task only checks wheel support at the policy endpoint; counts may differ. '
             'No robustness, trained stair policy, full-field or hardware acceptance claim.'))
    print(json.dumps(dict(passed=True,cases=len(rows),landed=sum(r['landed'] for r in rows),
        strict_parked=sum(r['parked'] for r in rows),optimizer_updates=0)),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='mode',required=True)
    g=sub.add_parser('geometry');g.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('stance');s.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--output',type=Path,required=True)
    r.add_argument('--checkpoint',type=Path,default=ROOT/SOURCE)
    r.add_argument('--command-height',type=float,default=.30)
    r.add_argument('--label',default='LOCAL3000')
    r.add_argument('--seconds',type=float,default=16.);r.add_argument('--deadline',type=float,required=True)
    r.add_argument('--heights',nargs='+',type=float,default=[.02,.05,.08,.12,.16,.20])
    r.add_argument('--speeds',nargs='+',type=float,default=[.3,.5]);r.add_argument('--angles',nargs='+',type=float,default=[0.,-10.,10.])
    r.add_argument('--directions',nargs='+',choices=['forward','reverse'],default=['forward','reverse'])
    r.add_argument('--friction',type=float,default=.8)
    r.add_argument('--high-speed',action='store_true',help='Separate v4 high-speed diagnostic; never the low-speed acceptance matrix')
    r.add_argument('--frictions',nargs='+',type=float,help='High-speed diagnostic batch friction values')
    v=sub.add_parser('render');v.add_argument('--recording',type=Path,required=True)
    v.add_argument('--case',required=True);v.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('render-geometry');q.add_argument('--recording',type=Path,required=True)
    q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('audit');c.add_argument('--recording',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='run' and (not 0<a.seconds<=20 or any(x not in FudanStepCfg().step.heights[1:] for x in a.heights)
        or any(not 0<x<=(2.5 if a.high_speed else .5) for x in a.speeds) or any(abs(x)>20 for x in a.angles) or not .4<=a.friction<=1.2
        or a.frictions and (not a.high_speed or any(not .4<=x<=1.2 for x in a.frictions))
        or not .22<=a.command_height<=.35):
        p.error('Bounded diagnostic requires0<seconds<=20, legal heights, speed<=.5 (high-speed:2.5), |angle|<=20, .4<=friction<=1.2')
    a.output.mkdir(parents=True,exist_ok=False)
    {'geometry':geometry,'stance':stance,'run':run,'render':render,'render-geometry':render_geometry,'audit':audit}[a.mode](a)


if __name__=='__main__':main()
