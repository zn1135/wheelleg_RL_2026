"""Small-batch StepEnv replay using existing frozen3000/PD/contact pipeline."""
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from .local_overhang import Hulls,MOTORS,CAPS,ROOT
from .diagnose_reverse import write,sha
from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,SOURCE,SOURCE_SHA,recipe_record
from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
from ct_mjlab.envs.fudan_ppo_step.runner import validate_source
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def run(args):
    out=args.output;selection=json.loads((out/'selected.json').read_text())['rows']
    assert args.batch in range(5)
    direction=('forward','reverse')[args.batch//2] if args.batch<4 else 'forward'
    guide=('additive','held')[args.batch%2] if args.batch<4 else 'height_matched'
    pose=next(r for r in selection if r['direction']==direction)
    grid=json.loads((out/'static/search.json').read_text())['rows']
    neutral=min((r for r in grid if r['pitch_deg']==0 and r['overhang_m'][0]<0),key=lambda r:abs(r['root_height_m']-.35)+abs(r['COM_minus_wheel_mid_m'][0]))
    folder=out/'gpu'/f'{direction}-{guide}';folder.mkdir(parents=True,exist_ok=False)
    cases=[dict(id=f'{direction}_{guide}_v{v:.2f}_a{ang:+.0f}',direction=direction,guide=guide,speed=v,angle=ang) for v in (.2,.3,.5) for ang in (0.,-10.)]
    cfg=FudanStepCfg();cfg.step.curriculum=False;cfg.step.side_obstacle_level=99
    torch.set_num_threads(1);checkpoint=ROOT/SOURCE;assert sha(checkpoint)==SOURCE_SHA
    policy,saved=load_policy(checkpoint,device='cuda:0');validate_source(saved,cfg)
    protocol=dict(backend='mujoco_warp_gpu',cases=cases,seconds=12.,physics_dt_s=.001,pd_dt_s=.005,policy_dt_s=.01,
        guide=guide,selection=pose,height_matched_neutral=neutral if guide=='height_matched' else None,source_checkpoint=str(checkpoint),source_sha256=SOURCE_SHA,optimizer_updates=0,
        actual_action_delay_ms=15.,actual_sensor_delay_ms=15.,friction=.8,root_state_forcing=False,
        guide_ramp_s=[1.,3.5],release_gate='At>=3.5s: actual overhang<=-2mm for100ms, root height within25mm, pitch within5deg, both wheels supported, nearest axle>=.4m before edge',
        held_reference='q_target=q_static+spring_compensated_gravity_torque/Kp; blend frozen NN leg output to this fixed reference; NN wheel output remains active',
        additive_reference='Add smooth(q_static - actual q_at_.9s)/leg_scale to all four live NN leg outputs',
        hard_limit_tolerance_rad=1e-6,recipe=recipe_record(cfg),deadline_unix_s=args.deadline)
    write(folder/'protocol.json',protocol)
    env=None;records=[];error=None
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Frozen measurement cannot learn')):
            env=StepEnv(6,cfg=cfg,seed=361116,randomized=False);ids=torch.arange(6,device=env.device)
            env.terrain.terrain_types.fill_(1);env.terrain.terrain_levels.fill_(cfg.step.heights.index(.20))
            env.env._step_fixed_yaw=torch.tensor([math.radians(c['angle'])+(math.pi if direction=='reverse' else 0) for c in cases],device=env.device)
            env.fixed_commands=torch.tensor([[0.,0.,.30]]*6,device=env.device)
            env.env.sim.model.geom_friction[:,:,0]=.8;env.env._fudan_parameters['friction'].fill_(.8)
            env.action.disturbances.action_steps.fill_(15);env.action.disturbances.sensor_steps.fill_(15)
            obs,_,history=env.reset();env.contacts.reset(ids);env.contacts.record_substeps=True
            model=env.env.sim.mj_model;origin=env.env.scene.env_origins.cpu().numpy().copy();initial=env.env.sim.data.qpos.cpu().numpy().copy()
            static=mujoco.MjModel.from_binary_path(str(out/'static/model.mjb'));sd=mujoco.MjData(static);hulls=Hulls(static)
            assert [model.joint(j).name.split('/')[-1] for j in range(model.njnt)]==[static.joint(j).name for j in range(static.njnt)]
            mids=[model.actuator('robot/'+n+'_motor').id for n in MOTORS]
            legs=env.action._leg_action_ids;desired=torch.tensor(pose['motor_positions'],dtype=torch.float32,device=env.device)
            gravity=torch.tensor(pose['spring_compensated_motor_Nm'],dtype=torch.float32,device=env.device)
            fixed=((desired[legs]+gravity[legs]/20.-env.action.nominal_positions[:,legs])/.5)
            limids=[j for j in range(model.njnt) if model.jnt_limited[j]]
            limq=torch.tensor(model.jnt_qposadr[limids],device=env.device);limits=torch.tensor(model.jnt_range[limids],device=env.device)
            alive=torch.ones(6,dtype=torch.bool,device=env.device);reasons=['duration']*6
            entered=np.full(6,np.nan);streak=np.zeros(6);qanchor=None;sub=[];sub_valid=[]
            previous_q=env.env.sim.data.qpos.clone();previous_v=env.env.sim.data.qvel.clone()
            original=env.contacts.update
            def update():
                nonlocal previous_q,previous_v
                value=original();pos=env.env.sim.data.qpos[:,limq]
                excess=torch.maximum((limits[:,0]-pos).clamp_min(0),(pos-limits[:,1]).clamp_min(0)).amax(-1)
                w=env.env.scene['step_wheels'].data
                sub.append(dict(solver_qpos=previous_q.clone(),solver_qvel=previous_v.clone(),
                    post_qpos=env.env.sim.data.qpos.clone(),wheel_pos=env.env.sim.data.xpos[:,env.wheel_ids].clone(),
                    actual_torque=env.env.sim.data.actuator_force[:,mids].clone(),ctrl=env.env.sim.data.ctrl[:,mids].clone(),
                    base_leg_force=torch.stack(env.contacts.step_forces[-1:]).squeeze(0).clone(),
                    base_leg_contact_pos=torch.stack(env.contacts.step_positions[-1:]).squeeze(0).clone(),
                    wheel_force=w.force.reshape(6,2,4,3).clone(),wheel_normal=w.normal.reshape(6,2,4,3).clone(),
                    wheel_contact_pos=w.pos.reshape(6,2,4,3).clone(),wheel_found=w.found.reshape(6,2,4).clone(),
                    limit_excess=excess.clone()))
                sub_valid.append(alive.clone())
                bad=alive&(excess>1e-6)
                for i in bad.nonzero().flatten().tolist():reasons[i]='hard_limit_excess'
                alive[bad]=False
                previous_q=env.env.sim.data.qpos.clone();previous_v=env.env.sim.data.qvel.clone()
                return value
            env.contacts.update=update
            with torch.inference_mode():
                for step in range(1200):
                    if time.time()>=args.deadline:raise TimeoutError('GPU replay deadline')
                    now=step*.01
                    qcpu=env.env.sim.data.qpos.cpu().numpy();ov=[];pitches=[]
                    for i,q in enumerate(qcpu):
                        sd.qpos[:]=q;sd.qpos[2]-=origin[i,2];mujoco.mj_kinematics(static,sd)
                        ov.append(hulls.measure(sd)[0][0])
                        R=sd.xmat[static.body('base_link').id].reshape(3,3);pitches.append(math.degrees(math.asin(np.clip(-R[2,0],-1,1))))
                    near=(origin[:,0]+1.2-env.env.sim.data.xpos[:,env.wheel_ids,0].cpu().numpy().max(-1))
                    if qanchor is None and now>=.9:qanchor=env.env.sim.data.qpos[:,env.action._motor_q_adr].clone()
                    for i in range(6):
                        if not alive[i] or math.isfinite(entered[i]):continue
                        good=now>=3.5 and ov[i]<=-.002 and abs(qcpu[i,2]-origin[i,2]-pose['root_height_m'])<=.025 and abs(pitches[i]-pose['pitch_deg'])<=5 and bool(env.contacts.supported[i].all())
                        streak[i]=streak[i]+.01 if good else 0.
                        if streak[i]>=.1 and near[i]>=.4:entered[i]=now
                        if near[i]<.4:alive[i]=False;reasons[i]='too_near_before_pose_entry'
                        elif now>=5.:alive[i]=False;reasons[i]='pose_entry_failed'
                    target=[]
                    for i,c in enumerate(cases):
                        factor=np.clip((now-entered[i])/.5,0,1) if math.isfinite(entered[i]) else 0.
                        target.append(c['speed']*factor*(-1 if direction=='reverse' else 1))
                    env.commands[:,0]=torch.tensor(target,device=env.device)
                    if guide=='height_matched':
                        u=np.clip((now-1)/2.5,0,1);env.commands[:,2]=.30+.05*u*u*(3-2*u)
                    env.commands[~alive,:2]=0.;env.fixed_commands.copy_(env.commands)
                    obs,_=env.observations();env.history[:,-25:]=env._history_frame
                    action,_=policy.act_inference(obs,env.history);baseline=action.clone()
                    blend=np.clip((now-1)/2.5,0,1);blend=blend*blend*(3-2*blend)
                    if qanchor is not None:
                        if guide=='height_matched':
                            neutralq=torch.tensor(neutral['motor_positions'],dtype=torch.float32,device=env.device)
                            action[:,legs]+=blend*(desired[legs]-neutralq[legs])/.5
                        elif guide=='additive':action[:,legs]+=blend*(desired[legs]-qanchor[:,legs])/.5
                        else:action[:,legs]=(1-blend)*action[:,legs]+blend*fixed
                    action[~alive]=0.;valid=alive.clone();sub.clear();sub_valid.clear()
                    _,_,_,done,info,_=env.step(action,auto_reset=False,resample=False)
                    assert torch.isfinite(env.env.sim.data.qpos).all() and torch.isfinite(env.env.sim.data.qvel).all()
                    for tick,s in enumerate(sub):
                        row={k:v.cpu().numpy().copy() for k,v in s.items()};row.update(valid=sub_valid[tick].cpu().numpy().copy(),
                            time=np.full(6,now+tick*.001),guide_blend=np.full(6,blend),released=np.isfinite(entered).copy(),
                            commanded_speed=np.array(target),policy_action=baseline.cpu().numpy().copy(),applied_action=action.cpu().numpy().copy())
                        records.append(row)
                    for i in (alive&done).nonzero().flatten().tolist():
                        reasons[i]='nonwheel_contact' if env.contacts.failed[i] else 'fall_boundary_or_missing_terrain'
                    alive&=~done
                    if not alive.any():break
                    dead=(~alive).nonzero().flatten()
                    if len(dead):
                        env.reset_idx(dead);previous_q=env.env.sim.data.qpos.clone();previous_v=env.env.sim.data.qvel.clone()
                    if step%100==0:print(json.dumps(dict(batch=args.batch,t=now,alive=alive.cpu().tolist(),entered=entered.tolist(),overhang_mm=(np.array(ov)*1000).round(1).tolist())),flush=True)
            weights=torch.load(checkpoint,map_location='cuda:0',weights_only=False)['model_state_dict']
            assert all(torch.equal(v,weights[k]) for k,v in policy.state_dict().items())
            protocol.update(networks_unchanged=True,entered_time_s=[None if not math.isfinite(v) else v for v in entered],
                stop_reasons=reasons,physics_ticks=env.contacts.total_ticks,capacity_overflow=bool(env.contacts.capacity_overflow.any()),
                actual_action_delay_steps=env.action.disturbances.action_steps.cpu().tolist(),actual_sensor_delay_steps=env.action.disturbances.sensor_steps.cpu().tolist())
            assert not protocol['capacity_overflow']
    except BaseException as exc:
        protocol['error']=repr(exc);error=exc
    finally:
        if records:
            (folder/'traces').mkdir(exist_ok=True)
            for i,c in enumerate(cases):
                trace={k:np.array([r[k][i] for r in records]) for k in records[0]};trace.update(origin=origin[i],initial=initial[i],stop_reason=np.array(reasons[i]))
                np.savez_compressed(folder/'traces'/(c['id']+'.npz'),**trace)
        if env is not None:env.close()
        write(folder/'protocol.json',protocol)
    if error:raise error
    print('DONE '+str(folder),flush=True)


def view_model():
    from ct_mjlab.envs.chuanliantui.model import load_spec
    spec=load_spec()
    spec.worldbody.add_geom(name='probe_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(5,5,.1),rgba=(.15,.2,.25,1))
    spec.worldbody.add_geom(name='probe_platform',type=mujoco.mjtGeom.mjGEOM_BOX,size=(1.045,.574,.1),pos=(1.045,0,.1),rgba=(.7,.45,.15,1))
    spec.worldbody.add_geom(name='query_edge',type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=(.000001,.574,0),
        pos=(0,0,.2),quat=(math.sqrt(.5),math.sqrt(.5),0,0),contype=0,conaffinity=0,rgba=(0,0,0,0))
    return spec.compile()


def analyze(args):
    from .local_overhang import closed_chain_residual
    from scripts.agent.fudan_ppo_step.evaluate import longest
    out=args.output;dest=out/'dynamic-final';dest.mkdir(exist_ok=False)
    m=view_model();d=mujoco.MjData(m);hulls=Hulls(m)
    edge=m.geom('query_edge').id
    gids=[g for g in range(m.ngeom) if m.geom_bodyid[g] and m.geom_bodyid[g] not in hulls.wids and (m.geom_contype[g] or m.geom_conaffinity[g])]
    names=[m.body(m.geom_bodyid[g]).name for g in gids]
    mujoco.mj_saveModel(m,str(dest/'view-model.mjb'));rows=[]
    for folder in sorted((out/'gpu').iterdir()):
        protocol=json.loads((folder/'protocol.json').read_text());assert 'error' not in protocol
        assert protocol['networks_unchanged'] and not protocol['capacity_overflow']
        for ci,case in enumerate(protocol['cases']):
            path=folder/'traces'/(case['id']+'.npz')
            with np.load(path) as a:t={k:a[k] for k in a.files}
            n=int(t['valid'].sum());assert t['valid'][:n].all() and not t['valid'][n:].any()
            limit=np.where(t['limit_excess'][:n]>1e-6)[0];nlegal=int(limit[0]) if len(limit) else n
            assert nlegal>0
            assert all(np.isfinite(v[:n]).all() for v in t.values() if v.ndim and v.dtype.kind in 'fc')
            assert (abs(t['actual_torque'][:n])<=CAPS+1e-4).all()
            origin=t['origin'];e=origin[0]+1.2
            fp=t['wheel_contact_pos'][:nlegal];wf=t['wheel_force'][:nlegal];wn=t['wheel_normal'][:nlegal]
            loaded=(t['wheel_found'][:nlegal]>0)&(np.linalg.norm(wf,axis=-1)>1)
            platform=(fp[...,0]>=e-.01)&(fp[...,0]<=e+2.10)&(fp[...,2]>.01)&(abs(fp[...,1]-origin[1])<.584)
            wheel=loaded&platform
            pos=t['base_leg_contact_pos'][:nlegal];force=t['base_leg_force'][:nlegal]
            npmask=(force>1)&(pos[...,0]>=e-.01)&(pos[...,0]<=e+2.10)&(pos[...,2]>.01)&(abs(pos[...,1]-origin[1])<.584)
            wi=np.where(wheel.any((1,2)))[0];ni=np.where(npmask.any(-1))[0]
            firstw=int(wi[0]) if len(wi) else None;firstn=int(ni[0]) if len(ni) else None
            event=min([a for a in (firstw,firstn) if a is not None],default=None)
            wheel_first=firstw is not None and (firstn is None or firstw<firstn)
            # Every2ms plus the exact neighboring1ms event samples.
            indices=np.unique(np.r_[np.arange(0,nlegal,2),[i for a in (firstw,firstn) if a is not None for i in (max(0,a-1),a)]]).astype(np.int64)
            geometry=[];distance=[];closure=[];kinerror=0.
            for i in indices:
                d.qpos[:]=t['solver_qpos'][i];d.qpos[:3]-=origin+[1.2,0,0];mujoco.mj_kinematics(m,d)
                ov=hulls.measure(d)[0][0];R=d.xmat[m.body('base_link').id].reshape(3,3)
                pitch=math.degrees(math.asin(np.clip(-R[2,0],-1,1)))
                geometry.append([t['time'][i],ov,d.qpos[2],pitch])
                distance.append([mujoco.mj_geomDistance(m,d,g,edge,10,None)+.000001 for g in gids])
                closure.append(closed_chain_residual(m,d))
                kinerror=max(kinerror,float(abs(d.xpos[hulls.wids]+origin+[1.2,0,0]-t['wheel_pos'][i]).max()))
            assert kinerror<2e-5
            geo=np.array(geometry);dist=np.array(distance);cl=np.array(closure)
            entered=protocol['entered_time_s'][ci];last=float(t['time'][nlegal-1])
            endpre=float(t['time'][event]) if event is not None else last+.001
            pre=(geo[:,0]<endpre)&(geo[:,0]>=0)
            moving=(geo[:,0]>=(entered if entered is not None else np.inf))&(geo[:,0]<endpre)
            speeds=np.linalg.norm(t['solver_qvel'][:nlegal,:2],axis=-1)
            geospeed=np.interp(geo[:,0],t['time'][:nlegal],speeds)
            held=moving&(geo[:,1]<=0)&(geospeed>.1)
            min_distance=float(dist[pre].min()) if pre.any() else None
            approach=(t['time'][:nlegal]>=endpre-.20)&(t['time'][:nlegal]<=endpre-.05) if event is not None else np.zeros(nlegal,bool)
            wall=wheel&(abs(wn[...,2])<.7)
            wallz=t['wheel_pos'][:nlegal,:,2]
            maxwall=[float(wallz[:,w][wall[:,w].any(-1)].max()) if wall[:,w].any() else None for w in (0,1)]
            if firstw is not None:
                if firstn is None or firstw<firstn:firstkind='wheel'
                elif firstn==firstw:firstkind='wheel+nonwheel_same_1ms'
                else:firstkind=['base','leg'][int(np.argmax(npmask[firstn]))]
            elif firstn is not None:firstkind=['base','leg'][int(np.argmax(npmask[firstn]))]
            else:firstkind='no_platform_contact'
            first_event_time=None if event is None else float(t['time'][event])
            row=dict(**case,valid_duration_s=last+.001,entered_s=entered,entered_axle_distance_m=None,
                first_contact=firstkind,first_contact_s=first_event_time,first_wheel_s=None if firstw is None else float(t['time'][firstw]),
                first_nonwheel_s=None if firstn is None else float(t['time'][firstn]),wheel_first=wheel_first,
                precontact_upper_edge_distance_min_m=min_distance,primary_wheel_first_legal_to_contact=bool(wheel_first and min_distance is not None and min_distance>0),
                whole_attempt_hard_limits_legal=not len(limit),first_limit_stop_s=None if not len(limit) else float(t['time'][limit[0]]),
                limit_excess_at_stop_rad=float(t['limit_excess'][:n].max()),stop_reason=t['stop_reason'].item(),
                moving_precontact_overhang_range_m=[float(geo[moving,1].min()),float(geo[moving,1].max())] if moving.any() else None,
                negative_overhang_moving_longest_s=longest(held)*.002,
                actual_approach_speed_m_s=float(speeds[approach].mean()) if approach.any() else None,
                actual_moving_speed_range_m_s=[float(geospeed[moving].min()),float(geospeed[moving].max())] if moving.any() else None,
                wall_loaded_axis_height_max_m=maxwall,wall_loaded_rise_max_m=[None if v is None else v-.06 for v in maxwall],
                dynamic_closed_chain_residual_max_m=float(cl.max()),kinematic_GPU_wheel_error_max_m=kinerror,
                torque_peak_Nm=abs(t['actual_torque'][:nlegal]).max(0).tolist(),
                actual_ctrl_force_error_Nm=float(abs(t['actual_torque'][:nlegal]-t['ctrl'][:nlegal]).max()),
                trace=str(path),trace_sha256=sha(path))
            if entered is not None:
                i=min(nlegal-1,round(entered/.001));row['entered_axle_distance_m']=float((e-t['wheel_pos'][i,:,0]).min())
            rows.append(row)
            np.savez_compressed(dest/(case['id']+'-geometry.npz'),time=geo[:,0],overhang=geo[:,1],root_z=geo[:,2],pitch_deg=geo[:,3],
                edge_distance=dist,edge_body_names=np.array(names),closed_chain_residual=cl,source_indices=indices)
            print(json.dumps({k:row[k] for k in ('id','first_contact','primary_wheel_first_legal_to_contact','negative_overhang_moving_longest_s','stop_reason')}),flush=True)
    write(dest/'metrics.json',dict(rows=rows,optimizer_updates=0,cpu_integration_steps=0,contacts_resolution_s=.001,
        geometry_resolution_s=.002,scope='First-contact diagnostic, not ascent or full motion acceptance; cases stop at hard-limit excess, no out-of-limit prefix counted.'))


def video(args):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    out=args.output;rows=json.loads((out/'dynamic-final/metrics.json').read_text())['rows']
    passing=[r for r in rows if r['primary_wheel_first_legal_to_contact']]
    r=max(passing or rows,key=lambda x:x['negative_overhang_moving_longest_s']);path=Path(r['trace'])
    with np.load(path) as a:t={k:a[k] for k in a.files}
    with np.load(out/'dynamic-final'/(r['id']+'-geometry.npz')) as a:g={k:a[k] for k in a.files}
    m=mujoco.MjModel.from_binary_path(str(out/'dynamic-final/view-model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=1000;m.vis.global_.offheight=600
    for gg in range(m.ngeom):
        name=m.body(m.geom_bodyid[gg]).name
        if name=='lfwheel':m.geom_rgba[gg]=[.1,.65,.85,1.]
        elif name=='rfwheel':m.geom_rgba[gg]=[.9,.2,.4,1.]
    m.vis.headlight.ambient[:]=[.6,.6,.6];m.vis.headlight.diffuse[:]=[.8,.8,.8]
    render=mujoco.Renderer(m,height=600,width=1000);cam=mujoco.MjvCamera();cam.azimuth=90.;cam.elevation=-12.;cam.distance=1.3
    dest=out/'video-final';dest.mkdir(exist_ok=False);fps=50;frames=math.ceil(r['valid_duration_s']*fps);vp=dest/'approach-1x.mp4'
    writer=imageio.get_writer(vp,fps=fps,codec='libx264',quality=8,macro_block_size=None)
    try:
        for j in range(frames):
            if time.time()>=args.deadline:raise TimeoutError('EGL video deadline')
            i=min(round(j/fps/.001),round(r['valid_duration_s']/.001)-1)
            d.qpos[:]=t['solver_qpos'][i];d.qpos[:3]-=t['origin']+[1.2,0,0];mujoco.mj_kinematics(m,d)
            cam.lookat[:]=d.qpos[:3]+[.08,0,-.02];render.update_scene(d,camera=cam)
            for ww in range(2):
                for cc in range(4):
                    if t['wheel_found'][i,ww,cc]<=0 or np.linalg.norm(t['wheel_force'][i,ww,cc])<=1:continue
                    point=t['wheel_contact_pos'][i,ww,cc]-t['origin']-[1.2,0,0]
                    gg=render.scene.geoms[render.scene.ngeom];render.scene.ngeom+=1
                    mujoco.mjv_initGeom(gg,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.003]*3),point,np.eye(3).ravel(),np.array([.1,1.,.2,1.]))
            im=Image.fromarray(render.render());pen=ImageDraw.Draw(im);pen.rectangle((0,0,1000,78),fill='#18212b')
            ov=np.interp(t['time'][i],g['time'],g['overhang'])*1000
            pen.text((8,5),f'Local frozen3000 + leg reference | GPU | 1x | {r["id"]} | t={t["time"][i]:.3f}s',fill='white')
            pen.text((8,24),f'overhang={ov:+.1f}mm | root z={d.qpos[2]:.3f}m | body/leg contact={t["base_leg_force"][i].round(1)}N',fill='white')
            pen.text((8,43),f'first platform contact={r["first_contact"]} | limit stop={r["first_limit_stop_s"]} | commanded {r["speed"]:.1f}m/s',fill='white')
            pen.text((8,61),f'Actual speed={np.linalg.norm(t["solver_qvel"][i,:2]):.2f}m/s | Hard-joint-safe prefix; stops before limit exceed.',fill='white')
            writer.append_data(np.asarray(im))
            if j in (0,round(3.5*fps),frames-1):im.save(dest/f'key-{j:04d}.png')
    finally:writer.close();render.close()
    reader=imageio.get_reader(vp);count=0
    for im in reader:count+=1
    md=reader.get_meta_data();reader.close();assert count==frames and md['fps']==fps
    write(dest/'video.json',dict(case=r['id'],fps=fps,frames=count,duration_s=count/fps,full_decode_verified=True,sha256=sha(vp),deadline_unix_s=args.deadline))
