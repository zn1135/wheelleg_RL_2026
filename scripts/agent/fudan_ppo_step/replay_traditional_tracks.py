"""传统台阶实录的只读映射、有限GPU参考回放与离线评分；不训练。"""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time
from unittest.mock import patch
from zipfile import ZipFile

import mujoco
import numpy as np

from .diagnose_reverse import sha, write

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'deployment/H7_RL/imcalib/config/machine_config.c'
XML = ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
MOTORS = ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')
CSV_JOINTS = ('lf','lr','rf','rr')


def firmware_pose(q):
    # 沿用既有H7五连杆前髋+pi定义，仅用于核对日志，不拟合零点/符号。
    q=q.reshape(-1,2,2)+[np.pi,0.]
    endpoints=.21*np.stack((np.cos(q),np.sin(q)),axis=-1)
    delta=endpoints[:,:,1]-endpoints[:,:,0]
    distance=np.linalg.norm(delta,axis=-1)
    normal=np.stack((-delta[...,1],delta[...,0]),axis=-1)/distance[...,None]
    point=endpoints.mean(-2)+np.sqrt(.25**2-distance**2/4)[...,None]*normal
    angle=np.pi/2-np.arctan2(point[...,1],point[...,0])
    return np.linalg.norm(point,axis=-1),np.arctan2(np.sin(angle),np.cos(angle))


def prepare(args):
    sys.path.insert(0,str(ROOT/'sim2sim'))
    import real2sim_huananhu as mirror
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'tracks').mkdir();(args.output/'inputs').mkdir()
    shutil.copy2(CONFIG,args.output/'inputs/machine_config.c')
    shutil.copy2(Path(__file__),args.output/'source-replay_traditional_tracks.py')
    # 当前固件已移除大小机数组外壳；只读唯一字段，兼容旧镜像解析器。
    text=CONFIG.read_text();arrays=[]
    for field,count in (('motor_sign',6),('motor_zero',4)):
        matches=re.findall(r'\.'+field+r'\s*=\s*\{([^}]+)\}',text)
        if len(matches)!=1:raise ValueError(f'Expected one {field} in supplied firmware')
        value=np.array([float(x.strip().rstrip('fF')) for x in matches[0].split(',') if x.strip()])
        if value.shape!=(count,):raise ValueError(f'Unexpected {field} shape')
        arrays.append(value)
    with patch.object(mirror,'load_mapping',return_value=tuple(arrays)):
        pose=mirror.MirrorPose(model_path=MODEL_XML,machine_config=CONFIG)
    model=pose.model
    mjids=[model.joint(n).qposadr[0] for n in mirror.MOTORS]
    protected={str(p):sha(p) for p in (CONFIG,XML,ROOT/'sim2sim/real2sim_huananhu.py')}
    report=dict(source_zip=str(args.source.resolve()),source_zip_sha256=sha(args.source),
        user_reported_stair_height_m=.2,field_measured_in_csv=False,
        mapping_sign=pose.sign.tolist(),mapping_zero_rad=pose.zero.tolist(),
        mapping_rule='q_model=sign*(firmware_calibrated_continuous_q-zero); no driver calibration applied twice',
        protected_sha256=protected,physics_steps=0,optimizer_updates=0,tracks=[])
    with ZipFile(args.source) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.csv'):continue
            raw=archive.read(name);stem=Path(name).stem
            (args.output/'inputs'/Path(name).name).write_bytes(raw)
            reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')));rows=list(reader)
            data={k:np.array([float(r[k]) for r in rows]) for k in reader.fieldnames}
            phase=data['stair_phase'].astype(int);active=np.isin(phase,[1,2,5])
            onset=int(np.flatnonzero(active)[0]);end=int(np.flatnonzero(active)[-1])+1
            t=(data['h7_us']-data['h7_us'][onset])*1e-6
            fwq=np.stack([data['angle_rad_'+j] for j in CSV_JOINTS],-1)
            fwv=np.stack([data['velocity_rad_s_'+j] for j in CSV_JOINTS],-1)
            qleg=(fwq-pose.zero)*pose.sign[:4];vleg=fwv*pose.sign[:4]
            wheelv=np.stack([data['wheel_velocity_l_rad_s'],data['wheel_velocity_r_rad_s']],-1)*pose.sign[4:]
            requests=np.stack([data['request_nm_'+j] for j in (*CSV_JOINTS,'lw','rw')],-1)*pose.sign
            lengths,angles=firmware_pose(fwq)
            logged_l=np.stack([data[s+'_length_m'] for s in ('left','right')],-1)
            logged_a=np.stack([data[s+'_angle_rad'] for s in ('left','right')],-1)
            geometry_l=float(abs(lengths-logged_l).max())
            geometry_a=float(abs(np.arctan2(np.sin(angles-logged_a),np.cos(angles-logged_a))).max())
            poses=[];closure=[];violations=[];wheel_local=[]
            pose.data.qpos[:]=model.qpos0
            for i,q in enumerate(qleg):
                pose.data.qpos[:7]=[0.,0.,.6,1.,0.,0.,0.]
                ok=all(pose._solve_leg(s,q[2*s:2*s+2]) for s in (0,1))
                if not ok:raise ValueError(f'{stem}: closed-chain solve failed at frame{i}')
                closure.append(max(pose.residual_m))
                violation=[]
                for j in range(model.njnt):
                    if model.jnt_limited[j]:
                        value=pose.data.qpos[model.jnt_qposadr[j]]
                        violation.append(max(model.jnt_range[j,0]-value,value-model.jnt_range[j,1],0.))
                violations.append(max(violation,default=0.))
                wheel_local.append([pose.data.body(w).xpos-pose.data.xanchor[model.joint(mirror.DRIVES[s][0]).id]
                                    for s,w in enumerate(mirror.WHEELS)])
                poses.append(pose.data.qpos.copy())
            poses=np.array(poses)
            qvel=np.zeros((len(t),model.nv))
            for j in range(1,model.njnt):
                qvel[:,model.jnt_dofadr[j]]=np.gradient(poses[:,model.jnt_qposadr[j]],t)
            qvel[:,[model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]]]=vleg
            qvel[:,[model.joint(n).dofadr[0] for n in mirror.WHEELS]]=wheelv
            np.savez_compressed(args.output/'tracks'/f'{stem}.npz',time=t,phase=phase,
                motor_q=qleg,motor_v=vleg,wheel_v=wheelv,request_nm=requests,
                qpos=poses,qvel=qvel,quat=np.stack([data['quat_'+a] for a in 'wxyz'],-1),
                gyro=np.stack([data['gyro_'+a+'_rad_s'] for a in 'xyz'],-1),
                closure_m=closure,limit_violation=violations,wheel_local=wheel_local,
                length=lengths,angle=angles)
            item=dict(id=stem,frames=len(t),onset_index=onset,end_index=end,
                source_csv_sha256=sha(args.output/'inputs'/Path(name).name),
                track_sha256=sha(args.output/'tracks'/f'{stem}.npz'),
                geometry_length_max_error_m=geometry_l,geometry_angle_max_error_rad=geometry_a,
                closure_max_m=float(max(closure)),limit_violation_peak=float(max(violations)),
                active_limit_violation_peak=float(np.array(violations)[active].max()),
                geometry_verified=geometry_l<.001 and geometry_a<.01,
                all_frames_within_limits=bool(max(violations)<1e-6))
            report['tracks'].append(item)
            print(json.dumps(item),flush=True)
    report['protected_unchanged']=all(sha(Path(p))==value for p,value in protected.items())
    assert report['protected_unchanged']
    write(args.output/'alignment.json',report)


def run(args):
    import torch
    from scipy.spatial.transform import Rotation
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,recipe_record
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    from .evaluate import longest
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'traces').mkdir()
    alignment=json.loads((args.alignment/'alignment.json').read_text())
    for path,digest in alignment['protected_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Mapping/model changed since alignment: '+path)
    tracks={}
    for item in alignment['tracks']:
        path=args.alignment/'tracks'/f'{item["id"]}.npz'
        assert sha(path)==item['track_sha256'] and item['geometry_verified']
        with np.load(path) as saved:tracks[item['id']]={k:saved[k] for k in saved.files}
    # 真实触发距离缺测：仅声明两个假设，不按结果调零点、修改轨迹或改限位。
    cases=[dict(id=f'{name}_{mode}_g{gap:.2f}',track=name,control=mode,gap_m=gap,
                height_m=.2,terrain='up',initial_position_assumed=True)
           for name in tracks for mode in ('positions','feedforward') for gap in (0.,.06)]
    if args.window=='active':
        cases += [dict(id=f'{name}_feedforward_flat',track=name,control='feedforward',gap_m=0.,
                       height_m=0.,terrain='flat',initial_position_assumed=True) for name in tracks]
    if args.support_sweep:
        assert args.window=='active'
        legal=[item['id'] for item in alignment['tracks'] if item['all_frames_within_limits']]
        # 只检验原限位内的5条记录；已抬高或已上台的初态均不具有完整登台资格。
        supports=('lower','corner30','corner60','top') if args.observe_contacts else ('corner30','corner60','top')
        cases=[dict(id=f'{name}_{support}',track=name,control='feedforward',gap_m=0.,height_m=.2,
                    terrain='up',support_assumption=support,ascent_eligible=support=='lower',initial_position_assumed=True)
               for name in legal for support in supports]
    seconds=3. if args.window=='full' else 2.
    cfg=FudanStepCfg();cfg.step.curriculum=False;cfg.step.side_obstacle_level=99
    cfg.step.max_approach_speed=3.
    report=dict(backend='mujoco_warp_gpu',cases=cases,recipe=recipe_record(cfg),
        alignment=str(args.alignment.resolve()),alignment_sha256=sha(args.alignment/'alignment.json'),
        seconds=seconds,window=args.window,physics_dt=.001,pd_hz=200,reference_action_hz=100,
        input_sampling_hz=200,action_delay_ms=15,sensor_delay_ms=15,
        initial_state_assumptions=['Recorded first hip pose and yaw-removed IMU orientation',
            'Lowest wheel center at60mm above floor; actual root height unmeasured',
            'Initial vx and pre-trigger displacement assume wheel no-slip',
            'At phase1 wheels tangent to edge or60mm before edge under no-slip pre-motion',
            'Root vertical/lateral initial velocity zero; no forcing after initial placement'],
        reference='Measured motion tracking, with optional recorded-torque feedforward through unchanged PD',
        source_torque_is_request=True,optimizer_updates=0,network_forwards=0,
        learned_skill=False,formal_acceptance=False,root_forcing_after_initialization=False,
        parking_assessed=False,trace_sha256={},contact_failure_observation_only=args.observe_contacts,
        task_contact_failure_not_waived=True)
    if args.support_sweep:
        report['initial_state_assumptions']=[
            'Unmeasured support is a counterfactual hypothesis, not reconstructed ground truth',
            'corner30/corner60: lowest wheel center on60mm circle outside200mm edge at30/60 degrees',
            'top: wheels initially100mm inside platform; never counts as ascent',
            'Recorded trigger pose/gyro/joint velocities; root linear velocity estimated from no-slip tangent minus joint motion',
            'No root forcing after initial placement; even successful elevated segments do not establish full ascent']
    runtime=args.output/'source-runtime';runtime.mkdir()
    sources=[]
    for folder in ('envs/fudan_ppo_step','envs/fudan_ppo_walk','envs/chuanliantui','fudan_ppo_walk'):
        sources.extend((ROOT/'mjlab_training/src/ct_mjlab'/folder).glob('*.py'))
    manifest={}
    for source in sources:
        relative=source.relative_to(ROOT);destination=runtime/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        manifest[str(relative)]=sha(source)
    write(args.output/'runtime-source-manifest.json',manifest)
    shutil.copy2(Path(__file__),args.output/'source-replay_traditional_tracks.py')
    write(args.output/'protocol.json',report)
    env=None;records=[];torch.set_num_threads(1)
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Replay cannot train')):
            env=StepEnv(len(cases),cfg=cfg,randomized=False,seed=361161)
            n=env.num_envs;device=env.device;ids=torch.arange(n,device=device)
            env.terrain.terrain_types.fill_(1);env.terrain.terrain_levels.fill_(7)
            for i,case in enumerate(cases):
                if case['terrain']=='flat':env.terrain.terrain_types[i]=0;env.terrain.terrain_levels[i]=0
            env.fixed_commands=torch.tensor([[0.,0.,.22]]*n,device=device)
            env.env.sim.model.geom_friction[:,:,0]=.8;env.env._fudan_parameters['friction'].fill_(.8)
            env.action.disturbances.action_steps.fill_(15);env.action.disturbances.sensor_steps.fill_(15)
            env.reset()
            model=env.env.sim.mj_model
            sys.path.insert(0,str(ROOT/'sim2sim'))
            import real2sim_huananhu as mirror
            local=mujoco.MjModel.from_xml_path(str(XML));local_data=mujoco.MjData(local)
            initial=env.env.sim.data.qpos.clone();velocity=env.env.sim.data.qvel.clone()
            origin=env.env.scene.env_origins.cpu().numpy().copy()
            init_rows=[]
            for i,case in enumerate(cases):
                track=tracks[case['track']]
                first=0 if args.window=='full' else int(np.flatnonzero(track['time']>=0)[0])
                q=track['qpos'][first].copy();v=track['qvel'][first].copy()
                quat=track['quat'][first];rotation=Rotation.from_quat(quat[[1,2,3,0]])
                yaw=rotation.as_euler('ZYX')[0]
                aligned=Rotation.from_euler('z',-yaw)*rotation
                aq=aligned.as_quat();q[3:7]=aq[[3,0,1,2]];q[:3]=0.
                local_data.qpos[:]=q;mujoco.mj_kinematics(local,local_data)
                wheels=np.array([local_data.body(w).xpos for w in mirror.WHEELS])
                wheel_direction=np.array([local.joint(w).axis[1] for w in mirror.WHEELS])
                speed=(track['wheel_v']*wheel_direction).mean(-1)*.06
                start=np.flatnonzero(track['time']>=0)[0]
                displacement=float(np.trapezoid(speed[first:start+1],track['time'][first:start+1])) if first<start else 0.
                q[0]=origin[i,0]+1.2-.06-case['gap_m']-displacement-wheels[:,0].max()
                q[1]=origin[i,1];q[2]=origin[i,2]+.06-wheels[:,2].min()
                v[:6]=[float(speed[first]),0.,0.,*track['gyro'][first]]
                if args.support_sweep and case['support_assumption']!='lower':
                    support=case['support_assumption'];edge=origin[i,0]+1.2
                    if support=='top':
                        q[2]=origin[i,2]+.26-wheels[:,2].min()
                        q[0]=edge+.10-wheels[:,0].min();desired=np.array([float(speed[first]),0.,0.])
                    else:
                        angle=np.deg2rad(float(support.removeprefix('corner')))
                        q[2]=origin[i,2]+.2+.06*np.sin(angle)-wheels[:,2].min()
                        dz=wheels[:,2]+q[2]-origin[i,2]-.2
                        q[0]=min(edge-np.sqrt(max(.06**2-z*z,0.))-x for x,z in zip(wheels[:,0],dz))
                        desired=np.array([float(speed[first])*np.sin(angle),0.,float(speed[first])*np.cos(angle)])
                    # 用相同关节/陀螺速度扣掉轮心相对机身运动；只是假设的无滑动初速。
                    v[:3]=0.;local_data.qpos[:]=q;mujoco.mj_kinematics(local,local_data);mujoco.mj_comPos(local,local_data)
                    relative=[]
                    for wheel in mirror.WHEELS:
                        jac=np.zeros((3,local.nv));mujoco.mj_jacBody(local,local_data,jac,None,local.body(wheel).id)
                        relative.append(jac@v)
                    v[:3]=desired-np.mean(relative,axis=0)
                initial[i,:7]=torch.tensor(q[:7],device=device);velocity[i,:6]=torch.tensor(v[:6],device=device)
                for j in range(1,local.njnt):
                    name=local.joint(j).name;target=model.joint('robot/'+name)
                    initial[i,target.qposadr[0]]=q[local.jnt_qposadr[j]]
                    velocity[i,target.dofadr[0]]=v[local.jnt_dofadr[j]]
                init_rows.append(dict(id=case['id'],first_record_index=first,estimated_initial_vx=float(speed[first]),
                    assumed_pretrigger_displacement_m=displacement,root=q[:7].tolist(),root_velocity=v[:6].tolist()))
            env.env.sim.data.qpos.copy_(initial);env.env.sim.data.qvel.copy_(velocity)
            env.env.sim.forward();env.env.scene.update(dt=0.);env.env.sim.sense();env._update_state()
            env.previous_position.copy_(env.position);env.previous_vel.copy_(env.velocity)
            env.action.last_sample_q.copy_(env.env.sim.data.qpos[:,env.action._motor_q_adr])
            env.action.sample_velocity.zero_()
            env.action.disturbances.sensors[:]=env.action.sensor_frame()[None]
            env.contacts.reset(ids);env.contacts.record_substeps=True
            mujoco.mj_saveModel(model,str(args.output/'model.mjb'),None)
            motor_ids=[model.actuator('robot/'+name+'_motor').id for name in MOTORS]
            hinge_ids=[j for j in range(model.njnt) if model.jnt_limited[j] and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
            hinge_adr=torch.tensor(model.jnt_qposadr[hinge_ids],dtype=torch.long,device=device)
            hinge_range=torch.tensor(model.jnt_range[hinge_ids],device=device)
            slide_ids=[j for j in range(model.njnt) if model.jnt_limited[j] and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE]
            slide_adr=torch.tensor(model.jnt_qposadr[slide_ids],dtype=torch.long,device=device)
            slide_range=torch.tensor(model.jnt_range[slide_ids],device=device)
            alive=torch.ones(n,dtype=torch.bool,device=device);reasons=['duration']*n
            sub=[];original_update=env.contacts.update
            def update():
                result=original_update()
                q=env.env.sim.data.qpos[:,hinge_adr]
                excess=torch.maximum((hinge_range[:,0]-q).clamp_min(0),(q-hinge_range[:,1]).clamp_min(0)).amax(-1)
                sq=env.env.sim.data.qpos[:,slide_adr]
                stroke=(torch.maximum((slide_range[:,0]-sq).clamp_min(0),(sq-slide_range[:,1]).clamp_min(0)).amax(-1)
                        if len(slide_ids) else torch.zeros(n,device=device))
                sub.append(torch.cat((env.env.sim.data.xpos[:,env.wheel_ids].flatten(1),
                    env.env.sim.data.ctrl[:,motor_ids],env.contacts.supported.float(),excess[:,None],stroke[:,None],
                    env.env.sim.data.xfrc_applied.abs().flatten(1).amax(-1)[:,None]),-1).detach().clone())
                return result
            env.contacts.update=update
            legs=env.action._leg_action_ids;wheels=env.action._wheel_action_ids
            pd_rows=[];original_apply=env.action.apply_actions
            def apply():
                q=env.env.sim.data.qpos[:,env.action._motor_q_adr].clone()
                v=env.env.sim.data.qvel[:,env.action._motor_v_adr].clone()
                tick=env.action._pd_tick
                original_apply()
                pd_rows.append(torch.cat((q,v,env.action._control_actions.clone(),env.action._torques.clone(),
                    q.new_full((n,1),tick)),dim=-1))
            env.action.apply_actions=apply
            report['pd_fixture']=dict(nominal=env.action.nominal_positions.cpu().tolist(),
                reference=env.action.reference_positions.cpu().tolist(),
                kp_scale=env.action.kp_scale.cpu().tolist(),kd_scale=env.action.kd_scale.cpu().tolist(),
                strength=env.action.motor_strength.cpu().tolist(),motor_q=env.action._motor_q_adr.cpu().tolist(),
                action_delay_steps=env.action.disturbances.action_steps.cpu().tolist(),
                wheel_response=env.action._wheel_response.cpu().tolist())
            report['initialization']=init_rows
            report['actual_material_friction_range']=[float(env.env.sim.model.geom_friction[:,:,0].min()),
                                                       float(env.env.sim.model.geom_friction[:,:,0].max())]
            with torch.inference_mode():
                for step in range(round(seconds/.01)):
                    if time.time()>=args.deadline:raise TimeoutError('Finite replay deadline reached')
                    action=torch.zeros(n,6,device=device);references=[];targets=[]
                    for i,case in enumerate(cases):
                        track=tracks[case['track']];when=float((track['time'][0] if args.window=='full' else 0.)+step*.01)
                        sample=lambda key:np.array([np.interp(when,track['time'],column) for column in track[key].T])
                        qref=sample('motor_q');vref=sample('motor_v');wvref=sample('wheel_v');tau=sample('request_nm')
                        target=qref.copy();wvtarget=wvref.copy()
                        if case['control']=='feedforward':
                            target+=(tau[:4]+vref)/20.
                            wvtarget+=tau[4:]/.1
                        action[i,legs]=(torch.tensor(target,device=device,dtype=action.dtype)-env.action.nominal_positions[i,legs])/.5
                        action[i,wheels]=torch.tensor(wvtarget,device=device,dtype=action.dtype)/10.
                        references.append(qref);targets.append(target)
                    action[~alive]=0.;valid=alive.clone();sub.clear();pd_rows.clear()
                    _,_,_,done,info,_=env.step(action,auto_reset=False,resample=False)
                    if args.observe_contacts:
                        # 只延长物理观察；非轮接触失败仍逐拍保存，评分不豁免撑沿。
                        relative=env.position[:,:2]-env.env.scene.env_origins[:,:2]
                        wp=env.env.sim.data.xpos[:,env.wheel_ids]
                        dist=wp[:,:,0]-env.env.scene.env_origins[:,0,None]-1.2
                        lane=(abs(wp[:,:,1]-env.env.scene.env_origins[:,1,None])<1.148/2-.04).all(-1)
                        outside=(relative[:,1].abs()>1.6)|(relative[:,0].abs()>3.5)
                        outside |= (dist.amax(-1)>-.05)&~lane
                        outside |= dist.amax(-1)>cfg.step.depth-.25
                        done=(env.fail_ticks>cfg.env.fail_ticks)|~info['terrain_valid']|outside|(env.max_approach_speed>3.)
                    ticks=torch.stack(sub,1)
                    assert ticks.shape[1]==10 and torch.isfinite(env.env.sim.data.qpos).all()
                    records.append(dict(qpos=env.env.sim.data.qpos.cpu().numpy().copy(),
                        qvel=env.env.sim.data.qvel.cpu().numpy().copy(),valid=valid.cpu().numpy(),
                        force=torch.stack(env.contacts.step_forces,1).cpu().numpy(),
                        contact_position=torch.stack(env.contacts.step_positions,1).cpu().numpy(),
                        wheel_sub=ticks[:,:,:6].reshape(n,10,2,3).cpu().numpy(),torque_sub=ticks[:,:,6:12].cpu().numpy(),
                        wheel_support_sub=ticks[:,:,12:14].cpu().numpy(),joint_limit_violation=ticks[:,:,14].cpu().numpy(),
                        slide_limit_violation=ticks[:,:,15].cpu().numpy(),external_wrench_peak=ticks[:,:,16].cpu().numpy(),
                        pd_sub=torch.stack(pd_rows,1).cpu().numpy(),
                        strict_contact_failed=env.contacts.failed.cpu().numpy().copy(),
                        task_contact_failed=env._contact_failure().cpu().numpy().copy(),
                        action=action.cpu().numpy(),reference_motor_q=np.array(references),target_motor_q=np.array(targets)))
                    for i in (alive&done).nonzero().flatten().tolist():
                        reasons[i]='missing_terrain' if not info['terrain_valid'][i] else (
                            'nonwheel_contact' if env._contact_failure()[i] and not args.observe_contacts else 'fall_or_boundary')
                    alive &= ~done
                    if not alive.any():break
                    dead=(~alive).nonzero().flatten()
                    if len(dead):env.reset_idx(dead)
            rows=[]
            for i,case in enumerate(cases):
                trace={key:np.stack([r[key][i] for r in records]) for key in records[0]}
                trace.update(origin=origin[i],initial=initial[i].cpu().numpy(),stop_reason=np.array(reasons[i]))
                file=args.output/'traces'/f'{case["id"]}.npz';np.savez_compressed(file,**trace)
                report['trace_sha256'][case['id']]=sha(file)
                count=int(trace['valid'].sum());wheel=trace['wheel_sub'][:count].reshape(-1,2,3)
                force=trace['force'][:count].reshape(-1,2)
                support=trace['wheel_support_sub'][:count].reshape(-1,2).astype(bool)
                top=((wheel[:,:,0]>origin[i,0]+1.26)&(abs(wheel[:,:,2]-origin[i,2]-.26)<.02)&support).all(-1)
                if case['terrain']=='flat':top[:]=False
                top &= (abs(wheel[:,:,1]-origin[i,1])<1.148/2-.04).all(-1)
                motor_q=trace['qpos'][:count,env.action._motor_q_adr[legs].cpu().numpy()]
                error=motor_q-trace['reference_motor_q'][:count]
                clean=bool(force.max()<=1.)
                limit=float(trace['joint_limit_violation'][:count].max())
                stroke=float(trace['slide_limit_violation'][:count].max())
                row=dict(**case,valid_seconds=count*.01,reason=reasons[i],
                    reference_joint_rmse_rad=np.sqrt((error*error).mean(0)).tolist(),
                    body_leg_peak_N=force.max(0).tolist(),body_leg_contact_longest_s=[longest(force[:,j]>1)*.001 for j in (0,1)],
                    wheel_max_height_above_floor_m=(wheel[:,:,2]-origin[i,2]).max(0).tolist(),
                    both_top_support_longest_s=longest(top)*.001,landed=bool(longest(top)>=100),
                    nonwheel_free=clean,joint_limit_violation_peak_rad=limit,slide_violation_peak_m=stroke,
                    clean_reference_ascent=bool(longest(top)>=100 and clean and limit<=.02 and stroke<=.001
                                               and case.get('ascent_eligible',True)))
                track=tracks[case['track']];source_end=float(track['time'][np.flatnonzero(np.isin(track['phase'],[1,2,5]))[-1]])+.005
                row['source_motion_end_s']=source_end
                row['reference_duration_covered']=bool(count*.01>=source_end)
                row['contactfree_duration_covered']=bool(count*.01>=source_end and clean and limit<=.02 and stroke<=.001)
                failed=np.flatnonzero(trace['task_contact_failed'][:count])
                row['first_task_contact_failure_s']=float((failed[0]+1)*.01) if len(failed) else None
                row['conditional_edge_transfer']=bool(case.get('support_assumption','').startswith('corner') and
                    longest(top)>=100 and clean and limit<=.02 and stroke<=.001)
                rows.append(row)
            write(args.output/'metrics.json',rows)
            report.update(physics_ticks=int(env.contacts.total_ticks),constraint_peak=int(env.contacts.constraint_peak),
                capacity_overflow=bool(env.contacts.capacity_overflow.any()),
                clean_reference_ascents=[r['id'] for r in rows if r['clean_reference_ascent']],
                protected_unchanged=all(sha(Path(p))==d for p,d in alignment['protected_sha256'].items()))
            assert report['protected_unchanged'] and not report['capacity_overflow']
            print(json.dumps(dict(cases=n,landed=sum(r['landed'] for r in rows),clean=report['clean_reference_ascents'],optimizer_updates=0)),flush=True)
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        write(args.output/'protocol.json',report)


def audit(args):
    from .evaluate import longest
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.recording/'protocol.json').read_text())
    fixture=protocol['pd_fixture'];rows=[];maximum_pd=0.;maximum_delay=0.;maximum_torque=0.
    caps=np.array([40.,40.,3.9,40.,40.,3.9]);legs=[0,1,3,4];wheels=[2,5]
    saved_metrics=json.loads((args.recording/'metrics.json').read_text())
    alignment=Path(protocol['alignment'])
    assert sha(alignment/'alignment.json')==protocol['alignment_sha256']
    geometry_rows=[]
    for path in sorted((alignment/'tracks').glob('*.npz')):
        with np.load(path) as track:
            vector=track['wheel_local']
            error_l=float(abs(np.linalg.norm(vector[:,:,[0,2]],axis=-1)-track['length']).max())
            angle=np.arctan2(vector[:,:,0],-vector[:,:,2])
            error_a=float(abs(np.arctan2(np.sin(angle-track['angle']),np.cos(angle-track['angle']))).max())
            assert error_l<1e-7 and error_a<1e-7
            geometry_rows.append(dict(track=path.stem,model_firmware_length_error_m=error_l,
                                      model_firmware_angle_error_rad=error_a))
    scene=mujoco.MjModel.from_binary_path(str(args.recording/'model.mjb'));state=mujoco.MjData(scene)
    robot_bodies={j for j in range(scene.nbody) if scene.body(j).name.startswith('robot/')}
    wheel_bodies={scene.body('robot/'+w).id for w in ('lfwheel','rfwheel')}
    for i,case in enumerate(protocol['cases']):
        path=args.recording/'traces'/f'{case["id"]}.npz'
        assert sha(path)==protocol['trace_sha256'][case['id']]
        with np.load(path) as saved:t={k:saved[k] for k in saved.files}
        count=int(t['valid'].sum());assert count and t['valid'][:count].all() and not t['valid'][count:].any()
        assert all(np.isfinite(value).all() for value in t.values() if value.dtype.kind in 'fc')
        # 只检查出生几何，不推进CPU积分，也不把CPU接触力当作GPU证据。
        state.qpos[:]=t['initial'];mujoco.mj_forward(scene,state)
        initial_penetration=0.
        for contact in state.contact:
            a,b=scene.geom_bodyid[contact.geom]
            if (a in robot_bodies)==(b in robot_bodies):continue
            body=a if a in robot_bodies else b
            if body not in wheel_bodies:initial_penetration=max(initial_penetration,float(-contact.dist))
        pd=t['pd_sub'][:count].reshape(-1,25);q,v,control,torque=pd[:,:6],pd[:,6:12],pd[:,12:18],pd[:,18:24]
        commanded=np.repeat(t['action'][:count].clip(-100,100),10,axis=0)
        delay=fixture['action_delay_steps'][i]
        delayed=np.zeros_like(commanded);delayed[delay:]=commanded[:-delay] if delay else commanded
        delay_error=float(abs(delayed-control).max());maximum_delay=max(maximum_delay,delay_error)
        ref=np.broadcast_to(np.array(fixture['reference']),(len(protocol['cases']),6))[i]
        nominal=np.array(fixture['nominal'][i])
        kp=np.array(fixture['kp_scale'][i]);kd=np.array(fixture['kd_scale'][i]);strength=np.array(fixture['strength'][i])
        target=np.clip(nominal[legs]+.5*control[:,legs],ref[legs]-3.14,ref[legs]+3.14)
        expected=np.zeros_like(torque)
        expected[:,legs]=20.*kp[legs]*(target-q[:,legs])-kd[legs]*v[:,legs]
        expected[:,wheels]=kd[wheels]*.1*(np.clip(10.*control[:,wheels],-150,150)-v[:,wheels])
        expected=np.clip(expected*strength,-caps,caps)
        updates=pd[:,24]==0
        pd_error=float(abs(expected[updates]-torque[updates]).max());maximum_pd=max(maximum_pd,pd_error)
        # 非更新拍必须保持上一拍的标称限幅力矩。
        hold=np.flatnonzero(~updates);assert not len(hold) or abs(torque[hold]-torque[hold-1]).max()<1e-6
        maximum_torque=max(maximum_torque,float((abs(torque)-caps).max()))
        assert delay_error<1e-6 and pd_error<5e-5
        assert (abs(torque)<=caps+1e-5).all() and t['external_wrench_peak'][:count].max()==0
        force=t['force'][:count].reshape(-1,2);wheel=t['wheel_sub'][:count].reshape(-1,2,3)
        supported=t['wheel_support_sub'][:count].reshape(-1,2).astype(bool);origin=t['origin']
        top=((wheel[:,:,0]>origin[0]+1.26)&(abs(wheel[:,:,2]-origin[2]-.26)<.02)&supported).all(-1)
        top &= (abs(wheel[:,:,1]-origin[1])<1.148/2-.04).all(-1)
        if case.get('terrain','up')=='flat':top[:]=False
        assert bool(longest(top)>=100)==saved_metrics[i]['landed']
        assert np.allclose(force.max(0),saved_metrics[i]['body_leg_peak_N'],rtol=0,atol=1e-6)
        if protocol.get('contact_failure_observation_only',False):
            failed=t['task_contact_failed'][:count]
            assert np.diff(failed.astype(int)).min(initial=0)>=0
            if (force>1).any():assert not saved_metrics[i]['clean_reference_ascent']
            if not case['ascent_eligible']:assert not saved_metrics[i]['clean_reference_ascent']
        first=np.flatnonzero((force>1).any(-1))
        motor=t['qpos'][:count,np.array(fixture['motor_q'])[legs]]
        error=motor-t['reference_motor_q'][:count]
        assert np.allclose(np.sqrt((error*error).mean(0)),saved_metrics[i]['reference_joint_rmse_rad'])
        rows.append(dict(id=case['id'],delay_error=delay_error,pd_error_Nm=pd_error,
            initial_nonwheel_penetration_m=initial_penetration,
            first_nonwheel_contact_time_from_record_start_s=float((first[0]+1)*.001) if len(first) else None,
            target_motor_rmse_deg=np.rad2deg(np.sqrt((error*error).mean(0))).tolist(),
            source_track_sha256=sha(alignment/'tracks'/f'{case["track"]}.npz'),
            diagnostic_stop_class=('duration' if t['stop_reason'].item()=='duration' else 'hard_stop')
                if protocol.get('contact_failure_observation_only',False) else t['stop_reason'].item(),
            both_top_support_longest_s=longest(top)*.001,landed=bool(longest(top)>=100)))
    protected=json.loads((alignment/'alignment.json').read_text())['protected_sha256']
    unchanged=all(sha(Path(p))==digest for p,digest in protected.items())
    assert unchanged and protocol['optimizer_updates']==0 and not protocol['capacity_overflow']
    write(args.output/'audit.json',dict(evidence_passed=True,reference_ascent_passed=bool(protocol['clean_reference_ascents']),
        cases=len(rows),maximum_pd_error_Nm=maximum_pd,maximum_action_fifo_error=maximum_delay,
        torque_cap_excess_Nm=max(0.,maximum_torque),external_wrench_zero=True,protected_unchanged=unchanged,
        optimizer_updates=0,physics_steps=0,kinematic_mapping_crosscheck=geometry_rows,
        run_source_sha256=sha(args.recording/'source-replay_traditional_tracks.py'),rows=rows))
    print(json.dumps(dict(evidence_passed=True,cases=len(rows),maximum_pd_error_Nm=maximum_pd,
                         reference_ascent_passed=bool(protocol['clean_reference_ascents']))))


def render(args):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    import math
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.recording/'protocol.json').read_text())
    path=args.recording/'traces'/f'{args.case}.npz';assert sha(path)==protocol['trace_sha256'][args.case]
    with np.load(path) as saved:t={k:saved[k] for k in saved.files}
    count=int(t['valid'].sum());model=mujoco.MjModel.from_binary_path(str(args.recording/'model.mjb'))
    data=mujoco.MjData(model);model.vis.global_.offwidth=800;model.vis.global_.offheight=480
    camera=mujoco.MjvCamera();camera.azimuth=90.;camera.elevation=-12.;camera.distance=1.7
    fps=25;frames=math.ceil(count*.01*fps);movie=args.output/'replay-1x.mp4'
    with mujoco.Renderer(model,height=480,width=800) as renderer:
        with imageio.get_writer(movie,fps=fps,codec='libx264',quality=8,macro_block_size=None) as writer:
            for frame in range(frames):
                index=min(count-1,round(frame/fps/.01));data.qpos[:]=t['qpos'][index]
                mujoco.mj_kinematics(model,data);camera.lookat[:]=data.qpos[:3]+[.12,0.,-.1]
                renderer.update_scene(data,camera=camera)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                force=t['force'][index].max(0)
                task_failed=bool(t['task_contact_failed'][index]) if 'task_contact_failed' in t else force.max()>1
                draw.rectangle((0,0,800,62),fill='#872126' if task_failed or force.max()>1 else '#172a35')
                offset=-.5 if protocol['window']=='full' else 0.
                draw.text((8,5),f'GPU 1x | RECORDED MOTION TRACKING | NO LEARNING | phase time={offset+(index+1)*.01:.2f}s',fill='white')
                draw.text((8,25),f'{args.case} | body/leg={force.round(1)} N',fill='white')
                allowed=protocol['recipe']['step'].get('allow_temporary_support',False)
                label=('TASK CONTACT FAILED; diagnostic continues' if task_failed else
                       ('Temporary support permitted' if allowed else 'Task contact threshold not reached'))
                draw.text((8,43),f'Assumed initial support | {label} | NOT FULL ASCENT',fill='white')
                writer.append_data(np.asarray(im))
                if frame in (0,frames//2,frames-1):im.save(args.output/f'key-{frame:03d}.png')
    reader=imageio.get_reader(movie);decoded=sum(1 for _ in reader);reader.close();assert decoded==frames
    write(args.output/'media.json',dict(case=args.case,fps=fps,frames=frames,duration_s=frames/fps,
        all_frames_decoded=True,video_sha256=sha(movie),trace_sha256=sha(path),new_physics_steps=0,optimizer_updates=0))


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.recording/'protocol.json').read_text())
    assert protocol['window']=='active'
    if any('support_assumption' in case for case in protocol['cases']):
        metrics={r['id']:r for r in json.loads((args.recording/'metrics.json').read_text())}
        ids=['A-12_lower' if 'A-12_lower' in metrics else 'A-12_corner30','A-12_corner60']
        figure,axes=plt.subplots(4,2,figsize=(11,10),sharex=True)
        for column,case in enumerate(ids):
            path=args.recording/'traces'/f'{case}.npz';assert sha(path)==protocol['trace_sha256'][case]
            with np.load(path) as saved:t={k:saved[k] for k in saved.files}
            count=int(t['valid'].sum());times=(np.arange(count)+1)*.01
            wheel=t['wheel_sub'][:count].reshape(-1,2,3);ft=(np.arange(len(wheel))+1)*.001
            force=t['force'][:count].reshape(-1,2)
            for side in (0,1):
                axes[0,column].plot(ft,wheel[:,side,2]-t['origin'][2],label=('left','right')[side])
                axes[1,column].plot(ft,wheel[:,side,0]-t['origin'][0]-1.2)
            axes[0,column].axhline(.26,color='black',linestyle=':',label='top wheel height')
            axes[1,column].axhline(0,color='black',linestyle=':')
            axes[2,column].plot(ft,force[:,0],label='body');axes[2,column].plot(ft,force[:,1],label='legs')
            motor_ids=np.array(protocol['pd_fixture']['motor_q'])[[0,1,3,4]]
            error=np.rad2deg(abs(t['qpos'][:count,motor_ids]-t['reference_motor_q'][:count])).mean(-1)
            axes[3,column].plot(times,error)
            failed=metrics[case]['first_task_contact_failure_s']
            for axis in axes[:,column]:
                axis.grid(alpha=.25);axis.set_xlim(0,2)
                if failed is not None:axis.axvspan(failed,2,color='red',alpha=.06)
            axes[0,column].set_title(case+'; task contact failure retained')
            axes[0,column].legend(fontsize=8);axes[2,column].legend(fontsize=8)
            axes[-1,column].set_xlabel('Time from recorded stair trigger [s]')
        for row,label in enumerate(['Wheel center above lower floor [m]','Wheel X minus edge [m]',
                                    'Nonwheel contact [N]','Mean absolute hip error [deg]']):axes[row,0].set_ylabel(label)
        figure.suptitle('A-12 support hypotheses; corner starts already elevated; no full ascent passed')
        figure.tight_layout();figure.savefig(args.output/'comparison.png',dpi=150);plt.close(figure)
        write(args.output/'plot.json',dict(cases=ids,new_physics_steps=0,optimizer_updates=0))
        return
    alignment=Path(protocol['alignment']);fixture=protocol['pd_fixture']
    with np.load(alignment/'tracks/A-12.npz') as saved:r={k:saved[k] for k in saved.files}
    figure,axes=plt.subplots(4,3,figsize=(13,10),sharex=True)
    ids=['A-12_positions_g0.00','A-12_feedforward_g0.00','A-12_feedforward_flat']
    for column,case in enumerate(ids):
        path=args.recording/'traces'/f'{case}.npz';assert sha(path)==protocol['trace_sha256'][case]
        with np.load(path) as saved:t={k:saved[k] for k in saved.files}
        count=int(t['valid'].sum());times=(np.arange(count)+1)*.01
        indices=np.array(fixture['motor_q'])[[0,1]]
        for j in (0,1):
            axes[0,column].plot(r['time'],r['motor_q'][:,j],linestyle='--',color=f'C{j}',alpha=.6)
            axes[0,column].plot(times,t['qpos'][:count,indices[j]],color=f'C{j}',label=('lf','lr')[j])
        q=r['quat'];rp=np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        q=t['qpos'][:count,3:7];sp=np.arcsin(np.clip(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        axes[1,column].plot(r['time'],np.rad2deg(rp),'--',color='C0',label='source quaternion')
        axes[1,column].plot(times,np.rad2deg(sp),color='C0',label='GPU quaternion')
        axes[2,column].plot(r['time'],r['request_nm'][:,0],'--',color='C0',label='recorded request')
        axes[2,column].plot(times,t['pd_sub'][:count,-1,18],color='C0',label='GPU nominal PD')
        force=t['force'][:count].reshape(-1,2);ft=(np.arange(len(force))+1)*.001
        axes[3,column].plot(ft,force[:,0],label='body');axes[3,column].plot(ft,force[:,1],label='leg')
        axes[0,column].set_title(case.replace('A-12_','')+'\n'+t['stop_reason'].item())
        for axis in axes[:,column]:axis.grid(alpha=.25);axis.set_xlim(0,1.2)
        axes[0,column].legend(fontsize=8);axes[3,column].legend(fontsize=8)
        axes[-1,column].set_xlabel('Time from recorded stair trigger [s]')
    for row,label in enumerate(['Left hip q [rad]','Quaternion pitch [deg]','Left-front torque [Nm]','Nonwheel contact [N]']):
        axes[row,0].set_ylabel(label)
    figure.suptitle('A-12 measured reference (dashed) vs GPU replay (solid); assumed root translation; no learning')
    figure.tight_layout();figure.savefig(args.output/'comparison.png',dpi=150);plt.close(figure)
    write(args.output/'plot.json',dict(source_recording=str(args.recording.resolve()),cases=ids,new_physics_steps=0,optimizer_updates=0))


def trigger(args):
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[CONFIG,ROOT/'deployment/H7_RL/imcalib/Algorithm/detection/stair_detection.c',
        ROOT/'deployment/H7_RL/imcalib/task/control/control_detection.c',
        ROOT/'deployment/H7_RL/tools/jetson/export_pd.py']
    manifest={}
    for path in sources:
        shutil.copy2(path,args.output/path.name);manifest[str(path.relative_to(ROOT))]=sha(path)
    rows=[]
    for path in sorted((args.alignment/'tracks').glob('*.npz')):
        with np.load(path) as track:
            before=(track['time']>=-.05)&(track['time']<0)
            after=(track['time']>=0)&(track['time']<.05)
            rows.append(dict(track=path.stem,
                wheel_v_model_mean_before_rad_s=track['wheel_v'][before].mean(0).tolist(),
                wheel_v_model_mean_after_rad_s=track['wheel_v'][after].mean(0).tolist(),
                request_mean_before_Nm=track['request_nm'][before].mean(0).tolist(),
                request_mean_after_Nm=track['request_nm'][after].mean(0).tolist()))
    with (args.alignment/'inputs/A-10.csv').open() as f:schema=next(csv.reader(f))
    exporter=sources[-1].read_text()
    report=dict(source_sha256=manifest,csv_columns=schema,rows=rows,
        existing_trigger_reads_contact_sensor=False,
        trigger_uses='World leg angle deviation plus filtered world leg angular acceleration; permitted by NORMAL/LQR/forward speed and length',
        missing=['World base translation/height','Wheel contact point and support surface','1kHz detection filter state and exact trigger sample','Firmware build identity'],
        local_exporter_has_stair_option='--stair' in exporter,
        source_version_bound_to_dataset=False,
        contact_surface_identifiable_from_csv=False,physics_steps=0,optimizer_updates=0)
    write(args.output/'trigger.json',report)
    print(json.dumps(dict(contact_surface_identifiable_from_csv=False,
        source_version_bound_to_dataset=False,tracks=len(rows),physics_steps=0)))


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--alignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--window',choices=('full','active'),default='full')
    p.add_argument('--support-sweep',action='store_true',help='Test support hypotheses; elevated starts cannot count as full ascent')
    p.add_argument('--observe-contacts',action='store_true',help='Continue bounded physics after contact failure, retaining failure labels')
    p=sub.add_parser('audit');p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('render');p.add_argument('--recording',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('plot');p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('trigger');p.add_argument('--alignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'audit':audit,'render':render,'plot':plot,'trigger':trigger}[args.mode](args)


if __name__=='__main__':main()
