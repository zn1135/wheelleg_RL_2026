"""Fixed-body GPU replay of recorded leg torque and length targets; no hardware."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from ct_mjlab.envs.chuanliantui import gas_spring as training_spring
from scripts.agent.real2sim import elastic_joint_resistance as elastic


def leg_state(q, v, lu=.21, lg=.25):
    """Same midpoint construction as H7; derivatives in firmware motor coordinates."""
    q = np.asarray(q, dtype=float) + [np.pi, 0.]
    ab = lu * np.column_stack((np.cos(q), np.sin(q)))
    delta = ab[1] - ab[0]; distance = np.linalg.norm(delta)
    height = np.sqrt(lg*lg - distance*distance/4)
    if distance < 1e-6 or not np.isfinite(height) or height < 1e-6:
        raise ValueError('Invalid or singular closed-chain geometry')
    rotate = lambda x: np.array([-x[1], x[0]])
    normal = rotate(delta)/distance
    point = (ab[0]+ab[1])/2 + height*normal
    length = np.linalg.norm(point); jac = []
    for j in range(2):
        da = rotate(ab[0]) if j==0 else np.zeros(2)
        db = rotate(ab[1]) if j==1 else np.zeros(2)
        dd = db-da; ds = delta@dd/distance
        dh = -distance*ds/(4*height)
        dn = rotate(dd)/distance-normal*ds/distance
        dp = (da+db)/2+dh*normal+height*dn
        jac.append(point@dp/length)
    jac = np.asarray(jac)
    return length, jac, float(jac@v)


def gains(config):
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {',1)[1].split('[MACHINE_ID_SMALL_WHEELLEG]',1)[0]
    match = re.search(r'\.leg_len\s*=\s*\{\{([^}]+)\},\s*\{([^}]+)\}',block)
    values = np.array([[float(x.strip().rstrip('fF')) for x in row.split(',')] for row in match.groups()])
    return values[:,0], values[:,2]*.001


def spring_state(slide_position, spring_model, slide_velocity=None, parameters=None):
    # XML零位已压缩10mm；滑动关节正方向为伸长，线性模型总行程90mm。
    compression=.010-np.asarray(slide_position)
    if spring_model=='linear-damped':
        if slide_velocity is None:raise ValueError('Damped spring needs actual simulated slider velocity')
        compression=np.maximum(compression,0.)
        if parameters is not None and 'spring_reference_force_n' in parameters:
            # 厂家额定推力候选：5mm压缩处为参考点；仅用于离线诊断，不改训练弹簧。
            if 'spring_force_knots_n' in parameters:
                grid=np.asarray(parameters['spring_compression_knots_m'])
                if grid.ndim>1:grid=grid[0]
                forces=np.broadcast_to(np.asarray(parameters['spring_force_knots_n']),compression.shape+(len(grid),))
                position=np.clip(compression,grid[0],grid[-1])
                index=np.clip(np.sum(position[...,None]>=grid,axis=-1)-1,0,len(grid)-2)
                lower=np.take_along_axis(forces,index[...,None],axis=-1)[...,0]
                upper=np.take_along_axis(forces,(index+1)[...,None],axis=-1)[...,0]
                elastic_force=lower+(upper-lower)*(position-grid[index])/(grid[index+1]-grid[index])
            else:
                elastic_force=np.asarray(parameters['spring_reference_force_n'])+np.asarray(parameters['spring_stiffness_n_m'])*(compression-.005)
            force=np.clip(elastic_force-
                          np.asarray(parameters.get('spring_damping_ns_m',50.))*np.asarray(slide_velocity),
                          0.,np.asarray(parameters['spring_force_limit_n']))
        else:
            bias=parameters.get('spring_bias_n',0.) if parameters is not None else 0.
            force=training_spring.gas_spring_force(slide_position,slide_velocity,bias)
    else:
        force=np.clip(150.*compression/.090,0.,150.) if spring_model=='linear' else np.full_like(compression,150.)
    return compression,force


def read_data(session, decoded, window_start=None, duration=8.):
    if not np.isfinite(duration) or duration<=0 or abs(duration/.001-round(duration/.001))>1e-6:
        raise ValueError('Duration must be a positive whole number of milliseconds')
    if window_start is not None and (not np.isfinite(window_start) or window_start<0):
        raise ValueError('Window start must be a finite nonnegative board time')
    meta, asset, config = mirror.load_session(session)
    report = json.loads((decoded/'report.json').read_text())
    if report['stream_sha256']!=meta['stream_sha256']:raise ValueError('Decoded data hash mismatch')
    parser = mirror.FrameParser(); frames = []
    for event in mirror.iter_session_events(session):
        if event['type']=='data':
            frames.extend(f for f in parser.feed(bytes.fromhex(event['data_hex'])) if int(f[152])==6)
    frames = np.asarray(frames)
    ft = np.array([mirror.read_u32(f[8:10])/1000 for f in frames])
    moved = np.flatnonzero(np.abs(frames[:,91])>1e-6)
    if not len(moved):raise ValueError('No active dial input')
    origin = ft[moved[0]]
    start=origin-.5 if window_start is None else window_start
    end=start+duration
    if ft[0]>start or ft[-1]<end:
        raise ValueError('Mirror data does not cover the entire requested window')
    window_frames=ft[(ft>=start)&(ft<=end)]
    if len(window_frames)<2 or np.max(np.diff(window_frames))>.03:
        raise ValueError('Mirror gap exceeds30ms in requested window')
    control=[]; state={}; geometry=[]; sequences=[]
    with (decoded/'outputs.csv').open() as f:
        for row in csv.DictReader(f):
            t=int(row['begin_ns'])/1e9
            if not start-.1<=t<=end+.1:continue
            sequences.append(int(row['sequence']))
            if row['strategy']!='6' or row['fault']!='0':
                if row['kind']!='can':raise ValueError('Window includes inactive or faulted control')
                continue
            key=int(row['id'])
            if row['kind']=='control':
                if int(row['flags_or_hal'])&62!=60:raise ValueError('Window has gated or failed output')
                control.append([t,key,*[float(row[f'request_{i}_firmware_nm']) for i in range(6)]])
            elif row['kind']=='leg_state':
                if max(float(row[f'leg_{i}_rx_age_us']) for i in range(4))>=5000:
                    raise ValueError('Window contains stale motor feedback')
                state[key]=[t,*[float(row[f'leg_{i}_position_rad']) for i in range(4)],*[float(row[f'leg_{i}_velocity_rad_s']) for i in range(4)]]
            elif row['kind']=='leg_geometry':
                if int(row['leg_valid_flags'])&15!=15:raise ValueError('Window has invalid leg geometry')
                geometry.append([t,key,float(row['left_leg_length_m']),float(row['right_leg_length_m']),
                    float(row['left_leg_test_target_m']),float(row['right_leg_test_target_m']),int(row['leg_valid_flags'])])
    if np.any(np.diff(sequences)!=1):raise ValueError('Replay window contains missing output records')
    control=np.asarray(control);geometry=np.asarray(geometry)
    if not len(control) or not len(geometry):raise ValueError('No control or paired geometry in requested window')
    if np.any(geometry[:,6].astype(int)&128):
        raise ValueError('10mm wheel-step capture: legacy continuous-rate feedback replay is unsupported; static analysis remains available')
    paired=np.asarray([state[int(g[1])] for g in geometry])
    if not all(np.isfinite(a).all() for a in (control,geometry,paired)):
        raise ValueError('Nonfinite input in requested window')
    if any(a[0,0]>start or a[-1,0]<end for a in (control,geometry,paired)):
        raise ValueError('Output data does not cover the entire requested window')
    if np.max(np.diff(control[:,0]))>.002:raise ValueError('Control input gap exceeds2ms')
    return meta,asset,config,report,frames,ft,control,geometry,paired,origin,start,end


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--decoded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-replay',type=Path,help='Recompute reports from saved GPU states; no additional physical steps')
    p.add_argument('--calibrated-model',type=Path,help='Explicit model override; only lf1/rf1 range changes allowed against the frozen recording model')
    p.add_argument('--response-parameters',type=Path,help='Diagnostic four-joint effective friction/damping JSON; never changes source XML or firmware')
    p.add_argument('--window-start-board-s',type=float,help='Explicit board time; integrate all reversals continuously from this initial state')
    p.add_argument('--duration-s',type=float,default=8.,help='Continuous replay duration; default preserves the original8s trial')
    p.add_argument('--prepare-inputs-only',action='store_true',help='Freeze verified inputs and measured initial state for finite candidate batches; no physical steps')
    p.add_argument('--spring-model',choices=('constant','linear','linear-damped'),default='constant',help='constant:150N; linear:elastic only; linear-damped:actual training 0..150N/90mm/CAD10mm plus damping50')
    limits=p.add_mutually_exclusive_group()
    limits.add_argument('--disable-knee-limits',action='store_true',help='Diagnostic copy only: disable lf1/rf1 limits without changing the source model')
    limits.add_argument('--extend-knee-limits',action='store_true',help='Diagnostic copy only: extend lf1 upper/rf1 lower bound from +/-0.12 to +/-0.15 rad')
    p.add_argument('--constraint-check',choices=('solver','stiff'),help='Diagnostic copy: tighter solver convergence; stiff also reduces compliance of all six loop connections')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    meta,asset,config,input_report,frames,ft,control,geometry,paired,origin,start,end=read_data(args.session,args.decoded,args.window_start_board_s,args.duration_s)
    recorded_asset=asset
    calibrated_changes={}
    if args.calibrated_model:
        if args.disable_knee_limits or args.extend_knee_limits:raise ValueError('Calibrated model cannot be mixed with historical knee overrides')
        original=ET.parse(asset).getroot();candidate=ET.parse(args.calibrated_model).getroot()
        for name in ('lf1','rf1'):
            old=original.find(f".//joint[@name='{name}']");new=candidate.find(f".//joint[@name='{name}']")
            calibrated_changes[name]={'old':old.get('range'),'new':new.get('range')}
            new.set('range',old.get('range'))
        if ET.tostring(original)!=ET.tostring(candidate):raise ValueError('Calibrated model changed geometry or parameters beyond two knee ranges')
        asset=args.calibrated_model
    kp,kd=gains(config);sign,zero=mirror.load_mapping(config)
    # 类型4 bit6区分新边界返回控制；旧实录仍按原越界锁止逻辑复现。
    boundary_flags=(geometry[:,6].astype(int)&64)!=0
    if np.any(boundary_flags!=boundary_flags[0]):raise ValueError('Mixed leg-boundary control versions in one window')
    soft_boundary=bool(boundary_flags[0])
    expected=[];actual=[];geom_error=[];gradient_error=[]
    controls={int(c[1]):c[2:6] for c in control}
    for g,s in zip(geometry,paired):
        torques=[]
        for side in range(2):
            q=s[1+side*2:3+side*2];v=s[5+side*2:7+side*2]
            length,jac,velocity=leg_state(q,v)
            geom_error.append(length-g[2+side])
            force=kp[side]*(g[4+side]-length)-kd[side]*velocity
            if int(g[6])&32:force=min(force,0.)
            if soft_boundary:
                if length>=.33:force=min(force,0.)
                elif length<=.14:force=max(force,0.)
            torques.extend(np.clip(jac*force,-40,40))
            for j in range(2):
                delta=np.zeros(2);delta[j]=1e-6
                numeric=(leg_state(q+delta,v)[0]-leg_state(q-delta,v)[0])/2e-6
                gradient_error.append(numeric-jac[j])
        expected.append(torques);actual.append(controls[int(g[1])])
    controller_error=np.asarray(expected)-actual
    if np.max(abs(controller_error))>.02 or np.max(abs(np.asarray(gradient_error)))>1e-6:
        raise ValueError('Controller reconstruction does not match recorded inputs')
    source_pose=mirror.MirrorPose(asset,config)
    init_index=max(0,np.searchsorted(ft,start,side='right')-1)
    frame=frames[init_index];source_pose.apply(frame,0.)
    initial_q=source_pose.data.qpos.copy();initial_residual=list(source_pose.residual_m)
    if any(x is None or x>1e-6 for x in initial_residual):raise ValueError('Initial pose closure failed')
    qinit=source_pose.sign[:4]*(paired[np.searchsorted(paired[:,0],start,side='right')-1,1:5]-source_pose.zero)
    for side in range(2):source_pose._solve_leg(side,qinit[2*side:2*side+2])
    initial_q=source_pose.data.qpos.copy()
    initial_velocity=paired[np.searchsorted(paired[:,0],start,side='right')-1,5:9]*sign[:4]
    eps=1e-5
    for side in range(2):source_pose._solve_leg(side,qinit[2*side:2*side+2]+eps*initial_velocity[2*side:2*side+2])
    full_velocity=np.zeros(source_pose.model.nv)
    mujoco.mj_differentiatePos(source_pose.model,full_velocity,eps,initial_q,source_pose.data.qpos)
    source_pose.data.qpos[:]=initial_q
    root=ET.parse(asset).getroot();compiler=root.find('compiler');compiler.set('meshdir',str((asset.parent/compiler.get('meshdir')).resolve()))
    base=root.find("worldbody/body[@name='base_link']");base.remove(base.find('freejoint'));base.set('pos','0 0 0.8');base.set('quat',' '.join(map(str,frame[22:26])))
    if args.disable_knee_limits:
        for name in ('lf1','rf1'):
            root.find(f".//joint[@name='{name}']").set('limited','false')  # 仅在本次诊断副本中关闭两膝限位。
    if args.extend_knee_limits:
        root.find(".//joint[@name='lf1']").set('range','-0.77 0.15')  # 覆盖本次记录需要的约0.139rad初态，保留屈腿端限位。
        root.find(".//joint[@name='rf1']").set('range','-0.15 0.77')
    if args.constraint_check:
        option=root.find('option')
        if option is None:option=ET.SubElement(root,'option')
        option.attrib.update(iterations='200',tolerance='1e-12',ls_iterations='100',ls_tolerance='0.0001')
    if args.constraint_check=='stiff':
        for connection in root.find('equality').findall('connect'):
            connection.set('solref','0.002 1')  # 只在诊断副本中减小连接点的软约束变形。
            connection.set('solimp','0.999 0.9999 0.001 0.5 2')
    response_parameters=None
    if args.response_parameters:
        if args.constraint_check!='stiff':raise ValueError('Response candidates require the verified stiff-loop diagnostic')
        response_parameters=json.loads(args.response_parameters.read_text())
        if not {'friction_nm','damping_nms_rad'}<=set(response_parameters) or not set(response_parameters)<={'friction_nm','damping_nms_rad','stiffness_nm_rad','armature_kg_m2'}:raise ValueError('Unexpected response parameter fields')
        resistance=np.asarray([response_parameters['friction_nm'],response_parameters['damping_nms_rad']],dtype=float)
        if resistance.shape!=(2,4) or not np.isfinite(resistance).all() or np.any(resistance<0):raise ValueError('Invalid resistance candidate')
        response_stiffness=np.asarray(response_parameters.get('stiffness_nm_rad',[0.]*4),dtype=float)
        if response_stiffness.shape!=(4,) or not np.isfinite(response_stiffness).all() or np.any(response_stiffness<0):raise ValueError('Invalid elastic stiffness')
        armature=np.asarray(response_parameters.get('armature_kg_m2',[.015795]*4),dtype=float)
        if armature.shape!=(4,) or not np.isfinite(armature).all() or np.any(armature<0):raise ValueError('Invalid effective armature')
        for i,name in enumerate(mirror.MOTORS[:4]):
            joint=root.find(f".//joint[@name='{name}']")
            joint.set('frictionloss',str(0. if response_stiffness[i]>0 else resistance[0,i]));joint.set('damping',str(resistance[1,i]))
            if 'armature_kg_m2' in response_parameters:joint.set('armature',str(armature[i]))
            joint.set('solreffriction','0.002 1');joint.set('solimpfriction','0.999 0.9999 0.001 0.5 2')
    xml=ET.tostring(root,encoding='unicode');(args.output/'fixed-base-model.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    model.geom_contype[:]=model.geom_conaffinity[:]=0
    model.opt.timestep=.001;model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    for j in range(model.njnt):
        name=model.joint(j).name;s=source_pose.model.joint(name)
        data.qpos[model.jnt_qposadr[j]]=initial_q[s.qposadr[0]]
        data.qvel[model.jnt_dofadr[j]]=full_velocity[s.dofadr[0]]
    mujoco.mj_forward(model,data)
    motorq=np.array([model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]])
    motorv=np.array([model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]])
    motors=np.array([model.actuator(n+'_motor').id for n in mirror.MOTORS])
    springs=np.array([model.actuator(n+'_gas_spring_motor').id for n in ('left','right')])
    springq=np.array([model.joint(n+'_spring2_joint').qposadr[0] for n in ('left','right')])
    springv=np.array([model.joint(n+'_spring2_joint').dofadr[0] for n in ('left','right')])
    spring_model={'constant':'constant_150N','linear':'linear_0_150N_90mm_CAD10mm','linear-damped':'linear_0_150N_90mm_CAD10mm_damping50'}[args.spring_model]
    limit_excess={}
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            q=data.qpos[model.jnt_qposadr[j]];lo,hi=model.jnt_range[j];excess=max(lo-q,q-hi,0.)
            if excess:limit_excess[model.joint(j).name]=float(excess)
    steps=round((end-start)/.001);ts=start+np.arange(steps+1)*.001
    if args.prepare_inputs_only:
        if args.reuse_replay:raise ValueError('Input preparation cannot reuse simulated states')
        indices=np.searchsorted(control[:,0],ts[:-1],side='right')-1
        if indices.min()<0:raise ValueError('Input preparation has no initial torque sample')
        requested=control[indices,2:8]*sign
        np.savez_compressed(args.output/'replay.npz',time_s=ts-start,qpos=data.qpos[None,None,:],
                            qvel=data.qvel[None,None,:],submitted_model_torque=requested[:,None,:],
                            real_geometry=geometry,real_motor_state=paired,initial_qpos=data.qpos,initial_qvel=data.qvel)
        np.savez_compressed(args.output/'rc-inputs.npz',time_s=ft,dial_rate=frames[:,91],stream_sha256=meta['stream_sha256'])
        summary=dict(input_only=True,environment_steps=0,physics_backend=None,
                     session=str(args.session),stream_sha256=meta['stream_sha256'],
                     source_model=str(asset),source_model_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                     config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                     spring_model=spring_model,training_spring_source_sha256=hashlib.sha256(Path(training_spring.__file__).read_bytes()).hexdigest(),
                     constraint_check=args.constraint_check,window_start_board_s=start,duration_s=end-start,dt_s=.001,
                     leg_boundary_mode='return_control' if soft_boundary else 'legacy_lockout',
                     controller_reconstruction_max_error_Nm=float(np.max(abs(controller_error))),
                     initial_joint_limit_excess_rad=limit_excess,initial_closure_m=initial_residual,
                     diagnostic_model_sha256=hashlib.sha256(xml.encode()).hexdigest())
        (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2));return
    qpos=np.empty((steps+1,2,model.nq));qvel=np.empty((steps+1,2,model.nv));submitted=np.zeros((steps,2,6))
    spring_compression=np.empty((steps,2,2));spring_forces=np.empty_like(spring_compression)
    solver_iterations=None
    knee_ids=np.array([model.joint(n).id for n in ('lf1','rf1')])
    knee_force=None;knee_present=None
    elastic_force=None;elastic_preload=None;minimum_loss=0.
    uses_elastic=response_parameters is not None and np.any(response_stiffness>0)
    if uses_elastic:
        ci=max(0,np.searchsorted(control[:,0],start,side='right')-1)
        preload,elastic_preload=elastic.initial_preload(model,data,control[ci,2:6]*sign[:4])
        elastic_mask=response_stiffness>0
        friction_state=np.tile(np.where(elastic_mask,np.clip(preload,-resistance[0],resistance[0]),0.),(2,1))
        applied=np.zeros((2,model.nv),dtype=np.float32)
    # 独立往返试验可能从正常控制段开始，按实录初态保留仅收腿标志。
    initial_geometry=max(0,np.searchsorted(geometry[:,0],start,side='right')-1)
    recovering=bool(int(geometry[initial_geometry,6])&32);blocked=False;began=time.monotonic();error=None;done=0
    if args.reuse_replay:
        saved=np.load(args.reuse_replay/'replay.npz');old=json.loads((args.reuse_replay/'report.json').read_text())
        if old.get('spring_model','constant_150N')!=spring_model:raise ValueError('Saved replay spring model does not match --spring-model')
        if old.get('response_parameters')!=response_parameters:raise ValueError('Saved response parameters differ')
        if old['stream_sha256']!=meta['stream_sha256'] or (args.reuse_replay/'fixed-base-model.xml').read_text()!=xml:raise ValueError('Saved replay source mismatch')
        if not np.array_equal(saved['time_s'],ts-start):raise ValueError('Saved replay time mismatch')
        qpos=saved['qpos'];qvel=saved['qvel'];submitted=saved['submitted_model_torque'];done=old['steps_per_world'];blocked=old['feedback_branch_blocked']
        spring_compression,spring_forces=spring_state(qpos[:-1,:,springq],args.spring_model,qvel[:-1,:,springv])
        if 'solver_iterations' in saved:solver_iterations=saved['solver_iterations']
        if 'knee_limit_force_nm' in saved:
            knee_force=saved['knee_limit_force_nm'];knee_present=saved['knee_limit_constraint_present']
        if uses_elastic:
            elastic_force=saved['elastic_resistance_nm'];minimum_loss=old['minimum_elastic_dissipation']
    else:
        import mujoco_warp as mw
        import warp as wp
        wp.init()
        solver_iterations=np.empty((steps,2),dtype=np.int32)
        knee_force=np.zeros((steps,2,2));knee_present=np.zeros((steps,2,2),dtype=bool)
        if uses_elastic:elastic_force=np.empty((steps,2,4))
        with wp.ScopedDevice('cuda:0'):
            gm=mw.put_model(model);gd=mw.put_data(model,data,nworld=2,nconmax=128,njmax=2048)
            with wp.ScopedCapture() as capture:mw.step(gm,gd)
            if np.any(gd.time.numpy()!=0):raise ValueError('Graph capture advanced the initial state')
            qp=gd.qpos.numpy();qv=gd.qvel.numpy();qpos[0]=qp;qvel[0]=qv
            for k,t in enumerate(ts[:-1]):
                if time.monotonic()-began>180:raise TimeoutError('Finite replay deadline exceeded')
                ci=max(0,np.searchsorted(control[:,0],t,side='right')-1);gi=max(0,np.searchsorted(geometry[:,0],t,side='right')-1);fi=max(0,np.searchsorted(ft,t,side='right')-1)
                ctrl=np.zeros((2,model.nu),np.float32)
                compression,spring_force=spring_state(qp[:,springq],args.spring_model,qv[:,springv])
                ctrl[:,springs]=spring_force
                spring_compression[k]=compression;spring_forces[k]=ctrl[:,springs]
                ctrl[0,motors]=control[ci,2:8]*sign
                tau=np.zeros(4);lengths=[]
                if not blocked:
                    try:
                        fwq=qp[1,motorq]*sign[:4]+zero;fwv=qv[1,motorv]*sign[:4]
                        for side in range(2):
                            length,jac,velocity=leg_state(fwq[2*side:2*side+2],fwv[2*side:2*side+2]);lengths.append(length)
                            if not soft_boundary and (length<.14 or (not recovering and length>.33)):raise ValueError('Feedback branch outside control envelope')
                            force=kp[side]*(geometry[gi,4+side]-length)-kd[side]*velocity
                            if recovering:force=min(force,0.)
                            if soft_boundary:
                                if length>=.33:force=min(force,0.)
                                elif length<=.14:force=max(force,0.)
                            tau[2*side:2*side+2]=np.clip(jac*force,-40,40)
                        if recovering and max(lengths)<=.33 and max(geometry[gi,4:6])<=.33 and frames[fi,91]==0:recovering=False
                    except ValueError:
                        blocked=True;tau[:]=0
                ctrl[1,motors[:4]]=tau*sign[:4]
                if uses_elastic:
                    previous_q=qp[:,motorq].copy();elastic_force[k]=friction_state
                    applied[:,motorv]=-friction_state;gd.qfrc_applied.assign(applied)
                submitted[k,0]=ctrl[0,motors];submitted[k,1]=ctrl[1,motors]
                gd.ctrl.assign(ctrl);wp.capture_launch(capture.graph)
                qp=gd.qpos.numpy();qv=gd.qvel.numpy();done+=1
                if uses_elastic:
                    friction_state,loss=elastic.advance(friction_state,np.where(elastic_mask,qp[:,motorq]-previous_q,0.),resistance[0],np.maximum(response_stiffness,1.))
                    minimum_loss=min(minimum_loss,loss)
                solver_iterations[k]=gd.solver_niter.numpy()
                # GPU限位约束的标量反力；铰链限位Jacobian为±1，因此单位为N·m。
                counts=gd.nefc.numpy();types=gd.efc.type.numpy();ids=gd.efc.id.numpy();forces=gd.efc.force.numpy()
                for branch in range(2):
                    for side,joint_id in enumerate(knee_ids):
                        mask=(types[branch,:counts[branch]]==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))&(ids[branch,:counts[branch]]==joint_id)
                        knee_present[k,branch,side]=np.any(mask)
                        knee_force[k,branch,side]=np.sum(np.abs(forces[branch,:counts[branch]][mask]))
                if not np.isfinite(qp).all() or not np.isfinite(qv).all():raise ValueError('Nonfinite GPU replay state')
                qpos[k+1]=qp;qvel[k+1]=qv
    extra={} if solver_iterations is None else {'solver_iterations':solver_iterations}
    extra.update(spring_force_n=spring_forces,spring_compression_m=spring_compression)
    if knee_force is not None:extra.update(knee_limit_force_nm=knee_force,knee_limit_constraint_present=knee_present)
    if elastic_force is not None:extra['elastic_resistance_nm']=elastic_force
    np.savez_compressed(args.output/'replay.npz',time_s=ts-start,qpos=qpos,qvel=qvel,submitted_model_torque=submitted,real_geometry=geometry,real_motor_state=paired,initial_qpos=data.qpos,initial_qvel=data.qvel,**extra)
    lengths=np.empty((len(ts),2,2));sensor_lengths=np.empty_like(lengths);sensor_q=np.empty((len(ts),2,4));closure_max=np.zeros((2,2))
    pin_residual=np.empty_like(lengths);spring_residual=np.empty_like(lengths)
    base_rotation=np.empty(9);mujoco.mju_quat2Mat(base_rotation,np.asarray(frame[22:26],dtype=float));base_rotation=base_rotation.reshape(3,3)
    for k in range(len(ts)):
        for branch in range(2):
            data.qpos[:]=qpos[k,branch];mujoco.mj_kinematics(model,data)
            sensor_q[k,branch]=qpos[k,branch,motorq]*sign[:4]+zero
            for side in range(2):
                delta=base_rotation.T@(data.body(mirror.WHEELS[side]).xpos-data.body(mirror.DRIVES[side][0]).xpos)
                lengths[k,branch,side]=np.hypot(delta[0],delta[2])
                sensor_lengths[k,branch,side]=leg_state(sensor_q[k,branch,2*side:2*side+2],np.zeros(2))[0]
                pin_residual[k,branch,side]=max(np.linalg.norm(data.site(a).xpos-data.site(b).xpos) for a,b in mirror.PIN_PAIRS[side])
                name=('left','right')[side]
                spring_residual[k,branch,side]=np.linalg.norm(data.site(f'{name}_spring_tip').xpos-data.site(f'{name}_gas_spring_lower').xpos)
                closure_max[branch,side]=max(closure_max[branch,side],pin_residual[k,branch,side])
    np.savez_compressed(args.output/'constraint-residuals.npz',time_s=ts-start,pin_residual_m=pin_residual,spring_residual_m=spring_residual)
    real=np.column_stack([np.interp(ts,geometry[:,0],geometry[:,2+i]) for i in range(2)])
    targets=np.column_stack([np.interp(ts,geometry[:,0],geometry[:,4+i]) for i in range(2)])
    real_q=np.column_stack([np.interp(ts,paired[:,0],paired[:,1+i]) for i in range(4)])
    metrics={}
    for b,name in enumerate(('recorded_torque','recorded_target_feedback')):
        metrics[name]={'encoder_solved_leg_length_rmse_m':np.sqrt(np.mean((sensor_lengths[:,b]-real)**2,axis=0)).tolist(),'final_encoder_solved_leg_length_m':sensor_lengths[-1,b].tolist(),'physical_endpoint_length_rmse_m_secondary':np.sqrt(np.mean((lengths[:,b]-real)**2,axis=0)).tolist(),'motor_position_rmse_rad':np.sqrt(np.mean((sensor_q[:,b]-real_q)**2,axis=0)).tolist(),'max_loop_pin_residual_m':closure_max[b].tolist(),'max_abs_model_torque_Nm':np.max(np.abs(submitted[:,b]),axis=0).tolist()}
        metrics[name].update(max_spring_loop_residual_m=np.max(spring_residual[:,b],axis=0).tolist(),rms_loop_pin_residual_m=np.sqrt(np.mean(pin_residual[:,b]**2,axis=0)).tolist())
        if solver_iterations is not None:
            metrics[name]['solver_iterations']={'max':int(solver_iterations[:,b].max()),'mean':float(solver_iterations[:,b].mean()),'at_iteration_cap':int(np.count_nonzero(solver_iterations[:,b]>=model.opt.iterations))}
        if knee_force is not None:
            metrics[name]['knee_limits']={'joint_names':['lf1','rf1'],'ranges_rad':model.jnt_range[knee_ids].tolist(),
                'constraint_present_time_s':(np.sum(knee_present[:,b],axis=0)*model.opt.timestep).tolist(),
                'nonzero_force_time_s':(np.sum(knee_force[:,b]>1e-6,axis=0)*model.opt.timestep).tolist(),
                'maximum_force_nm':np.max(knee_force[:,b],axis=0).tolist()}
    summary=dict(scope=f'GPU diagnostic under recorded mapping, fixed base and {args.spring_model} springs; '+('effective resistance candidate applied; no online fit or hardware change' if response_parameters else 'no parameter fit or hardware change'),
        spring_model=spring_model,spring_stroke_m=.090 if args.spring_model!='constant' else None,CAD_reference_compression_m=.010,
        spring_damping_ns_m=50. if args.spring_model=='linear-damped' else 0.,
        training_spring_source_sha256=hashlib.sha256(Path(training_spring.__file__).read_bytes()).hexdigest() if args.spring_model=='linear-damped' else None,
        spring_force_min_n=spring_forces.min(axis=0).tolist(),spring_force_max_n=spring_forces.max(axis=0).tolist(),
        spring_compression_min_m=spring_compression.min(axis=0).tolist(),spring_compression_max_m=spring_compression.max(axis=0).tolist(),
        session=str(args.session),stream_sha256=meta['stream_sha256'],original_whole_stream_complete=input_report['complete_observed_span'],replay_window_has_no_output_sequence_gaps=True,
        window_start_board_s=start,first_dial_board_s=origin,
        source_model=str(asset),source_model_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        recorded_source_model=str(recorded_asset),recorded_source_model_sha256=meta['model']['sha256'],calibrated_joint_range_changes=calibrated_changes,
        response_parameters=response_parameters,response_parameters_sha256=hashlib.sha256(args.response_parameters.read_bytes()).hexdigest() if args.response_parameters else None,
        elastic_preload=elastic_preload,minimum_elastic_dissipation=minimum_loss,
        disabled_joint_limits=['lf1','rf1'] if args.disable_knee_limits else [],
        extended_joint_limits_rad={'lf1':[-.77,.15],'rf1':[-.15,.77]} if args.extend_knee_limits else {},
        constraint_check=args.constraint_check,leg_boundary_mode='return_control' if soft_boundary else 'legacy_lockout',
        solver_options={n:getattr(model.opt,n) for n in ('iterations','tolerance','ls_iterations','ls_tolerance')},
        diagnostic_model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controller_reconstruction_max_error_Nm=float(np.max(abs(controller_error))),leg_solver_max_difference_m=float(np.max(abs(np.asarray(geom_error)))),
        jacobian_finite_difference_max_error=float(np.max(abs(np.asarray(gradient_error)))),initial_closure_m=initial_residual,initial_joint_limit_excess_rad=limit_excess,
        dt_s=.001,duration_s=end-start,physics_backend='mujoco_warp_gpu',worlds=2,steps_per_world=done,total_environment_steps=done*2,model_forwards=0,optimizer_updates=0,
        reused_replay=str(args.reuse_replay) if args.reuse_replay else None,new_environment_steps=0 if args.reuse_replay else done*2,
        feedback_branch_blocked=blocked,elapsed_replay_s=time.monotonic()-began,metrics=metrics,
        limitations=['Numerically matched registration is not independent physical calibration','Fixed base and selected spring law are diagnostic assumptions, not a measured force curve','Torque replay uses submitted requests, not measured shaft torque','Target replay uses the recorded target sequence; it does not recreate the dial target integrator',
            'Knee limits disabled only for diagnosis; this does not establish the physical travel range' if args.disable_knee_limits else
            'Knee limits extended only for diagnosis based on the recorded pose envelope; physical travel remains unverified' if args.extend_knee_limits else
            'Calibrated knee ranges and explicit effective resistance candidate used; geometry/zero/sign retained' if response_parameters else
            'Explicit calibrated knee ranges used; geometry and all other parameters unchanged' if args.calibrated_model else
            'Initial joint limits are retained; initial measured pose exceeds limits' if limit_excess else 'Initial joint limits are retained; initial pose within limits',
            'Single trial with no independent holdout; no identified physical parameters'])
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for side,ax in enumerate(axes):
        ax.plot(ts-origin,real[:,side]*1000,label='Recorded robot');ax.plot(ts-origin,targets[:,side]*1000,'--',label='Recorded target')
        ax.plot(ts-origin,sensor_lengths[:,0,side]*1000,label='GPU: recorded torque');ax.plot(ts-origin,sensor_lengths[:,1,side]*1000,label='GPU: target feedback')
        ax.set_ylabel(('Left' if side==0 else 'Right')+' leg (mm)');ax.grid(alpha=.3);ax.legend()
    axes[-1].set_xlabel('Time from first nonzero dial input (s)');fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for i,ax in enumerate(axes):
        ax.plot(ts-origin,np.degrees(real_q[:,i]),label='Recorded robot')
        ax.plot(ts-origin,np.degrees(sensor_q[:,0,i]),label='GPU: recorded torque');ax.plot(ts-origin,np.degrees(sensor_q[:,1,i]),label='GPU: target feedback')
        ax.set_ylabel(mirror.MOTORS[i]+' (deg)');ax.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('Time from first nonzero dial input (s)');fig.tight_layout();fig.savefig(args.output/'joint-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
