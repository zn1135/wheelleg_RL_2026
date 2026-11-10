"""Dual-branch fixed-base GPU replay with frozen firmware control and candidate plant loads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

PROGRAM_SOURCE = Path(__file__).read_bytes()

import mujoco
import numpy as np

from scripts.agent.real2sim.control_leg_pose_gpu import mirror,gas_spring_force,leg_pose
from scripts.agent.real2sim.nonlinear_spring import spring_forces as axial_spring_forces,total_length_load


def mit_torque_quantize(values):
    """Fixed ±54Nm/12bit encoder and decoder, with dm.c's float32 operation order."""
    value=np.clip(np.asarray(values,dtype=np.float32),np.float32(-54),np.float32(54))
    raw=((value+np.float32(54))/np.float32(108)*np.float32(4095)).astype(np.uint16)
    return raw.astype(np.float32)*np.float32(108)/np.float32(4095)-np.float32(54)


def mit_velocity_quantize(values):
    """Fixed ±45rad/s/12bit floor encoding; the zero cell decodes to-0.01099014."""
    value=np.clip(np.asarray(values,dtype=np.float32),np.float32(-45),np.float32(45))
    raw=((value+np.float32(45))/np.float32(90)*np.float32(4095)).astype(np.uint16)
    return raw.astype(np.float32)*np.float32(90)/np.float32(4095)-np.float32(45)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--parameters',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=35.)
    parser.add_argument('--feedback-io',choices=('ideal','measured'),default='measured')
    parser.add_argument('--save-traces',action='store_true')
    parser.add_argument('--save-signals',action='store_true',help='save only right-leg length/angle for search diagnostics')
    parser.add_argument('--driver-torque',type=Path,help='diagnostic: first branch uses paired driver torque estimates, not drive-entry requests')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    identity=json.loads((args.inputs/'report.json').read_text())
    assert hashlib.sha256((args.inputs/'inputs.npz').read_bytes()).hexdigest()==identity['input_npz_sha256']
    for path,digest in identity['protected_sources'].items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    with np.load(args.inputs/'inputs.npz') as source: a={k:source[k] for k in source.files}
    driver_torque=None;driver_identity=None
    if args.driver_torque:
        driver_source=args.driver_torque.read_bytes()
        with np.load(args.driver_torque) as source:
            assert np.array_equal(source['time_s'],a['time_s'])
            driver_torque=source['feedback_model_nm'].copy()
        assert driver_torque.shape==a['recorded_tau'].shape and np.isfinite(driver_torque).all()
        driver_identity=dict(path=str(args.driver_torque.resolve()),sha256=hashlib.sha256(driver_source).hexdigest())
    baseline_source=(args.inputs/'baseline-parameters.json').read_bytes()
    baseline=json.loads(baseline_source)
    parameter_source=args.parameters.read_bytes()
    candidates=json.loads(parameter_source)
    assert isinstance(candidates,list) and candidates
    count=len(candidates);nworld=count*2;dt=.002
    steps=min(int(round(args.seconds/dt)),len(a['time_s'])-1)
    assert steps>0 and abs(steps*dt-args.seconds)<1e-8
    source_model_path=args.inputs/'model.xml'
    input_model_sha256=hashlib.sha256(source_model_path.read_bytes()).hexdigest()
    routes=[p.get('seal_friction_route','slider') for p in candidates]
    assert set(routes)<= {'slider','spatial','dahl'}
    spatial_mask=np.tile(np.array([route=='spatial' for route in routes]),2)
    bristle_mask=np.tile(np.array([route=='dahl' for route in routes]),2)
    if spatial_mask.any():
        tree=ET.fromstring(source_model_path.read_text())
        tendons=tree.find('tendon')
        if tendons is None:tendons=ET.SubElement(tree,'tendon')
        span=ET.SubElement(tendons,'spatial',name='right_seal_span',frictionloss='0.000001',damping='0',
                           solreffriction='0.002 1',solimpfriction='0.999 0.9999 0.001 0.5 2')
        for site in ('right_gas_spring_upper','right_gas_spring_lower'):ET.SubElement(span,'site',site=site)
        source_model_path=args.output/'span-input.xml'
        source_model_path.write_text(ET.tostring(tree,encoding='unicode'))
    model=mujoco.MjModel.from_xml_path(str(source_model_path))
    data=mujoco.MjData(model)
    mq,mv=a['mq'].astype(int),a['mv'].astype(int)
    motors=[model.actuator(name+'_motor').id for name in mirror.MOTORS]
    springs=[model.actuator(side+'_gas_spring_motor').id for side in ('left','right')]
    curve_enabled=any('spring_curve_force_n' in p for p in candidates)
    total_load_enabled=any('right_total_length_load_nm' in p for p in candidates)
    no_baseline=np.tile(np.array([p.get('remove_baseline_right_load',False) for p in candidates]),2)
    if curve_enabled:
        ceiling=max(p.get('spring_force_limit_n',600.) for p in candidates if 'spring_curve_force_n' in p)
        model.actuator_ctrlrange[springs[1]]=[0,ceiling]
        model.actuator_forcerange[springs[1]]=[0,ceiling]
    sq=[model.joint(side+'_spring2_joint').qposadr[0] for side in ('left','right')]
    sv=[model.joint(side+'_spring2_joint').dofadr[0] for side in ('left','right')]
    span_id=model.tendon('right_seal_span').id if spatial_mask.any() else None
    spring_axis_z=model.body('right_spring2').pos[2]
    spring_lateral_y=model.body('right_spring2').pos[1]
    tendon_friction_table=np.tile(model.tendon_frictionloss,(nworld,1)).astype(np.float32)
    knee=model.joint('rf1');kq=knee.qposadr[0]
    static=np.array([p['static_nm'] for p in candidates],dtype=float)
    kinetic=np.array([p['kinetic_nm'] for p in candidates],dtype=float)
    vs=np.array([p['velocity_scale_rad_s'] for p in candidates],dtype=float)
    damping=np.array([p['damping_nms_rad'] for p in candidates],dtype=float)
    bias=np.array([p.get('bias_delta_nm',[0,0]) for p in candidates],dtype=float)
    length_k=np.array([p.get('length_stiffness_delta_nm_rad',0.) for p in candidates],dtype=float)
    cross_k=np.array([p.get('cross_stiffness_delta_nm_rad',0.) for p in candidates],dtype=float)
    length_k2=np.array([p.get('length_quadratic_delta_nm_rad2',0.) for p in candidates],dtype=float)
    recovery=np.array([np.broadcast_to(p.get('friction_recovery_s',0.),(2,)) for p in candidates],dtype=float)
    weakening=np.array([np.broadcast_to(p.get('friction_weakening_s',0.),(2,)) for p in candidates],dtype=float)
    elastic_k=np.array([p.get('elastic_stiffness_nm_rad',0.) for p in candidates],dtype=float)
    preload=np.array([p.get('elastic_preload_fraction',0.) for p in candidates],dtype=float)
    assert static.shape==kinetic.shape==vs.shape==damping.shape==bias.shape==(count,2)
    assert np.isfinite(np.r_[static.ravel(),kinetic.ravel(),vs.ravel(),damping.ravel(),bias.ravel(),length_k,cross_k,length_k2]).all()
    assert static.min()>0 and kinetic.min()>=0 and np.all(kinetic<=static) and vs.min()>0 and damping.min()>=0
    static,kinetic,vs,damping,bias=[np.tile(x,(2,1)) for x in (static,kinetic,vs,damping,bias)]
    length_k=np.tile(length_k,2)
    cross_k=np.tile(cross_k,2)
    length_k2=np.tile(length_k2,2)
    recovery,weakening=np.tile(recovery,(2,1)),np.tile(weakening,(2,1))
    elastic_k,preload=np.tile(elastic_k,2),np.tile(preload,2)
    elastic_mask=elastic_k>0
    assert elastic_k.min()>=0 and np.all((preload>=0)&(preload<=1))
    assert np.isfinite(recovery).all() and np.isfinite(weakening).all() and min(recovery.min(),weakening.min())>=0
    friction_state=np.ones((nworld,2))
    seal_static=np.tile(np.array([p.get('seal_static_n',0.) for p in candidates],dtype=float),2)
    seal_kinetic=np.tile(np.array([p.get('seal_kinetic_n',0.) for p in candidates],dtype=float),2)
    seal_vs=np.tile(np.array([p.get('seal_velocity_scale_m_s',.0005) for p in candidates],dtype=float),2)
    seal_recovery=np.tile(np.array([p.get('seal_recovery_s',3.) for p in candidates],dtype=float),2)
    seal_force_bias=np.tile(np.array([p.get('seal_force_bias_n',0.) for p in candidates],dtype=float),2)
    assert np.isfinite(seal_force_bias).all()
    assert np.all((seal_static>=seal_kinetic)&(seal_kinetic>=0)&(seal_vs>0)&(seal_recovery>0))
    seal_state=np.ones(nworld)
    seal_stiffness=np.tile(np.array([p.get('seal_bristle_stiffness_n_m',150000.) for p in candidates],dtype=float),2)
    seal_initial=np.tile(np.array([p.get('seal_initial_force_n',0.) for p in candidates],dtype=float),2)
    bristle_aging=np.tile(np.array([p.get('seal_bristle_aging',False) for p in candidates],dtype=np.int32),2)
    seal_weakening=np.tile(np.array([p.get('seal_weakening_s',.002) for p in candidates],dtype=float),2)
    assert seal_stiffness.min()>0 and np.all(abs(seal_initial)<=seal_static+1e-9)
    friction_table=np.tile(model.dof_frictionloss,(nworld,1)).astype(np.float32)
    damping_table=np.tile(model.dof_damping,(nworld,1)).astype(np.float32)
    damping_table[:,mv[2:]]=damping
    seal_damping=np.tile(np.array([p.get('seal_damping_ns_m',0.) for p in candidates],dtype=float),2)
    assert np.isfinite(seal_damping).all() and seal_damping.min()>=0
    damping_table[:,sv[1]]+=seal_damping
    inertia_scale=np.array([np.broadcast_to(p.get('armature_scale',1.),(2,)) for p in candidates],dtype=float)
    assert np.isfinite(inertia_scale).all() and inertia_scale.min()>0
    inertia_scale=np.tile(inertia_scale,(2,1))
    armature_table=np.tile(model.dof_armature,(nworld,1)).astype(np.float32)
    armature_table[:,mv[2:]]*=inertia_scale
    derived={}
    if np.any(inertia_scale!=1):
        # Inertia changes also require new reference inverse weights for the constraints.
        for field in ('dof_invweight0','body_invweight0','actuator_acc0'):
            derived[field]=[]
        if model.ntendon:derived['tendon_invweight0']=[]
        for case in range(count):
            candidate_model=mujoco.MjModel.from_xml_path(str(source_model_path))
            candidate_model.dof_armature[:]=armature_table[case]
            mujoco.mj_setConst(candidate_model,mujoco.MjData(candidate_model))
            for field in derived:derived[field].append(getattr(candidate_model,field).copy())
        derived={name:np.tile(np.array(rows,dtype=np.float32),(2,)+(1,)*(np.array(rows).ndim-1)) for name,rows in derived.items()}
        # Restore the template used by mj_saveLastXML after loading candidate models.
        model=mujoco.MjModel.from_xml_path(str(source_model_path))
        model.geom_contype[:]=model.geom_conaffinity[:]=0
    model.dof_frictionloss[mv[2:]]=1e-6  # Compile native friction constraints for every candidate.
    if np.any((seal_static>0)&~spatial_mask&~bristle_mask):
        model.dof_frictionloss[sv[1]]=1e-6
        model.dof_solref[sv[1]]=model.dof_solref[mv[2]]
        model.dof_solimp[sv[1]]=model.dof_solimp[mv[2]]
    model.geom_contype[:]=model.geom_conaffinity[:]=0
    mujoco.mj_saveLastXML(str(args.output/'model.xml'),model)
    # All worlds use the same first measured pose; no future measured state is assigned.
    initial=np.tile(a['initial_q'],(nworld,1)).astype(np.float32)
    initial_v=np.tile(a['initial_v'],(nworld,1)).astype(np.float32)
    data.qpos[:]=a['initial_q'];data.qvel[:]=a['initial_v']
    qref=np.array(baseline['reference_q_model_rad']);load_k=np.array(baseline['stiffness_nm_rad'])
    baseline_bias=np.array(baseline['bias_nm']);direction=np.array([-1.,1.])/np.sqrt(2)
    angle_direction=np.array([1.,1.])/np.sqrt(2)
    elastic_force=np.zeros((nworld,2))
    automatic_seal=np.tile(np.array([p.get('seal_friction_route')=='dahl' and 'seal_initial_force_n' not in p for p in candidates]),2)
    assert not np.any(elastic_mask&bristle_mask)
    if elastic_mask.any() or automatic_seal.any():
        from scripts.agent.real2sim.elastic_joint_resistance import initial_preload
        model.opt.gravity[:]=a['gravity'][0]
        initial_drive=a['initial_request'].copy()
        if args.feedback_io=='measured':initial_drive[2:]=mit_torque_quantize(initial_drive[2:])
        initial_residual,initial_info=initial_preload(model,data,initial_drive,return_tangent=True)
        initial_residual-=baseline_bias+(a['initial_q'][mq]-qref)@load_k.T
        x0=(a['initial_q'][mq[2:]]-qref[2:])@direction;y0=(a['initial_q'][mq[2:]]-qref[2:])@angle_direction
        initial_extra=bias+(length_k*x0+length_k2*x0*x0)[:,None]*direction+cross_k[:,None]*(x0*angle_direction+y0*direction)
        elastic_force=np.clip((initial_residual[2:]-initial_extra)*preload[:,None],-static,static)
        if automatic_seal.any():
            jac=np.array(initial_info['tangent'])[sv[1],2:]
            residual_right=np.tile(initial_residual[2:],(nworld,1))-initial_extra
            residual_right[no_baseline]+=(baseline_bias+(a['initial_q'][mq]-qref)@load_k.T)[2:]
            if curve_enabled:
                f0=axial_spring_forces(initial[:,sq],np.zeros((nworld,2)),candidates)[:,1]
                old_f0=gas_spring_force(a['initial_q'][sq],np.zeros(2))[1]
                residual_right+=(f0-old_f0)[:,None]*jac
            inferred=residual_right@jac/(jac@jac)+seal_force_bias
            seal_initial[automatic_seal]=np.clip(inferred[automatic_seal],-seal_static[automatic_seal],seal_static[automatic_seal])
    elastic_initial=elastic_force.copy()
    previous_q=initial[:,mq[2:]].copy()
    integral=np.zeros((count,2),dtype=np.float32)
    takeover=a['initial_request'][2:]+2*a['initial_sensor_v'][2:]-a['feedforward'][0,2:]
    history_q=np.tile(initial[:,mq][None],(4,1,1));history_v=np.zeros_like(history_q)
    block=np.zeros(nworld,dtype=bool);stop=[None]*nworld
    sum_q=np.zeros((nworld,2));sum_l=np.zeros(nworld);sum_angle=np.zeros(nworld)
    max_l=np.zeros(nworld);max_angle=np.zeros(nworld);peak_tau=np.zeros((nworld,2));peak_v=np.zeros((nworld,2))
    signed_l=np.zeros(nworld);signed_angle=np.zeros(nworld);signal_lengths=[];signal_angles=[]
    traces=[];velocities=[];requests=[];frictions=[];integrals=[];loads=[];spring_forces=[];elastic_forces=[];seal_capacities=[];seal_biases=[]
    native_friction_excess=np.zeros(model.nv)
    native_tendon_excess=np.zeros(model.ntendon)
    started=time.monotonic()
    import warp as wp
    import mujoco_warp as mw
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    wp.init()
    @wp.kernel
    def bristle_step(q:wp.array2d(dtype=wp.float32),v:wp.array2d(dtype=wp.float32),load:wp.array2d(dtype=wp.float32),
                     enabled:wp.array(dtype=wp.int32),force:wp.array(dtype=wp.float32),previous:wp.array(dtype=wp.float32),
                     fs:wp.array(dtype=wp.float32),fk:wp.array(dtype=wp.float32),scale:wp.array(dtype=wp.float32),
                     stiffness:wp.array(dtype=wp.float32),bias_force:wp.array(dtype=wp.float32),aging:wp.array(dtype=wp.int32),
                     age_state:wp.array(dtype=wp.float32),recovery:wp.array(dtype=wp.float32),weakening:wp.array(dtype=wp.float32),
                     counter:wp.array(dtype=wp.int32),audit:wp.array3d(dtype=wp.float32),save:int,qi:int,vi:int,physics_dt:float):
        world=wp.tid()
        if enabled[world]!=0:
            delta=q[world,qi]-previous[world]
            speed=v[world,vi]/scale[world]
            desired_age=wp.exp(-speed*speed)
            old_age=age_state[world]
            rate=recovery[world]
            if desired_age<old_age:rate=weakening[world]
            new_age=desired_age+(old_age-desired_age)*wp.exp(-physics_dt/rate)
            age_state[world]=new_age
            selected_age=desired_age
            if aging[world]!=0:selected_age=new_age
            capacity=fk[world]+(fs[world]-fk[world])*selected_age
            capacity=wp.max(capacity,1.e-6)
            target=wp.sign(delta)*capacity
            before=force[world]
            updated=target+(before-target)*wp.exp(-stiffness[world]*wp.abs(delta)/capacity)
            force[world]=updated;previous[world]=q[world,qi]
            load[world,vi]=bias_force[world]-updated
            if save!=0:
                row=counter[world]
                audit[row,world,0]=before;audit[row,world,1]=delta;audit[row,world,2]=capacity
                audit[row,world,3]=updated;audit[row,world,4]=q[world,qi]
                audit[row,world,5]=v[world,vi];audit[row,world,6]=load[world,vi]
                audit[row,world,7]=old_age;audit[row,world,8]=new_age
            counter[world]+=1
    with wp.ScopedDevice('cuda:0'):
        batch_sizes={'dof_frictionloss':nworld,'dof_damping':nworld,'dof_armature':nworld,**{name:nworld for name in derived}}
        if spatial_mask.any():batch_sizes['tendon_frictionloss']=nworld
        gm=mw.put_model(model,batch_sizes=batch_sizes)
        gm.dof_damping.assign(damping_table)
        assert np.array_equal(gm.dof_damping.numpy(),damping_table)
        gm.dof_armature.assign(armature_table)
        for name,values in derived.items():
            getattr(gm,name).assign(values)
            assert np.array_equal(getattr(gm,name).numpy(),values),name
        assert np.array_equal(gm.dof_armature.numpy(),armature_table)
        gm.opt.gravity=wp.array(np.tile(a['gravity'][0],(nworld,1)),dtype=wp.vec3)
        gd=mw.put_data(model,data,nworld=nworld,nconmax=128,njmax=256)
        gd.qpos.assign(initial);gd.qvel.assign(initial_v)
        storage=ordered.buffers(gm,gd)
        bristle_force=wp.array(seal_initial.astype(np.float32));bristle_previous=wp.array(initial[:,sq[1]])
        bristle_counter=wp.zeros(nworld,dtype=wp.int32)
        bristle_age_state=wp.ones(nworld,dtype=wp.float32)
        bristle_audit=wp.zeros((steps*10,nworld,9) if args.save_traces and bristle_mask.any() else (1,1,1),dtype=wp.float32)
        bristle_arguments=[gd.qpos,gd.qvel,gd.qfrc_applied,wp.array(bristle_mask.astype(np.int32)),bristle_force,bristle_previous,
                           wp.array(seal_static.astype(np.float32)),wp.array(seal_kinetic.astype(np.float32)),wp.array(seal_vs.astype(np.float32)),
                           wp.array(seal_stiffness.astype(np.float32)),wp.array(seal_force_bias.astype(np.float32)),wp.array(bristle_aging),
                           bristle_age_state,wp.array(seal_recovery.astype(np.float32)),wp.array(seal_weakening.astype(np.float32)),
                           bristle_counter,bristle_audit,int(args.save_traces),int(sq[1]),int(sv[1]),float(model.opt.timestep)]
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                if bristle_mask.any():wp.launch(bristle_step,dim=nworld,inputs=bristle_arguments)
                mw.step1(gm,gd);ordered.apply(gd,storage);mw.step2(gm,gd)
        assert not np.any(gd.time.numpy())
        for step in range(steps+1):
            q,v=gd.qpos.numpy(),gd.qvel.numpy()
            if args.save_traces and step>0 and step%10==0:
                kinds,identities,forces=gd.efc.type.numpy(),gd.efc.id.numpy(),gd.efc.force.numpy()
                constraint_counts=gd.nefc.numpy()
                for world in range(nworld):
                    rows=np.flatnonzero(kinds[world,:constraint_counts[world]]==int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
                    for row in rows:
                        dof=identities[world,row]
                        native_friction_excess[dof]=max(native_friction_excess[dof],abs(forces[world,row])-friction_table[world,dof])
                    rows=np.flatnonzero(kinds[world,:constraint_counts[world]]==int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON))
                    for row in rows:
                        tendon=identities[world,row]
                        native_tendon_excess[tendon]=max(native_tendon_excess[tendon],abs(forces[world,row])-tendon_friction_table[world,tendon])
            length,angle,_=leg_pose(q[:,mq]*a['sign']+a['zero'])
            wrong=(~np.isfinite(q).all(1)|~np.isfinite(v).all(1)|(length[:,1]<.14)|(length[:,1]>.34)|(q[:,kq]<knee.range[0])|(q[:,kq]>knee.range[1]))
            for world in np.flatnonzero(wrong&~block):stop[world]=step*dt
            block|=wrong
            history_q[step%4]=q[:,mq];history_v[step%4]=v[:,mv]
            if args.feedback_io=='measured':
                feedback_q=np.empty((count,2));feedback_v=np.empty((count,2))
                for j in range(2):
                    lag=float(a['delay_s'][step,j+2])/dt
                    older=min(int(lag),2);weight=lag-older
                    feedback_q[:,j]=(1-weight)*history_q[(step-older)%4,count:,j+2]+weight*history_q[(step-older-1)%4,count:,j+2]
                    feedback_v[:,j]=(1-weight)*history_v[(step-older)%4,count:,j+2]+weight*history_v[(step-older-1)%4,count:,j+2]
                phase=a['initial_sensor_q'][2:]
                feedback_q=phase+np.rint((feedback_q-phase)/a['position_quantum'][2:])*a['position_quantum'][2:]
                feedback_v=mit_velocity_quantize(feedback_v)
            else:feedback_q=q[count:,mq[2:]];feedback_v=v[count:,mv[2:]]
            error=a['target_q'][step,2:]-feedback_q
            integral_used=integral.copy()
            t=np.clip(step*dt/5,0,1);takeover_weight=1-t**3*(10-15*t+6*t*t)
            raw=a['feedforward'][step,2:]+60*error+2*(a['target_v'][step,2:]-feedback_v)+integral+takeover_weight*takeover
            tau=np.zeros((nworld,4),dtype=np.float32)
            tau[:count]=a['recorded_tau'][step]
            if driver_torque is not None:tau[:count,2:]=driver_torque[step,2:]
            tau[count:,2:]=np.clip(raw,-40,40)
            # MIT command encoder is fixed, not a fit variable; model/driver signs multiply to+1.
            if args.feedback_io=='measured':
                tau[count:,2:]=mit_torque_quantize(tau[count:,2:])
                if driver_torque is None:tau[:count,2:]=mit_torque_quantize(tau[:count,2:])
            tau[block]=0
            integral=np.clip(integral+.04*error*((abs(raw)<40)|(raw*error<0)),-8,8).astype(np.float32)
            qerror=q[:,mq[2:]]-a['real_q'][step,2:]
            lerror=length[:,1]-a['real_length'][step,1]
            aerror=angle[:,1]-a['real_angle'][step,1]
            sum_q+=qerror*qerror;sum_l+=lerror*lerror;sum_angle+=aerror*aerror
            signed_l+=lerror;signed_angle+=aerror
            if args.save_signals:signal_lengths.append(length[:,1].copy());signal_angles.append(angle[:,1].copy())
            max_l=np.maximum(max_l,abs(lerror));max_angle=np.maximum(max_angle,abs(aerror))
            peak_tau=np.maximum(peak_tau,abs(tau[:,2:]));peak_v=np.maximum(peak_v,abs(v[:,mv[2:]]))
            desired_state=np.exp(-(v[:,mv[2:]]/vs)**2)
            state_dt=np.where(desired_state<friction_state,weakening,recovery)
            # Capacity remains nonnegative; native dry friction still opposes simulated motion.
            friction_state=desired_state+(friction_state-desired_state)*np.exp(-dt/np.maximum(state_dt,1e-12))
            capacity=kinetic+(static-kinetic)*friction_state
            delta=q[:,mq[2:]]-previous_q
            target=np.sign(delta)*capacity
            elastic_force=target+(elastic_force-target)*np.exp(-elastic_k[:,None]*abs(delta)/np.maximum(capacity,1e-8))
            elastic_force[~elastic_mask]=0
            previous_q=q[:,mq[2:]].copy()
            friction_table[:,mv[2:]]=np.where(elastic_mask[:,None],1e-6,capacity)
            seal_desired=np.exp(-(v[:,sv[1]]/seal_vs)**2)
            seal_dt=np.where(seal_desired<seal_state,.002,seal_recovery)
            seal_state=seal_desired+(seal_state-seal_desired)*np.exp(-dt/seal_dt)
            seal_capacity=seal_kinetic+(seal_static-seal_kinetic)*seal_state
            seal_capacity=np.where(bristle_mask,seal_kinetic+(seal_static-seal_kinetic)*seal_desired,seal_capacity)
            friction_table[:,sv[1]]=np.where(spatial_mask|bristle_mask,0.,seal_capacity)
            if span_id is not None:
                axial=q[:,sq[1]].astype(np.float64)+spring_axis_z
                # The eyelets have a25mm lateral offset; preserve axial virtual work.
                force_ratio=np.hypot(axial,spring_lateral_y)/np.maximum(abs(axial),1e-8)
                tendon_friction_table[:,span_id]=np.where(spatial_mask,seal_capacity*force_ratio,0.)
            if args.save_traces:
                traces.append(q.copy());velocities.append(v.copy());requests.append(tau.copy());frictions.append(capacity.copy());integrals.append(integral_used);elastic_forces.append(elastic_force.copy());seal_capacities.append(seal_capacity.copy())
            if step==steps:break
            gm.dof_frictionloss.assign(friction_table)
            if span_id is not None:gm.tendon_frictionloss.assign(tendon_friction_table)
            gm.opt.gravity.assign(np.tile(a['gravity'][step],(nworld,1)).astype(np.float32))
            control=np.zeros((nworld,model.nu),dtype=np.float32)
            control[:,motors[:4]]=tau;control[:,springs]=axial_spring_forces(q[:,sq],v[:,sv],candidates) if curve_enabled or total_load_enabled else gas_spring_force(q[:,sq],v[:,sv])
            load=np.zeros((nworld,model.nv),dtype=np.float32)
            load[:,mv]=-(baseline_bias+(q[:,mq]-qref)@load_k.T)
            load[np.ix_(no_baseline,mv[2:])]=0.
            coordinate=(q[:,mq[2:]]-qref[2:])@direction
            angle_coordinate=(q[:,mq[2:]]-qref[2:])@angle_direction
            load[:,mv[2:]]-=bias+length_k[:,None]*coordinate[:,None]*direction
            load[:,mv[2:]]-=length_k2[:,None]*coordinate[:,None]**2*direction
            load[:,mv[2:]]-=cross_k[:,None]*(coordinate[:,None]*angle_direction+angle_coordinate[:,None]*direction)
            load[:,mv[2:]]-=elastic_force
            if total_load_enabled:load[:,mv[2:]]+=total_length_load(length[:,1],candidates)
            load[:,sv[1]]+=seal_force_bias
            gd.ctrl.assign(control);gd.qfrc_applied.assign(load)
            if args.save_traces:
                loads.append(load[:,mv].copy());spring_forces.append(control[:,springs].copy());seal_biases.append(load[:,sv[1]].copy())
            wp.capture_launch(capture.graph)
            if step%2500==0:print(json.dumps(dict(simulation_s=step*dt,wall_s=time.monotonic()-started,worlds=nworld,blocked=int(block.sum()))),flush=True)
        if args.save_traces and bristle_mask.any():
            np.savez_compressed(args.output/'bristle-audit.npz',audit=bristle_audit.numpy(),counts=bristle_counter.numpy(),
                                stiffness_n_m=seal_stiffness,initial_force_n=seal_initial,enabled=bristle_mask,
                                aging_enabled=bristle_aging,recovery_s=seal_recovery,weakening_s=seal_weakening,physics_dt_s=float(model.opt.timestep))
    rmse_q=np.degrees(np.sqrt(sum_q/(steps+1)))
    rmse_l=np.sqrt(sum_l/(steps+1))*1000;rmse_angle=np.degrees(np.sqrt(sum_angle/(steps+1)))
    scores=.5*np.sqrt(np.mean(rmse_q**2,axis=1))+.25*rmse_l/3+.25*rmse_angle
    scores=np.where(block,1000+scores,scores)
    reports=[]
    for case,p in enumerate(candidates):
        branches={}
        for branch,world in (('recorded_torque',case),('recorded_target',case+count)):
            branches[branch]=dict(joint_rmse_deg=rmse_q[world].tolist(),length_rmse_mm=float(rmse_l[world]),angle_rmse_deg=float(rmse_angle[world]),
                                 mean_length_error_mm=float(signed_l[world]/(steps+1)*1000),mean_angle_error_deg=float(np.degrees(signed_angle[world]/(steps+1))),
                                 final_length_mm=float(length[world,1]*1000),final_angle_deg=float(np.degrees(angle[world,1])),
                                 length_max_error_mm=float(max_l[world]*1000),angle_max_error_deg=float(np.degrees(max_angle[world])),
                                 score=float(scores[world]),guard_stop_s=stop[world],peak_request_nm=peak_tau[world].tolist(),peak_speed_rad_s=peak_v[world].tolist())
        reports.append(dict(parameters=p,score=float((scores[case]+scores[case+count])/2),branches=branches))
    report=dict(physics_backend='mujoco_warp_gpu',physics_dt_s=.0002,control_dt_s=.002,feedback_io=args.feedback_io,torque_encoder='dm_float32',sensor_velocity_quantizer='mit_floor',seconds=steps*dt,
                input_identity=identity['input_npz_sha256'],candidate_sha256=hashlib.sha256(parameter_source).hexdigest(),
                torque_input_source='driver_estimate' if driver_torque is not None else 'drive_entry_request',driver_torque_identity=driver_identity,
                input_model_sha256=input_model_sha256,model_sha256=hashlib.sha256((args.output/'model.xml').read_bytes()).hexdigest(),
                baseline_parameter_sha256=hashlib.sha256(baseline_source).hexdigest(),program_sha256=hashlib.sha256(PROGRAM_SOURCE).hexdigest(),
                candidate_count=count,trials=reports,right_motor_damping_nms_rad=damping_table[:,mv[2:]].tolist(),right_seal_damping_ns_m=damping_table[:,sv[1]].tolist(),right_armature_kg_m2=armature_table[:,mv[2:]].tolist(),native_friction_bound_excess=native_friction_excess.tolist() if args.save_traces else None,elastic_initial_force_nm=elastic_initial.tolist(),new_gpu_environment_steps=steps*10*nworld,policy_training_updates=0,hardware_commands=False,
                native_tendon_friction_bound_excess=native_tendon_excess.tolist() if args.save_traces else None,initial_seal_bristle_force_n=seal_initial.tolist(),
                nonlinear_axial_force_curve=curve_enabled,removed_right_baseline_load=no_baseline.tolist(),
                limitations=['Fitting this trajectory is not independent validation','No real leg state injected after initial conditions',
                             'Frozen controller and first-sample takeover; diagnostic plant parameters only, no shared-source update',
                             'Position quantizer uses a fixed first-sample phase; velocity uses fixed MIT floor encoding; continuous encoder phase and motor internal dynamics were not measured',
                             'Torque is the protocol drive-entry request, not independently measured shaft torque',
                             'Fixed body with recorded gravity direction; fixture translation and rotational inertial disturbances are not imposed'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'parameters.json').write_bytes(parameter_source)
    (args.output/Path(__file__).name).write_bytes(PROGRAM_SOURCE)
    if args.save_signals:np.savez_compressed(args.output/'signals.npz',time_s=a['time_s'][:steps+1],length_m=np.array(signal_lengths),angle_rad=np.array(signal_angles))
    if args.save_traces:np.savez_compressed(args.output/'traces.npz',time_s=a['time_s'][:steps+1],qpos=np.asarray(traces),qvel=np.asarray(velocities),request_model_nm=np.asarray(requests),friction_capacity_nm=np.asarray(frictions),integral_nm=np.asarray(integrals),applied_load_nm=np.asarray(loads),spring_force_n=np.asarray(spring_forces),elastic_force_nm=np.asarray(elastic_forces),seal_friction_capacity_n=np.asarray(seal_capacities),seal_force_bias_n=np.asarray(seal_biases))
    print(json.dumps(dict(best_index=int(np.argmin([r['score'] for r in reports])),best=min(reports,key=lambda r:r['score']),completed_steps=report['new_gpu_environment_steps']),indent=2),flush=True)


if __name__=='__main__':main()
