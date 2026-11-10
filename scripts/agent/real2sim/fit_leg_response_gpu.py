"""Finite batched GPU trials of effective joint friction/damping, without future-state forcing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay
from scripts.agent.real2sim import elastic_joint_resistance as elastic
from scripts.agent.real2sim import leg_load_potential as load_potential
from scripts.agent.real2sim import smooth_motor_resistance as smooth_resistance
from scripts.agent.real2sim import leg_memory_history as memory_history


def set_extension_stop_xml(root,candidates):
    """Compile limit switches from XML; mj_saveLastXML does not preserve boolean-only edits."""
    if not any('spring_extension_stop_m' in p for p in candidates):return
    stops=np.asarray([p['spring_extension_stop_m'] for p in candidates],dtype=float)
    margins=np.asarray([p.get('spring_extension_margin_m',[0.,0.]) for p in candidates],dtype=float)
    assert stops.shape==margins.shape==(len(candidates),2) and np.isfinite(stops).all() and np.isfinite(margins).all()
    assert np.all(stops==stops[0]) and np.all(margins==margins[0]) and margins.min()>=0
    for side,name in enumerate(('left','right')):
        joint=root.find(f".//joint[@name='{name}_spring2_joint']")
        joint.set('limited','true');joint.set('range',f'-.09 {stops[0,side]:.17g}')
        joint.set('margin',f'{margins[0,side]:.17g}');joint.set('solreflimit','.002 1');joint.set('solimplimit','.999 .9999 .001 .5 2')


def batch_leg_state(q, v):
    angles=q+np.array([np.pi,0.])
    ab=.21*np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    delta=ab[:,1]-ab[:,0]
    distance=np.maximum(np.linalg.norm(delta,axis=1),1e-12)
    height=np.sqrt(.25**2-distance**2/4)
    rotate=lambda x:np.column_stack((-x[:,1],x[:,0]))
    normal=rotate(delta)/distance[:,None]
    point=(ab[:,0]+ab[:,1])/2+height[:,None]*normal
    length=np.linalg.norm(point,axis=1)
    jac=np.empty_like(q)
    for j in range(2):
        da=rotate(ab[:,0]) if j==0 else np.zeros_like(delta)
        db=rotate(ab[:,1]) if j==1 else np.zeros_like(delta)
        dd=db-da;ds=np.sum(delta*dd,axis=1)/distance
        dh=-distance*ds/(4*height)
        dn=rotate(dd)/distance[:,None]-normal*(ds/distance)[:,None]
        dp=(da+db)/2+dh[:,None]*normal+height[:,None]*dn
        jac[:,j]=np.sum(point*dp,axis=1)/length
    return length,jac,np.sum(jac*v,axis=1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Completed calibrated stiff replay')
    parser.add_argument('--parameters', type=Path, required=True, help='JSON list of four-joint friction/damping candidates')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-until-s', type=float, default=3.5, help='Candidate selection window; later samples are diagnostics only')
    parser.add_argument('--max-final-error-deg', type=float, help='Optional selection criterion; only when the full trace is in the fitting window')
    parser.add_argument('--max-peak-error-deg', type=float, help='Optional peak criterion; only for full-window fitting')
    parser.add_argument('--feedback-validation', action='store_true', help='Fit both recorded-torque and recorded-target feedback branches')
    parser.add_argument('--rc-inputs', type=Path, help='Verified recorded time/dial cache; required for feedback branch')
    parser.add_argument('--friction-law', choices=('coulomb','elastic'), default='coulomb')
    parser.add_argument('--sample-stride',type=int,choices=(1,10),default=10,help='Saved diagnostic grid; integration always uses the reference dt')
    parser.add_argument('--memory-history',type=Path,help='Frozen measurements ending at replay start, for friction-state initialization only')
    args = parser.parse_args()
    reference = json.loads((args.reference / 'report.json').read_text())
    assert reference['spring_model'] == 'linear_0_150N_90mm_CAD10mm_damping50'
    assert reference['constraint_check'] == 'stiff'
    assert hashlib.sha256(Path(reference['source_model']).read_bytes()).hexdigest() == reference['source_model_sha256']
    saved = np.load(args.reference / 'replay.npz')
    candidates = json.loads(args.parameters.read_text())
    potential=None
    if any('load_potential_coefficients_nm' in p for p in candidates):
        potential=np.asarray([p['load_potential_coefficients_nm'] for p in candidates],dtype=float)
        assert potential.shape==(len(candidates),2,8) and np.isfinite(potential).all()
    assert 1 <= len(candidates) <= 64
    friction = np.array([p['friction_nm'] for p in candidates], dtype=np.float64)
    damping = np.array([p['damping_nms_rad'] for p in candidates], dtype=np.float32)
    motor_friction_bias=np.asarray([p.get('motor_friction_bias_nm',[0.]*4) for p in candidates],dtype=float)
    smooth_positive=None;smooth_negative=None;smooth_scale=None
    if any('motor_smooth_positive_nm' in p for p in candidates):
        smooth_positive=np.asarray([p['motor_smooth_positive_nm'] for p in candidates],dtype=float)
        smooth_negative=np.asarray([p['motor_smooth_negative_nm'] for p in candidates],dtype=float)
        smooth_scale=np.asarray([p['motor_smooth_velocity_scale_rad_s'] for p in candidates],dtype=float)
        assert args.friction_law=='coulomb' and np.all(friction==0) and not np.any(motor_friction_bias),'Continuous friction uses positive damping only'
        assert all(a.shape==friction.shape and np.isfinite(a).all() and a.min()>0 for a in (smooth_positive,smooth_negative,smooth_scale))
    assert motor_friction_bias.shape==friction.shape and np.isfinite(motor_friction_bias).all() and np.all(abs(motor_friction_bias)<=friction),'Asymmetric motor friction must oppose both motion directions'
    if np.any(motor_friction_bias):assert args.friction_law=='coulomb' and not any('motor_dynamic_positive_nm' in p for p in candidates),'Use the explicit static directional interval without a symmetric Stribeck threshold'
    static_excess=None;static_decay=None;static_velocity_scale=None
    if any('motor_static_excess_nm' in p for p in candidates):
        assert args.friction_law=='coulomb' and not any('motor_dynamic_positive_nm' in p for p in candidates)
        static_excess=np.asarray([p['motor_static_excess_nm'] for p in candidates],dtype=float)
        static_decay=np.asarray([p['motor_static_decay_m'] for p in candidates],dtype=float)
        static_velocity_scale=np.asarray([p['motor_static_velocity_scale_rad_s'] for p in candidates],dtype=float)
        assert static_excess.shape==static_velocity_scale.shape==friction.shape and static_decay.shape==(len(candidates),2)
        assert np.isfinite(static_excess).all() and static_excess.min()>=0 and np.isfinite(static_decay).all() and static_decay.min()>0
        assert np.isfinite(static_velocity_scale).all() and static_velocity_scale.min()>0
        static_decay=np.repeat(static_decay,2,axis=1)
    spring_bias=np.array([p.get('spring_bias_n',[0.,0.]) for p in candidates],dtype=np.float32)
    spring_friction=np.asarray([p.get('spring_friction_n',[0.,0.]) for p in candidates],dtype=float)
    assert spring_friction.shape==(len(candidates),2) and np.isfinite(spring_friction).all() and spring_friction.min()>=0
    seal_capacity=None;seal_stiffness=None;seal_positive=None;seal_negative=None;seal_scale=None
    if any('spring_seal_static_n' in p for p in candidates):
        if args.friction_law=='elastic':
            assert all(p.get('preload_rule')=='minimum_energy_shared' for p in candidates),'Joint memory needs one shared equilibrium allocation rule'
            assert not any('coordinate_friction_nm' in p for p in candidates),'Shared preload currently allocates motor and seal states only'
        fields={key:np.asarray([p[key] for p in candidates],dtype=float) for key in
                ('spring_seal_static_n','spring_seal_stiffness_n_m','spring_seal_positive_n','spring_seal_negative_n','spring_seal_velocity_scale_m_s')}
        assert all(a.shape==(len(candidates),2) and np.isfinite(a).all() and a.min()>0 for a in fields.values())
        seal_capacity=fields['spring_seal_static_n'];seal_stiffness=fields['spring_seal_stiffness_n_m']
        seal_positive=fields['spring_seal_positive_n'];seal_negative=fields['spring_seal_negative_n'];seal_scale=fields['spring_seal_velocity_scale_m_s']
        assert np.all(seal_positive<=seal_capacity) and np.all(seal_negative<=seal_capacity)
    assert spring_bias.shape==(len(candidates),2) and np.isfinite(spring_bias).all() and np.min(spring_bias)>=0 and np.max(spring_bias)<=150
    spring_parameters={'spring_bias_n':spring_bias}
    rated_spring=any('spring_reference_force_n' in p for p in candidates)
    if rated_spring:
        assert not np.any(spring_bias),'Do not combine the rated-force and old bias hypotheses'
        for key in ('spring_reference_force_n','spring_stiffness_n_m','spring_force_limit_n'):
            values=np.asarray([p[key] for p in candidates],dtype=float)
            assert values.shape==(len(candidates),2) and np.isfinite(values).all() and values.min()>=0
            spring_parameters[key]=values
        assert np.all(spring_parameters['spring_force_limit_n']>=spring_parameters['spring_reference_force_n'])
        spring_parameters['spring_damping_ns_m']=np.asarray([p.get('spring_damping_ns_m',[50.,50.]) for p in candidates],dtype=float)
        assert spring_parameters['spring_damping_ns_m'].shape==(len(candidates),2)
        assert np.isfinite(spring_parameters['spring_damping_ns_m']).all() and spring_parameters['spring_damping_ns_m'].min()>=0
        if any('spring_force_knots_n' in p for p in candidates):
            grids=np.asarray([p['spring_compression_knots_m'] for p in candidates],dtype=float)
            forces=np.asarray([p['spring_force_knots_n'] for p in candidates],dtype=float)
            assert grids.ndim==2 and grids.shape[1]>=2 and np.all(grids==grids[0]) and np.all(np.diff(grids,axis=1)>0)
            assert forces.shape==(len(candidates),2,grids.shape[1]) and np.isfinite(forces).all() and forces.min()>=0 and np.all(np.diff(forces,axis=-1)>=0)
            spring_parameters.update(spring_compression_knots_m=grids,spring_force_knots_n=forces)
    assert friction.shape == damping.shape == (len(candidates), 4)
    assert np.isfinite(friction).all() and np.isfinite(damping).all() and np.min(friction) >= 0 and np.min(damping) >= 0
    motor_dynamic_positive=None;motor_dynamic_negative=None;motor_velocity_scale=None
    motor_friction_decay=None
    if any('motor_dynamic_positive_nm' in p for p in candidates):
        motor_dynamic_positive=np.asarray([p['motor_dynamic_positive_nm'] for p in candidates],dtype=float)
        motor_dynamic_negative=np.asarray([p['motor_dynamic_negative_nm'] for p in candidates],dtype=float)
        motor_velocity_scale=np.asarray([p['motor_velocity_scale_rad_s'] for p in candidates],dtype=float)
        for dynamic in (motor_dynamic_positive,motor_dynamic_negative):
            assert dynamic.shape==friction.shape and np.isfinite(dynamic).all() and dynamic.min()>0 and np.all(dynamic<=friction)
        assert motor_velocity_scale.shape==friction.shape and np.isfinite(motor_velocity_scale).all() and motor_velocity_scale.min()>0
    if any('motor_friction_decay_m' in p for p in candidates):
        assert args.friction_law=='coulomb' and motor_dynamic_positive is not None,'Position-dependent native friction needs explicit kinetic capacities'
        motor_friction_decay=np.asarray([p['motor_friction_decay_m'] for p in candidates],dtype=float)
        assert motor_friction_decay.shape==(len(candidates),2) and np.isfinite(motor_friction_decay).all() and motor_friction_decay.min()>0
        motor_friction_decay=np.repeat(motor_friction_decay,2,axis=1)
    if args.friction_law=='elastic':
        resistance_rule=candidates[0].get('motor_resistance_rule','dahl')
        assert resistance_rule in ('dahl','plastic') and all(p.get('motor_resistance_rule','dahl')==resistance_rule for p in candidates)
        stiffness=np.array([p['stiffness_nm_rad'] for p in candidates],dtype=np.float64)
        assert stiffness.shape==friction.shape and np.min(stiffness)>=0 and np.any(stiffness>0) and np.min(friction)>0
    elif any(np.any(np.asarray(p.get('stiffness_nm_rad',[0.]*4))>0) for p in candidates):
        raise ValueError('Stiffness candidates require --friction-law elastic')
    args.output.mkdir(parents=True, exist_ok=False)
    payload=np.asarray([p.get('distal_payload_kg',[0.,0.]) for p in candidates],dtype=float)
    assert payload.shape==(len(candidates),2) and np.isfinite(payload).all() and payload.min()>=0
    assert np.all(payload==payload[0]),'Use separate batches for different mass constants'
    coordinate_friction=None
    coordinate_bias=None
    coordinate_stiffness=None;coordinate_damping=None
    coordinate_dynamic_positive=None;coordinate_dynamic_negative=None;coordinate_velocity_scale=None
    if any('coordinate_friction_nm' in p for p in candidates):
        coordinate_friction=np.asarray([p.get('coordinate_friction_nm',[0.]*4) for p in candidates],dtype=np.float64)
        assert coordinate_friction.shape==(len(candidates),4) and np.isfinite(coordinate_friction).all() and coordinate_friction.min()>=0
        coordinate_damping=np.asarray([p.get('coordinate_damping_nms_rad',[0.]*4) for p in candidates],dtype=float)
        assert coordinate_damping.shape==coordinate_friction.shape and np.isfinite(coordinate_damping).all() and coordinate_damping.min()>=0
        coordinate_bias=np.asarray([p.get('coordinate_bias_nm',[0.]*4) for p in candidates],dtype=np.float32)
        assert coordinate_bias.shape==coordinate_friction.shape and np.isfinite(coordinate_bias).all()
        assert np.all(np.abs(coordinate_bias)<=coordinate_friction), 'Directional friction must oppose both motion directions'
        if any('coordinate_stiffness_nm_rad' in p for p in candidates):
            coordinate_stiffness=np.asarray([p.get('coordinate_stiffness_nm_rad',[0.]*4) for p in candidates],dtype=float)
            assert coordinate_stiffness.shape==coordinate_friction.shape and np.isfinite(coordinate_stiffness).all() and coordinate_stiffness.min()>=0
            assert not np.any(coordinate_bias), 'Stateful coordinate resistance cannot use the native directional-bias rule'
            assert coordinate_friction.min()>0
        if any('coordinate_dynamic_positive_nm' in p for p in candidates):
            coordinate_dynamic_positive=np.asarray([p['coordinate_dynamic_positive_nm'] for p in candidates],dtype=float)
            coordinate_dynamic_negative=np.asarray([p['coordinate_dynamic_negative_nm'] for p in candidates],dtype=float)
            coordinate_velocity_scale=np.asarray([p['coordinate_velocity_scale_rad_s'] for p in candidates],dtype=float)
            if coordinate_velocity_scale.ndim==1:coordinate_velocity_scale=coordinate_velocity_scale[:,None]
            for dynamic in (coordinate_dynamic_positive,coordinate_dynamic_negative):
                assert dynamic.shape==coordinate_friction.shape and np.isfinite(dynamic).all() and dynamic.min()>0 and np.all(dynamic<=coordinate_friction)
            assert np.all(coordinate_bias>=-coordinate_dynamic_positive) and np.all(coordinate_bias<=coordinate_dynamic_negative), 'Directional bias must also preserve dissipative moving friction'
            assert coordinate_velocity_scale.shape in ((len(candidates),1),(len(candidates),4))
            assert np.isfinite(coordinate_velocity_scale).all() and coordinate_velocity_scale.min()>0
        root=ET.parse(args.reference/'fixed-base-model.xml').getroot()
        if np.any(payload):
            parents={child:parent for parent in root.iter() for child in parent}
            for side,wheel in enumerate(('lfwheel','rfwheel')):
                if payload[0,side]==0:continue
                wheel_body=root.find(f".//body[@name='{wheel}']")
                # 点质量放在轮轴同位置、固定在腿上，不改轮电机的转动惯量标定。
                load=ET.SubElement(parents[wheel_body],'body',name=f'diagnostic_distal_load_{side}',pos=wheel_body.get('pos','0 0 0'))
                ET.SubElement(load,'inertial',pos='0 0 0',mass=str(payload[0,side]),diaginertia='0.00000001 0.00000001 0.00000001')
        tendon=root.find('tendon')
        if tendon is None:tendon=ET.SubElement(root,'tendon')
        for side,names in enumerate((('lf0','lf00'),('rf0','rf00'))):
            for kind,coefficients in (('common',(.5,.5)),('difference',(.5,-.5))):
                fixed=ET.SubElement(tendon,'fixed',name=f'resistance_{side}_{kind}',frictionloss='0.000001',
                                    solreffriction='0.002 1',solimpfriction='0.999 0.9999 0.001 0.5 2')
                for name,coefficient in zip(names,coefficients):ET.SubElement(fixed,'joint',joint=name,coef=str(coefficient))
        set_extension_stop_xml(root,candidates)
        xml=ET.tostring(root,encoding='unicode');(args.output/'fixed-base-template.xml').write_text(xml)
        model=mujoco.MjModel.from_xml_string(xml)
    else:
        assert not np.any(payload),'Distal-load diagnostics currently require the coordinate-resistance model'
        if any('spring_extension_stop_m' in p for p in candidates):
            root=ET.parse(args.reference/'fixed-base-model.xml').getroot();set_extension_stop_xml(root,candidates)
            xml=ET.tostring(root,encoding='unicode');(args.output/'fixed-base-template.xml').write_text(xml)
            model=mujoco.MjModel.from_xml_string(xml)
        else:model = mujoco.MjModel.from_xml_path(str(args.reference / 'fixed-base-model.xml'))
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    model.opt.timestep = reference['dt_s']
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    mq = [model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]]
    mv = [model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[:4]]
    motor_ids = [model.actuator(n + '_motor').id for n in replay.mirror.MOTORS]
    sq = [model.joint(side + '_spring2_joint').qposadr[0] for side in ('left', 'right')]
    sv = [model.joint(side + '_spring2_joint').dofadr[0] for side in ('left', 'right')]
    spring_ids = [model.actuator(side + '_gas_spring_motor').id for side in ('left', 'right')]
    extension_stop=None
    if any('spring_extension_stop_m' in p for p in candidates):
        extension_stop=np.asarray([p['spring_extension_stop_m'] for p in candidates],dtype=float)
        assert extension_stop.shape==(len(candidates),2) and np.isfinite(extension_stop).all() and np.all(extension_stop==extension_stop[0]),'Use one stop geometry per batch'
        margins=np.asarray([p.get('spring_extension_margin_m',[0.,0.]) for p in candidates],dtype=float)
        assert margins.shape==extension_stop.shape and np.all(margins==margins[0]) and margins.min()>=0
        for side,name in enumerate(('left','right')):
            joint=model.joint(name+'_spring2_joint');jid=joint.id
            model.jnt_limited[jid]=True;model.jnt_range[jid]=[-.09,extension_stop[0,side]]
            model.jnt_margin[jid]=margins[0,side];model.jnt_solref[jid]=[.002,1.]
            model.jnt_solimp[jid]=[.999,.9999,.001,.5,2.]
    if rated_spring:
        # 原XML的150N夹限会掩盖压缩增力；只放宽这份诊断副本。
        limits=np.max(spring_parameters['spring_force_limit_n'],axis=0)
        model.actuator_ctrlrange[spring_ids]=np.column_stack((np.zeros(2),limits))
        model.actuator_forcerange[spring_ids]=np.column_stack((np.zeros(2),limits))
    if 'armature_kg_m2' in candidates[0]:
        armatures=np.asarray([p['armature_kg_m2'] for p in candidates],dtype=float)
        assert armatures.shape==(len(candidates),4) and np.isfinite(armatures).all() and np.min(armatures)>=0
        assert np.all(armatures==armatures[0]), 'Use separate batches for different inertias to rebuild all model constants'
        model.dof_armature[mv]=armatures[0]
        mujoco.mj_setConst(model,mujoco.MjData(model))
    # 静态编译表须包含四驱动摩擦自由度，实际每个world仍使用各自的候选值。
    model.dof_frictionloss[mv] = 1e-6
    if np.any(spring_friction):
        model.dof_frictionloss[sv]=1e-6
        model.dof_solref[sv]=[.002,1.]
        model.dof_solimp[sv]=[.999,.9999,.001,.5,2.]
    model.dof_solref[mv] = [.002, 1.]
    model.dof_solimp[mv] = [.999, .9999, .001, .5, 2.]
    mujoco.mj_saveLastXML(str(args.output/'candidate-model.xml'),model)
    data = mujoco.MjData(model)
    data.qpos[:] = saved['qpos'][0, 0]
    data.qvel[:] = saved['qvel'][0, 0]
    mujoco.mj_forward(model, data)  # 初始化约束，不推进CPU物理。
    ncase = len(candidates)
    nworld = ncase * (2 if args.feedback_validation else 1)
    if args.feedback_validation:
        friction=np.tile(friction,(2,1));damping=np.tile(damping,(2,1))
        motor_friction_bias=np.tile(motor_friction_bias,(2,1))
        if smooth_positive is not None:
            smooth_positive=np.tile(smooth_positive,(2,1));smooth_negative=np.tile(smooth_negative,(2,1));smooth_scale=np.tile(smooth_scale,(2,1))
        if static_excess is not None:
            static_excess=np.tile(static_excess,(2,1));static_decay=np.tile(static_decay,(2,1));static_velocity_scale=np.tile(static_velocity_scale,(2,1))
        spring_bias=np.tile(spring_bias,(2,1))
        spring_friction=np.tile(spring_friction,(2,1))
        if seal_capacity is not None:
            seal_capacity=np.tile(seal_capacity,(2,1));seal_stiffness=np.tile(seal_stiffness,(2,1))
            seal_positive=np.tile(seal_positive,(2,1));seal_negative=np.tile(seal_negative,(2,1));seal_scale=np.tile(seal_scale,(2,1))
        spring_parameters={key:np.tile(value,(2,)+(1,)*(value.ndim-1)) for key,value in spring_parameters.items()}
        if coordinate_friction is not None:coordinate_friction=np.tile(coordinate_friction,(2,1))
        if coordinate_bias is not None:coordinate_bias=np.tile(coordinate_bias,(2,1))
        if coordinate_stiffness is not None:coordinate_stiffness=np.tile(coordinate_stiffness,(2,1))
        if coordinate_damping is not None:coordinate_damping=np.tile(coordinate_damping,(2,1))
        if coordinate_dynamic_positive is not None:
            coordinate_dynamic_positive=np.tile(coordinate_dynamic_positive,(2,1))
            coordinate_dynamic_negative=np.tile(coordinate_dynamic_negative,(2,1))
            coordinate_velocity_scale=np.tile(coordinate_velocity_scale,(2,1))
        if args.friction_law=='elastic':stiffness=np.tile(stiffness,(2,1))
        if motor_dynamic_positive is not None:
            motor_dynamic_positive=np.tile(motor_dynamic_positive,(2,1));motor_dynamic_negative=np.tile(motor_dynamic_negative,(2,1))
            motor_velocity_scale=np.tile(motor_velocity_scale,(2,1))
        if motor_friction_decay is not None:motor_friction_decay=np.tile(motor_friction_decay,(2,1))
        if potential is not None:potential=np.tile(potential,(2,1,1))
    friction_table = np.tile(model.dof_frictionloss, (nworld, 1)).astype(np.float32)
    damping_table = np.tile(model.dof_damping, (nworld, 1)).astype(np.float32)
    friction_table[:,mv] = np.where(stiffness>0,0.,friction) if args.friction_law=='elastic' else friction
    friction_table[:,sv]=spring_friction
    damping_table[:,mv] = damping
    if coordinate_friction is not None:
        tendon_ids=[model.tendon(f'resistance_{side}_{kind}').id for side in range(2) for kind in ('common','difference')]
        tendon_table=np.tile(model.tendon_frictionloss,(nworld,1)).astype(np.float32)
        tendon_table[:,tendon_ids]=np.where(coordinate_stiffness>0,0.,coordinate_friction) if coordinate_stiffness is not None else coordinate_friction
    controls = saved['submitted_model_torque'][:,0].astype(np.float32)
    steps = len(controls)
    history_info=None;history_preloads=None;history_motor=None;history_seal=None
    if args.memory_history is not None:
        assert seal_capacity is not None and coordinate_friction is None and potential is None
        assert static_excess is None and motor_friction_decay is None and smooth_positive is None
        history,history_info=memory_history.load(args.memory_history,reference,data.qpos)
        history_motor,history_seal,history_preloads=memory_history.estimate(model,data,candidates,history,args.friction_law=='elastic')
    preload_info=None;minimum_loss=0.
    base_applied=np.zeros((nworld,model.nv),dtype=np.float32)
    base_applied[:,mv]=-motor_friction_bias
    seal_state=None;seal_preload_info=None;minimum_seal_loss=0.;minimum_discrete_seal_loss=0.
    seal_discrete_min=np.zeros((nworld,2))
    initial_equilibrium_error=np.zeros(ncase)
    if seal_capacity is not None and args.friction_law!='elastic':
        states=[];seal_preload_info=[]
        for case in range(ncase):
            initial_force=replay.spring_state(data.qpos[sq],'linear-damped',np.zeros(2),{k:v[case] for k,v in spring_parameters.items()})[1]
            residual,info=elastic.initial_preload(model,data,controls[0,:4],initial_spring_force=initial_force,return_tangent=True)
            tangent=np.asarray(info.pop('tangent'));jac=tangent[sv,:].T
            if candidates[case].get('preload_rule')=='minimum_seal_energy_native_interval':
                static_motor=friction[case].copy()
                if static_excess is not None:
                    compression=np.repeat(np.maximum(0.,.010-data.qpos[sq]),2)
                    static_motor+=static_excess[case]*np.exp(-compression/static_decay[case])
                initial,allocation=elastic.minimum_seal_energy_preload(residual,jac,static_motor,motor_friction_bias[case],seal_capacity[case],
                    prior=history_seal[case] if history_seal is not None else None)
                initial_equilibrium_error[case]=allocation['equilibrium_max_error_nm'];info['shared_native_interval_allocation']=allocation
            else:initial=np.linalg.lstsq(jac,residual,rcond=None)[0]
            states.append(np.clip(initial,-seal_capacity[case],seal_capacity[case]))
            info.update(projected_seal_friction_n=states[-1].tolist(),unclipped_seal_friction_n=initial.tolist())
            seal_preload_info.append(info)
        seal_state=np.asarray(states)
        if args.feedback_validation:seal_state=np.tile(seal_state,(2,1))
        applied=np.zeros((nworld,model.nv),dtype=np.float32)
    if coordinate_bias is not None:
        for side in range(2):
            common,difference=coordinate_bias[:,2*side],coordinate_bias[:,2*side+1]
            base_applied[:,mv[2*side]]-= (common+difference)/2
            base_applied[:,mv[2*side+1]]-= (common-difference)/2
    if args.friction_law=='elastic':
        preload=[];preload_info=[];shared_seal=[]
        for case in range(ncase):
            initial_force=replay.spring_state(data.qpos[sq],'linear-damped',np.zeros(2),{k:v[case] for k,v in spring_parameters.items()})[1]
            residual,info=elastic.initial_preload(model,data,controls[0,:4],initial_spring_force=initial_force,return_tangent=seal_capacity is not None)
            if potential is not None:
                correction=load_potential.evaluate(data.qpos[mq],potential[case])[1]
                residual-=correction;info['load_potential_torque_nm']=correction.tolist()
            if seal_capacity is not None:
                tangent=np.asarray(info.pop('tangent'))
                motor_initial,seal_initial,allocation=elastic.minimum_energy_preload(residual,tangent[sv,:].T,
                    stiffness[case],seal_stiffness[case],friction[case],seal_capacity[case],
                    prior=np.r_[history_motor[case],history_seal[case]] if history_motor is not None else None)
                initial_equilibrium_error[case]=allocation['equilibrium_max_error_nm']
                info['shared_preload_allocation']=allocation;preload.append(motor_initial);shared_seal.append(seal_initial)
            else:preload.append(residual)
            preload_info.append(info)
        preload=np.asarray(preload)
        if args.feedback_validation:preload=np.tile(preload,(2,1))
        if all(info==preload_info[0] for info in preload_info):preload_info=preload_info[0]
        elastic_mask=stiffness>0
        friction_state=np.where(elastic_mask,np.clip(preload,-friction,friction),0.)
        if seal_capacity is not None:
            seal_state=np.asarray(shared_seal)
            if args.feedback_validation:seal_state=np.tile(seal_state,(2,1))
            seal_preload_info=[info['shared_preload_allocation'] for info in (preload_info if isinstance(preload_info,list) else [preload_info])]
        applied=np.zeros((nworld,model.nv),dtype=np.float32)
    coordinate_state=None;coordinate_preload=None;minimum_coordinate_loss=0.
    minimum_discrete_coordinate_loss=0.
    discrete_coordinate_min=np.zeros((nworld,2))
    discrete_motor_min=np.zeros((nworld,2));minimum_coordinate_viscous_work=0.
    coordinate_transform=np.kron(np.eye(2),np.array([[.5,.5],[.5,-.5]]))
    if coordinate_damping is not None:applied=np.zeros((nworld,model.nv),dtype=np.float32)
    if potential is not None:applied=np.zeros((nworld,model.nv),dtype=np.float32)
    if coordinate_stiffness is not None and np.any(coordinate_stiffness>0):
        coordinate_preload=[];initial=[]
        for case in range(ncase):
            initial_force=replay.spring_state(data.qpos[sq],'linear-damped',np.zeros(2),{k:v[case] for k,v in spring_parameters.items()})[1]
            residual,info=elastic.initial_preload(model,data,controls[0,:4],initial_spring_force=initial_force)
            if potential is not None:
                correction=load_potential.evaluate(data.qpos[mq],potential[case])[1]
                residual-=correction;info['load_potential_torque_nm']=correction.tolist()
            initial.append(residual@np.linalg.inv(coordinate_transform))
            coordinate_preload.append(info)
        initial=np.asarray(initial)
        if args.feedback_validation:initial=np.tile(initial,(2,1))
        coordinate_state=np.where(coordinate_stiffness>0,np.clip(initial,-coordinate_friction,coordinate_friction),0.)
        applied=np.zeros((nworld,model.nv),dtype=np.float32)
    config = Path(json.loads((Path(reference['session'])/'meta.json').read_text())['machine_config']['path'])
    assert hashlib.sha256(config.read_bytes()).hexdigest() == reference['config_sha256']
    sign,zero = replay.mirror.load_mapping(config)
    if args.feedback_validation:
        assert args.rc_inputs is not None
        rc=np.load(args.rc_inputs)
        assert str(rc['stream_sha256'])==reference['stream_sha256']
        absolute_step=reference['window_start_board_s']+saved['time_s'][:-1]
        geometry=saved['real_geometry']
        geometry_indices=np.maximum(0,np.searchsorted(geometry[:,0],absolute_step,side='right')-1)
        targets=geometry[geometry_indices,4:6]
        boundary_flags=(geometry[:,6].astype(int)&64)!=0
        assert not np.any(boundary_flags!=boundary_flags[0]), 'Mixed boundary control versions'
        soft_boundary=bool(boundary_flags[0])
        dial=rc['dial_rate'][np.maximum(0,np.searchsorted(rc['time_s'],absolute_step,side='right')-1)]
        kp,kd=replay.gains(config)
        recovering=np.full(ncase,bool(int(geometry[geometry_indices[0],6])&32));blocked=np.zeros(ncase,dtype=bool)
    stride = args.sample_stride
    sample_indices = np.arange(0, steps + 1, stride)
    qpos = np.empty((len(sample_indices),nworld,model.nq),dtype=np.float32)
    qvel = np.empty((len(sample_indices),nworld,model.nv),dtype=np.float32)
    submitted_control=np.empty((len(sample_indices)-1,nworld,model.nu),dtype=np.float32)
    submitted_applied=np.empty((len(sample_indices)-1,nworld,model.nv),dtype=np.float32)
    submitted_motor_friction=np.empty((len(sample_indices)-1,nworld,4),dtype=np.float32)
    submitted_slider_friction=np.empty((len(sample_indices)-1,nworld,2),dtype=np.float32)
    submitted_motor_damping=np.empty_like(submitted_motor_friction)
    smooth_discrete_min=np.zeros((nworld,2))
    spring_min=np.full((nworld,2),np.inf);spring_max=np.full((nworld,2),-np.inf)
    import mujoco_warp as mw
    import warp as wp
    wp.init()
    started = time.monotonic()
    with wp.ScopedDevice('cuda:0'):
        batch_sizes={'dof_frictionloss':nworld,'dof_damping':nworld}
        if coordinate_friction is not None:batch_sizes['tendon_frictionloss']=nworld
        gm = mw.put_model(model, batch_sizes=batch_sizes)
        gm.dof_frictionloss.assign(friction_table)
        gm.dof_damping.assign(damping_table)
        if coordinate_friction is not None:gm.tendon_frictionloss.assign(tendon_table)
        gd = mw.put_data(model,data,nworld=nworld,nconmax=128,njmax=256)
        gd.qfrc_applied.assign(base_applied)
        with wp.ScopedCapture() as capture:
            mw.step(gm,gd)
        assert not np.any(gd.time.numpy())
        qp,qv = gd.qpos.numpy(),gd.qvel.numpy()
        qpos[0],qvel[0] = qp,qv
        ctrl = np.zeros((nworld,model.nu),dtype=np.float32)
        for k in range(steps):
            if time.monotonic()-started > 180:
                raise TimeoutError('Finite candidate batch exceeded180s')
            ctrl[:,motor_ids] = controls[k]
            applied_now=base_applied
            if smooth_positive is not None:
                damping_table[:,mv]=damping+smooth_resistance.damping(qv[:,mv],smooth_positive,smooth_negative,smooth_scale)
                gm.dof_damping.assign(damping_table)
                previous_smooth_q=qp[:,mq].copy()
            if static_excess is not None:
                compression=np.repeat(np.maximum(0.,.010-qp[:,sq]),2,axis=1)
                friction_table[:,mv]=friction+static_excess*np.exp(-compression/static_decay)*np.exp(-(qv[:,mv]/static_velocity_scale)**2)
                gm.dof_frictionloss.assign(friction_table)
            if motor_dynamic_positive is not None:
                kinetic=np.where(qv[:,mv]>=0,motor_dynamic_positive,motor_dynamic_negative)
                motor_limit=kinetic+(friction-kinetic)*np.exp(-(qv[:,mv]/motor_velocity_scale)**2)
                if motor_friction_decay is not None:
                    compression=np.repeat(np.maximum(0.,.010-qp[:,sq]),2,axis=1)
                    motor_limit*=np.exp(-compression/motor_friction_decay)
                friction_table[:,mv]=np.where(stiffness>0,0.,motor_limit) if args.friction_law=='elastic' else motor_limit
                gm.dof_frictionloss.assign(friction_table)
            if coordinate_dynamic_positive is not None:
                coordinate_velocity=qv[:,mv]@coordinate_transform.T
                kinetic=np.where(coordinate_velocity>=0,coordinate_dynamic_positive,coordinate_dynamic_negative)
                native_limit=kinetic+(coordinate_friction-kinetic)*np.exp(-(coordinate_velocity/coordinate_velocity_scale)**2)
                tendon_table[:,tendon_ids]=np.where(coordinate_stiffness>0,0.,native_limit) if coordinate_stiffness is not None else native_limit
                gm.tendon_frictionloss.assign(tendon_table)
            if args.friction_law=='elastic' or coordinate_state is not None or coordinate_damping is not None or potential is not None or seal_state is not None:
                previous_q=qp[:,mq].copy()
                if seal_state is not None:
                    previous_slide=qp[:,sq].copy();previous_slide_velocity=qv[:,sv].copy()
                    kinetic_seal=np.where(qv[:,sv]>=0,seal_positive,seal_negative)
                    seal_limit=kinetic_seal+(seal_capacity-kinetic_seal)*np.exp(-(qv[:,sv]/seal_scale)**2)
                if coordinate_dynamic_positive is not None:
                    coordinate_velocity=qv[:,mv]@coordinate_transform.T
                    kinetic=np.where(coordinate_velocity>=0,coordinate_dynamic_positive,coordinate_dynamic_negative)
                    coordinate_limit=kinetic+(coordinate_friction-kinetic)*np.exp(-(coordinate_velocity/coordinate_velocity_scale)**2)
                applied[:]=base_applied
                if seal_state is not None:applied[:,sv]-=seal_state
                if potential is not None:applied[:,mv]-=load_potential.evaluate(qp[:,mq],potential)[1]
                if args.friction_law=='elastic':applied[:,mv]-=friction_state
                if coordinate_state is not None:applied[:,mv]-=coordinate_state@coordinate_transform
                if coordinate_damping is not None:applied[:,mv]-=((qv[:,mv]@coordinate_transform.T)*coordinate_damping)@coordinate_transform
                if coordinate_damping is not None:previous_coordinate_velocity=qv[:,mv]@coordinate_transform.T
                gd.qfrc_applied.assign(applied)
                applied_now=applied
            ctrl[:,spring_ids] = replay.spring_state(qp[:,sq],'linear-damped',qv[:,sv],spring_parameters)[1]
            spring_min=np.minimum(spring_min,ctrl[:,spring_ids]);spring_max=np.maximum(spring_max,ctrl[:,spring_ids])
            if args.feedback_validation:
                fwq=qp[ncase:,mq]*sign[:4]+zero;fwv=qv[ncase:,mv]*sign[:4]
                lengths=[];tau=np.zeros((ncase,4))
                for side in range(2):
                    length,jac,velocity=batch_leg_state(fwq[:,2*side:2*side+2],fwv[:,2*side:2*side+2])
                    blocked |= ~np.isfinite(length)
                    if not soft_boundary:blocked |= (length<.14)|((~recovering)&(length>.33))
                    lengths.append(length)
                    force=kp[side]*(targets[k,side]-length)-kd[side]*velocity
                    force=np.where(recovering,np.minimum(force,0.),force)
                    if soft_boundary:
                        force=np.where(length>=.33,np.minimum(force,0.),np.where(length<=.14,np.maximum(force,0.),force))
                    tau[:,2*side:2*side+2]=np.clip(jac*force[:,None],-40,40)
                if dial[k]==0 and np.max(targets[k])<=.33:
                    recovering &= np.max(np.column_stack(lengths),axis=1)>.33
                tau[blocked]=0.
                ctrl[ncase:,motor_ids[:4]]=tau*sign[:4]
            if k%stride==0:
                submitted_control[k//stride]=ctrl;submitted_applied[k//stride]=applied_now
                submitted_motor_friction[k//stride]=friction_table[:,mv];submitted_motor_damping[k//stride]=damping_table[:,mv]
                submitted_slider_friction[k//stride]=friction_table[:,sv]
            gd.ctrl.assign(ctrl)
            wp.capture_launch(capture.graph)
            qp,qv = gd.qpos.numpy(),gd.qvel.numpy()
            if smooth_positive is not None:
                delta=qp[:,mq]-previous_smooth_q
                for side in (0,1):
                    section=slice(2*side,2*side+2)
                    work=np.sum(damping_table[:,mv][:,section]*qv[:,mv][:,section]*delta[:,section],axis=1)
                    smooth_discrete_min[:,side]=np.minimum(smooth_discrete_min[:,side],work)
            if seal_state is not None:
                delta_slide=qp[:,sq]-previous_slide;old_seal=seal_state.copy()
                seal_state,loss=elastic.advance(seal_state,delta_slide,seal_limit,seal_stiffness,capacity=seal_capacity)
                minimum_seal_loss=min(minimum_seal_loss,loss)
                stored=(seal_state**2-old_seal**2)/(2*seal_stiffness)
                step_loss=old_seal*delta_slide-stored
                # 与实际施加的轴向粘滞阻尼合并检查离散储能。
                if rated_spring:step_loss+=spring_parameters['spring_damping_ns_m']*previous_slide_velocity*delta_slide
                minimum_discrete_seal_loss=min(minimum_discrete_seal_loss,float(step_loss.min()))
                seal_discrete_min=np.minimum(seal_discrete_min,step_loss)
            if args.friction_law=='elastic':
                previous_force=friction_state.copy();delta=qp[:,mq]-previous_q
                friction_state,loss=elastic.advance(friction_state,np.where(elastic_mask,delta,0.),
                                                  motor_limit if motor_dynamic_positive is not None else friction,
                                                  np.maximum(stiffness,1.),capacity=friction,rule=resistance_rule)
                minimum_loss=min(minimum_loss,loss)
                for side in range(2):
                    section=slice(2*side,2*side+2)
                    stored=np.sum(np.where(elastic_mask[:,section],(friction_state[:,section]**2-previous_force[:,section]**2)/(2*np.maximum(stiffness[:,section],1.)),0.),axis=1)
                    work=np.sum(previous_force[:,section]*delta[:,section],axis=1)+np.sum(damping[:,section]*delta[:,section]**2,axis=1)/model.opt.timestep
                    discrete_motor_min[:,side]=np.minimum(discrete_motor_min[:,side],work-stored)
            if coordinate_state is not None:
                delta=(qp[:,mq]-previous_q)@coordinate_transform.T
                previous_force=coordinate_state.copy()
                coordinate_state,loss=elastic.advance(coordinate_state,np.where(coordinate_stiffness>0,delta,0.),
                                                      coordinate_limit if coordinate_dynamic_positive is not None else coordinate_friction,
                                                      np.maximum(coordinate_stiffness,1.),capacity=coordinate_friction)
                minimum_coordinate_loss=min(minimum_coordinate_loss,loss)
                mask=coordinate_stiffness>0
                stored=np.sum(np.where(mask,(coordinate_state**2-previous_force**2)/(2*np.maximum(coordinate_stiffness,1.)),0.),axis=1)
                work=np.sum(previous_force*delta,axis=1)+np.sum(damping*(qp[:,mq]-previous_q)**2,axis=1)/model.opt.timestep
                if coordinate_damping is not None:work+=np.sum(coordinate_damping*previous_coordinate_velocity*delta,axis=1)
                minimum_discrete_coordinate_loss=min(minimum_discrete_coordinate_loss,float(np.min(work-stored)))
                for side in range(2):
                    section=slice(2*side,2*side+2)
                    side_stored=np.sum(np.where(mask[:,section],(coordinate_state[:,section]**2-previous_force[:,section]**2)/(2*np.maximum(coordinate_stiffness[:,section],1.)),0.),axis=1)
                    side_work=np.sum(previous_force[:,section]*delta[:,section],axis=1)+np.sum(damping[:,section]*(qp[:,mq][:,section]-previous_q[:,section])**2,axis=1)/model.opt.timestep
                    if coordinate_damping is not None:side_work+=np.sum(coordinate_damping[:,section]*previous_coordinate_velocity[:,section]*delta[:,section],axis=1)
                    discrete_coordinate_min[:,side]=np.minimum(discrete_coordinate_min[:,side],side_work-side_stored)
            if coordinate_damping is not None:
                delta=(qp[:,mq]-previous_q)@coordinate_transform.T
                minimum_coordinate_viscous_work=min(minimum_coordinate_viscous_work,float(np.min(np.sum(coordinate_damping*previous_coordinate_velocity*delta,axis=1))))
            if not np.isfinite(qp).all() or not np.isfinite(qv).all():
                raise ValueError('Nonfinite GPU state')
            if (k+1)%stride == 0:
                qpos[(k+1)//stride],qvel[(k+1)//stride] = qp,qv
            if (k+1)%2000 == 0:
                print(json.dumps(dict(progress_step=k+1,worlds=nworld,elapsed_s=time.monotonic()-started)),flush=True)
    absolute = reference['window_start_board_s'] + saved['time_s'][sample_indices]
    measured = saved['real_motor_state']
    real = np.column_stack([np.interp(absolute,measured[:,0],measured[:,1+i]) for i in range(4)])
    predicted = qpos[:,:,mq]*sign[:4]+zero
    errors = predicted-real[:,None,:]
    assert 0 < args.fit_until_s <= saved['time_s'][-1]+1e-9
    fit_mask = saved['time_s'][sample_indices] <= args.fit_until_s+1e-9
    hold_mask = ~fit_mask
    fit_rmse = np.sqrt(np.mean(errors[fit_mask]**2,axis=0))
    held_rmse = np.sqrt(np.mean(errors[hold_mask]**2,axis=0)) if np.any(hold_mask) else None
    all_rmse = np.sqrt(np.mean(errors**2,axis=0))
    score = np.sqrt(np.mean(fit_rmse**2,axis=1))
    if args.feedback_validation:
        score=np.sqrt((score[:ncase]**2+score[ncase:]**2)/2)
        score[blocked]=np.inf
    score[initial_equilibrium_error>.1]=np.inf
    best = int(np.argmin(score))
    target_met = None
    if args.max_final_error_deg is not None or args.max_peak_error_deg is not None:
        assert not np.any(hold_mask), 'Do not select parameters using the held segment'
        eligible=np.ones(nworld,dtype=bool)
        if args.max_final_error_deg is not None:
            eligible &= np.max(np.abs(np.degrees(errors[-1])),axis=1) <= args.max_final_error_deg
        if args.max_peak_error_deg is not None:
            eligible &= np.max(np.abs(np.degrees(errors)),axis=(0,2)) <= args.max_peak_error_deg
        if args.feedback_validation:eligible=eligible[:ncase]&eligible[ncase:]&(~blocked)
        target_met = bool(np.any(eligible))
        if target_met:
            best = int(np.argmin(np.where(eligible,score,np.inf)))
    trials = []
    for i,p in enumerate(candidates):
        trials.append(dict(index=i,parameters=p,fit_motor_rmse_deg=np.degrees(fit_rmse[i]).tolist(),
                           held_motor_rmse_deg=np.degrees(held_rmse[i]).tolist() if held_rmse is not None else None,
                           all_motor_rmse_deg=np.degrees(all_rmse[i]).tolist(),
                           final_error_deg=np.degrees(errors[-1,i]).tolist(),fit_score_deg=float(np.degrees(score[i]))))
        trials[-1]['real_motor_span_deg']=np.degrees(np.ptp(real,axis=0)).tolist()
        trials[-1]['initial_static_equilibrium_error_nm']=float(initial_equilibrium_error[i])
        trials[-1]['initial_static_equilibrium_feasible']=bool(initial_equilibrium_error[i]<=.1)
        trials[-1]['all_motor_span_deg']=np.degrees(np.ptp(predicted[:,i],axis=0)).tolist()
        trials[-1].update(spring_force_min_n=spring_min[i].tolist(),spring_force_max_n=spring_max[i].tolist())
        discrete_min=np.minimum(np.minimum(np.minimum(discrete_coordinate_min,discrete_motor_min),seal_discrete_min),smooth_discrete_min)
        trials[-1]['minimum_discrete_dissipation_j']=np.minimum(discrete_min[i],discrete_min[ncase+i]).tolist() if args.feedback_validation else discrete_min[i].tolist()
        trials[-1]['peak_error_deg']=np.max(np.abs(np.degrees(errors[:,i])),axis=0).tolist()
        if args.feedback_validation:
            trials[-1].update(feedback_motor_rmse_deg=np.degrees(all_rmse[ncase+i]).tolist(),
                              feedback_final_error_deg=np.degrees(errors[-1,ncase+i]).tolist(),feedback_blocked=bool(blocked[i]))
            trials[-1]['feedback_peak_error_deg']=np.max(np.abs(np.degrees(errors[:,ncase+i])),axis=0).tolist()
            trials[-1]['feedback_motor_span_deg']=np.degrees(np.ptp(predicted[:,ncase+i],axis=0)).tolist()
    closure = np.zeros(2)
    selected=qpos[:,[best,best+ncase]] if args.feedback_validation else qpos[:,best:best+1]
    for q in selected.reshape(-1,model.nq):
        data.qpos[:] = q
        mujoco.mj_kinematics(model,data)
        for side in range(2):
            closure[side] = max(closure[side], max(np.linalg.norm(data.site(a).xpos-data.site(b).xpos) for a,b in replay.mirror.PIN_PAIRS[side]))
    summary = dict(reference=str(args.reference),source_model_sha256=reference['source_model_sha256'],
                   stream_sha256=reference['stream_sha256'],spring_model=reference['spring_model'],
                   worlds=nworld,steps_per_world=steps,new_environment_steps=nworld*steps,
                   fit_window_from_start_s=[0.,args.fit_until_s],validation_window_from_start_s=[args.fit_until_s,float(saved['time_s'][-1])],
                   selection_uses_validation=False,best_index=best,best=trials[best],trials=trials,
                   target_max_final_error_deg=args.max_final_error_deg,target_met=target_met,
                   target_max_peak_error_deg=args.max_peak_error_deg,
                   objective_branches=['recorded_torque','recorded_target_feedback'] if args.feedback_validation else ['recorded_torque'],
                   leg_boundary_mode=('return_control' if soft_boundary else 'legacy_lockout') if args.feedback_validation else 'recorded_torque_only',
                   friction_law=args.friction_law,preload_info=preload_info,minimum_elastic_dissipation=minimum_loss,
                   motor_resistance_rule=resistance_rule if args.friction_law=='elastic' else None,
                   conservative_load_correction=potential is not None,
                   spring_seal_preload_info=seal_preload_info,minimum_seal_analytic_dissipation_j=minimum_seal_loss,
                   minimum_seal_discrete_dissipation_j=minimum_discrete_seal_loss,
                   diagnostic_spring_bias_n=candidates[best].get('spring_bias_n',[0.,0.]),
                   diagnostic_spring_family='monotone_piecewise_effective_load' if 'spring_force_knots_n' in candidates[best] else ('reference_force_plus_compression' if rated_spring else 'zero_based_linear_with_optional_bias'),
                   native_motor_speed_dependent_friction=motor_dynamic_positive is not None,
                   native_motor_directional_static_friction=bool(np.any(motor_friction_bias)),
                   native_motor_configuration_dependent_static_excess=static_excess is not None,
                   smooth_motor_resistance=smooth_positive is not None,
                   memory_history=history_info,memory_history_preloads=history_preloads,
                   rc_inputs=str(args.rc_inputs) if args.rc_inputs is not None else None,
                   rc_inputs_sha256=hashlib.sha256(args.rc_inputs.read_bytes()).hexdigest() if args.rc_inputs is not None else None,
                   native_motor_configuration_dependent_friction=motor_friction_decay is not None,
                   diagnostic_spring_friction_n=candidates[best].get('spring_friction_n',[0.,0.]),
                   diagnostic_spring_extension_stop_m=extension_stop[0].tolist() if extension_stop is not None else None,
                   native_coordinate_speed_dependent_friction=coordinate_dynamic_positive is not None and coordinate_state is None,
                   saved_sample_stride=stride,
                   spring_reference_compression_m=.005 if rated_spring else None,
                   resistance_coordinates='common_difference' if coordinate_friction is not None else 'independent_motors',
                   diagnostic_distal_payload_kg=payload[0].tolist(),
                   coordinate_preload_info=coordinate_preload,minimum_coordinate_dissipation=minimum_coordinate_loss,
                   minimum_discrete_coordinate_dissipation=minimum_discrete_coordinate_loss,
                   minimum_explicit_coordinate_viscous_work_j=minimum_coordinate_viscous_work,
                   best_sampled_pin_error_m=closure.tolist(),
                   friction_constraint_solref=[.002,1.],friction_constraint_solimp=[.999,.9999,.001,.5,2.],
                   physics_backend='mujoco_warp_gpu',network_forwards=0,optimizer_updates=0,
                   elapsed_s=time.monotonic()-started,
                   limitations=['Equivalent resistance candidate, not independent friction calibration',
                                'Development data; a later segment, if present, is not an independent hardware trial',
                                'Only initial measured state is injected; later reference poses never force the simulation'])
    np.savez_compressed(args.output/'trials.npz',time_s=saved['time_s'][sample_indices],qpos=qpos,qvel=qvel,real_motor_q=real,
                        submitted_control=submitted_control,submitted_applied_force=submitted_applied,
                        submitted_motor_friction_capacity=submitted_motor_friction,submitted_motor_damping=submitted_motor_damping,
                        submitted_slider_friction_capacity=submitted_slider_friction)
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'best-parameters.json').write_text(json.dumps(candidates[best],indent=2)+'\n')
    shutil.copy2(args.parameters,args.output/'parameters.json')
    shutil.copy2(Path(__file__),args.output/Path(__file__).name)
    for source in (Path(replay.__file__),Path(elastic.__file__),Path(load_potential.__file__),Path(smooth_resistance.__file__),Path(memory_history.__file__)):
        shutil.copy2(source,args.output/source.name)
    print(json.dumps(summary['best'],indent=2),flush=True)


if __name__ == '__main__':
    main()
