"""Recheck saved GPU length-response states, frozen C control and applied loads; no stepping."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import mujoco
import numpy as np

from scripts.agent.real2sim.control_leg_pose_gpu import gas_spring_force,leg_pose,mirror
from scripts.agent.real2sim.precheck_right_leg_pose import State
from scripts.agent.real2sim.replay_length_response_gpu import mit_torque_quantize,mit_velocity_quantize


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--length-only',action='store_true',help='diagnose recorded-torque length response; angle is reported but not scored')
    args=parser.parse_args();output=args.run/'verification.json';assert not output.exists()
    identity=json.loads((args.inputs/'report.json').read_text());report=json.loads((args.run/'report.json').read_text())
    assert report['physics_backend']=='mujoco_warp_gpu' and report['new_gpu_environment_steps']>0
    assert hashlib.sha256((args.inputs/'inputs.npz').read_bytes()).hexdigest()==identity['input_npz_sha256']==report['input_identity']
    assert hashlib.sha256((args.run/'parameters.json').read_bytes()).hexdigest()==report['candidate_sha256']
    for field,path in [('model_sha256',args.run/'model.xml'),('input_model_sha256',args.inputs/'model.xml'),
                       ('baseline_parameter_sha256',args.inputs/'baseline-parameters.json'),('program_sha256',args.run/'replay_length_response_gpu.py')]:
        if field in report:assert hashlib.sha256(path.read_bytes()).hexdigest()==report[field],field
    for path,digest in identity['protected_sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    with np.load(args.inputs/'inputs.npz') as source:a={k:source[k] for k in source.files}
    with np.load(args.run/'traces.npz') as source:t={k:source[k] for k in source.files}
    assert all(np.isfinite(v).all() for v in t.values())
    model=mujoco.MjModel.from_xml_path(str(args.run/'model.xml'));data=mujoco.MjData(model)
    baseline=json.loads((args.inputs/'baseline-parameters.json').read_text())
    parameters=json.loads((args.run/'parameters.json').read_text());count=len(parameters)
    mq,mv=a['mq'].astype(int),a['mv'].astype(int);q=t['qpos'][:,:,mq];v=t['qvel'][:,:,mv];n=len(q)
    assert np.allclose(t['time_s'],a['time_s'][:n],atol=1e-12)
    assert np.max(abs(t['qpos'][0]-a['initial_q']))<1e-6 and np.max(abs(t['qvel'][0]-a['initial_v']))<1e-6
    assert not np.any(t['request_model_nm'][:,:,:2])
    stops=[report['trials'][world%count]['branches']['recorded_torque' if world<count else 'recorded_target']['guard_stop_s'] for world in range(count*2)]
    real_length,real_angle,_=leg_pose(a['real_q'][:n]*a['sign']+a['zero'])
    target_length,target_angle,_=leg_pose(a['target_q'][:n]*a['sign']+a['zero'])
    length,angle,_=leg_pose(q*a['sign']+a['zero'])
    metrics=[]
    for world in range(count*2):
        err=length[:,world,1]-real_length[:,1];ae=angle[:,world,1]-real_angle[:,1]
        moving=(t['time_s']>=5)&(t['time_s']<=25);hold=t['time_s']>=35
        fit=t['time_s']<=90;held_out=t['time_s']>90
        metrics.append(dict(length_rmse_mm=float(np.sqrt(np.mean(err**2))*1000),length_max_mm=float(abs(err).max()*1000),
                            angle_rmse_deg=float(np.degrees(np.sqrt(np.mean(ae**2)))),angle_max_deg=float(np.degrees(abs(ae).max())),
                            moving_length_rmse_mm=float(np.sqrt(np.mean(err[moving]**2))*1000) if moving.any() else None,
                            hold_length_rmse_mm=float(np.sqrt(np.mean(err[hold]**2))*1000) if hold.any() else None,
                            fit_first90_length_rmse_mm=float(np.sqrt(np.mean(err[fit]**2))*1000),
                            holdout_after90_length_rmse_mm=float(np.sqrt(np.mean(err[held_out]**2))*1000) if held_out.any() else None,
                            holdout_after90_angle_rmse_deg=float(np.degrees(np.sqrt(np.mean(ae[held_out]**2)))) if held_out.any() else None,
                            final_length_mm=float(length[-1,world,1]*1000),guard_stop_s=stops[world]))
    if report.get('torque_input_source')=='driver_estimate':
        driver_path=Path(report['driver_torque_identity']['path'])
        assert hashlib.sha256(driver_path.read_bytes()).hexdigest()==report['driver_torque_identity']['sha256']
        with np.load(driver_path) as source:
            assert np.array_equal(source['time_s'],a['time_s'])
            expected=source['feedback_model_nm'][:n,2:].astype(np.float32)
    else:expected=mit_torque_quantize(a['recorded_tau'][:n,2:])
    real_input_torque=expected.copy() if report.get('torque_input_source')=='driver_estimate' else a['recorded_tau'][:n,2:].copy()
    expected=np.tile(expected[:,None],(1,count,1))
    for world in range(count):
        if stops[world] is not None:expected[round(stops[world]/.002):,world]=0
    torque_error=float(abs(t['request_model_nm'][:,:count,2:]-expected).max())
    qref=np.array(baseline['reference_q_model_rad']);u=np.array([-1.,1.])/np.sqrt(2);w=np.array([1.,1.])/np.sqrt(2)
    expected_load=-(np.array(baseline['bias_nm'])+(q[:-1]-qref)@np.array(baseline['stiffness_nm_rad']).T)
    capacity=np.empty_like(t['friction_capacity_nm']);elastic=np.zeros_like(capacity);elastic_loss_min=0
    initial_residual=None;initial_seal_jac=None
    auto_seal=any(p.get('seal_friction_route')=='dahl' and 'seal_initial_force_n' not in p for p in parameters)
    if any(p.get('elastic_stiffness_nm_rad',0)>0 for p in parameters) or auto_seal:
        from scripts.agent.real2sim.elastic_joint_resistance import initial_preload
        data.qpos[:]=a['initial_q'];data.qvel[:]=0;model.opt.gravity[:]=a['gravity'][0]
        initial_drive=a['initial_request'].copy()
        if auto_seal and report['feedback_io']=='measured':initial_drive[2:]=mit_torque_quantize(initial_drive[2:])
        initial_residual,initial_info=initial_preload(model,data,initial_drive,return_tangent=True)
        initial_seal_jac=np.array(initial_info['tangent'])[model.joint('right_spring2_joint').dofadr[0],2:]
        initial_residual-=np.array(baseline['bias_nm'])+(a['initial_q'][mq]-qref)@np.array(baseline['stiffness_nm_rad']).T
    for world in range(count*2):
        p=parameters[world%count];delta=q[:,world,2:]-qref[2:];x,y=delta@u,delta@w
        if p.get('remove_baseline_right_load',False):expected_load[:,world,2:]=0.
        load=np.array(p['bias_delta_nm'])+p.get('length_stiffness_delta_nm_rad',0)*x[:,None]*u
        load+=p.get('length_quadratic_delta_nm_rad2',0)*x[:,None]**2*u
        load+=p.get('cross_stiffness_delta_nm_rad',0)*(x[:,None]*w+y[:,None]*u)
        expected_load[:,world,2:]-=load[:-1]
        if 'right_total_length_load_nm' in p:
            from scipy.interpolate import PchipInterpolator
            knots=p['right_total_length_knots_m']
            expected_load[:,world,2:]+=PchipInterpolator(knots,p['right_total_length_load_nm'],axis=0)(np.clip(length[:-1,world,1],knots[0],knots[-1]))
        state=np.ones(2)
        ek=p.get('elastic_stiffness_nm_rad',0);force=np.zeros(2)
        if ek:
            q0=a['initial_q'][mq[2:]]-qref[2:];x0,y0=q0@u,q0@w
            extra=np.array(p['bias_delta_nm'])+(p.get('length_stiffness_delta_nm_rad',0)*x0+p.get('length_quadratic_delta_nm_rad2',0)*x0*x0)*u+p.get('cross_stiffness_delta_nm_rad',0)*(x0*w+y0*u)
            force=np.clip((initial_residual[2:]-extra)*p.get('elastic_preload_fraction',0),-np.array(p['static_nm']),np.array(p['static_nm']))
        for i in range(n):
            desired=np.exp(-(v[i,world,2:]/p['velocity_scale_rad_s'])**2)
            dt=np.where(desired<state,p.get('friction_weakening_s',0),p.get('friction_recovery_s',0))
            state=desired+(state-desired)*np.exp(-.002/np.maximum(dt,1e-12))
            capacity[i,world]=np.array(p['kinetic_nm'])+(np.array(p['static_nm'])-p['kinetic_nm'])*state
            if ek:
                delta=(q[i,world,2:]-q[max(0,i-1),world,2:]).astype(np.float64)
                limit=capacity[i,world];factor=np.exp(-ek*abs(delta)/np.maximum(limit,1e-8));sign=np.sign(delta);target=sign*limit
                updated=target+(force-target)*factor
                work=limit*abs(delta)+sign*(force-target)*limit/ek*(1-factor)
                loss=work-(updated**2-force**2)/(2*ek);elastic_loss_min=min(elastic_loss_min,float(loss.min()))
                force=updated;elastic[i,world]=force
        expected_load[:,world,2:]-=elastic[:-1,world]
    load_error=float(abs(expected_load-t['applied_load_nm']).max())
    friction_error=float(abs(capacity-t['friction_capacity_nm']).max())
    elastic_error=float(abs(elastic-t.get('elastic_force_nm',np.zeros_like(elastic))).max())
    sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    seal_capacity=np.zeros((n,count*2))
    for world in range(count*2):
        p=parameters[world%count];state=1.;fs=p.get('seal_static_n',0);fk=p.get('seal_kinetic_n',0)
        for i in range(n):
            desired=np.exp(-(float(t['qvel'][i,world,sv[1]])/p.get('seal_velocity_scale_m_s',.0005))**2)
            state=desired+(state-desired)*np.exp(-.002/(.002 if desired<state else p.get('seal_recovery_s',3.)))
            seal_capacity[i,world]=fk+(fs-fk)*(desired if p.get('seal_friction_route')=='dahl' else state)
    seal_error=float(abs(seal_capacity-t.get('seal_friction_capacity_n',np.zeros_like(seal_capacity))).max())
    seal_bias_expected=np.array([p.get('seal_force_bias_n',0.) for p in parameters]*2)
    seal_bias_error=float(abs(t.get('seal_force_bias_n',np.zeros((n-1,count*2)))-seal_bias_expected).max())
    armature_expected=np.array([model.dof_armature[mv[2:]]*np.broadcast_to(p.get('armature_scale',1.),(2,)) for p in parameters]*2)
    armature_error=float(abs(armature_expected-np.array(report.get('right_armature_kg_m2',armature_expected))).max())
    damping_expected=np.array([p['damping_nms_rad'] for p in parameters]*2)
    seal_damping_expected=np.array([model.dof_damping[sv[1]]+p.get('seal_damping_ns_m',0.) for p in parameters]*2)
    damping_error=float(abs(damping_expected-np.array(report.get('right_motor_damping_nms_rad',damping_expected))).max())
    seal_damping_error=float(abs(seal_damping_expected-np.array(report.get('right_seal_damping_ns_m',seal_damping_expected))).max())
    friction_excess=float(np.max(report.get('native_friction_bound_excess') or [0]))
    tendon_excess=float(np.max(report.get('native_tendon_friction_bound_excess') or [0]))
    bristle_checks={};bristle_verified=True
    if any(p.get('seal_friction_route')=='dahl' for p in parameters):
        with np.load(args.run/'bristle-audit.npz') as source:b={k:source[k] for k in source.files}
        audit=b['audit'];enabled=b['enabled'].astype(bool);assert np.all(b['counts'][enabled]==(n-1)*10)
        pre,delta,cap,post,position,velocity,submitted=[audit[:,:,i].astype(np.float64) for i in range(7)]
        stiffness=b['stiffness_n_m'][None];initial=b['initial_force_n']
        sign=np.sign(delta);target=sign*cap;factor=np.exp(-stiffness*abs(delta)/np.maximum(cap,1e-6))
        predicted=target+(pre-target)*factor
        force_error=float(abs(predicted[:,enabled]-post[:,enabled]).max())
        state_error=float(abs(pre[1:,enabled]-post[:-1,enabled]).max())
        initial_error=float(abs(pre[0,enabled]-initial[enabled]).max())
        inferred_initial=[]
        for world in range(count*2):
            p=parameters[world%count]
            if p.get('seal_friction_route')!='dahl':inferred_initial.append(0.);continue
            if 'seal_initial_force_n' in p:inferred_initial.append(p['seal_initial_force_n']);continue
            q0=a['initial_q'][mq[2:]]-qref[2:];x0,y0=q0@u,q0@w
            extra=np.array(p['bias_delta_nm'])+(p.get('length_stiffness_delta_nm_rad',0)*x0+p.get('length_quadratic_delta_nm_rad2',0)*x0*x0)*u+p.get('cross_stiffness_delta_nm_rad',0)*(x0*w+y0*u)
            residual=initial_residual[2:]-extra
            if p.get('remove_baseline_right_load',False):
                residual+=(np.array(baseline['bias_nm'])+(a['initial_q'][mq]-qref)@np.array(baseline['stiffness_nm_rad']).T)[2:]
            if 'spring_curve_force_n' in p:
                compression=.01-a['initial_q'][sq[1]]
                if p.get('spring_curve_interpolation')=='pchip':
                    from scipy.interpolate import PchipInterpolator
                    knots=p['spring_curve_compression_m'];elastic=PchipInterpolator(knots,p['spring_curve_force_n'])(np.clip(compression,knots[0],knots[-1]))
                else:elastic=np.interp(compression,p['spring_curve_compression_m'],p['spring_curve_force_n'])
                new=np.clip(elastic,0,p.get('spring_force_limit_n',600.))
                old=gas_spring_force(a['initial_q'][sq],np.zeros(2))[1]
                residual+=(new-old)*initial_seal_jac
            f0=residual@initial_seal_jac/(initial_seal_jac@initial_seal_jac)+p.get('seal_force_bias_n',0.)
            inferred_initial.append(np.clip(f0,-p.get('seal_static_n',0),p.get('seal_static_n',0)))
        equilibrium_initial_error=float(abs(initial[enabled]-np.array(inferred_initial)[enabled]).max())
        delta_error=float(abs((position[1:,enabled]-position[:-1,enabled])-delta[1:,enabled]).max())
        control_parity=float(abs(position[::10,enabled]-t['qpos'][:-1,enabled, sq[1]]).max())
        bias=np.array([p.get('seal_force_bias_n',0.) for p in parameters]*2)
        applied_error=float(abs(submitted[:,enabled]-(bias-post)[:,enabled]).max())
        fs=np.array([p.get('seal_static_n',0.) for p in parameters]*2)
        fk=np.array([p.get('seal_kinetic_n',0.) for p in parameters]*2)
        vs=np.array([p.get('seal_velocity_scale_m_s',.0005) for p in parameters]*2)
        desired_age=np.exp(-(velocity/vs)**2);age_error=age_continuity=age_initial_error=0.
        if audit.shape[2]>=9:
            age_pre=audit[:,:,7].astype(np.float64);age_post=audit[:,:,8].astype(np.float64)
            recovery=b['recovery_s'][None];weakening=b['weakening_s'][None]
            time_constant=np.where(desired_age<age_pre,weakening,recovery)
            predicted_age=desired_age+(age_pre-desired_age)*np.exp(-float(b['physics_dt_s'])/time_constant)
            age_error=float(abs(predicted_age[:,enabled]-age_post[:,enabled]).max())
            age_continuity=float(abs(age_pre[1:,enabled]-age_post[:-1,enabled]).max())
            age_initial_error=float(abs(age_pre[0,enabled]-1).max())
            selected_age=np.where(b['aging_enabled'][None].astype(bool),age_post,desired_age)
        else:selected_age=desired_age
        expected_cap=fk+(fs-fk)*selected_age
        cap_error=float(abs(cap[:,enabled]-expected_cap[:,enabled]).max())
        excess=float(np.maximum(abs(post[:,enabled])-fs[enabled],0).max())
        work=cap*abs(delta)+sign*(pre-target)*cap/stiffness*(1-factor)
        loss=work-(post**2-pre**2)/(2*stiffness)
        loss_min=float(loss[:,enabled].min())
        bristle_checks=dict(physics_samples=int(b['counts'][enabled].sum()),force_update_max_n=force_error,
                            state_continuity_max_n=state_error,initial_state_max_n=initial_error,
                            initial_equilibrium_reconstruction_max_n=equilibrium_initial_error,
                            displacement_reconstruction_max_m=delta_error,control_state_parity_max_m=control_parity,
                            applied_force_max_n=applied_error,capacity_max_n=cap_error,force_bound_excess_n=excess,
                            capacity_state_update_max_error=age_error,capacity_state_continuity_max_error=age_continuity,
                            capacity_initial_state_max_error=age_initial_error,
                            constitutive_min_dissipation_j=loss_min)
        bristle_verified=bool(force_error<.001 and state_error<1e-6 and initial_error<.0001 and equilibrium_initial_error<.0001 and delta_error<1e-7
                              and control_parity<1e-7 and applied_error<.001 and cap_error<.001 and excess<.001 and loss_min>=-1e-6
                              and age_error<1e-5 and age_continuity<1e-6 and age_initial_error<1e-6)
    expected_spring=gas_spring_force(t['qpos'][:-1,:,sq],t['qvel'][:-1,:,sv])
    for world in range(count*2):
        p=parameters[world%count]
        if 'right_total_length_load_nm' in p:
            expected_spring[:,world,1]=0.
            continue
        if 'spring_curve_force_n' not in p:continue
        compression=.01-t['qpos'][:-1,world,sq[1]]
        if p.get('spring_curve_interpolation')=='pchip':
            from scipy.interpolate import PchipInterpolator
            knots=p['spring_curve_compression_m'];elastic=PchipInterpolator(knots,p['spring_curve_force_n'])(np.clip(compression,knots[0],knots[-1]))
        else:elastic=np.interp(compression,p['spring_curve_compression_m'],p['spring_curve_force_n'])
        expected_spring[:,world,1]=np.clip(elastic-p.get('spring_velocity_damping_ns_m',50.)*t['qvel'][:-1,world,sv[1]],0,p.get('spring_force_limit_n',600.))
        assert np.allclose(model.actuator('right_gas_spring_motor').ctrlrange,[0,p.get('spring_force_limit_n',600.)])
    spring_error=float(abs(expected_spring-t['spring_force_n']).max())
    closure=length_parity=angle_parity=0
    for i in range(0,n,10):
        for world in range(count*2):
            data.qpos[:]=t['qpos'][i,world];mujoco.mj_kinematics(model,data)
            for side in ('left','right'):
                pairs=[(f'{side}_rear_pin{k}',f'{side}_front_pin{k}') for k in (1,2)]+[(f'{side}_spring_tip',f'{side}_gas_spring_lower')]
                closure=max(closure,max(np.linalg.norm(data.site(x).xpos-data.site(y).xpos) for x,y in pairs))
            delta=data.body('base_link').xmat.reshape(3,3).T@(data.xanchor[model.joint('rfwheel').id]-data.xanchor[model.joint('rf0').id])
            length_parity=max(length_parity,abs(np.linalg.norm(delta[[0,2]])-length[i,world,1]))
            angle_parity=max(angle_parity,abs(np.arctan2(delta[0],-delta[2])-angle[i,world,1]))
    algorithm=Path(identity['source_release'])/'source/deployment/H7_RL_ros/imcalib/Algorithm'
    release=Path(identity['source_release']);manifest=json.loads((release/'firmware-manifest.json').read_text())
    driver=release/'source/deployment/H7_RL_ros/imcalib/drivers/dm.c'
    assert hashlib.sha256(driver.read_bytes()).hexdigest()==manifest['sources']['deployment/H7_RL_ros/imcalib/drivers/dm.c']
    for name in ('joint_capture.c','joint_capture.h','joint_chirp.c','joint_chirp.h','joint_pose_load.h'):
        assert hashlib.sha256((algorithm/name).read_bytes()).hexdigest()==manifest['sources']['deployment/H7_RL_ros/imcalib/Algorithm/'+name]
    f4=ctypes.c_float*4;s6=ctypes.c_int8*6;control_error=0;integral_error=0;quantized_error=0
    with tempfile.TemporaryDirectory() as temp:
        library=Path(temp)/'controller.so'
        source=driver.read_text();quantizer=Path(temp)/'quantizer.c'
        quantizer.write_text('#include <stdint.h>\n'+source[source.index('float Dm_Uint_To_Float('):source.index('/* 计圈数 */')]+\
                             '\nvoid quantize(int n,const float*x,float*y){for(int i=0;i<n;i++)y[i]=Dm_Uint_To_Float(Dm_Float_To_Uint(x[i],-54,54,12),-54,54,12);}\n'
                             'void quantize_velocity(int n,const float*x,float*y){for(int i=0;i<n;i++)y[i]=Dm_Uint_To_Float(Dm_Float_To_Uint(x[i],-45,45,12),-45,45,12);}\n')
        subprocess.run([os.environ.get('CC','cc'),'-std=c11','-O2','-shared','-fPIC',str(algorithm/'joint_capture.c'),str(algorithm/'joint_chirp.c'),str(quantizer),'-lm','-o',str(library)],check=True)
        lib=ctypes.CDLL(str(library));update=lib.Joint_Capture_Update
        quantize=lib.quantize;quantize.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
        def c_quantize(values):
            values=np.ascontiguousarray(values,dtype='f4');result=np.empty_like(values)
            quantize(values.size,values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            return result
        encoder_parity=float(abs(c_quantize(a['recorded_tau'][:,2:])-mit_torque_quantize(a['recorded_tau'][:,2:])).max())
        velocity_quantize=lib.quantize_velocity;velocity_quantize.argtypes=quantize.argtypes
        velocity_test=np.linspace(-45,45,10001,dtype='f4');velocity_result=np.empty_like(velocity_test)
        velocity_quantize(len(velocity_test),velocity_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),velocity_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        velocity_parity=float(abs(velocity_result-mit_velocity_quantize(velocity_test)).max())
        update.argtypes=[ctypes.POINTER(State),ctypes.c_uint32,ctypes.c_uint8,*([ctypes.POINTER(ctypes.c_float)]*3),ctypes.POINTER(ctypes.c_int8),*([ctypes.POINTER(ctypes.c_float)]*3),ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
        for world in range(count,count*2):
            state=State()
            for i in range(n):
                if stops[world] is not None and i>=round(stops[world]/.002):
                    assert not np.any(t['request_model_nm'][i:,world])
                    break
                feedback_q=q[i,world].copy();feedback_v=v[i,world].copy()
                if report['feedback_io']=='measured':
                    for j in (2,3):
                        lag=a['delay_s'][i,j]/.002;older=min(int(lag),2);weight=lag-older
                        feedback_q[j]=(1-weight)*q[max(0,i-older),world,j]+weight*q[max(0,i-older-1),world,j]
                        feedback_v[j]=(1-weight)*v[max(0,i-older),world,j]+weight*v[max(0,i-older-1),world,j]
                    feedback_q=a['initial_sensor_q']+np.rint((feedback_q-a['initial_sensor_q'])/a['position_quantum'])*a['position_quantum']
                    if report.get('sensor_velocity_quantizer')=='mit_floor':feedback_v=mit_velocity_quantize(feedback_v)
                    else:
                        phase=np.sign(a['initial_sensor_v'])*a['velocity_quantum']/2
                        feedback_v=phase+np.rint((feedback_v-phase)/a['velocity_quantum'])*a['velocity_quantum']
                c_output=f4();before=np.array(state.integral)
                assert update(ctypes.byref(state),i*2,1,f4(*(feedback_q*a['sign']+a['zero'])),f4(*(feedback_v*a['sign'])),f4(*(a['initial_request']*a['sign'])),s6(*a['sign'].astype(int)),f4(*a['zero']),f4(60,60,60,60),f4(2,2,2,2),.21,.25,40.,c_output)
                c=np.array(c_output[:])*a['sign'];integral_error=max(integral_error,float(abs(before[2:]-t['integral_nm'][i,world-count]).max()))
                err=a['target_q'][i,2:]-feedback_q[2:];z=np.clip(i*.002/5,0,1)
                takeover=a['initial_request'][2:]+2*a['initial_sensor_v'][2:]-a['feedforward'][0,2:]
                raw=a['feedforward'][i,2:]+60*err+2*(a['target_v'][i,2:]-feedback_v[2:])+t['integral_nm'][i,world-count]+takeover*(1-z**3*(10-15*z+6*z*z))
                control_error=max(control_error,float(abs(c[2:]-np.clip(raw,-40,40)).max()))
                if report['feedback_io']=='measured':c[2:]=c_quantize(c[2:])
                quantized_error=max(quantized_error,float(abs(c[2:]-t['request_model_nm'][i,world,2:]).max()))
    checks=dict(input_identity=report['input_identity'],metrics=metrics,new_physics_steps=0,hardware_access=False,
                recorded_torque_reconstruction_max_nm=torque_error,load_reconstruction_max_nm=load_error,friction_capacity_max_nm=friction_error,
                elastic_force_reconstruction_max_nm=elastic_error,constitutive_min_dissipation_j=elastic_loss_min,
                seal_capacity_reconstruction_max_n=seal_error,armature_parameter_max_error_kg_m2=armature_error,native_friction_bound_excess=friction_excess,
                motor_damping_parameter_max_error=damping_error,seal_damping_parameter_max_error=seal_damping_error,
                seal_force_bias_max_error_n=seal_bias_error,
                native_tendon_friction_bound_excess_n=tendon_excess,
                spring_force_max_n=spring_error,closure_max_mm=closure*1000,geometry_length_max_mm=length_parity*1000,geometry_angle_max_deg=float(np.degrees(angle_parity)),
                archived_dm_encoder_max_nm=encoder_parity,torque_encoder=report.get('torque_encoder','legacy_approximation'),
                archived_dm_velocity_encoder_max_rad_s=velocity_parity,sensor_velocity_quantizer=report.get('sensor_velocity_quantizer','legacy_nearest_initial_phase'),
                archived_c_unquantized_request_max_nm=control_error,archived_c_integral_max_nm=integral_error,archived_c_quantized_request_max_nm=quantized_error)
    checks['seal_bristle']=bristle_checks
    checks['software_verified']=bool(bristle_verified and torque_error<2e-5 and encoder_parity==0 and velocity_parity==0 and report.get('torque_encoder')=='dm_float32' and load_error<2e-5 and friction_error<1e-6 and elastic_error<1e-6 and elastic_loss_min>=-1e-7 and seal_error<1e-6 and seal_bias_error<1e-4 and armature_error<1e-7 and damping_error<1e-6 and seal_damping_error<1e-4 and friction_excess<.001 and tendon_excess<.001 and spring_error<1e-4 and closure<.0001 and length_parity<1e-5 and angle_parity<1e-4 and control_error<.001 and integral_error<.001 and quantized_error<.027)
    checks['response_aligned']=bool(checks['software_verified'] and t['time_s'][-1]>=35 and all(s is None for s in stops) and all(m['length_rmse_mm']<1 and m['length_max_mm']<3 and m['angle_rmse_deg']<.2 for m in metrics))
    checks['response_objective']='length_angle_both_branches'
    if args.length_only:
        checks['response_objective']='length_only_recorded_torque'
        checks['diagnostic_length_thresholds_mm']=dict(rmse=1.,maximum=3.,scope='diagnostic thresholds, not new user acceptance requirements')
        checks['response_aligned']=bool(checks['software_verified'] and t['time_s'][-1]>=35
            and all(stops[w] is None and metrics[w]['length_rmse_mm']<1 and metrics[w]['length_max_mm']<3 for w in range(count)))
    output.write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2));assert checks['software_verified']
    if count==1:
        config=next(Path(p) for p in identity['protected_sources'] if p.endswith('machine_config.c'))
        pose=mirror.MirrorPose(args.inputs/'model.xml',config)
        indices=np.arange(0,n,10);positions=[];rotations=[];states=[]
        for i in indices:
            for side in (0,1):assert pose._solve_leg(side,a['real_q'][i,side*2:side*2+2])
            rows=np.concatenate((pose.data.qpos[None],t['qpos'][i]),axis=0)
            pp=[];rr=[]
            for row in rows:
                data.qpos[:]=row;mujoco.mj_kinematics(model,data)
                pp.append(data.xpos.copy());rr.append(data.xmat.reshape(-1,3,3).copy())
            positions.append(pp);rotations.append(rr);states.append(rows)
        np.savez_compressed(args.run/'comparison.npz',time_s=t['time_s'][indices],qpos=np.array(states),xpos=np.array(positions),xmat=np.array(rotations),
                            length_m=np.column_stack((real_length[indices,1],length[indices,:,1])),
                            angle_rad=np.column_stack((real_angle[indices,1],angle[indices,:,1])),
                            target_length_m=target_length[indices,1],target_angle_rad=target_angle[indices,1],
                            request_nm=np.concatenate((real_input_torque[indices,None],t['request_model_nm'][indices,:,2:]),axis=1)*a['sign'][2:])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    series=[(real_length[:,1]*1000,length[:,:,1]*1000,'Right leg length [mm]'),
            (np.degrees(real_angle[:,1]),np.degrees(angle[:,:,1]),'Right leg angle [deg]'),
            (np.degrees(a['real_q'][:n,2]*a['sign'][2]+a['zero'][2]),np.degrees(q[:,:,2]*a['sign'][2]+a['zero'][2]),'Right front joint [firmware deg]'),
            (np.degrees(a['real_q'][:n,3]*a['sign'][3]+a['zero'][3]),np.degrees(q[:,:,3]*a['sign'][3]+a['zero'][3]),'Right rear joint [firmware deg]')]
    targets=[target_length[:,1]*1000,np.degrees(target_angle[:,1]),np.degrees(a['target_q'][:n,2]*a['sign'][2]+a['zero'][2]),np.degrees(a['target_q'][:n,3]*a['sign'][3]+a['zero'][3])]
    for ax,(real,sim,label),command in zip(axes.flat,series,targets):
        ax.plot(t['time_s'],command,color='green',ls=':',label='Target',lw=.8)
        ax.plot(t['time_s'],real,color='black',label='Real',lw=1.4)
        for world in range(count*2):
            torque_label='Driver torque estimate' if report.get('torque_input_source')=='driver_estimate' else 'Recorded request'
            ax.plot(t['time_s'],sim[:,world],label=(torque_label if world<count else 'Recorded target')+f' #{world%count}',lw=.9)
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_xlim(0,min(35,t['time_s'][-1]))
    axes[0,0].legend();axes[1,0].set_xlabel('Time [s]');axes[1,1].set_xlabel('Time [s]')
    fig.tight_layout();fig.savefig(args.run/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
