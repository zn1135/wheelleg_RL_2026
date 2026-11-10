"""Project measured-motion loads to four motors; no integration or hardware access."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import nnls,least_squares
from scipy.signal import savgol_filter

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def moving_fit(velocity,residual):
    results=[]
    for i in range(4):
        moving=np.abs(velocity[:,i])>.03
        quiet=(np.abs(velocity[:,i])<.005)
        design=np.column_stack((np.sign(velocity[moving,i]),velocity[moving,i]))
        fitted=nnls(design,residual[moving,i])[0] if moving.sum()>3 else [0.,0.]
        error=design@fitted-residual[moving,i]
        results.append(dict(moving_samples=int(moving.sum()),quiet_samples=int(quiet.sum()),
                            moving_coulomb_nm=float(fitted[0]),moving_viscosity_nms_rad=float(fitted[1]),
                            moving_fit_rmse_nm=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                            moving_negative_power_fraction=float(np.mean(residual[moving,i]*velocity[moving,i]<-.01)) if moving.any() else None,
                            quiet_abs_residual_p95_nm=float(np.percentile(np.abs(residual[quiet,i]),95)) if quiet.any() else None,
                            residual_range_nm=np.percentile(residual[:,i],[1,50,99]).tolist()))
    return results


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference',type=Path,required=True,help='Existing GPU reference with measured states and frozen fixed-base model')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.reference/'report.json').read_text())
    assert report['spring_model']=='linear_0_150N_90mm_CAD10mm_damping50'
    meta,_,config=replay.mirror.load_session(Path(report['session']))
    assert meta['stream_sha256']==report['stream_sha256']
    assert hashlib.sha256(config.read_bytes()).hexdigest()==report['config_sha256']
    with np.load(args.reference/'replay.npz') as saved:
        measured=saved['real_motor_state'];requested=saved['submitted_model_torque'][:,0].copy()
        initial=saved['initial_qpos'].copy()
    pose=replay.mirror.MirrorPose(args.reference/'fixed-base-model.xml',config)
    model,data=pose.model,pose.data
    assert model.nq==model.nv and not np.any(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    data.qpos[:]=initial
    mq=np.array([model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]])
    mv=np.array([model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[:4]])
    wheel=[model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[4:]]
    passive=[i for i in range(model.nv) if i not in list(mv)+wheel]
    sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    springs=[model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
    source_model=mujoco.MjModel.from_xml_path(report['source_model'])
    source_mv=[source_model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[:4]]
    # 只用原共享模型的名义负载，不把旧拟合的电机阻尼/反射惯量算进已知负载。
    model.dof_damping[mv]=source_model.dof_damping[source_mv]
    model.dof_armature[mv]=source_model.dof_armature[source_mv]
    model.dof_frictionloss[mv]=0.
    mujoco.mj_setConst(model,mujoco.MjData(model))
    count=round(report['duration_s']/.01)+1
    times=report['window_start_board_s']+np.arange(count)*.01
    measured_q=np.column_stack([np.interp(times,measured[:,0],measured[:,1+i]) for i in range(4)])
    # 零位换算沿现有映射；平滑仅用于离线求导，不改变原数据或自由回放输入。
    smoothed=savgol_filter((measured_q-pose.zero)*pose.sign[:4],11,3,axis=0)
    positions=[];closure=[]
    for q in smoothed:
        for side in range(2):
            assert pose._solve_leg(side,q[2*side:2*side+2]),'Measured pose cannot close'
        positions.append(data.qpos.copy());closure.append(max(pose.residual_m))
    positions=np.asarray(positions)
    velocity=savgol_filter(positions,11,3,deriv=1,delta=.01,axis=0)
    acceleration=savgol_filter(positions,11,3,deriv=2,delta=.01,axis=0)
    control_indices=np.minimum(np.arange(count)*10,len(requested)-1)
    drive=requested[control_indices,:4]
    holding=[];dynamic=[];gravity=[];spring=[];tangents=[];inertia=np.zeros(model.nv)
    tangent_error=[];velocity_error=[]
    for q,v,a in zip(positions,velocity,acceleration):
        data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=0.
        data.ctrl[springs]=replay.training_spring.gas_spring_force(q[sq],v[sv])
        mujoco.mj_forward(model,data)
        mask=data.efc_type[:data.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac=data.efc_J.reshape(data.nefc,model.nv)[mask]
        tangent=np.zeros((model.nv,4));tangent[mv]=np.eye(4)
        tangent[passive],_,rank,_=np.linalg.lstsq(jac[:,passive],-jac[:,mv],rcond=None)
        assert rank==len(passive)
        tangent_error.append(np.max(np.abs(jac@tangent)));velocity_error.append(np.max(np.abs(jac@v)))
        mujoco.mj_mulM(model,data,inertia,a)
        holding.append(tangent.T@(data.qfrc_bias-data.qfrc_passive-data.qfrc_actuator))
        dynamic.append(tangent.T@(inertia+data.qfrc_bias-data.qfrc_passive-data.qfrc_actuator))
        gravity.append(tangent.T@data.qfrc_bias);spring.append(tangent.T@data.qfrc_actuator)
        tangents.append(tangent)
    holding=np.asarray(holding);dynamic=np.asarray(dynamic);residual=drive-dynamic
    assert max(closure)<1e-6 and max(tangent_error)<1e-6
    # 两个坐标分别描述两髋同向转动和反向收伸；此变换严格保持力×位移的功。
    transform=np.kron(np.eye(2),np.array([[.5,.5],[.5,-.5]]))
    coordinate_velocity=velocity[:,mv]@transform.T
    coordinate_residual=residual@np.linalg.inv(transform)
    assert np.allclose(np.sum(residual*velocity[:,mv],axis=1),np.sum(coordinate_residual*coordinate_velocity,axis=1))
    interior=slice(10,-10)
    spring_basis=-np.asarray(tangents)[:,sv,:].transpose(0,2,1)
    base_force=replay.training_spring.gas_spring_force(positions[:,sq],velocity[:,sv])
    elastic_force=np.clip(150.*np.maximum(.01-positions[:,sq],0.)/.09,0.,150.)
    motor_velocity=velocity[:,mv]
    fit_mask=(np.abs(motor_velocity)>.03)
    fit_mask[:10]=fit_mask[-10:]=False
    def load_prediction(parameters):
        # 仅诊断预载假设；保持150N上限、原斜率和50N·s/m阻尼。
        force=np.clip(elastic_force+parameters[:2]-50.*velocity[:,sv],0.,150.)
        extra=np.einsum('tij,tj->ti',spring_basis,force-base_force)
        return extra+np.sign(motor_velocity)*parameters[2:6]+motor_velocity*parameters[6:10]
    fitted=least_squares(lambda x:(load_prediction(x)-residual)[fit_mask],
                         np.r_[100.,100.,np.full(4,1.),np.full(4,.1)],
                         bounds=(np.zeros(10),np.r_[np.full(2,150.),np.full(4,15.),np.full(4,10.)]),
                         loss='soft_l1',f_scale=.5,max_nfev=100)
    extra_force=np.clip(elastic_force+fitted.x[:2]-50.*velocity[:,sv],0.,150.)
    compensated=residual-np.einsum('tij,tj->ti',spring_basis,extra_force-base_force)
    preload_hypothesis=dict(scope='diagnostic candidate only; not measured gas force or adopted physical calibration',
                            preload_n=fitted.x[:2].tolist(),moving_friction_nm=fitted.x[2:6].tolist(),
                            moving_viscosity_nms_rad=fitted.x[6:10].tolist(),physical_fit_evaluations=fitted.nfev,
                            residual_rmse_nm=[float(np.sqrt(np.mean((load_prediction(fitted.x)[fit_mask[:,i],i]-residual[fit_mask[:,i],i])**2))) for i in range(4)],
                            compensated_moving_fit=moving_fit(motor_velocity[interior],compensated[interior]))
    result=dict(scope='Measured-pose inverse-load diagnostic; no free simulation, physical calibration or hardware change',
                stream_sha256=meta['stream_sha256'],reference=str(args.reference),dt_s=.01,smoothing_window_s=.11,
                maximum_pose_closure_m=float(max(closure)),maximum_virtual_work_error=float(max(tangent_error)),
                maximum_velocity_constraint_error=float(max(velocity_error)),
                motor_order=list(replay.mirror.MOTORS[:4]),
                independent_motor_fit=moving_fit(velocity[interior][:,mv],residual[interior]),
                coordinate_order=['left_common','left_difference','right_common','right_difference'],
                common_difference_fit=moving_fit(coordinate_velocity[interior],coordinate_residual[interior]),
                preload_hypothesis=preload_hypothesis,
                environment_steps=0,policy_optimizer_updates=0,
                limitations=['Derivative estimates depend on smoothing and nominal inertia/mass',
                             'Residuals also contain unmodelled support, load, transmission and torque error',
                             'Negative instantaneous residual power can include release of stored elastic friction energy',
                             'This is not identified independent motor friction or measured shaft torque'])
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'inverse-loads.npz',time_s=times-times[0],qpos=positions,qvel=velocity,qacc=acceleration,
                        drive_torque_nm=drive,holding_torque_nm=holding,dynamic_torque_nm=dynamic,residual_nm=residual,
                        gravity_bias_nm=gravity,spring_torque_nm=spring,coordinate_velocity=coordinate_velocity,
                        coordinate_residual_nm=coordinate_residual,motor_indices=mq,
                        spring_basis_nm_per_n=spring_basis,base_spring_force_n=base_force,compensated_residual_nm=compensated)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for i,ax in enumerate(axes):
        ax.plot(times-times[0],drive[:,i],label='Recorded request')
        ax.plot(times-times[0],holding[:,i],label='Nominal bias/spring load')
        ax.plot(times-times[0],dynamic[:,i],label='Nominal load + estimated inertia')
        ax.set_ylabel(replay.mirror.MOTORS[i]+' (Nm)');ax.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('Time in selected trajectory (s)')
    fig.tight_layout();fig.savefig(args.output/'inverse-loads.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
