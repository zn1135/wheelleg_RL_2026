"""Inverse-motion seeds for monotone axial curves, with gravity/inertia kept physical."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from scripts.agent.real2sim.control_leg_pose_gpu import mirror
from scripts.agent.real2sim.replay_length_response_gpu import mit_torque_quantize


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    identity=json.loads((args.inputs/'report.json').read_text())
    assert hashlib.sha256((args.inputs/'inputs.npz').read_bytes()).hexdigest()==identity['input_npz_sha256']
    with np.load(args.inputs/'inputs.npz') as src:a={k:src[k] for k in src.files}
    config=next(Path(p) for p in identity['protected_sources'] if p.endswith('machine_config.c'))
    pose=mirror.MirrorPose(args.inputs/'model.xml',config);m,d=pose.model,pose.data
    mq,mv=a['mq'].astype(int),a['mv'].astype(int)
    wheels=[m.joint(n).dofadr[0] for n in mirror.MOTORS[4:]]
    passive=[i for i in range(m.nv) if i not in list(mv)+wheels]
    sq=m.joint('right_spring2_joint').qposadr[0];sv=m.joint('right_spring2_joint').dofadr[0]
    m.dof_damping[mv]=0.;m.geom_contype[:]=m.geom_conaffinity[:]=0
    ix=np.arange(0,17501,10);dt=.02
    q=savgol_filter(a['real_q'][ix],11,3,axis=0)
    positions=[]
    for row in q:
        for side in (0,1):assert pose._solve_leg(side,row[side*2:side*2+2])
        positions.append(d.qpos.copy())
    positions=np.array(positions)
    velocity=savgol_filter(positions,11,3,deriv=1,delta=dt,axis=0)
    acceleration=savgol_filter(positions,11,3,deriv=2,delta=dt,axis=0)
    loads=[];axis=[];max_closure=0;mass_force=np.zeros(m.nv)
    for i in range(len(ix)):
        d.qpos[:]=positions[i];d.qvel[:]=velocity[i];d.ctrl[:]=0;m.opt.gravity[:]=a['gravity'][ix[i]]
        mujoco.mj_forward(m,d)
        eq=d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac=d.efc_J.reshape(d.nefc,m.nv)[eq]
        tangent=np.zeros((m.nv,4));tangent[mv]=np.eye(4)
        tangent[passive],_,rank,_=np.linalg.lstsq(jac[:,passive],-jac[:,mv],rcond=None)
        assert rank==len(passive)
        max_closure=max(max_closure,float(abs(jac@tangent).max()))
        mujoco.mj_mulM(m,d,mass_force,acceleration[i])
        loads.append((tangent.T@(mass_force+d.qfrc_bias-d.qfrc_passive))[2:])
        axis.append(tangent[sv,2:])
    axis=np.array(axis);required=mit_torque_quantize(a['recorded_tau'][ix,2:])-loads
    compression=.01-positions[:,sq];slide_v=velocity[:,sv];motor_v=velocity[:,mv[2:]]
    mask=(a['time_s'][ix]>8)&(a['time_s'][ix]<26)&(abs(slide_v)>.0005)
    knots=np.array([.027,.031,.035,.039,.0435])
    basis=np.column_stack((np.ones(len(ix)),*[np.clip((compression-knots[j-1])/(knots[j]-knots[j-1]),0,1) for j in range(1,len(knots))]))
    original=json.loads(Path('artifacts/length-response-fit/20261109-bristle-final-full-01/parameters.json').read_text())[0]
    candidates=[original];reports=[]
    for seal_damping in (200.,1000.,2000.,4000.):
        known=.5*np.sign(motor_v)+np.asarray(original['damping_nms_rad'])*motor_v
        known+=axis*(55.5*np.sign(slide_v)+(seal_damping+50.)*slide_v)[:,None]
        target=required-known
        design=-axis[:,:,None]*basis[:,None,:]
        matrix=design[mask].reshape(-1,len(knots));rhs=target[mask].ravel()
        # Nonnegative increments give a monotone force curve without a fixed linear slope.
        regular=np.zeros((len(knots)-2,len(knots)))
        for j in range(len(knots)-2):regular[j,j+1:j+3]=[.002,-.002]
        fit=lsq_linear(np.vstack((matrix,regular)),np.r_[rhs,np.zeros(len(regular))],
            bounds=(np.zeros(len(knots)),np.r_[400.,np.full(len(knots)-1,200.)]),tol=1e-10)
        values=np.minimum(np.cumsum(fit.x),600.)
        for scale in (.9,1.,1.1):
            p=dict(original,remove_baseline_right_load=True,bias_delta_nm=[0.,0.],
                length_stiffness_delta_nm_rad=0.,cross_stiffness_delta_nm_rad=0.,length_quadratic_delta_nm_rad2=0.,
                seal_damping_ns_m=seal_damping,spring_curve_compression_m=knots.tolist(),
                spring_curve_force_n=(values*scale).clip(0,600).tolist(),spring_velocity_damping_ns_m=50.,spring_force_limit_n=600.)
            candidates.append(p)
        prediction=-axis*np.interp(compression,knots,values)[:,None]
        transverse=target-axis*np.sum(target*axis,axis=1)[:,None]/np.sum(axis*axis,axis=1)[:,None]
        reports.append(dict(seal_damping_ns_m=seal_damping,curve_force_n=values.tolist(),
            inverse_residual_rmse_nm=float(np.sqrt(np.mean((prediction[mask]-target[mask])**2))),
            axial_unexplainable_component_rmse_nm=float(np.sqrt(np.mean(transverse[mask]**2)))))
    np.savez_compressed(args.output/'inverse.npz',time_s=a['time_s'][ix],qpos=positions,qvel=velocity,
        qacc=acceleration,axis_jacobian=axis,required_motor_nm=required,compression_m=compression,moving=mask)
    (args.output/'parameters.json').write_text(json.dumps(candidates,indent=2)+'\n')
    report=dict(input_identity=identity['input_npz_sha256'],samples=len(ix),moving_samples=int(mask.sum()),
        compression_range_m=[float(compression.min()),float(compression.max())],closure_virtual_work_max=max_closure,
        trials=reports,physics_steps=0,policy_training_updates=0,
        limitations=['220ms smooth differentiation is only a seed; final scores use free GPU integration',
            'Retraction alone cannot uniquely split axial pressure and Coulomb friction',
            'Curve is constant outside observed knots; no global gas spring specification inferred'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
