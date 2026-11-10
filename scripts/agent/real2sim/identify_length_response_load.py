"""Estimate right-leg load corrections from observed motion; no physics integration."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from scripts.agent.real2sim.control_leg_pose_gpu import mirror, gas_spring_force


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    identity=json.loads((args.inputs/'report.json').read_text())
    assert hashlib.sha256((args.inputs/'inputs.npz').read_bytes()).hexdigest()==identity['input_npz_sha256']
    config=next(Path(p) for p in identity['protected_sources'] if p.endswith('machine_config.c'))
    with np.load(args.inputs/'inputs.npz') as source:a={k:source[k] for k in source.files}
    baseline=json.loads((args.inputs/'baseline-parameters.json').read_text())
    pose=mirror.MirrorPose(args.inputs/'model.xml',config)
    model,data=pose.model,pose.data
    mq,mv=a['mq'].astype(int),a['mv'].astype(int)
    model.dof_damping[mv]=0
    wheels=[model.joint(n).dofadr[0] for n in mirror.MOTORS[4:]]
    passive=[i for i in range(model.nv) if i not in list(mv)+wheels]
    sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    springs=[model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
    ix=np.arange(0,17501,5);time=a['time_s'][ix];dt=.01
    q=savgol_filter(a['real_q'][ix],11,3,axis=0)
    positions=[];closure=0
    for angles in q:
        for side in (0,1):assert pose._solve_leg(side,angles[side*2:side*2+2])
        closure=max(closure,max(pose.residual_m));positions.append(data.qpos.copy())
    positions=np.array(positions)
    velocity=savgol_filter(positions,11,3,deriv=1,delta=dt,axis=0)
    acceleration=savgol_filter(positions,11,3,deriv=2,delta=dt,axis=0)
    nominal=[];inertia=np.zeros(model.nv);tangent_error=0
    for i,(qfull,vfull,afull) in enumerate(zip(positions,velocity,acceleration)):
        data.qpos[:]=qfull;data.qvel[:]=vfull;data.ctrl[:]=0
        model.opt.gravity[:]=a['gravity'][ix[i]]
        data.ctrl[springs]=gas_spring_force(qfull[sq],vfull[sv])
        mujoco.mj_forward(model,data)
        mask=data.efc_type[:data.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac=data.efc_J.reshape(data.nefc,model.nv)[mask]
        tangent=np.zeros((model.nv,4));tangent[mv]=np.eye(4)
        tangent[passive],_,rank,_=np.linalg.lstsq(jac[:,passive],-jac[:,mv],rcond=None)
        assert rank==len(passive)
        tangent_error=max(tangent_error,float(abs(jac@tangent).max()))
        mujoco.mj_mulM(model,data,inertia,afull)
        nominal.append(tangent.T@(inertia+data.qfrc_bias-data.qfrc_passive-data.qfrc_actuator))
    nominal=np.array(nominal)
    known=np.array(baseline['bias_nm'])+(q-np.array(baseline['reference_q_model_rad']))@np.array(baseline['stiffness_nm_rad']).T
    residual=a['recorded_tau'][ix]-nominal-known
    u=np.array([-1.,1.])/np.sqrt(2);w=np.array([1.,1.])/np.sqrt(2)
    dq=q[:,2:]-np.array(baseline['reference_q_model_rad'])[2:]
    x,y=dq@u,dq@w
    v=velocity[:,mv[2:]]
    design=np.zeros((len(q),2,8))
    design[:,0,0]=1;design[:,1,1]=1
    design[:,:,2]=x[:,None]*u
    design[:,:,3]=x[:,None]*w+y[:,None]*u
    for j in (0,1):
        design[:,j,4+j]=np.tanh(v[:,j]/.01)
        design[:,j,6+j]=v[:,j]
    moving=(time[:,None]>5)&(time[:,None]<30)&(abs(v)>.03)
    matrix,target=design[moving],residual[:,2:][moving]
    low=[-10,-10,-40,-40,0,0,0,0];high=[10,10,40,40,10,10,30,30]
    weight=np.ones(len(target));regularizer=np.diag([.005,.005,.01,.01,.01,.01,.02,.02])
    for _ in range(6):
        fit=lsq_linear(np.vstack((matrix*weight[:,None],regularizer)),np.r_[target*weight,np.zeros(8)],bounds=(low,high),tol=1e-11)
        error=matrix@fit.x-target;weight=np.sqrt(np.minimum(1,.15/np.maximum(abs(error),1e-12)))
    b0,b1,kl,kc,f0,f1,d0,d1=fit.x
    unexplained=residual[:,2:]-(design[:,:,:4]@fit.x[:4])
    still=(time[:,None]>1)&(abs(v)<.01)
    capacity=[max(fit.x[4+j],np.percentile(abs(unexplained[still[:,j],j]),99)) for j in (0,1)]
    candidate=dict(static_nm=capacity,kinetic_nm=[f0,f1],velocity_scale_rad_s=[.02,.02],damping_nms_rad=[d0,d1],bias_delta_nm=[b0,b1],length_stiffness_delta_nm_rad=kl,cross_stiffness_delta_nm_rad=kc)
    report=dict(scope='inverse-motion seed, not free-running simulation validation',samples=len(q),input_identity=identity['input_npz_sha256'],new_physics_steps=0,
                closure_max_m=closure,tangent_max_error=tangent_error,moving_points=moving.sum(0).tolist(),moving_residual_rmse_nm=float(np.sqrt(np.mean(error**2))),candidate=candidate)
    np.savez_compressed(args.output/'inverse.npz',time_s=time,q=q,v=v,residual_nm=residual,design=design,moving=moving,parameters=fit.x)
    (args.output/'parameters.json').write_text(json.dumps([candidate],indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
