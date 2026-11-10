"""Passive saturated elastic friction candidate; initial preload uses only the first pose.

df/dq = K*(1-sign(dq)*f/Fc), integrated exactly over each measured simulated displacement.
This is a simplified Dahl-type candidate, not a full LuGre model or a hardware calibration.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize,lsq_linear,minimize_scalar

from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force

MOTORS=('lf0','lf00','rf0','rf00')


def minimum_energy_preload(residual,slide_jacobian,motor_stiffness,seal_stiffness,motor_capacity,seal_capacity,prior=None):
    """Uniquely allocate the first static load between motor and seal elastic states."""
    matrix=np.column_stack((np.eye(4),np.asarray(slide_jacobian)))
    stiffness=np.r_[motor_stiffness,seal_stiffness];capacity=np.r_[motor_capacity,seal_capacity]
    scale=np.sqrt(stiffness);a=matrix*scale[None,:];bounds=capacity/scale
    prior=np.zeros(6) if prior is None else np.asarray(prior,dtype=float)
    assert prior.shape==(6,) and np.isfinite(prior).all()
    center=prior/scale
    initial=np.linalg.lstsq(a,np.asarray(residual),rcond=None)[0]
    result=minimize(lambda x:.5*np.dot(x-center,x-center),np.clip(initial,-bounds,bounds),jac=lambda x:x-center,
                    bounds=list(zip(-bounds,bounds)),constraints=[dict(type='eq',fun=lambda x:a@x-residual,jac=lambda x:a)],
                    method='SLSQP',options=dict(ftol=1e-11,maxiter=100))
    force=result.x*scale
    error=float(np.max(abs(matrix@force-residual)))
    if error>1e-5:
        # 容量不足时保存最小平衡残差，不能伪造满足静力平衡的初态。
        solve=lsq_linear(np.vstack((matrix*1e4,np.diag(1./scale))),np.r_[np.asarray(residual)*1e4,center],bounds=(-capacity,capacity),tol=1e-11)
        force=solve.x;error=float(np.max(abs(matrix@force-residual)))
    return force[:4],force[4:],dict(equilibrium_max_error_nm=error,elastic_energy_j=float(np.sum(force**2/(2*stiffness))),
                                   motor_state_nm=force[:4].tolist(),seal_state_n=force[4:].tolist(),solver_success=bool(result.success),
                                   prior_state=prior.tolist(),correction_energy_j=float(np.sum((force-prior)**2/(2*stiffness))))


def minimum_seal_energy_preload(residual,slide_jacobian,motor_capacity,motor_bias,seal_capacity,prior=None):
    """Smallest seal preload consistent with native motor static-friction intervals."""
    residual=np.asarray(residual)-motor_bias;jac=np.asarray(slide_jacobian);force=np.zeros(2)
    prior=np.zeros(2) if prior is None else np.asarray(prior,dtype=float)
    assert prior.shape==(2,) and np.isfinite(prior).all()
    for side in (0,1):
        low=-seal_capacity[side];high=seal_capacity[side]
        section=slice(2*side,2*side+2);coeff=jac[section,side];target=residual[section];cap=np.asarray(motor_capacity)[section]
        for c,r,f in zip(coeff,target,cap):
            if abs(c)<1e-12:continue
            bounds=sorted(((r-f)/c,(r+f)/c));low=max(low,bounds[0]);high=min(high,bounds[1])
        if low<=high:force[side]=np.clip(prior[side],low,high)
        else:
            result=minimize_scalar(lambda x:np.sum(np.maximum(abs(target-coeff*x)-cap,0.)**2),
                                   bounds=(-seal_capacity[side],seal_capacity[side]),method='bounded')
            force[side]=result.x
    remaining=residual-jac@force;error=float(np.max(np.maximum(abs(remaining)-motor_capacity,0.)))
    return force,dict(equilibrium_max_error_nm=error,native_motor_static_state_nm=remaining.tolist(),seal_state_n=force.tolist(),prior_state_n=prior.tolist())


def initial_preload(model,data,drive_torque,spring_bias=0.,initial_spring_force=None,return_tangent=False):
    assert not np.any(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE), 'Fixed-base diagnostic only'
    mv=[model.joint(n).dofadr[0] for n in MOTORS]
    assert np.max(np.abs(data.qvel[mv]))<.05, 'Initial preload requires a stationary preparation pose'
    wheel=[model.joint(n).dofadr[0] for n in ('lfwheel','rfwheel')]
    passive=[i for i in range(model.nv) if i not in mv+wheel]
    sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    springs=[model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
    original_v=data.qvel.copy();original_ctrl=data.ctrl.copy()
    data.qvel[:]=0.;data.ctrl[:]=0.
    data.ctrl[springs]=gas_spring_force(data.qpos[sq],np.zeros(2),spring_bias) if initial_spring_force is None else initial_spring_force
    mujoco.mj_forward(model,data)
    rows=data.efc_type[:data.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jac=data.efc_J.reshape(data.nefc,model.nv)[rows]
    tangent=np.zeros((model.nv,4));tangent[mv]=np.eye(4)
    solution,_,rank,_=np.linalg.lstsq(jac[:,passive],-jac[:,mv],rcond=None)
    assert rank==len(passive)
    tangent[passive]=solution
    residual=float(np.max(np.abs(jac@tangent)))
    assert residual<1e-6
    hold=tangent.T@(data.qfrc_bias-data.qfrc_passive-data.qfrc_actuator)
    data.qvel[:]=original_v;data.ctrl[:]=original_ctrl
    mujoco.mj_forward(model,data)
    info=dict(constraint_virtual_work_error=residual,holding_torque_nm=hold.tolist())
    if return_tangent:info['tangent']=tangent.tolist()
    return np.asarray(drive_torque)-hold,info


def advance(force,delta,limit,stiffness,capacity=None,rule='dahl'):
    """Return next force and minimum exact friction work minus stored-energy increase."""
    if rule=='plastic':
        # 先弹性变形，达到摩擦容量后才滑动；容量下降释放的储能计为耗散。
        projected=np.clip(force,-limit,limit)
        updated=np.clip(projected+stiffness*delta,-limit,limit)
        slip=delta-(updated-projected)/stiffness
        loss=limit*np.abs(slip)+(force**2-projected**2)/(2*stiffness)
        assert np.isfinite(updated).all() and np.min(loss)>-1e-8
        return updated,float(np.min(loss))
    assert rule=='dahl'
    sign=np.sign(delta);distance=np.abs(delta)
    factor=np.exp(-stiffness*distance/limit)
    target=sign*limit
    updated=target+(force-target)*factor
    work=limit*distance+sign*(force-target)*limit/stiffness*(1-factor)
    stored=(updated**2-force**2)/(2*stiffness)
    loss=work-stored
    bound=limit if capacity is None else capacity
    assert np.isfinite(updated).all() and np.all(np.abs(updated)<=bound+1e-8)
    assert np.min(loss)>-1e-8, 'Elastic resistance would create energy'
    return updated,float(np.min(loss))
