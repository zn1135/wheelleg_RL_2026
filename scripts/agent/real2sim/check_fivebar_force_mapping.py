"""Closed-chain coordinate/virtual-work check; CPU geometry only, no integration."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from .replay_usb_pd_targets import PoseSolver,sha,write


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);solver=PoseSolver();model=solver.model
    shutil.copy2(__file__,out/'check-source.py')
    with np.load(a.inputs) as f:current=f['current'].copy()
    rows=[]
    for index in (0,2500,5000,7500,10000):
        motor=current[index].copy();motor[[2,5]]=0.
        for prefix,motor_ids in (('lf',(0,1)),('rf',(3,4))):
            derivatives=[]
            # 较小步长会放大闭链位置求解的停止误差；用两档中心差分交叉核对解析约束导数。
            for eps in (1e-3,5e-4):
                joint=[];wheel=[]
                for j in motor_ids:
                    samples=[]
                    for direction in (1.,-1.):
                        trial=motor.copy();trial[j]+=direction*eps;pose,error=solver.solve(trial)
                        assert error<1e-7
                        samples.append((pose[[solver.qaddr[prefix+'0'],solver.qaddr[prefix+'1']]],
                            solver.data.body(prefix+'wheel').xpos[[0,2]].copy()))
                    joint.append((samples[0][0]-samples[1][0])/(2*eps))
                    wheel.append((samples[0][1]-samples[1][1])/(2*eps))
                derivatives.append((np.asarray(joint).T,np.asarray(wheel).T))
            # 对闭链有限差分，与主支链解析几何雅可比独立比较。
            q,error=solver.solve(motor);mujoco.mj_comPos(model,solver.data)
            jac=np.zeros((3,model.nv));mujoco.mj_jacBody(model,solver.data,jac,None,model.body(prefix+'wheel').id)
            tree_dofs=[int(model.joint(prefix+n).dofadr[0]) for n in ('0','1')]
            tree_jac=jac[np.ix_([0,2],tree_dofs)];wheel_jac=derivatives[1][1]
            dependent=[int(model.joint(prefix+n).dofadr[0]) for n in ('1','01','02','03')]
            active=[int(model.joint(prefix+n).dofadr[0]) for n in ('0','00')]
            side='left' if prefix=='lf' else 'right';constraints=[]
            for pin in (1,2):
                rear=np.zeros((3,model.nv));front=np.zeros_like(rear)
                mujoco.mj_jacSite(model,solver.data,rear,None,model.site(f'{side}_rear_pin{pin}').id)
                mujoco.mj_jacSite(model,solver.data,front,None,model.site(f'{side}_front_pin{pin}').id)
                constraints.append((rear-front)[[0,2]])
            constraint_jac=np.concatenate(constraints)
            passive_over_motor=np.linalg.solve(constraint_jac[:,dependent],-constraint_jac[:,active])
            coordinate_jac=np.vstack(([1.,0.],passive_over_motor[0]))
            chain_error=float(abs(tree_jac@coordinate_jac-wheel_jac).max())
            eps_error=float(max(abs(derivatives[k][0]-coordinate_jac).max() for k in (0,1)))
            force=np.asarray([10.,100.])
            direct=wheel_jac.T@force;mapped=coordinate_jac.T@(tree_jac.T@force)
            torque_error=float(abs(direct-mapped).max())
            if not (eps_error<1e-4 and chain_error<1e-5 and torque_error<.002):
                write(out/'failure.json',dict(time_s=index*.002,side=prefix,eps_error=eps_error,
                    chain_error=chain_error,torque_error=torque_error,coordinate_jac=coordinate_jac.tolist(),
                    tree_jac=tree_jac.tolist(),wheel_jac=wheel_jac.tolist()))
                raise AssertionError('Geometry/virtual-work check failed; see failure.json')
            actuators=[]
            for name in (prefix+'0',prefix+'00'):
                actuator=model.actuator(name+'_motor');joint=model.joint(name)
                assert model.actuator_trnid[actuator.id,0]==joint.id
                assert np.array_equal(model.actuator_gear[actuator.id],[1.,0.,0.,0.,0.,0.])
                actuators.append(dict(name=actuator.name,joint=joint.name,gear=1.))
            rows.append(dict(time_s=index*.002,side=prefix,coordinate_order=dict(motor=[prefix+'0',prefix+'00'],tree=[prefix+'0',prefix+'1']),
                coordinate_jacobian=coordinate_jac.tolist(),wheel_motor_jacobian_m_rad=wheel_jac.tolist(),
                finite_difference_epsilon_consistency=eps_error,chain_rule_error_m_rad=chain_error,
                test_force_XZ_N=force.tolist(),equivalent_motor_torque_Nm=direct.tolist(),virtual_work_torque_error_Nm=torque_error,
                actuator_transmissions=actuators,closure_m=float(error)))
    write(out/'result.json',dict(passed=True,rows=rows,input_sha256=sha(a.inputs),
        scope='Five sampled poses, CPU kinematics and virtual-work equivalence only. No physical integration, no GPU motion acceptance and no proof of real linkage dimensions.',
        physical_steps=0,training_updates=0))
    shutil.copy2(__file__,out/'check-source.py')
    print(json.dumps(dict(passed=True,samples=len(rows),maximum_chain_rule_error_m_rad=max(r['chain_rule_error_m_rad'] for r in rows),
        maximum_torque_equivalence_error_Nm=max(r['virtual_work_torque_error_Nm'] for r in rows)),indent=2))


if __name__=='__main__':main()
