"""Compare native/Warp force and acceleration evaluation at identical states; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--inverse',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_xml_path(str(args.reference/'fixed-base-model.xml'))
    model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    mv=[model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[:4]]
    sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    motors=[model.actuator(n+'_motor').id for n in replay.mirror.MOTORS[:4]]
    springs=[model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
    model.dof_armature[mv]=.015795;model.dof_damping[mv]=.2;model.dof_frictionloss[mv]=1e-6
    model.dof_solref[mv]=[.002,1.];model.dof_solimp[mv]=[.999,.9999,.001,.5,2.]
    mujoco.mj_setConst(model,mujoco.MjData(model))
    source=np.load(args.inverse/'inverse-loads.npz');indices=np.unique(np.linspace(0,len(source['qpos'])-1,21).astype(int))
    q=np.tile(source['qpos'][indices],(3,1));v=np.tile(source['qvel'][indices],(3,1));n=len(q)
    controls=np.zeros((n,model.nu));controls[:,motors]=np.tile(source['drive_torque_nm'][indices],(3,1))
    controls[:,springs]=np.clip(150.-50.*v[:,sv],0,150)
    friction=np.tile(model.dof_frictionloss,(n,1));friction[:,mv]=np.repeat([0.,1.,8.],len(indices))[:,None]
    names=['qacc','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint'];native={name:[] for name in names}
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    for i in range(n):
        model.dof_frictionloss[:]=friction[i]
        data=mujoco.MjData(model);data.qpos[:]=q[i];data.qvel[:]=v[i];data.ctrl[:]=controls[i]
        mujoco.mj_forward(model,data)
        for name in names:native[name].append(getattr(data,name).copy())
    model.dof_frictionloss[mv]=1e-6
    import mujoco_warp as mw
    import warp as wp
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        gm=mw.put_model(model,batch_sizes={'dof_frictionloss':n});gm.dof_frictionloss.assign(friction.astype(np.float32))
        data=mujoco.MjData(model);gd=mw.put_data(model,data,nworld=n,nconmax=128,njmax=256)
        gd.qpos.assign(q.astype(np.float32));gd.qvel.assign(v.astype(np.float32));gd.ctrl.assign(controls.astype(np.float32))
        mw.forward(gm,gd)
        assert not np.any(gd.time.numpy())
        gpu={name:getattr(gd,name).numpy().copy() for name in names}
    result={}
    for name in names:
        native[name]=np.asarray(native[name]);error=gpu[name]-native[name]
        result[name]=dict(max_abs_error=float(np.max(abs(error))),motor_rmse_error=np.sqrt(np.mean(error[:,mv]**2,axis=0)).tolist(),
                          motor_native_rms=np.sqrt(np.mean(native[name][:,mv]**2,axis=0)).tolist())
    report=dict(scope='identical-state force/acceleration parity diagnostic; not a trajectory or hardware acceptance',
                state_count=n,friction_cases_nm=[0.,1.,8.],environment_steps=0,source_sha256=hashlib.sha256((args.inverse/'inverse-loads.npz').read_bytes()).hexdigest(),
                results=result,versions=dict(mujoco=mujoco.__version__,warp=wp.__version__))
    np.savez_compressed(args.output/'comparison.npz',qpos=q,qvel=v,controls=controls,friction=friction,**{'native_'+k:z for k,z in native.items()},**{'gpu_'+k:z for k,z in gpu.items()})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
