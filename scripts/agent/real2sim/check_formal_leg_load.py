"""Verify the adopted load against archived applied forces and native/GPU lifecycles."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fitted_leg_load import (
    parameters, coefficients, conservative_load, memory_step, NativeLegLoad, parameter_record)
from ct_mjlab.envs.chuanliantui.model import MODEL_XML


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    with np.load('artifacts/length-response-fit/20261109-fudan8000-right-load-01/candidate-both/response-batch-0/response-audit.npz') as src:
        a=src['audit'][::7].astype(float)
    bundle=parameters();p=bundle['candidate'];c=coefficients(bundle,np.asarray);sign=np.array([-1.,1.])
    q=a[...,:2]*sign[None,None,:,None]
    native_load=conservative_load(q,c,p)
    old_load_error=float(abs(native_load-a[...,10:12]).max())
    memory=memory_step(a[...,4],a[...,5],a[...,4]-a[...,6],a[...,7],p)
    old_memory_error=float(abs(memory-a[...,9]).max())
    backend=[]
    for device in ('cpu','cuda:0'):
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        tc=coefficients(bundle,tensor)
        applied=conservative_load(tensor(q[:10]),tc,p).cpu().numpy()
        force=memory_step(tensor(a[:10,...,4]),tensor(a[:10,...,5]),
            tensor(a[:10,...,4]-a[:10,...,6]),tensor(a[:10,...,7]),p).cpu().numpy()
        backend.append(dict(device=device,load_max_nm=float(abs(applied-native_load[:10]).max()),
            memory_max_n=float(abs(force-memory[:10]).max())))
    model=mujoco.MjModel.from_xml_path(str(MODEL_XML));data=mujoco.MjData(model)
    native=NativeLegLoad(model,data)
    qi,sv=native.qi,native.sv
    data.qpos[qi]=q[0,0];data.qpos[native.sq]=a[0,0,:,4];data.qvel[sv]=a[0,0,:,5]
    native.apply()
    first_force=float(abs(native.force).max())
    data.qpos[qi]=q[1,0];data.qpos[native.sq]=a[1,0,:,4];data.qvel[sv]=a[1,0,:,5]
    native.apply()
    native_load_error=float(abs(data.qfrc_applied[native.vi]-conservative_load(q[1,0],c,p)).max())
    native.reset();data.qpos[native.sq]+=.05;native.apply()
    reset_force=float(abs(native.force).max())
    # Actual startup randomization must preserve an active default load and finite forces.
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
    from ct_mjlab.envs.chuanliantui.huananhu_env_cfg import make_huananhu_env_cfg
    cfg=make_huananhu_env_cfg(play=False)
    assert cfg.actions['pd'].fitted_leg_load
    wrapper=FudanEnv(20,randomized=True,disturbances=DisturbanceConfig())
    env=wrapper.env
    try:
        wrapper.reset();action=wrapper.action;assert action.leg_load is not None
        from ct_mjlab.envs.chuanliantui.huananhu_randomization import randomize_at_startup
        randomize_at_startup(env,None)
        dry=env.sim.model.dof_frictionloss[:,action.leg_load.vi]-torch.tensor(p['static_nm'],device='cuda:0')
        viscous=env.sim.model.dof_damping[:,action.leg_load.vi]-torch.tensor(p['damping_nms_rad'],device='cuda:0')
        randomization_ok=bool((dry>=.25-1e-6).all() and (dry<=1.+1e-6).all()
            and (viscous>=.05-1e-6).all() and (viscous<=.2+1e-6).all())
        assert randomization_ok
        wrapper.step(torch.zeros(20,6,device='cuda:0'),auto_reset=False,resample=False)
        training_finite=bool(torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(action.leg_load.force).all())
        training_profile=parameter_record(action.leg_load is not None)
    finally:env.close()
    result=dict(profile=parameter_record(),archived_load_max_nm=old_load_error,
        archived_memory_max_n=old_memory_error,backends=backend,native_applied_load_max_nm=native_load_error,
        native_initial_memory_max_n=first_force,native_reset_after_pose_change_max_n=reset_force,
        training_environment_finite=training_finite,training_profile=training_profile,
        randomization_around_fitted_baseline=randomization_ok,
        scope='Archived-force parity, native algebra/reset and actual20-world Fudan training-config GPU smoke; common Hu ground-reset behavior not tested',
        policy_training_updates=0,hardware_commands=0)
    result['passed']=bool(old_load_error<1e-4 and old_memory_error<.001 and native_load_error<1e-9
        and first_force==reset_force==0 and training_finite
        and all(x['load_max_nm']<1e-4 and x['memory_max_n']<.001 for x in backend))
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']


if __name__=='__main__':main()
