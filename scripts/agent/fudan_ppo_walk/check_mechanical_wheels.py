"""Verify unit wheel torque, native mechanical fields and finite GPU integration."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.model import load_spec
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv,make_env_cfg
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.envs.fudan_ppo_walk.wheel_physics import mechanical_spec,record


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    old=load_spec().compile();new=mechanical_spec().compile();params=record('mechanical');checks={}
    def check(name,value):checks[name]=bool(value);assert checks[name],name
    w=np.array([new.joint(n).dofadr[0] for n in ('lfwheel','rfwheel')])
    actuators=np.array([new.actuator(n+'_motor').id for n in ('lfwheel','rfwheel')])
    for field in ('body_mass','body_inertia','body_ipos','jnt_axis','jnt_range','eq_data','actuator_ctrlrange','actuator_gear'):
        check('preserved_'+field,np.array_equal(getattr(old,field),getattr(new,field)))
    for field in ('armature','damping','frictionloss'):
        check('mechanical_'+field,np.allclose(getattr(new,'dof_'+field)[w],params[field],atol=1e-12,rtol=0))
        check('nonwheel_'+field,np.array_equal(np.delete(getattr(old,'dof_'+field),w,axis=0),np.delete(getattr(new,'dof_'+field),w,axis=0)))
    check('unit_native_gain',np.all(new.actuator_gainprm[actuators,0]==1.))
    cfg=FudanWalkCfg();cfg.sim.wheel_motor_gain=.45
    try:make_env_cfg(task_cfg=cfg)
    except ValueError:check('reject_output045',True)
    else:check('reject_output045',False)
    with patch.object(FudanWalkCfg.control,'leg_stiffness',60.),patch.object(FudanWalkCfg.control,'wheel_stiffness',.045):
        cfg=FudanWalkCfg()
        clean=replace(from_task_config(cfg),action_delay=False,sensor_delay=False,pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
        env=FudanEnv(20,seed=30001,randomized=False,disturbances=clean,cfg=cfg)
    try:
        action=env.action;sim=env.env.sim;ids=action._wheel_action_ids;dofs=action._motor_v_adr[ids]
        acts=[sim.mj_model.actuator('robot/'+n+'_motor').id for n in ('lfwheel','rfwheel')]
        check('no_speed_curve',not action._wheel_speed_shaping)
        check('no_output_layer',not action.wheel_motor.enabled and action.wheel_motor.gain.eq(1).all())
        error=0.;forces=[]
        for speed in (-100.,-10.,0.,10.,100.):
            for requested in (-2.,0.,2.):
                sim.data.qvel[:,dofs]=speed
                raw=torch.zeros(20,6,device=env.device);raw[:,ids]=(speed+requested/.045)/10.
                action.process_actions(raw);action.apply_actions();env.env.scene.write_data_to_sim();sim.forward()
                control=sim.data.ctrl[:,acts];force=sim.data.actuator_force[:,acts]
                error=max(error,float(abs(control-requested).max()),float(abs(force-requested).max()))
                forces.append(dict(speed_rad_s=speed,requested_nm=requested,actual_nm=force[0].cpu().tolist()))
        check('unit_force_all_speeds_signs',error<1e-5)
        env.reset();action.reset(torch.tensor([0,3,7],device=env.device))
        check('mechanics_survive_reset',torch.allclose(sim.model.dof_armature[:,dofs],torch.tensor(params['armature'],device=env.device).expand_as(sim.model.dof_armature[:,dofs]),atol=1e-8,rtol=0))
        for _ in range(100):
            env.step(torch.zeros(20,6,device=env.device),auto_reset=True,resample=False)
            assert torch.isfinite(sim.data.qpos).all() and torch.isfinite(sim.data.qvel).all()
        check('finite_500_GPU_steps',True)
        result=dict(passed=all(checks.values()),checks=checks,parameters=params,force_probes=forces,max_force_error_nm=error,
            num_envs=20,explicit_forward_force_probes=15,explicit_physics_steps=500,policy_forwards=0,optimizer_updates=0,
            scope='Control/physics integration only; behavior scored separately')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
