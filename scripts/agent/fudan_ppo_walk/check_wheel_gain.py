"""Historical pre-limit equivalence only; current response checks use check_wheel_motor_response."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    env=FudanEnv(20,seed=30001,randomized=True)
    try:
        action=env.action;sim=env.env.sim;cfg=action.cfg;wheels=action._wheel_action_ids;legs=action._leg_action_ids
        assert cfg.wheel_stiffness==.045 and cfg.wheel_damping==0.
        strength=action.motor_strength.clone();assert strength.ge(.85).all() and strength.le(1.15).all()
        assert torch.allclose(sim.model.dof_armature[:,action._motor_v_adr[legs]],strength.new_full((20,4),.03159),atol=1e-8,rtol=0)
        dist=replace(cfg.disturbances,action_delay=False,pushes=False)
        requested_error=0.;ctrl_error=0.;leg_error=0.;cases=0
        for target in (-20.,-5.,-1.,0.,1.,5.,20.):
            for speed in (-100.,-20.,0.,20.,100.):
                raw=torch.zeros(20,6,device=env.device);raw[:,wheels]=target
                sim.data.qvel[:,action._motor_v_adr[wheels]]=speed
                action.process_actions(raw)
                action.cfg=replace(cfg,wheel_stiffness=.1,disturbances=dist)
                action.motor_strength.copy_(strength);action.motor_strength[:,wheels]*=.45
                action.apply_actions();env.env.scene.write_data_to_sim()
                old_torque=action._torques.clone();old_ctrl=sim.data.ctrl.clone()
                action.cfg=replace(cfg,disturbances=dist);action.motor_strength.copy_(strength)
                action.apply_actions()
                env.env.scene.write_data_to_sim()
                requested_error=max(requested_error,float((old_torque[:,wheels]-action._torques[:,wheels]).abs().max()))
                leg_error=max(leg_error,float((old_torque[:,legs]-action._torques[:,legs]).abs().max()))
                ctrl_error=max(ctrl_error,float((old_ctrl-sim.data.ctrl).abs().max()));cases+=1
        assert requested_error<1e-5 and ctrl_error<1e-5 and leg_error==0.
        action.reset(torch.tensor([0,3,7],device=env.device))
        assert torch.equal(action.motor_strength,strength)
        result=dict(passed=True,num_envs=20,control_cases=cases,physics_steps=0,model_forwards=0,optimizer_updates=0,
            wheel_p=cfg.wheel_stiffness,wheel_d=cfg.wheel_damping,leg_p=cfg.leg_stiffness,leg_d=cfg.leg_damping,
            requested_wheel_torque_max_error_nm=requested_error,actuator_ctrl_max_error=ctrl_error,leg_torque_max_error_nm=leg_error,
            raw_strength_range=[float(strength.min()),float(strength.max())],partial_reset_preserves_strength=True,
            scope='Actual GPU PD, limits and wheel response equivalence, including saturation; not behavior acceptance')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
