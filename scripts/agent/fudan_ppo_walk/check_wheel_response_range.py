"""GPU writeback, step response and partial reset checks for episode wheel tau."""
import argparse
import json
import math
from pathlib import Path

import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
from ct_mjlab.envs.fudan_ppo_walk.wheel_motor import response_record,configure_response


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.sim.wheel_motor_time_constant_range_ms=[25.,35.]
    record=response_record(cfg);restored=FudanWalkCfg();configure_response(restored,record)
    assert response_record(restored)==record
    configure_response(restored,None);assert restored.sim.wheel_motor_time_constant_range_ms is None
    env=FudanEnv(20,seed=30001,randomized=False,disturbances=DisturbanceConfig(),cfg=cfg)
    checks={}
    def check(name,value):checks[name]=bool(value);assert checks[name],name
    try:
        a=env.action;sim=env.env.sim;motor=a.wheel_motor;wheel=a._wheel_action_ids;legs=a._leg_action_ids
        acts=[sim.mj_model.actuator('robot/'+n+'_motor').id for n in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        tau=motor.tau_s.clone();check('initial_tau_range',((tau>=.025)&(tau<=.035)).all())
        check('unit_gain_no_curve',motor.gain.eq(1).all() and not a._wheel_speed_shaping)
        sim.data.qvel.zero_();raw=torch.zeros(20,6,device=env.device);raw[:,wheel]=1.
        error=0.;leg_error=0.
        for tick in range(1,151):
            a.process_actions(raw);a.apply_actions();env.env.scene.write_data_to_sim();sim.forward()
            expected=1.-torch.exp(-tick*.002/tau)
            error=max(error,float(abs(sim.data.actuator_force[:,acts][:,wheel]-expected[:,None]).max()))
            leg_error=max(leg_error,float(abs(sim.data.ctrl[:,acts][:,legs]-a._torques[:,legs]).max()))
        check('independent_step_formula',error<2e-6)
        check('leg_writeback_unchanged',leg_error==0.)
        check('tau_fixed_during_episode',torch.equal(tau,motor.tau_s))
        keep=motor.state.clone();ids=torch.tensor([0,3,7],device=env.device);other=torch.tensor([i for i in range(20) if i not in (0,3,7)],device=env.device)
        a.reset(ids)
        check('reset_only_selected_tau',torch.equal(tau[other],motor.tau_s[other]) and not torch.equal(tau[ids],motor.tau_s[ids]))
        check('reset_only_selected_state',motor.state[ids].eq(0).all() and torch.equal(keep[other],motor.state[other]))
        check('resampled_bounds',((motor.tau_s>=.025)&(motor.tau_s<=.035)).all())
        for options in (dict(gain=.45),dict(delay_ms=30.),dict(torque_limit_nm=1.)):
            try:motor.configure(**options)
            except ValueError:pass
            else:raise AssertionError('Mechanical restriction removed: '+str(options))
        # 固定0ms必须精确旁路；固定30ms的阶跃无需纯延迟即可立即产生小力矩。
        motor.configure();check('old_zero_response_bypass',not motor.enabled)
        motor.configure(time_constant_ms=30.);result=motor.apply(torch.ones(20,2,device=env.device),torch.zeros(20,2,device=env.device))
        check('fixed_tau_first_tick',torch.allclose(result,torch.full_like(result,1-math.exp(-.002/.03)),atol=1e-7))
        check('metadata_roundtrip_and_legacy',response_record(restored)['time_constant_range_ms'] is None)
        report=dict(passed=True,checks=checks,step_response_max_error_Nm=error,leg_writeback_error_Nm=leg_error,
            wheel_response=record,num_envs=20,physics_steps=0,policy_forwards=0,optimizer_updates=0,
            scope='GPU production control/actuator forward writes and reset behavior, not integrated motion acceptance')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':main()
