"""Actual GPU control writeback for independent wheel response; zero physics/policy updates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    with patch.object(FudanWalkCfg.control,'leg_stiffness',60.),patch.object(FudanWalkCfg.sim,'wheel_model','legacy_curve'):env=FudanEnv(20,seed=30001,randomized=False)
    checks={};values={}
    def verify(name,condition):checks[name]=bool(condition);assert checks[name],name
    try:
        a=env.action;sim=env.env.sim;wheels=a._wheel_action_ids;legs=a._leg_action_ids
        motor=a.wheel_motor;cfg=a.cfg
        a.cfg=replace(cfg,disturbances=replace(cfg.disturbances,action_delay=False,pushes=False))
        base_cfg=a.cfg;motors=[sim.mj_model.actuator('robot/'+n+'_motor').id for n in ('lfwheel','rfwheel')]
        verify('wheel_PD_restored',a.cfg.wheel_stiffness==.1 and a.cfg.wheel_damping==0.)
        verify('leg_PD_preserved',(a.cfg.leg_stiffness,a.cfg.leg_damping)==(60.,2.))
        verify('default_response_bypass',not motor.enabled)
        def submit(target,speed=0.,legacy=False):
            raw=torch.zeros(20,6,device=env.device);raw[:,wheels]=target
            sim.data.qvel[:,a._motor_v_adr[wheels]]=speed;a.process_actions(raw)
            if legacy:HuananhuPdAction.apply_actions(a)
            else:a.apply_actions()
            env.env.scene.write_data_to_sim()
            return a._torques.clone(),sim.data.ctrl[:,motors].clone()
        default_error=0.;post_error=0.;leg_error=0.;prelimit_error=0.
        strength=a.motor_strength.clone()
        for target in (-20.,-1.,-.1,0.,.1,1.,20.):
            for speed in (-100.,0.,100.):
                motor.configure();old_torque,old_ctrl=submit(target,speed,legacy=True)
                torque,ctrl=submit(target,speed);default_error=max(default_error,float(abs(ctrl-old_ctrl).max()))
                motor.configure(gain=.45);torque,ctrl=submit(target,speed)
                post_error=max(post_error,float(abs(ctrl-old_ctrl*.45).max()))
                leg_error=max(leg_error,float(abs(torque[:,legs]-old_torque[:,legs]).max()))
                motor.configure();a.cfg=replace(base_cfg,wheel_stiffness=.045)
                _,lower_p_ctrl=submit(target,speed)
                a.cfg=base_cfg;a.motor_strength[:,wheels]=strength[:,wheels]*.45
                _,prelimit_ctrl=submit(target,speed);a.motor_strength.copy_(strength)
                prelimit_error=max(prelimit_error,float(abs(lower_p_ctrl-prelimit_ctrl).max()))
        verify('identity_writeback_exact',default_error==0.)
        verify('gain_after_limit_actual_writeback',post_error<1e-6)
        verify('leg_torque_unchanged',leg_error==0.)
        verify('prelimit_strength_equals_lowerP_at_actual_writeback',prelimit_error<1e-6)
        motor.configure();a.cfg=replace(base_cfg,wheel_stiffness=.045)
        lower_p_request,lower_p=submit(20.)
        a.cfg=base_cfg;motor.configure(gain=.45);post_request,post=submit(20.)
        verify('saturated_PD_request_same',torch.equal(lower_p_request[:,wheels],post_request[:,wheels]))
        verify('post_gain_differs_from_lower_PD_when_saturated',torch.allclose(post,lower_p*.45,atol=1e-6) and not torch.allclose(post,lower_p))
        values['saturated_nominal_PD_request_nm']=post_request[0,wheels].cpu().tolist()
        values['lower_PD_written_ctrl']=lower_p[0].cpu().tolist();values['post_gain_written_ctrl']=post[0].cpu().tolist()
        cap=a._effort_limits[wheels]*a._wheel_response[:,0]*.45
        motor.configure(torque_limit_nm=cap);_,capped=submit(20.)
        verify('shaft_output_cap_honored',torch.allclose(capped*a._wheel_response[:,0],cap.expand_as(capped),atol=1e-6))
        _,cap_small=submit(.1);motor.configure(gain=.45);_,gain_small=submit(.1)
        verify('gain_and_cap_are_distinct_at_small_request',torch.allclose(gain_small,cap_small*.45,atol=1e-6))
        motor.configure(delay_ms=4.);pulse=[]
        for i in range(5):_,c=submit(.1 if i==0 else 0.);pulse.append(c)
        verify('4ms_delay_two_physics_ticks',pulse[0].eq(0).all() and pulse[1].eq(0).all() and pulse[2].gt(0).all() and pulse[3].eq(0).all())
        motor.configure(time_constant_ms=4.);alpha=1-torch.exp(torch.tensor(-.5));expected=torch.zeros(20,2,device=env.device);err=0.
        for _ in range(10):
            request,c=submit(.1);expected=expected+alpha*(request[:,wheels]-expected)
            err=max(err,float(abs(c-expected).max()))
        verify('first_order_response_formula',err<1e-6)
        motor.configure(gain=.6,delay_ms=4.,time_constant_ms=4.)
        for _ in range(5):submit(.1)
        state=motor.state.clone();history=motor.history.clone();head=motor.head
        ids=torch.tensor([0,3,7],device=env.device);other=torch.tensor([1,2,4],device=env.device)
        a.reset(ids)
        verify('selected_response_state_cleared',motor.state[ids].eq(0).all() and motor.history[:,ids].eq(0).all())
        verify('other_worlds_preserved',torch.equal(motor.state[other],state[other]) and torch.equal(motor.history[:,other],history[:,other]) and head==motor.head)
        verify('response_configuration_survives_reset',motor.gain.eq(.6).all() and motor.delay_steps.eq(2).all())
        fixture=motor.fixture();verify('fixture_parameters_finite',all(torch.isfinite(x).all() for x in fixture.values()))
        values.update(identity_ctrl_max_error=default_error,post_gain_ctrl_max_error=post_error,leg_torque_max_error=leg_error,
            prelimit_vs_lowerP_actual_ctrl_max_error=prelimit_error,
            time_constant_max_error=err,xml_wheel_gains=a._wheel_response[:,0].cpu().tolist())
        result=dict(passed=all(checks.values()),checks=checks,actual=values,num_envs=20,physics_steps=0,
            policy_forwards=0,optimizer_updates=0,scope='Actual GPU PD/response/writeback and reset integration; not physical calibration or behavior acceptance')
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
