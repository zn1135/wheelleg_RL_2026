#!/usr/bin/env python3
"""Check getup torque penalties before standing and unchanged weights/other rewards; no learning."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import rewards, runner
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_stand_v2 import rewards as shared
from scripts.agent.split.check_getup_nominal_state import source_module
from scripts.agent.split.check_chuanliantui_two_policy import rejected, digest


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    before=source_module(args.before_shared,'ct_mjlab.envs.chuanliantui_stand_v2._torques_before')
    contracts=json.loads(args.before_contracts.read_text())
    normalize=lambda value:json.loads(json.dumps(value))
    walk=normalize(runner.contracts(command_cfg('walk')))
    # 不把并行walk命令/课程版本变化算成本次getup奖励变化。
    for key in ('control_contract','observation_contract','switch_contract','physics_dt_s','policy_dt_s',
                'reward_recipe','reward_weights'):
        assert contracts['walk'][key]==walk[key]
    assert contracts['stand_v2_weights']==shared.WEIGHTS
    assert rewards.GETUP_WEIGHTS==contracts['getup']['reward_weights']
    assert rewards.GETUP_REWARD_RECIPE=='split-getup-torque2-prestand-full-nominal-state-v4'
    wrapped,_=task_registry.make_env('chuanliantui_getup',play=True,num_envs=4,seed=43,device=args.device)
    raw=wrapped.unwrapped;rows=[]
    try:
        current,previous=shared.RewardState(raw),before.RewardState(raw)
        state=raw.command_manager.get_term('standup')
        assert raw.cfg.rewards['torques'].weight==-2e-4
        state.reward_active.fill_(True);state.numerical_failure.zero_()
        torque=torch.zeros_like(raw.sim.data.qfrc_actuator)
        fake=SimpleNamespace(cfg=raw.cfg,scene=raw.scene,command_manager=raw.command_manager,
            action_manager=raw.action_manager,common_step_counter=0,
            sim=SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torque)))
        # 四腿电机的合成力矩，覆盖零值、未饱和及加权后-1饱和；不作物理步进。
        legs=[current.motor_vadr[i] for i in (0,1,3,4)]
        for stood in (False,True):
            state.has_stood.fill_(stood)
            for magnitude in (0.,1.,5.,10.,20.,30.,35.35534,40.):
                torque.zero_();torque[:,legs]=magnitude
                current.step=previous.step=-1
                actual=current.compute(fake);old=previous.compute(fake)
                assert all(torch.equal(actual[k],old[k]) for k in rewards.GETUP_WEIGHTS if k!='torques')
                weighted=actual['torques']*raw.cfg.rewards['torques'].weight
                expected=-min(4*magnitude*magnitude*2e-4,1.)
                assert torch.allclose(weighted,torch.full_like(weighted,expected))
                assert torch.allclose(weighted*raw.step_dt,torch.full_like(weighted,expected*.01))
                if stood:
                    assert torch.equal(actual['torques'],old['torques'])
                else:
                    assert not old['torques'].any()
                # 实际奖励管理器只能乘一次权重和dt，不能把-1误当成每10ms扣1分。
                raw.sim.data.qfrc_actuator.copy_(torque)
                raw._isaac_v2_rewards=current;current.step=-1
                total=raw.reward_manager.compute(raw.step_dt)
                index=tuple(raw.cfg.rewards).index('torques')
                assert torch.allclose(raw.reward_manager._step_reward[:,index],weighted)
                assert torch.allclose(total,raw.reward_manager._step_reward.sum(-1)*raw.step_dt,atol=1e-6)
                # 历史stand_v2仍沿用原权重、限幅与首次站稳门控。
                state.cfg.role=None;current.step=previous.step=-1
                historical=current.compute(fake);reference=previous.compute(fake)
                assert all(torch.equal(historical[k],reference[k]) for k in shared.WEIGHTS if k!='tracking_lin_vel_tight')
                state.cfg.role='getup'
                rows.append(dict(has_stood=stood,leg_torque_Nm=magnitude,weighted_before_dt=expected))
        # 集合仍是原六个主动电机，轮力矩不能漏算或混入弹簧/被动关节。
        state.has_stood.zero_();torque.zero_();torque[:,current.motor_vadr[[2,5]]]=3.9
        current.step=-1
        assert torch.allclose(current.compute(fake)['torques']*-2e-4,
            torch.full((4,),-2*3.9**2*2e-4,device=raw.device),atol=1e-7)
        for flag,value in (('reward_active',False),('numerical_failure',True)):
            getattr(state,flag).fill_(value);current.step=-1
            assert not current.compute(fake)['torques'].any()
            getattr(state,flag).fill_(not value)
        saved=torch.load(args.previous_checkpoint,map_location='cpu',weights_only=False)
        rejected(lambda:runner.validate(saved['infos'],command_cfg('getup')))
        report=dict(verified=True,parameter_updates=0,synthetic_cases=rows,
            torque_weight=-2e-4,previous_weight=-2e-4,weighted_minimum=-1.,
            pre_stand_penalty_enabled=True,post_stand_penalty_bitwise_equal=True,
            actual_reward_manager_weight_and_dt_once=True,all_six_active_motors_checked=True,
            other_18_getup_terms_bitwise_equal=True,
            standalone_stand_v2_unchanged=True,walk_control_and_reward_contract_fields_unchanged=True,
            numerical_and_inactive_masks_preserved=True,previous_getup_checkpoint_rejected=True,
            reward_recipe=rewards.GETUP_REWARD_RECIPE,source_sha256=digest(__file__),
            scope='Synthetic reward/contract checks; no trained behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:wrapped.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-shared',type=Path,required=True)
    parser.add_argument('--before-contracts',type=Path,required=True)
    parser.add_argument('--previous-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Check cannot train')):
        check(parser.parse_args())
