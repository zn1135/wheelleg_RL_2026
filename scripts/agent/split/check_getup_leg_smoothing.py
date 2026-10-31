#!/usr/bin/env python3
"""Check policy-rate leg smoothing through real getup PD, reset and metadata; no learning."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split.action import GetupPdAction, LEG_ACTION_ALPHA
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg, make_env_cfg
from ct_mjlab.envs.chuanliantui_split.runner import contracts, validate
from ct_mjlab.envs.chuanliantui_stand_v2.action import StandV2PdAction
from scripts.agent.split.check_chuanliantui_two_policy import rejected, digest


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    before=json.loads(args.before_contracts.read_text())
    normalize=lambda value:json.loads(json.dumps(value))
    walk=normalize(contracts(command_cfg('walk')))
    # 并行walk开发可更换命令/课程版本；此处只约束本次涉及的控制接口。
    for key in ('control_contract','observation_contract','switch_contract','physics_dt_s','policy_dt_s'):
        assert before['walk'][key]==walk[key]
    current=normalize(contracts(command_cfg('getup')))
    # 奖励配方的独立变更由check_getup_torques.py核对；此处核对其余配置及原EMA执行。
    unchanged=lambda value:{k:v for k,v in value.items() if k not in ('action_filter','action_filter_contract','reward_recipe')}
    assert unchanged(current)==unchanged(before['getup'])
    assert LEG_ACTION_ALPHA==.8
    assert make_env_cfg('walk').actions['pd'].class_type is StandV2PdAction
    env,_=task_registry.make_env('chuanliantui_getup',play=True,num_envs=4,seed=43,device=args.device)
    raw=env.unwrapped;rows=[]
    try:
        pd=raw.action_manager.get_term('pd');state=raw.command_manager.get_term('standup')
        assert type(pd) is GetupPdAction and raw.step_dt==.01 and raw.cfg.decimation==5
        legs,wheels=pd._leg_action_ids,pd._wheel_action_ids
        action=torch.full((4,6),.2,device=raw.device);action[:,wheels]=torch.tensor([.3,-.2],device=raw.device)
        raw.sim.data.qpos[:,pd._motor_q_adr]=pd.nominal_positions
        raw.sim.data.qvel.zero_();raw.sim.forward()
        state.ground_start.fill_(True)
        # 等待期不能预先把缓存累积到完整动作；第21拍才开始混合。
        for age in range(20):
            raw.episode_length_buf.fill_(age);pd.process_actions(action)
            for _ in range(5):pd.apply_actions();assert not pd.torques.any()
            assert not pd._filtered_leg_actions.any()
        raw.episode_length_buf.fill_(20)
        expected=torch.zeros((4,4),device=raw.device)
        for tick in range(1,4):
            pd.process_actions(action);expected=LEG_ACTION_ALPHA*action[:,legs]+(1.-LEG_ACTION_ALPHA)*expected
            assert torch.allclose(expected,torch.full_like(expected,(.16,.192,.1984)[tick-1]))
            assert torch.equal(pd.raw_action,action)
            assert torch.allclose(pd._control_actions[:,legs],expected)
            assert torch.equal(pd._control_actions[:,wheels],action[:,wheels])
            for _ in range(5):
                pd.apply_actions()
                assert torch.allclose(pd.torques[:,legs],60.*.5*expected,atol=2e-5)
                assert torch.allclose(pd.torques[:,wheels],torch.tensor([.3,-.2],device=raw.device),atol=1e-6)
                assert torch.allclose(pd._filtered_leg_actions,expected)
            rows.append(dict(policy_tick=tick,raw_leg_action=.2,filtered_leg_action=float(expected[0,0]),
                leg_target_offset_rad=float(expected[0,0]*.5),leg_torque_Nm=float(pd.torques[0,0])))
        cached=pd._filtered_leg_actions.clone();pd.reset(torch.tensor([0],device=raw.device))
        assert not pd._filtered_leg_actions[0].any() and torch.equal(pd._filtered_leg_actions[1:],cached[1:])
        pd.reset();pd.action_delay_steps.fill_(2);pd.process_actions(action)
        for substep in range(5):
            pd.apply_actions();expected_torque=0. if substep<2 else 4.8
            assert torch.allclose(pd.torques[:,legs],torch.full((4,4),expected_torque,device=raw.device),atol=2e-5)
        infos=contracts(state.cfg);infos['standup_curriculum']=state.checkpoint_state()
        for key in ('action_filter_contract','action_filter'):
            bad=copy.deepcopy(infos);bad.pop(key);rejected(lambda:validate(bad,state.cfg))
        for key,value in (('alpha',.25),('policy_dt_s',.002),('action_ids',(0,1,2,3,4,5))):
            bad=copy.deepcopy(infos);bad['action_filter'][key]=value;rejected(lambda:validate(bad,state.cfg))
        old=torch.load(args.previous_checkpoint,map_location='cpu',weights_only=False)
        rejected(lambda:validate(old['infos'],state.cfg))
        report=dict(verified=True,parameter_updates=0,alpha=LEG_ACTION_ALPHA,cases=rows,
            physical_pd_checked=True,one_filter_update_per_policy_tick=True,
            raw_action_and_wheels_preserved=True,ground_hold_clears_filter=True,
            partial_reset_isolated=True,original_physics_tick_delay_preserved=True,
            walk_control_contract_fields_and_action_unchanged=True,getup_contracts_except_filter_and_reward_recipe_unchanged=True,
            previous_action_recipe_and_changed_filter_rejected=True,
            source_sha256=digest(__file__),scope='Synthetic action inputs through actual PD; no trained behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-contracts',type=Path,required=True)
    parser.add_argument('--previous-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Check cannot train')):
        check(parser.parse_args())
