#!/usr/bin/env python3
"""Check full getup symmetry cost against the original 1999 reward snapshot; no learning."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import rewards, runner
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_stand_v2 import rewards as shared
from scripts.agent.split.check_chuanliantui_two_policy import rejected, digest


def source_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    old=source_module(args.original,'ct_mjlab.envs.chuanliantui_stand_v2._nominal_original')
    before=source_module(args.before,'ct_mjlab.envs.chuanliantui_stand_v2._nominal_before')
    old_runner=source_module(args.runner_before,'ct_mjlab.envs.chuanliantui_split._nominal_runner_before')
    assert old_runner.contracts(command_cfg('walk'))==runner.contracts(command_cfg('walk'))
    wrapped,_=task_registry.make_env('chuanliantui_getup',play=True,num_envs=20,seed=43,device=args.device)
    raw=wrapped.unwrapped
    rows=[]
    try:
        state=raw.command_manager.get_term('standup')
        current,original,previous=(m.RewardState(raw) for m in (shared,old,before))
        data=raw.scene['robot'].data
        positions=data.body_link_pos_w.clone()
        rotations=torch.zeros_like(data.root_link_quat_w);rotations[:,0]=1.
        fake=SimpleNamespace(body_link_pos_w=positions,root_link_quat_w=rotations,
            projected_gravity_b=data.projected_gravity_b,
            root_link_lin_vel_b=data.root_link_lin_vel_b,root_link_ang_vel_b=data.root_link_ang_vel_b,
            joint_pos=data.joint_pos,joint_vel=data.joint_vel)
        for reward_state in (current,original,previous):reward_state.robot=SimpleNamespace(data=fake)
        state.reward_active.fill_(True);state.numerical_failure.zero_();state.has_landed.fill_(True)
        for angle in (.1,1.5):
            # 合成小/大左右腿角差，分别覆盖未饱和及限幅后的惩罚。
            positions[:,current.hips]=0.
            positions[:,current.wheels]=torch.tensor(
                [[-torch.sin(torch.tensor(angle)),0.,-torch.cos(torch.tensor(angle))],[0.,0.,-1.]],device=raw.device)
            for stood in (False,True):
                state.has_stood.fill_(stood)
                for seconds in (0.,.125,.25,.49,.5,1.):
                    state.standing_time.fill_(seconds)
                    for value in (current,original,previous):value.step=-1
                    actual=current.compute(raw);reference=original.compute(raw);scaled=previous.compute(raw)
                    assert all(torch.equal(actual[name],reference[name]) for name in rewards.GETUP_WEIGHTS)
                    assert (actual['nominal_state']>0.).all()
                    assert all(torch.equal(actual[name],scaled[name]) for name in rewards.GETUP_WEIGHTS if name!='nominal_state')
                    factor=1. if stood else .1+.9*min(seconds/.5,1.)
                    assert torch.allclose(scaled['nominal_state'],actual['nominal_state']*factor)
                    # 历史stand_v2仍走原缩放，不能因getup回退而改写其行为。
                    state.cfg.role=None;current.step=-1
                    historical=current.compute(raw)
                    assert all(torch.equal(historical[name],scaled[name]) for name in rewards.GETUP_WEIGHTS)
                    state.cfg.role='getup'
                    rows.append(dict(angle_difference_rad=angle,has_stood=stood,standing_time_s=seconds,
                        full_weighted_cost=float(actual['nominal_state'][0]*rewards.GETUP_WEIGHTS['nominal_state']),
                        previous_scale=factor))
        state.numerical_failure.fill_(True);current.step=-1
        assert all(not value.any() for value in current.compute(raw).values())
        previous_model=torch.load(args.previous_checkpoint,map_location='cpu',weights_only=False)
        rejected(lambda:runner.validate(previous_model['infos'],command_cfg('getup')))
        report=dict(verified=True,parameter_updates=0,synthetic_cases=rows,
            all_19_getup_terms_match_original1999=True,other_18_terms_unchanged=True,
            standalone_stand_v2_unchanged=True,walk_contract_unchanged=True,
            previous_scaled_getup_checkpoint_rejected=True,numerical_mask_retained=True,
            nominal_state_weight=rewards.GETUP_WEIGHTS['nominal_state'],reward_recipe=rewards.GETUP_REWARD_RECIPE,
            original_sha256=digest(args.original),before_sha256=digest(args.before),source_sha256=digest(__file__),
            scope='Synthetic physical inputs and archived reward formulas; no trained behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='synthetic_cases'},indent=2))
    finally:raw.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--runner-before',type=Path,required=True)
    parser.add_argument('--previous-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Check cannot train')):check(args)
