#!/usr/bin/env python3
"""CPU synthetic reward equivalence check; no simulator, checkpoints or updates."""
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from ct_mjlab.envs.chuanliantui_split import rewards
from ct_mjlab.supermimic.analytic_reward import WEIGHTS, differentiable_reward
from ct_mjlab.supermimic.motion_adapter import MotionStateAdapter, MOTION_AUXILIARY_RECIPE
from ct_mjlab.supermimic.state import NOISE
from scripts.agent.supermimic.check_supermimic_motion import fixture


def main():
    torch.set_num_threads(1)
    removed = ('lin_vel_z', 'base_link_contact', 'wheels_airborne', 'action_rate', 'action_smooth')
    assert WEIGHTS == rewards.WALK_WEIGHTS and len(WEIGHTS) == 13
    assert all(name not in WEIGHTS for name in removed)
    assert {name:rewards.GETUP_WEIGHTS[name] for name in removed} == dict(
        lin_vel_z=-1., base_link_contact=-.3, wheels_airborne=-.3, action_rate=-.01, action_smooth=-.01)
    n = 4
    action = torch.zeros(n, 6)
    command = torch.zeros(n, 3); command[:, 2] = .25
    state = NS(command=command, cfg=NS(tracking_tight_kernel_denominator=.02),
        tracking_active=torch.ones(n, dtype=torch.bool), position_locked=torch.zeros(n, dtype=torch.bool),
        reward_active=torch.ones(n, dtype=torch.bool), numerical_failure=torch.zeros(n, dtype=torch.bool),
        height_above_ground=torch.full((n,), .25))
    positions = torch.zeros(n, 4, 3); positions[:, 2:, 2] = -.25
    gravity = torch.zeros(n, 3); gravity[:, 2] = -1.
    data = NS(body_link_pos_w=positions, root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_link_lin_vel_b=torch.zeros(n, 3), root_link_ang_vel_b=torch.zeros(n, 3),
        projected_gravity_b=gravity, joint_pos=torch.zeros(n, 6), joint_vel=torch.zeros(n, 6))
    support = torch.zeros(n, 2, 3); support[:, :, 2] = -10.
    env = NS(common_step_counter=0, command_manager=NS(get_term=lambda _: state),
        action_manager=NS(action=action, prev_action=action, prev_prev_action=action),
        scene={'wheel_contact': NS(data=NS(force=support))},
        sim=NS(data=NS(qfrc_actuator=torch.zeros(n, 6))))
    reward_state = rewards.WalkRewardState.__new__(rewards.WalkRewardState)
    reward_state.step = -1; reward_state.robot = NS(data=data)
    reward_state.hips = [0, 1]; reward_state.wheels = [2, 3]
    reward_state.legs = [0, 1, 2, 3]; reward_state.knees = [4, 5]
    reward_state.motor_vadr = list(range(6))
    reward_state.soft_limits = torch.tensor([[[-1., 1.]] * 4])
    reward_state.knee_limits = torch.tensor([[[-1., 1.]] * 2])
    with patch.object(rewards, 'base_contact_force', return_value=torch.zeros(n)):
        original = reward_state.compute(env)
        # 改变上下速度、轮支撑和原始动作，已删除的奖励项不应再产生扣分。
        data.root_link_lin_vel_b[:, 2] = torch.tensor([-3., -.1, .1, 3.])
        support.zero_(); action.fill_(2.)
        reward_state.step = -1
        changed = reward_state.compute(env)
    assert set(original) == set(changed) == set(WEIGHTS)
    for name in WEIGHTS:
        torch.testing.assert_close(original[name], changed[name], atol=0, rtol=0)
    features = torch.zeros(n, 31); features[:, 8] = -1.; features[:, 9] = .25; features[:, 29:31] = 1.
    features[:, 2] = data.root_link_lin_vel_b[:, 2]; features.requires_grad_()
    context = torch.zeros(n, 21); context[:, 2] = .25; context[:, 3] = context[:, 6] = 1.
    context[:, 7:19:2] = -1.; context[:, 8:19:2] = 1.; context[:, 19] = .02; context[:, 20] = .01
    terms, total = differentiable_reward(features, action, action, action, context)
    altered_terms, altered_total = differentiable_reward(features, action*3., -action, action*.5, context)
    assert set(terms) == set(altered_terms) == set(WEIGHTS)
    torch.testing.assert_close(total, altered_total, atol=0, rtol=0)
    expected = torch.zeros(n)
    for name, weight in WEIGHTS.items():
        torch.testing.assert_close(terms[name], changed[name], atol=0, rtol=0)
        expected += changed[name] * weight * .01
    torch.testing.assert_close(total, expected, atol=0, rtol=0)
    assert torch.count_nonzero(torch.autograd.grad(total.sum(), features)[0][:, 2]) == 0
    # 直接走当前世界模型适配器，确认没有继续访问已删除的奖励键。
    motion_state, parameters, motion_context = fixture(n=n)
    path = motion_state[:, None, :64].expand(-1, 5, -1).clone()
    zero_action = torch.zeros(n, 6)
    with torch.no_grad():
        imagined = MotionStateAdapter().advance(motion_state, parameters, path, torch.zeros(n, 3),
            zero_action, zero_action, zero_action, motion_context, motion_state[:, NOISE])
    assert torch.isfinite(imagined[2]).all()
    assert MOTION_AUXILIARY_RECIPE['disabled_model_reward_terms'] == []
    print(json.dumps(dict(passed=True, walk_terms=13, source_and_analytic_equal=True,
        removed_terms=list(removed), raw_action_history_reward_invariant=True,
        motion_adapter_finite_without_removed_reward_keys=True,
        vertical_velocity_reward_gradient_zero=True, getup_unchanged=True,
        optimizer_steps=0, scope='Synthetic reward checks only; no behavior acceptance')))


if __name__ == '__main__':
    main()
