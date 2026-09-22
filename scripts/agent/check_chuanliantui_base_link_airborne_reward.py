#!/usr/bin/env python3
"""CPU 检查起立任务 base_link 接地惩罚；不创建仿真或训练产物。"""

import isaacgym  # 必须先于 torch
import torch

from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup import (
    ChuanliantuiStandup,
)
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import (
    ChuanliantuiStandupCfg,
    ChuanliantuiStandupCfgPPO,
)


def main():
    terminal_keys = ChuanliantuiStandupCfgPPO.runner.terminal_episode_keys
    assert "rew_leg_angle" not in terminal_keys
    assert "rew_wheels_airborne" in terminal_keys
    assert "rew_base_link_airborne" in terminal_keys
    assert hasattr(ChuanliantuiStandup, "_reward_wheels_airborne")
    assert "standup_curriculum_target_height" in terminal_keys

    env = ChuanliantuiStandup.__new__(ChuanliantuiStandup)
    env.cfg = ChuanliantuiStandupCfg()
    assert env.cfg.standup.success_requires_base_contact_free
    assert env.cfg.standup_curriculum.unlock_recovered_rate == 0.30
    assert env.cfg.commands.ranges.height == [0.20, 0.20]
    assert env.cfg.rewards.scales.orientation == -1.0
    assert env.cfg.rewards.scales.base_height == 2.0
    assert env.cfg.rewards.scales.leg_angle == 0.0
    assert env.cfg.rewards.height_reward_sigma == 0.01
    assert env.cfg.rewards.height_reward_contact_factor == 0.2
    assert env.cfg.rewards.base_link_reward_force_scale == 5.0
    assert env.cfg.rewards.scales.base_link_contact == -0.3
    assert env.cfg.rewards.scales.base_link_airborne == 0.2
    assert env.cfg.rewards.scales.wheels_airborne == -0.3
    env.device = "cpu"
    env.num_envs = 4
    env.dt = env.cfg.sim.dt * env.cfg.control.decimation
    env.has_landed = torch.tensor([False, True, True, True])
    env.termination_contact_indices = torch.tensor([0])
    env.contact_forces = torch.zeros(4, 3, 3)
    threshold = env.cfg.standup.success_base_contact_force_threshold
    env.contact_forces[1, 0, 2] = 2.5
    env.contact_forces[2, 0, 2] = 5.0

    # 奖励使用连续接触比例；成功判据仍使用严格的 0.1 N 二值阈值。
    expected = torch.tensor([0.0, 0.5, 1.0, 0.0])
    torch.testing.assert_close(env._reward_base_link_contact(), expected)
    assert torch.equal(
        env._base_link_has_contact(threshold),
        torch.tensor([False, True, True, False]),
    )

    # 走基类实际的 dt 缩放与单项裁剪路径，验证接地项为负值。
    env.rew_buf = torch.zeros(env.num_envs)
    env.reward_scales = {"base_link_contact": env.cfg.rewards.scales.base_link_contact}
    env._prepare_reward_function()
    assert env.reward_names == ["base_link_contact"]
    env.compute_reward()
    assert torch.allclose(
        env.rew_buf, expected * env.cfg.rewards.scales.base_link_contact * env.dt
    )
    torch.testing.assert_close(env.episode_sums["base_link_contact"], env.rew_buf)

    expected_airborne = torch.tensor([0.0, 0.5, 0.0, 1.0])
    torch.testing.assert_close(env._reward_base_link_airborne(), expected_airborne)
    env.rew_buf = torch.zeros(env.num_envs)
    env.reward_scales = {"base_link_airborne": env.cfg.rewards.scales.base_link_airborne}
    env._prepare_reward_function()
    assert env.reward_names == ["base_link_airborne"]
    env.compute_reward()
    assert torch.allclose(env.rew_buf, expected_airborne * 0.2 * env.dt)

    # 双轮同时无有效支撑才扣分；首次自由落下阶段虽产生该原始项，但不进入 PPO 样本。
    env.feet_indices = torch.tensor([1, 2])
    env.contact_forces[1, 1, 2] = env.cfg.standup.wheel_contact_force_threshold + 0.01
    expected_wheels_airborne = torch.tensor([1.0, 0.0, 1.0, 1.0])
    assert torch.equal(env._reward_wheels_airborne(), expected_wheels_airborne)
    env.rew_buf = torch.zeros(env.num_envs)
    env.reward_scales = {"wheels_airborne": env.cfg.rewards.scales.wheels_airborne}
    env._prepare_reward_function()
    assert env.reward_names == ["wheels_airborne"]
    env.compute_reward()
    assert torch.allclose(
        env.rew_buf,
        expected_wheels_airborne * env.cfg.rewards.scales.wheels_airborne * env.dt,
    )

    # 固定 0.20 m 指令；高度门控随 0–5 N 接触力连续从 1 过渡到 0.2。
    target_height = 0.20
    env.base_height = torch.tensor([0.20, 0.20, 0.15, 0.25])
    env.commands = torch.full((4, 3), 0.0)
    env.commands[:, 2] = target_height
    env.contact_forces[:, 0, :] = 0.0
    env.contact_forces[1, 0, 2] = 2.5
    env.contact_forces[2, 0, 2] = 5.0
    expected_height_reward = torch.exp(
        -torch.square(env.base_height - target_height)
        / env.cfg.rewards.height_reward_sigma
    ) * torch.tensor([1.0, 0.6, 0.2, 1.0])
    torch.testing.assert_close(
        env._reward_base_height(),
        expected_height_reward,
    )

    # 腿倾角项已关闭，基类准备阶段会删除零权重项，训练中不调用该奖励函数。
    env.reward_scales = {"leg_angle": env.cfg.rewards.scales.leg_angle}
    env._prepare_reward_function()
    assert env.reward_names == []

    # 课程在完整 4096 回合窗口达到 30% 恢复率后，永久切换为 0.20 m / -10。
    env.standup_curriculum_unlocked = False
    env.standup_curriculum_completed_episodes = 0
    env.standup_curriculum_recovered_episodes = 0
    env.standup_curriculum_last_recovered_rate = 0.0
    env.reward_scales = {"orientation": env.cfg.rewards.scales.orientation * env.dt}
    env.commands = torch.zeros(2, 3)
    assert env._current_standup_target_height() == 0.30
    assert env._current_standup_success_height() == 0.28
    assert not env._update_standup_curriculum(4096, 1228)
    assert not env.standup_curriculum_unlocked
    assert env._update_standup_curriculum(4096, 1229)
    assert env.standup_curriculum_unlocked
    assert env._current_standup_target_height() == 0.20
    assert env._current_standup_success_height() == 0.18
    assert torch.equal(env.commands[:, 2], torch.full((2,), 0.20))
    assert env.reward_scales["orientation"] == -10.0 * env.dt
    assert not env._update_standup_curriculum(4096, 0)
    assert env.standup_curriculum_unlocked

    # 课程解锁与窗口统计随 checkpoint 保存、在新环境实例恢复。
    checkpoint_state = env.get_checkpoint_state()
    restored_env = ChuanliantuiStandup.__new__(ChuanliantuiStandup)
    restored_env.cfg = ChuanliantuiStandupCfg()
    restored_env.dt = env.dt
    restored_env.commands = torch.zeros(2, 3)
    restored_env.reward_scales = {
        "orientation": restored_env.cfg.rewards.scales.orientation * restored_env.dt
    }
    assert restored_env.load_checkpoint_state(checkpoint_state)
    assert restored_env.standup_curriculum_unlocked
    assert restored_env.standup_curriculum_completed_episodes == 0
    assert restored_env.standup_curriculum_recovered_episodes == 0
    assert restored_env.standup_curriculum_last_recovered_rate == 1229 / 4096
    assert torch.equal(restored_env.commands[:, 2], torch.full((2,), 0.20))
    assert restored_env.reward_scales["orientation"] == -10.0 * restored_env.dt

    print("PASS: 连续 base_link 奖励与可恢复站立课程均按配置生效")


if __name__ == "__main__":
    main()
