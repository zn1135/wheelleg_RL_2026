#!/usr/bin/env python3
"""CPU 回归检查：高度奖励不再裁平、起立后膝关节保留行程余量。

调用真实奖励准备/计算路径；不创建仿真或训练产物，不替代行为回放。
"""
from pathlib import Path
from xml.etree import ElementTree as ET

import isaacgym  # 必须先于 torch
import numpy as np
import torch

from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup import ChuanliantuiStandup
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import ChuanliantuiStandupCfg


def fixture(n):
    env = ChuanliantuiStandup.__new__(ChuanliantuiStandup)
    env.cfg = ChuanliantuiStandupCfg()
    env.device, env.num_envs, env.num_dof = "cpu", n, 6
    env.dof_names = ["rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel"]
    env.dt = env.cfg.sim.dt * env.cfg.control.decimation
    env.rew_buf = torch.zeros(n)
    env.has_stood = torch.ones(n, dtype=torch.bool)
    env.standup_curriculum_unlocked = True
    env.termination_contact_indices = torch.tensor([0])
    env.contact_forces = torch.zeros(n, 1, 3)
    env.commands = torch.zeros(n, 3)
    env.commands[:, 2] = env.cfg.standup_curriculum.post_unlock_target_height
    return env


def effective_reward(env, name):
    env.reward_scales = {name: getattr(env.cfg.rewards.scales, name)}
    env._prepare_reward_function()
    env.compute_reward()
    return env.rew_buf.clone()


def check_height():
    env = fixture(4)
    env.base_height = env.commands[:, 2] + torch.tensor([0., .008, .03, -.03])
    reward = effective_reward(env, "base_height")
    assert reward[0] > reward[1] > reward[2] > 0, (
        "高度目标与 8mm/3cm 误差必须在实际裁剪之后仍可区分", reward.tolist())
    torch.testing.assert_close(reward[2], reward[3])
    # 接触门控仍有效；初始课程保留宽容差，解锁及恢复后使用精细容差。
    env.contact_forces[:, 0, 2] = 5.
    torch.testing.assert_close(effective_reward(env, "base_height"), reward * .2)
    env.contact_forces.zero_()
    env.base_height = env.commands[:, 2] + .05
    env.standup_curriculum_unlocked = False
    broad = effective_reward(env, "base_height")
    env.standup_curriculum_unlocked = True
    narrow = effective_reward(env, "base_height")
    assert torch.all(broad > narrow), (broad, narrow)
    print("PASS: 高度精度、正负误差对称、接触门控及课程宽窄容差")


def check_knee_margin():
    env = fixture(8)
    urdf = Path(__file__).resolve().parents[2] / "resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf"
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    props = np.zeros(6, dtype=[(key, np.float32) for key in ("lower", "upper", "velocity", "effort")])
    for i, name in enumerate(env.dof_names):
        limit = joints[name].find("limit")
        for key in props.dtype.names:
            # 连续轮不提供位置限位，本检查不对轮使用位置奖励。
            props[key][i] = float(limit.get(key, "0"))
    original = props.copy()
    env._process_dof_props(props, 0)
    assert np.array_equal(props, original), "奖励余量不得改动物理关节限位"
    assert hasattr(env, "knee_hard_limits"), "需要保存 URDF 硬限位，不能使用 97% 软限位替代"
    torch.testing.assert_close(env.knee_hard_limits, torch.tensor([[-.12, .77], [-.77, .12]]))
    margin = env.cfg.rewards.knee_limit_margin_rad
    lower, upper = env.knee_hard_limits[:, 0], env.knee_hard_limits[:, 1]
    env.dof_pos = torch.zeros(8, 6)
    env.dof_pos[:, [1, 4]] = torch.stack([
        upper - margin, upper - margin / 2, upper, upper + margin / 10,
        lower + margin / 2, lower, torch.tensor([upper[0], 0.]), upper,
    ])
    env.has_stood[-1] = False
    reward = effective_reward(env, "knee_limit_margin")
    assert abs(reward[0]) < 1e-10 and reward[-1] == 0
    assert reward[0] > reward[1] > reward[2] > reward[3], reward.tolist()
    torch.testing.assert_close(reward[1], reward[4])
    torch.testing.assert_close(reward[2], reward[5])
    torch.testing.assert_close(reward[6], reward[2] / 2)
    # 真实端点处必须保留梯度；不能像旧高度项一样被单项裁剪成平顶。
    assert reward[3] > -env.cfg.rewards.clip_single_reward * env.dt
    print("PASS: 真实硬限位、余量边界、递增惩罚、左右/上下限对称及起立门控")


if __name__ == "__main__":
    failures = []
    for check in (check_height, check_knee_margin):
        try:
            check()
        except AssertionError as error:
            failures.append(f"{check.__name__}: {error}")
    if failures:
        raise AssertionError("\n".join(failures))
