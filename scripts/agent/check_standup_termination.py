"""CPU 检查起立任务的接地门控、持续站稳、跌倒及腾空终止；不创建仿真。"""

from unittest.mock import patch
from types import SimpleNamespace

import isaacgym  # 必须先于 torch
import torch

from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup import (
    ChuanliantuiStandup,
)
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import (
    ChuanliantuiStandupCfg,
)


def make_env(num_envs, unlocked=False):
    """只构造真实判定方法需要的 CPU 状态，课程字段与正常初始化一致。"""
    env = ChuanliantuiStandup.__new__(ChuanliantuiStandup)
    env.cfg = ChuanliantuiStandupCfg()
    env.num_envs = num_envs
    env.device = "cpu"
    env.dt = env.cfg.sim.dt * env.cfg.control.decimation
    env.max_episode_length = round(env.cfg.env.episode_length_s / env.dt)
    env.standup_curriculum_unlocked = unlocked
    env.standup_curriculum_completed_episodes = 0
    env.standup_curriculum_recovered_episodes = 0
    env.standup_curriculum_last_recovered_rate = 0.0
    env.commands = torch.zeros(num_envs, 3)
    env.commands[:, 2] = env._current_standup_target_height()
    env.reward_scales = {"orientation": env.cfg.rewards.scales.orientation * env.dt}
    env.projected_gravity = torch.zeros(num_envs, 3)
    env.projected_gravity[:, 2] = -1.0
    env.base_height = torch.full((num_envs,), env._current_standup_target_height())
    env.has_stood = torch.zeros(num_envs, dtype=torch.bool)
    env.standing_time = torch.zeros(num_envs)
    env.wheels_airborne_steps = torch.zeros(num_envs, dtype=torch.long)
    env.has_landed = torch.ones(num_envs, dtype=torch.bool)
    env.fail_buf = torch.zeros(num_envs)
    env.contact_forces = torch.zeros(num_envs, 3, 3)
    env.feet_indices = torch.tensor([1, 2])
    env.termination_contact_indices = torch.tensor([0])
    env.contact_forces[:, 1:, 2] = 2.0
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    env.time_out_buf = torch.zeros(num_envs, dtype=torch.bool)
    env.extras = {"episode": {}}
    env.num_bodies = 3
    env.envs, env.actor_handles = [None], [None]
    env.gym = SimpleNamespace(find_actor_rigid_body_handle=lambda *args: 0)
    env.rigid_body_external_forces = torch.zeros(num_envs, 3, 3)
    env.rigid_body_external_torques = torch.zeros_like(env.rigid_body_external_forces)
    env._init_standup_pushes()
    return env


def check_first_contact_and_episode_timeout():
    env = make_env(6)
    env.has_landed[:] = False
    env.projected_gravity[:, 2] = 1.0
    env.episode_length_buf[:] = env.max_episode_length + 1
    env.contact_forces[:] = 0.0
    env.contact_forces[4, 1:, 2] = 1.0  # 等于阈值不算支撑。
    env.contact_forces[5, 1:, 0] = 100.0  # 水平接触不算竖直支撑。
    env.check_termination()
    assert not env.get_policy_action_mask().any()
    assert not env.reset_buf.any() and not env.time_out_buf.any()
    assert (env.standing_time == 0).all()
    assert (env.wheels_airborne_steps == 0).all()

    # 当前低位后摆初态也使用同一门控；任一轮首次接触才开始策略回合。
    env.base_height[:] = env.cfg.init_state.pos[2]
    env.projected_gravity[:, 2] = -1.0
    env.contact_forces[0, 1, 2] = 1.1
    env.contact_forces[1, 2, 2] = 1.1
    env.contact_forces[2, 1:, 2] = 1.1
    env.check_termination()
    assert torch.equal(
        env.get_policy_action_mask(),
        torch.tensor([True, True, True, False, False, False]),
    )
    assert (env.episode_length_buf[:3] == 0).all()
    assert (env.episode_length_buf[3:] == env.max_episode_length + 1).all()
    assert not env.reset_buf.any()
    env.episode_length_buf[0] = env.max_episode_length
    env.episode_length_buf[1] = env.max_episode_length + 1
    env.check_termination()
    expected = torch.tensor([False, True, False, False, False, False])
    assert torch.equal(env.time_out_buf, expected)
    assert torch.equal(env.reset_buf, expected)
    print("PASS: first-wheel-contact policy gate and strict episode-timeout boundary")


def check_continuous_success_and_latch():
    # 输入覆盖课程切换前后的真实成功门槛，不再使用旧的固定 0.30 m 门槛。
    for unlocked, target_height, success_height in ((False, .30, .28), (True, .22, .20)):
        env = make_env(6, unlocked)
        assert torch.allclose(env.commands[:, 2], torch.full((6,), target_height))
        env.base_height[:] = success_height
        env.base_height[1] = success_height - .0001
        env.projected_gravity[:, 2] = -.90
        env.projected_gravity[2, 2] = -.8999
        env.contact_forces[3, 0, 0] = .1001
        env.contact_forces[4, 0, 0] = .1  # 合力严格超过阈值才算机身接地。
        env.has_landed[5] = False
        env.contact_forces[5] = 0.0
        hold_steps = round(.5 / env.dt)
        for _ in range(hold_steps - 1):
            env.check_termination()
        assert not env.has_stood.any()  # 0.49 s 尚未满足连续时长。
        # float32 连加 0.01 在名义第 50 步可能略小于 0.5；再一步应已锁存。
        env.check_termination()
        env.check_termination()
        expected = torch.tensor([True, False, False, False, True, False])
        assert torch.equal(env.has_stood, expected), env.has_stood
        assert (env.standing_time[~expected] == 0).all()
        assert not env.reset_buf.any()

        env.base_height[0] = success_height - .0001
        env.check_termination()
        assert env.standing_time[0] == 0 and env.has_stood[0]
        env.base_height[0] = success_height
        env.check_termination()
        assert torch.isclose(env.standing_time[0], torch.tensor(env.dt))
        assert env.has_stood[0]  # 历史成功标志不随一次失稳清除。

    # 单步跨过时长边界，避免仅靠“多跑几步”掩盖错误的比较符号。
    env = make_env(2, unlocked=True)
    env.standing_time[:] = torch.tensor([.49, .4899])
    env.check_termination()
    assert torch.equal(env.has_stood, torch.tensor([True, False]))
    env.base_height[1] = .1999
    env.check_termination()
    assert env.standing_time[1] == 0 and not env.has_stood[1]
    env.base_height[1] = .22
    env.check_termination()
    assert torch.isclose(env.standing_time[1], torch.tensor(env.dt))
    assert not env.has_stood[1]  # 两段不足 0.5 s 的稳站不能拼成一次成功。
    print("PASS: curriculum height/tilt/contact boundaries, continuous 0.5 s and success latch")


def check_delayed_fall_and_inversion():
    env = make_env(5)
    env.base_height[:] = .08
    env.has_stood[:] = True
    env.contact_forces[0, 0, 2] = 10.1
    env.projected_gravity[1, 2] = -.0999
    env.projected_gravity[2, 2] = -.1
    env.contact_forces[3, 0, 2] = 10.0
    env.contact_forces[4, 0, 2] = 11.0
    env.has_stood[4] = False
    delay_steps = round(1.0 / env.dt)
    for _ in range(delay_steps):
        env.check_termination()
        assert not env.reset_buf.any()
    assert torch.equal(env.fail_buf, torch.tensor([100., 100., 0., 0., 0.]))
    env.check_termination()
    assert torch.equal(env.reset_buf, torch.tensor([True, True, False, False, False]))
    assert not env.time_out_buf.any()
    env.contact_forces[:, 0] = 0.0
    env.projected_gravity[:, 2] = -1.0
    env.check_termination()
    assert (env.fail_buf == 0).all() and not env.reset_buf.any()

    env = make_env(4)
    env.base_height[:] = .08
    env.has_stood[:] = torch.tensor([False, True, False, True])
    env.projected_gravity[:, 2] = torch.tensor([1e-6, 1e-6, 0., 0.])
    env.check_termination()
    assert torch.equal(env.reset_buf, torch.tensor([True, True, False, False]))
    assert not env.time_out_buf.any()
    print("PASS: severe fall requires >1 s after success; pg_z>0 resets immediately")


def check_airborne_boundary_and_recovery():
    env = make_env(8)
    env.base_height[:] = .08
    env.has_stood[1] = True
    env.contact_forces[:] = 0.0
    env.contact_forces[2, 1, 2] = 1.1
    env.contact_forces[3, 2, 2] = 1.1
    env.contact_forces[4, 1:, 2] = 1.1
    env.contact_forces[5, 1:, 2] = 1.0
    env.contact_forces[6, 1:, 0] = 100.0
    expected = torch.tensor([True, True, False, False, False, True, True, True])
    air_steps = round(.2 / env.dt)
    for _ in range(air_steps - 1):
        env.check_termination()
        assert not env.reset_buf.any()
    assert (env.wheels_airborne_steps[expected] == 19).all()
    env.check_termination()
    assert torch.equal(env.reset_buf, expected)
    assert not env.time_out_buf.any()

    # 每次仅一轮重新接地都清除计时，随后必须重新连续离地 0.2 s。
    for foot in (1, 2):
        env.contact_forces[7, foot, 2] = 1.1
        env.check_termination()
        assert env.wheels_airborne_steps[7] == 0 and not env.reset_buf[7]
        env.contact_forces[7] = 0.0
        for _ in range(air_steps - 1):
            env.check_termination()
            assert not env.reset_buf[7]
        assert env.wheels_airborne_steps[7] == 19
    env.check_termination()
    assert env.reset_buf[7] and env.wheels_airborne_steps[7] == 20
    print("PASS: 0.2 s airborne boundary and either-wheel touchdown restart the timer")


def check_selected_reset_and_curriculum():
    env = make_env(4)
    env.has_stood[:] = torch.tensor([True, True, False, True])
    env.standing_time[:] = torch.tensor([.7, .8, .2, .9])
    env.wheels_airborne_steps[:] = torch.tensor([7, 8, 9, 10])
    env.standup_curriculum_completed_episodes = 10
    env.standup_curriculum_recovered_episodes = 4
    remaining = torch.tensor([0, 2])
    selected = torch.tensor([1, 3])
    before = {
        name: getattr(env, name)[remaining].clone()
        for name in ("has_stood", "standing_time", "wheels_airborne_steps", "has_landed")
    }
    # 父类 reset 涉及 Isaac 物理张量；仅隔离它，保留本类课程统计及清理逻辑。
    with patch.object(Chuanliantui, "reset_idx", return_value=None):
        env.reset_idx(torch.empty(0, dtype=torch.long))
        env.reset_idx(selected)
    assert not env.has_stood[selected].any()
    assert not env.has_landed[selected].any()
    assert (env.standing_time[selected] == 0).all()
    assert (env.wheels_airborne_steps[selected] == 0).all()
    for name, expected in before.items():
        assert torch.equal(getattr(env, name)[remaining], expected), name
    assert env.standup_curriculum_completed_episodes == 12
    assert env.standup_curriculum_recovered_episodes == 6
    assert env.extras["episode"]["recovered_rate"].item() == 1.0
    assert not env.standup_curriculum_unlocked
    print("PASS: selected reset clears local state and preserves other environments/global curriculum")


def main():
    check_first_contact_and_episode_timeout()
    check_continuous_success_and_latch()
    check_delayed_fall_and_inversion()
    check_airborne_boundary_and_recovery()
    check_selected_reset_and_curriculum()
    print("CPU numerical checks only; no physics, training, playback or sim2sim acceptance.")


if __name__ == "__main__":
    main()
