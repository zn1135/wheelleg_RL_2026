"""CPU 检查重力投影、未站稳、双轮离地及延迟判死；不创建仿真或训练产物。"""

from unittest.mock import patch

import isaacgym  # 必须先于 torch
import torch

from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui
from wheel_legged_gym.envs.chuanliantui.chuanliantui_config import ChuanliantuiCfg

from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup import (
    ChuanliantuiStandup,
)
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import (
    ChuanliantuiStandupCfg,
)


def main():
    env = ChuanliantuiStandup.__new__(ChuanliantuiStandup)
    env.cfg = ChuanliantuiStandupCfg()
    expected_stand_dofs = torch.tensor(
        [
            ChuanliantuiCfg.init_state.default_joint_angles[name]
            for name in ("lf0", "lf1", "lfwheel", "rf0", "rf1", "rfwheel")
        ]
    )
    assert env.cfg.init_state.pos == [0.0, 0.0, 1.0]
    assert env.cfg.standup.fall_start_height == 1.0
    assert torch.equal(torch.tensor(env.cfg.standup.initial_dof_pos), expected_stand_dofs)
    env.dt = env.cfg.sim.dt * env.cfg.control.decimation
    env.max_episode_length = round(env.cfg.env.episode_length_s / env.dt)
    # 分别覆盖正值、零、负值、已站起/未站起、接触与超时。
    pg_z = torch.tensor([1e-6, 1e-6, 0., 0., -.05, -.05, -1., -1., -1.])
    env.projected_gravity = torch.zeros(9, 3)
    env.projected_gravity[:, 2] = pg_z
    env.base_height = torch.full((9,), .08)
    env.has_stood = torch.tensor([False, True, False, True, False, True, True, False, False])
    env.standing_time = torch.zeros(9)
    env.standup_steps = torch.zeros(9, dtype=torch.long)
    env.wheels_airborne_steps = torch.zeros(9, dtype=torch.long)
    env.has_landed = torch.zeros(9, dtype=torch.bool)
    env.fail_buf = torch.zeros(9)
    delay_steps = env.cfg.env.fail_to_terminal_time_s / env.dt
    env.fail_buf[4:6] = delay_steps
    env.contact_forces = torch.zeros(9, 3, 3)
    env.feet_indices = torch.tensor([1, 2])
    env.contact_forces[:, 1:, 2] = 2.  # 非离地用例保证轮地支撑，不干扰其他判死检查。
    env.contact_forces[6:8, 0, 2] = 11.
    env.termination_contact_indices = torch.tensor([0])
    env.episode_length_buf = torch.zeros(9, dtype=torch.long)

    # 从 1 m 高空自由落下时，未接地环境不得推理、终止或累计起立/腾空计时；
    # 即便此时 pg_z 为正或 episode_length 已超过常规上限也应保持存活。
    env.projected_gravity[:, 2] = 1.0
    env.episode_length_buf[:] = env.max_episode_length + 1
    env.contact_forces[:] = 0
    env.check_termination()
    assert not env.get_policy_action_mask().any()
    assert not env.reset_buf.any() and not env.time_out_buf.any()
    assert (env.standup_steps == 0).all()
    assert (env.wheels_airborne_steps == 0).all()
    # 任一轮首次有效接触后，下一控制步才允许策略推理。
    env.contact_forces[0, 1, 2] = env.cfg.standup.wheel_contact_force_threshold + .1
    env.projected_gravity[:, 2] = -1.0
    env.episode_length_buf[:] = 0
    env.episode_length_buf[0] = env.max_episode_length + 1
    env.check_termination()
    assert env.get_policy_action_mask()[0] and not env.get_policy_action_mask()[1:].any()
    assert env.episode_length_buf[0] == 0
    print("PASS: 1 m free fall has no policy/termination before first wheel contact")

    env.projected_gravity[:, 2] = pg_z
    env.contact_forces[:] = 0
    env.contact_forces[:, 1:, 2] = 2.
    env.contact_forces[6:8, 0, 2] = 11.
    env.fail_buf[:] = 0
    env.fail_buf[4:6] = delay_steps
    env.episode_length_buf[-1] = env.max_episode_length + 1
    env.has_landed[:] = True
    env.check_termination()
    expected = torch.tensor([True, True, False, False, False, True, False, False, True])
    assert torch.equal(env.reset_buf, expected), env.reset_buf
    assert not env.time_out_buf[:8].any()
    assert env.fail_buf[4] == 0  # 未曾站起：保留原来的非正值倾倒豁免。
    print("PASS: pg_z>0 resets immediately before/after standing; zero is not positive")
    print("PASS: negative tilt/contact delay and timeout remain unchanged")

    # 已站起的触地状态仍须连续超过 1 秒，不能被改成全部立即判死。
    for _ in range(round(delay_steps) - 1):
        env.check_termination()
    assert env.fail_buf[6] == delay_steps
    assert not env.reset_buf[6]
    env.check_termination()
    assert env.reset_buf[6]
    env.contact_forces[6, 0] = 0
    env.check_termination()
    assert env.fail_buf[6] == 0 and not env.reset_buf[6]
    print("PASS: delayed contact termination and failure-counter clearing")

    # 2 s 是连续未站稳的控制步，而非随机初始化的 episode_length_buf。
    deadline_steps = round(env.cfg.standup.standup_timeout_s / env.dt)
    env.standup_steps[:] = deadline_steps
    env.standup_steps[0] = deadline_steps - 1
    env.standup_steps[[3, 6]] = 0
    env.has_stood[:] = False
    env.has_stood[2] = True
    env.standing_time[:] = 0
    env.standing_time[4] = env.cfg.standup.success_duration_s
    env.standing_time[5] = .1
    env.base_height[:] = .08
    env.base_height[[4, 5, 7]] = .3
    env.projected_gravity[:, 2] = -1.
    env.projected_gravity[6, 2] = 1e-6
    env.projected_gravity[7, 2] = -.5
    env.contact_forces[:] = 0
    env.contact_forces[:, 1:, 2] = 2.
    env.fail_buf[:] = 0
    env.episode_length_buf[:] = 0
    env.episode_length_buf[3] = env.max_episode_length // 2
    env.check_termination()
    expected = torch.tensor([False, True, True, False, False, True, True, True, True])
    assert torch.equal(env.reset_buf, expected), env.reset_buf
    assert not env.time_out_buf.any()  # 起立失败不能标记成 PPO 的超时截断。
    env.check_termination()
    assert env.reset_buf[0]  # 严格超过 2 s 才判死：当前为第 201 个控制步。
    assert env.reset_buf[2] and not env.reset_buf[3]
    assert env.standup_steps[4] == 0  # 当前重新站稳，才清零计时。
    print("PASS: exact 2 s boundary, no previous-standing exemption, timer independence")

    # 仅隔离父类物理重置，检查本任务的选定环境计时器是否清零。
    env.extras = {"episode": {}}
    env.wheels_airborne_steps[:] = 7
    env.has_landed[:] = True
    other_steps = env.standup_steps[2:].clone()
    other_air_steps = env.wheels_airborne_steps[2:].clone()
    with patch.object(Chuanliantui, "reset_idx", return_value=None):
        env.reset_idx(torch.tensor([0, 1]))
    assert (env.standup_steps[:2] == 0).all()
    assert (env.standing_time[:2] == 0).all() and not env.has_stood[:2].any()
    assert torch.equal(env.standup_steps[2:], other_steps)
    assert (env.wheels_airborne_steps[:2] == 0).all()
    assert not env.has_landed[:2].any()
    assert torch.equal(env.wheels_airborne_steps[2:], other_air_steps)
    print("PASS: selected reset clears the standup timer without touching other environments")

    # 当前成功标准：高度 >= 0.30 m 且 pg_z <= -0.90，持续时间仍为 0.5 s。
    assert env.cfg.standup.success_height == .30
    assert env.cfg.standup.success_projected_gravity_z == -.90
    assert env.cfg.standup.success_duration_s == .5
    env.base_height[:] = torch.tensor([.30, .2999, .30, .30, .31, .31, .30, .28, .30])
    env.projected_gravity[:, 2] = torch.tensor([-.90, -.90, -.8999, -.91, -.89, -.91, -.90, -.85, -1.])
    env.has_stood[:] = False
    env.standing_time[:] = .5
    env.standing_time[6] = .1
    env.standup_steps[:] = 0
    env.episode_length_buf[:] = 0
    env.check_termination()
    expected_stood = torch.tensor([True, False, False, True, False, True, False, False, True])
    assert torch.equal(env.has_stood, expected_stood), env.has_stood
    assert (env.standing_time[[1, 2, 4, 7]] == 0).all()
    assert not env.reset_buf.any()
    print("PASS: new height/pg_z equality boundaries, old criteria rejected, duration still required")

    # 已站稳 -> 轻微失稳 -> 完整站稳恢复 -> 再次失稳；不触发原1秒严重跌倒分支。
    env.base_height[:] = .31
    env.projected_gravity[:, 2] = -.95
    env.standing_time[:] = 0
    env.standup_steps[:] = 0
    hold_steps = round(env.cfg.standup.success_duration_s / env.dt) + 1
    for _ in range(hold_steps):
        env.check_termination()
    assert env.has_stood.all() and (env.standup_steps == 0).all()
    env.base_height[:] = .29
    for _ in range(100):
        env.check_termination()
    assert (env.standup_steps == 100).all() and not env.reset_buf.any()
    env.base_height[:] = .31
    env.check_termination()
    assert (env.standup_steps == 101).all()  # 瞬间达到高度/姿态不能抹掉失稳时间。
    for _ in range(hold_steps):
        env.check_termination()
    assert (env.standup_steps == 0).all() and not env.reset_buf.any()
    # 仅姿态不合格也会启动计时；历史成功标志保留，但不再屏蔽判死。
    env.projected_gravity[:, 2] = -.89
    for _ in range(deadline_steps):
        env.check_termination()
    assert (env.standup_steps == deadline_steps).all()
    assert env.has_stood.all() and not env.reset_buf.any()
    env.check_termination()
    assert env.reset_buf.all() and not env.time_out_buf.any()
    print("PASS: repeated standing-loss cycles, full recovery clears timer, 2 s rule remains active")

    # 双轮离地：当前 0.01 s 控制周期，19步不判死，第20步判死。
    env.base_height[:] = .31
    env.base_height[:2] = .08
    env.has_stood[:] = False
    env.has_stood[1] = True  # 对照尚未站起/曾站起，均不得豁免腾空判死。
    env.standing_time[:] = 0
    env.standup_steps[:] = 0
    env.wheels_airborne_steps[:] = 0
    env.fail_buf[:] = 0
    env.episode_length_buf[:] = 0
    env.projected_gravity[:, 2] = -1.
    env.contact_forces[:] = 0
    threshold = env.cfg.standup.wheel_contact_force_threshold
    env.contact_forces[2, 1, 2] = threshold + .1  # 仅左轮支撑。
    env.contact_forces[3, 2, 2] = threshold + .1  # 仅右轮支撑。
    env.contact_forces[[4, 8], 1:, 2] = threshold + .1
    env.contact_forces[5, 1:, 2] = threshold  # 等于阈值不算有效支撑。
    env.contact_forces[6, 1:, 0] = 100.  # 水平接触力不能冒充竖直支撑。
    air_limit = round(env.cfg.standup.wheels_airborne_timeout_s / env.dt)
    expected_air_failure = torch.tensor([True, True, False, False, False, True, True, True, False])
    # 离地第一步就有负奖励，无需等待离地计时或曾经站起。
    initial_air_steps = env.wheels_airborne_steps.clone()
    assert torch.equal(env._reward_wheels_airborne(), expected_air_failure.float())
    assert torch.equal(env.wheels_airborne_steps, initial_air_steps)  # 读奖励不能改变计时器。
    # 隔离其他奖励，走真实基类注册、dt缩放、裁剪、episode累加管线。
    env.num_envs = 9
    env.device = "cpu"
    env.rew_buf = torch.zeros(9)
    env.reward_scales = {"wheels_airborne": env.cfg.rewards.scales.wheels_airborne}
    env._prepare_reward_function()
    assert env.reward_names == ["wheels_airborne"]
    env.compute_reward()
    assert torch.allclose(env.rew_buf, -env.dt * expected_air_failure.float())
    assert torch.equal(env.episode_sums["wheels_airborne"], env.rew_buf)
    print("PASS: airborne penalty applies immediately, both before/after standing, with correct sign and dt")
    for _ in range(air_limit - 1):
        env.check_termination()
        assert not env.reset_buf.any()
    assert (env.wheels_airborne_steps[[0, 1, 5, 6, 7]] == air_limit - 1).all()
    env.check_termination()
    assert torch.equal(env.reset_buf, expected_air_failure)
    assert not env.time_out_buf.any()
    print("PASS: both wheels airborne for 0.2 s; single wheel support and force threshold boundaries")

    # 用第7个环境检验分段腾空不累计，任一轮恢复接触都会清零。
    env.contact_forces[7, 1, 2] = threshold + .1
    env.check_termination()
    assert env.wheels_airborne_steps[7] == 0 and not env.reset_buf[7]
    assert env._reward_wheels_airborne()[7] == 0
    for foot in (2, 1):
        env.contact_forces[7] = 0
        for _ in range(air_limit - 1):
            env.check_termination()
            assert not env.reset_buf[7]
        env.contact_forces[7, foot, 2] = threshold + .1
        env.check_termination()
        assert env.wheels_airborne_steps[7] == 0 and not env.reset_buf[7]
    print("PASS: interrupted airborne intervals never accumulate across either-wheel touchdown")
    print("Numerical checks only; no physics, training or GUI acceptance.")


if __name__ == "__main__":
    main()
