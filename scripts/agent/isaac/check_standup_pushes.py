"""CPU 检查接地后的分级水平推力；不创建物理仿真，不替代行为验收。"""

from types import SimpleNamespace
from unittest.mock import patch

import isaacgym  # 必须先于 torch
from isaacgym import gymapi, gymtorch
import torch

from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup import (
    ChuanliantuiStandup,
)
from wheel_legged_gym.envs.chuanliantui_standup.chuanliantui_standup_config import (
    ChuanliantuiStandupCfg,
)


def check_push_contract_exists():
    required = ("_init_standup_pushes", "_update_standup_pushes", "_reset_standup_pushes")
    missing = [name for name in required if not callable(getattr(ChuanliantuiStandup, name, None))]
    assert not missing, "尚未实现接地后水平推力接口: " + ", ".join(missing)


class ForceBoundary:
    """只隔离 Isaac 的句柄查询及张量提交，不模拟推力调度或物理响应。"""

    def __init__(self):
        self.calls = []

    def find_actor_rigid_body_handle(self, env_handle, actor_handle, name):
        assert name == "base_link"
        return 2  # 故意不是刚体0。

    def apply_rigid_body_force_tensors(self, sim, forces, torques, space):
        assert space == gymapi.ENV_SPACE, "水平力必须固定在环境坐标系"
        assert torch.isfinite(forces).all() and torch.isfinite(torques).all()
        self.calls.append((forces.clone(), torques.clone()))


def initialize_cpu_fields(env, cfg, num_envs):
    env.cfg, env.num_envs, env.device = cfg, num_envs, "cpu"
    env.dt = .01
    env.sim_params = SimpleNamespace(dt=.002)
    env.num_bodies = 5
    env.envs, env.actor_handles = list(range(num_envs)), list(range(num_envs))
    env.gym, env.sim = ForceBoundary(), object()
    env.feet_indices = torch.tensor([3, 4])
    env.rigid_body_external_forces = torch.zeros(num_envs, 5, 3)
    env.rigid_body_external_torques = torch.zeros(num_envs, 5, 3)
    env.base_quat = torch.zeros(num_envs, 4)
    env.base_quat[:, 3] = 1.
    env.base_lin_vel = torch.zeros(num_envs, 3)
    env.base_ang_vel = torch.zeros(num_envs, 3)
    env.commands = torch.zeros(num_envs, 3)
    env.time_out_buf = torch.zeros(num_envs, dtype=torch.bool)
    env.reward_scales = {"orientation": cfg.rewards.scales.orientation * env.dt}
    env.extras = {"episode": {}}


def make_env(num_envs=4):
    check_push_contract_exists()
    cfg = ChuanliantuiStandupCfg()

    def skip_physics_init(env, cfg, *args):
        initialize_cpu_fields(env, cfg, num_envs)

    # 只隔离父类创建 Isaac 资源；执行真实 standup 构造器，检测漏挂初始化。
    with patch.object(Chuanliantui, "__init__", skip_physics_init):
        env = ChuanliantuiStandup(cfg, None, None, "cpu", True)
    assert hasattr(env, "standup_push_force"), "standup __init__ 未初始化推力状态"
    return env


def submit_substep(env):
    with patch.object(gymtorch, "unwrap_tensor", side_effect=lambda tensor: tensor):
        env._push_robots()


def ready(env):
    env.has_landed[:] = True
    env.has_stood[:] = True
    env.standing_time[:] = env.cfg.standup.success_duration_s
    env.standup_push_wait_steps[:] = 0


def check_initialization_and_timer_gate():
    env = make_env()
    assert env.cfg.domain_rand.push_robots, "起立配置必须启用此训练扰动"
    assert env.standup_push_body_index == 2, "不能假定 base_link 的刚体索引为0"
    assert env.standup_push_force.shape == (4, 3)
    for field in (env.standup_push_steps_left, env.standup_push_wait_steps, env.standup_push_count):
        assert field.shape == (4,) and field.dtype in (torch.int32, torch.int64)
    assert not env.standup_push_force.any() and not env.standup_push_steps_left.any()
    assert not env.standup_push_count.any()
    assert ((env.standup_push_wait_steps >= 300) & (env.standup_push_wait_steps <= 500)).all()

    env.standup_push_wait_steps[:] = 7
    env.has_landed[:] = torch.tensor([False, True, True, False])
    env.has_stood[:] = torch.tensor([False, False, True, True])
    env.pre_physics_step()  # 真实策略步钩子及父类奖励快照。
    assert torch.equal(env.standup_push_wait_steps, torch.tensor([7, 6, 6, 7]))
    assert not env.standup_push_force.any(), "尚未到期就施加了力"
    assert hasattr(env, "rwd_linVelTrackPrev") and hasattr(env, "rwd_angVelTrackPrev")
    print("PASS: 构造器初始化、非零 base_link 索引及仅接地后按策略步计时")


def check_landing_and_force_bands():
    env = make_env(4096)
    ready(env)
    # 0: 初始空中；1: 已接地未站起；2: 历史站起但当前失稳；3: 当前稳定。
    env.has_landed[:1024] = False
    env.has_stood[1024:2048] = False
    # 分别独立检查 has_stood 和当前持续时间两个稳定条件。
    env.standing_time[2048:3072] = env.cfg.standup.success_duration_s - .0001
    env.pre_physics_step()
    force = env.standup_push_force
    assert not force[:1024].any() and not env.standup_push_count[:1024].any()
    assert (env.standup_push_count[1024:] == 1).all()
    assert (force[:, 2] == 0).all(), "推力出现竖直分量"
    for group, lower, upper in ((slice(1024, 3072), 1., 5.), (slice(3072, 4096), 5., 15.)):
        xy = force[group, :2]
        magnitude = torch.linalg.vector_norm(xy, dim=1)
        assert ((magnitude >= lower - 1e-5) & (magnitude <= upper + 1e-5)).all()
        # 固定种子的大批样本检查避免把x/y独立方框采样误当均匀方向/模长。
        assert abs(magnitude.mean().item() - (lower + upper) / 2) < (upper - lower) * .05
        assert abs((magnitude < (lower + upper) / 2).float().mean().item() - .5) < .07
        direction = xy / magnitude[:, None]
        assert (direction.mean(dim=0).abs() < .07).all()
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            fraction = ((sx * xy[:, 0] > 0) & (sy * xy[:, 1] > 0)).float().mean()
            assert .18 < fraction < .32
    print("PASS: 空中不推、接地后1–5N、当前稳定5–15N及均匀方向/模长抽样")


def check_pulse_duration_impulse_and_interval():
    env = make_env(2)
    ready(env)
    original_force = None
    impulse = torch.zeros(2, 3)
    for tick in range(10):
        env.pre_physics_step()
        if original_force is None:
            original_force = env.standup_push_force.clone()
            assert (torch.linalg.vector_norm(original_force, dim=1) >= 5.).all()
        torch.testing.assert_close(env.standup_push_force, original_force, rtol=0, atol=0)
        assert (env.standup_push_count == 1).all(), "活动脉冲重复触发"
        state_before = (env.standup_push_steps_left.clone(), env.standup_push_wait_steps.clone())
        for _ in range(5):
            submit_substep(env)
            forces, torques = env.gym.calls[-1]
            expected = torch.zeros_like(forces)
            expected[:, 2] = original_force
            torch.testing.assert_close(forces, expected, rtol=0, atol=0)
            assert not torques.any()
            impulse += forces[:, 2] * .002
        assert torch.equal(env.standup_push_steps_left, state_before[0])
        assert torch.equal(env.standup_push_wait_steps, state_before[1]), "物理子步推进了策略计时"
        # 脉冲内暂时失稳、机体旋转，均不能重采样/旋转/截断当前环境系力。
        env.standing_time[:] = 0.
        env.base_quat[:] = torch.tensor([.5, .5, .5, .5])
    torch.testing.assert_close(impulse, original_force * .1, rtol=1e-6, atol=1e-6)
    env.pre_physics_step()  # t=.10，前十段[0,.10)已经完整受力。
    assert not env.standup_push_force.any(), "脉冲超过0.1s"
    submit_substep(env)
    assert not env.gym.calls[-1][0].any(), "结束后仍向物理引擎提交旧力"
    next_start = torch.full((2,), -1, dtype=torch.long)
    for tick in range(11, 512):
        env.pre_physics_step()
        new = (env.standup_push_count >= 2) & (next_start < 0)
        next_start[new] = tick
        if (next_start >= 0).all():
            break
    assert ((next_start - 10 >= 300) & (next_start - 10 <= 500)).all(), next_start
    print("PASS: 10策略步×5子步恒力、F×0.1冲量、失稳不中断及结束后完整3–5s间隔")


def check_partial_reset_and_disable():
    env = make_env(4)
    ready(env)
    env.pre_physics_step()
    submit_substep(env)
    selected, remaining = torch.tensor([1, 3]), torch.tensor([0, 2])
    fields = ("standup_push_force", "standup_push_steps_left", "standup_push_wait_steps", "standup_push_count")
    before = {name: getattr(env, name)[remaining].clone() for name in fields}
    # 保留真实 standup reset 课程统计和清理，隔离父类 Isaac 张量复位。
    with patch.object(Chuanliantui, "reset_idx", return_value=None):
        env.reset_idx(selected)
    for name in ("standup_push_force", "standup_push_steps_left", "standup_push_count"):
        assert not getattr(env, name)[selected].any(), name
    assert ((env.standup_push_wait_steps[selected] >= 300) & (env.standup_push_wait_steps[selected] <= 500)).all()
    assert not env.rigid_body_external_forces[selected].any(), "reset残留已提交的外力缓冲"
    for name, expected in before.items():
        torch.testing.assert_close(getattr(env, name)[remaining], expected, rtol=0, atol=0)
    submit_substep(env)
    assert not env.gym.calls[-1][0][selected].any()
    torch.testing.assert_close(env.gym.calls[-1][0][remaining, 2], before["standup_push_force"])
    empty_before = {name: getattr(env, name).clone() for name in fields}
    env._reset_standup_pushes(torch.empty(0, dtype=torch.long))
    for name, expected in empty_before.items():
        torch.testing.assert_close(getattr(env, name), expected, rtol=0, atol=0)

    count_before = env.standup_push_count.clone()
    env.cfg.domain_rand.push_robots = False
    for _ in range(12):
        env.pre_physics_step()
        assert not env.standup_push_force.any() and not env.standup_push_steps_left.any()
        assert not env.rigid_body_external_forces.any(), "关闭push后外力缓冲未清零"
    assert torch.equal(env.standup_push_count, count_before), "关闭时仍触发脉冲"
    calls_before = len(env.gym.calls)
    submit_substep(env)
    assert all(not force.any() for force, _ in env.gym.calls[calls_before:])
    print("PASS: 真实reset钩子的部分清理、其他环境保留、空reset及关闭后无残留")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_push_contract_exists()
    for check in (check_initialization_and_timer_gate, check_landing_and_force_bands,
                  check_pulse_duration_impulse_and_interval, check_partial_reset_and_disable):
        check()
    print("CPU tensor/API contract checks only; no physics, training or behavior acceptance.")
