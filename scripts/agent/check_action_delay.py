"""CPU 检查真实源码中的动作队列片段；不创建仿真，不替代行为验收。"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import isaacgym  # 必须先于 torch
from isaacgym.torch_utils import torch_rand_float
import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[2] / "wheel_legged_gym/envs/base/legged_robot.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
ROBOT = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "LeggedRobot")
METHODS = {n.name: n for n in ROBOT.body if isinstance(n, ast.FunctionDef)}
GLOBALS = {"torch": torch, "np": np, "torch_rand_float": torch_rand_float}


def assigns(node, name):
    if not isinstance(node, ast.Assign):
        return False
    return any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == "self" and t.attr == name for t in node.targets)


def execute(nodes, env, **values):
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec")
    exec(code, GLOBALS, dict(self=env, **values))


def make_env(dt, maximum_ms, enabled=True):
    env = NS(num_envs=64, device="cpu", sim_params=NS(dt=dt),
             cfg=NS(env=NS(num_actions=6, send_timeouts=False),
                    domain_rand=NS(delay_ms_range=[0, maximum_ms],
                                   randomize_action_delay=enabled),
                    terrain=NS(curriculum=False), commands=NS(curriculum=False)))
    body = METHODS["_init_buffers"].body
    begin = next(i for i, n in enumerate(body) if assigns(n, "action_delay_idx"))
    end = next(i for i, n in enumerate(body) if assigns(n, "action_fifo"))
    execute(body[begin:end + 1], env)
    sampling = next(n for n in body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Attribute)
                    and n.test.attr == "randomize_action_delay")
    execute([sampling], env)
    env.actions = torch.zeros(64, 6)
    env.torques = torch.zeros_like(env.actions)
    env._compute_torques = lambda actions: actions
    return env


def advance(env, value):
    env.actions.fill_(value)
    loop = next(n for n in METHODS["step"].body if isinstance(n, ast.For))
    # 执行实际入队和动作选择，不复制队列实现；PD 用恒等探针显示选中的目标。
    execute([n for n in loop.body if assigns(n, "action_fifo") or assigns(n, "torques")], env)
    return env.torques


def check_delay(dt, maximum_ms, enabled=True):
    env = make_env(dt, maximum_ms, enabled)
    sampled = env.action_delay_idx
    assert sampled.min() >= 0 and sampled.max() < env.action_fifo.shape[1], "延迟采样越界"
    delays = list(range(round(maximum_ms / (dt * 1000)) + 1)) if enabled else [0]
    env.action_delay_idx[:] = torch.tensor([delays[i % len(delays)] for i in range(64)])
    # 每五个物理子步发布一次新动作，覆盖跨策略周期的延迟上界。
    for tick in range(20):
        actual = advance(env, tick // 5 + 1)
        expected = [0 if tick < delay else (tick - delay) // 5 + 1
                    for delay in env.action_delay_idx.tolist()]
        assert torch.equal(actual[:, 0], torch.tensor(expected, dtype=torch.float))
    print(f"PASS: dt={dt:g}s, range=0..{maximum_ms:g}ms, enabled={enabled}")


def check_reset():
    env = make_env(0.002, 10)
    env.action_fifo.fill_(7)
    env.actions.fill_(7)
    env.last_actions = torch.full((64, 6, 2), 7.0)
    for name in ("last_dof_vel", "feet_air_time", "last_dof_pos", "dof_pos",
                 "last_base_position", "base_position"):
        setattr(env, name, torch.zeros(64, 6))
    for name in ("episode_length_buf", "reset_buf", "fail_buf", "envs_steps_buf"):
        setattr(env, name, torch.zeros(64))
    env.obs_history_length = 5
    env.obs_history = torch.zeros(64, 30)
    env.extras, env.episode_sums = {}, {}
    env.compute_proprioception_observations = lambda: env.actions
    for name in ("_reset_dofs", "_reset_root_states", "_resample_commands"):
        setattr(env, name, lambda ids: None)
    namespace = dict(GLOBALS)
    exec(compile(ast.Module(body=[METHODS["reset_idx"]], type_ignores=[]), str(SOURCE), "exec"), namespace)
    ids = torch.tensor([1, 4])
    namespace["reset_idx"](env, ids)
    assert not env.action_fifo[ids].any(), "新回合残留旧延迟动作"
    assert not env.actions[ids].any() and not env.obs_history[ids].any(), "初始观测残留旧动作"
    assert (env.action_fifo[0] == 7).all() and (env.actions[0] == 7).all(), "误清未重置环境"
    env.action_delay_idx[:] = 5
    assert not advance(env, 0)[ids].any(), "重置后仍执行上一回合目标"
    print("PASS: partial reset clears delayed targets and initial action history")


if __name__ == "__main__":
    torch.manual_seed(0)
    failures = []
    for check in (lambda: check_delay(0.002, 10), lambda: check_delay(0.005, 10),
                  lambda: check_delay(0.002, 0), lambda: check_delay(0.002, 10, False),
                  check_reset):
        try:
            check()
        except (AssertionError, IndexError) as exc:
            failures.append(str(exc))
            print("FAIL:", exc)
    if failures:
        raise SystemExit(f"{len(failures)} queue checks failed")
    print("CPU queue checks only; training, Isaac playback and sim2sim remain separate.")
