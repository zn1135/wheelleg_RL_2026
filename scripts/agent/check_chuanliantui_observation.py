#!/usr/bin/env python3
"""检查 chuanliantui 25 维 actor 观测、历史和 MuJoCo 部署构造契约。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM2SIM_PATH = REPO_ROOT / "sim2sim" / "mj_sim2sim_ct.py"
ENV_PATH = REPO_ROOT / "wheel_legged_gym" / "envs" / "chuanliantui" / "chuanliantui.py"
CONFIG_PATH = REPO_ROOT / "wheel_legged_gym" / "envs" / "chuanliantui" / "chuanliantui_config.py"
LEG_POSITION_INDICES = (0, 1, 3, 4)


def load_sim2sim_module():
    spec = importlib.util.spec_from_file_location("mj_sim2sim_ct", SIM2SIM_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_observation_methods():
    """以最小 stub 加载任务类，避免本检查触发 Isaac Gym 的 gymtorch JIT 编译。"""
    package_names = (
        "wheel_legged_gym",
        "wheel_legged_gym.envs",
        "wheel_legged_gym.envs.base",
        "wheel_legged_gym.envs.chuanliantui",
        "isaacgym",
    )
    saved = {name: sys.modules.get(name) for name in package_names}
    saved["wheel_legged_gym.envs.base.legged_robot"] = sys.modules.get(
        "wheel_legged_gym.envs.base.legged_robot"
    )
    saved["isaacgym.torch_utils"] = sys.modules.get("isaacgym.torch_utils")
    try:
        for name in package_names:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        legged_robot = types.ModuleType("wheel_legged_gym.envs.base.legged_robot")
        legged_robot.LeggedRobot = type("LeggedRobot", (), {})
        sys.modules[legged_robot.__name__] = legged_robot
        torch_utils = types.ModuleType("isaacgym.torch_utils")
        torch_utils.quat_rotate_inverse = lambda quat, vector: vector
        sys.modules[torch_utils.__name__] = torch_utils
        spec = importlib.util.spec_from_file_location(
            "wheel_legged_gym.envs.chuanliantui.chuanliantui", ENV_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.Chuanliantui
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def read_config_contract():
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChuanliantuiCfg")
    classes = {
        node.name: node for node in cfg.body
        if isinstance(node, ast.ClassDef) and node.name in {"env", "control", "sim"}
    }
    env = classes["env"]
    values = {}
    for node in env.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"num_observations", "obs_history_length", "num_privileged_obs"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    for class_name, field in (("control", "decimation"), ("sim", "dt")):
        node = next(
            item for item in classes[class_name].body
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == field
        )
        values[field] = ast.literal_eval(node.value)
    return values


def read_policy_encoder_obs_contract():
    """读取 PPO 的 encoder 输入定义，防止环境观测改了而网络仍用旧维度。"""
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    ppo_cfg = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChuanliantuiCfgPPO"
    )
    policy = next(
        node for node in ppo_cfg.body
        if isinstance(node, ast.ClassDef) and node.name == "policy"
    )
    assignment = next(
        node for node in policy.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "num_encoder_obs"
    )
    expression = assignment.value
    assert isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult)

    def dotted_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        assert isinstance(node, ast.Name)
        parts.append(node.id)
        return ".".join(reversed(parts))

    return {dotted_name(expression.left), dotted_name(expression.right)}


def main() -> None:
    contract = read_config_contract()
    assert contract == {
        "num_observations": 25,
        "obs_history_length": 5,
        "num_privileged_obs": 65,
        "decimation": 5,
        "dt": 0.002,
    }, contract
    assert read_policy_encoder_obs_contract() == {
        "ChuanliantuiCfg.env.obs_history_length",
        "ChuanliantuiCfg.env.num_observations",
    }
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            num_observations=contract["num_observations"],
            obs_history_length=contract["obs_history_length"],
            num_privileged_obs=contract["num_privileged_obs"],
        ),
        control=SimpleNamespace(decimation=contract["decimation"]),
        noise=SimpleNamespace(
            add_noise=True,
            noise_level=1.0,
            noise_scales=SimpleNamespace(dof_pos=0.02, dof_vel=1.5, ang_vel=0.2, gravity=0.05),
        ),
    )

    # 不创建 Isaac 仿真，直接用与环境等价的批量状态检查布局、保留索引和噪声切片。
    Chuanliantui = load_observation_methods()
    env = object.__new__(Chuanliantui)
    batch_size = 2
    env.cfg = cfg
    env.obs_scales = SimpleNamespace(
        lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05,
        dof_acc=0.0025, height_measurements=5.0, torque=0.05,
    )
    env.base_ang_vel = torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3)
    env.projected_gravity = torch.full((batch_size, 3), -1.0)
    env.commands = torch.full((batch_size, 3), 2.0)
    env.commands_scale = torch.tensor((2.0, 0.25, 5.0))
    env.dof_pos = torch.tensor(
        ((10.0, 11.0, 12.0, 13.0, 14.0, 15.0), (20.0, 21.0, 22.0, 23.0, 24.0, 25.0))
    )
    env.default_dof_pos = torch.zeros((batch_size, 6))
    env.dof_vel = torch.ones((batch_size, 6))
    env.actions = torch.full((batch_size, 6), 3.0)
    env.obs_buf = torch.zeros((batch_size, cfg.env.num_observations))

    obs = env.compute_proprioception_observations()
    assert obs.shape == (batch_size, 25)
    torch.testing.assert_close(obs[:, 9:13], env.dof_pos[:, LEG_POSITION_INDICES])
    assert not torch.any(obs == 12.0) and not torch.any(obs == 15.0)

    env.noise_scale_vec = env._get_noise_scale_vec(cfg)
    assert env.noise_scale_vec.shape == (25,)
    torch.testing.assert_close(env.noise_scale_vec[9:13], torch.full((4,), 0.02))
    torch.testing.assert_close(env.noise_scale_vec[13:19], torch.full((6,), 0.075))
    torch.testing.assert_close(env.noise_scale_vec[19:25], torch.zeros(6))

    # 基类 compute_observations 将 actor obs 接入 critic；平地 heights 为 1 维。
    assert 3 + 25 + 12 + 6 + 1 + 6 + 1 + 3 + 6 + 1 + 1 == 65
    assert 3 + 25 + 12 + 6 + 77 + 6 + 1 + 3 + 6 + 1 + 1 == 141
    assert 25 * 5 == 125
    assert contract["dt"] * contract["decimation"] == 0.01

    sim = load_sim2sim_module()
    dof_pos = np.array((10.0, 11.0, 12.0, 13.0, 14.0, 15.0))
    sim_obs = sim.build_obs(
        np.array((0.0, 0.0, 0.0, 1.0)),
        np.zeros(3), dof_pos, np.zeros(6), np.zeros(3), np.zeros(6),
    )
    assert sim_obs.shape == (25,)
    np.testing.assert_allclose(sim_obs[9:13], dof_pos[list(LEG_POSITION_INDICES)] - sim.DEFAULT_DOF_POS[list(LEG_POSITION_INDICES)])
    assert sim.NUM_ENCODER_OBS == 125
    assert sim.SIM_DT == contract["dt"]
    assert sim.DECIMATION == contract["decimation"]
    assert sim.SIM_DT * sim.DECIMATION == 0.01
    assert sim.STANDUP_START_HEIGHT == 0.15
    np.testing.assert_allclose(
        sim.STANDUP_INITIAL_DOF_POS,
        np.array((11.0, 0.0, 0.0, -11.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(
        sim.wrap_to_pi(sim.STANDUP_INITIAL_DOF_POS),
        np.array((11.0 - 4.0 * np.pi, 0.0, 0.0, -11.0 + 4.0 * np.pi, 0.0, 0.0)),
    )

    legacy_checkpoint = REPO_ROOT / "logs" / "chuanliantui" / "Sep08_12-56-46_new1_train_proxy_v1_resume" / "model_3000.pt"
    if legacy_checkpoint.is_file():
        try:
            sim.load_policy(str(legacy_checkpoint))
        except ValueError as error:
            assert "观测接口不匹配" in str(error)
        else:
            raise AssertionError("历史 27 维 checkpoint 不应加载为 25 维策略")

    print("chuanliantui 25 维观测、125 维历史、65 维 critic 与 MuJoCo 构造检查通过")


if __name__ == "__main__":
    main()
