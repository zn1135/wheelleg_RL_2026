"""Task-specific observation and reward terms for flat velocity tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")


def root_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def track_linear_velocity(env: ManagerBasedRlEnv, *, command_name: str, std: float) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.sum(torch.square(command[:, :2] - robot.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_yaw_velocity(env: ManagerBasedRlEnv, *, command_name: str, std: float) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.square(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def upright(env: ManagerBasedRlEnv, *, std: float) -> torch.Tensor:
    gravity_b = env.scene["robot"].data.projected_gravity_b
    return torch.exp(-torch.sum(torch.square(gravity_b[:, :2]), dim=1) / std**2)


def nominal_height(env: ManagerBasedRlEnv, *, target: float, std: float) -> torch.Tensor:
    error = torch.square(env.scene["robot"].data.root_link_pos_w[:, 2] - target)
    return torch.exp(-error / std**2)


def native_torque_l2(
    env: ManagerBasedRlEnv, *, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_effort_target[:, asset_cfg.joint_ids]), dim=1)
