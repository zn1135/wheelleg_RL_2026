"""Offline, physically closed V14-style reset poses for the CAD robot.

The source leg length is mapped by normalized extension to the CAD linkage.
The source swing angle (and, for ground poses, the height/length ratio) is
preserved.  Expensive MuJoCo closure runs only while building the bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import brentq
import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .model import MODEL_XML


LEFT_JOINTS = (
    "lf0", "lf1", "lfwheel", "left_spring1_joint", "left_spring2_joint",
    "lf00", "lf01", "lf02", "lf03",
)
RIGHT_JOINTS = (
    "rf0", "rf1", "rfwheel", "right_spring1_joint", "right_spring2_joint",
    "rf00", "rf01", "rf02", "rf03",
)
POSE_JOINT_NAMES = LEFT_JOINTS + RIGHT_JOINTS
MODES = ("normal", "positive", "negative")
NORMAL_ROOT_HEIGHT = 0.38
GROUND_ROOT_HEIGHT = 0.25
_TILT = float(np.arccos(np.cos(0.15) ** 2))
_CLEARANCE = 0.002
_KNEE_INTERIOR_MARGIN = 2e-5
_CACHE_VERSION = 4


@dataclass(frozen=True)
class PoseBank:
    """Arrays have shape (count, 9) per side, in LEFT/RIGHT_JOINTS order.

    ``ground_min_z`` is the lowest collision-mesh vertex for that side at a
    0.25 m, level root.  ``normal_tilt_min_z`` is a conservative lower bound
    at zero root height for any roll/pitch in [-0.15, 0.15] radians.
    """

    positions: dict[str, dict[str, np.ndarray]]
    ground_min_z: dict[str, dict[str, np.ndarray]]
    normal_tilt_min_z: dict[str, dict[str, np.ndarray]]
    base_tilt_min_z: float
    generation_stats: dict[str, dict[str, int]]

    def to_torch(self, device: str | torch.device) -> TorchPoseBank:
        return TorchPoseBank(
            {mode: {side: torch.as_tensor(self.positions[mode][side], device=device)
                    for side in ("left", "right")} for mode in MODES},
            {mode: {side: torch.as_tensor(self.normal_tilt_min_z[mode][side], device=device)
                    for side in ("left", "right")} for mode in MODES},
            self.base_tilt_min_z,
            {mode: {side: torch.as_tensor(self.ground_min_z[mode][side] - GROUND_ROOT_HEIGHT,
                                         device=device) for side in ('left', 'right')} for mode in MODES},
        )


@dataclass(frozen=True)
class TorchPoseBank:
    """GPU-resident pose tables; no MuJoCo or CPU transfer on the reset path."""

    positions: dict[str, dict[str, torch.Tensor]]
    normal_tilt_min_z: dict[str, dict[str, torch.Tensor]]
    base_tilt_min_z: float
    level_min_z: dict[str, dict[str, torch.Tensor]]

    def sample(self, mode: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently sample left/right rows; return (joint_pos, root_z).

        Columns follow POSE_JOINT_NAMES.  Ground signs may differ between
        legs: call ``sample_sides(left_mode, right_mode, count)`` for that case.
        """
        return self.sample_sides(mode, mode, count)

    def sample_sides(
        self, left_mode: str, right_mode: str, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, root_z, _, _ = self.sample_with_bounds(left_mode, right_mode, count)
        return positions, root_z

    def sample_with_bounds(self, left_mode, right_mode, count):
        """Return sampled rows and per-leg clearance bounds with the same draws."""
        if left_mode not in MODES or right_mode not in MODES:
            raise ValueError(f"Unknown V14 reset mode: {left_mode}/{right_mode}")
        if (left_mode == "normal") != (right_mode == "normal"):
            raise ValueError("Normal/ground mode is selected per robot, not per leg")
        left = self.positions[left_mode]["left"]
        right = self.positions[right_mode]["right"]
        left_id = torch.randint(left.shape[0], (count,), device=left.device)
        right_id = torch.randint(right.shape[0], (count,), device=right.device)
        joint_pos = torch.cat((left[left_id], right[right_id]), dim=1)
        if left_mode == right_mode == "normal":
            left_low = self.normal_tilt_min_z["normal"]["left"][left_id]
            right_low = self.normal_tilt_min_z["normal"]["right"][right_id]
            minimum = torch.minimum(left_low, right_low).clamp(max=self.base_tilt_min_z)
            root_z = torch.maximum(torch.full_like(minimum, NORMAL_ROOT_HEIGHT), _CLEARANCE - minimum)
        else:
            root_z = torch.full((count,), GROUND_ROOT_HEIGHT, device=left.device, dtype=left.dtype)
        level = torch.stack((self.level_min_z[left_mode]['left'][left_id],
                             self.level_min_z[right_mode]['right'][right_id]), -1)
        tilt = torch.stack((self.normal_tilt_min_z[left_mode]['left'][left_id],
                            self.normal_tilt_min_z[right_mode]['right'][right_id]), -1)
        return joint_pos, root_z, level, tilt


def ground_clearance_height(level, tilt, roll, pitch, requested_height, base_tilt_min_z):
    """Conservative interpolation for actual tilt within the bank's 0.15 rad range.

    The sine ratio bounds every vertex's downward displacement, including all
    wheel spin angles. Rigid root transforms preserve every closed-chain loop.
    """
    angle = torch.acos((roll.cos() * pitch.cos()).clamp(-1., 1.))
    fraction = (angle.sin() / np.sin(_TILT)).clamp(0., 1.)
    minimum = (level + fraction[:, None] * (tilt - level)).amin(-1).clamp(max=base_tilt_min_z)
    return torch.maximum(requested_height, _CLEARANCE - minimum)


def make_pose_bank(*, cache_path=None):
    """Read the locally verified CAD bank; never generate poses at runtime."""
    path = Path(__file__).resolve().parents[1] / 'reset_poses.npz'
    with np.load(path, allow_pickle=False) as bank:
        if str(bank['model_hash']) != hashlib.sha256(MODEL_XML.read_bytes()).hexdigest():
            raise ValueError('Reset pose bank asset mismatch')
        if tuple(bank['joint_names'].tolist()) != POSE_JOINT_NAMES:
            raise ValueError('Reset pose bank joint order mismatch')
        return PoseBank(
            {mode: {side: bank[f'{mode}_{side}_pos'].copy() for side in ('left','right')} for mode in MODES},
            {mode: {side: bank[f'{mode}_{side}_ground'].copy() for side in ('left','right')} for mode in MODES},
            {mode: {side: bank[f'{mode}_{side}_tilt'].copy() for side in ('left','right')} for mode in MODES},
            float(bank['base_tilt']), {})


def sample_reset_state(env, env_ids: torch.Tensor | None, ground_bool: torch.Tensor) -> None:
    """Write V14-style reset state for selected envs, using a GPU-resident bank.

    Ground selection is per environment.  Its positive/negative angle branch
    is selected independently for each leg with 0.6/0.4 conditional weights.
    Normal root roll/pitch/yaw are uniform in +/-0.15,+/-0.15,+/-pi;
    ground roots default to level at 0.25 m, with optional bounded rigid-body
    perturbations and collision clearance correction. Passive speeds are zero;
    active motor speeds are sampled independently without projecting velocity
    onto the closed-chain tangent space, matching the source reset convention.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    count = len(env_ids)
    if count == 0:
        return
    ground = torch.as_tensor(ground_bool, device=env.device, dtype=torch.bool)
    if ground.ndim == 0:
        ground = ground.expand(count)
    if ground.shape != (count,):
        raise ValueError(f"ground_bool must have shape ({count},), got {tuple(ground.shape)}")

    bank = getattr(env, "_huananhu_pose_bank", None)
    if bank is None:
        cache = MODEL_XML.parent / "huananhu_pose_bank.npz"
        bank = make_pose_bank(cache_path=cache).to_torch(env.device)
        env._huananhu_pose_bank = bank
    normal_pos, normal_z = bank.sample("normal", count)
    positive_pos, _, positive_level, positive_tilt = bank.sample_with_bounds('positive', 'positive', count)
    negative_pos, _, negative_level, negative_tilt = bank.sample_with_bounds('negative', 'negative', count)
    left_positive = torch.rand(count, device=env.device) < 0.6
    right_positive = torch.rand(count, device=env.device) < 0.6
    left = torch.where(left_positive[:, None], positive_pos[:, :9], negative_pos[:, :9])
    right = torch.where(right_positive[:, None], positive_pos[:, 9:], negative_pos[:, 9:])
    joint_pos = torch.where(ground[:, None], torch.cat((left, right), dim=1), normal_pos)
    root_z = torch.where(ground, torch.full_like(normal_z, GROUND_ROOT_HEIGHT), normal_z)

    robot = env.scene["robot"]
    name_to_col = {name: i for i, name in enumerate(POSE_JOINT_NAMES)}
    joint_names = tuple(robot.joint_names)
    if set(joint_names) != set(POSE_JOINT_NAMES):
        raise RuntimeError(f"Unexpected robot joint names for pose bank: {joint_names}")
    joint_pos = joint_pos[:, [name_to_col[name] for name in joint_names]].clone()
    joint_vel = torch.zeros_like(joint_pos)
    for name in ("lf0", "lf00", "rf0", "rf00"):
        joint_vel[:, joint_names.index(name)] = (torch.rand(count, device=env.device) * 2 - 1) * (np.pi / 2)
    for name in ("lfwheel", "rfwheel"):
        joint_pos[:, joint_names.index(name)] = (torch.rand(count, device=env.device) * 2 - 1) * (2 * np.pi)
        joint_vel[:, joint_names.index(name)] = (torch.rand(count, device=env.device) * 2 - 1) * 50.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += root_z
    roll = (torch.rand(count, device=env.device) * 2 - 1) * 0.15
    pitch = (torch.rand(count, device=env.device) * 2 - 1) * 0.15
    yaw = (torch.rand(count, device=env.device) * 2 - 1) * np.pi
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    identity = torch.zeros_like(quat)
    identity[:, 0] = 1.0
    root_state[:, 3:7] = torch.where(ground[:, None], identity, quat)
    command = getattr(env, 'command_manager', None)
    cfg = command.get_term('standup').cfg if command is not None else None
    probability = getattr(cfg, 'ground_randomization_probability', 0.)
    if hasattr(env, '_recover_reset_randomized'):
        env._recover_reset_randomized[env_ids] = False
        env._recover_reset_height_lift[env_ids] = 0.
    if probability > 0.:
        randomized = ground & (torch.rand(count, device=env.device) < probability)
        maximum = cfg.ground_tilt_max_rad
        ground_roll = (torch.rand(count, device=env.device) * 2 - 1) * maximum
        ground_pitch = (torch.rand(count, device=env.device) * 2 - 1) * maximum
        ground_yaw = (torch.rand(count, device=env.device) * 2 - 1) * cfg.ground_yaw_max_rad
        offset = torch.empty(count, device=env.device).uniform_(*cfg.ground_height_offset_range)
        ground_quat = quat_from_euler_xyz(ground_roll, ground_pitch, ground_yaw)
        level = torch.stack((torch.where(left_positive, positive_level[:, 0], negative_level[:, 0]),
                             torch.where(right_positive, positive_level[:, 1], negative_level[:, 1])), -1)
        tilt = torch.stack((torch.where(left_positive, positive_tilt[:, 0], negative_tilt[:, 0]),
                            torch.where(right_positive, positive_tilt[:, 1], negative_tilt[:, 1])), -1)
        requested = torch.full_like(root_z, GROUND_ROOT_HEIGHT) + offset
        safe_height = ground_clearance_height(level, tilt, ground_roll, ground_pitch, requested,
                                              bank.base_tilt_min_z)
        root_state[:, 3:7] = torch.where(randomized[:, None], ground_quat, root_state[:, 3:7])
        root_state[:, 2] = torch.where(randomized, env.scene.env_origins[env_ids, 2] + safe_height,
                                     root_state[:, 2])
        if not hasattr(env, '_recover_reset_randomized'):
            env._recover_reset_randomized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env._recover_reset_height_lift = torch.zeros(env.num_envs, device=env.device)
        env._recover_reset_randomized[env_ids] = randomized
        env._recover_reset_height_lift[env_ids] = torch.where(randomized, safe_height - requested, 0.)
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
