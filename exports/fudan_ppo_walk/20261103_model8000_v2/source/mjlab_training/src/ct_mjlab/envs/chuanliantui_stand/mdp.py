"""Fixed commands, closed standing resets and one disturbance per episode."""

from dataclasses import dataclass
from functools import lru_cache
import math

import mujoco
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz

from ..chuanliantui.huananhu_poses import _front_kinematics, _mesh_vertices, make_pose_bank, GROUND_ROOT_HEIGHT
from ..chuanliantui.model import MODEL_XML, load_model, solve_standing_joint_positions, closed_chain_residual
from ..chuanliantui.multimode_mdp import MultimodeCommand, MultimodeCommandCfg, reset_pose as original_reset_pose


HEIGHT_COMMAND_M = .27
RESET_CLEARANCE_M = .002
RESET_TILT_RAD = math.radians(3.)
LEG_MOTORS = ("lf0", "lf00", "rf0", "rf00")
CURRICULUM_VERSION = "stand-performance-stages-v2"
LEGACY_CURRICULUM_VERSION = "stand-still-push-knockdown-v1"
STEPS_PER_ITERATION = 24


@lru_cache(maxsize=1)
def standing_pose(target_height=HEIGHT_COMMAND_M):
    """Solve wheel-supported geometry without changing the shared XML."""
    model = load_model()
    data = mujoco.MjData(model)
    wheel_bodies = {model.body(name).id for name in ("lfwheel", "rfwheel")}
    geoms = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    wheel_geoms = [i for i in geoms if model.geom_bodyid[i] in wheel_bodies]

    def solve(knee):
        _, phi = _front_kinematics(model, data, knee)
        front = {"left": (phi, knee), "right": (-phi, -knee)}
        positions = solve_standing_joint_positions(model, front_joint_targets=front)
        data.qpos[:] = model.qpos0
        data.qpos[:7] = (0., 0., 0., 1., 0., 0., 0.)
        for name, value in positions.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        bottom = min(float(_mesh_vertices(model, data, i)[:, 2].min()) for i in wheel_geoms)
        return -bottom, front, positions

    lower, upper = model.joint("lf1").range
    knee = brentq(lambda q: solve(q)[0] - target_height, lower + 1e-4, upper - 1e-4)
    height, front, positions = solve(knee)
    vertices = np.concatenate([_mesh_vertices(model, data, i) for i in geoms])
    # A linear height minimum is identical on the mesh union and its hull.
    # Cache only its extreme vertices; box corners would add excess clearance.
    clearance_vertices = vertices[ConvexHull(vertices).vertices]
    for name, value in positions.items():
        joint = model.joint(name)
        if joint.limited and not joint.range[0] <= value <= joint.range[1]:
            raise ValueError(f"Standing reset exceeds {name} limits")
    residual = closed_chain_residual(model, data)
    if residual > 1e-8:
        raise ValueError(f"Standing reset is not closed: {residual}")
    return {"height": height, "front_joint_targets": front,
            "joint_positions": positions, "clearance_vertices": clearance_vertices.tolist(),
            "closed_chain_residual_m": residual}


def supported_pose_mask(env):
    """Wheel-supported upright pose, without a velocity gate on learning."""
    robot = env.scene["robot"]
    data = robot.data
    gravity = data.projected_gravity_b
    tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
    # Contact sensors report robot-on-ground force, hence negative Z is support.
    support = -env.scene["wheel_contact"].data.force[:, :, 2]
    weight = (env.sim.model.body_mass[:, robot.indexing.body_ids].sum(-1)
              * float(np.linalg.norm(env.sim.mj_model.opt.gravity)))
    base_force = env.scene["base_contact"].data.force.norm(dim=-1).amax(-1)
    height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ((support > 5.).all(-1) & (support.sum(-1) >= .7 * weight)
            & (base_force <= 1.) & (tilt <= math.radians(25.))
            & ((height - HEIGHT_COMMAND_M).abs() <= .03))


def standing_mask(env):
    """Physical standing gate; callers separately exclude recover/airborne."""
    data = env.scene["robot"].data
    return (supported_pose_mask(env)
            & (data.root_link_lin_vel_b[:, :2].norm(dim=-1) <= .05)
            & (data.root_link_ang_vel_b[:, :2].abs().amax(-1) <= .5))


@dataclass(kw_only=True)
class StandCommandCfg(MultimodeCommandCfg):
    zero_hold_probability: float = 1.
    ground_start_probability: float = .5
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    disturbances: bool = True
    curriculum_stage: int | None = None

    def build(self, env):
        return StandCommand(self, env)


class StandCommand(MultimodeCommand):
    def __init__(self, cfg, env):
        if cfg.curriculum_stage not in (None, 0, 1, 2):
            raise ValueError("Standing curriculum stage must be 0, 1 or 2")
        super().__init__(cfg, env)
        # New runs start easy; compatible checkpoints restore their actual stage.
        self.curriculum_stage = 0 if cfg.curriculum_stage is None else cfg.curriculum_stage
        self.strict_stable_time = torch.zeros(self.num_envs, device=self.device)
        self.quiet_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.settle_required = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ground_clearance_lift = torch.zeros(self.num_envs, device=self.device)
        self.hold_anchor_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.hold_anchor_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 0=quiet, 1=weak velocity, 2=strong velocity, 3=knockdown.
        self.disturbance_kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.disturbance_time = torch.zeros(self.num_envs, device=self.device)
        self.push_applied = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.push_skipped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.push_time = torch.full((self.num_envs,), -1., device=self.device)
        self.leg_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self.leg_absolute_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self.leg_speed_peak = torch.zeros(self.num_envs, device=self.device)
        self.leg_velocity_count = 0
        self.leg_velocity_step = -1
        self.metrics["curriculum_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["disturbance_applied"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["disturbance_skipped"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strict_standing_fraction"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def effective_iteration(self):
        return self.training_iteration + self._env.common_step_counter // STEPS_PER_ITERATION

    def assign_disturbances(self, env_ids):
        stage = self.curriculum_stage
        draw = torch.rand(len(env_ids), device=self.device)
        kind = torch.zeros_like(env_ids)
        if self.cfg.disturbances and stage == 1:
            kind[draw >= .5] = 1
        elif self.cfg.disturbances and stage == 2:
            kind[(draw >= .5) & (draw < .75)] = 2
            kind[draw >= .75] = 3
        self.disturbance_kind[env_ids] = kind
        self.disturbance_time[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(5., 10.)
        self.push_applied[env_ids] = False
        self.push_skipped[env_ids] = False
        self.push_time[env_ids] = -1.

    def reset(self, env_ids):
        result = super().reset(env_ids)
        self.strict_stable_time[env_ids] = 0.
        self.quiet_ready[env_ids] = False
        self.hold_anchor_valid[env_ids] = False
        # Reset events run before command.reset; retain their selected start mode.
        self.settle_required[env_ids] = self.ground_start[env_ids]
        self.leg_velocity_sum[env_ids] = 0.
        self.leg_absolute_velocity_sum[env_ids] = 0.
        self.leg_speed_peak[env_ids] = 0.
        self.assign_disturbances(env_ids)
        return result

    def _resample_command(self, env_ids):
        self.sampled[env_ids] = torch.tensor((0., 0., HEIGHT_COMMAND_M), device=self.device)
        self.standing[env_ids] = True
        self.heading[env_ids] = False
        self.special_mode[env_ids] = -1
        self.zero_hold[env_ids] = True

    def advance(self):
        super().advance()
        self.metrics["curriculum_stage"].fill_(self.curriculum_stage)
        stable = standing_mask(self._env) & ~self.recover & ~self.airborne
        self.strict_stable_time = torch.where(stable, self.strict_stable_time + self._env.step_dt, 0.)
        self.quiet_ready.copy_(self.strict_stable_time >= .5)
        # Keep all hold rewards off throughout get-up, including the final settling.
        self.settle_required |= self.recover | self.fallen
        self.settle_required &= ~self.quiet_ready
        self.metrics["strict_standing_fraction"] += (
            stable.float() * self._env.step_dt / self._env.cfg.episode_length_s)


def reset_pose(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    state = env.command_manager.get_term("standup")
    probability = state.cfg.ground_start_probability
    state.ground_clearance_lift[env_ids] = 0.
    # No extra random draw in the pinned near-start acceptance path.
    if probability in (0., 1.):
        ground = torch.full((len(env_ids),), probability == 1., device=env.device)
    else:
        ground = torch.rand(len(env_ids), device=env.device) < probability
    ground_ids, standing_ids = env_ids[ground], env_ids[~ground]
    if len(ground_ids):
        original_reset_pose(env, ground_ids, ground_probability=1.)
        _clear_ground_penetration(env, ground_ids)
        state.settle_required[ground_ids] = True
    if len(standing_ids):
        _reset_near_standing_pose(env, standing_ids)
        state.settle_required[standing_ids] = False


def _clear_ground_penetration(env, env_ids):
    """Reuse source-bank minima; preserve its sampled joints and motor velocities."""
    robot = env.scene["robot"]
    state = env.command_manager.get_term("standup")
    if not hasattr(env, "_stand_ground_clearance"):
        bank = make_pose_bank(cache_path=MODEL_XML.parent / "huananhu_pose_bank.npz")
        tables = []
        for side, names in (("left", ("lf0", "lf1")), ("right", ("rf0", "rf1"))):
            ids, _ = robot.find_joints(names, preserve_order=True)
            poses = np.concatenate([bank.positions[mode][side][:, :2] for mode in ("positive", "negative")])
            minima = np.concatenate([bank.ground_min_z[mode][side] for mode in ("positive", "negative")])
            tables.append((robot.indexing.joint_q_adr[ids],
                           torch.tensor(poses, device=env.device), torch.tensor(minima, device=env.device)))
        env._stand_ground_clearance = (tables, GROUND_ROOT_HEIGHT + bank.base_tilt_min_z)
    tables, base_minimum = env._stand_ground_clearance
    minimum = torch.full((len(env_ids),), base_minimum, device=env.device)
    for addresses, poses, minima in tables:
        actual = env.sim.data.qpos[env_ids][:, addresses]
        error, rows = (actual[:, None, :] - poses[None, :, :]).abs().amax(-1).min(-1)
        if bool((error > 2e-6).any()):
            raise ValueError("Ground reset did not match the original closed-chain pose bank")
        minimum = torch.minimum(minimum, minima[rows])
    lift = (RESET_CLEARANCE_M - minimum).clamp_min(0.)
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :7] = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    root[:, 2] += lift
    root[:, 7:13] = 0.
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    state.ground_clearance_lift[env_ids] = lift


def _reset_near_standing_pose(env, env_ids, *, pose=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    state = env.command_manager.get_term("standup")
    state.ground_start[env_ids] = False
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    count = len(env_ids)
    roll = torch.empty(count, device=env.device).uniform_(-RESET_TILT_RAD, RESET_TILT_RAD)
    pitch = torch.empty(count, device=env.device).uniform_(-RESET_TILT_RAD, RESET_TILT_RAD)
    yaw = torch.empty(count, device=env.device).uniform_(-math.pi, math.pi)
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    pose = standing_pose() if pose is None else pose
    vertices = torch.tensor(pose["clearance_vertices"], device=env.device, dtype=joint_pos.dtype)
    world_up = torch.tensor((0., 0., 1.), device=env.device, dtype=joint_pos.dtype).expand(count, -1)
    local_up = quat_apply_inverse(quat, world_up)
    # Preserve historical reset heights while policy matrix products remain IEEE.
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        minimum_z = (local_up @ vertices.T).amin(-1)
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, 2] += RESET_CLEARANCE_M - minimum_z
    root[:, 3:7] = quat
    root[:, 7:13] = 0.
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    if hasattr(env, "_huananhu_rewards"):
        env._huananhu_rewards.step = -1


def _mark_disturbance(env, env_ids):
    state = env.command_manager.get_term("standup")
    state.push_applied[env_ids] = True
    state.push_time[env_ids] = env.episode_length_buf[env_ids] * env.step_dt
    state.strict_stable_time[env_ids] = 0.
    state.quiet_ready[env_ids] = False
    state.hold_anchor_valid[env_ids] = False
    state.metrics["disturbance_applied"][env_ids] += 1.


def apply_velocity_push(env, env_ids, velocity_xy):
    """Set world XY root velocity; retain vertical and angular velocities."""
    robot = env.scene["robot"]
    velocity = robot.data.root_link_vel_w[env_ids].clone()
    velocity[:, :2] = velocity_xy
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    _mark_disturbance(env, env_ids)


def apply_knockdown(env, env_ids, angular_xy):
    """Set world roll/pitch angular velocity once, without changing pose."""
    robot = env.scene["robot"]
    velocity = robot.data.root_link_vel_w[env_ids].clone()
    velocity[:, 3:5] = angular_xy
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    _mark_disturbance(env, env_ids)


def disturb_once(env, env_ids):
    state = env.command_manager.get_term("standup")
    elapsed = env.episode_length_buf * env.step_dt
    missed = ((state.disturbance_kind > 0) & ~state.push_applied & ~state.push_skipped
              & (elapsed > 10.))
    state.push_skipped |= missed
    state.metrics["disturbance_skipped"] += missed.float()
    eligible = ((state.disturbance_kind > 0) & ~state.push_applied & ~state.push_skipped
                & (elapsed >= state.disturbance_time) & (elapsed <= 10.)
                & (state.strict_stable_time >= .5) & standing_mask(env)
                & ~state.recover & ~state.airborne & ~env.termination_manager.dones)
    for kind, magnitude in ((1, .3), (2, .6)):
        ids = (eligible & (state.disturbance_kind == kind)).nonzero(as_tuple=False).flatten()
        if len(ids):
            values = torch.empty(len(ids), 2, device=env.device).uniform_(-magnitude, magnitude)
            apply_velocity_push(env, ids, values)
    ids = (eligible & (state.disturbance_kind == 3)).nonzero(as_tuple=False).flatten()
    if len(ids):
        angle = torch.rand(len(ids), device=env.device) * (2. * math.pi)
        speed = torch.empty(len(ids), device=env.device).uniform_(12., 16.)
        values = torch.stack((speed * angle.cos(), speed * angle.sin()), dim=-1)
        apply_knockdown(env, ids, values)
