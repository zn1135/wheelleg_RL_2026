"""Ground standup state, observations, rewards and terminations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import mdp as env_mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


STABLE_DURATION_S = 0.5
AIRBORNE_TIMEOUT_S = 0.2
FALLEN_TIMEOUT_S = 1.0


@dataclass(kw_only=True)
class StandupCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
    unlock_recovered_rate: float = 0.30
    unlock_window_episodes: int = 4096
    pre_unlock_target_height: float = 0.30
    post_unlock_target_height: float = 0.22
    pre_unlock_success_height: float = 0.28
    post_unlock_success_height: float = 0.20

    def build(self, env):
        return StandupCommand(self, env)


class StandupCommand(CommandTerm):
    """Fixed [vx, yaw, height] goal and episode-local standup state."""

    cfg: StandupCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self._command = torch.zeros(n, 3, device=self.device)
        self.has_landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.policy_active = torch.zeros_like(self.has_landed)
        self.has_stood = torch.zeros_like(self.has_landed)
        self.completed = torch.zeros_like(self.has_landed)
        self.standing_time = torch.zeros(n, device=self.device)
        self.airborne_time = torch.zeros(n, device=self.device)
        self.fallen_time = torch.zeros(n, device=self.device)
        self.unlocked = False
        self.completed_episodes = 0
        self.recovered_episodes = 0
        self.last_recovered_rate = 0.0

    @property
    def command(self):
        return self._command

    @property
    def target_height(self):
        return (
            self.cfg.post_unlock_target_height
            if self.unlocked else self.cfg.pre_unlock_target_height
        )

    @property
    def success_height(self):
        return (
            self.cfg.post_unlock_success_height
            if self.unlocked else self.cfg.pre_unlock_success_height
        )

    def _resample_command(self, env_ids):
        self._command[env_ids, 2] = self.target_height

    def _update_command(self, env_ids):
        self._command[:, 2] = self.target_height

    def _update_metrics(self):
        pass

    def update_curriculum(self, env_ids):
        completed = self.completed[env_ids]
        count = int(completed.sum().item())
        recovered = int((completed & self.has_stood[env_ids]).sum().item())
        if not self.unlocked:
            self.completed_episodes += count
            self.recovered_episodes += recovered
            if self.completed_episodes >= self.cfg.unlock_window_episodes:
                self.last_recovered_rate = self.recovered_episodes / self.completed_episodes
                self.completed_episodes = 0
                self.recovered_episodes = 0
                if self.last_recovered_rate >= self.cfg.unlock_recovered_rate:
                    self.unlocked = True
                    self._command[:, 2] = self.target_height
        return {
            "unlocked": float(self.unlocked),
            "recent_recovered_rate": self.last_recovered_rate,
            "target_height": self.target_height,
            "recovered_rate": recovered / count if count else 0.0,
        }

    def clear_episodes(self, env_ids):
        self.has_landed[env_ids] = False
        self.policy_active[env_ids] = False
        self.has_stood[env_ids] = False
        self.completed[env_ids] = False
        self.standing_time[env_ids] = 0.0
        self.airborne_time[env_ids] = 0.0
        self.fallen_time[env_ids] = 0.0

    def checkpoint_state(self):
        return {
            "unlocked": self.unlocked,
            "completed_episodes": self.completed_episodes,
            "recovered_episodes": self.recovered_episodes,
            "last_recovered_rate": self.last_recovered_rate,
        }

    def load_checkpoint_state(self, state):
        self.unlocked = bool(state["unlocked"])
        self.completed_episodes = int(state["completed_episodes"])
        self.recovered_episodes = int(state["recovered_episodes"])
        self.last_recovered_rate = float(state["last_recovered_rate"])
        self._command[:, 2] = self.target_height


def standup(env) -> StandupCommand:
    return env.command_manager.get_term("standup")


def _active(env, value):
    return standup(env).policy_active.float() * value


def clipped_reward(env, *, func, weight, params=None):
    """Source clips each weighted reward rate to +/-1 before multiplying dt."""
    return func(env, **(params or {})).clamp(-1.0 / abs(weight), 1.0 / abs(weight))


def reset_standup_state(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    standup(env).clear_episodes(env_ids)


def update_standup_curriculum(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    return standup(env).update_curriculum(env_ids)


def wheel_contact(env):
    force = env.scene["wheel_contact"].data.force
    assert force is not None and force.shape[1] == 2
    # ContactSensor's net wrench points from the primary wheel to the terrain.
    return force[:, :, 2].abs() > 1.0


def base_contact_force(env):
    force = env.scene["base_contact"].data.force
    assert force is not None and force.shape[1] == 1
    return torch.linalg.vector_norm(force[:, 0, :], dim=1)


def standup_failure(env):
    """Advance state exactly once per policy step, before reward evaluation."""
    state = standup(env)
    landed_now = ~state.has_landed & wheel_contact(env).any(dim=1)
    state.has_landed |= landed_now
    env.episode_length_buf[landed_now] = 0
    robot = env.scene["robot"]
    height = robot.data.root_link_pos_w[:, 2]
    gravity_z = robot.data.projected_gravity_b[:, 2]
    stable = (
        state.has_landed
        & (height >= state.success_height)
        & (gravity_z <= -0.90)
        & (base_contact_force(env) <= 0.1)
    )
    state.standing_time = torch.where(
        stable, state.standing_time + env.step_dt, torch.zeros_like(state.standing_time)
    )
    state.has_stood |= state.standing_time >= STABLE_DURATION_S
    airborne = state.has_landed & ~wheel_contact(env).any(dim=1)
    state.airborne_time = torch.where(
        airborne, state.airborne_time + env.step_dt, torch.zeros_like(state.airborne_time)
    )
    fallen = state.has_stood & ((base_contact_force(env) > 10.0) | (gravity_z > -0.1))
    state.fallen_time = torch.where(
        fallen, state.fallen_time + env.step_dt, torch.zeros_like(state.fallen_time)
    )
    failure = state.has_landed & (
        (gravity_z > 0.0) | (state.airborne_time >= AIRBORNE_TIMEOUT_S)
        | (state.fallen_time > FALLEN_TIMEOUT_S)
    )
    state.completed |= failure | (state.has_landed & (env.episode_length_buf >= env.max_episode_length))
    return failure


def standup_timeout(env):
    state = standup(env)
    return state.has_landed & (env.episode_length_buf >= env.max_episode_length)


def actor_observation(env):
    from .v14_observations import actor_observation as v14_actor
    return v14_actor(env)


def height_reward(env):
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    error_sq = torch.square(height - standup(env).command[:, 2])
    contact_ratio = (base_contact_force(env) / 5.0).clamp(0.0, 1.0)
    sigma = 0.0025 if standup(env).unlocked else 0.01
    value = torch.exp(-error_sq / sigma) * (1.0 - 0.8 * contact_ratio)
    return _active(env, value)


def orientation_penalty(env):
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    target = standup(env).target_height
    progress = ((height - 0.15) / (target - 0.15)).clamp(0.0, 1.0)
    gravity = env.scene["robot"].data.projected_gravity_b
    scale = 10.0 if standup(env).unlocked else 1.0
    value = scale * progress * torch.sum(torch.square(gravity[:, :2]), dim=1)
    return _active(env, value.clamp(max=1.0))


def base_contact_penalty(env):
    return _active(env, (base_contact_force(env) / 5.0).clamp(0.0, 1.0))


def base_airborne_reward(env):
    return _active(env, 1.0 - (base_contact_force(env) / 5.0).clamp(0.0, 1.0))


def wheels_airborne_penalty(env):
    return _active(env, (~wheel_contact(env).any(dim=1)).float())


def recovered_reward(env):
    return _active(env, (standup(env).standing_time / STABLE_DURATION_S).clamp(0.0, 1.0))


def nominal_leg_symmetry(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("lf0", "lf1", "rf0", "rf1"), preserve_order=True)
    q = robot.data.joint_pos[:, ids]
    # This geometric formula was derived in CAD coordinates, before the motor
    # zeros moved to horizontal. Passive knee coordinates are unchanged.
    q[:, [0, 2]] -= env.action_manager.get_term("pd").reference_positions[:, [0, 3]]
    off_hip, off_knee = 0.664720554456, 1.626002936793
    # Left is the +Y assembly (positive joint axis +Y); right is the -Y assembly.
    theta1 = torch.stack((-q[:, 0] + off_hip, q[:, 2] + off_hip), dim=1)
    theta2 = torch.stack((-q[:, 1] + off_knee, q[:, 3] + off_knee), dim=1)
    x = 0.21 * torch.cos(theta1) + 0.25 * torch.cos(theta1 + theta2)
    z = 0.21 * torch.sin(theta1) + 0.25 * torch.sin(theta1 + theta2)
    swing = torch.atan2(z, x) - torch.pi / 2
    return _active(env, torch.square(swing[:, 0] - swing[:, 1]).clamp(max=1.0 / 3.0))


def gated_torque_penalty(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel"), preserve_order=True)
    return _active(env, standup(env).has_stood.float() * torch.sum(
        torch.square(robot.data.joint_effort_target[:, ids]), dim=1
    ))


def gated_action_rate(env):
    """Huananhu raw first differences, enabled when the policy takes control."""
    return _active(env, env_mdp.action_rate_l2(env))


def gated_action_smooth(env, *, action_ids=(0, 1, 3, 4)):
    """Raw second differences for physical leg or wheel action channels."""
    manager = env.action_manager
    acceleration = (manager.action[:, action_ids] - 2 * manager.prev_action[:, action_ids]
                    + manager.prev_prev_action[:, action_ids])
    return _active(env, torch.square(acceleration).sum(dim=1))


def knee_limit_margin(env, *, margin=0.05):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("lf1", "rf1"), preserve_order=True)
    q = robot.data.joint_pos[:, ids]
    limits = robot.data.joint_pos_limits[:, ids]
    penetration = torch.maximum(limits[:, :, 0] + margin - q,
                                q - limits[:, :, 1] + margin).clamp_min(0.)
    return _stood(env, torch.square(penetration / margin).mean(dim=1))


def _stood(env, value):
    return _active(env, standup(env).has_stood.float() * value)


def track_still_linear(env, *, enhance: bool = False):
    vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    value = torch.exp(-torch.square(vx) / (2.5 if enhance else 0.25))
    return _stood(env, value - 1.0 if enhance else value)


def track_still_yaw(env, *, enhance: bool = False):
    yaw = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    value = torch.exp(-torch.square(yaw) / (2.5 if enhance else 0.25))
    return _stood(env, value - 1.0 if enhance else value)


def vertical_velocity_penalty(env):
    vel = env.scene["robot"].data.root_link_lin_vel_b[:, 2]
    return _stood(env, torch.square(vel).clamp(max=1.0))


def tilt_rate_penalty(env):
    vel = env.scene["robot"].data.root_link_ang_vel_b[:, :2]
    return _stood(env, torch.sum(torch.square(vel), dim=1).clamp(max=5.0))


def leg_velocity_penalty(env):
    robot = env.scene["robot"]
    # Match the source front-chain hip/knee motion, despite direct motor control.
    ids, _ = robot.find_joints(("lf0", "lf1", "rf0", "rf1"), preserve_order=True)
    return _stood(env, torch.sum(torch.square(robot.data.joint_vel[:, ids]), dim=1))


def joint_limit_penalty(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("lf0", "lf1", "rf0", "rf1"), preserve_order=True)
    position = robot.data.joint_pos[:, ids]
    limits = robot.data.soft_joint_pos_limits[:, ids, :]
    excess = (limits[:, :, 0] - position).clamp_min(0.0) + (
        position - limits[:, :, 1]
    ).clamp_min(0.0)
    return _active(env, torch.sum(excess, dim=1).clamp(max=1.0))
