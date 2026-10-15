"""Flat V14 commands and reset/failure timing, independent of standup curriculum."""

from dataclasses import dataclass
import math

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from .model import ACTUATED_JOINTS


# Source 0.20–0.42 m exceeds this robot's static reach (≈0.21–0.37 m wheel-to-root,
# ~3.5 cm PD sag). Starting at 0.27 m keeps leg-link kneeling (≈0.167 m) outside
# the 10 cm velocity gate.
HEIGHT_COMMAND_RANGE = (.27, .34)


def consecutive_failure(counter, failed, steps=20):
    counter = torch.where(failed, (counter + 1).clamp_max(steps), 0)
    return counter, counter >= steps


@dataclass(kw_only=True)
class HuananhuCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (5., 15.)

    def build(self, env):
        return HuananhuCommand(self, env)


class HuananhuCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, device = self.num_envs, self.device
        self._command = torch.zeros(n, 3, device=device)
        self.sampled = self._command.clone()
        self.ground_command = torch.zeros(n, 2, device=device)
        self.heading_target = torch.zeros(n, device=device)
        self.heading = torch.zeros(n, device=device, dtype=torch.bool)
        self.standing = self.heading.clone()
        self.ground_start = self.heading.clone()
        self.numerical_failure = self.heading.clone()
        self.special_mode = torch.full((n,), -1, dtype=torch.long, device=device)
        self.failure_count = torch.zeros(n, dtype=torch.long, device=device)
        self.stable_time = torch.zeros(n, device=device)
        self.training_iteration = 0  # offset; runner restores it on resume
        self._resetting = False
        self.metrics = {name: torch.zeros(n, device=device) for name in (
            'ground_start', 'recovered', 'numerical_failure', 'upright_fraction',
            'height_error', 'velocity_error')}

    @property
    def command(self):
        return self._command

    def reset(self, env_ids):
        self._resetting = True
        result = super().reset(env_ids)
        self._resetting = False
        self.failure_count[env_ids] = 0
        self.numerical_failure[env_ids] = False
        self.stable_time[env_ids] = 0
        self.ground_command[env_ids] = torch.empty(len(env_ids), 2, device=self.device).uniform_(-1., 1.)
        return result

    def _resample_command(self, env_ids):
        n = len(env_ids)
        if not n:
            return
        draw = lambda low, high: torch.empty(n, device=self.device).uniform_(low, high)
        self.sampled[env_ids, 0] = draw(-2.7, 2.7)
        self.sampled[env_ids, 1] = draw(-2*math.pi, 2*math.pi)
        self.sampled[env_ids, 2] = draw(*HEIGHT_COMMAND_RANGE)
        self.heading_target[env_ids] = draw(-math.pi, math.pi)
        self.heading[env_ids] = draw(0., 1.) < .5
        standing = draw(0., 1.) < .1
        self.special_mode[env_ids] = -1
        eligible = ~standing & (self._env.episode_length_buf[env_ids] >= math.ceil(5. / self._env.step_dt))
        if self._resetting:
            eligible[:] = False
        iteration = self.training_iteration + self._env.common_step_counter // 24
        bucket = draw(0., 1.)
        cursor = 0.
        # Source disjoint buckets. Shuffling bucket order preserves this distribution.
        for mode, start, probability in ((0, 3000, .15), (1, 4000, .15), (2, 2000, .30)):
            if iteration < start:
                continue
            mask = eligible & (bucket >= cursor) & (bucket < cursor + probability)
            cursor += probability
            selected = env_ids[mask]
            self.special_mode[selected] = mode
            if mode == 2:
                self.sampled[selected, 0] = (draw(2., 3.) * torch.where(draw(0., 1.) < .5, -1., 1.))[mask]
            else:
                low, high = ((2., 3.25), (3.25, 4.5))[mode]
                self.sampled[selected, 0] = draw(-.1, .1)[mask]
                self.sampled[selected, 1] = (draw(low*math.pi, high*math.pi) * torch.where(draw(0., 1.) < .5, -1., 1.))[mask]
        # The source delegates normal envs to UniformVelocityCommand, which also
        # samples its 10% standing flag. Keep that second draw (19% at iteration 0).
        standing |= (self.special_mode[env_ids] < 0) & ~standing & (draw(0., 1.) < .1)
        self.standing[env_ids] = standing

    def _update_command(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self._command[ids] = self.sampled[ids]
        yaw = self._env.scene['robot'].data.heading_w[ids]
        heading_command = (5. * wrap_to_pi(self.heading_target[ids] - yaw)).clamp(-2*math.pi, 2*math.pi)
        use_heading = self.heading[ids] & (self.special_mode[ids] < 0)
        self._command[ids, 1] = torch.where(use_heading, heading_command, self._command[ids, 1])
        self._command[ids, :2] = torch.where(self.standing[ids, None], 0., self._command[ids, :2])
        ground_override = self.ground_start[ids] & (self._env.episode_length_buf[ids] < math.ceil(1.5 / self._env.step_dt))
        self._command[ids, :2] = torch.where(ground_override[:, None], self.ground_command[ids], self._command[ids, :2])

    def _update_metrics(self):
        pass  # Advanced once per transition in failure(), before autoreset.


def failure(env):
    state = env.command_manager.get_term('standup')
    robot = env.scene['robot']
    if not hasattr(state, 'motor_v_adr'):
        ids, _ = robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        state.motor_v_adr = robot.indexing.joint_v_adr[ids]
    velocity = env.sim.data.qvel[:, state.motor_v_adr]
    linear, angular = robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b
    finite = torch.isfinite(env.sim.data.qpos).all(-1) & torch.isfinite(env.sim.data.qvel).all(-1)
    numerical = ~finite | (velocity.abs().amax(-1) > 500.) | (linear.abs().amax(-1) > 100.) | (angular.abs().amax(-1) > 200.)
    state.numerical_failure.copy_(numerical)
    # atan2 retains >90 degree roll information that asin alone would lose.
    gravity = robot.data.projected_gravity_b
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
    contact = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1) > 1.
    tilted = (roll.abs() > math.radians(40)) | (pitch.abs() > math.radians(40))
    state.failure_count, delayed = consecutive_failure(state.failure_count, (contact | tilted) & ~numerical)
    height = getattr(state, 'height_above_ground', robot.data.root_link_pos_w[:, 2])
    stable = ~contact & ~tilted & (height > .20)
    state.stable_time = torch.where(stable, state.stable_time + env.step_dt, 0.)
    state.metrics['ground_start'].copy_(state.ground_start)
    state.metrics['recovered'].copy_(torch.maximum(state.metrics['recovered'], (state.stable_time >= .5).float()))
    state.metrics['numerical_failure'].copy_(numerical)
    state.metrics['upright_fraction'] += stable.float() * env.step_dt / env.cfg.episode_length_s
    state.metrics['height_error'] += (height - state.command[:, 2]).abs() * env.step_dt / env.cfg.episode_length_s
    state.metrics['velocity_error'] += (linear[:, 0] - state.command[:, 0]).abs() * env.step_dt / env.cfg.episode_length_s
    return delayed | numerical


def reset_pose(env, env_ids):
    """Sample independent legal legs from the retargeted, closed-chain pose bank."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    from .huananhu_poses import sample_reset_state
    state = env.command_manager.get_term('standup')
    ground = torch.rand(len(env_ids), device=env.device) < .5
    state.ground_start[env_ids] = ground
    sample_reset_state(env, env_ids, ground)
    if hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards.step = -1
