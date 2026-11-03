"""Upright-first curriculum and Isaac failure timers on mjlab reset/control clocks."""

from dataclasses import dataclass
import math

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from ..chuanliantui.multimode_mdp import MultimodeCommand
from ..chuanliantui.standup_mdp import StandupCommand, base_contact_force
from ..chuanliantui_stand.mdp import StandCommand, StandCommandCfg


CURRICULUM_VERSION = "stand-v2-upright-before-height-v4"
RECOVERY_CONTRACT = "stand-v2-normal-wheel-weight-v1"
LEGACY_RECOVERY_CONTRACT = "stand-v2-normal-root40-height20-v1"
NORMAL_EACH_WHEEL_WEIGHT_FRACTION = .1
NORMAL_TOTAL_WHEEL_WEIGHT_FRACTION = .8
NORMAL_EACH_WHEEL_MIN_N = 5.
ZERO_COMMAND_CONTRACT = "stand-v2-zero-command-v1"
TRACKING_COMMAND_CONTRACT = "stand-v2-post-stand-half-vx01to03-v1"
SPEED_COMMAND_CONTRACT = "stand-v2-post-stand-half-vx15-slew-v2"
YAW_COMMAND_CONTRACT = "stand-v2-post-stand-vx-yaw-heading-hold-v3"
POSITION_COMMAND_CONTRACT = "stand-v2-post-stand-vx-yaw-position-hold-v4"


@dataclass(kw_only=True)
class StandV2CommandCfg(StandCommandCfg):
    # Like Isaac, every episode starts on the ground, so the unlock rate measures get-up.
    ground_start_probability: float = 1.
    disturbances: bool = False
    unlock_recovered_rate: float = .30
    unlock_window_episodes: int = 4096
    pre_unlock_target_height: float = .30
    post_unlock_target_height: float = .22
    pre_unlock_success_height: float = .20
    post_unlock_success_height: float = .20
    # 显式实验配方开关；旧零速模型和普通入口继续使用原配方。
    post_stand_tracking: bool = False
    speed_curriculum: bool = False
    tracking_weight_while_active: bool = False
    tracking_tight_term: bool = False
    tracking_tight_kernel_denominator: float = .02
    tracking_tight_weight: float = .5
    tracking_acceleration_range: tuple[float, float] = (.5, 2.)
    tracking_interval_range: tuple[float, float] = (2., 4.)
    speed_limits: tuple[float, ...] = (.5, 1., 1.5)
    speed_error_threshold: float = .30
    speed_promotion_segments: int = 256
    speed_promotion_fraction: float = .8
    yaw_control: bool = False
    yaw_rate_limit: float = .5
    yaw_acceleration: float = 1.
    heading_gain: float = 1.5
    position_hold: bool = False
    position_gain: float = 1.
    position_velocity_limit: float = .1
    position_tolerance: float = .02
    position_lock_speed: float = .05
    position_lock_duration: float = .5

    def build(self, env):
        if self.tracking_tight_term:
            from .rewards import configure_tight_reward
            configure_tight_reward(env.cfg, self)
        return StandV2Command(self, env)


class StandV2Command(StandCommand):
    target_height = StandupCommand.target_height
    success_height = StandupCommand.success_height
    update_curriculum = StandupCommand.update_curriculum

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.yaw_control:
            self.yaw_config()
        if cfg.position_hold:
            self.position_config()
        self.unlocked = False
        self.completed_episodes = self.recovered_episodes = 0
        self.last_recovered_rate = 0.
        self.has_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_stood = torch.zeros_like(self.has_landed)
        self.completed = torch.zeros_like(self.has_landed)
        self.standing_time = torch.zeros(self.num_envs, device=self.device)
        self.wheels_airborne_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.isaac_fallen_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.landed_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.observation_stable_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.observation_recovered = torch.zeros_like(self.has_landed)
        self.reward_active = torch.zeros_like(self.has_landed)
        self.tracking_episode = torch.zeros_like(self.has_landed)
        self.tracking_velocity = torch.zeros_like(self.standing_time)
        self.tracking_ready_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.tracking_active = torch.zeros_like(self.has_landed)
        self.tracking_yaw_rate = torch.zeros_like(self.standing_time)
        self.heading_hold = torch.zeros_like(self.has_landed)
        self.yaw_active = torch.zeros_like(self.has_landed)
        self._yaw_previous_command = torch.zeros_like(self.standing_time)
        self.position_anchor = torch.zeros((self.num_envs, 2), device=self.device)
        self.position_locked = torch.zeros_like(self.has_landed)
        self.position_ready_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.position_error_forward = torch.zeros_like(self.standing_time)
        self.position_error_lateral = torch.zeros_like(self.standing_time)
        self.position_velocity = torch.zeros_like(self.standing_time)
        self._last_tracking_step = -1
        self._last_failure_step = -1
        self.speed_stage = self.speed_segments = self.speed_successes = 0
        self.last_speed_success_fraction = 0.
        self.speed_curriculum_frozen = False
        self.tracking_acceleration = torch.zeros_like(self.standing_time)
        self.tracking_interval_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.tracking_transition = torch.full_like(self.wheels_airborne_steps, -1)
        self.segment_steps = torch.zeros_like(self.wheels_airborne_steps)
        self.segment_error = torch.zeros_like(self.standing_time)
        self.segment_started = torch.zeros_like(self.has_landed)
        self.segment_failed = torch.zeros_like(self.has_landed)
        self.segment_stage = torch.zeros_like(self.wheels_airborne_steps)
        self._speed_settle_steps = torch.zeros_like(self.wheels_airborne_steps)
        for name in ("isaac_curriculum_unlocked", "isaac_target_height", "isaac_recovered"):
            self.metrics[name] = torch.zeros_like(self.standing_time)

    def reset(self, env_ids):
        if self.cfg.speed_curriculum:
            self.segment_failed[env_ids] |= self.recover[env_ids] | self.fallen[env_ids]
            if hasattr(self, "_isaac_failure"):
                self.segment_failed[env_ids] |= self._isaac_failure[env_ids]
            self._finish_speed_segments(env_ids)
        curriculum = self.update_curriculum(env_ids)
        result = super().reset(env_ids)
        for name in ("has_landed", "has_stood", "completed", "observation_recovered", "reward_active"):
            getattr(self, name)[env_ids] = False
        self.standing_time[env_ids] = 0.
        for name in ("wheels_airborne_steps", "isaac_fallen_steps", "landed_steps", "observation_stable_steps"):
            getattr(self, name)[env_ids] = 0
        if hasattr(self, "_isaac_failure"):
            self._isaac_failure[env_ids] = False
        if hasattr(self._env, "_isaac_v2_rewards"):
            self._env._isaac_v2_rewards.reset(env_ids)
        self.metrics["isaac_curriculum_unlocked"].fill_(curriculum["unlocked"])
        self.metrics["isaac_target_height"].fill_(self.target_height)
        return result

    def _resample_command(self, env_ids):
        if self.cfg.position_hold:
            self.position_config()
            self._clear_position(env_ids)
        if self.cfg.yaw_control:
            self.yaw_config()
            self.tracking_yaw_rate[env_ids] = 0.
            self.heading_target[env_ids] = 0.
            self.heading_hold[env_ids] = self.yaw_active[env_ids] = False
            self._yaw_previous_command[env_ids] = self._command[env_ids, 1] = 0.
        self.sampled[env_ids] = torch.tensor((0., 0., self.target_height), device=self.device)
        self.standing[env_ids] = self.zero_hold[env_ids] = True
        self.heading[env_ids] = False
        self.special_mode[env_ids] = -1
        self.tracking_episode[env_ids] = False
        self.tracking_velocity[env_ids] = 0.
        self.tracking_ready_steps[env_ids] = 0
        self.tracking_active[env_ids] = False
        if self.cfg.post_stand_tracking:
            n = len(env_ids)
            self.tracking_episode[env_ids] = torch.rand(n, device=self.device) < .5
            magnitude = torch.empty(n, device=self.device).uniform_(.1, .3)
            direction = torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
            self.tracking_velocity[env_ids] = magnitude * direction
            self.zero_hold[env_ids] = ~self.tracking_episode[env_ids]
        if self.cfg.speed_curriculum:
            low, high = self.cfg.tracking_acceleration_range
            if not self.cfg.post_stand_tracking or not 0. < low <= high:
                raise ValueError("Speed curriculum requires tracking and positive acceleration range")
            self.tracking_acceleration[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)
            self.tracking_velocity[env_ids] = 0.
            self.tracking_interval_steps[env_ids] = 0
            self.tracking_transition[env_ids] = -1
            for name in ("segment_steps", "segment_error", "segment_started", "segment_failed"):
                getattr(self, name)[env_ids] = 0
            self._speed_settle_steps[env_ids] = 0

    def _update_command(self, env_ids):
        if self.cfg.speed_curriculum:
            self._speed_previous_command = self._command[:, 0].clone()
        if self.cfg.yaw_control:
            self._yaw_previous_command = self._command[:, 1].clone()
        # Update every environment immediately when the global curriculum unlocks.
        self.sampled[:, 2] = self.target_height
        super()._update_command(env_ids)
        if self.cfg.post_stand_tracking:
            # failure()已更新当前NORMAL；失稳立即清零，再就绪须重新连续保持50拍。
            ready = self.has_stood & ~self.recover & ~self.fallen
            advance = env_ids is None and self._last_tracking_step != self._env.common_step_counter
            if advance:
                self.tracking_ready_steps = torch.where(ready, self.tracking_ready_steps + 1, 0)
                self._last_tracking_step = self._env.common_step_counter
            ids = slice(None) if env_ids is None else env_ids
            active = self.tracking_episode & ready & (self.tracking_ready_steps >= round(.5 / self._env.step_dt))
            self.tracking_active[ids] = active[ids]
            if self.cfg.speed_curriculum:
                if advance:
                    self._advance_speed_command(active)
                else:
                    self._command[:, 0] = self._speed_previous_command
                if self.cfg.position_hold:
                    if not advance:
                        allowed = ready & (self.tracking_ready_steps >= round(.5 / self._env.step_dt)) & (self.tracking_velocity == 0.)
                        selected = torch.arange(self.num_envs, device=self.device)[ids]
                        self._clear_position(selected[~allowed[selected]])
                        self.tracking_active[ids] = (active | self.position_locked)[ids]
                    applied_active = self.tracking_active
                else:
                    applied_active = active
                self._command[ids, 0] = torch.where(applied_active, self._command[:, 0], 0.)[ids]
            else:
                self._command[ids, 0] = torch.where(active, self.tracking_velocity, 0.)[ids]
            if self.cfg.yaw_control:
                # 静止角色也锁航向；vx奖励仍只使用原tracking_active门控。
                yaw_ready = ready & (self.tracking_ready_steps >= round(.5 / self._env.step_dt))
                self.yaw_active[ids] = yaw_ready[ids]
                if advance:
                    self._advance_yaw_command(yaw_ready)
                else:
                    self._command[:, 1] = self._yaw_previous_command
                    self._command[ids, 1] = torch.where(yaw_ready, self._command[:, 1], 0.)[ids]
                    inactive = ~yaw_ready[ids]
                    selected = torch.arange(self.num_envs, device=self.device)[ids][inactive]
                    self.tracking_yaw_rate[selected] = 0.
                    self.heading_hold[selected] = False
            else:
                self._command[ids, 1] = 0.

    def speed_config(self):
        return {name: getattr(self.cfg, name) for name in ("tracking_acceleration_range",
            "tracking_interval_range", "speed_limits", "speed_error_threshold",
            "speed_promotion_segments", "speed_promotion_fraction")}

    def yaw_config(self):
        if not self.cfg.post_stand_tracking or not self.cfg.speed_curriculum:
            raise ValueError("Yaw control requires post-stand tracking and speed curriculum")
        config = {name: getattr(self.cfg, name) for name in
                  ("yaw_rate_limit", "yaw_acceleration", "heading_gain")}
        if any(not math.isfinite(value) or value <= 0. for value in config.values()):
            raise ValueError("Yaw control parameters must be finite and positive")
        if config["yaw_rate_limit"] < .1:
            raise ValueError("Yaw rate limit must be at least 0.1 rad/s")
        return config

    def checkpoint_state(self):
        saved = StandupCommand.checkpoint_state(self)
        if self.cfg.speed_curriculum:
            saved["speed_curriculum"] = dict(stage=self.speed_stage, segments=self.speed_segments,
                successes=self.speed_successes, last_success_fraction=self.last_speed_success_fraction,
                config=self.speed_config())
        if self.cfg.yaw_control:
            # 航向参考属于当前物理回合，不随模型恢复到新的复位状态。
            saved["yaw_control"] = self.yaw_config()
        if self.cfg.position_hold:
            saved['position_hold'] = self.position_config()
        return saved

    def load_checkpoint_state(self, saved):
        if self.cfg.position_hold and saved.get('position_hold') != self.position_config():
            raise ValueError('Position hold configuration differs from checkpoint')
        if not self.cfg.position_hold and 'position_hold' in saved:
            raise ValueError('Position hold checkpoint requires position hold')
        if self.cfg.yaw_control and saved.get("yaw_control") != self.yaw_config():
            raise ValueError("Yaw control configuration differs from checkpoint")
        if not self.cfg.yaw_control and "yaw_control" in saved:
            raise ValueError("Yaw control checkpoint requires yaw control")
        StandupCommand.load_checkpoint_state(self, saved)
        if self.cfg.position_hold:
            # 模型恢复不携带旧物理回合的停车点。
            self._clear_position(torch.arange(self.num_envs, device=self.device))
        if self.cfg.speed_curriculum:
            speed = saved["speed_curriculum"]
            if speed["config"] != self.speed_config():
                raise ValueError("Speed curriculum configuration differs from checkpoint")
            self.speed_stage = int(speed["stage"])
            self.speed_segments, self.speed_successes = int(speed["segments"]), int(speed["successes"])
            self.last_speed_success_fraction = float(speed["last_success_fraction"])

    def _finish_speed_segments(self, env_ids):
        # 只用接近本档上限的非零段晋级；失稳或提前结束算失败，不能被筛掉。
        eligible = self.segment_started[env_ids] & (self.segment_stage[env_ids] == self.speed_stage)
        steps = self.segment_steps[env_ids]
        mean_error = self.segment_error[env_ids] / steps.clamp_min(1)
        good = (~self.segment_failed[env_ids] & (steps >= round(.5 / self._env.step_dt))
                & (mean_error <= self.cfg.speed_error_threshold))
        if not self.speed_curriculum_frozen:
            self.speed_segments += int(eligible.sum())
            self.speed_successes += int((eligible & good).sum())
            if self.speed_segments >= self.cfg.speed_promotion_segments:
                self.last_speed_success_fraction = self.speed_successes / self.speed_segments
                if self.last_speed_success_fraction >= self.cfg.speed_promotion_fraction:
                    self.speed_stage = min(self.speed_stage + 1, len(self.cfg.speed_limits)-1)
                self.speed_segments = self.speed_successes = 0
        for name in ("segment_steps", "segment_error", "segment_started", "segment_failed"):
            getattr(self, name)[env_ids] = 0

    def _sample_speed_target(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.yaw_control:
            self._sample_yaw_speed_target(env_ids)
            return
        previous = self.tracking_velocity[env_ids]
        kind = torch.randint(0, 4, (len(env_ids),), device=self.device)
        # 0急走、1急停、2换向、3保持；只采样当前状态能够发生的转换。
        moving = previous != 0.
        kind = torch.where(~moving & (kind != 3), 0, kind)
        kind = torch.where(moving & (kind == 0), 3, kind)
        vmax = self.cfg.speed_limits[self.speed_stage]
        magnitude = torch.empty(len(env_ids), device=self.device).uniform_(.1, vmax)
        direction = torch.where(torch.rand(len(env_ids), device=self.device) < .5, -1., 1.)
        target = torch.where(kind == 0, magnitude * direction, previous)
        target = torch.where(kind == 1, 0., target)
        target = torch.where(kind == 2, -previous, target)
        self.tracking_velocity[env_ids] = target
        self.tracking_transition[env_ids] = kind
        low, high = self.cfg.tracking_interval_range
        self.tracking_interval_steps[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            low, high).div(self._env.step_dt).round().long()
        self.segment_started[env_ids] = target.abs() >= .8 * vmax
        self.segment_stage[env_ids] = self.speed_stage

    def _sample_yaw_speed_target(self, env_ids):
        # 新配方四类等概率：0停、1直行、2原地转、3弯行。
        count = len(env_ids)
        kind = torch.randint(0, 4, (count,), device=self.device)
        vmax = self.cfg.speed_limits[self.speed_stage]
        magnitude = torch.empty(count, device=self.device).uniform_(.1, vmax)
        direction = torch.where(torch.rand(count, device=self.device) < .5, -1., 1.)
        target = torch.where((kind == 1) | (kind == 3), magnitude * direction, 0.)
        yaw_magnitude = torch.empty(count, device=self.device).uniform_(.1, self.cfg.yaw_rate_limit)
        yaw_direction = torch.where(torch.rand(count, device=self.device) < .5, -1., 1.)
        self.tracking_yaw_rate[env_ids] = torch.where(
            (kind == 2) | (kind == 3), yaw_magnitude * yaw_direction, 0.)
        self.tracking_velocity[env_ids] = target
        self.tracking_transition[env_ids] = kind
        low, high = self.cfg.tracking_interval_range
        self.tracking_interval_steps[env_ids] = torch.empty(count, device=self.device).uniform_(
            low, high).div(self._env.step_dt).round().long()
        self.segment_started[env_ids] = target.abs() >= .8 * vmax
        self.segment_stage[env_ids] = self.speed_stage

    def _advance_speed_command(self, active):
        # super()._update_command会重建命令，斜率必须从上一拍真实下发值开始。
        previous = self._speed_previous_command
        lost = ~active & self.segment_started
        self.segment_failed |= lost
        self._finish_speed_segments(lost.nonzero(as_tuple=False).flatten())
        self.tracking_velocity[~active] = 0.
        self.tracking_interval_steps[~active] = 0
        measured = active & self.segment_started
        error = (self._env.scene["robot"].data.root_link_lin_vel_b[:, 0] - previous).abs()
        # 正常跟踪段：限斜率完成后等待0.5秒，再测满0.5秒，和验收口径一致。
        settled = measured & ((previous-self.tracking_velocity).abs() < 1e-5)
        self._speed_settle_steps = torch.where(settled, self._speed_settle_steps + 1, 0)
        measure = settled & (self._speed_settle_steps > round(.5 / self._env.step_dt))
        self.segment_steps += measure.long()
        self.segment_error += torch.where(measure, error, 0.)
        self.tracking_interval_steps[active] -= 1
        due = (active & (self.tracking_interval_steps <= 0)).nonzero(as_tuple=False).flatten()
        self._finish_speed_segments(due)
        self._sample_speed_target(due)
        self._speed_settle_steps[due] = 0
        target = self.tracking_velocity
        if self.cfg.position_hold:
            # 采样和晋级仍只看运动角色，停车纠偏只扩大奖励/实际命令门控。
            ready = self.has_stood & ~self.recover & ~self.fallen & (self.tracking_ready_steps >= round(.5 / self._env.step_dt))
            parking = self._advance_position_command(ready)
            target = torch.where(parking, self.position_velocity, target)
            active = active | parking
            self.tracking_active.copy_(active)
        delta = self.tracking_acceleration * self._env.step_dt
        next_command = previous + (target-previous).clamp(-delta, delta)
        self._command[:, 0] = torch.where(active, next_command, 0.)

    def position_config(self):
        if not self.cfg.yaw_control:
            raise ValueError('Position hold requires yaw control')
        self.yaw_config()
        config = {name: getattr(self.cfg, name) for name in ('position_gain', 'position_velocity_limit',
            'position_tolerance', 'position_lock_speed', 'position_lock_duration')}
        if any(not math.isfinite(value) or value <= 0. for name, value in config.items() if name != 'position_tolerance'):
            raise ValueError('Position hold gain, limit, lock speed and duration must be finite and positive')
        if not math.isfinite(config['position_tolerance']) or config['position_tolerance'] < 0.:
            raise ValueError('Position tolerance must be finite and nonnegative')
        return config

    def _clear_position(self, env_ids):
        self.position_anchor[env_ids] = 0.
        self.position_locked[env_ids] = False
        self.position_ready_steps[env_ids] = 0
        self.position_error_forward[env_ids] = self.position_error_lateral[env_ids] = 0.
        self.position_velocity[env_ids] = 0.

    def _position_measurement(self):
        data = self._env.scene['robot'].data
        return data.root_link_pos_w[:, :2], data.root_link_lin_vel_b[:, 0]

    def _advance_position_command(self, ready):
        xy, velocity = self._position_measurement()
        allowed = ready & (self.tracking_velocity == 0.)
        self._clear_position((~allowed).nonzero(as_tuple=False).flatten())
        waiting = allowed & ~self.position_locked
        low = waiting & (velocity.abs() < self.cfg.position_lock_speed)
        self.position_ready_steps[waiting] = torch.where(low[waiting], self.position_ready_steps[waiting] + 1, 0)
        lock = low & (self.position_ready_steps >= max(1, math.ceil(self.cfg.position_lock_duration / self._env.step_dt)))
        # 先等滑行结束，再把这一拍的位置作为停车点；锁定后不追随当前位置。
        self.position_anchor[lock] = xy[lock]
        self.position_locked[lock] = True
        yaw = self._env.scene['robot'].data.heading_w
        offset = xy - self.position_anchor
        forward = offset[:, 0] * yaw.cos() + offset[:, 1] * yaw.sin()
        lateral = -offset[:, 0] * yaw.sin() + offset[:, 1] * yaw.cos()
        self.position_error_forward.copy_(torch.where(self.position_locked, forward, 0.))
        self.position_error_lateral.copy_(torch.where(self.position_locked, lateral, 0.))
        correction = (-self.cfg.position_gain * forward).clamp(-self.cfg.position_velocity_limit, self.cfg.position_velocity_limit)
        self.position_velocity.copy_(torch.where(self.position_locked & (forward.abs() > self.cfg.position_tolerance), correction, 0.))
        return self.position_locked

    def _advance_yaw_command(self, active):
        yaw = self._env.scene["robot"].data.heading_w
        self.tracking_yaw_rate[~active | ~self.tracking_episode] = 0.
        turning = active & (self.tracking_yaw_rate != 0.)
        hold = active & ~turning
        # 松开转向的当拍锁当前朝向；持续保持时不追随漂移更新参考。
        relock = turning | (hold & ~self.heading_hold)
        self.heading_target[relock] = yaw[relock]
        self.heading_hold.copy_(hold)
        correction = (self.cfg.heading_gain * wrap_to_pi(self.heading_target-yaw)).clamp(
            -self.cfg.yaw_rate_limit, self.cfg.yaw_rate_limit)
        target = torch.where(turning, self.tracking_yaw_rate, correction).clamp(
            -self.cfg.yaw_rate_limit, self.cfg.yaw_rate_limit)
        delta = self.cfg.yaw_acceleration * self._env.step_dt
        next_command = self._yaw_previous_command + (target-self._yaw_previous_command).clamp(-delta, delta)
        # 失稳清零优先于正常转向斜率，恢复后由heading_hold=False重新锁定。
        self._command[:, 1] = torch.where(active, next_command, 0.)

    def advance(self):
        MultimodeCommand.advance(self)
        # This task has no airborne assist; keep its inherited privileged flag at zero.
        self.airborne.zero_()

    @property
    def recover(self):
        # Observation mode is independent of Isaac reward/success/termination memory.
        return ~self.observation_recovered


def wheel_load(env):
    # Use upward support, not force magnitude; tangential contact cannot hold the robot up.
    forces = -env.scene["wheel_contact"].data.force[:, :, 2]
    gravity = math.sqrt(sum(float(value)**2 for value in env.sim.mj_model.opt.gravity))
    weight = env.sim.model.body_mass.sum(-1) * gravity
    return forces, weight


def normal_wheel_support(env):
    forces, weight = wheel_load(env)
    minimum = (weight * NORMAL_EACH_WHEEL_WEIGHT_FRACTION).clamp_min(NORMAL_EACH_WHEEL_MIN_N)
    return ((forces >= minimum[:, None]).all(-1)
            & (forces.sum(-1) >= weight * NORMAL_TOTAL_WHEEL_WEIGHT_FRACTION))


def failure(env):
    state = env.command_manager.get_term("standup")
    if state._last_failure_step == env.common_step_counter:
        return state._isaac_failure
    robot = env.scene["robot"]
    data = robot.data
    support = (-env.scene["wheel_contact"].data.force[:, :, 2] > 1.).any(-1)
    landed_now = ~state.has_landed & support
    state.has_landed |= support
    # Keep episode_length_buf untouched: mjlab's 0.2 s motor hold uses that clock.
    state.landed_steps = torch.where(landed_now, 0,
        state.landed_steps + state.has_landed.long())
    height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    gravity_z = data.projected_gravity_b[:, 2]
    gravity = data.projected_gravity_b
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
    observation_down = ((roll.abs() > math.radians(40))
                        | (pitch.abs() > math.radians(40)) | (height <= .20)
                        | ~normal_wheel_support(env))
    state.observation_stable_steps = torch.where(observation_down, 0,
        state.observation_stable_steps + 1)
    state.observation_recovered = (~observation_down & (state.observation_recovered
        | (state.observation_stable_steps >= round(.5 / env.step_dt))))
    force = base_contact_force(env)
    # Success now measures an upright, wheel-supported stance, not a 0.28 m get-up.
    stable = (state.has_landed & (height >= state.success_height)
              & (gravity_z <= -math.cos(math.radians(15))) & (force <= .1)
              & normal_wheel_support(env))
    # Count whole policy ticks so float accumulation cannot delay the 50th tick.
    state.standing_time = torch.where(stable,
        (torch.round(state.standing_time / env.step_dt) + 1) * env.step_dt, 0.)
    state.has_stood |= state.standing_time >= .5
    state.stable_time.copy_(state.standing_time)
    airborne = state.has_landed & ~support
    state.wheels_airborne_steps = torch.where(airborne, state.wheels_airborne_steps + 1, 0)
    fallen = state.has_landed & state.has_stood & ((force > 10.) | (gravity_z > -.1))
    state.isaac_fallen_steps = torch.where(fallen, state.isaac_fallen_steps + 1, 0)
    state.knockdown.copy_(fallen & ~state.fallen)
    state.fallen.copy_(fallen)
    state.height_above_ground.copy_(height)
    state.numerical_failure.copy_(~torch.isfinite(env.sim.data.qpos).all(-1)
                                 | ~torch.isfinite(env.sim.data.qvel).all(-1))
    # Integer counts avoid float accumulation missing the source's 0.2/1.0 s boundaries.
    # Before the first successful stand, leaving the ground/inverting is allowed.
    state._isaac_failure = (state.has_landed & state.has_stood & ((gravity_z > 0.)
        | (state.wheels_airborne_steps >= .2 / env.step_dt)
        | (state.isaac_fallen_steps > 1. / env.step_dt))) | state.numerical_failure
    state.completed |= state._isaac_failure | timeout(env)
    state.metrics["isaac_recovered"].copy_(state.has_stood)
    state._last_failure_step = env.common_step_counter
    return state._isaac_failure


def timeout(env):
    # A never-landed or unsuccessful get-up episode must also finish and be counted.
    return env.episode_length_buf >= env.max_episode_length
