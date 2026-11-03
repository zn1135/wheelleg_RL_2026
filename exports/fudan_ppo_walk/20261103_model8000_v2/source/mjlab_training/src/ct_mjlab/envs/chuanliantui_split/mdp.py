"""Separate role reset/termination, sharing the established physical command loops."""

from dataclasses import dataclass
import math
import torch
from ..chuanliantui_stand_v2 import mdp as stand
from ..chuanliantui_stand import mdp as resets
from .switch import physical_conditions


WALK_INITIAL_HEIGHT = .30


def reset_walk_pose(env, env_ids):
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    state = env.command_manager.get_term('standup')
    state.ground_clearance_lift[ids] = 0.
    # 腿部闭链姿态与机身高度一同改变，不能只把旧站姿悬空抬高。
    resets._reset_near_standing_pose(env, ids, pose=resets.standing_pose(WALK_INITIAL_HEIGHT))
    state.settle_required[ids] = False


def walk_speed_config(cfg):
    return {name: getattr(cfg, name) for name in ('tracking_acceleration_range',
        'tracking_interval_range', 'forward_speed_min', 'forward_speed_limits',
        'backward_speed_min', 'backward_speed_limits', 'speed_error_threshold',
        'speed_promotion_segments', 'speed_promotion_fraction')}


def validate_walk_speed_state(speed, cfg):
    if not isinstance(speed, dict) or set(speed) != {'config', 'forward', 'backward'}:
        raise ValueError('Walking requires separate forward/backward curriculum state')
    if speed['config'] != walk_speed_config(cfg):
        raise ValueError('Walking direction speed configuration differs from checkpoint')
    for direction in ('forward', 'backward'):
        state = speed[direction]
        if not isinstance(state, dict) or set(state) != {'stage', 'segments', 'successes', 'last_success_fraction'}:
            raise ValueError('Invalid walking direction counters')
        stage, segments, successes = (state[key] for key in ('stage', 'segments', 'successes'))
        fraction = state['last_success_fraction']
        if (any(type(v) is not int for v in (stage, segments, successes))
                or not 0 <= stage < len(getattr(cfg, direction+'_speed_limits'))
                or not 0 <= successes <= segments < cfg.speed_promotion_segments
                or not isinstance(fraction, (int, float)) or not math.isfinite(fraction)
                or not 0. <= fraction <= 1.):
            raise ValueError('Invalid walking direction stage/counters/fraction')


@dataclass(kw_only=True)
class SplitCommandCfg(stand.StandV2CommandCfg):
    role: str = 'getup'
    pre_unlock_target_height: float = .30
    post_unlock_target_height: float = .22
    pre_unlock_success_height: float = .28
    height_range: tuple[float, float] | None = None
    forward_speed_min: float = .30
    forward_speed_limits: tuple[float, ...] = (.40, .60, .80, 1., 1.50)
    backward_speed_min: float = 1.
    backward_speed_limits: tuple[float, ...] = (1., 1.50)
    moving_episode_probability: float = .5

    def build(self, env):
        if self.role not in ('getup', 'walk'):
            raise ValueError(f'Unknown split policy role: {self.role}')
        if self.role == 'walk' and (self.height_range is None or len(self.height_range) != 2
                or not all(math.isfinite(v) for v in self.height_range)
                or not .22 <= self.height_range[0] <= self.height_range[1] <= .30):
            raise ValueError('Walking height range must stay within [0.22, 0.30] m')
        if self.role == 'getup' and self.height_range is not None:
            raise ValueError('Get-up uses its height curriculum, not a random height range')
        if self.role == 'walk':
            for direction in ('forward', 'backward'):
                minimum = getattr(self, direction+'_speed_min')
                limits = getattr(self, direction+'_speed_limits')
                if (not math.isfinite(minimum) or minimum <= 0. or not limits
                        or any(not math.isfinite(v) or v < minimum for v in limits)
                        or any(a >= b for a, b in zip(limits, limits[1:]))):
                    raise ValueError('Walking direction speed minimum/tables must be positive and increasing')
        if not 0. <= self.moving_episode_probability <= 1.:
            raise ValueError('Moving episode probability must be within [0, 1]')
        if self.tracking_tight_term:
            from ..chuanliantui_stand_v2.rewards import configure_tight_reward
            configure_tight_reward(env.cfg, self)
            if self.role == 'walk':
                from .rewards import reward
                env.cfg.rewards['tracking_lin_vel_tight'].func = reward
        return SplitCommand(self, env)


class SplitCommand(stand.StandV2Command):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 起立复用原0.30→0.22m课程；行走仍从各自随机目标高度开始。
        self.unlocked = cfg.role == 'walk'
        self.walk_down_ticks = torch.zeros_like(self.landed_steps)
        self._last_walk_failure_step = -1
        self.walk_ready = torch.zeros_like(self.has_landed)
        self._walk_failure = torch.zeros_like(self.has_landed)
        if cfg.role == 'walk':
            self.episode_height = torch.full_like(self.standing_time, .22)
            self.direction_curriculum = {direction: dict(stage=0, segments=0, successes=0,
                last_success_fraction=0.) for direction in ('forward', 'backward')}
            self.segment_direction = torch.full_like(self.segment_stage, -1)
            # walk不暴露旧的公共档位，防止下游误读始终为0的继承字段。
            for name in ('speed_stage', 'speed_segments', 'speed_successes', 'last_speed_success_fraction'):
                delattr(self, name)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self.cfg.role == 'walk':
            # 覆盖原50%静止回合比例；其余回合仍等概率停车/直行/原地转。
            moving = torch.rand(len(env_ids), device=self.device) < self.cfg.moving_episode_probability
            self.tracking_episode[env_ids] = moving
            self.zero_hold[env_ids] = ~moving
            # Draw once at reset; velocity-segment changes do not change height.
            self.episode_height[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.height_range)
            self._apply_height(env_ids)
            self.segment_direction[env_ids] = -1

    def _apply_height(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.sampled[:, 2] = self.episode_height
        self._command[ids, 2] = self.episode_height[ids]
        self.metrics['isaac_target_height'].copy_(self.episode_height)

    def _update_command(self, env_ids):
        if self.cfg.role != 'walk':
            return super()._update_command(env_ids)
        _update_walk_state(self._env)
        if self.cfg.speed_curriculum:
            self._speed_previous_command = self._command[:, 0].clone()
        if self.cfg.yaw_control:
            self._yaw_previous_command = self._command[:, 1].clone()
        # Update every environment immediately when the global curriculum unlocks.
        self.sampled[:, 2] = self.target_height
        stand.StandCommand._update_command(self, env_ids)
        if self.cfg.post_stand_tracking:
            # 使用本拍终止检查缓存的walk状态；短于100ms的掉载不打断50拍等待。
            ready = self.walk_ready
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

        self._apply_height(env_ids)
        # 终止/急停清掉外环参考；下一次就绪重新锁定，不沿用旧速度或航向。
        inactive = (~self.walk_ready).nonzero(as_tuple=False).flatten()
        self.heading_target[inactive] = 0.
        self._speed_previous_command[inactive] = 0.
        self._yaw_previous_command[inactive] = 0.

    def _advance_speed_command(self, active):
        if self.cfg.role != 'walk':
            return super()._advance_speed_command(active)
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
            ready = self.walk_ready & (self.tracking_ready_steps >= round(.5 / self._env.step_dt))
            parking = self._advance_position_command(ready)
            target = torch.where(parking, self.position_velocity, target)
            active = active | parking
            self.tracking_active.copy_(active)
        delta = self.tracking_acceleration * self._env.step_dt
        next_command = previous + (target-previous).clamp(-delta, delta)
        self._command[:, 0] = torch.where(active, next_command, 0.)

    def speed_config(self):
        if self.cfg.role != 'walk':
            return super().speed_config()
        return walk_speed_config(self.cfg)

    def checkpoint_state(self):
        if self.cfg.role != 'walk':
            return super().checkpoint_state()
        saved = stand.StandupCommand.checkpoint_state(self)
        if self.cfg.speed_curriculum:
            saved['speed_curriculum'] = dict(config=self.speed_config(),
                **{name: dict(value) for name, value in self.direction_curriculum.items()})
        if self.cfg.yaw_control:
            saved['yaw_control'] = self.yaw_config()
        if self.cfg.position_hold:
            saved['position_hold'] = self.position_config()
        return saved

    def load_checkpoint_state(self, saved):
        if self.cfg.role != 'walk':
            return super().load_checkpoint_state(saved)
        validate_walk_speed_state(saved.get('speed_curriculum'), self.cfg)
        if saved.get('yaw_control') != self.yaw_config():
            raise ValueError('Yaw control configuration differs from checkpoint')
        if saved.get('position_hold') != self.position_config():
            raise ValueError('Position hold configuration differs from checkpoint')
        stand.StandupCommand.load_checkpoint_state(self, saved)
        self._clear_position(torch.arange(self.num_envs, device=self.device))
        self.direction_curriculum = {name: dict(saved['speed_curriculum'][name])
                                     for name in ('forward', 'backward')}

    def _finish_speed_segments(self, env_ids):
        if self.cfg.role != 'walk':
            return super()._finish_speed_segments(env_ids)
        steps = self.segment_steps[env_ids]
        good = (~self.segment_failed[env_ids] & (steps >= round(.5 / self._env.step_dt))
                & (self.segment_error[env_ids] / steps.clamp_min(1) <= self.cfg.speed_error_threshold))
        if not self.speed_curriculum_frozen:
            for index, direction in enumerate(('forward', 'backward')):
                state = self.direction_curriculum[direction]
                # 旧档仍在途的段只属于旧档，不能记到刚晋级的新档；另一方向独立处理。
                eligible = (self.segment_started[env_ids] & (self.segment_direction[env_ids] == index)
                            & (self.segment_stage[env_ids] == state['stage']))
                state['segments'] += int(eligible.sum())
                state['successes'] += int((eligible & good).sum())
                if state['segments'] >= self.cfg.speed_promotion_segments:
                    state['last_success_fraction'] = state['successes'] / state['segments']
                    if state['last_success_fraction'] >= self.cfg.speed_promotion_fraction:
                        limits = getattr(self.cfg, direction+'_speed_limits')
                        state['stage'] = min(state['stage']+1, len(limits)-1)
                    state['segments'] = state['successes'] = 0
        for name in ('segment_steps', 'segment_error', 'segment_started', 'segment_failed'):
            getattr(self, name)[env_ids] = 0

    def _sample_yaw_speed_target(self, env_ids):
        if self.cfg.role != 'walk':
            return super()._sample_yaw_speed_target(env_ids)
        count = len(env_ids)
        # Only stop, straight travel and spin; no simultaneous motion request.
        kind = torch.randint(0, 3, (count,), device=self.device)
        # 0前进、1后退等概率；各自下限/档位只约束直行目标，不改变停车或斜率过程。
        backward = torch.rand(count, device=self.device) < .5
        stages = [self.direction_curriculum[name]['stage'] for name in ('forward', 'backward')]
        vmax = torch.where(backward, self.cfg.backward_speed_limits[stages[1]],
                           self.cfg.forward_speed_limits[stages[0]])
        minimum = torch.where(backward, self.cfg.backward_speed_min, self.cfg.forward_speed_min)
        magnitude = minimum + torch.rand(count, device=self.device) * (vmax-minimum)
        direction = torch.where(backward, -1., 1.)
        target = torch.where(kind == 1, magnitude * direction, 0.)
        yaw = torch.empty(count, device=self.device).uniform_(.1, self.cfg.yaw_rate_limit)
        yaw_direction = torch.where(torch.rand(count, device=self.device) < .5, -1., 1.)
        self.tracking_yaw_rate[env_ids] = torch.where(kind == 2, yaw * yaw_direction, 0.)
        self.tracking_velocity[env_ids] = target
        self.tracking_transition[env_ids] = kind
        self.tracking_interval_steps[env_ids] = torch.empty(count, device=self.device).uniform_(
            *self.cfg.tracking_interval_range).div(self._env.step_dt).round().long()
        self.segment_started[env_ids] = target.abs() >= .8 * vmax
        self.segment_stage[env_ids] = torch.where(backward, stages[1], stages[0])
        self.segment_direction[env_ids] = backward.long()

    def reset(self, env_ids):
        if self.cfg.role != 'walk':
            return super().reset(env_ids)
        if self.cfg.speed_curriculum:
            self.segment_failed[env_ids] |= ~self.walk_ready[env_ids]
            if hasattr(self, "_isaac_failure"):
                self.segment_failed[env_ids] |= self._isaac_failure[env_ids]
            self._finish_speed_segments(env_ids)
        curriculum = self.update_curriculum(env_ids)
        result = stand.StandCommand.reset(self, env_ids)
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
        self.walk_down_ticks[env_ids] = 0
        self.walk_ready[env_ids] = False
        self._walk_failure[env_ids] = False
        self.unlocked = True
        self._apply_height(env_ids)
        if hasattr(self._env, '_split_walk_rewards'):
            self._env._split_walk_rewards.reset(env_ids)
        return result



def failure(env):
    old = stand.failure(env)
    state = env.command_manager.get_term('standup')
    if state.cfg.role == 'getup':
        # 与Isaac一致：首次轮触地后，倒置或连续腾空0.2秒即失败，无需曾站稳。
        gravity_z = env.scene['robot'].data.projected_gravity_b[:, 2]
        failed = old | (state.has_landed & ((gravity_z > 0.)
            | (state.wheels_airborne_steps >= .2 / env.step_dt)))
        state._isaac_failure = failed
        state.completed |= failed
        return failed
    if state.cfg.role == 'walk':
        failed = _update_walk_state(env)
        state.completed |= failed
        return failed
    return old


def _update_walk_state(env):
    state = env.command_manager.get_term('standup')
    # 终止在forward/命令之前运行；同一拍只读这一份物理判定，不能重复累加。
    old = stand.failure(env)
    if state._last_walk_failure_step != env.common_step_counter:
        _, down, emergency, finite = physical_conditions(env)
        armed = env.episode_length_buf >= round(1. / env.step_dt)
        state.walk_down_ticks.copy_(torch.where(down & armed, state.walk_down_ticks + 1, 0))
        state._walk_failure = old | emergency | ~finite | (state.walk_down_ticks >= round(.1 / env.step_dt))
        state.walk_ready.copy_(armed & ~state._walk_failure)
        state._last_walk_failure_step = env.common_step_counter
    return state._walk_failure
