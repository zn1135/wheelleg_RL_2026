"""Terrain-aware Huananhu command modes on the existing physical robot.

Mode state advances only from the command manager's control-step update. Sensor
and observation reads do not mutate it. The source V14 mode channels are kept
in order, while previously unused stair/slope/recover channels have real local
semantics documented in docs/ai/huananhu_multimode.md.
"""

from dataclasses import dataclass

import torch

from . import huananhu_mdp
from .huananhu_mdp import HuananhuCommand, HuananhuCommandCfg
from .huananhu_poses import sample_reset_state


TERRAIN_NAMES = ("flat", "slope", "step", "stair")
MODE_NAMES = ("normal", "stair", "slope", "recover", "jump")
WHEEL_RADIUS = .06


@dataclass(kw_only=True)
class MultimodeCommandCfg(HuananhuCommandCfg):
    def build(self, env):
        return MultimodeCommand(self, env)


def mode_one_hot(*, jump, stair, recover, slope):
    """Mutually exclusive modes, priority jump > stair > recover > slope."""
    stair = stair & ~jump
    recover = recover & ~jump & ~stair
    slope = slope & ~jump & ~stair & ~recover
    normal = ~(jump | stair | recover | slope)
    return torch.stack((normal, stair, slope, recover, jump), dim=-1).float()


class MultimodeCommand(HuananhuCommand):
    """Four terrain families and independent airborne, jump and step assists."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, device = self.num_envs, self.device
        self.terrain_kind = torch.zeros(n, device=device, dtype=torch.long)
        self.ground_height = torch.zeros(n, device=device)
        self.height_above_ground = torch.zeros(n, device=device)
        self.airborne_time = torch.zeros(n, device=device)
        self.airborne = torch.zeros(n, device=device, dtype=torch.bool)
        self.step_hold = torch.zeros(n, device=device)
        self.obstacle_seen = torch.zeros(n, device=device, dtype=torch.bool)
        self.stair_time = torch.zeros(n, device=device)
        self.stair_active = torch.zeros(n, device=device, dtype=torch.bool)
        self.jump_phase = torch.zeros(n, device=device, dtype=torch.long)
        self.jump_time = torch.zeros(n, device=device)
        self.jump_target = torch.zeros(n, device=device)
        self.jump_peak = torch.zeros(n, device=device)
        self.jump_cooldown = torch.zeros(n, device=device)
        self.recovered = torch.zeros(n, device=device, dtype=torch.bool)
        self.trigger_counts = {name: torch.zeros(n, device=device) for name in
                               ("airborne", "jump", "step", "stair")}
        for name in self.trigger_counts:
            self.metrics[f"mode_{name}"] = self.trigger_counts[name]
        self._last_mode_step = -1

    @property
    def policy_active(self):
        return ~self.ground_start | (self._env.episode_length_buf * self._env.step_dt >= .2)

    @property
    def recover(self):
        return self.ground_start & ~self.recovered

    @property
    def step_active(self):
        return self.step_hold > 0

    def mode(self):
        one_hot = mode_one_hot(jump=self.jump_phase > 0,
                               stair=self.stair_active | self.step_active,
                               recover=self.recover,
                               slope=self.terrain_kind == 1)
        jump = self.jump_phase > 0
        return torch.cat((one_hot,
                          torch.where(jump, self.jump_target, 0.).unsqueeze(-1) * 5.,
                          torch.where(jump, self.jump_time, 0.).unsqueeze(-1)), dim=-1)

    def extra_flags(self):
        return torch.stack((self.ground_start, self.policy_active,
                            self.airborne, self.step_active), dim=-1).float()

    def reset(self, env_ids):
        result = super().reset(env_ids)
        for name in ("airborne_time", "step_hold", "stair_time", "jump_time", "jump_target",
                     "jump_peak", "jump_cooldown"):
            getattr(self, name)[env_ids] = 0
        for name in ("airborne", "stair_active", "recovered", "obstacle_seen"):
            getattr(self, name)[env_ids] = False
        self.jump_phase[env_ids] = 0
        terrain = self._env.scene.terrain
        self.terrain_kind[env_ids] = terrain.terrain_types[env_ids]
        self.ground_height[env_ids] = self._env.scene.env_origins[env_ids, 2]
        root_z_adr = self._env.scene["robot"].indexing.free_joint_q_adr[2]
        self.height_above_ground[env_ids] = (
            self._env.sim.data.qpos[env_ids, root_z_adr] - self.ground_height[env_ids])
        return result

    def _update_command(self, env_ids):
        super()._update_command(env_ids)
        if not hasattr(self, "jump_phase") or env_ids is not None:
            return
        step = self._env.common_step_counter
        if step == self._last_mode_step:
            return
        self._last_mode_step = step
        self.advance()

    def advance(self):
        env = self._env
        dt = env.step_dt
        robot = env.scene["robot"]
        scan = env.scene["terrain_scan"].data
        # Seven yaw-aligned vertical rays at x=-.6,-.4,-.2,0,.2,.4,.6.
        valid = scan.distances >= 0
        hit_z = scan.hit_pos_w[:, :, 2]
        center = torch.where(valid[:, 3], hit_z[:, 3], env.scene.env_origins[:, 2])
        ahead = torch.where(valid[:, 5], hit_z[:, 5], center)
        self.ground_height.copy_(center)
        self.height_above_ground.copy_(robot.data.root_link_pos_w[:, 2] - center)
        wheel = env.scene["wheel_height"].data.heights
        clear = torch.isfinite(wheel).all(-1) & (wheel > WHEEL_RADIUS + .02).all(-1)
        wheel_force = env.scene["wheel_contact"].data.force_history.norm(dim=-1).amax(dim=-1)
        no_contact = (wheel_force < 5.).all(-1)
        self.airborne_time = torch.where(clear & no_contact,
                                         self.airborne_time + dt, 0.)
        entered_air = ~self.airborne & (self.airborne_time >= .02)
        self.airborne = self.airborne_time >= .02
        self.trigger_counts["airborne"] += entered_air.float()
        self.recovered |= self.stable_time >= .5

        rise = ahead - center
        can_step = (rise >= .035) & (rise <= .09) & (self.terrain_kind >= 2)
        # Trigger on a new scan hit, not every time the hold timer expires
        # while the same obstacle remains in front of the wheels.
        first_step = can_step & ~self.obstacle_seen & (self.jump_phase == 0)
        self.obstacle_seen.copy_(can_step)
        self.step_hold = (self.step_hold - dt).clamp_min(0.)
        self.step_hold = torch.where(first_step, 1.0, self.step_hold)
        self.trigger_counts["step"] += first_step.float()
        stair_seen = first_step & (self.terrain_kind == 3)
        self.stair_active |= stair_seen
        self.stair_time = torch.where(self.stair_active, self.stair_time + dt, 0.)
        stair_done = self.stair_time >= 2.
        self.stair_active &= ~stair_done
        self.trigger_counts["stair"] += stair_seen.float()
        height_cmd = torch.where(self.step_active | self.stair_active,
                                 (self.sampled[:, 2] + .04).clamp(max=.34),
                                 self._command[:, 2])
        self._command[:, 2] = height_cmd

        self.jump_cooldown = (self.jump_cooldown - dt).clamp_min(0.)
        episode_t = env.episode_length_buf * dt
        # A short, explicit takeoff request on a subset of stable flat episodes.
        can_jump = ((self.terrain_kind == 0) & (self.jump_phase == 0)
                    & (self.jump_cooldown <= 0) & (episode_t >= 2.)
                    & (self.stable_time >= .5) & ~self.ground_start)
        request = can_jump & (torch.rand(self.num_envs, device=self.device) < dt * .08)
        self.jump_phase = torch.where(request, 1, self.jump_phase)
        self.jump_target = torch.where(request,
                                       torch.empty_like(self.jump_target).uniform_(.40, .46),
                                       self.jump_target)
        self.jump_peak = torch.maximum(self.jump_peak, self.height_above_ground)
        self.jump_time = torch.where(self.jump_phase > 0, self.jump_time + dt, 0.)
        self.trigger_counts["jump"] += request.float()
        to_tuck = (self.jump_phase == 1) & ((self.jump_time >= .18) | self.airborne)
        self.jump_phase = torch.where(to_tuck, 2, self.jump_phase)
        done = (self.jump_phase == 2) & ((~self.airborne & (self.jump_time > .25))
                                          | (self.jump_time >= 1.2))
        self.jump_phase = torch.where(done, 0, self.jump_phase)
        self.jump_cooldown = torch.where(done, 3., self.jump_cooldown)
        self.jump_target = torch.where(done, 0., self.jump_target)
        self.jump_peak = torch.where(done, 0., self.jump_peak)


def reset_pose(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state = env.command_manager.get_term("standup")
    ground = torch.rand(len(env_ids), device=env.device) < .5
    state.ground_start[env_ids] = ground
    # Existing closed-chain pose bank is retained; the generated terrain origin
    # is applied by sample_reset_state. No joints are reset to zero.
    sample_reset_state(env, env_ids, ground)
    if hasattr(env, "_huananhu_rewards"):
        env._huananhu_rewards.step = -1


def failure(env):
    failed = huananhu_mdp.failure(env)
    robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
    origin_xy = env.scene.env_origins[:, :2]
    out_of_patch = (robot_xy - origin_xy).abs().amax(dim=-1) > 2.5
    return failed | out_of_patch
