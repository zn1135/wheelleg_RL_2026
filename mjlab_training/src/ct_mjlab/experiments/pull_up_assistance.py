"""Training-only world-up COM assistance, composed with existing disturbances."""
from dataclasses import dataclass, field
import math

import torch

from ..reference_training.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg


PARAMETERS = dict(max_weight_ratio=1., end_updates=3000,
                  promotion_max_weight_ratio=.05,
                  direction='world +z', application='base_link center of mass',
                  stages=('righting', 'standing'), playback_enabled=False,
                  respects_ground_motor_hold=True,
                  auxiliary_torque=False)


def pull_up_force(height, target, weight, updates, *, stage=0, enabled=True,
                  max_weight_ratio=1., end_updates=3000):
    """Source height deficit and linear time fade; max force scaled to this robot."""
    fade = max(0., min(1., 1. - updates / end_updates))
    deficit = ((target - height) / target.clamp_min(1e-6)).clamp(0., 1.)
    force = weight * max_weight_ratio * deficit * fade
    if not enabled or stage >= 2:
        force = torch.zeros_like(force)
    valid = (torch.isfinite(height) & torch.isfinite(target) & torch.isfinite(weight)
             & (target > 0.) & (weight > 0.))
    return torch.nan_to_num(force, nan=0., posinf=0., neginf=0.).masked_fill(~valid, 0.)


@dataclass(kw_only=True)
class AssistedPdActionCfg(HuananhuPdActionCfg):
    class_type: type = field(default_factory=lambda: AssistedPdAction)
    assistance_enabled: bool = True
    max_weight_ratio: float = PARAMETERS['max_weight_ratio']
    end_updates: int = PARAMETERS['end_updates']

    def build(self, env):
        if not math.isfinite(self.max_weight_ratio) or not 0. <= self.max_weight_ratio <= 1.:
            raise ValueError('Auxiliary force ratio must be between 0 and one robot weight')
        if self.end_updates < 1:
            raise ValueError('Auxiliary force end_updates must be positive')
        return super().build(env)


class AssistedPdAction(HuananhuPdAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene['robot']
        self._base_id = int(robot.indexing.body_ids[robot.body_names.index('base_link')])
        self._body_ids = robot.indexing.body_ids
        self._gravity = float(sum(float(v) ** 2 for v in env.sim.mj_model.opt.gravity) ** .5)
        self.assistance_force = torch.zeros(self.num_envs, device=self.device)

    def robot_weight(self):
        return self._env.sim.model.body_mass[:, self._body_ids].sum(-1) * self._gravity

    def apply_actions(self):
        # The parent writes motor/spring controls and freshly rotates the
        # persistent local wrench each physical tick. Add assistance afterward,
        # so it neither accumulates nor overwrites the existing wrench.
        super().apply_actions()
        state = self._env.command_manager.get_term('standup')
        if self.cfg.assistance_enabled and not hasattr(self._env, '_huananhu_randomization'):
            raise RuntimeError('Training assistance requires the configured disturbance composer')
        height = self._entity.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        updates = state.training_iteration + self._env.common_step_counter / 24
        self.assistance_force.copy_(pull_up_force(
            height, state.command[:, 2], self.robot_weight(), updates,
            stage=state.curriculum.stage, enabled=self.cfg.assistance_enabled,
            max_weight_ratio=self.cfg.max_weight_ratio, end_updates=self.cfg.end_updates))
        self.assistance_force.masked_fill_(self._ground_hold(), 0.)
        self._env.sim.data.xfrc_applied[:, self._base_id, 2] += self.assistance_force
        state.metrics['pull_up_force_n'].copy_(self.assistance_force)
        fade = max(0., min(1., 1. - updates / self.cfg.end_updates))
        state.metrics['pull_up_budget_ratio'].fill_(
            fade * self.cfg.max_weight_ratio if self.cfg.assistance_enabled and state.curriculum.stage < 2 else 0.)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = self._env_ids if env_ids is None else self._env_ids[env_ids]
        if hasattr(self, 'assistance_force'):
            self.assistance_force[ids] = 0.
            # Training's existing reset event clears the local wrench. Clear
            # base external forces here too, so forward() sees no stale aid.
            self._env.sim.data.xfrc_applied[ids, self._base_id] = 0.
