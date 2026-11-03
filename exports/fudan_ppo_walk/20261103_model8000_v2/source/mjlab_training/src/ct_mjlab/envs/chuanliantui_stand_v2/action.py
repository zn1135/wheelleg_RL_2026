"""Keep mjlab PD; sample Isaac motor parameters once per environment."""

from dataclasses import dataclass, field, replace

import torch

from ..chuanliantui.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg
from ..chuanliantui.pd_action import ChuanliantuiPdAction
from ..chuanliantui import standup_randomization


@dataclass(kw_only=True)
class StandV2PdActionCfg(HuananhuPdActionCfg):
    class_type: type = field(default_factory=lambda: StandV2PdAction)


class StandV2PdAction(HuananhuPdAction):
    def __init__(self, cfg, env):
        super().__init__(replace(cfg, motor_randomization=False, randomize_delays=False,
            action_delay_s=0., nominal_leg_offset_range=(0., 0.)), env)
        self.cfg = cfg
        if cfg.motor_randomization:
            self.kp_scale.uniform_(.95, 1.05)
            self.kd_scale.uniform_(.95, 1.05)
            self.motor_strength.uniform_(.8, 1.2)
            self.nominal_offset.uniform_(-.03, .03)
            self._nominal_pos.add_(self.nominal_offset)
        maximum_lag = round(.010 / env.physics_dt)
        self._action_history = torch.zeros(maximum_lag + 1, self.num_envs,
            self.action_dim, device=self.device)
        if cfg.randomize_delays:
            # Match Isaac's continuous uniform draw followed by nearest-tick rounding.
            self.action_delay_steps.copy_(torch.empty(self.num_envs, device=self.device)
                .uniform_(0., .010 / env.physics_dt).round().long())
        self.wheel_action_delay_steps.copy_(self.action_delay_steps)

    def process_actions(self, actions):
        # Latch before physics increments episode_length_buf (avoid a boundary leak).
        self._env.command_manager.get_term("standup").reward_active.copy_(~self._ground_hold())
        super().process_actions(actions)
        if self.cfg.standup_pushes:
            standup_randomization.get_state(self._env).update_push()

    def reset(self, env_ids=None):
        # Clear episode histories without resampling the environment's motor parameters.
        ChuanliantuiPdAction.reset(self, env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._motor_response_torque[ids] = 0.
