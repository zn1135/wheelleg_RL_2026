"""Standing-only action delay measured in complete policy steps."""

from dataclasses import dataclass, field, replace

import torch

from ..chuanliantui.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg


@dataclass(kw_only=True)
class StandPdActionCfg(HuananhuPdActionCfg):
    class_type: type = field(default_factory=lambda: StandPdAction)
    action_delay_policy_steps: tuple[int, ...] = (1, 2)
    randomization_scale: float = .25


class StandPdAction(HuananhuPdAction):
    cfg: StandPdActionCfg

    def __init__(self, cfg, env):
        # Reuse the unchanged PD, reset randomization and physical action FIFO.
        super().__init__(replace(cfg, action_delay_s=0.), env)
        self.cfg = cfg
        # Integer decimation counts complete action steps, without seconds/rounding.
        self._delay_choices = torch.tensor(cfg.action_delay_policy_steps,
            dtype=torch.long, device=self.device) * env.cfg.decimation
        self._action_history = torch.zeros(
            int(self._delay_choices.max()) + 1, self.num_envs, self.action_dim,
            device=self.device)

    def reset(self, env_ids=None):
        ids = self._env_ids if env_ids is None else self._env_ids[env_ids]
        previous = self._last_parameter_reset[ids].clone()
        super().reset(env_ids)
        changed = ids[self._last_parameter_reset[ids] != previous]
        scale = self.cfg.randomization_scale
        if scale == 1.:
            return
        for values in (self.kp_scale, self.kd_scale, self.motor_strength):
            values[changed] = 1 + scale * (values[changed] - 1)
        from ..chuanliantui.gas_spring import SPRING_DAMPING
        self.spring_damping[changed] = (SPRING_DAMPING
            + scale * (self.spring_damping[changed] - SPRING_DAMPING))
