"""Policy-rate leg smoothing for getup; keep raw observations and wheel control."""

import torch

from ..chuanliantui_stand_v2.action import StandV2PdAction


LEG_ACTION_ALPHA = .8
ACTION_FILTER_CONTRACT = 'split-getup-leg-ema80-policy100hz-hold-reset-v2'


class GetupPdAction(StandV2PdAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered_leg_actions = torch.zeros(self.num_envs, 4, device=self.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        # 每个策略拍只混合一次；等待期清零，避免首次出力前已累积成全量动作。
        self._filtered_leg_actions.mul_(1.-LEG_ACTION_ALPHA).add_(
            actions[:, self._leg_action_ids], alpha=LEG_ACTION_ALPHA)
        self._filtered_leg_actions[self._ground_hold()] = 0.
        self._control_actions[:, self._leg_action_ids] = self._filtered_leg_actions

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._filtered_leg_actions[slice(None) if env_ids is None else env_ids] = 0.
