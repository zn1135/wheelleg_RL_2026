"""Shared physical observations; task selection is never a network input."""

import torch
from ..chuanliantui.huananhu_observations import HuananhuObservationState
from ..chuanliantui_stand_v2.observations import StandV2ObservationState, velocity_target
from ..chuanliantui.v14_observations import V14ObservationCfg

ACTOR_DIM, CRITIC_DIM, HISTORY_LENGTH = 25, 71, 5
OBSERVATION_CONTRACT = 'chuanliantui-split-physical25-history125-critic71-v1'


class SplitObservationState(StandV2ObservationState):
    def mode(self):
        return torch.empty((self.env.num_envs, 0), device=self.env.device)

    def critic(self):
        # Only physical privileged fields: omit mode, extra flags and task phase flags.
        value = HuananhuObservationState.critic(self)
        value[:, 31] = self.command.height_above_ground.clamp(-10., 10.) * 5.
        return value


def get_state(env, cfg=None):
    if not hasattr(env, '_multimode_observations'):
        env._multimode_observations = SplitObservationState(env, cfg or V14ObservationCfg())
    return env._multimode_observations


def actor_observation(env, cfg=None):
    return get_state(env, cfg).actor()


def critic_observation(env, cfg=None):
    return get_state(env, cfg).critic()


def reset_observations(env, env_ids, cfg=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    get_state(env, cfg).reset(env_ids)


def sample_observations(env):
    value = get_state(env)
    value.sample_substep()
    return value.delay.lags.float().mean(-1) * env.physics_dt * 1000.
