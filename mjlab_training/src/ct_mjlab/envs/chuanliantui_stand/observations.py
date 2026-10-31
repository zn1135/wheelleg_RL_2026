"""Standing-only 39/82 observations with the original wheel-angle placeholders."""
from dataclasses import dataclass

import torch

from ..chuanliantui.multimode_observations import MultimodeObservationState
from ..chuanliantui.v14_observations import V14ObservationCfg

OBSERVATION_CONTRACT = "chuanliantui-huananhu-multimode-39-82-v1"


@dataclass(frozen=True)
class StandObservationCfg(V14ObservationCfg):
    noise_scale: float = .4


class StandObservationState(MultimodeObservationState):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.noise_bounds.mul_(cfg.noise_scale)
        self.noise.mul_(cfg.noise_scale)


def get_state(env, cfg=None):
    if not hasattr(env, "_multimode_observations"):
        env._multimode_observations = StandObservationState(env, cfg or StandObservationCfg())
    return env._multimode_observations


def actor_observation(env, cfg=None):
    return get_state(env, cfg).actor()


def critic_observation(env, cfg=None):
    return get_state(env, cfg).critic()


def reset_observations(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    get_state(env).reset(env_ids)


def sample_observations(env):
    state = get_state(env)
    state.sample_substep()
    return state.delay.lags.float().mean(dim=-1) * env.physics_dt * 1000.
