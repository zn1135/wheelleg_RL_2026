"""Huananhu 39/82 fixed-scale observations with live control-mode signals."""

import torch

from .huananhu_observations import HuananhuObservationState
from .v14_observations import V14ObservationCfg


OBSERVATION_CONTRACT = "chuanliantui-huananhu-multimode-39-82-v1"


class MultimodeObservationState(HuananhuObservationState):
    @property
    def command(self):
        return self.env.command_manager.get_term("standup")

    def mode(self):
        return self.command.mode()

    def actor(self):
        return torch.cat((super().actor(), self.command.extra_flags()), dim=-1)

    def critic(self):
        old = super().critic().clone()
        # The existing privileged height slot is terrain-relative in this task.
        old[:, 31] = self.command.height_above_ground.clamp(-10., 10.) * 5.
        return torch.cat((old, self.command.extra_flags()), dim=-1)


def get_state(env, cfg: V14ObservationCfg | None = None) -> MultimodeObservationState:
    if not hasattr(env, "_multimode_observations"):
        env._multimode_observations = MultimodeObservationState(env, cfg or V14ObservationCfg())
    return env._multimode_observations


def actor_observation(env, cfg: V14ObservationCfg | None = None):
    return get_state(env, cfg).actor()


def critic_observation(env, cfg: V14ObservationCfg | None = None):
    return get_state(env, cfg).critic()


def reset_observations(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    get_state(env).reset(env_ids)


def sample_observations(env):
    state = get_state(env)
    state.sample_substep()
    return state.delay.lags.float().mean(dim=-1) * env.physics_dt * 1000.
