"""Isaac 25-channel layout plus seven modes; physical motor coordinates stay mjlab."""

import torch

from ..chuanliantui.multimode_observations import MultimodeObservationState
from ..chuanliantui.v14_observations import V14ObservationCfg

OBSERVATION_CONTRACT = "stand-v2-isaac32-history160-critic89-physical-v3"
ACTOR_DIM, CRITIC_DIM, HISTORY_LENGTH = 32, 89, 5


class StandV2ObservationState(MultimodeObservationState):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        # Isaac has no wheel-position inputs: keep both mjlab placeholders noiseless.
        self.noise_bounds.copy_(torch.tensor(
            [.2]*3 + [.05]*3 + [.02]*4 + [0.]*2 + [1.5]*6, device=env.device))
        self.noise.uniform_(-1., 1.).mul_(self.noise_bounds)

    def actor(self):
        sensors = self.delay.read()
        if self.cfg.noise:
            sensors = sensors + self.noise
        # Isaac ordering/scales, but q/dq are the four physical leg motors, not virtual knees.
        command = self.command.command * torch.tensor((2., .25, 5.), device=self.env.device)
        velocity = sensors[:, 12:18][:, (0, 1, 4, 2, 3, 5)]
        return torch.cat((sensors[:, :3] * .25, sensors[:, 3:6], command,
            sensors[:, 6:10], velocity * .05, self.action.raw_action, self.mode()), -1).clamp(-100., 100.)

    def critic(self):
        state = self.command
        # Preserve privileged physical fields and expose the exact reward/terminal memory.
        phase = torch.stack((state.has_landed.float(), state.has_stood.float(),
            (state.standing_time / .5).clamp(0., 1.),
            (state.wheels_airborne_steps * self.env.step_dt / .2).clamp(0., 1.),
            (state.isaac_fallen_steps * self.env.step_dt).clamp(0., 1.),
            (state.observation_stable_steps * self.env.step_dt / .5).clamp(0., 1.),
            (self.env.episode_length_buf / self.env.max_episode_length).clamp(0., 1.)), -1)
        return torch.cat((super().critic(), phase), -1)


def get_state(env, cfg=None):
    if not hasattr(env, "_multimode_observations"):
        env._multimode_observations = StandV2ObservationState(env, cfg or V14ObservationCfg())
    return env._multimode_observations


def actor_observation(env, cfg=None):
    return get_state(env, cfg).actor()


def critic_observation(env, cfg=None):
    return get_state(env, cfg).critic()


def velocity_target(env):
    # Training supervision only. The actor never reads this group or critic observations.
    return env.scene["robot"].data.root_com_lin_vel_b * 2.  # Isaac velocity supervision scale.


def reset_observations(env, env_ids, cfg=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    get_state(env, cfg).reset(env_ids)


def sample_observations(env):
    state = get_state(env)
    state.sample_substep()
    return state.delay.lags.float().mean(-1) * env.physics_dt * 1000.
