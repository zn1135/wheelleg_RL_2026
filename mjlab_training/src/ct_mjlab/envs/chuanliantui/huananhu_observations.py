"""Flat Huananhu Actor 35 / Critic 78 observations for the MuJoCo robot.

The first 78 Critic channels match the existing V14 fixed-scale layout. The
standup-only six phase channels are omitted. MuJoCo exposes each wheel's native
sliding, torsional and rolling friction values; these six material channels
retain that native schema instead of pretending to be Isaac Lab materials.
"""

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .v14_observations import V14ObservationCfg, V14ObservationState


OBSERVATION_CONTRACT = "chuanliantui-huananhu-flat-35-78-v1"


class HuananhuObservationState(V14ObservationState):
    """Share the V14 sensor delay/noise and Actor layout, without standup phase."""

    def critic(self) -> torch.Tensor:
        data = self.robot.data
        kp = self.action.position_gains[:, self.action_order]
        kd = self.action.velocity_gains[:, self.action_order]

        wheel_vel = data.body_com_lin_vel_w[:, self.wheel_ids] - data.root_com_lin_vel_w[:, None, :]
        quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        wheel_vel = quat_apply_inverse(quat.reshape(-1, 4), wheel_vel.reshape(-1, 3)).reshape(-1, 2, 3)
        wheel_vel[:, :, 1] = 0.0

        force = self.env.scene["wheel_contact"].data.force_history
        assert force is not None
        contact = (force[:, self.contact_ids].norm(dim=-1).amax(dim=-1) > 5.0).float()
        base_mass = self.env.sim.model.body_mass[:, self.base_id:self.base_id + 1]
        mass_scale = base_mass / self.env.sim.get_default_field("body_mass")[self.base_id]
        material = self.env.sim.model.geom_friction[:, self.wheel_geom_ids].reshape(self.env.num_envs, 6)

        # Source Critic channels 61:63 hold the two action delay groups. The
        # physical action term exposes each delay separately in physics steps.
        leg_delay = self.action.action_delay_steps[:, None].float()
        wheel_delay = self.action.wheel_action_delay_steps[:, None].float()

        return torch.cat((
            self.basic(self.raw_sensors()),
            data.root_com_lin_vel_b.clamp(-100, 100),
            data.root_link_pos_w[:, 2:3].clamp(-10, 10) * 5,
            self.mode(),
            kp.clamp(-100, 100), kd.clamp(-100, 100),
            self.env.sim.data.qfrc_actuator[:, self._motor_v_adr].clamp(-100, 100) * .05,
            self.delay.lags.float(),
            leg_delay, wheel_delay,
            wheel_vel.reshape(-1, 6).clamp(-100, 100), contact,
            mass_scale.clamp(-10, 10), material.clamp(-100, 100),
        ), dim=-1)


def get_state(env, cfg: V14ObservationCfg | None = None) -> HuananhuObservationState:
    if not hasattr(env, "_huananhu_observations"):
        env._huananhu_observations = HuananhuObservationState(env, cfg or V14ObservationCfg())
    return env._huananhu_observations


def actor_observation(env, cfg: V14ObservationCfg | None = None) -> torch.Tensor:
    return get_state(env, cfg).actor()


def critic_observation(env, cfg: V14ObservationCfg | None = None) -> torch.Tensor:
    return get_state(env, cfg).critic()


def reset_observations(env, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    get_state(env).reset(env_ids)


def sample_observations(env) -> torch.Tensor:
    state = get_state(env)
    state.sample_substep()
    return state.delay.lags.float().mean(dim=-1) * env.physics_dt * 1000
