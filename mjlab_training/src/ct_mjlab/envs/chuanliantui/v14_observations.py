"""V14 Flat-v0 observation scheme adapted to the MuJoCo standup task.

See docs/ai/standup_observations.md for the indexed schema and engine changes.
No external source repository is required at runtime.
"""

from dataclasses import dataclass
import math

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .standup_mdp import AIRBORNE_TIMEOUT_S, FALLEN_TIMEOUT_S, STABLE_DURATION_S


OBSERVATION_CONTRACT = "chuanliantui-standup-v14-mujoco-35-84-leftplusy-phase-rawaction-fixedscale-v5"
MOTOR_NAMES = ("lf0", "lf00", "rf0", "rf00", "lfwheel", "rfwheel")
ACTION_ORDER = (0, 1, 3, 4, 2, 5)
SENSOR_SLICES = (slice(0, 3), slice(3, 6), slice(6, 12), slice(12, 18))


@dataclass(frozen=True)
class V14ObservationCfg:
    noise: bool = True
    delay: bool = True
    delay_seconds: tuple[float, ...] = (.005, .010, .015)
    wheel_angle_displacement: bool = False
    wheel_angle_handover_reset: bool = False


class GroupedSensorDelay:
    """Four independent, episode-local delays in physics samples."""

    def __init__(self, initial: torch.Tensor, max_lag: int):
        self.history = initial.unsqueeze(0).repeat(max_lag + 1, 1, 1)
        self.lags = torch.zeros(initial.shape[0], 4, dtype=torch.long, device=initial.device)
        self.env_ids = torch.arange(initial.shape[0], device=initial.device)
        self.head = 0

    def push(self, sample: torch.Tensor) -> None:
        self.head = (self.head + 1) % self.history.shape[0]
        self.history[self.head].copy_(sample)

    def read(self) -> torch.Tensor:
        return torch.cat([
            self.history[(self.head - self.lags[:, group]) % self.history.shape[0], self.env_ids, columns]
            for group, columns in enumerate(SENSOR_SLICES)
        ], dim=-1)

    def reset(self, env_ids: torch.Tensor, sample: torch.Tensor) -> None:
        # Do not rewind a shared cursor or retain any previous-episode samples.
        self.history[:, env_ids] = sample.unsqueeze(0)


class V14ObservationState:
    def __init__(self, env, cfg: V14ObservationCfg):
        self.env, self.cfg = env, cfg
        self.robot = env.scene["robot"]
        self.action = env.action_manager.get_term("pd")
        self.joint_ids, names = self.robot.find_joints(MOTOR_NAMES, preserve_order=True)
        if tuple(names) != MOTOR_NAMES:
            raise ValueError(f"Unexpected observation motor names: {names}")
        self.joint_ids = torch.tensor(self.joint_ids, device=env.device)
        self.action_order = torch.tensor(ACTION_ORDER, device=env.device)
        self._motor_q_adr = self.robot.indexing.joint_q_adr[self.joint_ids]
        self._motor_v_adr = self.robot.indexing.joint_v_adr[self.joint_ids]
        self.wheel_ids, _ = self.robot.find_bodies(("lfwheel", "rfwheel"), preserve_order=True)
        self.wheel_ids = torch.tensor(self.wheel_ids, device=env.device)
        base_ids, _ = self.robot.find_bodies(("base_link",), preserve_order=True)
        self.base_id = int(self.robot.indexing.body_ids[base_ids[0]])
        wheel_body_ids = self.robot.indexing.body_ids[self.wheel_ids].tolist()
        model = env.sim.mj_model
        self.wheel_geom_ids = []
        for body_id in wheel_body_ids:
            ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id
                   and (model.geom_contype[i] or model.geom_conaffinity[i])]
            if len(ids) != 1:
                raise ValueError(f"Expected one wheel collision geom, got {ids}")
            self.wheel_geom_ids.extend(ids)
        self.wheel_geom_ids = torch.tensor(self.wheel_geom_ids, device=env.device)
        sensor = env.scene["wheel_contact"]
        self.contact_ids = torch.tensor(
            [sensor.primary_names.index(name) for name in ("lfwheel", "rfwheel")], device=env.device)
        self._mode = torch.zeros(env.num_envs, 7, device=env.device)
        self._mode[:, 0] = 1.0
        self.lag_choices = torch.tensor(
            [int(math.floor(seconds / env.physics_dt + .5)) for seconds in cfg.delay_seconds],
            dtype=torch.long, device=env.device,
        )
        self.delay = GroupedSensorDelay(self.raw_sensors(), int(self.lag_choices.max()))
        self.noise_bounds = torch.tensor(
            [.25]*3 + [.05]*3 + [.025]*6 + [.5]*4 + [1.]*2, device=env.device
        )
        self.noise = torch.zeros(env.num_envs, 18, device=env.device)
        self.reset(torch.arange(env.num_envs, device=env.device))

    def raw_sensors(self) -> torch.Tensor:
        # Derived xquat/cvel lag the integration mid-substep. Read qpos/qvel.
        indexing = self.robot.indexing
        quat = self.env.sim.data.qpos[:, indexing.free_joint_q_adr[3:7]]
        ang_vel = self.env.sim.data.qvel[:, indexing.free_joint_v_adr[3:6]]
        gravity = quat_apply_inverse(quat, self.robot.data.gravity_vec_w)
        q = self.env.sim.data.qpos[:, self._motor_q_adr] - self.action.nominal_positions[:, self.action_order]
        q[:, 4:] = 0.0
        qd = self.env.sim.data.qvel[:, self._motor_v_adr]
        return torch.cat((ang_vel, gravity, q, qd), dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.delay.reset(env_ids, self.raw_sensors()[env_ids])
        if self.cfg.delay:
            choices = torch.randint(len(self.lag_choices), (len(env_ids), 4), device=self.env.device)
            self.delay.lags[env_ids] = self.lag_choices[choices]
        else:
            self.delay.lags[env_ids] = 0
        self.noise[env_ids] = torch.empty(len(env_ids), 18, device=self.env.device).uniform_(-1, 1) * self.noise_bounds

    def sample_substep(self) -> None:
        self.delay.push(self.raw_sensors())
        # Cached so repeated observation queries never resample or advance time.
        if self.cfg.noise:
            self.noise.uniform_(-1, 1).mul_(self.noise_bounds)

    def basic(self, sensors: torch.Tensor) -> torch.Tensor:
        command = self.env.command_manager.get_term("standup").command
        velocity_command = torch.stack((command[:, 0], torch.zeros_like(command[:, 0]), command[:, 1]), dim=-1)
        return torch.cat((
            velocity_command.clamp(-100, 100),
            command[:, 2:3].clamp(0, 1) * 5,
            sensors[:, :3].clamp(-100, 100) * .5,
            sensors[:, 3:12].clamp(-100, 100),
            sensors[:, 12:18].clamp(-200, 200) * .1,
            self.action.raw_action[:, self.action_order].clamp(-100, 100),
        ), dim=-1)

    def mode(self) -> torch.Tensor:
        # V14's ordinary mode. Its recover flag is a placeholder even in source.
        return self._mode

    def actor(self) -> torch.Tensor:
        sensors = self.delay.read()
        if self.cfg.noise:
            sensors = sensors + self.noise
        return torch.cat((self.basic(sensors), self.mode()), dim=-1)

    def critic_phase(self) -> torch.Tensor:
        """Current task memory, without sensor lag, noise or state advancement.

        Read timers afresh: the command replaces their tensors on each step.
        Keep the one-step difference between landing and policy activation.
        """
        state = self.env.command_manager.get_term("standup")
        return torch.stack((
            state.has_landed.float(),
            state.policy_active.float(),
            state.has_stood.float(),
            (state.standing_time / STABLE_DURATION_S).clamp(0.0, 1.0),
            (state.airborne_time / AIRBORNE_TIMEOUT_S).clamp(0.0, 1.0),
            (state.fallen_time / FALLEN_TIMEOUT_S).clamp(0.0, 1.0),
        ), dim=-1)

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
        base_mass = self.env.sim.model.body_mass[:, self.base_id:self.base_id+1]
        mass_scale = base_mass / self.env.sim.get_default_field("body_mass")[self.base_id]
        # MuJoCo native material schema: sliding, torsional, rolling friction.
        material = self.env.sim.model.geom_friction[:, self.wheel_geom_ids].reshape(self.env.num_envs, 6)
        return torch.cat((
            self.basic(self.raw_sensors()),
            data.root_com_lin_vel_b.clamp(-100, 100),
            data.root_link_pos_w[:, 2:3].clamp(-10, 10) * 5,
            self.mode(),
            kp.clamp(-100, 100), kd.clamp(-100, 100),
            self.env.sim.data.qfrc_actuator[:, self._motor_v_adr].clamp(-100, 100) * .05,
            self.delay.lags.float(),
            self.action.action_delay_steps[:, None].expand(-1, 2).float(),
            wheel_vel.reshape(-1, 6).clamp(-100, 100), contact,
            mass_scale.clamp(-10, 10), material.clamp(-100, 100),
            self.critic_phase(),
        ), dim=-1)


def get_state(env, cfg: V14ObservationCfg | None = None) -> V14ObservationState:
    if not hasattr(env, "_standup_v14_observations"):
        env._standup_v14_observations = V14ObservationState(env, cfg or V14ObservationCfg())
    return env._standup_v14_observations


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
    return state.delay.lags.float().mean(dim=-1) * env.physics_dt * 1000
