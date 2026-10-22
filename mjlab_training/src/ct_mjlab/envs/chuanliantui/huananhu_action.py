"""Huananhu V14 controls on the physical Chuanliantui motor/spring hardware."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

from .gas_spring import (
    SPRING_DAMPING,
    SPRING_FORCE_LIMIT,
    SPRING_FORCE_MAX_REFERENCE,
    SPRING_FORCE_MIN,
    SPRING_INITIAL_COMPRESSION,
    SPRING_REFERENCE_STROKE,
)
from .pd_action import ChuanliantuiPdAction, ChuanliantuiPdActionCfg
from .model import wheel_response_parameters


def wheel_response_input(command: torch.Tensor, velocity: torch.Tensor,
                         parameters: torch.Tensor) -> torch.Tensor:
    """Nominal Nm to actuator input; XML fixed gain supplies G afterward."""
    along = torch.clamp(torch.sign(command) * velocity, min=0.)
    fraction = torch.clamp(1. - along / parameters[:, 1], min=0.)
    return command * fraction.pow(parameters[:, 2])


@dataclass(kw_only=True)
class HuananhuPdActionCfg(ChuanliantuiPdActionCfg):
    """Unbounded policy actions with separate leg and wheel actuator delays.

    Random delay choices round upward to physical simulation ticks; a fixed
    measured action_delay_s replaces them and rounds to the nearest tick.
    Nominal motor gains and all effort limits retain this robot's hardware.
    """

    class_type: type = field(default_factory=lambda: HuananhuPdAction)
    wheel_stiffness: float = .1
    wheel_damping: float = .0
    motor_randomization: bool = True
    nominal_leg_offset_range: tuple[float, float] = (0., 0.)
    randomize_delays: bool = True
    action_delay_choices_s: tuple[float, ...] = (.005, .010)
    action_delay_s: float | None = None  # Fixed measured delay overrides episode randomization.
    motor_response_time_constant_s: float = 0.  # Unmeasured first-order response; zero bypasses it.
    ground_start_hold_s: float = .2
    parameter_reset_interval_steps: int = 720

    def build(self, env) -> HuananhuPdAction:
        return self.class_type(self, env)


class HuananhuPdAction(ChuanliantuiPdAction):
    cfg: HuananhuPdActionCfg

    def __init__(self, cfg: HuananhuPdActionCfg, env):
        # Reuse hardware indexing, then apply this task's own randomization.
        super().__init__(replace(cfg, motor_randomization=False,
                                 standup_pushes=False, standup_landing_gate=False), env)
        self.cfg = cfg
        low, high = cfg.nominal_leg_offset_range
        if low != 0. or high != 0.:
            # Sample once per environment/joint; PD and observations share this reference.
            legs = self._leg_action_ids
            self.nominal_offset[:, legs] = torch.empty_like(self.nominal_offset[:, legs]).uniform_(low, high)
            self._nominal_pos.add_(self.nominal_offset)
        tau = cfg.motor_response_time_constant_s
        if not math.isfinite(tau) or tau < 0.:
            raise ValueError("Motor response time constant must be finite and nonnegative")
        self._motor_response_alpha = -math.expm1(-env.physics_dt / tau) if tau else 1.
        self._motor_response_torque = torch.zeros_like(self._torques)
        self._wheel_response = torch.as_tensor(
            wheel_response_parameters(env.sim.mj_model),
            dtype=self._torques.dtype, device=self.device)
        self._wheel_action_delay_steps = torch.zeros_like(self.action_delay_steps)
        self._delay_choices = torch.tensor(
            ([int(math.floor(cfg.action_delay_s / env.physics_dt + .5))]
             if cfg.action_delay_s is not None else
             [math.ceil(delay / env.physics_dt) for delay in cfg.action_delay_choices_s]),
            dtype=torch.long, device=self.device)
        self._action_history = torch.zeros(
            int(self._delay_choices.max()) + 1, self.num_envs, self.action_dim,
            device=self.device)
        self.spring_damping = torch.full_like(self._spring_force_bias, SPRING_DAMPING)
        # Integer episode ticks avoid a float32 20 * .01 < .2 boundary error.
        self._ground_hold_steps = math.ceil(cfg.ground_start_hold_s / env.step_dt - 1e-9)
        self._last_parameter_reset = torch.full(
            (self.num_envs,), -cfg.parameter_reset_interval_steps,
            dtype=torch.long, device=self.device)

    @property
    def wheel_action_delay_steps(self) -> torch.Tensor:
        return self._wheel_action_delay_steps

    def _ground_hold(self) -> torch.Tensor:
        command = self._env.command_manager.get_term("standup")
        return command.ground_start & (self._env.episode_length_buf < self._ground_hold_steps)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._control_actions.copy_(actions)
        command = self._env.command_manager.get_term("standup")
        if hasattr(command, "policy_active"):
            command.policy_active.copy_(~self._ground_hold())

    def apply_actions(self) -> None:
        self._history_head = (self._history_head + 1) % len(self._action_history)
        self._action_history[self._history_head].copy_(self._control_actions)
        leg_delayed = self._action_history[
            (self._history_head - self.action_delay_steps) % len(self._action_history), self._env_ids]
        wheel_delayed = self._action_history[
            (self._history_head - self.wheel_action_delay_steps) % len(self._action_history), self._env_ids]
        q = self._env.sim.data.qpos[:, self._motor_q_adr]
        qd = self._env.sim.data.qvel[:, self._motor_v_adr]
        legs, wheels = self._leg_action_ids, self._wheel_action_ids
        reference = self._reference_pos[:, legs]
        target = torch.clamp(
            self._nominal_pos[:, legs] + self.cfg.leg_position_scale * leg_delayed[:, legs],
            reference - 3.14, reference + 3.14)
        self._torques[:, legs] = (
            self.cfg.leg_stiffness * self.kp_scale[:, legs] * (target - q[:, legs])
            - self.cfg.leg_damping * self.kd_scale[:, legs] * qd[:, legs])
        target_velocity = torch.clamp(
            self.cfg.wheel_velocity_scale * wheel_delayed[:, wheels],
            -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity)
        self._torques[:, wheels] = self.kd_scale[:, wheels] * (
            self.cfg.wheel_stiffness * (target_velocity - qd[:, wheels])
            - self.cfg.wheel_damping * qd[:, wheels])
        self._torques.mul_(self.motor_strength)
        self._torques.copy_(torch.clamp(self._torques, -self._effort_limits, self._effort_limits))
        hold = self._ground_hold()
        self._torques[hold] = 0.
        # Candidate drive response after PD/limits, separate from the action FIFO.
        if self.cfg.motor_response_time_constant_s:
            self._motor_response_torque.add_(
                self._motor_response_alpha * (self._torques - self._motor_response_torque))
        else:
            self._motor_response_torque.copy_(self._torques)
        self._motor_response_torque[hold] = 0.
        # Preserve nominal PD/limits and diagnostics; only the simulated motor
        # response changes. XML gain G is not multiplied here a second time.
        inputs = self._motor_response_torque.clone()
        inputs[:, wheels] = wheel_response_input(
            self._motor_response_torque[:, wheels], qd[:, wheels], self._wheel_response)
        self._entity.set_joint_effort_target(inputs, joint_ids=self._physical_joint_ids)

        compression = (SPRING_INITIAL_COMPRESSION
                       - self._env.sim.data.qpos[:, self._spring_q_adr]).clamp(min=0.)
        stiffness = (SPRING_FORCE_MAX_REFERENCE - SPRING_FORCE_MIN) / SPRING_REFERENCE_STROKE
        elastic = (SPRING_FORCE_MIN + stiffness * compression).clamp(0., SPRING_FORCE_LIMIT)
        spring_force = (elastic + self._spring_force_bias
                        - self.spring_damping * self._env.sim.data.qvel[:, self._spring_v_adr])
        self._env.sim.data.ctrl[:, self._spring_ids] = spring_force.clamp(0., SPRING_FORCE_LIMIT)
        state = getattr(self._env, "_huananhu_randomization", None)
        if state is not None:
            state.apply_local_wrench()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = self._env_ids if env_ids is None else self._env_ids[env_ids]
        self._motor_response_torque[ids] = 0.
        if self.cfg.action_delay_s is not None:
            self.action_delay_steps[ids] = self._delay_choices[0]
            self.wheel_action_delay_steps[ids] = self._delay_choices[0]
        elif self.cfg.randomize_delays:
            choices = torch.randint(len(self._delay_choices), (len(ids), 2), device=self.device)
            self.action_delay_steps[ids] = self._delay_choices[choices[:, 0]]
            self.wheel_action_delay_steps[ids] = self._delay_choices[choices[:, 1]]
        if self.cfg.motor_randomization:
            step = self._env.common_step_counter
            eligible = ids[(step - self._last_parameter_reset[ids])
                           >= self.cfg.parameter_reset_interval_steps]
            count = len(eligible)
            if count:
                self.kp_scale[eligible] = torch.empty(count, 6, device=self.device).uniform_(.75, 1.25)
                self.kd_scale[eligible] = torch.empty(count, 6, device=self.device).uniform_(.75, 1.25)
                strength = torch.empty(count, 6, device=self.device).uniform_(.8, 1.1)
                strength[:, self._wheel_action_ids] = torch.empty(count, 2, device=self.device).uniform_(.9, 1.1)
                self.motor_strength[eligible] = strength
                self.spring_damping[eligible] = torch.empty(count, 2, device=self.device).uniform_(25., 75.)
                self._last_parameter_reset[eligible] = step
