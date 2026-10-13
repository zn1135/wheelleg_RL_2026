"""Direct position PD on chuanliantui's physical closed-chain motor joints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .gas_spring import gas_spring_force
from .model import ACTUATED_JOINTS, POLICY_JOINTS, GAS_SPRING_JOINTS, gas_spring_actuator_ids

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


CONTROL_CONTRACT = "chuanliantui-direct-motor-pd-physical-spring-0to150n-preload10mm-xzero-reset-leftplusy-rawaction-sourcexml-spring90mm-legsfirst-v11"
_LEG_ACTION_IDS = (0, 1, 3, 4)
_WHEEL_ACTION_IDS = (2, 5)


@dataclass(kw_only=True)
class ChuanliantuiPdActionCfg(ActionTermCfg):
    """Policy order: (lf0, lf00, rf0, rf00, lfwheel, rfwheel).

    Leg commands are physical motor angles relative to a closed standing pose.
    Raw policy outputs are unbounded; only physical position/velocity targets
    and the resulting motor efforts are limited.
    """

    policy_joint_order: tuple[str, ...] = POLICY_JOINTS
    leg_position_scale: float = 0.5
    wheel_velocity_scale: float = 10.0
    max_wheel_velocity: float = 150.0
    leg_stiffness: float = 60.0
    leg_damping: float = 2.0
    wheel_stiffness: float = 0.12
    wheel_damping: float = 0.02
    standup_landing_gate: bool = False
    nominal_leg_positions: tuple[float, float, float, float] | None = None
    spring_force_bias_range: tuple[float, float] = (-50.0, 50.0)
    motor_randomization: bool = False
    standup_pushes: bool = False

    def build(self, env: ManagerBasedRlEnv) -> "ChuanliantuiPdAction":
        return ChuanliantuiPdAction(self, env)


class ChuanliantuiPdAction(ActionTerm):
    cfg: ChuanliantuiPdActionCfg

    def __init__(self, cfg: ChuanliantuiPdActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.policy_joint_order = tuple(cfg.policy_joint_order)
        if self.policy_joint_order not in (POLICY_JOINTS, ACTUATED_JOINTS):
            raise ValueError(f"Unsupported policy motor order: {self.policy_joint_order}")
        self.policy_to_physical = torch.tensor(
            [self.policy_joint_order.index(name) for name in ACTUATED_JOINTS],
            dtype=torch.long, device=self.device)
        joint_ids, names = self._entity.find_joints(ACTUATED_JOINTS, preserve_order=True)
        if tuple(names) != ACTUATED_JOINTS:
            raise RuntimeError(f"Unexpected physical motor order: {tuple(names)}")
        self._physical_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self._leg_action_ids = torch.tensor(_LEG_ACTION_IDS, device=self.device)
        self._wheel_action_ids = torch.tensor(_WHEEL_ACTION_IDS, device=self.device)
        self._motor_q_adr = self._entity.indexing.joint_q_adr[self._physical_joint_ids]
        self._motor_v_adr = self._entity.indexing.joint_v_adr[self._physical_joint_ids]
        spring_joint_ids, spring_names = self._entity.find_joints(GAS_SPRING_JOINTS, preserve_order=True)
        if tuple(spring_names) != GAS_SPRING_JOINTS:
            raise RuntimeError(f"Unexpected spring joints: {tuple(spring_names)}")
        self._spring_joint_ids = torch.tensor(spring_joint_ids, dtype=torch.long, device=self.device)
        self._spring_q_adr = self._entity.indexing.joint_q_adr[self._spring_joint_ids]
        self._spring_v_adr = self._entity.indexing.joint_v_adr[self._spring_joint_ids]
        self._spring_ids = torch.as_tensor(gas_spring_actuator_ids(env.sim.mj_model), device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._control_actions = torch.zeros_like(self._raw_actions)
        self._torques = torch.zeros_like(self._raw_actions)
        self._nominal_pos = self._entity.data.default_joint_pos[:, self._physical_joint_ids].clone()
        q_addresses = self._entity.indexing.joint_q_adr[self._physical_joint_ids].cpu().numpy()
        self._reference_pos = torch.tensor(
            env.sim.mj_model.qpos0[q_addresses], dtype=self._nominal_pos.dtype, device=self.device
        ).unsqueeze(0)
        if cfg.nominal_leg_positions is not None:
            self._nominal_pos[:, list(_LEG_ACTION_IDS)] = torch.tensor(
                cfg.nominal_leg_positions, dtype=self._nominal_pos.dtype, device=self.device
            )
        self.kp_scale = torch.ones_like(self._nominal_pos)
        self.kd_scale = torch.ones_like(self._nominal_pos)
        self.motor_strength = torch.ones_like(self._nominal_pos)
        self.nominal_offset = torch.zeros_like(self._nominal_pos)
        self.action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.motor_randomization:
            self.kp_scale.uniform_(.95, 1.05)
            self.kd_scale.uniform_(.95, 1.05)
            self.motor_strength.uniform_(.8, 1.2)
            self.nominal_offset.uniform_(-.03, .03)
            self._nominal_pos += self.nominal_offset
            self.action_delay_steps = torch.empty(self.num_envs, device=self.device).uniform_(
                0., .010 / env.physics_dt).round().long()
        self._action_history = torch.zeros(
            int(self.action_delay_steps.max()) + 1, self.num_envs, self.action_dim, device=self.device)
        self._history_head = 0
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._effort_limits = torch.tensor((40., 40., 3.9, 40., 40., 3.9), device=self.device)
        # Configured gains are fixed after initialization; runtime randomization
        # modifies kp_scale/kd_scale, which the properties still read on each call.
        self._base_position_gains = torch.tensor(
            [cfg.leg_stiffness, cfg.leg_stiffness, 0., cfg.leg_stiffness, cfg.leg_stiffness, 0.],
            device=self.device)
        wheel_gain = cfg.wheel_stiffness + cfg.wheel_damping
        self._base_velocity_gains = torch.tensor(
            [cfg.leg_damping, cfg.leg_damping, wheel_gain, cfg.leg_damping, cfg.leg_damping, wheel_gain],
            device=self.device)
        self._spring_force_bias = torch.zeros(self.num_envs, 2, device=self.device)
        low, high = cfg.spring_force_bias_range
        if low > high:
            raise ValueError("Spring force bias lower bound exceeds upper bound")

    @property
    def action_dim(self) -> int:
        return len(ACTUATED_JOINTS)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def torques(self) -> torch.Tensor:
        return self._torques

    @property
    def nominal_positions(self) -> torch.Tensor:
        """Nominal positions in physical action order, also used by observations."""
        return self._nominal_pos

    @property
    def reference_positions(self) -> torch.Tensor:
        """CAD assembly angles in the current motor coordinate system."""
        return self._reference_pos

    @property
    def position_gains(self) -> torch.Tensor:
        return self.kp_scale * self._base_position_gains

    @property
    def velocity_gains(self) -> torch.Tensor:
        return self.kd_scale * self._base_velocity_gains

    def process_actions(self, actions: torch.Tensor) -> None:
        # Keep policy samples intact for observations and raw-action penalties.
        self._raw_actions.copy_(actions)
        self._control_actions.copy_(actions[:, self.policy_to_physical])
        if self.cfg.standup_landing_gate:
            state = self._env.command_manager.get_term("standup")
            state.policy_active.copy_(state.has_landed)
            self._control_actions[~state.has_landed] = 0.0
        if self.cfg.standup_pushes:
            from .standup_randomization import get_state
            get_state(self._env).update_push()

    def apply_actions(self) -> None:
        self._history_head = (self._history_head + 1) % len(self._action_history)
        self._action_history[self._history_head].copy_(self._control_actions)
        delayed_actions = self._action_history[
            (self._history_head - self.action_delay_steps) % len(self._action_history), self._env_ids]
        # Entity joint_pos/vel first gather every joint. Read only the six motors.
        q = self._env.sim.data.qpos[:, self._motor_q_adr]
        qd = self._env.sim.data.qvel[:, self._motor_v_adr]
        legs, wheels = self._leg_action_ids, self._wheel_action_ids
        reference = self._reference_pos[:, legs]
        target = torch.clamp(
            self._nominal_pos[:, legs] + self.cfg.leg_position_scale * delayed_actions[:, legs],
            reference - 3.14, reference + 3.14,
        )
        self._torques[:, legs] = (self.cfg.leg_stiffness * self.kp_scale[:, legs] * (target - q[:, legs])
                                  - self.cfg.leg_damping * self.kd_scale[:, legs] * qd[:, legs])
        target_velocity = torch.clamp(
            self.cfg.wheel_velocity_scale * delayed_actions[:, wheels],
            -self.cfg.max_wheel_velocity, self.cfg.max_wheel_velocity,
        )
        self._torques[:, wheels] = self.kd_scale[:, wheels] * (
            self.cfg.wheel_stiffness * (target_velocity - qd[:, wheels]) - self.cfg.wheel_damping * qd[:, wheels])
        self._torques.mul_(self.motor_strength)
        self._torques.copy_(torch.clamp(self._torques, -self._effort_limits, self._effort_limits))
        if self.cfg.standup_landing_gate:
            state = self._env.command_manager.get_term("standup")
            self._torques[~state.has_landed] = 0.0
        self._entity.set_joint_effort_target(self._torques, joint_ids=self._physical_joint_ids)
        # Internal spring actuators stay outside the six policy outputs.
        self._env.sim.data.ctrl[:, self._spring_ids] = gas_spring_force(
            self._env.sim.data.qpos[:, self._spring_q_adr],
            self._env.sim.data.qvel[:, self._spring_v_adr],
            self._spring_force_bias,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._control_actions[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._action_history[:, env_ids] = 0.0
        low, high = self.cfg.spring_force_bias_range
        self._spring_force_bias[env_ids] = torch.empty_like(self._spring_force_bias[env_ids]).uniform_(low, high)
