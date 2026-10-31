"""Stand upright first, then track height; retain the other Isaac reward terms."""

import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ...smp.isaac_standup_rewards import WEIGHTS as ISAAC_WEIGHTS, RewardState as IsaacRewardState, reward_values
from ..chuanliantui.standup_mdp import base_contact_force


RECIPE_VERSION = "stand-v2-upright-first-height-gate-v3"
TRACKING_RECIPE_VERSION = "stand-v2-upright-first-tracking-weight1-v4"
ACTIVE_TRACKING_RECIPE_VERSION = "stand-v2-upright-first-tracking-active-weight1-v5"
TIGHT_TRACKING_RECIPE_VERSION = "stand-v2-upright-first-tracking-active-weight1-tight-v6"
WEIGHTS = {**ISAAC_WEIGHTS, "orientation": -1., "tracking_lin_vel_tight": .5}


def configure_tight_reward(env_cfg, command_cfg):
    from mjlab.managers.reward_manager import RewardTermCfg
    if not command_cfg.tracking_weight_while_active:
        raise ValueError("tracking_tight_term requires tracking_weight_while_active")
    for value in (command_cfg.tracking_tight_kernel_denominator, command_cfg.tracking_tight_weight):
        if not math.isfinite(value) or value <= 0.:
            raise ValueError("Tight kernel denominator and weight must be finite and positive")
    env_cfg.rewards["tracking_lin_vel_tight"] = RewardTermCfg(func=reward,
        weight=command_cfg.tracking_tight_weight, params={"name": "tracking_lin_vel_tight"})


def restore_tracking_reward_cfg(cfg, infos):
    cfg.tracking_tight_term = infos.get("recipe_version") == TIGHT_TRACKING_RECIPE_VERSION
    cfg.tracking_weight_while_active = cfg.tracking_tight_term or infos.get("recipe_version") == ACTIVE_TRACKING_RECIPE_VERSION
    if cfg.tracking_tight_term:
        config = infos["tracking_tight_config"]
        cfg.tracking_tight_kernel_denominator = config["kernel_denominator"]
        cfg.tracking_tight_weight = config["weight"]
        if any(not math.isfinite(value) or value <= 0. for value in config.values()):
            raise ValueError("Tight checkpoint kernel denominator and weight must be finite and positive")


class RewardState(IsaacRewardState):
    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        state = env.command_manager.get_term("standup")
        data = self.robot.data
        force = base_contact_force(env)
        support = -env.scene["wheel_contact"].data.force[:, :, 2] > 1.
        # Use physical hip-to-wheel geometry, not the serial URDF's angle zeros.
        relative = data.body_link_pos_w[:, self.wheels] - data.body_link_pos_w[:, self.hips]
        quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        relative = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 2, 3)
        swing = torch.atan2(-relative[..., 0], -relative[..., 2])
        manager = env.action_manager
        values = reward_values(command=state.command, height=state.height_above_ground,
            gravity=data.projected_gravity_b, linear_velocity=data.root_link_lin_vel_b,
            angular_velocity=data.root_link_ang_vel_b, base_force=force, wheel_supported=support,
            landed=state.has_landed, stood=state.has_stood, standing_time=state.standing_time,
            leg_swing=swing, leg_q=data.joint_pos[:, self.legs], leg_dq=data.joint_vel[:, self.legs],
            soft_limits=self.soft_limits, knee_q=data.joint_pos[:, self.knees], knee_limits=self.knee_limits,
            torque=env.sim.data.qfrc_actuator[:, self.motor_vadr], action=manager.action,
            previous_action=manager.prev_action, previous_previous_action=manager.prev_prev_action)
        gravity_z = data.projected_gravity_b[:, 2].clamp(-1., 1.)
        tilt = torch.acos(-gravity_z)
        upright = ((math.radians(40) - tilt) / math.radians(25)).clamp(0., 1.)
        # Global unlock alone is insufficient: each reset must earn a current stable stance.
        precise = state.unlocked & (state.standing_time >= .5)
        denominator = torch.where(precise, .0025, .01)
        values["base_height"] = torch.exp(
            -(state.height_above_ground - state.command[:, 2]).square() / denominator
        ) * upright * (1. - .8 * (force / 5.).clamp(0., 1.))
        # This remains distinct through inversion and is not clipped to a weighted -1.
        values["orientation"] = 1. + gravity_z
        damping = torch.where(state.has_stood, 1., .1)
        values["lin_vel_z"] = damping * data.root_link_lin_vel_b[:, 2].square()
        values["ang_vel_xy"] = damping * data.root_link_ang_vel_b[:, :2].square().sum(-1)
        if state.cfg.post_stand_tracking:
            # 基础权重0.2；新配方在整个跟踪激活期乘5，包括急停后的零速段。
            boosted = state.tracking_active if state.cfg.tracking_weight_while_active else state.command[:, 0] != 0.
            values["tracking_lin_vel"] *= torch.where(boosted, 5., 1.)
        if state.cfg.tracking_tight_term:
            # 单独注册，避免与已经达到限幅上限的宽核主项相加。
            error = state.command[:, 0] - data.root_link_lin_vel_b[:, 0]
            values["tracking_lin_vel_tight"] = torch.exp(
                -error.square() / state.cfg.tracking_tight_kernel_denominator
            ).masked_fill(~state.tracking_active, 0.)
        # RewardManager multiplies weight and dt once; the tight kernel is already in [0,1].
        self.values = {name: (value if name in ("orientation", "tracking_lin_vel_tight") else
            value.clamp(-1. / abs(WEIGHTS[name]), 1. / abs(WEIGHTS[name])))
            .masked_fill(state.numerical_failure | ~state.reward_active, 0.) for name, value in values.items()}
        if getattr(state.cfg, "role", None) == "getup":
            # 起立出力后即惩罚六电机实测力矩；保留等待期/异常屏蔽和加权-1上限。
            self.values["torques"] = env.sim.data.qfrc_actuator[:, self.motor_vadr].square().sum(-1).clamp(
                max=1. / abs(env.cfg.rewards["torques"].weight)
            ).masked_fill(state.numerical_failure | ~state.reward_active, 0.)
        else:
            progress = (state.standing_time / .5).clamp(0., 1.)
            progress = torch.where(state.has_stood, 1., progress)
            # 旧stand_v2保留渐进缩放；独立getup恢复原1999的全强度惩罚。
            self.values["nominal_state"] *= .1 + .9 * progress
        self.step = env.common_step_counter
        return self.values


def reward(env, *, name):
    if not hasattr(env, "_isaac_v2_rewards"):
        env._isaac_v2_rewards = RewardState(env)
    return env._isaac_v2_rewards.compute(env)[name]
