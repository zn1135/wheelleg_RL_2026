"""Get-up retains its recipe; walking rewards have no get-up phase dependence."""

import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from ..chuanliantui_stand_v2.rewards import WEIGHTS, reward as getup_reward
from ...smp.isaac_standup_rewards import RewardState as PhysicalRewardState
from ..chuanliantui.standup_mdp import base_contact_force
from ..chuanliantui.model import ACTUATED_JOINTS

# Two independent recipe dictionaries; changing one role does not change the other.
GETUP_WEIGHTS = {key:value for key,value in WEIGHTS.items() if key != 'tracking_lin_vel_tight'}
GETUP_WEIGHTS['torques'] = -2e-4
GETUP_REWARD_RECIPE = 'split-getup-torque2-prestand-full-nominal-state-v4'
WALK_WEIGHTS = {key:value for key,value in WEIGHTS.items()
                if key not in ('recovered', 'base_link_airborne', 'lin_vel_z',
                               'base_link_contact', 'wheels_airborne', 'action_rate', 'action_smooth')}
WALK_WEIGHTS['nominal_state'] = -3.
WALK_WEIGHTS['orientation'] = -100.
WALK_WEIGHTS['tracking_ang_vel'] = 1.
WALK_REWARD_RECIPE = 'split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-no-vz-contact-rate-smooth-v7'
# 0～5°倾斜不罚；超出后按原1−cos继续增长，加权后到−5（约19°）才饱和。
WALK_ORIENTATION_FREE_RAD = math.radians(5.)
WALK_WEIGHTED_LIMITS = {'orientation': 5., 'nominal_state': 10.}


class WalkRewardState(PhysicalRewardState):
    def __init__(self, env):
        super().__init__(env)
        # Penalize the four driven leg joints; passive knees keep their own margin term.
        names = tuple(ACTUATED_JOINTS[i] for i in (0, 1, 3, 4))
        self.legs = self.robot.find_joints(names, preserve_order=True)[0]
        limits = self.robot.data.default_joint_pos_limits[:, self.legs].clone()
        # Unlimited XML joints stay unlimited; avoid inf-inf when forming soft bounds.
        limited = limits.isfinite().all(-1, keepdim=True)
        finite_limits = limits.masked_fill(~limited, 0.)
        mid, half = finite_limits.mean(-1), (finite_limits[..., 1]-finite_limits[..., 0])*.5*.97
        self.soft_limits = torch.where(limited, torch.stack((mid-half, mid+half), -1), limits)

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        state = env.command_manager.get_term('standup')
        data = self.robot.data
        # 与起立奖励相同：将髋到轮的几何向量转到机体系，计算左右腿摆角。
        relative = data.body_link_pos_w[:, self.wheels] - data.body_link_pos_w[:, self.hips]
        quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        relative = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 2, 3)
        swing = torch.atan2(-relative[..., 0], -relative[..., 2])
        contact = (base_contact_force(env)/5.).clamp(0., 1.)
        vx_error = (state.command[:, 0]-data.root_link_lin_vel_b[:, 0]).square()
        yaw_error = (state.command[:, 1]-data.root_link_ang_vel_b[:, 2]).square()
        gravity_z = data.projected_gravity_b[:, 2].clamp(-1., 1.)
        upright = ((math.radians(40)-torch.acos(-gravity_z))/math.radians(25)).clamp(0., 1.)
        leg_q = data.joint_pos[:, self.legs]
        knee_q = data.joint_pos[:, self.knees]
        penetration = torch.maximum(self.knee_limits[..., 0]+.05-knee_q,
                                    knee_q-self.knee_limits[..., 1]+.05).clamp_min(0.)
        # 停车锁点后不再×5，运动段的跟踪收益明显高于原地站立。
        boost = torch.where(state.tracking_active & ~state.position_locked, 5., 1.)
        # |vx命令|≤0.5时仍为原0.25核；更高速时按相对误差放宽，避免站着与慢跑同分。
        vx_kernel = state.command[:, 0].square().clamp_min(.25)
        values = dict(
            tracking_lin_vel=boost*torch.exp(-vx_error/vx_kernel),
            tracking_lin_vel_enhance=torch.exp(-vx_error/2.5)-1.,
            tracking_ang_vel=torch.exp(-yaw_error/.25),
            tracking_ang_vel_enhance=torch.exp(-yaw_error/2.5)-1.,
            tracking_lin_vel_tight=torch.exp(-vx_error/state.cfg.tracking_tight_kernel_denominator)
                .masked_fill(~state.tracking_active, 0.),
            # Walking uses precise height tracking from the first policy step.
            base_height=torch.exp(-(state.height_above_ground-state.command[:, 2]).square()/.0025)
                *upright*(1.-.8*contact),
            nominal_state=(swing[:, 0]-swing[:, 1]).square(),
            orientation=(1.+gravity_z-(1.-math.cos(WALK_ORIENTATION_FREE_RAD))).clamp_min(0.),
            dof_pos_limits=((self.soft_limits[..., 0]-leg_q).clamp_min(0.)+
                            (leg_q-self.soft_limits[..., 1]).clamp_min(0.)).sum(-1),
            knee_limit_margin=(penetration/.05).square().mean(-1),
            ang_vel_xy=data.root_link_ang_vel_b[:, :2].square().sum(-1),
            dof_vel=data.joint_vel[:, self.legs].square().sum(-1),
            torques=env.sim.data.qfrc_actuator[:, self.motor_vadr].square().sum(-1))
        self.values = {name:(value if name == 'tracking_lin_vel_tight' else
            value.clamp(-WALK_WEIGHTED_LIMITS.get(name, 1.)/abs(WALK_WEIGHTS[name]),
                        WALK_WEIGHTED_LIMITS.get(name, 1.)/abs(WALK_WEIGHTS[name])))
            .masked_fill(state.numerical_failure | ~state.reward_active, 0.) for name,value in values.items()}
        self.step = env.common_step_counter
        return self.values


def reward(env, *, name):
    if env.command_manager.get_term('standup').cfg.role == 'getup':
        return getup_reward(env, name=name)
    if not hasattr(env, '_split_walk_rewards'):
        env._split_walk_rewards = WalkRewardState(env)
    return env._split_walk_rewards.compute(env)[name]
