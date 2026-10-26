"""Isaac ChuanliantuiStandup pre-unlock rewards on the current mjlab robot.

Only the reward recipe is ported: current commands, resets, termination and
0.2 s motor hold stay unchanged. No automatic 0.22 m curriculum is introduced.
"""
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from ..envs.chuanliantui.model import ACTUATED_JOINTS
from ..envs.chuanliantui.standup_mdp import base_contact_force

WEIGHTS = dict(tracking_lin_vel=.2, tracking_lin_vel_enhance=.2,
    tracking_ang_vel=.2, tracking_ang_vel_enhance=.2, base_height=1.,
    orientation=-1., nominal_state=-3., dof_pos_limits=-1., knee_limit_margin=-.3,
    recovered=1., base_link_contact=-.3, base_link_airborne=.2, wheels_airborne=-.3,
    lin_vel_z=-1., ang_vel_xy=-.2, dof_vel=-5e-5, torques=-1e-4,
    action_rate=-.01, action_smooth=-.01)
TIGHT_WEIGHTS = {**WEIGHTS, 'tracking_lin_vel': 1.}


def velocity_scores(command, linear_velocity, stood, *, tight=False, convex=False):
    error = (command[:, 0]-linear_velocity[:, 0]).square()
    if tight:
        # 当前零速站立同时抑制前后和侧向漂移；宽尺度项保留较大误差的减速反馈。
        error = error+linear_velocity[:, 1].square()
        main, enhance = torch.exp(-error/.02**2), torch.exp(-error/.2**2)-1.
        if convex:
            # 沿用原站稳门控；线性+平方速度成本不会奖励“停一下再冲一下”。
            main = 1.-10.*((error+1e-12).sqrt()-1e-6)
            enhance = -error/.2**2
        return stood*main, stood*enhance
    return stood*torch.exp(-error/.25), stood*(torch.exp(-error/2.5)-1.)


def reward_values(*, command, height, gravity, linear_velocity, angular_velocity,
                  base_force, wheel_supported, landed, stood, standing_time,
                  leg_swing, leg_q, leg_dq, soft_limits, knee_q, knee_limits,
                  torque, action, previous_action, previous_previous_action, tight_velocity=False,
                  convex_velocity=False):
    contact = (base_force/5.).clamp(0., 1.)
    linear, enhance = velocity_scores(command, linear_velocity, stood, tight=tight_velocity, convex=convex_velocity)
    yaw_error = (command[:, 1]-angular_velocity[:, 2]).square()
    penetration = torch.maximum(knee_limits[..., 0]+.05-knee_q,
                                knee_q-knee_limits[..., 1]+.05).clamp_min(0.)
    second = action-2*previous_action+previous_previous_action
    # 沿用源代码的高度渐进姿态惩罚；目标取当前mjlab命令，不改高度课程。
    progress = ((height-.15)/(command[:, 2]-.15)).clamp(0., 1.)
    return dict(
        tracking_lin_vel=linear,
        tracking_lin_vel_enhance=enhance,
        tracking_ang_vel=stood*torch.exp(-yaw_error/.25),
        tracking_ang_vel_enhance=stood*(torch.exp(-yaw_error/2.5)-1.),
        base_height=torch.exp(-(height-command[:, 2]).square()/.01)*(1.-.8*contact),
        orientation=progress*gravity[:, :2].square().sum(-1),
        nominal_state=(leg_swing[:, 0]-leg_swing[:, 1]).square(),
        dof_pos_limits=((soft_limits[..., 0]-leg_q).clamp_min(0.)+
                        (leg_q-soft_limits[..., 1]).clamp_min(0.)).sum(-1),
        knee_limit_margin=stood*(penetration/.05).square().mean(-1),
        recovered=(standing_time/.5).clamp(0., 1.),
        base_link_contact=landed*contact, base_link_airborne=landed*(1.-contact),
        wheels_airborne=(~wheel_supported.any(-1)).float(),
        lin_vel_z=stood*linear_velocity[:, 2].square(),
        ang_vel_xy=stood*angular_velocity[:, :2].square().sum(-1),
        dof_vel=stood*leg_dq.square().sum(-1), torques=stood*torque.square().sum(-1),
        action_rate=stood*(action-previous_action).square().sum(-1),
        action_smooth=stood*second[:, (0, 1, 3, 4)].square().sum(-1))


class RewardState:
    def __init__(self, env, *, tight_velocity=False, convex_velocity=False):
        self.tight_velocity = tight_velocity
        self.convex_velocity = convex_velocity
        self.weights = TIGHT_WEIGHTS if tight_velocity else WEIGHTS
        self.robot = env.scene['robot']
        self.legs = self.robot.find_joints(('lf0', 'lf1', 'rf0', 'rf1'), preserve_order=True)[0]
        self.knees = self.robot.find_joints(('lf1', 'rf1'), preserve_order=True)[0]
        motors = self.robot.find_joints(ACTUATED_JOINTS, preserve_order=True)[0]
        self.motor_vadr = self.robot.indexing.joint_v_adr[motors]
        self.hips = self.robot.find_bodies(('lf0', 'rf0'), preserve_order=True)[0]
        self.wheels = self.robot.find_bodies(('lfwheel', 'rfwheel'), preserve_order=True)[0]
        limits = self.robot.data.default_joint_pos_limits[:, self.legs].clone()
        mid, half = limits.mean(-1), (limits[..., 1]-limits[..., 0])*.5*.97
        self.soft_limits = torch.stack((mid-half, mid+half), -1)
        self.knee_limits = self.robot.data.default_joint_pos_limits[:, self.knees].clone()
        self.landed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stood = torch.zeros_like(self.landed)
        self.standing_time = torch.zeros(env.num_envs, device=env.device)
        self.step = -1
        self.values = {}

    def reset(self, ids):
        self.landed[ids] = self.stood[ids] = False
        self.standing_time[ids] = 0.
        self.step = -1

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        state = env.command_manager.get_term('standup')
        data = self.robot.data
        height = state.height_above_ground
        force = base_contact_force(env)
        # netforce为世界系、从轮指向地面；取反后才是Isaac的向上支撑力。
        support = -env.scene['wheel_contact'].data.force[..., 2] > 1.
        self.landed |= support.any(-1)
        # 保留源代码“目标高度下2cm、直立且机身离地连续0.5s”的奖励门控。
        stable = (self.landed & (height >= state.command[:, 2]-.02)
                  & (data.projected_gravity_b[:, 2] <= -.9) & (force <= .1))
        self.standing_time = torch.where(stable, self.standing_time+env.step_dt, 0.)
        self.stood |= self.standing_time >= .5
        # 当前XML是闭链、双电机腿；直接用真实髋到轮心的机体系向量，避免套旧URDF零点。
        relative = data.body_link_pos_w[:, self.wheels]-data.body_link_pos_w[:, self.hips]
        quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        relative = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 2, 3)
        swing = torch.atan2(-relative[..., 0], -relative[..., 2])
        manager = env.action_manager
        values = reward_values(command=state.command, height=height, gravity=data.projected_gravity_b,
            linear_velocity=data.root_link_lin_vel_b, angular_velocity=data.root_link_ang_vel_b,
            base_force=force, wheel_supported=support, landed=self.landed, stood=self.stood,
            standing_time=self.standing_time, leg_swing=swing,
            leg_q=data.joint_pos[:, self.legs], leg_dq=data.joint_vel[:, self.legs],
            soft_limits=self.soft_limits, knee_q=data.joint_pos[:, self.knees], knee_limits=self.knee_limits,
            torque=env.sim.data.qfrc_actuator[:, self.motor_vadr], action=manager.action,
            previous_action=manager.prev_action, previous_previous_action=manager.prev_prev_action,
            tight_velocity=self.tight_velocity,
            convex_velocity=self.convex_velocity)
        # 与Isaac每项加权后限幅±1等价；RewardManager再统一乘一次dt。
        self.values = {name: (value if self.convex_velocity and name in ('tracking_lin_vel', 'tracking_lin_vel_enhance')
                             else value.clamp(-1./abs(self.weights[name]), 1./abs(self.weights[name])))
                       * state.policy_active for name, value in values.items()}
        self.step = env.common_step_counter
        return self.values


def reward(env, *, name, tight_velocity=False, convex_velocity=False):
    if not hasattr(env, '_isaac_standup_rewards'):
        env._isaac_standup_rewards = RewardState(env, tight_velocity=tight_velocity, convex_velocity=convex_velocity)
    return env._isaac_standup_rewards.compute(env)[name]


def configure_rewards(cfg, *, tight_velocity=False, convex_velocity=False):
    # 替换整个任务奖励，避免混入旧华南虎项／停车项；SMP仍独立保留。
    cfg.rewards = {'smp_motion': cfg.rewards['smp_motion'], **{
        name: RewardTermCfg(func=reward, weight=weight, params={'name': name,
            **({'tight_velocity': True} if tight_velocity else {}),
            **({'convex_velocity': True} if convex_velocity else {})})
        for name, weight in (TIGHT_WEIGHTS if tight_velocity else WEIGHTS).items()}}
