# chuanliantui 串联腿轮足平衡机器人环境。
#
# 训练范式照复旦大学星云EGA战队开源方案(plane,github.com/yly-true/fudan_rl_wheel-leg):
#   纯串联二连杆模型训练,训练端不做任何串联→并联转换;"串联力矩转并联力矩"只发生在
#   部署适配层(实车并联五连杆时经雅可比映射,见 sim2sim 阶段)。VMC 极坐标
#   (虚拟腿长 L0 / 摆角 theta0)仅作为正运动学派生量进入 nominal_state 奖励,
#   不进观测、不参与动作解释。
#
# 与基类的唯一差异:post_physics_step 中二连杆 FK 的零位偏置。基类公式
#   theta1 = dof_hip, theta2 = dof_knee + π/2
# 是 wl/imcawl 腿形的约定;chuanliantui 的 URDF 零位构型(大腿角 a1=atan2(0.12977,
# 0.1651)≈0.6662,小腿绝对角 a2=atan2(0.18773,−0.1651)≈2.2921,由重排 URDF 几何推得,
# 数值验证 q=0 时 L0=0.3175、theta0=0)对应:
#   theta1 = dof_hip + fk_offset_hip, theta2 = dof_knee + fk_offset_knee
# 偏置从 cfg.asset.fk_offset_* 读取(l1=0.21/l2=0.25 由 URDF 连杆矢量模长实测)。
#
# 其余逻辑——混合 PD(_compute_torques 的腿位置环+轮速度环索引恰与 DOF 字母序兼容)、
# 终止判定、奖励装配、观测构造——全部继承基类,不覆写。

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from wheel_legged_gym.envs.base.legged_robot import LeggedRobot


class Chuanliantui(LeggedRobot):

    def post_physics_step(self):
        """与基类 legged_robot.post_physics_step 逐行一致,仅 FK 段换成带零位偏置的版本。"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

        # ---- chuanliantui 二连杆 FK(唯一改动段)----
        off_hip = self.cfg.asset.fk_offset_hip
        off_knee = self.cfg.asset.fk_offset_knee
        theta1 = torch.cat(
            (
                (self.dof_pos[:, 0] + off_hip).unsqueeze(1),
                (-self.dof_pos[:, 3] + off_hip).unsqueeze(1),
            ),
            dim=1,
        )
        theta2 = torch.cat(
            (
                (self.dof_pos[:, 1] + off_knee).unsqueeze(1),
                (-self.dof_pos[:, 4] + off_knee).unsqueeze(1),
            ),
            dim=1,
        )
        end_x = (
            self.cfg.asset.offset
            + self.cfg.asset.l1 * torch.cos(theta1)
            + self.cfg.asset.l2 * torch.cos(theta1 + theta2)
        )
        end_y = self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            theta1 + theta2
        )
        self.L0 = torch.sqrt(end_x**2 + end_y**2)
        self.theta0 = torch.arctan2(end_y, end_x) - self.pi / 2
        # ---- FK 段结束 ----

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
