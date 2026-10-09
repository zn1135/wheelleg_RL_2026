# chuanliantui 串联腿轮足平衡机器人环境。
#
# 训练范式照复旦大学星云EGA战队开源方案(plane,github.com/yly-true/fudan_rl_wheel-leg):
#   纯串联二连杆模型训练,训练端不做任何串联→并联转换;"串联力矩转并联力矩"只发生在
#   部署适配层(实车并联五连杆时经雅可比映射,见 sim2sim 阶段)。VMC 极坐标
#   (虚拟腿长 L0 / 摆角 theta0)仅作为正运动学派生量进入 nominal_state 奖励,
#   不进观测、不参与动作解释。
#
# post_physics_step 中使用二连杆 FK 的 CAD 零位偏置。基类公式
#   theta1 = dof_hip, theta2 = dof_knee + π/2
# 是 wl/imcawl 腿形的约定;chuanliantui 的 URDF 零位构型(大腿角 a1=atan2(0.12977,
# 0.1651)≈0.6662,小腿绝对角 a2=atan2(0.18773,−0.1651)≈2.2921,由重排 URDF 几何推得,
# 数值验证 q=0 时 L0=0.3175、theta0=0)对应:
#   theta1 = dof_hip + fk_offset_hip, theta2 = dof_knee + fk_offset_knee
# 偏置从 cfg.asset.fk_offset_* 读取(l1=0.21/l2=0.25 由 URDF 连杆矢量模长实测)。
#
# 混合 PD 继承基类；另构造 25 维观测，并在每个物理子步施加 CAD 气弹簧
# 等效膝力矩。弹簧与电机输出分开，起立子类的随机推力也在此统一提交。

import math

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.chuanliantui_gas_spring import GasSpringGeometry


class Chuanliantui(LeggedRobot):

    _expected_dof_names = ("rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel")
    _leg_position_indices = (0, 1, 3, 4)

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if (
            self.num_dofs != len(self._expected_dof_names)
            or tuple(self.dof_names) != self._expected_dof_names
        ):
            raise RuntimeError(
                "chuanliantui_train.urdf DOF contract changed: "
                f"expected {self._expected_dof_names}, got {tuple(self.dof_names)}"
            )
        self._init_gas_spring()

    def _init_gas_spring(self):
        self.gas_spring_force_n = float(self.cfg.gas_spring.force_n)
        if not math.isfinite(self.gas_spring_force_n) or not 0 <= self.gas_spring_force_n <= 150:
            raise ValueError("gas_spring.force_n 必须是 0~150 N 的有限数")
        urdf_path = self.cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        self.gas_spring_geometry = GasSpringGeometry(urdf_path, device=self.device)
        self.gas_spring_upper_bodies = []
        self.gas_spring_lower_bodies = []
        for target, names in ((self.gas_spring_upper_bodies, self.gas_spring_geometry.upper_bodies),
                              (self.gas_spring_lower_bodies, self.gas_spring_geometry.lower_bodies)):
            for name in names:
                index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                if not 0 <= index < self.num_bodies:
                    raise RuntimeError("Gas spring requires rigid body: " + name)
                target.append(index)
        # 独立记录末个物理子步的被动力矩；self.torques 仍只记录电机。
        self.gas_spring_knee_torques = torch.zeros(self.num_envs, 2, device=self.device)

    def _apply_external_forces(self):
        """合并水平推扰和弹簧，每个 2 ms 子步只提交一次刚体外力。"""
        self.rigid_body_external_forces.zero_()
        self.rigid_body_external_torques.zero_()
        if self.cfg.domain_rand.push_robots:
            self._accumulate_push_forces()
        self.gas_spring_knee_torques[:] = self.gas_spring_geometry.knee_torques(
            self.dof_pos[:, [1, 4]], self.gas_spring_force_n
        )
        # root tensor 必须刷新到当前子步；不能沿用上个 100 Hz 策略拍的姿态。
        self.gym.refresh_actor_root_state_tensor(self.sim)
        axis_local = self.gas_spring_geometry.knee_axes_in_base(self.dof_pos[:, [0, 3]])
        quat = self.root_states[:, 3:7].unsqueeze(1).expand(-1, 2, -1)
        axis_world = quat_rotate(quat.reshape(-1, 4), axis_local.reshape(-1, 3)).view(self.num_envs, 2, 3)
        torque_world = axis_world * self.gas_spring_knee_torques.unsqueeze(-1)
        # 对上下刚体施加等大反向膝轴扭矩，在理想串联铰链上与端点力等效。
        # 独立于 DOF motor effort 限幅，不把弹簧计入电机能耗或限矩奖励。
        self.rigid_body_external_torques[:, self.gas_spring_lower_bodies] += torque_world
        self.rigid_body_external_torques[:, self.gas_spring_upper_bodies] -= torque_world
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques), gymapi.ENV_SPACE,
        )

    def compute_proprioception_observations(self):
        """构造 25 维 actor 观测，连续轮的位置不参与策略或历史编码。"""
        dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        return torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                dof_pos_obs[:, self._leg_position_indices],
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    def _get_noise_scale_vec(self, cfg):
        """与 25 维观测布局一一对应；轮位置已删除，轮速仍有传感器噪声。"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # [vx, yaw_rate, height] commands
        noise_vec[9:13] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[13:19] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[19:25] = 0.0  # previous actions
        return noise_vec

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

        # 虚拟腿量由两侧髋/膝关节角的正运动学得到。
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
        end_x = self.cfg.asset.l1 * torch.cos(theta1) + self.cfg.asset.l2 * torch.cos(
            theta1 + theta2
        )
        end_z = self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            theta1 + theta2
        )
        self.L0 = torch.sqrt(end_x**2 + end_z**2)
        self.theta0 = torch.arctan2(end_z, end_x) - self.pi / 2

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
