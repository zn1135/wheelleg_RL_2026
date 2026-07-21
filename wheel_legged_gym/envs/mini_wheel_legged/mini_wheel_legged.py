# imcawl 迷你轮足机器人环境子类。
#
# 为什么需要子类而不是直接用 LeggedRobot：
#   基类是为 wl 机器人写的，其 URDF 腿装在机体 y=±0.17（前进方向 = 机体 x）。
#   imcawl 的 URDF 腿装在机体 x=±0.179（x 是左右方向），轮子绕 x 轴转动，
#   只能沿机体 y 方向滚动 —— 前进方向是 y 而非 x。
#   基类 _reward_tracking_lin_vel 跟踪 base_lin_vel[:, 0]（对 imcawl 是不可驱动的
#   横向速度），导致策略学不会跟速度命令，只学会"前倾斜靠停机"骗取存活/高度奖励
#   （2026-07-21 sim2sim 排查结论，同 c354431 修过的 ang_vel wrong index 是同款问题）。

import torch

from wheel_legged_gym.envs.base.legged_robot import LeggedRobot


class MiniWheelLegged(LeggedRobot):

    def _reward_tracking_lin_vel(self):
        # imcawl 前进方向为机体 y（基类的 [:, 0] 是轮子不可驱动的横向）
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_enhance(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def check_termination(self):
        """在基类逻辑基础上收紧倾角失败阈值。

        基类只在 pg_z > -0.1（倾角 >84°）时记失败，策略曾利用这个宽松阈值学会
        "前倾 ~40°、腿蹭地当支架"的停机姿态骗过终止。这里把阈值收紧到
        cfg.env.fail_tilt_pg_z（默认 -0.85，约 32°），持续超限
        fail_to_terminal_time_s 秒才终止，不影响机动中的短暂大倾角。
        """
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fail_buf |= self.projected_gravity[:, 2] > self.cfg.env.fail_tilt_pg_z
        self.fail_buf *= fail_buf
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.edge_reset_buf
        )
