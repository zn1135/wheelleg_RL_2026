from isaacgym import gymtorch
import torch

from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui


class ChuanliantuiStandup(Chuanliantui):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.has_stood = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.standing_time = torch.zeros(self.num_envs, device=self.device)
        # 连续未站稳的控制步数；不复用 runner 会随机初始化的 episode_length_buf。
        self.standup_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        if self.feet_indices.numel() != 2:
            raise RuntimeError("Standup wheel-airborne termination requires exactly two wheels")
        self.wheels_airborne_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # 每次 reset 后先自由落下；首次任一轮有效接地后才允许策略控制和判死。
        self.has_landed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = torch.tensor(
            self.cfg.standup.initial_dof_pos,
            dtype=self.dof_pos.dtype,
            device=self.device,
        )
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def check_termination(self):
        # 在本控制步内已经触地的环境，从下一控制步起交给策略。这里仍不把
        # 本步的零动作当作策略样本（runner 在 step 前读取 action mask）。
        newly_landed = ~self.has_landed & self._has_wheel_contact()
        self.has_landed |= newly_landed
        # 自由落体是策略回合外的物理预备阶段。首次落地才开始计入 2 秒
        # 课程和 episode timeout，避免 train.py 的随机初始回合长度在触地瞬间
        # 触发超时重置。
        self.episode_length_buf[newly_landed] = 0

        standing = (
            self.has_landed
            &
            (self.base_height >= self.cfg.standup.success_height)
            & (
                self.projected_gravity[:, 2]
                <= self.cfg.standup.success_projected_gravity_z
            )
        )
        self.standing_time = torch.where(
            standing,
            self.standing_time + self.dt,
            torch.zeros_like(self.standing_time),
        )
        stable_now = (
            self.standing_time >= self.cfg.standup.success_duration_s
        )
        self.has_stood |= stable_now
        # 全回合生效：每次重新站稳才清零；短暂满足高度/姿态不清零。
        self.standup_steps = torch.where(
            ~self.has_landed,
            torch.zeros_like(self.standup_steps),
            torch.where(
                stable_now,
                torch.zeros_like(self.standup_steps),
                self.standup_steps + 1,
            ),
        )

        base_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fallen = self.has_landed & self.has_stood & (
            base_contact | (self.projected_gravity[:, 2] > -0.1)
        )
        self.fail_buf *= fallen
        self.fail_buf += fallen

        # pg_z 严格为正时立即终止，不依赖 has_stood，也不等待失败计时。
        # 在每个控制步的终止检查中生效；pg_z == 0 不触发本条件。
        inverted = self.has_landed & (self.projected_gravity[:, 2] > 0.0)
        # 连续超过期限未站稳，按失败终止；曾站起过也不豁免。
        failed_to_stand = (
            self.standup_steps > self.cfg.standup.standup_timeout_s / self.dt
        )
        # 当前地形为平地且关闭自碰撞，用世界 z 向轮接触力近似判断地面支撑。
        # 任一轮恢复支撑就清零；不累计多段腾空，也不受历史站起标志限制。
        both_airborne = self.has_landed & self._both_wheels_airborne()
        self.wheels_airborne_steps = torch.where(
            both_airborne,
            self.wheels_airborne_steps + 1,
            torch.zeros_like(self.wheels_airborne_steps),
        )
        airborne_failure = (
            self.wheels_airborne_steps >= self.cfg.standup.wheels_airborne_timeout_s / self.dt
        )
        self.time_out_buf = self.has_landed & (
            self.episode_length_buf > self.max_episode_length
        )
        self.reset_buf = (
            self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt
        ) | inverted | failed_to_stand | airborne_failure | self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        recovered_rate = self.has_stood[env_ids].float().mean()
        super().reset_idx(env_ids)
        self.extras["episode"]["recovered_rate"] = recovered_rate
        self.has_stood[env_ids] = False
        self.standing_time[env_ids] = 0.0
        self.standup_steps[env_ids] = 0
        self.wheels_airborne_steps[env_ids] = 0
        self.has_landed[env_ids] = False

    def get_policy_action_mask(self):
        """返回本控制步可进入 actor/critic 的环境；首次轮接地前为 False。"""
        return self.has_landed

    def _has_wheel_contact(self):
        """任一轮 z 向接触力超过阈值，即视为首次落地。"""
        wheel_contacts = (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.standup.wheel_contact_force_threshold
        )
        return torch.any(wheel_contacts, dim=1)

    def _both_wheels_airborne(self):
        """平地支撑代理，供离地奖励和判死共用；任一轮竖直接触力超过阈值即接地。"""
        return ~self._has_wheel_contact()

    def _reward_wheels_airborne(self):
        # 原始值为0/1，负权重与dt由基类统一应用；不等待0.2秒，也不依赖has_stood。
        return self._both_wheels_airborne().float()

    def _reward_recovered(self):
        # 奖励累积时间可独立调整；has_stood 始终由成功持续时间决定。
        return (self.standing_time / self.cfg.rewards.recovered_reward_duration_s).clip(
            0.0, 1.0
        )

    def _reward_base_height(self):
        initial_height = self.cfg.standup.initial_base_height
        reward_height = self.commands[:, 2] - self.cfg.rewards.height_reward_tolerance
        return (
            (self.base_height - initial_height)
            / (reward_height - initial_height).clamp(min=1e-6)
        ).clip(0.0, 1.0)

    def _reward_orientation(self):
        initial_height = self.cfg.standup.initial_base_height
        height_progress = (
            (self.base_height - initial_height)
            / (self.commands[:, 2] - initial_height)
        ).clip(0.0, 1.0)
        return height_progress * super()._reward_orientation()

    def _reward_tracking_lin_vel(self):
        return self.has_stood * super()._reward_tracking_lin_vel()

    def _reward_tracking_lin_vel_enhance(self):
        return self.has_stood * super()._reward_tracking_lin_vel_enhance()

    def _reward_tracking_ang_vel(self):
        return self.has_stood * super()._reward_tracking_ang_vel()

    def _reward_tracking_ang_vel_enhance(self):
        return self.has_stood * super()._reward_tracking_ang_vel_enhance()

    def _reward_lin_vel_z(self):
        return self.has_stood * super()._reward_lin_vel_z()

    def _reward_ang_vel_xy(self):
        return self.has_stood * super()._reward_ang_vel_xy()

    def _reward_dof_vel(self):
        return self.has_stood * super()._reward_dof_vel()

    def _reward_torques(self):
        return self.has_stood * super()._reward_torques()

    def _reward_action_rate(self):
        return self.has_stood * super()._reward_action_rate()

    def _reward_action_smooth(self):
        return self.has_stood * super()._reward_action_smooth()
