from isaacgym import gymtorch
import torch

from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui


class ChuanliantuiStandup(Chuanliantui):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.has_stood = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.standing_time = torch.zeros(self.num_envs, device=self.device)
        if self.feet_indices.numel() != 2:
            raise RuntimeError("Standup wheel-airborne termination requires exactly two wheels")
        self.wheels_airborne_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # 每次 reset 后先自由落下；首次任一轮有效接地后才允许策略控制和判死。
        self.has_landed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # 课程状态属于全局训练状态，不随单个环境 reset 回退，并随 checkpoint 保存。
        self.standup_curriculum_unlocked = False
        self.standup_curriculum_completed_episodes = 0
        self.standup_curriculum_recovered_episodes = 0
        self.standup_curriculum_last_recovered_rate = 0.0

    def get_checkpoint_state(self):
        """返回必须跨训练进程延续的站立课程状态。"""
        return {
            "standup_curriculum": {
                "unlocked": self.standup_curriculum_unlocked,
                "completed_episodes": self.standup_curriculum_completed_episodes,
                "recovered_episodes": self.standup_curriculum_recovered_episodes,
                "last_recovered_rate": self.standup_curriculum_last_recovered_rate,
            }
        }

    def load_checkpoint_state(self, state):
        """恢复站立课程，并把当前全部环境同步到恢复后的命令阶段。"""
        curriculum_state = state.get("standup_curriculum") if state else None
        if curriculum_state is None:
            return False

        self.standup_curriculum_unlocked = bool(curriculum_state["unlocked"])
        self.standup_curriculum_completed_episodes = int(
            curriculum_state["completed_episodes"]
        )
        self.standup_curriculum_recovered_episodes = int(
            curriculum_state["recovered_episodes"]
        )
        self.standup_curriculum_last_recovered_rate = float(
            curriculum_state["last_recovered_rate"]
        )
        if self.standup_curriculum_unlocked:
            # reward_scales 已在父类初始化时乘过 dt；只恢复切换后的有效权重。
            self.reward_scales["orientation"] = (
                self.cfg.standup_curriculum.post_unlock_orientation_scale * self.dt
            )
        self.commands[:, 2] = self._current_standup_target_height()
        return True

    def _current_standup_target_height(self):
        curriculum = self.cfg.standup_curriculum
        return (
            curriculum.post_unlock_target_height
            if self.standup_curriculum_unlocked
            else curriculum.pre_unlock_target_height
        )

    def _current_standup_success_height(self):
        curriculum = self.cfg.standup_curriculum
        return (
            curriculum.post_unlock_success_height
            if self.standup_curriculum_unlocked
            else curriculum.pre_unlock_success_height
        )

    def _resample_commands(self, env_ids):
        """重置时按全局站立课程写入固定高度命令。"""
        super()._resample_commands(env_ids)
        self.commands[env_ids, 2] = self._current_standup_target_height()

    def _update_standup_curriculum(self, completed_episodes, recovered_episodes):
        """按完整全局回合窗口评估恢复率；解锁后不再回退。"""
        if self.standup_curriculum_unlocked:
            return False
        self.standup_curriculum_completed_episodes += completed_episodes
        self.standup_curriculum_recovered_episodes += recovered_episodes
        if (
            self.standup_curriculum_completed_episodes
            < self.cfg.standup_curriculum.unlock_window_episodes
        ):
            return False

        recovered_rate = (
            self.standup_curriculum_recovered_episodes
            / self.standup_curriculum_completed_episodes
        )
        self.standup_curriculum_last_recovered_rate = recovered_rate
        self.standup_curriculum_completed_episodes = 0
        self.standup_curriculum_recovered_episodes = 0
        if recovered_rate < self.cfg.standup_curriculum.unlock_recovered_rate:
            return False

        self.standup_curriculum_unlocked = True
        # reward_scales 已在父类初始化时乘过 dt；此处只切换姿态项的有效权重。
        self.reward_scales["orientation"] = (
            self.cfg.standup_curriculum.post_unlock_orientation_scale * self.dt
        )
        # 正在运行的环境立即切换，避免等待各自 reset 后混用两套课程。
        self.commands[:, 2] = self._current_standup_target_height()
        return True

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
        # 首次轮触地前是策略回合外的物理预备阶段。首次落地才开始计入 episode
        # timeout，避免 train.py 的随机初始回合长度在触地瞬间触发超时重置。
        self.episode_length_buf[newly_landed] = 0

        base_contact = self._base_link_has_contact(
            self.cfg.standup.success_base_contact_force_threshold
        )
        upright = (
            self.projected_gravity[:, 2]
            <= self.cfg.standup.success_projected_gravity_z
        )
        # 起立首先必须达到目标高度且机身竖直。ground-standup 还要求
        # base_link 脱离地面，避免低姿态仅由轮子支撑被误判为“已起立”。
        standing = (
            self.has_landed
            & (self.base_height >= self._current_standup_success_height())
            & upright
        )
        if self.cfg.standup.success_requires_base_contact_free:
            standing &= ~base_contact
        self.standing_time = torch.where(
            standing,
            self.standing_time + self.dt,
            torch.zeros_like(self.standing_time),
        )
        stable_now = (
            self.standing_time >= self.cfg.standup.success_duration_s
        )
        self.has_stood |= stable_now

        base_contact = self._base_link_has_contact(10.0)
        fallen = self.has_landed & self.has_stood & (
            base_contact | (self.projected_gravity[:, 2] > -0.1)
        )
        self.fail_buf *= fallen
        self.fail_buf += fallen

        # pg_z 严格为正时立即终止，不依赖 has_stood，也不等待失败计时。
        # 在每个控制步的终止检查中生效；pg_z == 0 不触发本条件。
        inverted = self.has_landed & (self.projected_gravity[:, 2] > 0.0)
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
        ) | inverted | airborne_failure | self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        recovered_rate = self.has_stood[env_ids].float().mean()
        unlocked_now = self._update_standup_curriculum(
            completed_episodes=len(env_ids),
            recovered_episodes=int(self.has_stood[env_ids].sum().item()),
        )
        super().reset_idx(env_ids)
        self.extras["episode"]["recovered_rate"] = recovered_rate
        self.extras["episode"]["standup_curriculum_unlocked"] = float(
            self.standup_curriculum_unlocked
        )
        self.extras["episode"]["standup_curriculum_recent_recovered_rate"] = (
            self.standup_curriculum_last_recovered_rate
        )
        self.extras["episode"]["standup_curriculum_target_height"] = (
            self._current_standup_target_height()
        )
        if unlocked_now:
            print(
                "[standup curriculum] recovered_rate={:.3f}，已永久解锁："
                "height=0.20 m，orientation=-10".format(
                    self.standup_curriculum_last_recovered_rate
                )
            )
        self.has_stood[env_ids] = False
        self.standing_time[env_ids] = 0.0
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
        """平地支撑代理，供双轮离地判死使用；任一轮竖直接触力超过阈值即接地。"""
        return ~self._has_wheel_contact()

    def _base_link_has_contact(self, force_threshold):
        """base_link 任一接触力超过阈值即视为仍在地面上。"""
        return self._base_link_contact_force() > force_threshold

    def _base_link_contact_force(self):
        """返回 base_link 各接触点中的最大合力 [N]。"""
        base_contact_forces = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        )
        return torch.amax(base_contact_forces, dim=1)

    def _base_link_contact_ratio(self):
        """奖励用连续接触比例：0 N 为离地，达到配置力值后为满接触。"""
        return (
            self._base_link_contact_force()
            / self.cfg.rewards.base_link_reward_force_scale
        ).clip(0.0, 1.0)

    def _reward_base_link_contact(self):
        """首次轮触地后，按 base_link 连续接触比例返回惩罚。"""
        return self.has_landed.float() * self._base_link_contact_ratio()

    def _reward_base_link_airborne(self):
        """首次轮触地后，按连续离地比例返回正奖励。"""
        return self.has_landed.float() * (1.0 - self._base_link_contact_ratio())

    def _reward_wheels_airborne(self):
        """双轮同时无有效支撑时返回惩罚指示。"""
        return self._both_wheels_airborne().float()

    def _reward_recovered(self):
        # 奖励累积时间可独立调整；has_stood 始终由成功持续时间决定。
        return (self.standing_time / self.cfg.rewards.recovered_reward_duration_s).clip(
            0.0, 1.0
        )

    def _reward_base_height(self):
        """按高度误差给指数奖励，并用连续 base_link 接触力软门控。"""
        height_error_sq = torch.square(self.base_height - self.commands[:, 2])
        height_reward = torch.exp(
            -height_error_sq / self.cfg.rewards.height_reward_sigma
        )
        base_link_airborne = 1.0 - self._base_link_contact_ratio()
        gate = self.cfg.rewards.height_reward_contact_factor + (
            1.0 - self.cfg.rewards.height_reward_contact_factor
        ) * base_link_airborne
        return height_reward * gate

    def _reward_orientation(self):
        initial_height = self.cfg.standup.initial_base_height
        height_progress = (
            (self.base_height - initial_height)
            / (self.commands[:, 2] - initial_height)
        ).clip(0.0, 1.0)
        return height_progress * super()._reward_orientation()

    def _reward_leg_angle(self):
        """奖励左右虚拟腿接近 theta0=0 的绝对摆角，不只奖励两腿彼此对称。"""
        angle_error_sq = torch.sum(torch.square(self.theta0), dim=1)
        return torch.exp(-angle_error_sq / self.cfg.rewards.leg_angle_reward_sigma)

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
