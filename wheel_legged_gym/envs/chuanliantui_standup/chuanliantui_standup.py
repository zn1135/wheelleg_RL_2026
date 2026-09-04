from isaacgym import gymtorch
import torch

from wheel_legged_gym.envs.chuanliantui.chuanliantui import Chuanliantui


class ChuanliantuiStandup(Chuanliantui):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.has_stood = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.standing_time = torch.zeros(self.num_envs, device=self.device)

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
        standing = (
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
        self.has_stood |= (
            self.standing_time >= self.cfg.standup.success_duration_s
        )

        base_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fallen = self.has_stood & (
            base_contact | (self.projected_gravity[:, 2] > -0.1)
        )
        self.fail_buf *= fallen
        self.fail_buf += fallen

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = (
            self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt
        ) | self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        recovered_rate = self.has_stood[env_ids].float().mean()
        super().reset_idx(env_ids)
        self.extras["episode"]["recovered_rate"] = recovered_rate
        self.has_stood[env_ids] = False
        self.standing_time[env_ids] = 0.0

    def _reward_recovered(self):
        return (self.standing_time / self.cfg.standup.success_duration_s).clip(
            0.0, 1.0
        )

    def _reward_base_height(self):
        initial_height = self.cfg.standup.initial_base_height
        return (
            (self.base_height - initial_height)
            / (self.commands[:, 2] - initial_height)
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
