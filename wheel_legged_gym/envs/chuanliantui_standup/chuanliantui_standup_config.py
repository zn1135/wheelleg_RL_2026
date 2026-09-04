from wheel_legged_gym.envs.chuanliantui.chuanliantui_config import (
    ChuanliantuiCfg,
    ChuanliantuiCfgPPO,
)


class ChuanliantuiStandupCfg(ChuanliantuiCfg):
    class commands(ChuanliantuiCfg.commands):
        curriculum = False

        class ranges(ChuanliantuiCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.3276, 0.3276]

    class init_state(ChuanliantuiCfg.init_state):
        pos = [0.0, 0.0, 0.08]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

    class standup:
        initial_base_height = 0.08
        target_base_height = 0.3276
        initial_dof_pos = [11.0, 0.0, 0.0, -11.0, 0.0, 0.0]
        success_height = 0.2876
        success_projected_gravity_z = -0.85
        success_duration_s = 0.5

    class rewards(ChuanliantuiCfg.rewards):
        class scales(ChuanliantuiCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0
            tracking_ang_vel_enhance = 1.0
            base_height = 1.0
            orientation = -1.0
            recovered = 1.0
            stand_still = 0.0
            collision = 0.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.2
            dof_vel = -5e-5
            dof_acc = 0.0
            torques = -1e-4
            action_rate = -0.01
            action_smooth = -0.01


class ChuanliantuiStandupCfgPPO(ChuanliantuiCfgPPO):
    class runner(ChuanliantuiCfgPPO.runner):
        experiment_name = "chuanliantui_standup"
        max_iterations = 3000
