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
            height = [0.32, 0.32]

    class init_state(ChuanliantuiCfg.init_state):
        # 与 ground-standup 相同：后摆姿态的碰撞网格贴近地面，避免初始穿地。
        pos = [0.0, 0.0, 0.15]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

    class standup:
        fall_start_height = 0.15
        initial_base_height = 0.15
        target_base_height = 0.32
        # 与 ground-standup 相同的后摆初态；顺序为
        # [lf0, lf1, lfwheel, rf0, rf1, rfwheel]。
        initial_dof_pos = [11.0, 0.0, 0.0, -11.0, 0.0, 0.0]
        success_height = 0.30
        success_projected_gravity_z = -0.90
        success_requires_base_contact_free = False
        success_base_contact_force_threshold = 0.1
        success_duration_s = 0.5
        wheels_airborne_timeout_s = 0.2  # 双轮连续同时无有效支撑达到该时间即判死。
        wheel_contact_force_threshold = 1.0  # [N] 当前平地任务：世界 z 向接触力 > 1 N 视为接地。

    class rewards(ChuanliantuiCfg.rewards):
        # 恢复放宽前的奖励强度，用于已学会起立策略的续训；不改成功判据或终止课程。
        tracking_sigma = 0.25  # 由 0.5 收紧，线速度/偏航速度及 enhance 项共用。
        height_reward_tolerance = 0.0  # [m] 达到命令高度才获得满额高度奖励。
        recovered_reward_duration_s = 0.5  # [s] 满额稳站奖励与成功持续时间一致。

        class scales(ChuanliantuiCfg.rewards.scales):
            tracking_lin_vel = 0.2
            tracking_lin_vel_enhance = 0.2
            tracking_ang_vel = 0.2
            tracking_ang_vel_enhance = 0.2

            base_height = 1.0
            # 9 项负权重恢复到放宽前的值，原先禁用的惩罚不额外启用。
            orientation = -10.0
            nominal_state = -3.0
            dof_pos_limits = -1.0
            recovered = 1.0
            wheels_airborne = -1.0  # 双轮同时无有效支撑时每步扣分；接地阈值与离地判死共用。
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
    class algorithm(ChuanliantuiCfgPPO.algorithm):
        # 恢复已完成 3000→6000 轮训练的更新强度；不把它视为物理 NaN 修复。
        learning_rate = 3.0e-4
        extra_learning_rate = 3.0e-4
        value_loss_coef = 0.25
        num_learning_epochs = 3

    class runner(ChuanliantuiCfgPPO.runner):
        experiment_name = "chuanliantui_standup"
        max_iterations = 3000
