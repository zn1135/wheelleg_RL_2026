from wheel_legged_gym.envs.chuanliantui.chuanliantui_config import (
    ChuanliantuiCfg,
    ChuanliantuiCfgPPO,
)


class ChuanliantuiStandupCfg(ChuanliantuiCfg):
    class terrain(ChuanliantuiCfg.terrain):
        static_friction = 0.4
        dynamic_friction = 0.4

    class domain_rand(ChuanliantuiCfg.domain_rand):
        # 平面摩擦 0.4，PhysX 平均合成；轮地接触分别覆盖 0.20–0.30 和 0.45–0.55。
        friction_range = [0.0, 0.7]
        friction_ranges = [[0.0, 0.2], [0.5, 0.7]]
        # 水平恒力脉冲，作用于 base_link 质心；首次轮接地后才开始计时。
        push_robots = True
        standup_push_unstable_force_range = [1.0, 5.0]  # [N] 未站稳（含受扰失稳）
        standup_push_force_range = [5.0, 15.0]  # [N] 当前连续站稳后
        standup_push_duration_s = 0.10
        standup_push_interval_s_range = [3.0, 5.0]  # 两次脉冲之间的无外力空档

    class commands(ChuanliantuiCfg.commands):
        curriculum = False

        class ranges(ChuanliantuiCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.22, 0.22]

    class standup_curriculum:
        # 每满一整个全局回合窗口评估一次；达标后在该训练进程内永久切换。
        unlock_recovered_rate = 0.30
        unlock_window_episodes = 4096
        pre_unlock_target_height = 0.30
        post_unlock_target_height = 0.22
        pre_unlock_success_height = 0.28
        post_unlock_success_height = 0.20
        pre_unlock_orientation_scale = -1.0
        post_unlock_orientation_scale = -10.0

    class init_state(ChuanliantuiCfg.init_state):
        # 后摆姿态的碰撞网格贴近地面，避免初始穿地。
        pos = [0.0, 0.0, 0.15]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

    class standup:
        fall_start_height = 0.15
        initial_base_height = 0.15
        target_base_height = 0.22
        # 后摆初态；顺序为 [rf0, rf1, rfwheel, lf0, lf1, lfwheel]。
        initial_dof_pos = [-1.566, 0.0, 0.0, 1.566, 0.0, 0.0]
        # 解锁后的 0.22 m 命令下留 2 cm 裕量；解锁前使用课程的 0.28 m 门槛。
        success_height = 0.20
        success_projected_gravity_z = -0.90
        # 高度、直立和 base_link 离地必须同时连续满足，才记为已站稳。
        success_requires_base_contact_free = True
        success_base_contact_force_threshold = 0.1
        success_duration_s = 0.5
        wheels_airborne_timeout_s = 0.2  # 双轮连续同时无有效支撑达到该时间即判死。
        wheel_contact_force_threshold = 1.0  # [N] 当前平地任务：世界 z 向接触力 > 1 N 视为接地。

    class rewards(ChuanliantuiCfg.rewards):
        # 0.22 m 精确站高与膝关节余量续训；保持初始后摆姿态和起立课程。
        tracking_sigma = 0.25  # 由 0.5 收紧，线速度/偏航速度及 enhance 项共用。
        # 高度误差平方的指数分母 [m²]；误差 sqrt(0.01)=10 cm 时奖励为 e^-1。
        height_reward_sigma = 0.01
        # 解锁后误差 5 cm 时降为 e^-1；解锁前保留上面的宽容差帮助起立。
        height_reward_sigma_post_unlock = 0.0025
        # 当前模型留 0.05 rad 膝余量、pitch≈0 时最低站高约 0.218 m，
        # 因此命令使用 0.22 m。仅作为奖励安全区，不修改物理限位或裁剪 PD 目标。
        knee_limit_margin_rad = 0.05
        # base_link 满接触力时保留的高度奖励比例；离地时始终为满额。
        height_reward_contact_factor = 0.2
        # [N] base_link 奖励门控从离地到满接触的连续过渡区间。
        base_link_reward_force_scale = 5.0
        # 两条虚拟腿摆角平方和的指数分母 [rad²]；合成误差 sqrt(0.3)=0.548 rad 时为 e^-1。
        leg_angle_reward_sigma = 0.3
        recovered_reward_duration_s = 0.5  # [s] 满额稳站奖励与成功持续时间一致。

        class scales(ChuanliantuiCfg.rewards.scales):
            tracking_lin_vel = 0.2
            tracking_lin_vel_enhance = 0.2
            tracking_ang_vel = 0.2
            tracking_ang_vel_enhance = 0.2

            # 最大值不超过 clip_single_reward=1，避免目标附近被单项裁平。
            base_height = 1.0
            orientation = -1.0
            nominal_state = -3.0
            # 关闭绝对腿倾角奖励；仅保留 nominal_state 的左右腿角差约束。
            leg_angle = 0.0
            dof_pos_limits = -1.0
            knee_limit_margin = -0.3
            recovered = 1.0
            # base_link 接触地面时每个策略步扣分；首次轮触地前不进入 PPO 样本。
            base_link_contact = -0.3
            # base_link 离地时提供直接正反馈，协助策略获得高度奖励的前提条件。
            base_link_airborne = 0.2
            # 双轮同时无有效支撑时每步扣分；双轮离地 0.2 s 判死仍保留。
            wheels_airborne = -0.3
            stand_still = 0.0
            # 不再通过 collision 直接惩罚 base_link 碰地。
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
        # TensorBoard 仍记录全部项；终端只显示起立课程的关键反馈。
        terminal_episode_keys = [
            "standup_curriculum_unlocked",
            "standup_curriculum_target_height",
            "standup_curriculum_recent_recovered_rate",
            "recovered_rate",
            "rew_base_height",
            "rew_orientation",
            "rew_base_link_contact",
            "rew_base_link_airborne",
            "rew_wheels_airborne",
            "rew_recovered",
            "rew_nominal_state",
            "rew_dof_pos_limits",
            "rew_knee_limit_margin",
        ]
        terminal_episode_labels = {
            "standup_curriculum_unlocked": "课程已解锁 (0/1):",
            "standup_curriculum_target_height": "当前目标高度 [m]:",
            "standup_curriculum_recent_recovered_rate": "最近窗口恢复率:",
            "recovered_rate": "本批恢复率:",
            "rew_base_height": "高度奖励:",
            "rew_orientation": "机身姿态项:",
            "rew_base_link_contact": "base_link 接地项:",
            "rew_base_link_airborne": "base_link 离地奖励:",
            "rew_wheels_airborne": "双轮离地项:",
            "rew_recovered": "持续稳站奖励:",
            "rew_nominal_state": "左右腿对称项:",
            "rew_dof_pos_limits": "关节限位项:",
            "rew_knee_limit_margin": "站立膝余量项:",
        }
