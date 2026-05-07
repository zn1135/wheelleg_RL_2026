from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)



class Mini_WheelLeggedCfg(LeggedRobotCfg):
#todo: 需要调整初始位置和关节角度以适应迷你轮足机器人
    class init_state(LeggedRobotCfg.init_state):
        # 直接 spawn 在自然站立高度，wheel 几乎贴地，去掉落体过程避免落地反弹
        # 之前 0.20/0.17 都会引起明显冲击 + Kp 过强的弹性反弹
        pos = [0.0, 0.0, 0.155]  # 机器人初始基座位置 x,y,z [m]
        default_joint_angles = {  # 动作为 0.0 时各关节的目标角度
            "ll_Joint0": 0.9,   # 左腿髋关节
            "ll1_Joint": -0.94, # 左腿膝关节
            "lw_Joint": 0.0,    # 左轮关节
            "rl_Joint0": -0.9,  # 右腿髋关节
            "rl_Joint1": 0.94,  # 右腿膝关节
            "rw_Joint": 0.0,    # 右轮关节
        }

    class control(LeggedRobotCfg.control):
        # 位置/速度动作缩放系数，用于将策略输出映射到控制目标
        pos_action_scale = 0.5  # 关节位置动作缩放
        vel_action_scale = 10.0   # 轮速动作缩放
        # PD 驱动参数：刚度决定"拉回目标位置"的强度
        # 注意 xwl URDF 左膝关节命名是 `ll1_Joint`（不规则），用关节全名做 key 才能保证
        # legged_robot.py:_init_buffers 的子串匹配命中所有 6 个关节（避免左膝 kp=0 失控）
        # 腿段质量很轻（大腿 0.054 / 小腿 0.081 kg），Kp=80 会引起落地猛烈反弹；
        # 取 Kp=40、Kd=3：自然频率 10Hz、ζ ≈ 1（临界阻尼），既快又不过冲
        stiffness = {
            "ll_Joint0": 40.0, "ll1_Joint": 40.0, "lw_Joint": 0.0,
            "rl_Joint0": 40.0, "rl_Joint1": 40.0, "rw_Joint": 0.0,
        }  # [N*m/rad]
        # PD 驱动参数：阻尼抑制振荡，提升控制稳定性
        damping = {
            "ll_Joint0": 3.0, "ll1_Joint": 3.0, "lw_Joint": 0.5,
            "rl_Joint0": 3.0, "rl_Joint1": 3.0, "rw_Joint": 0.5,
        }  # [N*m*s/rad]


    class asset(LeggedRobotCfg.asset):
        # 机器人模型与基础几何参数
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/xwl/urdf/xwl.urdf"
        name = "Mini_WheelLegged"
        offset = 0.00  # 机体几何偏置参数
        l1 = 0.0911  # 连杆长度参数 1（ll_0 -> ll1）
        l2 = 0.1531  # 连杆长度参数 2（ll1 -> lw）
        penalize_contacts_on = ["base_link"]#rl ll  # 这些部位接触会被加入惩罚
        terminate_after_contacts_on = ["base_link"]  # 这些部位发生接触后终止回合
        self_collisions = 1  # 自碰撞开关：1 关闭，0 开启（位掩码过滤）
        flip_visual_attachments = False     #影响可视化网格朝向的开关，不改动力学参数

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"  # 使用平地，先保证策略稳定学习
        curriculum = False   # 关闭地形课程：基类默认 True 会触发 update_command_curriculum 中扩大 lin_vel_x 范围的逻辑

    class env(LeggedRobotCfg.env):
        # 先降低并行环境数做稳定性验证
        num_envs =4096
        # 翻倒后较快终止（0.5s ≈ 25 仿真步），既不让机器人在地上打滚太久，
        # 也给 policy 一点恢复机会；之前 0.1s 太激进，policy 学不出动作
        fail_to_terminal_time_s = 0.5

    class commands(LeggedRobotCfg.commands):
        # 命令幅度从小起步，让策略先学会站立再学跟踪
        curriculum = True
        resampling_time = 5.0
        heading_command = False  # 关闭 heading 模式，避免航向误差自动塞 yaw 速度命令

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]     # 暂时关闭线速度跟踪命令，先让 policy 专心学站立
            ang_vel_yaw = [-0.5, 0.5]  # yaw 命令大幅缩小，原 ±3.14 是翻倒主因之一
            height = [0.18, 0.18]      # 高度先固定到目标值，去掉一个变量
            heading = [0.0, 0.0]

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            ang_vel_xy = -0.15   # 比原 -0.05 严但不至于扼杀必要的 roll/pitch 微调
            termination = -2.0   # 翻倒一次性大额惩罚，替代连续累积负奖励的方式
            alive = 3.0          # 上一轮 0.5 太弱，alive 仅 +0.045 / 负奖励 -0.40，每 ep 净亏 0.31；加到 3.0 让活着真正赚到
            dof_acc = -5e-8      # 从 -2.5e-7 减到 -5e-8（5 倍）：之前过重，policy 不敢动腿
            lin_vel_z = -1.0     # 从 -2.0 减到 -1.0：学习阶段允许小幅 z 方向运动
            tracking_lin_vel_enhance = 0  # 关闭：函数最大值为 0、其他时候为负，会持续扣分干扰学习

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 20.0  # 观测裁剪，防止异常值输入网络
        clip_actions = 8.0        # 动作裁剪：从 20 收紧到 8，比 5 更宽松一点，给 policy 探索空间

    class noise(LeggedRobotCfg.noise):
        add_noise = False  # 先关闭观测噪声，便于排查不稳定来源

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False         # 摩擦随机化
        randomize_restitution = False      # 弹性系数随机化
        randomize_base_mass = False        # 机体质量随机化
        randomize_inertia = False          # 惯量随机化
        randomize_base_com = False         # 质心随机化
        push_robots = False                # 外力推扰
        randomize_Kp = False               # Kp 随机化
        randomize_Kd = False               # Kd 随机化
        randomize_motor_torque = False     # 电机力矩随机化
        randomize_default_dof_pos = False  # 默认关节角随机化
        randomize_action_delay = False     # 动作延迟随机化



class Mini_WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.05  # 初始策略探索噪声（小一点防 spawn 时狂抖）

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 3.0e-4  # 学习率
        entropy_coef = 0.005    # 从 0.001 调回 0.005：上轮 noise_std 塌到 0.01，policy 不再探索；恢复一些熵奖励

    class runner(LeggedRobotCfgPPO.runner):
        # 日志实验名
        experiment_name = "mini_wheel_legged"
        max_iterations = 3000

