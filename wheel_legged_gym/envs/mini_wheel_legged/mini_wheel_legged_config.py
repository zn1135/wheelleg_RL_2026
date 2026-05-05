from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)



class Mini_WheelLeggedCfg(LeggedRobotCfg):
#todo: 需要调整初始位置和关节角度以适应迷你轮足机器人
    class init_state(LeggedRobotCfg.init_state):
        # 降低初始落体冲击，避免起步瞬间接触力过大导致发散
        pos = [0.0, 0.0, 0.03]  # 机器人初始基座位置 x,y,z [m]
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
        # PD 驱动参数：刚度决定“拉回目标位置”的强度
        stiffness = {"Joint0": 2.0, "Joint1": 2.0, "w_Joint": 0}  # [N*m/rad]
        # PD 驱动参数：阻尼抑制振荡，提升控制稳定性
        damping = {"Joint0": 1.0, "Joint1": 1.0, "w_Joint": 0.5}  # [N*m*s/rad]


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

    class env(LeggedRobotCfg.env):
        # 先降低并行环境数做稳定性验证
        num_envs =4096

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 20.0  # 观测裁剪，防止异常值输入网络
        clip_actions = 20.0       # 动作裁剪，防止控制指令过大

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
        init_noise_std = 0.2  # 初始探索噪声标准差

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 3.0e-4  # 学习率
        entropy_coef = 0.005    # 熵奖励系数，平衡探索/收敛

    class runner(LeggedRobotCfgPPO.runner):
        # 日志实验名
        experiment_name = "mini_wheel_legged"
        max_iterations = 500        

