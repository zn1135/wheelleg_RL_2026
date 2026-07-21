from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)



class Mini_WheelLeggedCfg(LeggedRobotCfg):
#todo: 需要调整初始位置和关节角度以适应迷你轮足机器人
    class env(LeggedRobotCfg.env):
        # 倾角失败阈值（pg_z 超过它记为失败，持续 fail_to_terminal_time_s 秒终止）。
        # 基类 -0.1(84°) 过于宽松，策略曾学会 ~40° 前倾斜靠停机；-0.85 约 32°。
        fail_tilt_pg_z = -0.85

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            # 放宽膝限位(±2.4)后腿长可收到约0.17m，站高可行范围约0.15~0.44m
            height = [0.20, 0.40]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]  # 机器人初始基座位置 x,y,z [m]
        default_joint_angles = {  # 动作为 0.0 时各关节的目标角度
            # 零位=腿垂直伸直；髋后摆+膝反折使轮心位于髋正下方，站高约 0.30m
            "lf0_joint": 0.9,
            "lf1_joint": -1.62,
            "l_wheel_joint": 0.0,
            "rf0_joint": -0.9,
            "rf1_joint": 1.62,
            "r_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 位置/速度动作缩放系数，用于将策略输出映射到控制目标
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD 驱动参数：刚度决定“拉回目标位置”的强度
        stiffness = {"f0": 60.0, "f1": 60.0, "wheel": 0}  # [N*m/rad]
        # PD 驱动参数：阻尼抑制振荡，提升控制稳定性
        damping = {"f0": 2.0, "f1": 2.0, "wheel": 0.5}  # [N*m*s/rad]


    class rewards(LeggedRobotCfg.rewards):
        # 放宽跟踪评分宽容度：误差0.5m/s时得分从0.37提高到0.49，梯度更平缓利于课程解锁
        tracking_sigma = 0.35
        # 单项奖励每步裁剪上限须高于 0.7*tracking_lin_vel 权重，否则速度课程永远无法解锁
        clip_single_reward = 2

        class scales(LeggedRobotCfg.rewards.scales):
            # 提高线速度跟踪权重，压过 dof_acc 等正则项的对抗
            tracking_lin_vel = 1.5

    class sim(LeggedRobotCfg.sim):
        # 第二层：把 Isaac Gym(PhysX) 的接触调"软"，向 MuJoCo 的软约束接触靠拢，缩小 sim2sim gap。
        # 这些是全局 sim 参数（一个 sim 一个值，无法逐环境随机化）；逐环境的接触刚度随机化见 domain_rand.compliance。
        class physx(LeggedRobotCfg.sim.physx):  # 继承以保留未列出的字段（solver_type/num_threads 等）
            contact_offset = 0.02  # 基类 0.01：加大接触生成距离，接触更早触发、更"软"
            rest_offset = 0.001  # 基类 0.0：留微小静止间隙，减少穿透抖动
            num_position_iterations = 8  # 基类 4：更多位置求解迭代，接触更稳、更接近隐式求解
            bounce_threshold_velocity = 0.2  # 基类 0.5：降低反弹阈值，硬地行为更贴近 MuJoCo

    class asset(LeggedRobotCfg.asset):        # 机器人模型与基础几何参数
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/imcawl/urdf/imcawl.urdf"
        name = "Mini_WheelLegged"
        offset = 0.00  # 机体几何偏置参数
        l1 = 0.21  # 大腿连杆长度：lf0->lf1 关节距离 [m]
        l2 = 0.25  # 小腿连杆长度：lf1->轮心距离 [m]
        penalize_contacts_on = ["lf", "rf", "base"]  # 这些部位接触会被加入惩罚
        terminate_after_contacts_on = ["base"]  # 这些部位发生接触后终止回合
        self_collisions = 1  # 自碰撞开关：1 关闭，0 开启（位掩码过滤）
        flip_visual_attachments = False     #影响可视化网格朝向的开关，不改动力学参数

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 为缩小 sim2sim gap（策略在 MuJoCo 里滑向低蹲位、不追踪速度）而加宽/收窄的域随机化。
        # 目标：让策略不依赖 Isaac Gym(PhysX) 特有的接触/摩擦动力学，从而在 MuJoCo/真机也鲁棒。
        # 仅覆盖以下 4 项，其余（push_robots/base_mass/base_com/inertia/motor_torque/action_delay）继承基类。

        # 摩擦收到更贴近部署的区间（MuJoCo 实测站立需 ~0.5），避免策略依赖极端高/低摩擦
        friction_range = [0.3, 1.5]  # 基类 [0.1, 2.0]
        # 硬地反弹小，收窄 restitution，避免适配 IG 的高反弹
        restitution_range = [0.0, 0.3]  # 基类 [0.0, 1.0]
        # 加宽 PD 增益随机化：PD 等效刚度是两引擎差异的主要体现，加宽让策略对增益不敏感（闭合 gap 的关键杠杆）
        randomize_Kp_range = [0.8, 1.2]  # 基类 [0.9, 1.1]
        randomize_Kd_range = [0.8, 1.2]  # 基类 [0.9, 1.1]
        # 加宽站立姿态扰动，避免锁死单一平衡点（对应 MuJoCo 里滑向 0.12 低位的问题）
        randomize_default_dof_pos_range = [-0.08, 0.08]  # 基类 [-0.05, 0.05]
        # 逐环境接触柔度随机化：让 4096 个环境见到分布式的接触硬度，策略学会不依赖某一种接触刚度。
        # compliance 是 PhysX per-shape 字段，越大越软/越像 MuJoCo（0=硬，PhysX 默认）。这是闭合 gap 最对症的手段。
        # 由 legged_robot._process_rigid_shape_props 用 getattr 守卫读取，其他任务无此字段则自动跳过。
        randomize_compliance = True
        compliance_range = [0.0, 0.02]


class Mini_WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # 日志实验名
        experiment_name = "mini_wheel_legged"
        max_iterations = 1000  # 策略更新总迭代数        