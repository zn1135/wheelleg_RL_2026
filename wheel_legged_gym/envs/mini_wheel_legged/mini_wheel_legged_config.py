from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)



class Mini_WheelLeggedCfg(LeggedRobotCfg):
#todo: 需要调整初始位置和关节角度以适应迷你轮足机器人
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


class Mini_WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # 日志实验名
        experiment_name = "mini_wheel_legged"
        max_iterations = 1000  # 策略更新总迭代数        