from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)



class Mini_WheelLeggedCfg(LeggedRobotCfg):
#todo: 需要调整初始位置和关节角度以适应迷你轮足机器人
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.50]  # 机器人初始基座位置 x,y,z [m]
        default_joint_angles = {  # 动作为 0.0 时各关节的目标角度
            "ll_Joint0": 0.9,
            "ll1_Joint": -0.94,
            "lw_Joint": 0.0,
            "rl_Joint0": -0.9,
            "rl_Joint1": 0.94,
            "rw_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 位置/速度动作缩放系数，用于将策略输出映射到控制目标
        pos_action_scale = 0.5
        vel_action_scale = 1.0
        # PD 驱动参数：刚度决定“拉回目标位置”的强度
        stiffness = {"Joint0": 5.0, "Joint1": 5.0, "w_Joint": 0}  # [N*m/rad]
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
        mesh_type = "plane"



class Mini_WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # 日志实验名
        experiment_name = "mini_wheel_legged"        

