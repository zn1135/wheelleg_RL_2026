# chuanliantui 串联腿轮足平衡机器人配置。
#
# 模板 = 复旦星云EGA开源 plane 方案(方法论与数值出处见任务 T-20260823-01 调研),
# 机器人相关参数全部换为 chuanliantui 实测值(URDF a43140d / STL 实测):
#   - 6 DOF,DOF 字母序 [lf0, lf1, lfwheel, rf0, rf1, rfwheel],恰与基类
#     _compute_torques 的"索引 2/5 为轮、其余为腿位置环"约定兼容;
#   - 连杆 l1=0.21(髋→膝)、l2=0.25(膝→轮心),FK 零位偏置 (0.666151, 1.625993);
#   - 默认站姿 lf0=∓0.06 / lf1=±0.10(rf 取反镜像):微蹲 L0≈0.297m、摆角≈0,
#     base 站高 ≈0.324m;height 命令域覆盖其上下;
#   - 电机:腿 DM8009P(effort 40)、轮改装 M3508 减速比16.6(effort 3.9),
#     URDF limit 由 Isaac 自动读入 torque_limits。
# 首版平地(mesh_type="plane")跑通管线用;多地形课程后续再开。

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class ChuanliantuiCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 平地版 measure_heights=False 时地形采样项塌缩成 1 维,
        # 特权观测实际拼接 = 3+27+12+6+1+6+1+3+6+1+1 = 67;
        # 切多地形(measure_heights=True)时改回基类公式 143
        num_privileged_obs = 67

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"  # 首版平地;跑通后切 trimesh 多地形课程
        curriculum = False
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = True  # 指令课程照复旦 plane(逐环境独立放宽 lin_vel_x)
        num_commands = 3  # [前进速度 vx, 偏航角速度, 目标高度]
        heading_command = False  # 复旦不用航向保持外环
        resampling_time = 5.0

        class ranges(LeggedRobotCfg.commands.ranges):
            # yaw 采样域收窄到 ±2(复旦 ±15 依赖地形+课程夹持,平地起步先保守)
            ang_vel_yaw = [-2.0, 2.0]
            # 站高 0.324m 的上下约 ±10%
            height = [0.28, 0.36]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.33]  # 站高 0.324,留少量下落余量
        default_joint_angles = {  # action=0 时的目标角;微蹲、轮心位于髋正下方(theta0≈0)
            "lf0": -0.06,
            "lf1": 0.10,
            "lfwheel": 0.0,
            "rf0": 0.06,
            "rf1": -0.10,
            "rfwheel": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5  # 腿:位置增量目标 [rad]
        vel_action_scale = 10.0  # 轮:速度目标 [rad/s]
        stiffness = {"f0": 10.0, "f1": 10.0, "wheel": 0}  # 复旦 plane 口径
        damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.1}

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/chuanliantui/urdf/chuanliantui.urdf"
        name = "chuanliantui"
        foot_name = "wheel"
        offset = 0.0
        l1 = 0.21  # 髋→膝连杆长 [m]
        l2 = 0.25  # 膝→轮心连杆长 [m]
        # 二连杆 FK 零位偏置(chuanliantui.py post_physics_step 使用)
        fk_offset_hip = 0.666151
        fk_offset_knee = 1.625993
        penalize_contacts_on = ["base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1  # 关闭自碰撞
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 照复旦 plane 全套
        friction_range = [0.6, 1.4]
        restitution_range = [0.6, 1.0]
        added_mass_range = [-1.0, 2.0]
        randomize_inertia_range = [0.9, 1.1]
        rand_com_vec = [0.02, 0.02, 0.02]
        push_robots = False  # 平地首版关推搡(复旦 plane 同)
        randomize_Kp_range = [0.95, 1.05]
        randomize_Kd_range = [0.95, 1.05]
        randomize_default_dof_pos_range = [-0.03, 0.03]
        randomize_action_delay = False

    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma = 0.25
        clip_single_reward = 1.0
        only_positive_rewards = False
        base_height_target = 0.32  # 语义参考;height 奖励实际跟踪 commands[:,2]

        class scales(LeggedRobotCfg.rewards.scales):
            # 照复旦 plane;基类默认不同的项显式覆盖
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0
            tracking_ang_vel_enhance = 1.0
            base_height = 1.0
            nominal_state = -1.0  # 左右虚拟摆角差(依赖 chuanliantui 版 FK)
            lin_vel_z = -1.0
            ang_vel_xy = -0.2
            orientation = -100.0
            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -1e-4
            action_rate = -0.01
            action_smooth = -0.01
            collision = -1.0
            dof_pos_limits = -1.0

    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02  # 复旦口径


class ChuanliantuiCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "chuanliantui"
        max_iterations = 3000  # 首轮训练;效果评估后再拉长
