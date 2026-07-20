# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig






class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # 环境数量
        num_observations = 27  # 单个环境的观测维度
        num_privileged_obs = (
            num_observations + 7 * 11 + 3 + 6 * 5 + 3 + 3
        )  # 特权观测维度；若不为 None，step() 会返回 priviledge_obs_buf（用于非对称训练的 critic 观测），否则返回 None
        obs_history_length = 5  # 堆叠的历史观测帧数
        obs_history_dec = 1  # 历史观测采样间隔
        num_actions = 6  # 动作维度
        env_spacing = 3.0  # 环境间距（高度图/网格地形下不使用）
        send_timeouts = True  # 是否向算法发送超时信息
        episode_length_s = 20  # 每个回合的时长（秒）
        dof_vel_use_pos_diff = True  # 是否使用位置差分近似关节速度
        fail_to_terminal_time_s = 1  # 失败后终止的延迟时间（秒）

    class terrain:
        mesh_type = "trimesh"  # 地形类型："heightfield" / "none" / "plane" / "heightfield" / "trimesh"
        horizontal_scale = 0.1  # 水平分辨率 [m]
        vertical_scale = 0.005  # 垂直分辨率 [m]
        border_size = 25  # 边界大小 [m]
        curriculum = True  # 是否启用课程学习
        static_friction = 0.5  # 静摩擦系数
        dynamic_friction = 0.5  # 动摩擦系数
        restitution = 0.5  # 恢复系数
        # 仅适用于粗糙地形：
        measure_heights = True  # 是否测量高度
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1m x 1.6m 的矩形采样点（不含中心线）
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = False  # 是否选择单一地形类型，并传入全部参数
        terrain_kwargs = None  # 选定地形时使用的参数字典
        max_init_terrain_level = 5  # 初始课程等级
        terrain_length = 8.0  # 地形长度
        terrain_width = 8.0  # 地形宽度
        num_rows = 10  # 地形行数（等级数）
        num_cols = 20  # 地形列数（类型数）
        # 地形类型：[平地、平缓坡面、粗糙坡面、台阶下、台阶上、离散地形]
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 仅适用于 trimesh：
        slope_treshold = (
            0.75  # 超过该阈值的斜坡会被修正为垂直表面
        )

    class commands:
        curriculum = True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3  # 默认命令维度：lin_vel_x, lin_vel_y, ang_vel_yaw（heading 模式下由航向误差重算 yaw 角速度）
        resampling_time = 5.0  # 命令重采样周期 [s]
        heading_command = True  # 为 True 时，根据航向误差计算角速度命令

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # 最小/最大线速度 [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # 最小/最大偏航角速度 [rad/s]
            height = [0.1, 0.25]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.0]  # 初始位置 x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始姿态四元数 x,y,z,w
        lin_vel = [0.0, 0.0, 0.0]  # 初始线速度 x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # 初始角速度 x,y,z [rad/s]
        default_joint_angles = {  # action = 0.0 时的目标关节角
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

    class control:
        control_type = "P"  # 控制模式：P 位置、V 速度、T 力矩
        # PD 驱动参数
        stiffness = {"joint_a": 10.0, "joint_b": 15.0}  # [N*m/rad]
        damping = {"joint_a": 1.0, "joint_b": 1.5}  # [N*m*s/rad]
        # 动作缩放：目标角 = action_scale * action + default_angle
        action_scale = 0.5
        # 降采样比：每个策略步内执行的控制更新次数（按仿真 dt）
        decimation = 2

    class asset:
        file = ""
        name = "legged_robot"  # actor 名称
        foot_name = "None"  # 足端刚体名称，用于索引刚体状态与接触力张量
        offset = 0
        l1 = 0
        l2 = 0
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # 合并由固定关节连接的刚体；可在 URDF 中用 dont_collapse="true" 保留
        fix_base_link = False  # 是否固定机器人基座
        default_dof_drive_mode = 3  # 关节驱动模式，见 GymDofDriveModeFlags（0 无，1 位置，2 速度，3 力矩）
        self_collisions = 0  # 自碰撞开关（位掩码）：1 关闭，0 开启
        replace_cylinder_with_capsule = True  # 用胶囊体替换碰撞圆柱，通常更快且更稳定
        flip_visual_attachments = (
            True  # 部分 .obj 网格需要从 y-up 翻转到 z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = True
        randomize_Kd_range = [0.9, 1.1]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 10]

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0

            base_height = 1.0
            nominal_state = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -10.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.0001
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1.0

        only_positive_rewards = False  # 为 True 时将负总奖励裁剪为 0（可减少因早停导致的问题）
        clip_single_reward = 1
        tracking_sigma = 0.25  # 跟踪奖励形式：exp(-error^2 / sigma)
        soft_dof_pos_limit = (
            0.97  # URDF 关节位置上限比例，超过该阈值开始惩罚
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.18
        max_contact_force = 100.0  # 超过该接触力阈值会被惩罚

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0  # 噪声总强度缩放系数

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # 可视化相机参数
    class viewer:
        ref_env = 0
        pos = [0, -2, 1]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 求解器类型：0 pgs，1 tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 可支持约 8000+ 环境时的接触对上限
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 接触采集模式：0 从不，1 最后子步，2 所有子步（默认 2）
            )


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"  # 激活函数：elu/relu/selu/crelu/lrelu/tanh/sigmoid

        # 仅用于 ActorCriticSequence
        num_encoder_obs = (
            LeggedRobotCfg.env.obs_history_length * LeggedRobotCfg.env.num_observations
        )
        latent_dim = 3  # 至少为 3，便于估计机体线速度
        encoder_hidden_dims = [128, 64]

    class algorithm:
        # 训练超参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini-batch 大小 = num_envs * nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # 学习率策略：adaptive 或 fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.005
        max_grad_norm = 1.0

        extra_learning_rate = 1e-3

    class runner:
        policy_class_name = (
            "ActorCriticSequence"  # 策略类：ActorCritic 或 ActorCriticSequence
        )
        algorithm_class_name = "PPO"
        num_steps_per_env = 48  # 每次迭代每个环境采样步数
        max_iterations = 5000  # 策略更新总迭代数

        # 日志与保存
        save_interval = 100  # 每隔多少迭代检查并保存一次模型
        experiment_name = "test"
        run_name = ""
        # 加载与断点续训
        resume = False
        load_run = -1  # -1 表示加载最近一次运行
        checkpoint = -1  # -1 表示加载最新保存模型
        resume_path = None  # 由 load_run 和 checkpoint 解析得到
