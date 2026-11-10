"""复旦行走任务配置：修改这里的开关和参数后再运行。"""


class FudanWalkCfg:
    class inference:
        # GPU评估未指定checkpoint时，使用用户选定的模型和图外停车参考。
        checkpoint = 'logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt'
        checkpoint_sha256 = '39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9'
        parking_pitch_deg = 2.

    class sim2sim:
        # 仅GPU sim2sim生效；上下限相同表示固定延迟，不改训练侧domain_rand。
        randomize_action_delay = True
        delay_ms_range = [10., 10.]
        randomize_sensor_delay = True
        sensor_delay_ms_range = [4., 4.]

    class parking_integral:
        # 独立停车积分试验；训练、普通评估和导出不自动接入。
        enabled = False
        gain = 1.  # s^-1；积分单位m，输出速度补偿m/s。
        integral_limit = .1
        compensation_limit = .1
        policy_dt = .01
        contact_dt = .002
        contact_window_s = .02
        gate_duration_s = .5
        angle_limit_deg = 10.
        wheel_contact_min_N = 1.
        body_contact_max_N = 1.

    class commands:
        velocity_initial = 2.
        velocity_max = 2.5
        velocity_increment = .1
        yaw_range = [-2., 2.]
        height_range = [.22, .30]
        resampling_steps = 500
        curriculum_threshold = .7
        curriculum_yaw_factor = .8
        # 按完整回合分配：10%原地保持、10%行驶后停车，其余沿用原采样。
        standstill_env_ratio = .1
        stop_env_ratio = .1
        stop_drive_time_s = [3.,5.]
        stop_speed_range = [.25,1.]  # m/s；前进/后退各半，之后保持零速至复位。
        parking_heights = [.22,.26,.30]  # 停车回合按三个高度均匀抽取。

    class domain_rand:
        kp_range = [.85, 1.15]
        kd_range = [.85, 1.15]
        motor_strength_range = [.85, 1.15]
        nominal_offset_range = [-.05, .05]
        added_mass_range = [-1.5, 3.]
        mass_scale_range = [.85, 1.15]
        com_range = [-.03, .03]
        friction_range = [.4, 1.6]
        restitution_range = [.4, 1.]
        friction_buckets = 64
        # 下一阶段围绕10ms扩大范围；每环境初始化抽取，非逐拍抖动。
        randomize_action_delay = True
        delay_ms_range = [6., 14.]

        # 板端接收后数据龄约1.3ms，按物理步长近似为2ms。
        randomize_sensor_delay = True
        sensor_delay_ms_range = [2., 2.]

        # 与复旦配置同名：主动推扰开关、间隔和冲量速度尺度。
        push_robots = True
        push_interval_s = 7.
        max_push_vel_xy = .3

        # 机身坐标系各轴每回合重抽，作用于机身质心并持续整回合。
        external_force_range = [-10., 10.]  # N
        external_torque_range = [-10., 10.]  # N·m

    class env:
        num_envs = 512
        play_num_envs = 20
        episode_length_s = 20.
        fail_ticks = 0  # 沿用倒地角度判据，倒地即失败重置，避免持续外矩下等待时发散。
        terminal_gravity_z = -.1
        reset_velocity_range = [-.5, .5]

    class sim:
        dt = .002
        fitted_leg_load = False  # Roll back to the original linear-spring plant.

    class control:
        decimation = 5
        leg_stiffness = 60.
        leg_damping = 2.
        leg_position_scale = .5
        wheel_stiffness = .1
        wheel_damping = 0.
        wheel_velocity_scale = 10.
        max_wheel_velocity = 150.
        nominal_leg_positions = (-.6047205544565015, .6083241726212976, .6047205544560037, -.6083241726229305)

    class normalization:
        clip = 100.
        angular_velocity = .25
        command = [2., .25, 5.]
        joint_velocity = .05
        acceleration = .0025
        torques = .05
        height_center = .5
        height_scale = 5.
        noise_scales = [.075]*6 + [0.]*3 + [.03]*4 + [.1125]*6 + [0.]*6

    class rewards:
        tracking_sigma = .25
        tracking_enhance_sigma = 2.5
        height_sigma = .001
        soft_limit_half_span = .485
        scales = dict(tracking_lin_vel=1., tracking_lin_vel_enhance=1.,
            tracking_ang_vel=1., tracking_ang_vel_enhance=1., base_height=1., nominal_state=-1.,
            lin_vel_z=-1., ang_vel_xy=-.2, orientation=-100., dof_vel=-5e-5, dof_acc=-2.5e-7,
            torques=-.0001, action_rate=-.01, action_smooth=-.01, collision=-1., dof_pos_limits=-1.)

    def __init__(self):
        from copy import deepcopy
        from types import SimpleNamespace
        for name, section in vars(type(self)).items():
            if isinstance(section, type):
                setattr(
                    self,
                    name,
                    SimpleNamespace(**{
                        key: deepcopy(value) for key,
                        value in vars(section).items() if not key.startswith('_')
                    })
                )




class FudanWalkCfgPPO:
    seed = 1
    experiment_name = 'fudan_ppo_walk'
    run_name = ''
    @property
    def clip_actions(self):
        # 仅兼容旧配置访问；实际动作限幅读取环境normalization配置。
        return FudanWalkCfg.normalization.clip

    class policy:
        encoder_hidden_dims = [128, 64]
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'
        init_noise_std = .5

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = .2
        gamma = .99
        lam = .95
        value_loss_coef = 1.
        entropy_coef = .01
        learning_rate = .001
        extra_learning_rate = .001
        max_grad_norm = 1.
        use_clipped_value_loss = True
        schedule = 'adaptive'
        desired_kl = .005

    class runner:
        num_steps_per_env = 48
        save_interval = 100
        stage_interval_s = 600.
        print_interval = 10
        minimum_iteration_s = 5.
        iteration_time_factor = 1.3

    __init__ = FudanWalkCfg.__init__


def section_dict(section):
    return {name: value for name, value in vars(section).items() if not name.startswith('_')}


def policy_dt(cfg=None):
    cfg = FudanWalkCfg() if cfg is None else cfg
    return cfg.sim.dt * cfg.control.decimation


def episode_steps(cfg=None):
    cfg = FudanWalkCfg() if cfg is None else cfg
    return round(cfg.env.episode_length_s / policy_dt(cfg))
