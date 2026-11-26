"""单级平台候选；这里的阈值是实验设计，不是已通过的运动成绩。"""
from types import SimpleNamespace
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO

TASK = 'fudan_ppo_step'
RECIPE = 'fixed200-restart-2mps-v11'
TIMED_RECIPE = 'fixed200-timed-restart-2mps-v12'
SOURCE = 'logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01/model_final.pt'
SOURCE_SHA = 'dfb221c09648d278d5c5e5c7b78e4d729b7a7a5a56458762b7313dc50831c4f8'


def FudanStepCfg(version=RECIPE):
    # 创建独立实例，不能修改正在使用的平地类属性。
    cfg = FudanWalkCfg()
    cfg.inference.checkpoint = SOURCE
    cfg.inference.checkpoint_sha256 = SOURCE_SHA
    cfg.sim.dt = .001
    cfg.control.decimation = 10
    cfg.control.pd_decimation = 5
    cfg.control.leg_stiffness = 20.
    cfg.control.leg_damping = 1.
    cfg.env.num_envs = 4096  # 用户指定的正式训练并行数；仍先做零学习容量检查。
    cfg.env.reset_velocity_range = [0., 0.]
    cfg.domain_rand.delay_ms_range = [10., 20.]
    cfg.domain_rand.sensor_delay_ms_range = [10., 20.]
    cfg.domain_rand.ground_friction_range = [.7, .9]
    # 参数/噪声幅度缩至v5的五分之一；保留源3000的双延迟接口。
    for name, value in dict(kp_range=[.985,1.015], kd_range=[.985,1.015],
            motor_strength_range=[.985,1.015], nominal_offset_range=[-.005,.005],
            added_mass_range=[-.15,.30], mass_scale_range=[.985,1.015],
            com_range=[-.003,.003], friction_range=[.7,.9], restitution_range=[.67,.73]).items():
        setattr(cfg.domain_rand, name, value)
    cfg.normalization.noise_scales = [.0075]*6+[0.]*3+[.003]*4+[.01125]*6+[0.]*6
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.external_force_range = [0., 0.]
    cfg.domain_rand.external_torque_range = [0., 0.]
    cfg.commands.velocity_initial = 2.5
    cfg.commands.velocity_max = 2.5
    cfg.commands.velocity_increment = .1
    cfg.commands.yaw_range = [0., 0.]
    cfg.commands.stop_speed_range = [2., 2.]
    cfg.commands.height_range = [.22, .35]
    cfg.commands.standstill_env_ratio = 0.
    cfg.commands.stop_env_ratio = 0.
    cfg.rewards.height_sigma = .001
    # 南科大原版速度式奖励；控制/闭链限位仍来自本机，不复制其机器人参数。
    cfg.rewards.scales = dict(tracking_lin_vel=1., tracking_lin_vel_enhance=1.,
        tracking_ang_vel=1., tracking_ang_vel_enhance=0., base_height=1., base_height_enhance=0.,
        nominal_state=-.1, lin_vel_z=-2., ang_vel_xy=-.05, orientation=-10.,
        dof_vel=-5e-5, dof_acc=-2.5e-7, torques=-1e-4,
        action_rate=-.01, action_smooth=-.01, collision=-1., dof_pos_limits=-1.,
        reference_motion=.2, target_support=1.)
    cfg.step = SimpleNamespace(
        version=RECIPE, heights=(0., .20),
        width=1.148, depth=2.09, patch_size=(26., 4.),
        border_width=2.,
        contact_capacity=64, constraint_capacity=256,
        terrain_weights=(.2, .4, .4), start_level=1, minimum_stair_level=1, curriculum=False,
        terrain_assignment='fixed_types', reward_profile='clearlab-velocity-adapted-v1',
        reference_commit='c354431e5633eb98d9663fa3fbe444201d2de383',
        command_curriculum='fixed_forward_2mps',
        approach_distance=(0., 1.2, .6),
        side_obstacle_level=1, side_obstacle_height=.55,
        approach_speed=(2., 2.), approach_angle_deg=20., max_approach_speed=3.,
        reverse_probability=0., height_range=(.35,.35), landing_s=.1,
        crossing_symmetry_factor=0.,
        contact_threshold_N=1., contact_timeout_s=.20,
        support_weight_fraction=.1, support_timeout_s=.10,
        parking_s=2., parking_speed=.05, parking_angle_deg=5.,
        # 越障允许临时搭沿；传感器原失败标记仍保留作诊断，停车必须脱离接触。
        allow_temporary_support=True, landing_clearance_m=.15,
        reference_motion_path='artifacts/fudan_ppo_step/20261116-contact-exploration-v7-01/reference/motion.npz',
        reference_motion_sha256='4ea3cc892b1a3035e106053d194371d4942a5b53a47a46a5f1f9feb0738183c3',
        flat_teacher_path=SOURCE, flat_teacher_sha256=SOURCE_SHA, flat_preservation_weight=.1,
        flat_regular_height_range=(.22,.30), turn_height=.35, turn_height_probability=.5,
        flat_teacher_height_max=.30, flat_cruise_s=7., flat_forward_limit_m=18.,
        restart_flat_probability=.5, delayed_stair_probability=.25,
        restart_initial_wait_s=(.75, 2.),
        restart_stable_s=.20, restart_first_drive_s=2.5, restart_second_drive_s=3.,
    )
    if version == TIMED_RECIPE:
        cfg.step.version = version
        del cfg.step.restart_stable_s
        cfg.step.restart_pause_s = (1.5, 3.)
        cfg.step.flat_spawn_yaw_deg = 0.
        cfg.step.flat_boundary_timeout = True
        cfg.step.down_lane_until_edge = True
    elif version != RECIPE:
        raise ValueError(f'Unknown step recipe: {version}')
    return cfg


def FudanStepCfgPPO():
    cfg = FudanWalkCfgPPO()
    cfg.experiment_name = TASK
    return cfg


def recipe_record(cfg):
    def plain(value):
        if isinstance(value, SimpleNamespace):
            return {key: plain(val) for key, val in vars(value).items()}
        if isinstance(value, dict):
            return {key: plain(val) for key, val in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(val) for val in value]
        return value
    return {name: plain(getattr(cfg, name)) for name in
            ('step', 'env', 'sim', 'control', 'normalization', 'commands', 'domain_rand', 'rewards')}
