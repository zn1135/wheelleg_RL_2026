"""复旦 Jump 平地训练；横条只用于冻结策略评估。"""
import json
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO

TASK = 'fudan_ppo_jump'
RECIPE = 'fudan-jump-8204e853-physical-v1'
LANDING_RECIPE = 'fudan-jump-landing-v2'
STOP_SINK_RECIPE = 'fudan-jump-stop-sink-v3'
PRELOAD_RECIPE = 'fudan-jump-preload-v4'
PEAK_RECIPE = 'fudan-jump-wheel-peak-v5'
ADDITIVE_PEAK_RECIPE = 'fudan-jump-wheel-peak-additive-v6'
LANDING_RECIPES = (LANDING_RECIPE, STOP_SINK_RECIPE, PRELOAD_RECIPE, PEAK_RECIPE, ADDITIVE_PEAK_RECIPE)
LANDING = dict(air_ticks=3, buffer_ticks=20, max_jump_ticks=140,
               reach_m=.30, compressed_m=.22, length_sigma_m=.045,
               wheel_level_sigma_m=.035, pitch_sigma_rad=.18)
TUCK, EXTEND, FLIGHT_HEIGHT = .16, .31, .65
BAR_X, BAR_DEPTH, BAR_HALF_WIDTH = 4., .05, 1.
HEIGHTS = (.05, .10)
TRIGGER, JUMP_TICKS = .9, 48


def jump_config(play=False, *, landing=False, stop_sink=False, preload=False, peak=False, peak_additive=False):
    cfg = FudanWalkCfg()
    preload = preload or peak or peak_additive
    cfg.sim.dt = .001
    cfg.control.decimation = 10
    cfg.control.pd_decimation = 5
    cfg.control.leg_stiffness = 20.
    cfg.control.leg_damping = 1.
    cfg.commands.velocity_initial = cfg.commands.velocity_max = 2.1
    cfg.commands.yaw_range = [-2., 2.]
    cfg.commands.resampling_steps = 2000
    cfg.commands.standstill_env_ratio = cfg.commands.stop_env_ratio = 0.
    cfg.normalization.command = [3., .25, 5.]
    cfg.normalization.noise_scales = [.05]*6 + [0.]*3 + [.02]*4 + [.075]*6 + [0.]*6
    cfg.domain_rand.randomize_action_delay = cfg.domain_rand.randomize_sensor_delay = False
    cfg.domain_rand.kp_range = cfg.domain_rand.kd_range = [.9, 1.1]
    cfg.domain_rand.motor_strength_range = [.9, 1.1]
    cfg.domain_rand.added_mass_range = [-2., 3.]
    cfg.domain_rand.mass_scale_range = [.8, 1.2]
    cfg.domain_rand.com_range = [-.05, .05]
    cfg.domain_rand.ground_friction_range = None
    cfg.domain_rand.friction_range = [.1, 2.]
    cfg.domain_rand.restitution_range = [.5, 1.]
    # 本机实体适配首版关闭外推，避免外力制造假起跳。
    cfg.domain_rand.push_robots = False
    cfg.rewards.scales = dict(tracking_lin_vel=1., tracking_lin_vel_enhance=1.,
        tracking_ang_vel=1., flight=.15, encourage_jump=1., base_height_flight=6.,
        leg_tuck=1.7, takeoff_extend=.5, line_z=6., pen_theta_no0=-2.,
        action_rate=-.04, torques=-.00005, orientation=-25., ang_vel_xy=-.1,
        nominal_state=-1., collision=-1.)
    if landing or stop_sink or preload:
        cfg.rewards.landing = LANDING.copy()
        cfg.rewards.scales.update(landing_reach=2.5, landing_level=1.5,
            landing_upright=1.5, landing_compression=1.5, landing_support=2.5,
            landing_body=-2.5, landing_bounce=-1.5)
    if stop_sink or preload:
        cfg.rewards.landing.update(buffer_ticks=35, compressed_m=.25,
            stop_sink=dict(min_clearance_m=.06, clearance_width_m=.06, velocity_scale_m_s=.8))
        cfg.rewards.scales.update(landing_clearance=-2.5, landing_sink=-2.5, landing_support_loss=-1.5)
    if preload:
        cfg.rewards.landing['preload'] = dict(lead_s=.10, height_margin_m=.02,
            nominal_diff_rad=1.279, lead_diff_rad=.30, max_diff_rad=1.42,
            sigma_diff_rad=.25, support_diff_Nm=10., length_floor_m=.29)
        cfg.rewards.scales.update(landing_motor_target=2.5, landing_drive=2.5,
            landing_first_drive=2.5, landing_early_length=2.)
    if peak and not peak_additive:
        del cfg.rewards.scales['base_height_flight']
    if peak or peak_additive:
        cfg.rewards.scales['wheel_peak_gain'] = 64.
    return cfg


def jump_ppo():
    cfg = FudanWalkCfgPPO()
    cfg.experiment_name = TASK
    return cfg

def recipe_record(cfg):
    sections = ('sim', 'control', 'normalization', 'commands', 'domain_rand', 'env', 'rewards')
    record = {name: vars(getattr(cfg, name)) for name in sections}
    landing = getattr(cfg.rewards,'landing',None)
    version = RECIPE
    if landing:
        version = STOP_SINK_RECIPE if 'stop_sink' in landing else LANDING_RECIPE
    if landing and 'preload' in landing:
        version = PRELOAD_RECIPE
    if 'wheel_peak_gain' in cfg.rewards.scales:
        version = ADDITIVE_PEAK_RECIPE if 'base_height_flight' in cfg.rewards.scales else PEAK_RECIPE
    record.update(version=version, tuck_m=TUCK, extend_m=EXTEND, flight_height_m=FLIGHT_HEIGHT,
                  clip_single_reward=2.5)
    # 日志和检查点都用同一份可序列化配置，比对时列表与元组等价。
    return json.loads(json.dumps(record))
