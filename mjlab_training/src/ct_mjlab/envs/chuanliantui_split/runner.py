"""Strict role/interface/config contracts and complete independent optimizer resumes."""

import torch
from mjlab.utils.torch import configure_torch_backends
from ..chuanliantui.runner import DirectMotorOnPolicyRunner
from ..chuanliantui.pd_action import CONTROL_CONTRACT
from .observations import OBSERVATION_CONTRACT
from .switch import SWITCH_CONTRACT
from .network import WALK_ACTOR_STD_MIN
from .mdp import walk_speed_config, validate_walk_speed_state
from .action import ACTION_FILTER_CONTRACT, LEG_ACTION_ALPHA
from .rewards import GETUP_WEIGHTS, WALK_WEIGHTS, GETUP_REWARD_RECIPE, WALK_REWARD_RECIPE

CONFIG_FIELDS = ('ground_start_probability', 'pre_unlock_target_height', 'post_unlock_target_height',
    'pre_unlock_success_height', 'post_unlock_success_height', 'post_stand_tracking', 'speed_curriculum',
    'tracking_weight_while_active', 'tracking_tight_term', 'tracking_tight_kernel_denominator',
    'tracking_tight_weight', 'tracking_acceleration_range', 'tracking_interval_range', 'speed_limits',
    'speed_error_threshold', 'speed_promotion_segments', 'speed_promotion_fraction', 'yaw_control',
    'yaw_rate_limit', 'yaw_acceleration', 'heading_gain', 'position_hold', 'position_gain',
    'position_velocity_limit', 'position_tolerance', 'position_lock_speed', 'position_lock_duration')


def contracts(cfg):
    command_config = {key:getattr(cfg, key) for key in CONFIG_FIELDS}
    if cfg.role == 'walk':
        command_config['height_range'] = cfg.height_range
        command_config.pop('speed_limits')  # 继承字段只服务旧任务，walk契约不保留单档位表。
        command_config.update(walk_speed_config(cfg))
        command_config['moving_episode_probability'] = cfg.moving_episode_probability
    else:
        command_config.update(unlock_recovered_rate=cfg.unlock_recovered_rate,
                              unlock_window_episodes=cfg.unlock_window_episodes)
    result = dict(observation_contract=OBSERVATION_CONTRACT, control_contract=CONTROL_CONTRACT,
        experiment_profile_contract=f'chuanliantui-{cfg.role}-physical25-v{8 if cfg.role == "walk" else 2}', policy_role=cfg.role,
        switch_contract=SWITCH_CONTRACT, physics_dt_s=.002, policy_dt_s=.01,
        reward_recipe=WALK_REWARD_RECIPE if cfg.role == 'walk' else GETUP_REWARD_RECIPE,
        reward_weights=dict(GETUP_WEIGHTS if cfg.role == 'getup' else WALK_WEIGHTS),
        command_config=command_config)
    if cfg.role == 'walk':
        result['command_contract'] = 'split-walk-episode-height-stop-straight-spin-moving75-walk-ready-armed1s-down100ms-dirsplit-fwd30-40-bwd100-err10-v6'
        result['exploration'] = dict(actor_std_min=WALK_ACTOR_STD_MIN)
    else:
        result['curriculum_version'] = 'split-getup-height30to22-recovered30pct-window4096-v1'
        result['termination_contract'] = 'split-getup-post-contact-inverted-airborne200ms-v1'
        result['action_filter_contract'] = ACTION_FILTER_CONTRACT
        result['action_filter'] = dict(alpha=LEG_ACTION_ALPHA, action_ids=(0,1,3,4),
            policy_dt_s=.01, reset_during_ground_hold=True, observation_action='raw')
    return result


def validate(infos, cfg):
    for key, value in contracts(cfg).items():
        if infos.get(key) != value:
            raise ValueError(f'Incompatible split {key}: {infos.get(key)!r}, expected {value!r}')
    curriculum = infos.get('standup_curriculum') or {}
    if cfg.role == 'walk' and curriculum.get('unlocked') is not True:
        raise ValueError('Walking does not use the height unlock curriculum')
    if cfg.role == 'getup' and not isinstance(curriculum.get('unlocked'), bool):
        raise ValueError('Get-up checkpoint must save its height curriculum stage')
    nested = dict(
        speed_curriculum={key:getattr(cfg,key) for key in ('tracking_acceleration_range',
            'tracking_interval_range','speed_limits','speed_error_threshold','speed_promotion_segments',
            'speed_promotion_fraction')},
        yaw_control={key:getattr(cfg,key) for key in ('yaw_rate_limit','yaw_acceleration','heading_gain')},
        position_hold={key:getattr(cfg,key) for key in ('position_gain','position_velocity_limit',
            'position_tolerance','position_lock_speed','position_lock_duration')})
    if cfg.role == 'walk':
        nested['speed_curriculum'] = walk_speed_config(cfg)
    for key, expected in nested.items():
        if getattr(cfg,key):
            actual = curriculum.get(key)
            if key == 'speed_curriculum':
                actual = (actual or {}).get('config')
            if actual != expected:
                raise ValueError(f'Split nested {key} configuration differs from checkpoint')
            if key == 'speed_curriculum':
                if cfg.role == 'walk':
                    validate_walk_speed_state(curriculum[key], cfg)
                else:
                    stage = curriculum[key].get('stage')
                    if not isinstance(stage,int) or not 0 <= stage < len(cfg.speed_limits):
                        raise ValueError('Invalid split speed curriculum stage')
        elif key in curriculum:
            raise ValueError(f'Unexpected split {key} in {cfg.role} checkpoint')


class SplitOnPolicyRunner(DirectMotorOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        configure_torch_backends(allow_tf32=False)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def save(self, path, infos=None):
        state = self.env.unwrapped.command_manager.get_term('standup')
        super().save(path, {**(infos or {}), **contracts(state.cfg),
            'standup_curriculum':state.checkpoint_state()})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state = self.env.unwrapped.command_manager.get_term('standup')
        validate(saved.get('infos') or {}, state.cfg)
        curriculum = saved['infos']['standup_curriculum']
        if state.cfg.speed_curriculum and curriculum['speed_curriculum']['config'] != state.speed_config():
            raise ValueError('Split speed curriculum differs from checkpoint')
        if state.cfg.yaw_control and curriculum.get('yaw_control') != state.yaw_config():
            raise ValueError('Split yaw curriculum differs from checkpoint')
        if state.cfg.position_hold and curriculum.get('position_hold') != state.position_config():
            raise ValueError('Split position curriculum differs from checkpoint')
        restored = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        state.load_checkpoint_state(curriculum)
        self.env.unwrapped.observation_manager.reset()
        self.env.unwrapped.obs_buf = self.env.unwrapped.observation_manager.compute(update_history=True)
        if load_cfg is None or load_cfg.get('iteration', False):
            self.current_learning_iteration += 1
        return restored
