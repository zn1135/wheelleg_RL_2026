"""Two independent recipes and log namespaces; no CLI recipe switches."""

import re

from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from ..chuanliantui_stand_v2.env_cfg import make_stand_v2_env_cfg, make_stand_v2_ppo_cfg
from . import mdp, observations, rewards
from .action import GetupPdAction
from .network import WALK_ACTOR_STD_MIN


def command_cfg(role):
    return mdp.SplitCommandCfg(role=role, height_range=(.22, .30) if role == 'walk' else None,
        pre_unlock_target_height=.30 if role == 'getup' else .22,
        pre_unlock_success_height=.28 if role == 'getup' else .20,
        ground_start_probability=1. if role == 'getup' else 0.,
        speed_error_threshold=.10 if role == 'walk' else .30,
        post_stand_tracking=role == 'walk', speed_curriculum=role == 'walk',
        tracking_weight_while_active=role == 'walk', tracking_tight_term=role == 'walk',
        tracking_tight_weight=rewards.WALK_WEIGHTS['tracking_lin_vel_tight'],
        yaw_control=role == 'walk', position_hold=role == 'walk',
        moving_episode_probability=.75 if role == 'walk' else .5)


def make_env_cfg(role, *, play=False):
    cfg = make_stand_v2_env_cfg(play=play)
    if role == 'getup':
        cfg.actions['pd'].class_type = GetupPdAction
    else:
        pose = mdp.resets.standing_pose(mdp.WALK_INITIAL_HEIGHT)
        initial = cfg.scene.entities['robot'].init_state
        initial.pos = (0., 0., pose['height'] + mdp.resets.RESET_CLEARANCE_M)
        initial.joint_pos = {f'^{re.escape(name)}$':value for name,value in pose['joint_positions'].items()}
        cfg.events['reset_pose'].func = mdp.reset_walk_pose
    cfg.commands['standup'] = command_cfg(role)
    weights = rewards.GETUP_WEIGHTS if role == 'getup' else rewards.WALK_WEIGHTS
    cfg.rewards = {name:RewardTermCfg(func=rewards.reward, weight=weight, params={'name':name})
                   for name,weight in weights.items()}
    for name, func in (('actor', observations.actor_observation), ('critic', observations.critic_observation),
                       ('history', observations.actor_observation)):
        cfg.observations[name].terms['proprioception'].func = func
    cfg.events['reset_observations'].func = observations.reset_observations
    cfg.metrics['observation_delay_ms'].func = observations.sample_observations
    cfg.terminations['failure'] = TerminationTermCfg(func=mdp.failure)
    return cfg


def make_ppo_cfg(role):
    cfg = make_stand_v2_ppo_cfg()
    cfg.experiment_name = f'chuanliantui_{role}'
    cfg.actor.class_name = 'ct_mjlab.envs.chuanliantui_split.network:HistoryActor'
    if role == 'walk':
        # 原训练200轮内std已降到0.13；保留探索下限，部署只用均值动作不受影响。
        cfg.actor.distribution_cfg['std_range'] = (WALK_ACTOR_STD_MIN, 1e6)
    return cfg


def make_getup_env_cfg(*, play=False):
    return make_env_cfg('getup', play=play)


def make_walk_env_cfg(*, play=False):
    return make_env_cfg('walk', play=play)


def make_getup_ppo_cfg():
    return make_ppo_cfg('getup')


def make_walk_ppo_cfg():
    return make_ppo_cfg('walk')
