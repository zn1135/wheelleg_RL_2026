"""Separate roll/pitch penalties, with a soft forward-pitch interval."""
import math
import torch
from .fudan_ppo_walk_config import FudanWalkCfg

LEGACY = dict(version='fudan-orientation-gravity-xy-v1')
VERSION = 'fudan-roll-pitch-soft-band-v1'


def from_record(record):
    if record is None or record == LEGACY:
        return dict(LEGACY)
    if set(record) != {'version', 'pitch_range_deg', 'pitch_inner_ratio', 'roll_scale', 'pitch_scale'} or record['version'] != VERSION:
        raise ValueError('Unsupported posture reward profile')
    band = [float(x) for x in record['pitch_range_deg']]
    ratio = float(record['pitch_inner_ratio'])
    scales = [float(record[x]) for x in ('roll_scale', 'pitch_scale')]
    if len(band) != 2 or not all(math.isfinite(x) for x in [*band, ratio, *scales]) or band[0] > band[1] or not 0 <= ratio <= 1:
        raise ValueError('Invalid roll/pitch reward parameters')
    return dict(version=VERSION, pitch_range_deg=band, pitch_inner_ratio=ratio,
                roll_scale=scales[0], pitch_scale=scales[1])


def from_task_config(task_cfg=None):
    task = FudanWalkCfg() if task_cfg is None else task_cfg
    cfg = task.rewards
    if 'orientation' in cfg.scales:
        if 'roll' in cfg.scales or 'pitch' in cfg.scales:
            raise ValueError('Cannot combine legacy and split posture penalties')
        return dict(LEGACY)
    return from_record(dict(version=VERSION, pitch_range_deg=cfg.pitch_soft_range_deg,
        pitch_inner_ratio=cfg.pitch_inner_penalty_ratio, roll_scale=cfg.scales['roll'], pitch_scale=cfg.scales['pitch']))


def configure(task, record):
    """Restore checkpoint reward semantics without changing observations or physics."""
    profile = from_record(record)
    scales = {}
    for name, value in task.rewards.scales.items():
        if name in ('orientation', 'roll', 'pitch'):
            if any(x in scales for x in ('orientation', 'roll')):
                continue
            if profile == LEGACY:
                scales['orientation'] = -100.
            else:
                scales['roll'] = profile['roll_scale']; scales['pitch'] = profile['pitch_scale']
        else:
            scales[name] = value
    task.rewards.scales = scales
    if profile != LEGACY:
        task.rewards.pitch_soft_range_deg = list(profile['pitch_range_deg'])
        task.rewards.pitch_inner_penalty_ratio = profile['pitch_inner_ratio']
    return profile


def penalties(env):
    profile = getattr(env, 'posture_reward_profile', LEGACY)
    gravity = env.gravity
    if profile == LEGACY:
        return dict(orientation=gravity[:, :2].square().sum(-1))
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
    low, high = [math.radians(x) for x in profile['pitch_range_deg']]
    excess = pitch - pitch.clamp(low, high)
    ratio = profile['pitch_inner_ratio']
    # 区间内弱罚真实pitch；越界部分二次增长，边界处不跳变。
    return dict(roll=roll.square(), pitch=ratio * pitch.square() + (1. - ratio) * excess.square())


def reward_names(record):
    task = FudanWalkCfg()
    configure(task, record)
    return list(task.rewards.scales)
