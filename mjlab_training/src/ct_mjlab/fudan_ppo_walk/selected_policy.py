"""User-selected checkpoint and inference-only parking pitch reference."""
import hashlib
from pathlib import Path

import torch

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg


def selection_record():
    cfg = FudanWalkCfg()
    selection = cfg.inference
    if hashlib.sha256(Path(selection.checkpoint).read_bytes()).hexdigest() != selection.checkpoint_sha256:
        raise ValueError('Selected checkpoint SHA256 differs from the adopted model')
    if cfg.parking_integral.enabled:
        raise ValueError('The adopted parking recipe requires integral compensation disabled')
    return dict(version='fudan-selected-parking-pitch-reference-v1',
        checkpoint=selection.checkpoint, checkpoint_sha256=selection.checkpoint_sha256,
        parking_pitch_deg=selection.parking_pitch_deg, ramp_s=.5, initial_delay_s=2.,
        integral_enabled=False, implementation='Gravity reference on actor current frame and encoder history only')


def reference_angles(t, starts, target):
    return ((t - starts) / .5).clamp(0., 1.) * target


def rotate_gravity(frame, angle):
    """Subtract the reference pitch from the gravity input; preserve every other slot."""
    result = frame.clone()
    c, s = angle.cos(), angle.sin()
    x, z = frame[..., 3], frame[..., 5]
    result[..., 3] = c*x + s*z
    result[..., 5] = -s*x + c*z
    return result


def policy_inputs(obs, history, t, starts, target):
    current = rotate_gravity(obs, reference_angles(t, starts, target))
    # 每个历史帧使用对应时刻的偏置，保持与已验证的斜坡时序相同。
    times = t + torch.arange(-4, 1, device=obs.device, dtype=obs.dtype)[None] * .01
    angles = reference_angles(times, starts[:, None], target[:, None])
    past = rotate_gravity(history.reshape(-1, 5, 25), angles).reshape(-1, 125)
    return current, past


def reference_starts(items, obs):
    starts = []
    for sample in items:
        segments = sample['case']['segments']
        first = next((i for i,s in enumerate(segments) if s['command'][0] == s['command'][1] == 0), None)
        if first is None:
            starts.append(float('inf'))
            continue
        if any(s['command'][0] != 0 or s['command'][1] != 0 for s in segments[first:]):
            raise ValueError('Selected pitch recipe has not validated leaving a parking segment')
        start = segments[first]['start']
        starts.append(2. if start == 0 else start)
    return obs.new_tensor(starts)
