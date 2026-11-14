"""Height reward parameters, including original semantics for older checkpoints."""
import math
from .fudan_ppo_walk_config import FudanWalkCfg

VERSION='fudan-height-exp-scaled-v1'
LEGACY=dict(version=VERSION,scale=1.,sigma=.001)


def from_record(record=None):
    values=dict(LEGACY if record is None else record)
    if set(values)!={'version','scale','sigma'} or values['version']!=VERSION:
        raise ValueError('Unsupported height reward profile')
    scale,sigma=float(values['scale']),float(values['sigma'])
    if not math.isfinite(scale) or scale<0 or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('Invalid height reward parameters')
    return dict(version=VERSION,scale=scale,sigma=sigma)


def from_task_config(task_cfg=None):
    cfg=(FudanWalkCfg() if task_cfg is None else task_cfg).rewards
    return from_record(dict(version=VERSION,scale=cfg.scales['base_height'],sigma=cfg.height_sigma))


def configure(task,record=None):
    values=from_record(record)
    task.rewards.scales['base_height']=values['scale']
    task.rewards.height_sigma=values['sigma']
    return values
