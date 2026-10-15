"""Command-line arguments shared by the mjlab scripts, mirroring wheel_legged_gym/utils/helpers.py."""
import argparse
from pathlib import Path


def get_args(description, *, defaults=None, extra=()):
    """Parse the common task/device/seed arguments plus script-specific ones.

    extra: iterable of (flags, kwargs) passed to add_argument.
    """
    from ..envs import task_registry

    defaults = defaults or {}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--task', choices=task_registry.names(), default='chuanliantui')
    parser.add_argument('--num-envs', type=int, default=defaults.get('num_envs'))
    parser.add_argument('--seed', type=int, default=None, help='Override the task PPO seed')
    parser.add_argument('--device', default='cuda:0')
    for flags, kwargs in extra:
        parser.add_argument(*flags, **kwargs)
    return parser.parse_args()


def checkpoint_arg(required=False):
    return (('--checkpoint',), dict(type=Path, required=required, help='model_*.pt'))
