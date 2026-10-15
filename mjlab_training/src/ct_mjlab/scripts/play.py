"""Replay a checkpoint in the play configuration (same flow as wheel_legged_gym/scripts/play.py)."""
import os

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.utils.helpers import checkpoint_arg, get_args


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, play=True)
    if args.seed is not None:
        env_cfg.seed = args.seed
    env, _ = task_registry.make_env(
        args.task, env_cfg=env_cfg, train_cfg=train_cfg, play=True,
        num_envs=args.num_envs, device=args.device)
    try:
        if args.checkpoint is None:
            shape = env.unwrapped.action_space.shape
            policy = lambda obs: torch.zeros(shape, device=env.unwrapped.device)  # noqa: E731
        else:
            runner, _ = task_registry.make_alg_runner(env, args.task, train_cfg=train_cfg, device=args.device)
            runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
        viewer = args.viewer
        if viewer == 'auto':
            viewer = 'native' if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') else 'viser'
        if viewer == 'native':
            from mjlab.viewer import NativeMujocoViewer
            NativeMujocoViewer(env, policy).run()
        else:
            from mjlab.viewer import ViserPlayViewer
            ViserPlayViewer(env, policy).run()
    finally:
        env.close()


def main():
    args = get_args(__doc__, defaults={'num_envs': 1}, extra=(
        checkpoint_arg(),
        (('--viewer',), dict(choices=('auto', 'native', 'viser'), default='auto')),
    ))
    play(args)


if __name__ == '__main__':
    main()
