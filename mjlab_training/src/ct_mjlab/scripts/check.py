"""Finite physics/interface check: build the task, optionally load a checkpoint, step a few times.

This is not a behavior acceptance; it never calls learn().
"""
import json

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.utils.helpers import checkpoint_arg, get_args


def check(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env, _ = task_registry.make_env(
        args.task, env_cfg=env_cfg, train_cfg=train_cfg, num_envs=args.num_envs,
        seed=train_cfg.seed if args.seed is None else args.seed, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, args.task, train_cfg=train_cfg, device=args.device)
        if args.checkpoint is not None:
            runner.load(str(args.checkpoint), strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.get_observations(), None
        n = env.num_envs
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, reward, _, _ = env.step(policy(obs))
                for name in ('actor', 'critic'):
                    if not torch.isfinite(obs[name]).all():
                        raise RuntimeError(f'Non-finite {name} observation')
                if not torch.isfinite(reward).all():
                    raise RuntimeError('Non-finite reward')
        shapes = {name: list(obs[name].shape) for name in ('actor', 'critic')}
        expected = {'actor': [n, 39], 'critic': [n, 82]}
        if shapes != expected:
            raise RuntimeError(f'Observation shapes {shapes}, expected {expected}')
        print(json.dumps(dict(kind='finite physics/interface check; no training', task=args.task,
                              steps=args.steps, num_envs=n, checkpoint=str(args.checkpoint),
                              observation_shapes=shapes, action_dim=env.num_actions,
                              physics_dt_s=env.unwrapped.physics_dt,
                              policy_dt_s=env.unwrapped.step_dt), indent=2))
    finally:
        env.close()


def main():
    args = get_args(__doc__, defaults={'num_envs': 2}, extra=(
        checkpoint_arg(), (('--steps',), dict(type=int, default=32)),
    ))
    check(args)


if __name__ == '__main__':
    main()
