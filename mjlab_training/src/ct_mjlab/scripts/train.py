"""Train a registered mjlab task (same flow as wheel_legged_gym/scripts/train.py)."""
from dataclasses import asdict
from pathlib import Path

from ct_mjlab.envs import task_registry
from ct_mjlab.utils.helpers import get_args


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.iterations is not None:
        train_cfg.max_iterations = args.iterations
    env_cfg.seed = train_cfg.seed
    env, env_cfg = task_registry.make_env(
        args.task, env_cfg=env_cfg, train_cfg=train_cfg, num_envs=args.num_envs, device=args.device)
    try:
        runner, train_cfg = task_registry.make_alg_runner(
            env, args.task, train_cfg=train_cfg, log_dir='default', resume=args.resume,
            run_name=args.run_name, device=args.device)
        from mjlab.utils.os import dump_yaml
        log_dir = Path(runner.logger.log_dir)
        dump_yaml(log_dir / 'params' / 'env.yaml', asdict(env_cfg))
        dump_yaml(log_dir / 'params' / 'agent.yaml', asdict(train_cfg))
        runner.add_git_repo_to_log(__file__)
        print(f'Training output: {log_dir}', flush=True)
        runner.learn(num_learning_iterations=train_cfg.max_iterations)
    finally:
        env.close()


def main():
    args = get_args(__doc__, defaults={'num_envs': 4096}, extra=(
        (('--iterations',), dict(type=int, default=None, help='Override max_iterations')),
        (('--run-name',), dict(default='')),
        (('--resume',), dict(type=Path, default=None, help='Continue from a multimode model_*.pt')),
    ))
    # Fail before building the GPU environment.
    if '/' in args.run_name or '\\' in args.run_name:
        raise SystemExit('--run-name must be a label, not a path')
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f'--resume checkpoint not found: {args.resume}')
    train(args)


if __name__ == '__main__':
    main()
