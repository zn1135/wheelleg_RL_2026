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
    if args.smp_prior is not None:
        from ct_mjlab.smp.ppo import (configure_smp, policy_observation_contract,
                                     checkpoint_smp_weight, checkpoint_bc_finetune)
        contract=policy_observation_contract(args.bc_init or getattr(args, 'ppo_init', None) or args.resume,
                                             bc=args.bc_init is not None)
        weight = getattr(args, 'smp_weight', None)
        if weight is None:
            weight = checkpoint_smp_weight(args.resume)
        configure_smp(env_cfg, train_cfg, args.smp_prior,policy_contract=contract,reward_weight=weight,
                      bc_finetune=getattr(args, 'bc_finetune', False) or checkpoint_bc_finetune(args.resume))
    env, env_cfg = task_registry.make_env(
        args.task, env_cfg=env_cfg, train_cfg=train_cfg, num_envs=args.num_envs, device=args.device)
    try:
        runner, train_cfg = task_registry.make_alg_runner(
            env, args.task, train_cfg=train_cfg, log_dir='default', resume=args.resume,
            run_name=args.run_name, device=args.device)
        if args.bc_init is not None:
            runner.initialize_from_bc(args.bc_init)
        if getattr(args, 'ppo_init', None) is not None:
            runner.initialize_from_ppo(args.ppo_init)
        if getattr(args, 'phase_mix', False):
            runner.enable_phase_mixing()
        if getattr(args, 'wheel_std', None) is not None:
            runner.alg.set_wheel_std(args.wheel_std)
        from mjlab.utils.os import dump_yaml
        log_dir = Path(runner.logger.log_dir)
        dump_yaml(log_dir / 'params' / 'env.yaml', asdict(env_cfg))
        dump_yaml(log_dir / 'params' / 'agent.yaml', asdict(train_cfg))
        runner.add_git_repo_to_log(__file__)
        if args.smp_prior is not None:
            import shutil
            shutil.copy2(args.smp_prior, log_dir / 'smp_prior.pt')
            source = Path(__file__).resolve().parents[1]
            shutil.copytree(source, log_dir / 'snapshot' / 'ct_mjlab',
                            ignore=shutil.ignore_patterns('__pycache__'))
        print(f'Training output: {log_dir}', flush=True)
        runner.learn(num_learning_iterations=train_cfg.max_iterations)
    finally:
        env.close()


def main():
    args = get_args(__doc__, defaults={'num_envs': 4096}, extra=(
        (('--iterations',), dict(type=int, default=None, help='Override max_iterations')),
        (('--run-name',), dict(default='')),
        (('--resume',), dict(type=Path, default=None, help='Continue from a multimode model_*.pt')),
        (('--smp-prior',), dict(type=Path, default=None, help='Frozen prior_N.pt; enable SMP PPO with normal ground starts')),
        (('--smp-weight',), dict(type=float, choices=(0., 1.), default=None,
                               help='0: matched task-only comparison; default1 or resume checkpoint value')),
        (('--bc-init',), dict(type=Path, default=None, help='Initialize the SMP PPO actor from an aligned BC checkpoint')),
        (('--ppo-init',), dict(type=Path, default=None, help='Adjacent Isaac reward recipe: keep actor, fresh critic/optimizer')),
        (('--phase-mix',), dict(action='store_true', help='Physically stagger normal20s episodes before tight-v2 PPO sampling')),
        (('--wheel-std',), dict(type=float, choices=(.005, .01, .02, .05), default=None,
                              help='Tight-v2 fixed wheel exploration; leg std stays0.001')),
        (('--bc-finetune',), dict(nargs='?', const='v1', default=False, choices=('v1', 'v2', 'isaac-standup', 'isaac-standup-v2', 'isaac-standup-v3'),
                                help='Nominal BC refinement; isaac-standup uses Isaac pre-unlock task rewards')),
    ))
    # Fail before building the GPU environment.
    if '/' in args.run_name or '\\' in args.run_name:
        raise SystemExit('--run-name must be a label, not a path')
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f'--resume checkpoint not found: {args.resume}')
    if args.smp_prior is not None and (args.iterations is None or args.iterations < 1):
        raise SystemExit('SMP PPO requires an explicit positive --iterations budget')
    if args.smp_weight is not None and args.smp_prior is None:
        raise SystemExit('--smp-weight requires --smp-prior')
    if args.bc_init is not None and (args.resume is not None or args.smp_prior is None or not args.bc_init.is_file()):
        raise SystemExit('--bc-init requires an existing BC checkpoint, --smp-prior and no --resume')
    if args.ppo_init is not None and (args.bc_init is not None or args.resume is not None or
            args.smp_prior is None or args.bc_finetune not in ('isaac-standup-v2', 'isaac-standup-v3') or not args.ppo_init.is_file()):
        raise SystemExit('--ppo-init requires Isaac v1 checkpoint, --smp-prior, --bc-finetune isaac-standup-v2/v3 and no --bc-init/--resume')
    if args.bc_finetune and (args.smp_prior is None or (args.bc_init is None and args.resume is None and args.ppo_init is None)):
        raise SystemExit('--bc-finetune requires --smp-prior and --bc-init or a refinement --resume')
    if args.phase_mix and (args.smp_prior is None or args.bc_finetune not in ('isaac-standup-v2', 'isaac-standup-v3')):
        raise SystemExit('--phase-mix requires --smp-prior and --bc-finetune isaac-standup-v2/v3')
    if args.wheel_std is not None and (args.smp_prior is None or args.bc_finetune not in ('isaac-standup-v2', 'isaac-standup-v3')):
        raise SystemExit('--wheel-std requires --smp-prior and --bc-finetune isaac-standup-v2/v3')
    train(args)


if __name__ == '__main__':
    main()
