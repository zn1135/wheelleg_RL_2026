"""独立台阶训练分发，必须给定预算及当前配方的GPU预检。"""
import json
import time
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import snapshot, sha
from .config import TASK, SOURCE, SOURCE_SHA, recipe_record


def train(args):
    if any(getattr(args, name, None) for name in ('smp_prior', 'smp_weight', 'bc_init', 'phase_mix',
           'wheel_std', 'bc_finetune', 'preserve_history_v2', 'migrate_history_v2', 'viewer')):
        raise ValueError('SMP/BC/legacy history options do not apply to the platform task')
    if args.resume and args.ppo_init:
        raise ValueError('--resume and --ppo-init are mutually exclusive')
    if args.log_dir is None or args.precheck is None or args.num_envs is None:
        raise ValueError('Step training requires --log-dir, --precheck and --num-envs')
    if (args.iterations is None) == (args.deadline is None):
        raise ValueError('Supply exactly one --iterations or --deadline budget')
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError('Iteration budget must be positive')
    if args.deadline is not None and args.deadline <= time.time():
        raise ValueError('Deadline expired')
    if not args.device.startswith('cuda') or args.num_envs < 4 or args.num_envs % 4:
        raise ValueError('Use GPU and a positive number of environments divisible by4')
    cfg, train_cfg = task_registry.get_cfgs(TASK)
    if args.seed is not None:
        train_cfg.seed = args.seed
    proof = json.loads(args.precheck.read_text())
    if (not proof.get('passed') or proof.get('kind') != 'step_gpu_precheck' or
            proof.get('recipe') != recipe_record(cfg) or proof.get('num_envs') != 20 or
            proof.get('physics_steps', 0) < 2000 or proof.get('optimizer_updates') != 0):
        raise ValueError('A matching20-environment no-learning GPU precheck is required')
    from scripts.agent.fudan_ppo_step.check import source_fingerprint
    if proof.get('source_fingerprint') != source_fingerprint():
        raise ValueError('Step precheck source differs from the current implementation')
    source = args.ppo_init or SOURCE
    if not args.resume and not args.ppo_init and sha(source) != SOURCE_SHA:
        raise ValueError('Default3000 initialization SHA differs')
    args.log_dir.mkdir(parents=True, exist_ok=False)
    snapshot(args.log_dir)
    torch.set_num_threads(1)
    torch.manual_seed(train_cfg.seed)
    env = None
    try:
        env, _ = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=train_cfg,
            num_envs=args.num_envs, seed=train_cfg.seed, device=args.device)
        runner, _ = task_registry.make_alg_runner(env, TASK, train_cfg=train_cfg,
            log_dir=args.log_dir, resume=args.resume, device=args.device)
        if not args.resume:
            runner.initialize_from_ppo(source)
        result = runner.learn_until(args.deadline if args.deadline is not None else float('inf'),
            max_iterations=None if args.iterations is None else runner.iteration+args.iterations)
        print(json.dumps(result, indent=2))
    finally:
        if env is not None:
            env.close()
