"""Train a registered mjlab task (same flow as wheel_legged_gym/scripts/train.py)."""
from dataclasses import asdict
from pathlib import Path

from ct_mjlab.envs import task_registry
from ct_mjlab.utils.helpers import get_args


def training_viewer(env, port):
    """Display the actual rollout states; PPO remains the only simulation driver."""
    import time
    import viser
    from mjlab.viewer.viser.scene import MjlabViserScene

    server = viser.ViserServer(host='127.0.0.1', port=port, label='mjlab training')
    try:
        sim = env.unwrapped.sim
        scene = MjlabViserScene(server, sim.mj_model, env.num_envs,
            sim_model=sim.model, expanded_fields=sim.expanded_fields)
        scene.show_only_selected = False
        scene.create_scene_gui(camera_distance=6., show_debug_viz_control=False)
        server.gui.add_markdown(f'**Training: {env.num_envs} environments** — live PPO rollouts')
        scene.update(sim.data)
        original_step = env.step
        last_render = time.perf_counter()

        def step(actions):
            nonlocal last_render
            started = time.perf_counter()
            result = original_step(actions)
            # Limit simulation speed for viewing without adding steps or changing actions.
            time.sleep(max(0., env.unwrapped.step_dt - (time.perf_counter() - started)))
            if time.perf_counter() - last_render >= 1. / 30.:
                with server.atomic():
                    scene.update(sim.data)
                last_render = time.perf_counter()
            return result

        env.step = step
        print(f'Training viewer: http://127.0.0.1:{server.get_port()}', flush=True)
        return server, original_step
    except BaseException:
        server.stop()
        raise


def train(args):
    if args.task == "fudan_ppo_walk":
        return train_fudan(args)
    if args.task != "smp" or args.smp_prior is None:
        raise ValueError("Only retained BC/SMP PPO is supported; use --task smp --smp-prior, or ct_mjlab.supermimic")
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
    viewer = None
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
        if args.smp_prior is not None:
            import shutil
            source = Path(__file__).resolve().parents[1]
            shutil.copytree(source, log_dir / 'snapshot' / 'ct_mjlab',
                            ignore=shutil.ignore_patterns('__pycache__'))
        print(f'Training output: {log_dir}', flush=True)
        if args.viewer == 'viser':
            viewer, original_step = training_viewer(env, args.viser_port)
        runner.learn(num_learning_iterations=train_cfg.max_iterations)
    finally:
        if viewer is not None:
            env.step = original_step
            viewer.stop()
        env.close()



def train_fudan(args):
    import json
    import time
    import torch
    from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT, RESET_V2_CONTRACT
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config, from_record
    from ct_mjlab.envs.fudan_ppo_walk.commands import command_config, LEGACY_COMMAND_SAMPLING
    from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import snapshot
    if args.smp_weight is not None or any(getattr(args, name, None) for name in ('smp_prior', 'bc_init', 'ppo_init', 'phase_mix', 'wheel_std', 'bc_finetune', 'viewer')):
        raise ValueError('SMP/BC options and the generic viewer do not apply to Fudan training')
    if args.num_envs is None or args.log_dir is None or args.precheck is None:
        raise ValueError('Fudan training requires --num-envs, --log-dir and --precheck')
    if (args.deadline is None) == (args.iterations is None):
        raise ValueError('Fudan training requires exactly one deadline or iteration budget')
    if args.preserve_history_v2 and args.migrate_history_v2:
        raise ValueError('History preservation and migration are mutually exclusive')
    if not args.device.startswith('cuda'):
        raise ValueError('Fudan training requires the GPU physical backend')
    torch.set_num_threads(1)
    source = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume is not None else {}
    if args.preserve_history_v2 and (args.resume is None or source.get('reset_contract') != RESET_V2_CONTRACT):
        raise ValueError('--preserve-history-v2 requires a source v2 checkpoint')
    target_reset = RESET_V2_CONTRACT if args.preserve_history_v2 else RESET_CONTRACT
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.seed is not None:
        train_cfg.seed = args.seed
    env_cfg.reset_contract = target_reset
    disturbances = from_task_config(env_cfg)
    proof = json.loads(args.precheck.read_text())
    if not proof['passed'] or proof.get('reset_contract') != target_reset or proof.get('kind') != 'physical-visible-precheck' or (proof.get('num_envs') != 20) or (proof.get('physics_steps', 0) <= 0):
        raise ValueError('A passed current reset/history physical precheck is required; older prechecks cannot approve this recipe')
    if from_record(proof.get('disturbances')) != disturbances:
        raise ValueError('Physical precheck disturbance settings differ from the requested training settings')
    if proof.get('command_sampling', LEGACY_COMMAND_SAMPLING) != command_config(env_cfg):
        raise ValueError('Physical precheck command sampling differs from the requested training settings')
    if args.migrate_history_v2 and args.resume is None:
        raise ValueError('--migrate-history-v2 requires --resume')
    if args.num_envs % train_cfg.algorithm.num_mini_batches:
        raise ValueError('Four equal reference minibatches required')
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError('Iteration budget must be positive')
    if args.deadline is not None and time.time() >= args.deadline:
        raise ValueError('Training budget already expired')
    torch.manual_seed(train_cfg.seed)
    args.log_dir.mkdir(parents=True, exist_ok=False)
    snapshot(args.log_dir)
    env = None
    try:
        env, env_cfg = task_registry.make_env(
            args.task,
            env_cfg=env_cfg,
            train_cfg=train_cfg,
            num_envs=args.num_envs,
            seed=train_cfg.seed,
            device=args.device
        )
        options = {}
        if args.migrate_history_v2:
            options['migrate_history_v2'] = True
        if from_record(source.get('disturbances')) != disturbances:
            options['allow_disturbance_change'] = True
        runner, train_cfg = task_registry.make_alg_runner(
            env,
            args.task,
            train_cfg=train_cfg,
            log_dir=args.log_dir,
            resume=args.resume,
            resume_options=options,
            device=args.device
        )
        report = runner.learn_until(float('inf'), max_iterations=runner.iteration + args.iterations) if args.iterations is not None else runner.learn_until(args.deadline)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        (args.log_dir / 'failure.json').write_text(json.dumps(dict(error=repr(exc), time=time.time()), indent=2) + '\n')
        if env is not None:
            import numpy as np
            np.savez_compressed(
                args.log_dir / 'failure_state.npz',
                qpos=env.env.sim.data.qpos.cpu().numpy(),
                qvel=env.env.sim.data.qvel.cpu().numpy()
            )
        raise
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    args = get_args(__doc__, argv=argv, extra=(
        (('--log-dir',), dict(type=Path, default=None)),
        (('--deadline',), dict(type=float, default=None)),
        (('--precheck',), dict(type=Path, default=None)),
        (('--preserve-history-v2',), dict(action='store_true')),
        (('--migrate-history-v2',), dict(action='store_true')),
        (('--iterations',), dict(type=int, default=None, help='Override max_iterations')),
        (('--run-name',), dict(default='')),
        (('--viewer',), dict(choices=('viser',), default=None,
                            help='Show the actual training rollouts in a browser')),
        (('--viser-port',), dict(type=int, default=8093)),
        (('--resume',), dict(type=Path, default=None, help='Continue from a model_*.pt of the selected task')),
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
    if args.task == 'fudan_ppo_walk':
        if args.preserve_history_v2 and args.migrate_history_v2:
            raise SystemExit(2)
        return train(args)
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
