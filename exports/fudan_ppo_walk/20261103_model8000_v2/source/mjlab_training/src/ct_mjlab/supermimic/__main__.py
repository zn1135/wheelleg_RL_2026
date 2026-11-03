"""Independent entry; never constructs a PPO runner."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import torch

from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from ct_mjlab.envs import task_registry
from ct_mjlab.validation import require_gpu
from .buffer import ReplayBuffer
from .learner import (Learner, ALGORITHM_CONTRACT, WORLD_STEPS_PER_UPDATE,
                      WORLD_LOSS_RECIPE, ALTERNATING_RECIPE)
from .networks import act, clip_action
from .state import capture_state, capture_parameters, physical_snapshot, SUBSTEPS

CONTACT_TRAINING_RECIPE = dict(name='supermimic_real_nominal_solref_mixture',
    nominal_fraction=.25, environment_selection='env_id % 4 == 0',
    nominal_source='compiled_scene_default_geom_solref',
    scope='Real training contact mixture affects world, actor and critic; all other original randomization remains active')


@requires_model_fields('geom_solref')
def supermimic_nominal_contact(env, env_ids):
    # Run after the original randomization; these are real simulated contacts.
    selected = torch.arange(env.num_envs,device=env.device)[::4]
    geoms = env._standup_randomization.collision_geom_ids
    nominal = env.sim.get_default_field('geom_solref')
    env.sim.model.geom_solref[selected[:,None],geoms[None,:]] = nominal[geoms][None,:]


def validate_contact_encoding(env):
    """Lock the real solver assumptions used by the reference K/B encoding."""
    import mujoco
    template, model = env.sim.mj_model, env.sim.model
    timestep = model.opt.timestep[:]
    if (not math.isclose(env.physics_dt,.002,rel_tol=0.,abs_tol=1e-12)
            or not math.isclose(float(template.opt.timestep),.002,rel_tol=0.,abs_tol=1e-12)
            or not torch.isclose(timestep,timestep.new_tensor(.002),rtol=0.,atol=1e-9).all()):
        raise ValueError('Contact encoding requires the actual 2ms physics timestep')
    for option in (template.opt,model.opt):
        if (int(option.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
                or int(option.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)):
            raise ValueError('Contact encoding requires REFSAFE and no contact override')
    if template.npair != 0 or model.npair != 0:
        raise ValueError('Contact encoding does not support explicit contact pairs')
    bodies = set(env.scene['robot'].indexing.body_ids.tolist())
    collision = [i for i in range(template.ngeom)
                 if template.geom_contype[i] or template.geom_conaffinity[i]]
    robot = [i for i in collision if int(template.geom_bodyid[i]) in bodies]
    terrain = [i for i in collision if int(template.geom_bodyid[i]) not in bodies]
    if not robot or not terrain:
        raise ValueError('Expected robot and terrain collision geometries')
    priority = model.geom_priority
    if not (priority[...,robot] == 1).all() or not (priority[...,terrain] == 0).all():
        raise ValueError('Contact encoding requires robot priority1 and terrain priority0')
    solimp = model.geom_solimp[...,collision,:]
    expected = solimp.new_tensor((.9,.95,.001,.5,2.))
    if not torch.equal(solimp,expected.expand_as(solimp)):
        raise ValueError('Contact encoding requires the unchanged common default solimp')
    if not (model.geom_solmix[...,collision] == 1.).all():
        raise ValueError('Contact encoding requires default contact mixing weights')
    solref = model.geom_solref[...,robot,:]
    reference = solref[...,:1,:]
    valid_format = ((reference > 0.).all(-1) | (reference < 0.).all(-1))
    if (not torch.isfinite(solref).all() or not valid_format.all()
            or not torch.equal(solref,reference.expand_as(solref))):
        raise ValueError('Robot collision SOLREF must share one valid pair per environment')
    floor = model.geom_solref[...,terrain,:]
    nominal = floor.new_tensor((.02,1.))
    if not torch.equal(floor,nominal.expand_as(floor)):
        raise ValueError('Contact encoding requires unchanged nominal terrain SOLREF')
    return dict(physics_dt_s=.002,refsafe_enabled=True,contact_override=False,
        explicit_contact_pairs=0,common_geom_solimp=expected.cpu().tolist(),
        common_dmax=float(expected[1]),refsafe_min_timeconst_s=.004,
        model_contact_encoding='reference_K_B_before_dmax',
        robot_priority=1,terrain_priority=0,geom_solmix=1.,terrain_solref=[.02,1.],
        robot_contact_solref_shared_per_environment=True)


@torch.no_grad()
def capture_initial_physics(env, buffer, *, task, seed):
    """Read the actual initial training state/materials without stepping or sampling."""
    cpu_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state_all()
    counter_before=int(env.common_step_counter)
    state,parameters=capture_state(env),capture_parameters(env)
    geoms=env._standup_randomization.collision_geom_ids
    solref=env.sim.model.geom_solref[:,geoms].clone()
    nominal=env.sim.get_default_field('geom_solref')[geoms].clone()
    selected=(solref==nominal[None]).all(dim=(-1,-2)).nonzero().flatten()
    if not torch.equal(state,capture_state(env)) or not torch.equal(parameters,capture_parameters(env)):
        raise RuntimeError('Initial physics capture changed the physical state or parameters')
    cpu_after=torch.get_rng_state().clone();cuda_after=torch.cuda.get_rng_state_all()
    counter_after=int(env.common_step_counter)
    if (not torch.equal(cpu_before,cpu_after) or counter_before!=counter_after
            or len(cuda_before)!=len(cuda_after)
            or not all(torch.equal(a,b) for a,b in zip(cuda_before,cuda_after))):
        raise RuntimeError('Initial physics capture changed RNG or stepped the environment')
    return dict(schema='supermimic-initial-physics-v1',algorithm_contract=ALGORITHM_CONTRACT,
        task=task,seed=seed,num_envs=env.num_envs,policy_dt_s=env.step_dt,physics_dt_s=env.physics_dt,
        state=state.cpu(),parameters=parameters.cpu(),robot_collision_geom_ids=geoms.cpu(),
        robot_collision_solref=solref.cpu(),default_robot_collision_solref=nominal.cpu(),
        original_restitution=env._standup_randomization.restitution.cpu().clone(),
        train_env_ids=buffer.train_ids.cpu(),validation_env_ids=buffer.validation_ids.cpu(),
        nominal_contact_env_ids=selected.cpu(),contact_training_recipe=CONTACT_TRAINING_RECIPE.copy(),
        contact_encoding_assumptions=env._supermimic_contact_encoding,
        torch_rng_before=cpu_before,torch_rng_after=cpu_after,cuda_rng_before=cuda_before,cuda_rng_after=cuda_after,
        common_step_counter_before=counter_before,common_step_counter_after=counter_after,
        rng_unchanged_during_capture=True,state_and_parameters_unchanged_during_capture=True)


@torch.no_grad()
def collect(env, learner, buffer, obs, *, exploration=.2):
    before = dict(history=obs['history'].clone(),privileged=obs['critic'].clone(),
                  velocity=obs['velocity_target'].clone())
    action = act(learner.actor,before['history'],unclipped=True)
    if exploration:
        action = action + exploration*torch.randn_like(action)
    if not torch.isfinite(action).all():
        raise FloatingPointError('Nonfinite action; not sent to simulation')
    # Clip after exploration, then use this same action for simulation and replay.
    action = clip_action(learner.actor,action)
    raw = env.unwrapped
    if buffer is not None:
        before.update(state=capture_state(raw),parameters=capture_parameters(raw))
        path = []
        original_substep = raw.metrics_manager.compute_substep
        def record_substep():
            original_substep()
            if not path:
                # process_actions has now sampled the exogenous push for this tick.
                before['parameters'] = capture_parameters(raw)
            # Observe the existing step; never call forward() or alter physics.
            path.append(physical_snapshot(raw))
        raw.metrics_manager.compute_substep = record_substep
    try:
        obs, reward, done, extras = env.step(action)
    finally:
        if buffer is not None:
            raw.metrics_manager.compute_substep = original_substep
    if not all(torch.isfinite(v).all() for v in (obs['history'],obs['critic'],reward)):
        raise FloatingPointError('Nonfinite simulator transition')
    if buffer is not None:
        from .analytic_reward import differentiable_reward
        record = env.unwrapped._split_walk_rewards.record
        if ((~record['features'].isfinite().all(-1)) & (record['context'][:,5] < .5)).any():
            raise FloatingPointError('Nonfinite reward features without a numerical-failure mask')
        terms, measured = differentiable_reward(record['features'],record['action'],
            record['previous'],record['previous_previous'],record['context'])
        # Prove the training label uses the unchanged reward, at its original physical instant.
        torch.testing.assert_close(measured,reward,atol=2e-7,rtol=2e-5)
        torch.testing.assert_close(torch.stack(list(terms.values())),
            torch.stack([env.unwrapped._split_walk_rewards.values[name] for name in terms]),
            atol=2e-6,rtol=2e-5)
        buffer.add(**before,action=action,next_frame=obs['actor'],next_privileged=obs['critic'],
            reward=reward,done=done.bool(),failure=env.unwrapped.reset_terminated.bool(),
            next_state=record['state'],path=torch.stack(path,1).flatten(-2),
            **{name:record[name] for name in ('features','context','previous','previous_previous')})
    return obs, reward, done


def make_env(args, *, play=False):
    from mjlab.utils.torch import configure_torch_backends
    cfg, train = task_registry.get_cfgs('supermimic',play=play)
    if not play:
        if 'isaac_domain_randomization' not in cfg.events:
            raise ValueError('SuperMimic contact mixture requires the original training randomization')
        cfg.events['supermimic_nominal_contact'] = EventTermCfg(
            func=supermimic_nominal_contact,mode='startup')
    env, _ = task_registry.make_env('supermimic',env_cfg=cfg,train_cfg=train,
                                   num_envs=args.num_envs,seed=args.seed,device=args.device)
    configure_torch_backends(allow_tf32=False)
    try:
        env.unwrapped._supermimic_contact_encoding = validate_contact_encoding(env.unwrapped)
    except Exception:
        env.close()
        raise
    if not play:
        from ct_mjlab.envs.chuanliantui_split.rewards import WalkRewardState
        from .analytic_reward import capture
        class ObservedWalkRewardState(WalkRewardState):
            def compute(self, raw):
                if self.step != raw.common_step_counter:
                    # Capture before the original reward, reset, forward and command update.
                    self.record = capture(raw,self)
                    self.record['state'] = capture_state(raw)
                return super().compute(raw)
        env.unwrapped._split_walk_rewards = ObservedWalkRewardState(env.unwrapped)
    if env.unwrapped.step_dt != .01 or env.unwrapped.physics_dt != .002 or env.clip_actions is not None:
        env.close()
        raise ValueError('Expected unbounded six raw actions, 100Hz policy and 500Hz physics')
    from .motion_adapter import validate_motion_pd_config
    try:
        validate_motion_pd_config(cfg.actions['pd'])
    except Exception:
        env.close()
        raise
    return env, cfg


def train(args):
    from torch.utils.tensorboard import SummaryWriter
    from mjlab.utils.os import dump_yaml
    if args.log_dir.exists():
        raise FileExistsError(f'Use a new log directory: {args.log_dir}')
    if args.init_world and (args.resume or not args.init_actor):
        raise ValueError('World transfer requires --init-actor and cannot combine with --resume')
    env, cfg = make_env(args)
    writer = None
    started, iteration, transitions, world_updates = None, 0, 0, 0
    try:
        learner = Learner(args.device,actor_lr=args.actor_lr)
        command = env.unwrapped.command_manager.get_term('standup')
        initialization = dict(kind='fresh_from_zero',fresh_from_zero=True)
        if args.resume:
            # Reject changed training settings before restoring any learned state or RNG.
            saved = torch.load(args.resume,map_location='cpu',weights_only=False)
            previous = saved['config']
            if previous.get('world_steps_per_update') != WORLD_STEPS_PER_UPDATE:
                raise ValueError('Resume world fitting ratio mismatch')
            if previous.get('world_loss_recipe') != WORLD_LOSS_RECIPE:
                raise ValueError('Resume configuration mismatch: world_loss_recipe')
            if previous.get('alternating_recipe') != ALTERNATING_RECIPE:
                raise ValueError('Resume configuration mismatch: alternating_recipe')
            if previous.get('actor_lr',3e-4) != args.actor_lr:
                raise ValueError('Resume configuration mismatch: actor_lr')
            if previous.get('contact_training_recipe') != CONTACT_TRAINING_RECIPE:
                raise ValueError('Resume configuration mismatch: contact_training_recipe')
            if previous.get('contact_encoding_assumptions') != env.unwrapped._supermimic_contact_encoding:
                raise ValueError('Resume configuration mismatch: contact_encoding_assumptions')
            for key in ('task','num_envs','seed','horizon','window','batch','buffer_steps','policy_warmup_updates'):
                if previous[key] != getattr(args,key):
                    raise ValueError(f'Resume configuration mismatch: {key}')
            learner.load(args.resume,command)
            iteration, transitions = saved['iteration'], saved['transitions']
            world_updates = saved['world_updates']
            initialization = dict(kind='resume',parent_checkpoint=str(args.resume.resolve()),
                parent_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                parent_algorithm_contract=saved['algorithm_contract'],
                inherited_initialization=saved['initialization'],fresh_from_zero=False)
            env.reset()  # New episodes/replay; restored curriculum and all optimizers are retained.
        elif args.init_actor:
            initialization = learner.init_actor(args.init_actor,command,task=args.task)
            if args.init_world:
                actor_transfer = initialization
                world_transfer = learner.init_world(args.init_world,command,task=args.task)
                initialization = dict(kind='actor_and_world_transfer_fresh_optimizers',
                    actor_transfer=actor_transfer,world_transfer=world_transfer,
                    inherited=['actor','world'],reset=['critic','opt_actor','opt_encoder',
                        'opt_world','opt_critic','iteration','transitions','world_updates',
                        'curriculum','replay','rng'],fresh_from_zero=False)
        world_updates = learner.optimizer_update_counts()['world']
        obs = env.get_observations()
        buffer = ReplayBuffer(args.buffer_steps,args.num_envs,args.device)
        args.log_dir.mkdir(parents=True,exist_ok=False)
        # Keep the actual training source with this run, including uncommitted task code.
        source = Path(__file__).resolve().parents[1]
        hashes = {}
        for path in sorted(source.rglob('*.py')):
            relative = path.relative_to(source)
            destination = args.log_dir/'snapshot'/'ct_mjlab'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.log_dir/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config['initialization'] = initialization
        config['world_steps_per_update'] = WORLD_STEPS_PER_UPDATE
        config['world_loss_recipe'] = WORLD_LOSS_RECIPE.copy()
        config['alternating_recipe'] = ALTERNATING_RECIPE.copy()
        config['contact_training_recipe'] = CONTACT_TRAINING_RECIPE.copy()
        config['contact_encoding_assumptions'] = env.unwrapped._supermimic_contact_encoding
        physics_path=args.log_dir/'initial_physics.pt'
        torch.save(capture_initial_physics(env.unwrapped,buffer,task=args.task,seed=args.seed),physics_path)
        config['initial_physics_sha256']=hashlib.sha256(physics_path.read_bytes()).hexdigest()
        (args.log_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        dump_yaml(args.log_dir/'env.yaml',asdict(cfg))
        writer = SummaryWriter(str(args.log_dir/'tensorboard'))
        writer.add_text('algorithm',ALGORITHM_CONTRACT)
        learner.save(args.log_dir/'initial.pt',command,config=config,iteration=iteration,
                     transitions=transitions,world_updates=world_updates)
        phase = learner.training_phase(args.policy_warmup_updates)
        writer.add_text('training_phase',phase,iteration)
        prewarm_saved = phase == 'alternating'
        postwarm_saved = phase == 'alternating'
        if phase == 'alternating_ready':
            learner.save(args.log_dir/'prewarm.pt',command,config=config,iteration=iteration,
                         transitions=transitions,world_updates=world_updates)
            prewarm_saved = True
        started = time.monotonic()
        deadline, last_save = started+args.seconds, started
        actor_target = (learner.optimizer_update_counts()['actor']+args.actor_updates
            if getattr(args,'actor_updates',None) is not None else None)
        updates = 0
        with (args.log_dir/'metrics.jsonl').open('x') as metrics_file:
            while time.monotonic() < deadline and (actor_target is None or learner.optimizer_update_counts()['actor'] < actor_target):
                reward_sum = done_sum = 0.
                collected = 0
                for _ in range(8):
                    if time.monotonic() >= deadline:
                        break
                    obs, reward, done = collect(env,learner,buffer,obs)
                    reward_sum += float(reward.mean()); done_sum += float(done.sum())
                    transitions += args.num_envs; collected += 1
                metrics = {}
                # Fit dynamics/estimator first; then enable imagined policy/value updates.
                if buffer.size >= max(128,args.horizon,args.window):
                    for _ in range(4):
                        if time.monotonic() >= deadline or (actor_target is not None and learner.optimizer_update_counts()['actor'] >= actor_target):
                            break
                        # Source-world training never counts toward these local Adam steps.
                        ready = learner.optimizer_update_counts()['world'] >= args.policy_warmup_updates
                        metrics = learner.update(buffer,batch=args.batch,window=args.window,
                            horizon=args.horizon,policy=ready,encoder=ready)
                        updates += 1
                        counts = learner.optimizer_update_counts()
                        world_updates = counts['world']
                        current_phase = learner.training_phase(args.policy_warmup_updates)
                        if not prewarm_saved and current_phase == 'alternating_ready':
                            learner.save(args.log_dir/'prewarm.pt',command,config=config,iteration=iteration+1,
                                         transitions=transitions,world_updates=world_updates)
                            prewarm_saved = True
                        if not postwarm_saved and counts['actor'] > 0:
                            learner.save(args.log_dir/'postwarm.pt',command,config=config,iteration=iteration+1,
                                         transitions=transitions,world_updates=world_updates)
                            postwarm_saved = True
                        if current_phase != phase:
                            writer.add_text('training_phase',current_phase,iteration+1)
                            phase = current_phase
                iteration += 1
                elapsed = time.monotonic()-started
                counts = learner.optimizer_update_counts()
                world_updates = counts['world']
                metrics.update(iteration=iteration,transitions=transitions,elapsed_s=elapsed,
                               real_reward=reward_sum/max(collected,1),resets=done_sum,
                               updates_this_run=updates,training_phase=learner.training_phase(args.policy_warmup_updates),
                               **{name+'_updates':value for name,value in counts.items()})
                data = env.unwrapped.scene['robot'].data
                metrics.update(real_vx_mae=float((data.root_link_lin_vel_b[:,0]-command.command[:,0]).abs().mean()),
                    real_horizontal_speed_max=float(data.root_link_lin_vel_w[:,:2].norm(dim=-1).max()),
                    real_ready_fraction=float(command.walk_ready.float().mean()),
                    real_episode_length_s=float(env.unwrapped.episode_length_buf.float().mean()*env.unwrapped.step_dt))
                metrics_file.write(json.dumps(metrics)+'\n'); metrics_file.flush()
                for key,value in metrics.items():
                    if key != 'training_phase':
                        writer.add_scalar(key,value,iteration)
                if iteration%10==0:
                    print(json.dumps(metrics),flush=True)
                if time.monotonic()-last_save >= 60:
                    learner.save(args.log_dir/f'checkpoint_{iteration:06d}.pt',command,
                        config=config,iteration=iteration,transitions=transitions,world_updates=world_updates)
                    last_save = time.monotonic()
        learner.save(args.log_dir/'final.pt',command,config=config,iteration=iteration,
                     transitions=transitions,world_updates=world_updates)
        print(json.dumps(dict(log_dir=str(args.log_dir),elapsed_s=time.monotonic()-started,
                              transitions=transitions,iteration=iteration)),flush=True)
    except (KeyboardInterrupt, Exception):
        if started is not None:
            # An interrupt can land between the four model steps; preserve the actual count.
            world_updates = learner.optimizer_update_counts()['world']
            learner.save(args.log_dir/'interrupted.pt',command,config=config,iteration=iteration,
                         transitions=transitions,world_updates=world_updates,interrupted=True)
        raise
    finally:
        if writer is not None:
            writer.close()
        env.close()


@torch.inference_mode()
def play(args):
    validation = require_gpu(args.device)
    env, _ = make_env(args,play=True)
    try:
        learner = Learner(args.device)
        command = env.unwrapped.command_manager.get_term('standup')
        learner.load(args.checkpoint,command,inference=True)
        command.speed_curriculum_frozen = True
        env.reset()
        obs = env.get_observations()
        total, failures = 0., 0
        started = time.monotonic()
        for _ in range(args.steps):
            obs, reward, _ = collect(env,learner,None,obs,exploration=0.)
            total += float(reward.mean())
            failures += int(env.unwrapped.reset_terminated.sum())
        report = dict(**validation,steps=args.steps,num_envs=args.num_envs,mean_return=total,
                      physical_failures=failures,wall_seconds=time.monotonic()-started,
                      scope='Frozen-policy mjlab rollout; not wheel-balance behavior acceptance')
        if args.report:
            if args.report.exists():
                raise FileExistsError(args.report)
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    for name in ('train','play'):
        p = sub.add_parser(name)
        # Walk only in this first port; get-up has a separate height curriculum and acceptance.
        p.set_defaults(task='chuanliantui_walk')  # 已有SM模型的内部合同标识，非公开task选择
        p.add_argument('--num-envs',type=int,default=128 if name=='train' else 20)
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--device',default='cuda:0')
        if name=='train':
            p.add_argument('--seconds',type=float,required=True,help='Explicit learning budget, including collection')
            p.add_argument('--actor-updates',type=int,help='Stop after this many additional actual actor Adam updates, or at the time budget')
            p.add_argument('--log-dir',type=Path,required=True)
            initialization = p.add_mutually_exclusive_group()
            initialization.add_argument('--resume',type=Path)
            initialization.add_argument('--init-actor',type=Path)
            p.add_argument('--init-world',type=Path,
                help='Verified completed V12 world and norms; requires --init-actor, with fresh optimizers')
            p.add_argument('--policy-warmup-updates',type=int,default=256,
                help='Local world Adam steps before the next world4 group enables encoder/actor/value')
            p.add_argument('--horizon',type=int,default=8)
            p.add_argument('--window',type=int,default=4)
            p.add_argument('--batch',type=int,default=128)
            p.add_argument('--actor-lr',type=float,default=3e-4,help='Learning rate for the control MLP only')
            p.add_argument('--buffer-steps',type=int,default=512)
        else:
            p.add_argument('--checkpoint',type=Path,required=True)
            p.add_argument('--steps',type=int,default=1000)
            p.add_argument('--report',type=Path)
    p = sub.add_parser('export')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.mode=='train':
        if args.actor_updates is not None and args.actor_updates<1:
            parser.error('--actor-updates must be positive')
        if not math.isfinite(args.actor_lr) or args.actor_lr<=0:
            parser.error('--actor-lr must be positive and finite')
        if (not math.isfinite(args.seconds) or args.seconds<=0 or args.num_envs<2
                or min(args.batch,args.horizon,args.window)<1 or args.policy_warmup_updates<0):
            parser.error('Positive budget/dimensions and at least two environments are required')
        if args.buffer_steps<max(128,args.horizon,args.window):
            parser.error('Replay capacity must cover warmup and rollout lengths')
        if args.init_world and (args.resume or not args.init_actor):
            parser.error('--init-world requires --init-actor and cannot combine with --resume')
        train(args)
    elif args.mode=='play':
        if min(args.steps,args.num_envs)<1:
            parser.error('Positive steps/environment count required')
        play(args)
    else:
        from .export import export
        print(json.dumps(export(args.checkpoint,args.output),indent=2))


if __name__=='__main__':
    main()
