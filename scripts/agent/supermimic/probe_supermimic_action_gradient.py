#!/usr/bin/env python3
"""Frozen-policy action-gradient probe using native resets and verified prefix replay.

The default reuses one nominal mjlab batch; --fresh-env-per-branch constructs
a new complete environment for each branch. It does not claim a mid-step
snapshot restore: every branch must reproduce the observed control/physics
prefix, and a separate zero-perturb branch must reproduce subsequent transfers.
No optimizer is constructed and no model statistics are updated.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import weakref

import numpy as np
import torch
from torch import nn


MOTORS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
HORIZONS = (1, 4, 8)
# sample_leg_velocity creates these lazily. They feed old stand-only rewards or
# diagnostics, not the active split-walk reward, command or termination paths.
DIAGNOSTIC_ONLY_COMMAND_TENSORS = ('body_speed_peak', 'wheel_velocity_sum')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_rollout(actor, world, history, privileged, commands, done, first_action,
                  policy, gamma, reward_context=None, previous=None, previous_previous=None):
    plain = torch.zeros(history.shape[0], device=history.device)
    expected, alive, result = plain.clone(), torch.ones_like(plain), {}
    for step in range(commands.shape[1]):
        action = first_action if step == 0 else policy(actor, history)
        command = torch.where(done[:, step, None], history[:, -19:-16], commands[:, step])
        if reward_context is None:
            history, privileged, reward, logit = world.step(history, privileged, action, command)
        else:
            history, privileged, reward, logit = world.step(history, privileged, action, command,
                                                           reward_context, previous, previous_previous)
            previous_previous, previous = previous, action
            next_physical_command = torch.stack((command[:, 0] / 2., command[:, 1] * 4., command[:, 2] / 5.), -1)
            reward_context = torch.cat((next_physical_command, reward_context[:, 3:]), -1)
        plain = plain + gamma ** step * reward
        expected = expected + gamma ** step * alive * reward
        alive = alive * (1. - logit.sigmoid())
        if step + 1 in HORIZONS:
            result[step + 1] = dict(plain=plain, expected=expected)
    return result


def model_gradients(actor, world, history, privileged, commands, done, policy, gamma,
                    reward_context=None, previous=None, previous_previous=None):
    first = policy(actor, history).detach().clone().requires_grad_(True)
    predictions = model_rollout(actor, world, history, privileged, commands, done, first, policy, gamma,
                                reward_context, previous, previous_previous)
    result = {}
    for horizon, row in predictions.items():
        result[horizon] = {}
        for name, value in row.items():
            derivative = torch.autograd.grad(value.sum(), first, retain_graph=True)[0]
            if not torch.isfinite(value).all() or not torch.isfinite(derivative).all():
                raise FloatingPointError('Nonfinite model return/action gradient')
            result[horizon][name] = value.detach().cpu().numpy()
            result[horizon][name + '_gradient'] = derivative.detach().cpu().numpy()
    return result


def tensor_signature(raw, obs, full=True):
    """Observed state audit; shared ring cursors are expressed in chronological order."""
    values = {f'obs/{key}': value.detach().cpu().clone() for key, value in obs.items()}
    names = ('qpos', 'qvel', 'ctrl') if not full else (
        'qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'act', 'qfrc_applied', 'xfrc_applied')
    for name in names:
        try:
            values['physics/' + name] = getattr(raw.sim.data, name).detach().cpu().clone()
        except AttributeError:
            pass
    values['episode_length'] = raw.episode_length_buf.detach().cpu().clone()
    if not full:
        return values
    manager = raw.action_manager
    for name in ('action', 'prev_action', 'prev_prev_action'):
        values['action_manager/' + name] = getattr(manager, name).detach().cpu().clone()
    pd = manager.get_term('pd')
    for name in ('_raw_actions', '_control_actions', '_torques', '_motor_response_torque',
                 '_nominal_pos', 'kp_scale', 'kd_scale', 'motor_strength',
                 'action_delay_steps', 'wheel_action_delay_steps'):
        values['pd/' + name] = getattr(pd, name).detach().cpu().clone()
    rows = (torch.arange(len(pd._action_history), device=raw.device) + pd._history_head + 1) % len(pd._action_history)
    values['pd/history'] = pd._action_history[rows].transpose(0, 1).detach().cpu().clone()
    sensors = raw._multimode_observations
    rows = (torch.arange(len(sensors.delay.history), device=raw.device) + sensors.delay.head + 1) % len(sensors.delay.history)
    values['sensor/history'] = sensors.delay.history[rows].transpose(0, 1).detach().cpu().clone()
    for name in ('lags',):
        values['sensor/' + name] = getattr(sensors.delay, name).detach().cpu().clone()
    command = raw.command_manager.get_term('standup')
    # Tensor task memory affects commands, terminations and reward phase. Absolute
    # cache counters and shared circular cursors differ harmlessly across resets.
    for name, value in vars(command).items():
        if name in DIAGNOSTIC_ONLY_COMMAND_TENSORS or name.endswith('_adr'):
            continue  # Static index metadata is not an environment's evolving state.
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == raw.num_envs:
            values['command/' + name] = value.detach().cpu().clone()
    for name in ('wheel_contact', 'base_contact'):
        data = raw.scene[name].data
        values[name + '/force'] = data.force.detach().cpu().clone()
        if data.force_history is not None:
            values[name + '/force_history'] = data.force_history.detach().cpu().clone()
    return values


def signature_difference(actual, expected, num_envs, atol):
    if actual.keys() != expected.keys():
        raise ValueError('State-audit fields changed between branches: '
                         f'added={sorted(actual.keys()-expected.keys())}, '
                         f'missing={sorted(expected.keys()-actual.keys())}')
    maximum = torch.zeros(num_envs, dtype=torch.float64)
    fields = {}
    for key in actual:
        first, second = actual[key], expected[key]
        if first.shape != second.shape or first.ndim == 0 or first.shape[0] != num_envs:
            raise ValueError(f'Unexpected batch dimension for {key}: {first.shape}, {second.shape}')
        delta = (first.double() - second.double()).abs().reshape(num_envs, -1)
        error = torch.zeros(num_envs, dtype=torch.float64) if not delta.shape[1] else delta.amax(-1)
        error = torch.nan_to_num(error, nan=math.inf, posinf=math.inf, neginf=math.inf)
        maximum = torch.maximum(maximum, error)
        if error.max() > 0.:
            fields[key] = float(error.max())
    return maximum.numpy(), fields


@torch.no_grad()
def real_step(env, actor, policy, obs, first_action=None):
    action = policy(actor, obs['history']) if first_action is None else first_action
    if not torch.isfinite(action).all():
        raise FloatingPointError('Nonfinite action; not sent to simulation')
    obs, reward, done, _ = env.step(action)
    if not torch.isfinite(reward).all():
        raise FloatingPointError('Nonfinite real reward')
    reward_state = getattr(env.unwrapped, '_split_walk_rewards', None)
    if reward_state is not None and hasattr(reward_state, 'record'):
        from ct_mjlab.supermimic.analytic_reward import differentiable_reward
        record = reward_state.record
        _, reproduced = differentiable_reward(record['features'], record['action'],
            record['previous'], record['previous_previous'], record['context'])
        torch.testing.assert_close(reproduced, reward, atol=2e-7, rtol=2e-5)
    return obs, reward.detach().cpu().numpy(), done.bool().detach().cpu().numpy(), action.detach().cpu()


def returns(rewards, done, gamma):
    alive, total, result = np.ones(rewards.shape[1], dtype=bool), np.zeros(rewards.shape[1]), {}
    for step in range(len(rewards)):
        total += gamma ** step * alive * rewards[step]
        alive &= ~done[step]
        if step + 1 in HORIZONS:
            result[step + 1] = dict(value=total.copy(), survived=alive.copy())
    return result


def source_manifest():
    import ct_mjlab
    root = Path(ct_mjlab.__file__).resolve().parent
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*.py'))}


def state_digest(*modules):
    result = hashlib.sha256()
    for module in modules:
        for key, value in sorted(module.state_dict().items()):
            result.update(key.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run_probe(args):
    from ct_mjlab.supermimic.__main__ import make_env
    from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, asset_digest
    from ct_mjlab.supermimic.networks import WorldModel, make_actor, act
    from ct_mjlab.envs.chuanliantui_split.runner import contracts

    if args.output.exists():
        raise FileExistsError(args.output)
    initial_sources = source_manifest()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('algorithm_contract') != ALGORITHM_CONTRACT or saved.get('asset_sha256') != asset_digest():
        raise ValueError('Checkpoint algorithm/XML mismatch')
    training_manifest = args.checkpoint.parent / 'source_sha256.json'
    if not training_manifest.is_file():
        raise FileNotFoundError('Training source manifest is required for this probe')
    training_sources = json.loads(training_manifest.read_text())
    if any(initial_sources.get(key) != value for key, value in training_sources.items()):
        raise ValueError('Current ct_mjlab source differs from this checkpoint training source')
    uses_analytic = 'analytic-reward' in ALGORITHM_CONTRACT
    def build_env():
        created, _ = make_env(args, play=True)
        try:
            command = created.unwrapped.command_manager.get_term('standup')
            if saved.get('infos') != contracts(command.cfg):
                raise ValueError('Checkpoint task contract mismatch')
            command.load_checkpoint_state(saved['curriculum'])
            command.speed_curriculum_frozen = True
            if uses_analytic:
                from ct_mjlab.envs.chuanliantui_split.rewards import WalkRewardState
                from ct_mjlab.supermimic.analytic_reward import capture
                class ProbeRewardState(WalkRewardState):
                    def compute(self, raw):
                        if self.step != raw.common_step_counter:
                            self.record = capture(raw, self)
                        return super().compute(raw)
                # Observe the original reward at its original physical instant;
                # the inherited formula, control and physical parameters stay unchanged.
                created.unwrapped._split_walk_rewards = ProbeRewardState(created.unwrapped)
            return created
        except Exception:
            created.close()
            raise
    env = build_env()
    raw, count, obs = env.unwrapped, args.num_envs, None
    command = raw.command_manager.get_term('standup')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        actor, world = make_actor(args.device), WorldModel().to(args.device)
        actor.load_state_dict(saved['actor']); world.load_state_dict(saved['world'])
        for module in (actor, world):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        weights_before = state_digest(actor, world)
        runtime = dict(policy_dt_s=raw.step_dt, physics_dt_s=raw.physics_dt,
            action_scales=dict(leg_position_rad=float(raw.action_manager.get_term('pd').cfg.leg_position_scale),
                               wheel_velocity_rad_s=float(raw.action_manager.get_term('pd').cfg.wheel_velocity_scale)))
        environment_builds = 1
        def release_env():
            nonlocal env, raw, command, obs
            if str(args.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
            previous_env = env
            wrapper_reference = weakref.ref(previous_env) if previous_env is not None else None
            raw_reference = weakref.ref(raw) if raw is not None else None
            env = raw = command = obs = None
            if previous_env is not None:
                previous_env.close()
            del previous_env
            gc.collect()  # Managers hold cycles pointing to their environment.
            if ((wrapper_reference is not None and wrapper_reference() is not None)
                    or (raw_reference is not None and raw_reference() is not None)):
                raise RuntimeError('Previous environment is still retained after close/GC; refusing to build another solver')
            if str(args.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        prefix_steps = [round(seconds / raw.step_dt) for seconds in args.prefix_seconds]
        if any(abs(steps * raw.step_dt - seconds) > 1e-7 for steps, seconds in zip(prefix_steps, args.prefix_seconds)):
            raise ValueError('Prefix times must be integer policy ticks')

        obs, _ = raw.reset(seed=args.seed)
        reference_states = [tensor_signature(raw, obs)]
        reference_obs, reference_records, actions, rewards, dones = [], [], [], [], []
        for _ in range(max(prefix_steps) + max(HORIZONS)):
            reference_obs.append({key: value.detach().clone() for key, value in obs.items()})
            obs, reward, done, action = real_step(env, actor, act, obs)
            if uses_analytic:
                reference_records.append({key: value.detach().clone() for key, value in raw._split_walk_rewards.record.items()})
            actions.append(action); rewards.append(reward); dones.append(done)
            reference_states.append(tensor_signature(raw, obs))
        rewards, dones = np.stack(rewards), np.stack(dones)
        extra = {key: torch.stack([record[key] for record in reference_records]).cpu().numpy()
                 for key in ('features', 'context', 'previous', 'previous_previous')} if uses_analytic else {}
        np.savez_compressed(args.output / 'reference.npz', actions=torch.stack(actions).numpy(),
                            rewards=rewards, done=dones,
                            history=np.stack([item['history'].cpu().numpy() for item in reference_obs]),
                            privileged=np.stack([item['critic'].cpu().numpy() for item in reference_obs]), **extra)

        def branch(prefix, delta, verify_post=False):
            nonlocal env, raw, command, environment_builds
            if args.fresh_env_per_branch:
                release_env()
                env = build_env()
                raw = env.unwrapped
                command = raw.command_manager.get_term('standup')
                environment_builds += 1
            obs, _ = raw.reset(seed=args.seed)
            maxima, fields = signature_difference(tensor_signature(raw, obs), reference_states[0], count, args.pair_atol)
            for step in range(prefix):
                obs, reward, done, _ = real_step(env, actor, act, obs, actions[step].to(args.device))
                observed = tensor_signature(raw, obs, full=step + 1 == prefix)
                expected = {key: reference_states[step + 1][key] for key in observed}
                error, changed = signature_difference(observed, expected, count, args.pair_atol)
                error = np.maximum(error, np.abs(reward - rewards[step]))
                error = np.where(done == dones[step], error, np.inf)
                maxima = np.maximum(maxima, error)
                for key, value in changed.items():
                    fields[key] = max(fields.get(key, 0.), value)
            trajectory, terminal, post_error = [], [], np.zeros(count)
            first = actions[prefix].to(args.device) + delta
            for step in range(max(HORIZONS)):
                obs, reward, done, _ = real_step(env, actor, act, obs, first if step == 0 else None)
                trajectory.append(reward); terminal.append(done)
                if verify_post:
                    error, _ = signature_difference(tensor_signature(raw, obs), reference_states[prefix + step + 1], count, args.pair_atol)
                    post_error = np.maximum(post_error, error)
            return dict(prefix_max_abs=maxima, changed_prefix_fields=fields,
                        post_max_abs=post_error, rewards=np.stack(trajectory), done=np.stack(terminal))

        rows, pair_audits, artifacts, stopped_early = [], [], {}, None
        zero = torch.zeros(count, 6, device=args.device)
        for prefix, seconds in zip(prefix_steps, args.prefix_seconds):
            baseline = returns(rewards[prefix:prefix + max(HORIZONS)], dones[prefix:prefix + max(HORIZONS)], args.gamma)
            zero_branch = branch(prefix, zero, verify_post=True)
            zero_returns = returns(zero_branch['rewards'], zero_branch['done'], args.gamma)
            prefix_alive = ~dones[:prefix].any(0)
            zero_reward_error = np.abs(zero_branch['rewards'] - rewards[prefix:prefix + max(HORIZONS)]).max(0)
            zero_done_equal = (zero_branch['done'] == dones[prefix:prefix + max(HORIZONS)]).all(0)
            zero_ok = ((zero_branch['prefix_max_abs'] <= args.pair_atol)
                       & (zero_branch['post_max_abs'] <= args.pair_atol)
                       & (zero_reward_error <= args.pair_atol) & zero_done_equal)
            initial = reference_obs[prefix]
            commands = torch.stack([reference_states[prefix + step + 1]['obs/actor'][:, 6:9]
                                    for step in range(max(HORIZONS))], 1).to(args.device)
            real_done = torch.as_tensor(dones[prefix:prefix + max(HORIZONS)].T, device=args.device)
            context_arguments = {key: reference_records[prefix][key] for key in ('previous', 'previous_previous')} if uses_analytic else {}
            if uses_analytic:
                context_arguments['reward_context'] = reference_records[prefix]['context']
            predicted = model_gradients(actor, world, initial['history'], initial['critic'],
                                        commands, real_done, act, args.gamma, **context_arguments)
            audit = dict(prefix_seconds=seconds, prefix_steps=prefix,
                         prefix_alive_envs=np.flatnonzero(prefix_alive).tolist(),
                         zero_perturb_prefix_max_abs=zero_branch['prefix_max_abs'].tolist(),
                         zero_perturb_post_max_abs=zero_branch['post_max_abs'].tolist(),
                         zero_perturb_reward_max_abs=zero_reward_error.tolist(),
                         zero_perturb_done_equal=zero_done_equal.tolist(),
                         zero_perturb_consistent_envs=np.flatnonzero(zero_ok).tolist(),
                         observed_state_fields=sorted(reference_states[0]),
                         changed_prefix_fields=zero_branch['changed_prefix_fields'])
            pair_audits.append(audit)
            if args.fresh_env_per_branch and not (zero_ok & prefix_alive).any():
                stopped_early = 'Freshly constructed zero-perturb branch did not establish a paired transfer; '
                stopped_early += 'no direction conclusion and no additional perturbation branches.'
                print(json.dumps(dict(prefix_seconds=seconds, stopped_early=stopped_early)), flush=True)
                break
            for epsilon in args.epsilons:
                for direction, motor in enumerate(MOTORS):
                    perturbation = zero.clone(); perturbation[:, direction] = epsilon
                    plus, minus = branch(prefix, perturbation), branch(prefix, -perturbation)
                    plus_returns = returns(plus['rewards'], plus['done'], args.gamma)
                    minus_returns = returns(minus['rewards'], minus['done'], args.gamma)
                    pair_ok = (zero_ok & (plus['prefix_max_abs'] <= args.pair_atol)
                               & (minus['prefix_max_abs'] <= args.pair_atol) & prefix_alive)
                    key = f't{prefix}_e{epsilon:g}_motor{direction}'
                    artifacts[key + '_rewards_plus'] = plus['rewards']
                    artifacts[key + '_rewards_minus'] = minus['rewards']
                    artifacts[key + '_done_plus'] = plus['done']
                    artifacts[key + '_done_minus'] = minus['done']
                    for horizon in HORIZONS:
                        context_ok = np.ones(count, dtype=bool)
                        if uses_analytic:
                            flags = torch.stack([record['context'][:, 3:7] for record in reference_records[prefix:prefix+horizon]])
                            context_ok = ((flags == flags[:1]).all(0).all(-1)
                                          & (reference_records[prefix]['context'][:, 5] < .5)).cpu().numpy()
                        valid_pair = pair_ok & context_ok
                        numerical = (plus_returns[horizon]['value'] - minus_returns[horizon]['value']) / (2 * epsilon)
                        model = predicted[horizon]['expected_gradient'][:, direction]
                        plain = predicted[horizon]['plain_gradient'][:, direction]
                        noise = np.abs(zero_returns[horizon]['value'] - baseline[horizon]['value'])
                        threshold = np.maximum(args.gradient_floor, 5 * noise / (2 * epsilon))
                        identifiable = valid_pair & (np.abs(numerical) > threshold)
                        agree = (np.sign(model) == np.sign(numerical)) & (np.abs(model) > args.gradient_floor)
                        smooth = plus_returns[horizon]['survived'] & minus_returns[horizon]['survived']
                        eligible = identifiable & smooth
                        row = dict(prefix_seconds=seconds, horizon=horizon, motor=motor, epsilon=epsilon,
                            paired_prefix_envs=int(valid_pair.sum()), identifiable_envs=int(identifiable.sum()),
                            smooth_identifiable_envs=int(eligible.sum()),
                            smooth_sign_agreement=float(agree[eligible].mean()) if eligible.any() else None,
                            model_expected_gradient=model.tolist(), model_plain_reward_gradient=plain.tolist(),
                            real_central_difference=numerical.tolist(), finite_difference_floor=threshold.tolist(),
                            eligible_pair=valid_pair.tolist(), identifiable=identifiable.tolist(),
                            model_context_consistent=context_ok.tolist(),
                            survived_both_branches=smooth.tolist(),
                            sign_agrees=[bool(agree[index]) if identifiable[index] else None for index in range(count)],
                            real_return_plus=plus_returns[horizon]['value'].tolist(),
                            real_return_minus=minus_returns[horizon]['value'].tolist(),
                            real_return_baseline=baseline[horizon]['value'].tolist(),
                            predicted_return=predicted[horizon]['expected'].tolist(),
                            predicted_plain_return=predicted[horizon]['plain'].tolist(),
                            prefix_plus_max_abs=plus['prefix_max_abs'].tolist(),
                            prefix_minus_max_abs=minus['prefix_max_abs'].tolist())
                        rows.append(row)
            print(json.dumps(dict(prefix_seconds=seconds, completed=True,
                                  prefix_alive=int(prefix_alive.sum()), zero_perturb_consistent=int(zero_ok.sum()),
                                  elapsed_s=time.monotonic()-started)), flush=True)
        if state_digest(actor, world) != weights_before:
            raise AssertionError('Actor/world parameters or running statistics changed')
        if source_manifest() != initial_sources:
            raise AssertionError('Source changed while probe was running')
        np.savez_compressed(args.output / 'branches.npz', **artifacts)
        verified_pairs = sum(row['paired_prefix_envs'] for row in rows)
        status = 'paired_gradient_measurement' if verified_pairs else 'paired_unverified_no_conclusion'
        report = dict(status=status, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
            asset_sha256=asset_digest(), algorithm_contract=ALGORITHM_CONTRACT,
            source_sha256=initial_sources, probe_sha256=digest(__file__),
            optimizer_steps=0, models_and_running_statistics_unchanged=True,
            num_envs=count, seed=args.seed, **runtime,
            gamma=args.gamma, nominal=True, horizons=list(HORIZONS), epsilons=args.epsilons,
            elapsed_s=time.monotonic()-started,
            fresh_env_per_branch=args.fresh_env_per_branch, environment_builds=environment_builds,
            environments_alive_at_once=1, previous_environment_collection_verified=args.fresh_env_per_branch,
            stopped_early=stopped_early,
            method=('Every branch constructs a new complete nominal mjlab environment after closing and releasing its predecessor. '
                    if args.fresh_env_per_branch else 'Same nominal mjlab batch is reused. ')
                   + 'Native seeded reset and exact recorded prefix actions are applied. '
                   'Observed prefix state plus zero-perturb successor transitions are checked. '
                   'This is not a complete mid-step physics/control snapshot restore.',
            excluded_state_audit_fields=dict(command_diagnostic_statistics=list(DIAGNOSTIC_ONLY_COMMAND_TENSORS),
                command_static_index_metadata='*_adr; joint-address caches are not per-environment dynamic state'),
            objective=('Discounted original analytic reward of predicted physical features' if uses_analytic else 'Discounted learned scalar reward')
                      + ' with predicted survival; no critic. Plain reward gradients also reported.',
            analytic_reward_reproduced_at_original_reward_time=uses_analytic,
            scope='Early pre-walk-activation prefixes; does not establish walking-policy gradient accuracy or behavior acceptance.',
            pair_atol=args.pair_atol, pair_audits=pair_audits, directions=rows)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(output=str(args.output), status=status,
                              optimizer_steps=0, direction_rows=len(rows))), flush=True)
        return verified_pairs > 0
    finally:
        if env is not None:
            env.close()


def selfcheck(output):
    class ToyWorld(nn.Module):
        def step(self, history, privileged, action, command):
            frame = history[:, -25:].clone()
            frame[:, 0:6] = frame[:, 0:6] + .2 * action
            frame[:, 6:9] = command
            frame[:, 19:25] = action
            history = torch.cat((history[:, 25:], frame), -1)
            reward = -((action - .3) ** 2).sum(-1) - .1 * frame[:, :6].square().sum(-1)
            return history, privileged, reward, torch.full_like(reward, -2.)
    def policy(actor, history):
        return .1 * history[:, -25:-19] + .2 * history[:, -6:]
    torch.manual_seed(3)
    history, privileged = torch.randn(3, 125, dtype=torch.float64), torch.zeros(3, 71, dtype=torch.float64)
    commands, done = torch.zeros(3, 8, 3, dtype=torch.float64), torch.zeros(3, 8, dtype=torch.bool)
    actor, world, gamma = nn.Identity(), ToyWorld(), .99
    result = model_gradients(actor, world, history, privileged, commands, done, policy, gamma)
    first = policy(actor, history)
    maximum = 0.
    for direction in range(6):
        delta = torch.zeros_like(first); delta[:, direction] = .001
        plus = model_rollout(actor, world, history, privileged, commands, done, first + delta, policy, gamma)
        minus = model_rollout(actor, world, history, privileged, commands, done, first - delta, policy, gamma)
        for horizon in HORIZONS:
            for name in ('plain', 'expected'):
                numerical = ((plus[horizon][name] - minus[horizon][name]) / .002).numpy()
                gradient = result[horizon][name + '_gradient'][:, direction]
                np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-7)
                maximum = max(maximum, float(np.abs(gradient - numerical).max()))
    rewards = np.ones((8, 2)); terminal = np.zeros((8, 2), dtype=bool); terminal[1, 0] = True
    actual = returns(rewards, terminal, .9)
    assert math.isclose(actual[8]['value'][0], 1.9)
    assert math.isclose(actual[8]['value'][1], sum(.9 ** step for step in range(8)))
    difference, _ = signature_difference({'state': torch.zeros(2, 3)}, {'state': torch.ones(2, 3)}, 2, 1e-6)
    assert np.array_equal(difference, np.ones(2))
    # Exercise the actual v3 world.step API and analytic reward on CPU. These
    # are untrained random weights; no environment, capture callback or optimizer.
    from ct_mjlab.supermimic.networks import WorldModel
    analytic_world = WorldModel().double()
    for parameter in analytic_world.parameters():
        parameter.requires_grad_(False)
    context = torch.zeros(3, 21, dtype=torch.float64)
    context[:, 2] = .26; context[:, 6] = 1.
    context[:, 7:15] = torch.tensor([-math.inf, math.inf] * 4)
    context[:, 15:19] = torch.tensor([-.77, .12, -.12, .77])
    context[:, 19] = .02; context[:, 20] = .01
    previous, previous_previous = torch.full_like(first, .03), torch.full_like(first, -.02)
    arguments = dict(reward_context=context, previous=previous, previous_previous=previous_previous)
    analytic_result = model_gradients(actor, analytic_world, history, privileged, commands, done, policy, gamma, **arguments)
    analytic_maximum = 0.
    for direction in range(6):
        delta = torch.zeros_like(first); delta[:, direction] = .001
        plus = model_rollout(actor, analytic_world, history, privileged, commands, done, first + delta, policy, gamma, **arguments)
        minus = model_rollout(actor, analytic_world, history, privileged, commands, done, first - delta, policy, gamma, **arguments)
        for horizon in HORIZONS:
            for name in ('plain', 'expected'):
                numerical = ((plus[horizon][name] - minus[horizon][name]) / .002).numpy()
                gradient = analytic_result[horizon][name + '_gradient'][:, direction]
                np.testing.assert_allclose(gradient, numerical, atol=1e-7, rtol=1e-5)
                analytic_maximum = max(analytic_maximum, float(np.abs(gradient - numerical).max()))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(device='cpu', simulator_steps=0, optimizer_steps=0,
                  model_first_action_derivatives_match_finite_difference=True,
                  finite_difference_max_abs_error=maximum, first_episode_return_mask_correct=True,
                  observed_state_mismatch_detected=True,
                  actual_v3_analytic_world_api_checked=True,
                  actual_v3_finite_difference_max_abs_error=analytic_maximum,
                  scope='CPU helper checks; mjlab real paired branches have not been run')
    (output / 'selfcheck.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--prefix-seconds', type=float, nargs='+', default=[.2, .5, .8])
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.05])
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--pair-atol', type=float, default=1e-6)
    parser.add_argument('--gradient-floor', type=float, default=1e-6)
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--fresh-env-per-branch', action='store_true',
                        help='Close/release each old environment and construct a new complete one for every branch')
    args = parser.parse_args()
    if args.num_envs < 1 or not 0. < args.gamma <= 1. or min(args.pair_atol, args.gradient_floor) <= 0.:
        parser.error('Positive environment count, tolerances and gamma in (0,1] required')
    if (any(not math.isfinite(value) or value <= 0. for value in args.epsilons)
            or any(not math.isfinite(value) or value < 0. for value in args.prefix_seconds)):
        parser.error('Finite nonnegative prefixes and finite positive perturbations required')
    args.task = 'chuanliantui_walk'
    torch.set_num_threads(1)
    if args.selfcheck:
        selfcheck(args.output)
    elif args.checkpoint is None:
        parser.error('--checkpoint is required for real simulation')
    else:
        if not run_probe(args):
            raise SystemExit(2)  # A finished diagnostic is not a verified paired measurement.


if __name__ == '__main__':
    main()
