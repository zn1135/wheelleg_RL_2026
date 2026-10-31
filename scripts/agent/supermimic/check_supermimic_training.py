#!/usr/bin/env python3
"""Read-only CPU audit of a SuperMimic run; never starts training or simulation."""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import torch


NETWORKS = ('actor', 'world', 'critic')
OPTIMIZERS = ('opt_actor', 'opt_encoder', 'opt_world', 'opt_critic')
V3_CONTRACT = 'chuanliantui-supermimic-physical-features-bptt-analytic-reward-v3'
V5_CONTRACT = 'chuanliantui-supermimic-goal-isolated-physical-features-bptt-analytic-reward-v5'
V6_CONTRACT = 'chuanliantui-supermimic-goal-isolated-bounded-value-bptt-analytic-reward-v6'
V7_CONTRACT = 'chuanliantui-supermimic-worldfit4-goal-isolated-bounded-value-bptt-analytic-reward-v7'
V8_CONTRACT = 'chuanliantui-supermimic-contact-target-cap5-worldfit4-goal-isolated-bounded-value-bptt-analytic-reward-v8'
V9_CONTRACT = 'chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9'
V10_CONTRACT = 'chuanliantui-supermimic-canonical-physics-contact-kb-fixed-scale-nominal-mix-v10'
V11_CONTRACT = 'chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return f'nonfinite:{value}'
    return value


def finite_tree(value, path='root'):
    problems = []
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            problems.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            problems.extend(finite_tree(item, f'{path}/{key}'))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            problems.extend(finite_tree(item, f'{path}/{index}'))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            problems.append(path)
    return problems


def equal(first, second):
    if isinstance(first, torch.Tensor):
        return (isinstance(second, torch.Tensor) and first.dtype == second.dtype
                and first.shape == second.shape and torch.equal(first, second))
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)):
        return isinstance(second, type(first)) and len(first) == len(second) and all(equal(a, b) for a, b in zip(first, second))
    return first == second


def optimizer_steps(optimizer):
    identifiers = [key for group in optimizer['param_groups'] for key in group['params']]
    if len(identifiers) != len(set(identifiers)) or not set(optimizer['state']).issubset(identifiers):
        raise ValueError('Optimizer parameter identities are invalid')
    values = []
    for state in optimizer['state'].values():
        step = float(state['step'])
        if not math.isfinite(step) or step < 0 or step != int(step):
            raise ValueError('Optimizer step is not a finite nonnegative integer')
        if not {'exp_avg', 'exp_avg_sq', 'step'}.issubset(state):
            raise ValueError('Adam state is incomplete')
        values.append(int(step))
    if values and (len(values) != len(identifiers) or len(set(values)) != 1):
        raise ValueError('Optimizer slots do not have one consistent update count')
    return values[0] if values else 0


def load_checkpoint(path):
    before = path.stat()
    state = torch.load(path, map_location='cpu', weights_only=False)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Checkpoint changed during reading: {path}')
    return state


def parameters(state):
    # The three current MLPs contain Linear/LayerNorm weight and bias parameters;
    # mean/variance/count tensors are running statistics, not trainable weights.
    return {key: value for key, value in state.items() if key.endswith(('.weight', '.bias'))}


def parameter_change(initial, current, prefix=''):
    first = {key: value for key, value in parameters(initial).items() if key.startswith(prefix)}
    second = {key: value for key, value in parameters(current).items() if key.startswith(prefix)}
    if first.keys() != second.keys() or any(first[key].shape != second[key].shape for key in first):
        raise ValueError('Network parameter keys/shapes changed')
    changed = [key for key in first if not torch.equal(first[key], second[key])]
    maximum = max((float((first[key].double() - second[key].double()).abs().max()) for key in changed), default=0.)
    return dict(parameter_tensors=len(first), changed_tensors=len(changed), changed_names=changed,
                max_abs_change=maximum if math.isfinite(maximum) else None)


def source_contract(snapshot):
    tree = ast.parse((snapshot / 'supermimic/learner.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(name, ast.Name) and name.id == 'ALGORITHM_CONTRACT' for name in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Archived source does not define ALGORITHM_CONTRACT')


def flag(command, name):
    if name not in command:
        return None
    return command[command.index(name) + 1]


def audit(args, report):
    run = args.run.resolve()
    checks = report['checks']
    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    config = json.loads((run / 'config.json').read_text())
    check('finite_config', not finite_tree(config), finite_tree(config))
    check('positive_explicit_budget', type(config.get('seconds')) in (int, float)
          and math.isfinite(config['seconds']) and config['seconds'] > 0)
    check('training_walk_config', config.get('mode') == 'train' and config.get('task') == 'chuanliantui_walk')
    actor_lr = config.get('actor_lr',3e-4)
    check('positive_finite_actor_lr', type(actor_lr) in (int,float) and math.isfinite(actor_lr) and actor_lr > 0)
    check('config_log_directory', Path(config['log_dir']).resolve() == run)
    report['config'] = config

    snapshot = run / 'snapshot/ct_mjlab'
    manifest = json.loads((run / 'source_sha256.json').read_text())
    safe_manifest = (isinstance(manifest, dict) and bool(manifest) and all(
        isinstance(key, str) and not Path(key).is_absolute() and '..' not in Path(key).parts
        and isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for key, value in manifest.items()))
    check('valid_source_manifest', safe_manifest)
    if not safe_manifest:
        raise ValueError('Cannot read unsafe/empty source manifest')
    archived_files = {str(path.relative_to(snapshot)) for path in snapshot.rglob('*.py')}
    check('snapshot_file_set', archived_files == set(manifest),
          dict(extra=sorted(archived_files - set(manifest)), missing=sorted(set(manifest) - archived_files)))
    snapshot_mismatches = [key for key, value in manifest.items() if not (snapshot / key).is_file() or sha256(snapshot / key) != value]
    check('snapshot_file_hashes', not snapshot_mismatches, snapshot_mismatches)
    import ct_mjlab
    live = Path(ct_mjlab.__file__).resolve().parent
    live_mismatches = [key for key, value in manifest.items() if not (live / key).is_file() or sha256(live / key) != value]
    check('current_source_matches_run', not live_mismatches, live_mismatches)
    report['source'] = dict(manifest_sha256=sha256(run / 'source_sha256.json'),
                            archived_file_count=len(archived_files), current_source_root=str(live))

    initial_path = run / 'initial.pt'
    initial = load_checkpoint(initial_path)
    candidates = sorted(run.glob('checkpoint_*.pt'), key=lambda path: int(re.fullmatch(r'checkpoint_(\d+)\.pt', path.name).group(1)))
    selected_path = run / 'final.pt' if (run / 'final.pt').is_file() else candidates[-1] if candidates else initial_path
    selected = initial if selected_path == initial_path else load_checkpoint(selected_path)
    check('checkpoint_counters_valid', all(type(state[key]) is int and state[key] >= 0
          for state in (initial, selected) for key in ('iteration', 'transitions')))
    report['selected_checkpoint'] = dict(path=str(selected_path), sha256=sha256(selected_path),
        iteration=selected['iteration'], transitions=selected['transitions'],
        immutable_selection='Only initial.pt, checkpoint_N.pt or atomically published final.pt; .tmp is ignored')
    if selected_path.name.startswith('checkpoint_'):
        check('checkpoint_filename_iteration', int(selected_path.stem.split('_')[1]) == selected['iteration'])
    check('checkpoint_configs_match', initial['config'] == config and selected['config'] == config)
    archived_algorithm = source_contract(snapshot)
    check('algorithm_matches_archived_source', initial['algorithm_contract'] == selected['algorithm_contract'] == archived_algorithm)
    report['algorithm_contract'] = archived_algorithm
    v8 = archived_algorithm == V8_CONTRACT
    v11 = archived_algorithm == V11_CONTRACT
    v10 = archived_algorithm in (V10_CONTRACT, V11_CONTRACT)
    v9 = archived_algorithm in (V9_CONTRACT, V10_CONTRACT, V11_CONTRACT)
    v7 = archived_algorithm in (V7_CONTRACT, V8_CONTRACT, V9_CONTRACT, V10_CONTRACT, V11_CONTRACT)
    v6 = archived_algorithm in (V6_CONTRACT, V7_CONTRACT, V8_CONTRACT, V9_CONTRACT, V10_CONTRACT, V11_CONTRACT)  # Shared bounded-value objective.
    v5 = archived_algorithm in (V5_CONTRACT, V6_CONTRACT, V7_CONTRACT, V8_CONTRACT, V9_CONTRACT, V10_CONTRACT, V11_CONTRACT)
    if v11:
        from ct_mjlab.supermimic.learner import WORLD_LOSS_RECIPE
        check('v11_fixed_world_loss_recipe', config.get('world_loss_recipe') == WORLD_LOSS_RECIPE)
    if v10:
        from ct_mjlab.supermimic.__main__ import CONTACT_TRAINING_RECIPE
        check('v10_real_nominal_contact_recipe', config.get('contact_training_recipe') == CONTACT_TRAINING_RECIPE)
        expected_contact = dict(physics_dt_s=.002,refsafe_enabled=True,contact_override=False,
            explicit_contact_pairs=0,common_geom_solimp=torch.tensor((.9,.95,.001,.5,2.)).tolist(),
            common_dmax=float(torch.tensor(.95)),refsafe_min_timeconst_s=.004,
            model_contact_encoding='reference_K_B_before_dmax',robot_priority=1,terrain_priority=0,
            geom_solmix=1.,terrain_solref=[.02,1.],robot_contact_solref_shared_per_environment=True)
        check('v10_contact_encoding_assumptions', config.get('contact_encoding_assumptions') == expected_contact)
    world_steps_per_update = 4 if v7 else 1
    if v7:
        check('v7_fixed_world_steps_per_update', config.get('world_steps_per_update') == 4)
        check('v7_complete_checkpoint_groups', not initial.get('interrupted',False)
              and not selected.get('interrupted',False))
        check('v7_complete_world_groups', all(state['world_updates'] % 4 == 0 for state in (initial, selected)))
    if v5:
        check('v5_initialization_config_fields', {'resume', 'init_actor', 'initialization'}.issubset(config))
        warmup = config.get('policy_warmup_updates')
        if type(warmup) is not int or warmup < 0:
            raise ValueError('v5 requires a nonnegative integer policy_warmup_updates')
        if not all(type(state.get('world_updates')) is int and state['world_updates'] >= 0
                   for state in (initial, selected)):
            raise ValueError('v5 checkpoints require cumulative integer world_updates')
        check('checkpoint_initialization_matches_config', initial.get('initialization')
              == selected.get('initialization') == config.get('initialization'))
        report['selected_checkpoint']['world_updates'] = selected['world_updates']
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    xml_sha = sha256(MODEL_XML)
    check('xml_hashes_match', initial['asset_sha256'] == selected['asset_sha256'] == xml_sha)
    report['xml'] = dict(path=str(MODEL_XML), sha256=xml_sha)
    from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
    from ct_mjlab.envs.chuanliantui_split.runner import contracts
    from ct_mjlab.envs.chuanliantui_split.mdp import validate_walk_speed_state
    command = command_cfg('walk')  # Configuration only; no environment or model is constructed.
    expected_contracts = contracts(command)
    check('task_contracts_match', initial['infos'] == selected['infos'] == expected_contracts)
    for label, state in (('initial', initial), ('selected', selected)):
        groups = state['opt_actor']['param_groups']
        check(label + '_actor_lr_matches_config', bool(groups) and all(
            type(group.get('lr')) in (int,float) and math.isfinite(group['lr'])
            and group['lr'] > 0 and group['lr'] == actor_lr for group in groups))
        check(label + '_curriculum_unlocked', state['curriculum'].get('unlocked') is True)
        validate_walk_speed_state(state['curriculum']['speed_curriculum'], command)
        for name in NETWORKS + OPTIMIZERS:
            problems = finite_tree(state[name], name)
            check(f'{label}_{name}_finite', not problems, problems)
    initial_steps = {name: optimizer_steps(initial[name]) for name in OPTIMIZERS}
    selected_steps = {name: optimizer_steps(selected[name]) for name in OPTIMIZERS}
    deltas = {name: selected_steps[name] - initial_steps[name] for name in OPTIMIZERS}
    report['optimizer_steps'] = dict(initial=initial_steps, selected=selected_steps, added=deltas)
    check('optimizer_steps_nonnegative', all(value >= 0 for value in deltas.values()))

    resume = config.get('resume')
    init_actor = config.get('init_actor')
    if v5:
        check('initialization_flags_exclusive', not (resume and init_actor))
        if resume and init_actor:
            raise ValueError('v5 resume and init_actor are mutually exclusive')
    if resume is None:
        check('fresh_initial_counters', initial['iteration'] == 0 and initial['transitions'] == 0)
        check('fresh_initial_optimizers', all(value == 0 and not initial[name]['state'] for name, value in initial_steps.items()))
        fresh_norms = {f'{name}/{key}': float(value) for name in ('world', 'critic')
                       for key, value in initial[name].items() if key.endswith('.count')}
        check('fresh_initial_running_statistics', all(value == 0 for value in fresh_norms.values()), fresh_norms)
        report['initialization'] = dict(kind='fresh', evidence='No resume path; zero counters, empty optimizer slots and unused running statistics; archived training source retained')
        if v5:
            check('cold_initial_world_updates', initial['world_updates'] == 0)
            if init_actor is None:
                expected_initialization = dict(kind='fresh_from_zero', fresh_from_zero=True)
            else:
                parent_path = Path(init_actor).resolve()
                parent = load_checkpoint(parent_path)
                parent_sha = sha256(parent_path)
                expected_initialization = dict(kind='actor_transfer_world_coldstart',
                    parent_checkpoint=str(parent_path), parent_sha256=parent_sha,
                    parent_algorithm_contract=parent['algorithm_contract'], inherited=['actor'],
                    reset=['world', 'critic', 'opt_actor', 'opt_encoder', 'opt_world', 'opt_critic',
                           'iteration', 'transitions', 'world_updates', 'curriculum', 'replay', 'rng'],
                    fresh_from_zero=False)
                transfer_contracts = ((V3_CONTRACT, V5_CONTRACT, V6_CONTRACT, V7_CONTRACT, V8_CONTRACT, V9_CONTRACT, V10_CONTRACT, V11_CONTRACT) if v11 else
                                      (V3_CONTRACT, V5_CONTRACT, V6_CONTRACT, V7_CONTRACT, V8_CONTRACT, V9_CONTRACT, V10_CONTRACT) if v10 else
                                      (V3_CONTRACT, V5_CONTRACT, V6_CONTRACT, V7_CONTRACT, V8_CONTRACT, V9_CONTRACT) if v9 else
                                      (V3_CONTRACT, V5_CONTRACT, V6_CONTRACT, V7_CONTRACT, V8_CONTRACT) if v8 else
                                      (V3_CONTRACT, V5_CONTRACT, V6_CONTRACT, V7_CONTRACT) if v7 else
                                      (V3_CONTRACT, V5_CONTRACT, V6_CONTRACT) if v6 else (V3_CONTRACT, V5_CONTRACT))
                check('actor_transfer_parent_contract', parent['algorithm_contract'] in transfer_contracts
                      and parent.get('config', {}).get('mode') == 'train'
                      and parent.get('config', {}).get('task') == config['task']
                      and parent.get('asset_sha256') == xml_sha and parent.get('infos') == expected_contracts
                      and all(name in parent for name in NETWORKS + OPTIMIZERS))
                check('actor_transfer_full_actor_restored', equal(initial['actor'], parent.get('actor')))
                check('actor_transfer_parent_actor_finite', not finite_tree(parent.get('actor')))
            check('cold_initialization_provenance', config.get('initialization') == expected_initialization)
            report['initialization'] = dict(**expected_initialization,
                evidence='Actor parent state and hash checked when transferred; empty optimizers and unused world/value statistics prove cold update state, not bitwise independent random initialization')
    else:
        parent_path = Path(resume).resolve()
        parent = load_checkpoint(parent_path)
        check('resume_initial_counters', initial['iteration'] == parent['iteration'] and initial['transitions'] == parent['transitions'])
        check('resume_networks_and_optimizers_restored', all(equal(initial[name], parent[name]) for name in NETWORKS + OPTIMIZERS))
        check('resume_algorithm_xml_contracts', all(initial[key] == parent[key] for key in ('algorithm_contract', 'asset_sha256', 'infos')))
        compatible = ('task', 'num_envs', 'seed', 'horizon', 'window', 'batch', 'buffer_steps')
        if v11:
            compatible += ('world_loss_recipe',)
        if v10:
            compatible += ('contact_training_recipe','contact_encoding_assumptions')
        if v7:
            compatible += ('world_steps_per_update',)
            check('resume_parent_complete_update_groups', not parent.get('interrupted',False))
        if v5:
            compatible += ('policy_warmup_updates',)
            check('resume_initial_world_updates', initial['world_updates'] == parent.get('world_updates'))
            expected_initialization = dict(kind='resume', parent_checkpoint=str(parent_path),
                parent_sha256=sha256(parent_path), parent_algorithm_contract=parent['algorithm_contract'],
                inherited_initialization=parent.get('initialization'), fresh_from_zero=False)
            check('resume_initialization_provenance', config.get('initialization') == expected_initialization)
        check('resume_configuration_compatible', all(config[key] == parent['config'][key] for key in compatible)
              and actor_lr == parent['config'].get('actor_lr',3e-4))
        report['initialization'] = dict(kind='resume', parent=str(parent_path), parent_sha256=sha256(parent_path),
            physical_continuation='Replay and physical episodes restart; RNG need not equal parent after native reset')
        if v5:
            report['initialization'].update(expected_initialization)
    if v5:
        for label, state, steps in (('initial', initial, initial_steps), ('selected', selected, selected_steps)):
            cumulative_policy = max(0, state['world_updates'] - warmup) // world_steps_per_update
            check(label + '_cumulative_world_optimizer_steps', steps['opt_world'] == state['world_updates'])
            check(label + '_cumulative_encoder_optimizer_steps', steps['opt_encoder'] == cumulative_policy)
            check(label + '_cumulative_policy_optimizer_steps', 0 <= steps['opt_actor']
                  == steps['opt_critic'] <= cumulative_policy)

    contents = (run / 'metrics.jsonl').read_bytes()
    lines = contents.splitlines(keepends=True)
    ignored_tail = bool(lines and not lines[-1].endswith(b'\n'))
    if ignored_tail and args.startup:
        lines = lines[:-1]  # A live append may be observed before its newline is published.
    else:
        check('metrics_final_newline', not ignored_tail)
    metrics = [json.loads(line) for line in lines if line.strip()]
    check('metrics_values_finite', not finite_tree(metrics), finite_tree(metrics))
    report['metrics'] = dict(observed_rows=len(metrics), ignored_unpublished_tail=args.startup and ignored_tail)
    previous_iteration, previous_transitions = initial['iteration'], initial['transitions']
    previous_elapsed = previous_updates = 0
    iteration_durations = []
    row_for_checkpoint = None
    observed_policy_updates = []
    actor_metrics = []
    for row in metrics:
        iteration, transitions = row['iteration'], row['transitions']
        elapsed, updates = row['elapsed_s'], row['updates_this_run']
        if v6 and 'actor' in row:
            actor_metric = row['actor']
            # Actor records negative return. Rate [-24,3.5] at dt=.01/gamma=.99
            # gives return [-24,3.5], hence actor loss [-3.5,24]. Critic loss is raw.
            if (type(actor_metric) not in (int, float) or not math.isfinite(actor_metric)
                    or not -3.5 - 1e-5 <= actor_metric <= 24. + 1e-5):
                raise ValueError(f'Actor metric violates the v6 theoretical return bound at iteration {iteration}: {actor_metric}')
            actor_metrics.append(actor_metric)
        if not (type(iteration) is int and iteration == previous_iteration + 1
                and type(transitions) is int and transitions >= previous_transitions
                and (transitions - previous_transitions) % config['num_envs'] == 0
                and transitions - previous_transitions <= 8 * config['num_envs']
                and elapsed > previous_elapsed and type(updates) is int
                and 0 <= updates - previous_updates <= 4):
            raise ValueError(f'Metrics counters/time are inconsistent at iteration {iteration}')
        if v5 and (type(row.get('world_updates')) is not int
                   or row['world_updates'] != initial['world_updates'] + updates * world_steps_per_update):
            raise ValueError(f'Cumulative world updates disagree with run updates at iteration {iteration}')
        if not v5 and updates <= 100 and 'actor' in row:
            raise ValueError('Policy metric appeared before this run completed its 100 model-only updates')
        policy_metrics = {'actor', 'critic', 'actor_grad', 'imagination_eligible_fraction'}
        policy_ready = ((row['world_updates'] - world_steps_per_update >= warmup)
                        if v5 else updates > 100)
        if v5 and (updates == previous_updates or not policy_ready) and policy_metrics.intersection(row):
            raise ValueError(f'Policy metric appeared without an eligible update at iteration {iteration}')
        if updates > previous_updates:
            required = {'world', 'encoder', 'dynamics', 'reward', 'failure',
                        'heldout_dynamics', 'heldout_reward', 'heldout_failure'}
            if archived_algorithm.endswith('analytic-reward-v3') or v5:
                required |= {'features', 'heldout_features'}
            if policy_ready:
                required |= {'actor', 'critic', 'actor_grad'}
                if v5:
                    required.add('imagination_eligible_fraction')
            if not required.issubset(row):
                raise ValueError(f'Missing learning metrics at iteration {iteration}: {sorted(required-set(row))}')
            if v5 and policy_ready:
                eligible = row['imagination_eligible_fraction']
                if type(eligible) not in (int, float) or not math.isfinite(eligible) or not 0 <= eligible <= 1:
                    raise ValueError(f'Invalid imagination eligible fraction at iteration {iteration}')
                if eligible == 0 and row['actor_grad'] != 0:
                    raise ValueError(f'Actor gradient reported for an all-invalid update at iteration {iteration}')
                if iteration <= selected['iteration']:
                    observed_policy_updates.append(eligible > 0)
        iteration_durations.append(elapsed - previous_elapsed)
        if iteration == selected['iteration']:
            row_for_checkpoint = row
        previous_iteration, previous_transitions = iteration, transitions
        previous_elapsed, previous_updates = elapsed, updates
    check('metrics_counters_monotonic', True)
    if v6:
        report['actor_metric_theoretical_bound'] = dict(minimum=-3.5, maximum=24., absolute_float_tolerance=1e-5,
            gamma=.99, policy_dt_s=.01, observed_count=len(actor_metrics),
            observed_min=min(actor_metrics) if actor_metrics else None,
            observed_max=max(actor_metrics) if actor_metrics else None,
            scope='Negative bounded-return objective only; finite values alone do not suffice. Raw critic regression loss has no actor-loss bound.')
    if selected['iteration'] == initial['iteration']:
        checkpoint_updates = 0
        check('initial_checkpoint_transitions', selected['transitions'] == initial['transitions'])
    else:
        check('checkpoint_has_matching_metric', row_for_checkpoint is not None)
        if row_for_checkpoint is None:
            raise ValueError('Selected checkpoint iteration has no published metric')
        check('checkpoint_transitions_match_metric', row_for_checkpoint['transitions'] == selected['transitions'])
        checkpoint_updates = row_for_checkpoint['updates_this_run']
    if v5:
        before_world, after_world = initial['world_updates'], selected['world_updates']
        policy_maximum = (max(0, after_world - warmup) // world_steps_per_update
                          - max(0, before_world - warmup) // world_steps_per_update)
        check('world_updates_match_metrics', after_world - before_world
              == checkpoint_updates * world_steps_per_update == deltas['opt_world'])
        check('encoder_updates_respect_cumulative_warmup', deltas['opt_encoder'] == policy_maximum)
        if row_for_checkpoint is not None:
            check('checkpoint_world_updates_match_metric', row_for_checkpoint['world_updates'] == after_world)
        valid_last = sum(observed_policy_updates)
        invalid_last = len(observed_policy_updates) - valid_last
        check('policy_steps_match_observed_last_updates', valid_last <= deltas['opt_actor'] <= policy_maximum - invalid_last)
        report['world_updates'] = dict(initial=before_world, selected=after_world,
            added=checkpoint_updates * world_steps_per_update, update_groups=checkpoint_updates,
            world_steps_per_update=world_steps_per_update, policy_warmup_updates=warmup)
    else:
        check('world_encoder_updates_match_metrics', deltas['opt_world'] == deltas['opt_encoder'] == checkpoint_updates)
        policy_maximum = max(0, checkpoint_updates - 100)
    check('actor_critic_updates_match', deltas['opt_actor'] == deltas['opt_critic'])
    check('policy_updates_respect_warmup', 0 <= deltas['opt_actor'] <= policy_maximum)
    report['policy_updates'] = dict(maximum_after_warmup=policy_maximum, actual=deltas['opt_actor'],
        skipped=policy_maximum - deltas['opt_actor'],
        skip_scope='Source permits skipping a policy/value update when every sampled context is invalid; metrics retain only the last of up to four updates per iteration')
    if v5:
        report['policy_updates'].update(encoder_updates=deltas['opt_encoder'],
            observed_valid_last_updates=valid_last, observed_invalid_last_updates=invalid_last,
            phase=('warmup_only' if policy_maximum == 0 else 'all_policy_updates_skipped_context_invalid'
                   if deltas['opt_actor'] == 0 else 'partially_skipped_context_invalid'
                   if deltas['opt_actor'] < policy_maximum else 'updated'))

    change = {name: parameter_change(initial[name], selected[name]) for name in NETWORKS}
    change['actor_mlp'] = parameter_change(initial['actor'], selected['actor'], 'mlp.')
    change['encoder'] = parameter_change(initial['actor'], selected['actor'], 'encoder.')
    for name, optimizer in (('world', 'opt_world'), ('encoder', 'opt_encoder'), ('actor_mlp', 'opt_actor'), ('critic', 'opt_critic')):
        check(name + '_parameter_changes_match_updates', bool(change[name]['changed_tensors']) == bool(deltas[optimizer]))
    report['parameter_change'] = change

    exit_record = json.loads(args.exit_record.read_text()) if args.exit_record and args.exit_record.is_file() else None
    report['actual_exit'] = dict(record=str(args.exit_record.resolve()) if args.exit_record else None,
                                present=exit_record is not None, contents=exit_record)
    if exit_record is not None:
        check('exit_record_finite', not finite_tree(exit_record), finite_tree(exit_record))
        check('exit_requested_budget_matches', exit_record['requested_seconds'] == config['seconds'])
        check('exit_run_matches', Path(exit_record['log_dir']).resolve() == run)
        command_line = exit_record['command']
        check('exit_command_is_training', '-m' in command_line and flag(command_line, '-m') == 'ct_mjlab.supermimic' and 'train' in command_line)
        check('exit_command_budget_matches', float(flag(command_line, '--seconds')) == config['seconds'])
        check('exit_command_actor_lr_matches', command_line.count('--actor-lr') <= 1
              and (float(flag(command_line, '--actor-lr')) if flag(command_line, '--actor-lr') is not None else 3e-4) == actor_lr)
        check('exit_command_logdir_matches', Path(flag(command_line, '--log-dir')).resolve() == run)
        check('exit_command_resume_matches', (Path(flag(command_line, '--resume')).resolve() if flag(command_line, '--resume') else None)
              == (Path(resume).resolve() if resume else None))
        if v5:
            check('exit_command_v5_flags_unique', all(command_line.count(name) <= 1
                  for name in ('--resume', '--init-actor', '--policy-warmup-updates')))
            check('exit_command_init_actor_matches', (Path(flag(command_line, '--init-actor')).resolve()
                  if flag(command_line, '--init-actor') else None) == (Path(init_actor).resolve() if init_actor else None))
            check('exit_command_initialization_flags_exclusive', not ('--resume' in command_line and '--init-actor' in command_line))
            check('exit_command_warmup_matches', (int(flag(command_line, '--policy-warmup-updates'))
                  if flag(command_line, '--policy-warmup-updates') is not None else 1000) == warmup)
        explicit_flags = {'--num-envs':'num_envs', '--horizon':'horizon', '--window':'window',
                          '--batch':'batch', '--buffer-steps':'buffer_steps', '--seed':'seed'}
        check('exit_explicit_training_flags_match', all(flag(command_line, name) is None
              or int(flag(command_line, name)) == config[key] for name, key in explicit_flags.items()))
        check('actual_successful_exit', type(exit_record.get('returncode')) is int
              and exit_record['returncode'] == 0 and exit_record.get('final_checkpoint_exists') is True)
        check('exit_final_file_claim_matches', bool(exit_record.get('final_checkpoint_exists')) == (run/'final.pt').is_file())
        check('exit_timeline_valid', datetime.fromisoformat(exit_record['finished_at']) >= datetime.fromisoformat(exit_record['started_at'])
              and exit_record['wall_elapsed_s'] >= exit_record['training_elapsed_s'] >= 0.)
        if metrics:
            check('exit_elapsed_matches_last_metric', math.isclose(exit_record['training_elapsed_s'], metrics[-1]['elapsed_s'], rel_tol=0., abs_tol=1e-6))
            timing_bound = max(iteration_durations)
            difference = exit_record['training_elapsed_s'] - config['seconds']
            check('budget_consistent_with_boundary_checks', abs(difference) <= timing_bound)
            report['budget'] = dict(requested_s=config['seconds'], last_metric_elapsed_s=exit_record['training_elapsed_s'],
                difference_s=difference, measured_max_iteration_s=timing_bound,
                timing_scope='Deadline is checked before each collection/update; elapsed is sampled before metric writing/checkpoint I/O. One observed iteration bounds reporting uncertainty, not an extra training budget.')
    if not args.startup:
        check('actual_exit_record_required_for_completion', exit_record is not None)
        check('final_checkpoint_required_for_completion', selected_path.name == 'final.pt')
        check('complete_metrics_required', bool(metrics))
        if metrics:
            check('final_counters_match_last_metric', selected['iteration'] == metrics[-1]['iteration']
                  and selected['transitions'] == metrics[-1]['transitions'])
            if v5:
                check('final_world_updates_match_last_metric', selected['world_updates'] == metrics[-1]['world_updates'])
    report['all_checks_passed'] = all(item['passed'] for item in checks)
    report['completion_verified'] = bool(not args.startup and report['all_checks_passed'])
    report['status'] = ('startup_evidence_only_completion_unverified' if args.startup and report['all_checks_passed']
        else 'training_completed_checks_passed_behavior_unverified' if report['completion_verified'] else 'training_audit_failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--exit-record', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    report = dict(run=str(args.run.resolve()), startup=args.startup, checks=[], checker_sha256=sha256(__file__),
                  device='cpu', optimizer_steps_performed=0,
                  simulator_steps_performed=0, scope='Artifact/exit/finite-value/source audit only; no convergence or behavior acceptance')
    try:
        audit(args, report)
    except Exception as error:
        report.update(status='training_audit_failed', all_checks_passed=False, completion_verified=False,
                      error=dict(type=type(error).__name__, message=str(error)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), status=report['status'],
                         checks=len(report['checks']), failures=[item['name'] for item in report['checks'] if not item['passed']],
                         error=report.get('error'), selected_checkpoint=report.get('selected_checkpoint')), indent=2))
    if not report['all_checks_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
