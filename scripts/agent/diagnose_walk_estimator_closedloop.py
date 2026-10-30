#!/usr/bin/env python3
"""One bounded saved-trace diagnosis; frozen encoder inference, no fitting or physics."""

import argparse
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from rsl_rl.modules import MLP

import train_walk_estimator_only as training_source
from evaluate_walk_estimator_final import bitwise_equal
from walk_estimator_offline import rebuild_history, file_sha


PARENT_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
CANDIDATE_SHA = 'eeac74b114132b1e2d6781bca7a0864e287897abb4a7dedcb2e2d5aa34da7216'
INPUT_GROUPS = dict(gyro=slice(0, 3), gravity=slice(3, 6), native_command=slice(6, 9),
    leg_position=slice(9, 13), motor_velocity=slice(13, 19), previous_raw_action=slice(19, 25))
PRESERVATION_CHECKS = ('tracking_mae', 'tracking_p95', 'transient_pitch_peak',
    'transient_pitch_total_over5', 'transient_pitch_longest_over5')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def differences(left, right):
    delta = np.asarray(right, dtype=np.float64)-np.asarray(left, dtype=np.float64)
    if not len(delta):
        return None
    return dict(mean_absolute=np.mean(abs(delta), axis=0).tolist(),
        p95_absolute=np.percentile(abs(delta), 95, axis=0).tolist(),
        maximum_absolute=np.max(abs(delta), axis=0).tolist(), signed_mean=delta.mean(axis=0).tolist(),
        parent_mean=np.asarray(left).mean(axis=0).tolist(), candidate_mean=np.asarray(right).mean(axis=0).tolist(),
        parent_std=np.asarray(left).std(axis=0).tolist(), candidate_std=np.asarray(right).std(axis=0).tolist())


def first_difference(left, right, valid, times):
    maximum = abs(np.asarray(right, dtype=float)-np.asarray(left, dtype=float)).reshape(len(times), -1).max(-1)
    any_ids = np.flatnonzero(valid & (maximum > 0.))
    numeric_ids = np.flatnonzero(valid & (maximum > 1e-7))
    return dict(first_nonidentical_s=None if not len(any_ids) else float(times[any_ids[0]]),
        first_difference_gt_1e_7_s=None if not len(numeric_ids) else float(times[numeric_ids[0]]),
        maximum_absolute=None if not valid.any() else float(maximum[valid].max()),
        tolerance_scope='1e-7 is a numerical identity reference only, not a new behavior gate or causal onset.')


def error_metrics(predicted, target):
    if not len(target):
        return None
    error = np.asarray(predicted, dtype=float)-np.asarray(target, dtype=float)
    return dict(mae_m_s=abs(error).mean(0).tolist(), p95_absolute_error_m_s=np.percentile(abs(error), 95, axis=0).tolist(),
        bias_m_s=error.mean(0).tolist(), peak_absolute_error_m_s=abs(error).max(0).tolist())


def load(folder, branch):
    folder = folder.resolve()
    meta = read(folder/'encoder.json')
    receipt = read(folder/'result.json')
    assert receipt['verified'] and meta['parameter_updates'] == 0
    assert meta['feedback_branch'] == branch and meta['controller_unchanged']
    assert meta['parent_checkpoint_sha256'] == PARENT_SHA
    assert meta['candidate_checkpoint_sha256'] == CANDIDATE_SHA
    assert meta['actor_unchanged'] and meta['encoder_unchanged']
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
    assert meta['checkpoint_sha256'] == (PARENT_SHA if branch == 'parent' else CANDIDATE_SHA)
    assert not meta['native_observation_cfg']['noise'] and not meta['native_observation_cfg']['delay']
    assert meta['seconds'] == 30. and meta['policy_dt'] == .01 and meta['dt'] == .002
    with np.load(folder/'encoder.npz', allow_pickle=False) as saved:
        data = {key:saved[key] for key in ('policy_observation', 'native_history', 'policy_valid',
            'policy_time_s', 'native_action', 'encoder_m_s', 'target_com_m_s', 'values')}
        data['fixed'] = {key:saved[key] for key in saved.files if key.startswith('fixed_')}
    frames, history, valid = data['policy_observation'], data['native_history'], data['policy_valid']
    assert frames.shape == (3000, 96, 25) and history.shape == (3000, 96, 125)
    assert valid.shape == (3000, 96) and valid.dtype == np.bool_
    assert np.array_equal(history[valid], rebuild_history(frames)[valid])
    assert np.array_equal(data['policy_time_s'], np.arange(3000)*.01)
    assert np.isfinite(history).all() and np.isfinite(data['values']).all()
    actual = data['native_action']
    lo, hi = meta['layout']['raw_action']
    assert np.array_equal(actual[valid], data['values'][::5, :, lo:hi][valid])
    same_episode = valid[1:] & valid[:-1]
    assert np.array_equal(frames[1:, :, 19:25][same_episode], actual[:-1][same_episode]), \
        'Previous action does not come from this branch at the preceding policy tick'
    data.update(meta=meta, folder=folder, files={str(folder/name):file_sha(folder/name)
        for name in ('encoder.json', 'encoder.npz', 'result.json', 'source.py')})
    return data


def encoder_from(saved, device):
    state = {key.removeprefix('encoder.'):value for key,value in saved['actor_state_dict'].items()
        if key.startswith('encoder.')}
    model = MLP(125, 3, (128, 64), 'elu').to(device)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, training_source.tree_digest(state)


@torch.inference_mode()
def predict(model, history):
    device = next(model.parameters()).device
    flat = history.reshape(-1, 125)
    values = np.empty((len(flat), 3), dtype=np.float32)
    for begin in range(0, len(flat), 8192):
        end = min(begin+8192, len(flat))
        batch = torch.from_numpy(np.array(flat[begin:end], copy=True)).to(device)
        values[begin:end] = (model(batch)/2.).cpu().numpy()
    assert np.isfinite(values).all()
    return values.reshape(*history.shape[:2], 3)


def phase_key(phase):
    return phase['env'], phase['segment'], phase['kind'], phase['start_s'], phase['end_s']


def selected_events(rows):
    order = lambda row:(row['phase']['start_s'], row['env'], row['phase']['end_s'])
    ordered = sorted(rows, key=order)
    failed = [row for row in ordered if row['failed_existing_checks']]
    preserved = [row for row in ordered if row['failed_preservation_checks']]
    late_back = [row for row in ordered if row['phase']['kind'] == 'start'
        and row['phase']['requested_speed'] < 0. and row['completed_parking_phases_before_start'] >= 2]
    failed_late = [row for row in late_back if row['failed_existing_checks']]
    result = {}
    for name, choices in (('earliest_failed_locked_phase', failed),
            ('earliest_preservation_failure', preserved), ('earliest_repeated_stop_backward_start', late_back),
            ('earliest_failed_repeated_stop_backward_start', failed_late)):
        result[name] = None if not choices else choices[0]
    return result


def preservation_failures(compared, screening):
    failed = [key for key,value in compared['checks'].items() if not value]
    baseline = compared['baseline']
    if not baseline['complete']:
        return []
    result = [key for key in failed if key in PRESERVATION_CHECKS]
    if 'complete_phase' in failed:
        result.append('complete_phase')
    metrics = baseline['metrics']
    if 'steady_pitch' in failed and metrics['pitch_abs_max_deg'] <= screening['steady_pitch_abs_max_deg']:
        result.append('steady_pitch')
    if ('no_sustained_swing' in failed and baseline['sustained']['coverage_sufficient']
            and not baseline['sustained']['detected']):
        result.append('no_sustained_swing')
    if ('drift_target' in failed and metrics['world_horizontal_speed_max_m_s']
            <= screening['drift_horizontal_peak_max_m_s']):
        result.append('drift_target')
    return result


def summarize_phase(parent, candidate, phase, compared, predictions):
    env, start, end = phase['env'], phase['start_s'], phase['end_s']
    times = parent['policy_time_s']
    planned = (times >= start-1e-9) & (times < end-1e-9)
    common = planned & parent['policy_valid'][:, env] & candidate['policy_valid'][:, env]
    complete = bool(planned.any() and np.array_equal(planned, common))
    group_changes = {name:differences(parent['policy_observation'][common, env, columns],
        candidate['policy_observation'][common, env, columns]) for name,columns in INPUT_GROUPS.items()}
    errors = {}
    for input_branch, target in (('parent', parent), ('candidate', candidate)):
        for model in ('parent', 'candidate'):
            key = model+'_on_'+input_branch
            errors[key] = error_metrics(predictions[key][common, env], target['target_com_m_s'][common, env])
    # This is an algebraic output decomposition, not a causal attribution of robot motion.
    network_effect = predictions['candidate_on_parent'][common, env]-predictions['parent_on_parent'][common, env]
    history_effect = predictions['candidate_on_candidate'][common, env]-predictions['candidate_on_parent'][common, env]
    actual_effect = predictions['candidate_on_candidate'][common, env]-predictions['parent_on_parent'][common, env]
    assert np.allclose(network_effect+history_effect, actual_effect, rtol=0., atol=2e-7)
    completed_parking = sum(item['kind'] == 'parking' and item['end_s'] <= start+1e-9
        for item in parent['window_lock']['phases'] if item['env'] == env)
    failures = [key for key,value in compared['checks'].items() if not value]
    return dict(seed=parent['meta']['seed'], env=env, phase=phase,
        expected_policy_samples=int(planned.sum()), common_valid_policy_samples=int(common.sum()),
        complete=complete, completed_parking_phases_before_start=completed_parking,
        failed_existing_checks=failures,
        failed_preservation_checks=preservation_failures(compared, parent['screening']),
        unchanged_existing_behavior_comparison=compared,
        input_group_changes_scaled_actor_units=group_changes,
        previous_action_feedback_change=differences(parent['policy_observation'][common, env, 19:25],
            candidate['policy_observation'][common, env, 19:25]),
        actual_action_change=differences(parent['native_action'][common, env], candidate['native_action'][common, env]),
        same_input_estimation_errors_xyz=errors,
        estimator_output_algebra_xyz=dict(network_at_parent_history=differences(np.zeros_like(network_effect), network_effect),
            history_under_candidate_network=differences(np.zeros_like(history_effect), history_effect),
            actual_network_and_history_change=differences(np.zeros_like(actual_effect), actual_effect)),
        interpretation='Same-input errors use that input branch own COM truth. Algebra is descriptive: feedback histories, commands, actions and dynamics co-evolve; it is not an isolated causal estimate.')


def save_event(path, parent, candidate, row, predictions):
    phase, env = row['phase'], row['env']
    start, end = max(0., phase['start_s']-1.5), phase['end_s']
    pt = parent['policy_time_s']
    policy = (pt >= start-1e-9) & (pt < end-1e-9)
    physics_t = (np.arange(15000)+1)*.002
    physics = (physics_t > start+1e-9) & (physics_t <= end+1e-9)
    arrays = dict(policy_time_s=pt[policy], physics_time_s=physics_t[physics],
        input_window_s=np.asarray([start, end]), env=np.asarray(env))
    for branch, data in (('parent', parent), ('candidate', candidate)):
        for name in ('policy_observation', 'native_history', 'policy_valid', 'native_action', 'encoder_m_s', 'target_com_m_s'):
            arrays[branch+'_'+name] = data[name][policy, env]
        arrays[branch+'_behavior_values'] = data['values'][physics, env]
    for name, value in predictions.items():
        arrays[name+'_m_s'] = value[policy, env]
    np.savez_compressed(path, **arrays)


def main(args):
    torch.set_num_threads(1)
    assert len(args.pair) == 3 and {int(row[0]) for row in args.pair} == {43, 48, 49}
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    assert file_sha(args.parent_checkpoint) == PARENT_SHA
    assert file_sha(args.candidate_checkpoint) == CANDIDATE_SHA
    parent_saved = torch.load(args.parent_checkpoint, map_location='cpu', weights_only=False)
    candidate_saved = torch.load(args.candidate_checkpoint, map_location='cpu', weights_only=False)
    assert bitwise_equal(training_source.frozen_parts(parent_saved), training_source.frozen_parts(candidate_saved))
    parent_model, parent_state = encoder_from(parent_saved, args.device)
    candidate_model, candidate_state = encoder_from(candidate_saved, args.device)
    sources = {str(Path(path).resolve()):file_sha(path) for path in
        (__file__, training_source.__file__, inspect.getfile(bitwise_equal), inspect.getfile(rebuild_history), inspect.getfile(MLP))}
    lock = dict(scope='One bounded diagnostic on already-collected paired trajectories. No new fit, simulation, gate or threshold.',
        parent_sha256=PARENT_SHA, candidate_sha256=CANDIDATE_SHA, sources_sha256=sources,
        pair_arguments=args.pair, optimizer_updates=0, physics_steps=0,
        phases='Existing parent-window lock only; earliest failed phase start is not exact failure onset.',
        input_groups={key:[value.start, value.stop] for key,value in INPUT_GROUPS.items()},
        technical_numeric_identity_reference=1e-7)
    write(args.output/'diagnosis-lock.json', lock)
    shutil.copy2(__file__, args.output/'source.py')
    results, all_events, input_files = [], [], {}
    for seed_text, parent_folder, candidate_folder, window_path, comparison_path in sorted(args.pair, key=lambda row:int(row[0])):
        seed = int(seed_text)
        parent, candidate = load(Path(parent_folder), 'parent'), load(Path(candidate_folder), 'candidate')
        windows, comparison = read(window_path), read(comparison_path)
        screening_card_path = Path(windows['execution_card']).resolve()
        assert file_sha(screening_card_path) == windows['execution_card_sha256']
        assert comparison['execution_card_sha256'] == windows['execution_card_sha256']
        screening = read(screening_card_path)['closed_loop_screening']
        assert parent['meta']['seed'] == candidate['meta']['seed'] == windows['seed'] == seed
        assert comparison['verified'] and comparison['optimizer_updates'] == 0
        assert comparison['baseline_files'] == parent['files'] and comparison['candidate_files'] == candidate['files']
        assert windows['baseline_files'] == parent['files']
        assert comparison['pairing']['windows_sha256'] == file_sha(window_path)
        assert comparison['pairing']['terminal_candidate_checkpoint_sha256'] == CANDIDATE_SHA
        assert parent['meta']['protocol_identity'] == candidate['meta']['protocol_identity']
        assert parent['meta']['engine_identity'] == candidate['meta']['engine_identity']
        assert parent['fixed'].keys() == candidate['fixed'].keys()
        assert all(bitwise_equal(parent['fixed'][key], candidate['fixed'][key]) for key in parent['fixed'])
        for data in (parent, candidate):
            data['window_lock'] = windows
            data['screening'] = screening
        input_files.update(parent['files']); input_files.update(candidate['files'])
        input_files.update({str(Path(path).resolve()):file_sha(path) for path in (window_path, comparison_path)})
        input_files[str(screening_card_path)] = windows['execution_card_sha256']
        outputs = dict(parent_on_parent=predict(parent_model, parent['native_history']),
            candidate_on_parent=predict(candidate_model, parent['native_history']),
            parent_on_candidate=predict(parent_model, candidate['native_history']),
            candidate_on_candidate=predict(candidate_model, candidate['native_history']))
        parity = {}
        for branch,data in (('parent', parent), ('candidate', candidate)):
            residual = float(abs(outputs[branch+'_on_'+branch][data['policy_valid']]-data['encoder_m_s'][data['policy_valid']]).max())
            assert residual <= 1e-5, 'Own-branch frozen encoder does not reproduce native feedback'
            parity[branch] = residual
        np.savez_compressed(args.output/f'seed{seed}-four-encoder-predictions.npz', **outputs)
        compared = {phase_key(row['baseline']):row for row in comparison['per_phase']}
        assert len(compared) == len(comparison['per_phase']) == len(windows['phases'])
        rows = [summarize_phase(parent, candidate, phase, compared[phase_key(phase)], outputs) for phase in windows['phases']]
        events = selected_events(rows)
        common = parent['policy_valid'] & candidate['policy_valid']
        first = []
        for env in range(96):
            first.append(dict(env=env,
                current_frame=first_difference(parent['policy_observation'][:, env], candidate['policy_observation'][:, env], common[:, env], parent['policy_time_s']),
                full_history=first_difference(parent['native_history'][:, env], candidate['native_history'][:, env], common[:, env], parent['policy_time_s']),
                native_action=first_difference(parent['native_action'][:, env], candidate['native_action'][:, env], common[:, env], parent['policy_time_s'])))
        result = dict(seed=seed, existing_comparison_passed=comparison['passed'], own_feedback_parity_max_m_s=parity,
            earliest_events=events, first_input_action_difference_by_environment=first, phases=rows)
        write(args.output/f'seed{seed}-diagnosis.json', result)
        saved = set()
        for name,row in events.items():
            if row is None or phase_key(row['phase']) in saved:
                continue
            saved.add(phase_key(row['phase']))
            path = args.output/f'seed{seed}-{name}-env{row["env"]}.npz'
            save_event(path, parent, candidate, row, outputs)
            all_events.append(dict(seed=seed, event=name, env=row['env'], phase=row['phase'], trace=str(path), sha256=file_sha(path)))
        results.append(dict(seed=seed, comparison_passed=comparison['passed'], phase_count=len(rows),
            failed_existing_phase_count=sum(bool(row['failed_existing_checks']) for row in rows),
            preservation_failed_phase_count=sum(bool(row['failed_preservation_checks']) for row in rows),
            late_backward_start_count=sum(row['phase']['kind'] == 'start' and row['phase']['requested_speed'] < 0. and
                row['completed_parking_phases_before_start'] >= 2 for row in rows), events=events,
            own_feedback_parity_max_m_s=parity))
        print('seed', seed, 'saved-trace diagnosis complete; no training/physics', flush=True)
        del parent, candidate, outputs, rows
    assert training_source.tree_digest(training_source.cpu_state(parent_model)) == parent_state
    assert training_source.tree_digest(training_source.cpu_state(candidate_model)) == candidate_state
    assert all(file_sha(path) == expected for path,expected in sources.items())
    assert all(file_sha(path) == expected for path,expected in input_files.items())
    write(args.output/'result.json', dict(verified=True, runs=results, event_traces=all_events,
        input_files_sha256=input_files, sources_sha256=sources, optimizer_updates=0, physics_steps=0,
        frozen_encoder_inference_only=True, existing_gates_unchanged=True, offline_failure_not_reclassified=True,
        limitations=['Earliest failed aggregate phase is not an exact causal onset.',
            'Cross-encoder estimates on identical histories isolate network output differences only.',
            'Closed-loop histories include changed native commands, actions and dynamics; these cannot be uniquely attributed to estimator error.',
            'These are already-observed trajectories, not new holdout evidence. No fitting, input-channel ablation, additional rollout or repeated diagnosis follows automatically.'],
        height='Backward height remains a separate unresolved issue.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=5, action='append', required=True,
        metavar=('SEED', 'PARENT_DIR', 'CANDIDATE_DIR', 'WINDOW_LOCK', 'COMPARISON_JSON'))
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--candidate-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    main(parser.parse_args())
