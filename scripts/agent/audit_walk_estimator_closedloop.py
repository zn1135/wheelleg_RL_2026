#!/usr/bin/env python3
"""Independently audit all three saved diagnostic pairs; never runs a policy."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


PARENT = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
CANDIDATE = 'eeac74b114132b1e2d6781bca7a0864e287897abb4a7dedcb2e2d5aa34da7216'
COMPARATOR = '89fe74747496847e5b6d64a38d050566873a5742bfe5875046887b53c010fd71'
COLLECTOR = '658d95573beebcb9060c77f52dbdef0ab45976b5499162315112010d7971417a'
OLD_CARD = '89b433fafa0a151e88a8fb60096ea02237c01e68ccf0166ff34051afca57b48e'
OLD_RESULT = 'd3df3330712dc03eb093b177fbec076a5ceac146a301c163ccb2e00b0db9d7f0'
MOTORS = ('lf0', 'lf00', 'rf0', 'rf00')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def match(actual, expected, label):
    if expected is None:
        assert actual is None, label
    elif isinstance(expected, (int, float, list)):
        assert np.allclose(np.asarray(actual), np.asarray(expected), rtol=0., atol=1e-6), label
    else:
        assert actual == expected, label


def longest(mask):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    return int(max(ends-starts, default=0))


def turning_events(angle, times, threshold):
    direction = 0
    anchor = extremum = float(angle[0])
    peak_time = float(times[0])
    result = []
    for time, value in zip(times[1:], angle[1:]):
        value = float(value)
        if direction == 0:
            if abs(value-anchor) >= threshold:
                direction = 1 if value > anchor else -1
                extremum, peak_time = value, float(time)
        elif direction*(value-extremum) >= 0.:
            extremum, peak_time = value, float(time)
        elif direction*(extremum-value) >= threshold:
            result.append(dict(turn_s=peak_time, confirmed_s=float(time), excursion_deg=abs(extremum-value)))
            direction = -direction
            extremum, peak_time = value, float(time)
    return result


def sustained(data, env, windows, rule):
    flagged, eligible = [], 0
    for i in range(len(windows)-2):
        three = windows[i:i+3]
        if not all(window['complete'] for window in three) or [window['start_s'] for window in three] != list(range(three[0]['start_s'], three[0]['start_s']+3)):
            continue
        eligible += 1
        start, end = three[0]['start_s'], three[-1]['end_s']
        mask = (data['time'] > start+1e-9) & (data['time'] <= end+1e-9)
        for motor, name in enumerate(MOTORS):
            if not all(window['motor_p2p_deg'][motor] > rule['window_p2p_gt_deg'] for window in three):
                continue
            angle = np.rad2deg(np.unwrap(data['motor'][mask, env, motor]))
            turns = turning_events(angle, data['time'][mask], rule['each_reverse_excursion_at_least_deg'])
            if len(turns) >= rule['reversals_in_3s_at_least']:
                flagged.append(dict(start_s=start, end_s=end, motor=name, confirmed_reversals=turns))
    return dict(detected=bool(flagged), eligible_three_window_groups=eligible, flagged_groups=flagged)


def initial43(folder):
    meta = read(folder/'encoder.json')
    assert meta['seed'] == 43 and meta['feedback_branch'] == 'parent' and meta['checkpoint_sha256'] == PARENT
    historical = Path(__file__).resolve().parents[2]/'artifacts/walk25-goal-wheel-vx-probe-20261030-r1/encoder.npz'
    keys = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
    with np.load(folder/'encoder.npz', allow_pickle=False) as current, np.load(historical, allow_pickle=False) as old:
        checks = {}
        for key in keys:
            a, b = current['fixed_'+key], old['fixed_'+key]
            checks[key] = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            assert checks[key], 'Historical seed43 initial/physical array differs: '+key
    return dict(verified=True, seed=43, original_success_ids=list(range(96)),
        historical_path=str(historical), historical_trace_sha256=sha(historical),
        new_parent_trace_sha256=sha(folder/'encoder.npz'), all_eight_fixed_arrays_byte_equal=checks,
        optimizer_updates=0, network_inference=False, simulation=False)


def raw(folder):
    meta = read(folder/'encoder.json')
    receipt = read(folder/'result.json')
    assert receipt['verified'] and receipt['collector_source_sha256'] == sha(folder/'source.py') == COLLECTOR
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] and meta['encoder_unchanged']
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
    assert sha(meta['checkpoint']) == meta['checkpoint_sha256']
    assert meta['parent_checkpoint_sha256'] == PARENT and meta['candidate_checkpoint_sha256'] == CANDIDATE
    assert sha(meta['candidate_receipt_path']) == meta['candidate_receipt_sha256']
    assert meta['dt'] == .002 and meta['policy_dt'] == .01 and meta['seconds'] == 30.
    for dep in meta['runtime_dependency_sources'].values():
        assert sha(dep['path']) == dep['sha256']
    with np.load(folder/'encoder.npz', allow_pickle=False) as saved:
        values = saved['values']
        fixed = {name:saved[name] for name in saved.files if name.startswith('fixed_')}
        qpos, qtime = saved['physics_qpos'], saved['physics_qpos_time_s']
        history, observations = saved['native_history'], saved['policy_observation']
        actions, prev = saved['native_action'], saved['policy_previous_action']
        policy_valid = saved['policy_valid']
    assert values.shape[:2] == (15000, 96) and np.isfinite(values).all()
    assert history.shape == (3000, 96, 125)
    assert np.array_equal(history[policy_valid, -25:], observations[policy_valid])
    assert np.array_equal(observations[..., -6:], prev)
    assert np.array_equal(prev[1:][policy_valid[1:]], actions[:-1][policy_valid[1:]])
    def get(name):
        return values[..., slice(*meta['layout'][name])]
    valid = get('valid')[..., 0] > .5
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    assert np.array_equal(valid.sum(0), np.rint(np.asarray(meta['episode_end'])/.002).astype(int))
    assert np.array_equal(get('policy_velocity')[valid], get('encoder_velocity')[valid])
    motor_ids = [meta['joint_names'].index(name) for name in MOTORS]
    result = dict(folder=folder, meta=meta, fixed=fixed, valid=valid, time=(np.arange(15000)+1)*.002,
        vx=get('true_velocity')[..., 0], world=get('root_linear_velocity_w'), pitch=get('roll_pitch')[..., 1],
        height=get('height')[..., 0], motor=get('joint_q')[..., motor_ids], dq=get('joint_dq'),
        angular=get('angular_velocity_xy'), length=get('leg_length'), command=get('applied_command')[..., 0],
        xy=qpos[..., np.asarray(meta['root_qpos_adr'])[:2]], qtime=qtime)
    assert result['xy'].shape == (15001, 96, 2)
    return result


def window_metrics(data, env):
    output = []
    for start in range(30):
        planned = (data['time'] > start+1e-9) & (data['time'] <= start+1.+1e-9)
        live = planned & data['valid'][:, env]
        complete = bool(planned.sum() == 500 and live.sum() == 500)
        angle = np.rad2deg(np.unwrap(data['motor'][live, env], axis=0)) if live.any() else None
        amplitude = None if angle is None else np.ptp(angle, axis=0)
        output.append(dict(start_s=start, end_s=start+1., complete=complete, observed_samples=int(live.sum()),
            motor_p2p_deg=None if amplitude is None else amplitude.tolist(),
            four_motor_max_p2p_deg=None if amplitude is None else float(amplitude.max()),
            leg_length_p2p_m=None if not live.any() else np.ptp(data['length'][live, env], axis=0).tolist()))
    return output


def phase_metrics(data, row, all_windows, rule):
    env, start, end = row['env'], row['start_s'], row['end_s']
    planned = (data['time'] > start+1e-9) & (data['time'] <= end+1e-9)
    live = planned & data['valid'][:, env]
    complete = bool(planned.any() and live.sum() == planned.sum())
    assert row['complete'] == complete and row['expected_samples'] == int(planned.sum()) and row['observed_samples'] == int(live.sum())
    metrics = None
    if live.any():
        vx, horizontal = data['vx'][live, env], np.linalg.norm(data['world'][live, env, :2], axis=-1)
        pitch, height = abs(data['pitch'][live, env]), data['height'][live, env]
        error = abs(vx-row['requested_speed'])
        metrics = dict(vx_mean_m_s=float(vx.mean()), vx_mean_bias_m_s=float((vx-row['requested_speed']).mean()),
            vx_mae_m_s=float(error.mean()), vx_abs_error_p95_m_s=float(np.percentile(error, 95)), vx_abs_error_max_m_s=float(error.max()),
            world_horizontal_speed_max_m_s=float(horizontal.max()), world_horizontal_speed_p95_m_s=float(np.percentile(horizontal, 95)),
            world_horizontal_speed_over_0_05_duration_s=float((horizontal > .05).sum()*.002),
            pitch_abs_max_deg=float(pitch.max()), pitch_abs_p95_deg=float(np.percentile(pitch, 95)),
            pitch_abs_over_5_duration_s=float((pitch > 5.).sum()*.002), pitch_abs_over_5_longest_s=longest(pitch > 5.)*.002,
            height_mean_m=float(height.mean()), height_mae_m=float(abs(height-.26).mean()),
            native_command_vx_mae_m_s=float(abs(vx-data['command'][live, env]).mean()),
            native_command_vx_p95_m_s=float(np.percentile(abs(vx-data['command'][live, env]), 95)),
            leg_length_mean_m=data['length'][live, env].mean(0).tolist(),
            leg_length_min_m=data['length'][live, env].min(0).tolist(), leg_length_max_m=data['length'][live, env].max(0).tolist())
        edge_xy = np.asarray([[np.interp(t, data['qtime'], data['xy'][:, env, axis]) for axis in range(2)] for t in (start, end)])
        metrics['xy_displacement_m'] = (edge_xy[1]-edge_xy[0]).tolist() if complete else None
        metrics['xy_displacement_norm_m'] = float(np.linalg.norm(edge_xy[1]-edge_xy[0])) if complete else None
        if complete:
            path = np.concatenate((edge_xy[:1], data['xy'][(data['qtime'] > start) & (data['qtime'] < end), env], edge_xy[1:]))
            metrics['xy_path_length_m'] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            metrics['xy_max_distance_from_start_m'] = float(np.linalg.norm(path-path[0], axis=1).max())
        saved = row['metrics'] if complete else row['partial_observed_metrics']
        for key, value in metrics.items():
            match(saved[key], value, (env, row['kind'], key))
        t, over = data['time'][live], pitch > 5.
        exceeded = np.flatnonzero(over)
        first_return, final_return = 0., 0.
        if len(exceeded):
            back = np.flatnonzero((np.arange(len(t)) > exceeded[0]) & ~over)
            first_return = None if not len(back) else float(t[back[0]]-start)
            final_return = None if exceeded[-1]+1 >= len(t) else float(t[exceeded[-1]+1]-start)
        recovery = row['pitch_recovery']
        match(recovery['first_return_after_exceedance_s'], first_return, (env, row['kind'], 'first_return'))
        match(recovery['final_observed_return_after_last_exceedance_s'], final_return, (env, row['kind'], 'final_return'))
        match(recovery['pitch_abs_over_5_duration_s'], metrics['pitch_abs_over_5_duration_s'], (env, row['kind'], 'pitch_duration'))
    else:
        assert row['metrics'] is None and row['partial_observed_metrics'] is None
    selected = [window for window in all_windows if window['start_s'] >= start-1e-9 and window['end_s'] <= end+1e-9]
    assert selected == row['one_second_windows']
    detected = sustained(data, env, selected, rule)
    for key, value in detected.items():
        assert row['sustained'][key] == value, (env, row['kind'], key)
    assert row['sustained']['coverage_sufficient'] == (detected['eligible_three_window_groups'] > 0)
    assert row['sustained']['assessment'] == ('sustained' if detected['detected'] else 'none_observed' if detected['eligible_three_window_groups'] else 'coverage_insufficient')
    return complete, metrics, selected


def audit_pair(comparison_path):
    comparison = read(comparison_path)
    assert comparison['verified'] and comparison['comparator_source_sha256'] == COMPARATOR
    folders = {}
    for branch in ('baseline', 'candidate'):
        for path, expected in comparison[branch+'_files'].items():
            assert sha(path) == expected
            if Path(path).name == 'encoder.json':
                folders[branch] = Path(path).parent
    baseline, candidate = raw(folders['baseline']), raw(folders['candidate'])
    bm, cm = baseline['meta'], candidate['meta']
    historical_initials = initial43(folders['baseline']) if bm['seed'] == 43 else None
    assert bm['feedback_branch'] == 'parent' and cm['feedback_branch'] == 'candidate'
    assert bm['checkpoint_sha256'] == PARENT and cm['checkpoint_sha256'] == CANDIDATE
    for key in ('seed', 'cases', 'layout', 'joint_names', 'protocol_identity', 'engine_identity', 'controller_sha256'):
        assert bm[key] == cm[key], key
    assert baseline['fixed'].keys() == candidate['fixed'].keys()
    for key, value in baseline['fixed'].items():
        other = candidate['fixed'][key]
        assert value.shape == other.shape and value.dtype == other.dtype and value.tobytes() == other.tobytes(), key
    window_path = Path(cm['window_lock']['path'])
    assert sha(window_path) == cm['window_lock']['sha256'] == comparison['pairing']['windows_sha256']
    locked = read(window_path)
    card_path = Path(locked['execution_card'])
    assert sha(card_path) == locked['execution_card_sha256'] == OLD_CARD
    card = read(card_path)
    rule = card['closed_loop_screening']
    locked_keys = {(row['env'], row['kind'], row['segment']):(row['start_s'], row['end_s']) for row in locked['phases']}
    counts, phase_checks, by_env, windows_total = Counter(), {}, {}, 0
    branch_windows = {}
    for branch, data in (('baseline', baseline), ('candidate', candidate)):
        branch_windows[branch] = {}
        for env in range(96):
            windows = window_metrics(data, env)
            assert windows == comparison['all_t0_anchored_one_second_windows'][str(env)][branch]
            branch_windows[branch][env] = windows
            windows_total += len(windows)
    for item in comparison['per_phase']:
        env = item['env']
        b, c = item['baseline'], item['candidate']
        assert (b['start_s'], b['end_s']) == (c['start_s'], c['end_s']) == locked_keys[(env, b['kind'], b['segment'])]
        bc, mb, bw = phase_metrics(baseline, b, branch_windows['baseline'][env], rule['sustained_swing'])
        cc, mc, cw = phase_metrics(candidate, c, branch_windows['candidate'][env], rule['sustained_swing'])
        checks = dict(complete_phase=bc and cc)
        kind = b['kind']
        if bc and cc:
            if kind in ('parking', 'cruise_forward', 'cruise_backward'):
                checks['steady_pitch'] = mc['pitch_abs_max_deg'] <= rule['steady_pitch_abs_max_deg']
                if c['sustained']['coverage_sufficient']:
                    checks['no_sustained_swing'] = not c['sustained']['detected']
            if kind == 'parking':
                checks['drift_target'] = mc['world_horizontal_speed_max_m_s'] <= rule['drift_horizontal_peak_max_m_s']
            elif kind in ('cruise_forward', 'cruise_backward'):
                checks['tracking_mae'] = mc['vx_mae_m_s'] <= mb['vx_mae_m_s']+rule['tracking_mae_increase_max_m_s']
                checks['tracking_p95'] = mc['vx_abs_error_p95_m_s'] <= mb['vx_abs_error_p95_m_s']+rule['tracking_p95_increase_max_m_s']
            elif kind in ('start', 'brake'):
                checks['transient_pitch_peak'] = mc['pitch_abs_max_deg'] <= mb['pitch_abs_max_deg']+rule['transient_pitch_peak_increase_max_deg']
                checks['transient_pitch_total_over5'] = mc['pitch_abs_over_5_duration_s'] <= mb['pitch_abs_over_5_duration_s']+rule['transient_pitch_exceedance_duration_increase_max_s']
                checks['transient_pitch_longest_over5'] = mc['pitch_abs_over_5_longest_s'] <= mb['pitch_abs_over_5_longest_s']+rule['transient_pitch_exceedance_duration_increase_max_s']
        assert checks == item['checks'] and item['passed'] == all(checks.values())
        counts.update(key for key, value in checks.items() if not value)
        phase_checks.setdefault(env, []).append(item)
        by_env[env] = by_env.get(env, 0)+1
    assert set(by_env) == set(range(96)) and len(comparison['per_phase']) == len(locked['phases'])
    for row in comparison['per_environment']:
        env = row['env']
        parking = [phase for phase in phase_checks[env] if phase['baseline']['kind'] == 'parking']
        bamp = [window['four_motor_max_p2p_deg'] for phase in parking for window in phase['baseline']['one_second_windows'] if window['complete']]
        camp = [window['four_motor_max_p2p_deg'] for phase in parking for window in phase['candidate']['one_second_windows'] if window['complete']]
        bp95, cp95 = (None if not values else float(np.percentile(values, 95)) for values in (bamp, camp))
        bdrift = max((phase['baseline']['metrics']['world_horizontal_speed_max_m_s'] for phase in parking if phase['baseline']['metrics'] is not None), default=None)
        cdrift = max((phase['candidate']['metrics']['world_horizontal_speed_max_m_s'] for phase in parking if phase['candidate']['metrics'] is not None), default=None)
        checks = dict(original_survival_retained=bool(bm['episode_done'][env] or not cm['episode_done'][env]),
            parking_leg_amplitude_coverage=bool(bamp and camp and len(bamp) == len(camp)),
            sustained_parking_coverage=any(phase['candidate']['sustained']['coverage_sufficient'] for phase in parking),
            parking_leg_p95_improvement=bp95 is not None and cp95 is not None and cp95 <= bp95*rule['leg_p2p_p95_ratio_max'],
            parking_drift_peak_improvement=bdrift is not None and cdrift is not None and cdrift <= bdrift*rule['drift_peak_ratio_max'])
        assert checks == row['checks']
        assert row['passed'] == (all(checks.values()) and all(item['passed'] for item in phase_checks[env]))
        assert row['original_survived'] == (not bm['episode_done'][env]) and row['candidate_survived'] == (not cm['episode_done'][env])
        assert row['candidate_first_termination'] == cm['first_termination'][env]
        counts.update(key for key, value in checks.items() if not value)
    assert len(comparison['per_environment']) == 96
    assert dict(counts) == comparison['failure_counts']
    assert comparison['passed'] == all(row['passed'] for row in comparison['per_environment'])
    return dict(seed=bm['seed'], baseline_survived=bm['survived'], candidate_survived=cm['survived'],
        passed=comparison['passed'], failure_counts=dict(counts), phase_rows=len(comparison['per_phase']),
        full_one_second_windows_recomputed=windows_total, fixed_array_pairs_byte_equal=len(baseline['fixed']),
        historical_seed43_initial_verification=historical_initials,
        repeated_backward_start_phase_rows=sum(row['baseline']['kind'] == 'start' and row['baseline']['segment'] == 3 and row['baseline']['requested_speed'] == -.5 for row in comparison['per_phase']),
        comparison=str(comparison_path), comparison_sha256=sha(comparison_path))


def main(args):
    artifact = args.artifact.resolve()
    if args.initial43 is not None:
        result = initial43(args.initial43.resolve())
        result['audit_source_sha256'] = sha(__file__)
        with args.output.open('x') as destination:
            json.dump(result, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write('\n')
        print(json.dumps(dict(verified=True, historical_seed43_fixed_arrays_equal=True, optimizer_updates=0, network_inference=False, simulation=False)), flush=True)
        return
    paths = sorted(artifact.rglob('comparison.json'))
    assert len(paths) == 3, 'Wait for all three diagnostic groups; do not audit a selected subset'
    results = [audit_pair(path) for path in paths]
    assert {row['seed'] for row in results} == {43, 48, 49}
    diagnostic = read(artifact/'diagnostic-card.json')
    assert diagnostic['training_updates_authorized'] == 0 and not diagnostic['offline_gate_or_threshold_change']
    assert diagnostic['behavior_failure_does_not_skip_remaining_diagnostic_runs']
    assert {row['seed'] for row in diagnostic['runs']} == {43, 48, 49}
    assert diagnostic['parent_sha256'] == PARENT and diagnostic['candidate_sha256'] == CANDIDATE
    for path, expected in diagnostic['source_sha256'].items():
        assert sha(path) == expected
    for row in results:
        for prefix in ('candidate', 'compare'):
            assert read(artifact/f'{prefix}{row["seed"]}-exit.json')['returncode'] == 0
    assert read(artifact/'parent43-exit.json')['returncode'] == 0
    old = artifact.parent/'walk25-estimator-only-20261031-r1'
    assert sha(old/'experiment-card.json') == OLD_CARD
    assert sha(old/'final-offline-r1/result.json') == OLD_RESULT
    assert not read(old/'final-offline-r1/result.json')['closed_loop_release']
    result = dict(verified=True, audit_source_sha256=sha(__file__), groups=results,
        diagnostic_card_sha256=sha(artifact/'diagnostic-card.json'), all_candidate_and_compare_returncodes=0,
        all_three_groups_present=True, old_offline_gate_unchanged_and_failed=True,
        candidate_adoption=False, optimizer_updates=0, network_inference=False, simulation=False,
        same_parent_windows=True, independent_raw_metric_window_and_gate_recomputation=True,
        scope='Separate newly authorized diagnostic only. Raw saved500Hz traces audited; every failure remains. No claim of model adoption or reversal of the earlier failed offline gate.')
    with args.output.open('x') as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps(dict(verified=True, groups=results, network_inference=False, simulation=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial43', type=Path)
    main(parser.parse_args())
