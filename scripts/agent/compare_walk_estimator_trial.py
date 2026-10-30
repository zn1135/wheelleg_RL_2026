#!/usr/bin/env python3
"""Saved-trace behavior comparison with parent-locked windows; no simulation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_walk_velocity_compensation import phase_specs
from summarize_walk_goal_feedback import scalar_metrics, transition_pitch


PARENT_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
BUCKETS = ('initial', 'parking', 'cruise_forward', 'cruise_backward', 'start', 'brake')
FIXED_KEYS = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction',
              'body_ipos', 'body_inertia', 'body_iquat')
MOTORS = ('lf0', 'lf00', 'rf0', 'rf00')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write('\n')


def same_array(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def exact(a, b):
    assert type(a) is type(b)
    if torch.is_tensor(a):
        assert same_array(a.detach().cpu().contiguous().numpy(), b.detach().cpu().contiguous().numpy())
    elif isinstance(a, np.ndarray):
        assert same_array(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def load(folder):
    folder = Path(folder).resolve()
    meta = read(folder/'encoder.json')
    receipt = read(folder/'result.json')
    assert receipt['verified'] and meta['parameter_updates'] == 0
    assert sha(folder/'source.py') == receipt['collector_source_sha256'] == meta['protocol_identity']['collector_source_sha256']
    assert meta['actor_unchanged'] and meta['encoder_unchanged']
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
    assert meta['seconds'] == 30. and meta['dt'] == .002 and meta['policy_dt'] == .01
    assert meta['height'] == .26
    assert sha(meta['checkpoint']) == meta['checkpoint_sha256']
    with np.load(folder/'encoder.npz', allow_pickle=False) as stored:
        values = stored['values']
        assert values.shape[:2] == (15000, 96) and np.isfinite(values).all()
        fixed = {key.removeprefix('fixed_'):stored[key] for key in stored.files if key.startswith('fixed_')}
        assert all(key in fixed for key in FIXED_KEYS)
        actual_model = {key:stored[key] for key in stored.files if key.startswith('actual_model_')}
        observation, policy_valid = stored['policy_observation'], stored['policy_valid']
        assert observation.shape == (3000, 96, 25) and policy_valid.shape == (3000, 96)
        positions, position_times = stored['qpos'], stored['qpos_time_s']
        if 'physics_qpos' in stored.files:
            positions, position_times = stored['physics_qpos'], stored['physics_qpos_time_s']
            assert positions.shape[:2] == (15001, 96)
        root_adrs = np.asarray(meta['root_qpos_adr'], dtype=int)
        assert root_adrs.shape == (7,)
        xy = positions[..., root_adrs[:2]]
        final_live = stored['qpos_live'][-1]
    def get(name):
        return values[..., slice(*meta['layout'][name])]
    times = (np.arange(len(values))+1)*.002
    valid = get('valid')[..., 0] > .5
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    done, ends = np.asarray(meta['episode_done'], bool), np.asarray(meta['episode_end'])
    assert done.shape == ends.shape == (96,)
    assert np.array_equal(final_live, ~done) and meta['survived'] == int((~done).sum())
    assert np.array_equal(valid.sum(0), np.rint(ends/.002).astype(int))
    assert np.array_equal(get('policy_velocity')[valid], get('encoder_velocity')[valid])
    assert not meta['native_observation_cfg']['noise'] and not meta['native_observation_cfg']['delay']
    assert all(value == 0. for value in meta['exact_alignment_residuals'].values())
    ids = [meta['joint_names'].index(name) for name in MOTORS]
    data = dict(times=times, dt=.002, target_height=.26, valid=valid,
        vx=get('true_velocity')[..., 0], world_velocity=get('root_linear_velocity_w'),
        pitch=get('roll_pitch')[..., 1], height=get('height')[..., 0],
        motor=get('joint_q')[..., ids], joint_dq=get('joint_dq'),
        angular_velocity_xy=get('angular_velocity_xy'), swing=get('swing'), length=get('leg_length'),
        applied_vx=get('applied_command')[..., 0], position_times=position_times, xy=xy,
        observation=observation, policy_valid=policy_valid, done=done, ends=ends,
        fixed=fixed, actual_model=actual_model, meta=meta, folder=folder)
    assert np.isfinite(xy).all() and np.array_equal(position_times, np.arange(len(position_times))*(.002 if len(position_times) == 15001 else .01))
    data['files'] = {str(folder/name):sha(folder/name) for name in ('encoder.json', 'encoder.npz', 'result.json', 'source.py')}
    return data


def validate_schedule(data, card):
    meta = data['meta']
    if meta['seed'] == 43:
        accelerations = (.5, 2.)
        intervals = [(0., 2., 0), (2., 16., 1), (16., 30., 0)]
    else:
        rules = [row for row in card['future_final_runs'] if row['seed'] == meta['seed']]
        assert len(rules) == 1
        accelerations = rules[0]['accelerations']
        intervals = [(row['start'], row['end'], row['speed_multiplier']) for row in rules[0]['request_intervals']]
    assert len(meta['cases']) == 6
    covered, conditions = [], set()
    for case in meta['cases']:
        assert len(case['env_ids']) == 16
        covered.extend(case['env_ids'])
        speed, acceleration = case['speed'], case['acceleration']
        conditions.add((speed, acceleration))
        assert [(row['start'], row['end'], row['speed']) for row in case['segments']] == [(a, b, speed*m) for a, b, m in intervals]
        assert all(row['acceleration'] == acceleration for row in case['segments'])
    assert sorted(covered) == list(range(96))
    assert conditions == {(speed, acceleration) for speed in (0., .5, -.5) for acceleration in accelerations}


def merge_segments(case):
    result = []
    for item in case['segments']:
        row = dict(start=float(item['start']), end=float(item['end']), speed=float(item['speed']),
            acceleration=float(item.get('acceleration', case['acceleration'])))
        assert row['end'] > row['start'] and row['acceleration'] > 0.
        if result:
            assert result[-1]['end'] == row['start']
        if result and result[-1]['speed'] == row['speed']:
            result[-1]['end'] = row['end']
        else:
            result.append(row)
    assert result[0]['start'] == result[0]['speed'] == 0. and result[-1]['end'] == 30.
    return result


def make_lock(data, execution_card):
    meta, phases, applicability, covered = data['meta'], [], {}, []
    assert meta.get('feedback_branch') == 'parent' and meta['checkpoint_sha256'] == PARENT_SHA
    policy_times = np.arange(3000)*.01
    for case in meta['cases']:
        segments = merge_segments(case)
        for env in case['env_ids']:
            covered.append(env)
            observed = set()
            for spec in phase_specs(segments, policy_times, data['observation'][:, env, 6]/2., data['policy_valid'][:, env]):
                kind = spec['kind']
                if kind == 'cruise':
                    kind += '_forward' if spec['requested_speed'] > 0. else '_backward'
                observed.add(kind)
                phases.append(dict(env=env, case=case['name'], kind=kind, segment=spec['segment'],
                    start_s=spec['start'], end_s=spec['end'], requested_speed=spec['requested_speed'],
                    phase_detail=spec))
            applicability[str(env)] = {kind:kind in observed for kind in BUCKETS}
    assert sorted(covered) == list(range(96))
    return dict(schema='walk-estimator-parent-windows-v1', baseline_folder=str(data['folder']),
        baseline_files=data['files'], execution_card=str(execution_card.resolve()), execution_card_sha256=sha(execution_card),
        seed=meta['seed'], cases=meta['cases'], phases=phases, applicability=applicability,
        one_second_windows=[[i, i+1] for i in range(30)], window_anchor_s=0.,
        phase_rule='Parent native100Hz command reaches target, max(nominal ramp end, observed end)+1.5s; initial0-1.5s; merged unchanged requests',
        scoring_samples='500Hz post-step t>start and t<=end; same parent windows for both branches',
        helper_source_sha256=sha(Path(__file__).with_name('analyze_walk_velocity_compensation.py')),
        comparator_source_sha256=sha(__file__), candidate_dependent_windows=False)


def longest_true(mask):
    longest = current = 0
    for value in mask:
        current = current+1 if value else 0
        longest = max(longest, current)
    return longest


def reversals(angle_deg, times, minimum):
    """Count confirmed reversals after an actual opposite excursion >=minimum."""
    direction, extreme, extreme_time = 0, float(angle_deg[0]), float(times[0])
    anchor, result = extreme, []
    for t, angle in zip(times[1:], angle_deg[1:]):
        angle = float(angle)
        if direction == 0:
            if abs(angle-anchor) >= minimum:
                direction = 1 if angle > anchor else -1
                extreme, extreme_time = angle, float(t)
        elif direction*(angle-extreme) >= 0.:
            extreme, extreme_time = angle, float(t)
        elif direction*(extreme-angle) >= minimum:
            result.append(dict(turn_s=extreme_time, confirmed_s=float(t), excursion_deg=abs(extreme-angle)))
            direction, extreme, extreme_time = -direction, angle, float(t)
    return result


def windows(data, env, rule):
    result = []
    for start in range(30):
        mask = (data['times'] > start+1e-9) & (data['times'] <= start+1.+1e-9)
        live = mask & data['valid'][:, env]
        complete = bool(mask.sum() == 500 and live.sum() == 500)
        motors = np.rad2deg(np.unwrap(data['motor'][live, env], axis=0)) if live.any() else None
        p2p = None if motors is None else np.ptp(motors, axis=0)
        result.append(dict(start_s=start, end_s=start+1., complete=complete, observed_samples=int(live.sum()),
            motor_p2p_deg=None if p2p is None else p2p.tolist(),
            four_motor_max_p2p_deg=None if p2p is None else float(p2p.max()),
            leg_length_p2p_m=None if not live.any() else np.ptp(data['length'][live, env], axis=0).tolist()))
    return result


def sustained(data, env, selected, rule):
    triples, eligible = [], 0
    for i in range(len(selected)-2):
        group = selected[i:i+3]
        if not all(item['complete'] for item in group) or group[0]['end_s'] != group[1]['start_s'] or group[1]['end_s'] != group[2]['start_s']:
            continue
        eligible += 1
        start, end = group[0]['start_s'], group[-1]['end_s']
        mask = (data['times'] > start+1e-9) & (data['times'] <= end+1e-9)
        for motor, name in enumerate(MOTORS):
            if not all(item['motor_p2p_deg'][motor] > rule['window_p2p_gt_deg'] for item in group):
                continue
            angle = np.rad2deg(np.unwrap(data['motor'][mask, env, motor]))
            turns = reversals(angle, data['times'][mask], rule['each_reverse_excursion_at_least_deg'])
            if len(turns) >= rule['reversals_in_3s_at_least']:
                triples.append(dict(start_s=start, end_s=end, motor=name, confirmed_reversals=turns))
    spans = []
    for name in MOTORS:
        for item in [item for item in triples if item['motor'] == name]:
            if spans and spans[-1]['motor'] == name and item['start_s'] <= spans[-1]['end_s']:
                spans[-1]['end_s'] = max(spans[-1]['end_s'], item['end_s'])
            else:
                spans.append(dict(motor=name, start_s=item['start_s'], end_s=item['end_s']))
    return dict(detected=bool(triples), assessment='sustained' if triples else 'none_observed' if eligible else 'coverage_insufficient',
        coverage_sufficient=eligible > 0, eligible_three_window_groups=eligible,
        flagged_groups=triples, same_motor_flagged_spans=spans,
        longest_flagged_span_s=max((item['end_s']-item['start_s'] for item in spans), default=0.))


def phase_summary(data, phase, all_windows, screening):
    env, start, end = phase['env'], phase['start_s'], phase['end_s']
    planned = (data['times'] > start+1e-9) & (data['times'] <= end+1e-9)
    live = planned & data['valid'][:, env]
    complete = bool(planned.any() and planned.sum() == live.sum())
    metrics = scalar_metrics(data, env, live, phase['requested_speed'])
    if metrics is not None:
        pitch = abs(data['pitch'][live, env])
        metrics['pitch_abs_over_5_longest_s'] = longest_true(pitch > 5.)*data['dt']
        metrics['native_command_vx_mae_m_s'] = float(abs(data['vx'][live, env]-data['applied_vx'][live, env]).mean())
        metrics['native_command_vx_p95_m_s'] = float(np.percentile(abs(data['vx'][live, env]-data['applied_vx'][live, env]), 95))
        boundary_xy = np.asarray([[np.interp(t, data['position_times'], data['xy'][:, env, axis]) for axis in range(2)] for t in (start, end)])
        metrics['xy_displacement_m'] = (boundary_xy[1]-boundary_xy[0]).tolist() if complete else None
        metrics['xy_displacement_norm_m'] = float(np.linalg.norm(boundary_xy[1]-boundary_xy[0])) if complete else None
        if complete:
            xy_path = np.concatenate((boundary_xy[:1], data['xy'][(data['position_times'] > start) & (data['position_times'] < end), env], boundary_xy[1:]))
            metrics['xy_path_length_m'] = float(np.linalg.norm(np.diff(xy_path, axis=0), axis=1).sum())
            metrics['xy_max_distance_from_start_m'] = float(np.linalg.norm(xy_path-xy_path[0], axis=1).max())
        metrics['leg_length_mean_m'] = data['length'][live, env].mean(0).tolist()
        metrics['leg_length_min_m'] = data['length'][live, env].min(0).tolist()
        metrics['leg_length_max_m'] = data['length'][live, env].max(0).tolist()
    selected = [item for item in all_windows if item['start_s'] >= start-1e-9 and item['end_s'] <= end+1e-9]
    amplitudes = [item['four_motor_max_p2p_deg'] for item in selected if item['complete']]
    return dict(**phase, complete=complete, expected_samples=int(planned.sum()), observed_samples=int(live.sum()),
        metrics=metrics if complete else None, partial_observed_metrics=metrics if not complete else None,
        pitch_recovery=transition_pitch(data, env, start, end), one_second_windows=selected,
        leg_window_max_p2p_p95_deg=None if not amplitudes else float(np.percentile(amplitudes, 95)),
        leg_window_max_p2p_max_deg=None if not amplitudes else float(max(amplitudes)),
        sustained=sustained(data, env, selected, screening['sustained_swing']))


def validate_pair(baseline, candidate, card_path, window_path):
    bm, cm = baseline['meta'], candidate['meta']
    assert bm.get('feedback_branch') == 'parent' and cm.get('feedback_branch') == 'candidate'
    assert bm['checkpoint_sha256'] == PARENT_SHA
    for key in ('seed', 'seconds', 'height', 'dt', 'policy_dt', 'cases', 'layout', 'joint_names', 'protocol_identity', 'engine_identity'):
        assert bm[key] == cm[key], key
    assert cm['controller_unchanged'] and cm['candidate_verification']['parent_fields_except_encoder_bitwise_equal']
    assert cm['parent_checkpoint_sha256'] == PARENT_SHA
    assert bm['controller_sha256'] == cm['controller_sha256']
    receipt = read(card_path.parent/'training-r1/checkpoint-1000.json')
    assert cm['checkpoint_sha256'] == cm['candidate_checkpoint_sha256'] == receipt['sha256']
    assert sha(receipt['path']) == receipt['sha256']
    assert Path(cm['candidate_receipt_path']).resolve() == card_path.parent/'training-r1/checkpoint-1000.json'
    assert sha(cm['candidate_receipt_path']) == cm['candidate_receipt_sha256']
    parent = torch.load(bm['checkpoint'], map_location='cpu', weights_only=False)
    paired = torch.load(cm['checkpoint'], map_location='cpu', weights_only=False)
    assert parent.keys() == paired.keys()
    for key in parent:
        if key == 'actor_state_dict':
            assert parent[key].keys() == paired[key].keys()
            for name in parent[key]:
                if not name.startswith('encoder.'):
                    exact(parent[key][name], paired[key][name])
        else:
            exact(parent[key], paired[key])
    assert baseline['fixed'].keys() == candidate['fixed'].keys()
    for key in baseline['fixed']:
        assert same_array(baseline['fixed'][key], candidate['fixed'][key]), key
    assert baseline['actual_model'].keys() == candidate['actual_model'].keys()
    for key in baseline['actual_model']:
        assert same_array(baseline['actual_model'][key], candidate['actual_model'][key]), key
    assert cm['window_lock'] is not None and cm['window_lock']['sha256'] == sha(window_path)
    assert Path(cm['window_lock']['path']).resolve() == window_path.resolve()
    pairing = cm['baseline_pairing']
    assert pairing['checked_before_first_action'] and pairing['protocol_engine_and_controller_equal']
    assert Path(pairing['path']).resolve() == baseline['folder']
    assert pairing['metadata_sha256'] == sha(baseline['folder']/'encoder.json')
    assert pairing['trace_sha256'] == sha(baseline['folder']/'encoder.npz')
    assert all(pairing['all_fixed_arrays_bitwise_equal'].values())
    for name in bm['runtime_dependency_sources']:
        assert bm['runtime_dependency_sources'][name]['sha256'] == cm['runtime_dependency_sources'][name]['sha256']
    return dict(initial_and_physical_arrays_byte_equal=True, model_fields_except_encoder_byte_equal=True,
        terminal_candidate_checkpoint_sha256=receipt['sha256'], windows_sha256=sha(window_path),
        actual_model_fields=sum(key.startswith('actual_model_') for key in baseline['fixed']), fixed_fields=list(baseline['fixed']))


def compare(baseline, candidate, locked, card):
    screening = card['closed_loop_screening']
    assert screening['sustained_swing']['same_motor_windows'] == 3
    rows, failures, all_windows = [], Counter(), {}
    per_env = []
    for env in range(96):
        bw, cw = windows(baseline, env, screening), windows(candidate, env, screening)
        all_windows[env] = dict(baseline=bw, candidate=cw)
        env_rows = []
        for phase in [item for item in locked['phases'] if item['env'] == env]:
            b = phase_summary(baseline, phase, bw, screening)
            c = phase_summary(candidate, phase, cw, screening)
            checks = {'complete_phase':b['complete'] and c['complete']}
            if checks['complete_phase']:
                mb, mc, kind = b['metrics'], c['metrics'], phase['kind']
                if kind in ('parking', 'cruise_forward', 'cruise_backward'):
                    checks['steady_pitch'] = mc['pitch_abs_max_deg'] <= screening['steady_pitch_abs_max_deg']
                    if c['sustained']['coverage_sufficient']:
                        checks['no_sustained_swing'] = not c['sustained']['detected']
                if kind == 'parking':
                    checks['drift_target'] = mc['world_horizontal_speed_max_m_s'] <= screening['drift_horizontal_peak_max_m_s']
                elif kind in ('cruise_forward', 'cruise_backward'):
                    checks['tracking_mae'] = mc['vx_mae_m_s'] <= mb['vx_mae_m_s']+screening['tracking_mae_increase_max_m_s']
                    checks['tracking_p95'] = mc['vx_abs_error_p95_m_s'] <= mb['vx_abs_error_p95_m_s']+screening['tracking_p95_increase_max_m_s']
                elif kind in ('start', 'brake'):
                    checks['transient_pitch_peak'] = mc['pitch_abs_max_deg'] <= mb['pitch_abs_max_deg']+screening['transient_pitch_peak_increase_max_deg']
                    checks['transient_pitch_total_over5'] = mc['pitch_abs_over_5_duration_s'] <= mb['pitch_abs_over_5_duration_s']+screening['transient_pitch_exceedance_duration_increase_max_s']
                    checks['transient_pitch_longest_over5'] = mc['pitch_abs_over_5_longest_s'] <= mb['pitch_abs_over_5_longest_s']+screening['transient_pitch_exceedance_duration_increase_max_s']
            failures.update(name for name, passed in checks.items() if not passed)
            changes = {} if b['metrics'] is None or c['metrics'] is None else {
                key:dict(delta=c['metrics'][key]-value, ratio=None if value == 0. else c['metrics'][key]/value)
                for key, value in b['metrics'].items() if isinstance(value, (int, float))}
            row = dict(env=env, baseline=b, candidate=c, paired_metric_changes=changes, checks=checks, passed=all(checks.values()))
            rows.append(row)
            env_rows.append(row)
        parking = [row for row in env_rows if row['baseline']['kind'] == 'parking']
        bamp = [item['four_motor_max_p2p_deg'] for row in parking for item in row['baseline']['one_second_windows'] if item['complete']]
        camp = [item['four_motor_max_p2p_deg'] for row in parking for item in row['candidate']['one_second_windows'] if item['complete']]
        checks = dict(original_survival_retained=bool(baseline['done'][env] or not candidate['done'][env]),
            parking_leg_amplitude_coverage=bool(bamp and camp and len(bamp) == len(camp)),
            sustained_parking_coverage=any(row['candidate']['sustained']['coverage_sufficient'] for row in parking))
        bp95 = None if not bamp else float(np.percentile(bamp, 95))
        cp95 = None if not camp else float(np.percentile(camp, 95))
        checks['parking_leg_p95_improvement'] = bp95 is not None and cp95 is not None and cp95 <= bp95*screening['leg_p2p_p95_ratio_max']
        bdrift = max((row['baseline']['metrics']['world_horizontal_speed_max_m_s'] for row in parking if row['baseline']['metrics'] is not None), default=None)
        cdrift = max((row['candidate']['metrics']['world_horizontal_speed_max_m_s'] for row in parking if row['candidate']['metrics'] is not None), default=None)
        checks['parking_drift_peak_improvement'] = bdrift is not None and cdrift is not None and cdrift <= bdrift*screening['drift_peak_ratio_max']
        failures.update(name for name, passed in checks.items() if not passed)
        per_env.append(dict(env=env, original_survived=not bool(baseline['done'][env]), candidate_survived=not bool(candidate['done'][env]),
            original_end_s=float(baseline['ends'][env]), candidate_end_s=float(candidate['ends'][env]),
            original_first_termination=baseline['meta']['first_termination'][env],
            candidate_first_termination=candidate['meta']['first_termination'][env],
            checks=checks, phase_passed=all(row['passed'] for row in env_rows),
            passed=all(checks.values()) and all(row['passed'] for row in env_rows),
            parking_leg_window_p2p_p95_deg=dict(baseline=bp95, candidate=cp95),
            all_parking_horizontal_peak_m_s=dict(baseline=bdrift, candidate=cdrift),
            not_applicable=[kind for kind, applicable in locked['applicability'][str(env)].items() if not applicable]))
    cases = [dict(name=case['name'], env_ids=case['env_ids'], original_survived=sum(not baseline['done'][env] for env in case['env_ids']),
        candidate_survived=sum(not candidate['done'][env] for env in case['env_ids']),
        passed=sum(per_env[env]['passed'] for env in case['env_ids']),
        failed_env_ids=[env for env in case['env_ids'] if not per_env[env]['passed']]) for case in baseline['meta']['cases']]
    return dict(passed=all(row['passed'] for row in per_env), failure_counts=dict(failures), cases=cases,
        per_environment=per_env, per_phase=rows, all_t0_anchored_one_second_windows=all_windows,
        metric_scope='True root-link body vx and world horizontal root-link speed; COM estimator errors are separate offline results.',
        leg_p95_scope='Each environment, all complete parent-window parking1s bins; each steady phase separately checked for sustained swing.',
        drift_ratio_scope='Each environment maximum over all complete parking phases; every individual parking phase must also meet0.05m/s absolute peak.',
        initialization_scope='Complete initialization metrics and peaks retained; no new independent initialization-pitch tolerance invented.',
        height='Backward low height remains independently unresolved; height error reported without reward change.',
        interpretation='Fixed experimental screening only; no training, inference, simulation, deployment or true-robot acceptance.')


def selfcheck():
    times = (np.arange(15000)+1)*.002
    base = dict(times=times, valid=np.ones((15000, 1), dtype=bool),
        length=np.full((15000, 1, 2), .26), motor=np.zeros((15000, 1, 4)))
    rule = dict(same_motor_windows=3, window_p2p_gt_deg=2.,
        reversals_in_3s_at_least=2, each_reverse_excursion_at_least_deg=1.)
    base['motor'][:, 0, 0] = np.deg2rad(3.*np.sin(2.*np.pi*times))
    selected = windows(base, 0, rule)[:3]
    assert sustained(base, 0, selected, rule)['detected']
    assert all(item['observed_samples'] == 500 for item in selected)
    base['motor'][:, 0, 0] = np.deg2rad(4.*times)
    assert not sustained(base, 0, windows(base, 0, rule)[:3], rule)['detected']
    assert not sustained(base, 0, windows(base, 0, rule)[:2], rule)['coverage_sufficient']
    assert merge_segments(dict(acceleration=1., segments=[dict(start=0., end=2., speed=0.),
        dict(start=2., end=30., speed=-0.)])) == [dict(start=0., end=30., speed=0., acceleration=1.)]
    sentinel = dict(meta=dict(feedback_branch='parent', checkpoint_sha256=PARENT_SHA, seed=43,
        cases=[dict(name='sentinel', acceleration=1., env_ids=list(range(96)),
            segments=[dict(start=0., end=30., speed=0., acceleration=1.)])]),
        observation=np.zeros((3000, 96, 25), dtype=np.float32),
        policy_valid=np.ones((3000, 96), dtype=bool), folder=Path(__file__).resolve().parent,
        files={})
    locked = make_lock(sentinel, Path(__file__))
    assert json.loads(json.dumps(locked)) == locked, 'Window lock must survive JSON write/read without key-type changes'
    return dict(verified=True, true_recurrence_detected=True, monotonic_large_motion_not_called_recurrent=True,
        short_segment_coverage_insufficient=True, t0_nonoverlapping_500_samples=True,
        constant_zero_segments_merged=True, parent_window_lock_JSON_roundtrip=True,
        optimizer_updates=0, network_inference=False, simulation=False)


def main(args):
    args.execution_card = args.execution_card.resolve()
    card = read(args.execution_card)
    assert card['parent_checkpoint_sha256'] == PARENT_SHA
    baseline = load(args.baseline)
    validate_schedule(baseline, card)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.lock_only:
        assert args.candidate is None and args.window_lock is None
        write(args.output/'window-lock.json', make_lock(baseline, args.execution_card))
        print(json.dumps(dict(locked=True, seed=baseline['meta']['seed'], optimizer_steps=0,
            window_lock=str((args.output/'window-lock.json').resolve())), ensure_ascii=False), flush=True)
        return
    assert args.candidate is not None and args.window_lock is not None
    locked = read(args.window_lock)
    assert locked == make_lock(baseline, args.execution_card), 'Baseline/card/window definition changed after lock'
    candidate = load(args.candidate)
    validate_schedule(candidate, card)
    pair = validate_pair(baseline, candidate, args.execution_card, args.window_lock)
    result = compare(baseline, candidate, locked, card)
    result.update(verified=True, pairing=pair, baseline_files=baseline['files'], candidate_files=candidate['files'],
        execution_card_sha256=sha(args.execution_card), comparator_source_sha256=sha(__file__),
        optimizer_updates=0, network_inference=False, simulation=False)
    write(args.output/'comparison.json', result)
    print(json.dumps(dict(verified=True, passed=result['passed'], failure_counts=result['failure_counts'],
        original_survived=sum(not item for item in baseline['done']), candidate_survived=sum(not item for item in candidate['done']),
        output=str(args.output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--execution-card', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--window-lock', type=Path)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        assert all(value is None for value in (args.baseline, args.candidate, args.execution_card, args.output, args.window_lock)) and not args.lock_only
        print(json.dumps(selfcheck(), ensure_ascii=False))
    else:
        assert all(value is not None for value in (args.baseline, args.execution_card, args.output))
        main(args)
