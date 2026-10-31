#!/usr/bin/env python3
"""Score a frozen sensor-only velocity formula on two separate saved runs, CPU only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_CANDIDATE_SHA256 = '522be70ee5e35960441d241d9bf69643464609eeff281a166a106493324bdb19'
EXPECTED_GEOMETRY_CODE_SHA256 = '9e0c51d7699e1e753a58a75afedaeab6fca47a710deaea87b7394e7eefa56e0f'
METHODS = ('encoder', 'wheel', 'compensated')
ERROR_KEYS = ('mae_m_s', 'p95_absolute_error_m_s', 'bias_m_s', 'abs_bias_m_s', 'maximum_absolute_error_m_s')
GATE_SPEC = dict(
    version='walk-sensor-velocity-offline-entry-v1',
    status='Technical diagnostic entry gate only; not user-confirmed task acceptance.',
    parking=dict(mae_max_m_s=.020, p95_max_m_s=.050, abs_bias_max_m_s=.010,
        relative_rule_encoder_mae_at_least_m_s=.010, maximum_encoder_mae_ratio=.70,
        maximum_encoder_p95_ratio=.80),
    cruise=dict(maximum_mae_increase_vs_encoder_m_s=.002, maximum_p95_increase_vs_encoder_m_s=.005),
    transition=dict(maximum_encoder_mae_ratio=.80, maximum_encoder_p95_ratio=1.,
        applies_to='Real requested-speed changes only; unchanged zero-speed segments have no transition gate.'),
    initial=dict(start_s=0., end_s=1.5, maximum_mae_increase_vs_encoder_m_s=.002,
        maximum_p95_increase_vs_encoder_m_s=.005, maximum_peak_increase_vs_encoder_m_s=.020),
    phase=dict(settle_after_ramp_s=1.5, command_target_tolerance_m_s=1e-4,
        rule='For a real transition, use max(nominal ramp end, first observed Actor vx command reaching target)+1.5s. Unreached/missing platforms fail coverage. Request/sensor command only, never target velocity labels.'),
    lag=dict(search_max_ms=150, resolution_ms=10, minimum_effective_duration_s=1.,
        minimum_true_std_m_s=.005, minimum_peak_correlation=.6,
        near_optimal_correlation_gap=.02, maximum_near_optimal_peak_width_ms=40,
        positive_sign='Prediction lags truth: corr(truth[t], prediction[t+shift]).',
        rules='Separate complete contiguous phases; constant/low-excitation, short, low-correlation, boundary and broad peaks are unknown. No time shift is used in error scores.'),
    decision='Every environment and required complete segment must satisfy its gate in both runs; no case regression, missing required phase, candidate invalid sample or baseline episode failure is averaged away.',
    fitting='None; fixed analytical formula and source hashes. No bias fitting, filtering, fusion, parameter search or validation-driven repair.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_candidate_source(path):
    # 原公式锁不变；只接受本次目录迁移导致的 ROOT 层级修改。
    with tarfile.open(ROOT/'scripts/archive/20261101-script-cleanup/source.tar.gz') as bundle:
        original = bundle.extractfile('scripts/agent/walk_sensor_velocity.py').read()
    assert hashlib.sha256(original).hexdigest() == EXPECTED_CANDIDATE_SHA256
    relocated = original.replace(b'Path(__file__).resolve().parents[2]', b'Path(__file__).resolve().parents[3]')
    assert relocated != original and path.read_bytes() == relocated, 'Unexpected candidate source change'
    return dict(original_sha256=EXPECTED_CANDIDATE_SHA256, executed_sha256=digest(path),
                allowed_change='Repository ROOT parent index 2 -> 3 only; formula bytes unchanged')


def write_json(path, value):
    with path.open('x') as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def describe(values, worst_mode='maximum'):
    observed = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if not observed:
        return dict(observed=0, missing=len(values), median=None, minimum=None, maximum=None, worst=None, worst_index=None)
    a = np.asarray([v for _, v in observed])
    index = int(np.argmax(abs(a))) if worst_mode == 'absolute' else int(np.argmin(a) if worst_mode == 'minimum' else np.argmax(a))
    return dict(observed=len(a), missing=len(values)-len(a), median=float(np.median(a)),
        minimum=float(a.min()), maximum=float(a.max()), worst=float(a[index]), worst_index=observed[index][0])


def read_context(folder, role):
    folder = folder.resolve()
    metadata, trace = folder/'encoder.json', folder/'encoder.npz'
    final_path = folder/'result.json'
    final = json.loads(final_path.read_text())
    assert final['verified'] is True, 'Incomplete collection'
    meta = json.loads(metadata.read_text())
    assert meta['schedule'] == role and meta['seed'] == (44 if role == 'discovery' else 45)
    assert meta['feedback_mode'] == meta['velocity_source'] == 'encoder'
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] is True
    assert meta['encoder_unchanged'] is True
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
    assert meta['seconds'] == 30. and meta['policy_dt'] == .01
    scene = Path(meta['scene_mjb']).resolve()
    assert digest(scene) == meta['scene_sha256']
    checkpoint = Path(meta['checkpoint']).resolve()
    assert digest(checkpoint) == meta['checkpoint_sha256']
    with np.load(trace, allow_pickle=False) as saved:
        # Deliberately load only estimator inputs here; target/encoder labels come after both predictions.
        observation = saved['policy_observation']
        nominal = saved['fixed_nominal']
        times = saved['policy_time_s']
        valid = saved['policy_valid']
        final_live = saved['qpos_live'][-1]
        arrays = {key: saved['fixed_'+key] for key in ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
                  if 'fixed_'+key in saved.files}
    assert observation.shape == (3000, 96, 25) and nominal.shape == (96, 6)
    assert valid.shape == (3000, 96) and valid.dtype == np.bool_
    assert times.shape == (3000,) and np.allclose(times, np.arange(3000)*.01, rtol=0., atol=1e-10)
    assert np.isfinite(observation).all() and np.isfinite(nominal).all()
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    done = np.asarray(meta['episode_done'], dtype=bool)
    ends = np.asarray(meta['episode_end'])
    assert done.shape == ends.shape == (96,)
    assert np.array_equal(final_live, ~done) and int(final_live.sum()) == meta['survived']
    assert np.array_equal(valid.sum(0), np.rint(ends/.01).astype(int))
    assert not ((ends < 30.-1e-6) & ~done).any()
    return dict(folder=folder, metadata=meta, trace=trace, scene=scene,
        observation=observation, nominal=nominal, times=times, valid=valid, done=done,
        inputs={str(p): digest(p) for p in (metadata, trace, final_path, scene)},
        fixed_arrays={key: dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())
                      for key, a in arrays.items()})


def predict_with_invalid(estimator, observation, nominal, eligible, progress_name):
    shape = observation.shape[:2]
    flat_obs = observation.reshape(-1, 25)
    flat_nominal = np.broadcast_to(nominal, (*shape, 6)).reshape(-1, 6)
    indices = np.flatnonzero(eligible.reshape(-1))
    values = np.full(shape[0]*shape[1], np.nan)
    invalid, examples = Counter(), []

    def batch(ids):
        if not len(ids):
            return
        try:
            predicted = np.asarray(estimator.predict(flat_obs[ids], flat_nominal[ids]), dtype=float)
            assert predicted.shape == (len(ids),), 'Candidate output shape differs'
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            if len(ids) > 1:
                midpoint = len(ids)//2
                batch(ids[:midpoint])
                batch(ids[midpoint:])
                return
            message = str(exc)[:500]
            invalid[message] += 1
            if len(examples) < 100:
                examples.append(dict(tick=int(ids[0]//shape[1]), env=int(ids[0]%shape[1]), reason=message))
            return
        finite = np.isfinite(predicted)
        values[ids[finite]] = predicted[finite]
        for index in ids[~finite]:
            invalid['Candidate returned nonfinite vx'] += 1
            if len(examples) < 100:
                examples.append(dict(tick=int(index//shape[1]), env=int(index%shape[1]), reason='Candidate returned nonfinite vx'))

    last = time.monotonic()
    print(progress_name, 'sensor-only prediction begins;', len(indices), 'eligible samples', flush=True)
    for offset in range(0, len(indices), 4096):
        batch(indices[offset:offset+4096])
        now = time.monotonic()
        if now-last >= 15.:
            print(progress_name, min(offset+4096, len(indices)), '/', len(indices), 'samples; invalid', sum(invalid.values()), flush=True)
            last = now
    values = values.reshape(shape)
    available = eligible & np.isfinite(values)
    print(progress_name, 'sensor-only prediction complete; invalid', int((eligible & ~available).sum()), flush=True)
    return values, dict(invalid_samples=int((eligible & ~available).sum()), reasons=dict(invalid), examples=examples,
        per_environment_invalid_samples=(eligible & ~available).sum(0).tolist())


def lag_measurement(predicted, truth):
    rule = GATE_SPEC['lag']
    n, margin = len(truth), rule['search_max_ms']//rule['resolution_ms']
    result = dict(lag_ms=None, raw_best_lag_ms=None, raw_best_correlation=None,
        true_std_m_s=None if not n else float(np.std(truth)), effective_duration_s=max(0, n-2*margin)*.01,
        time_resolution_ms=10, derived_truth_timing_uncertainty_ms=2, unknown_reasons=[])
    if n-2*margin < round(rule['minimum_effective_duration_s']/.01):
        result['unknown_reasons'].append('less_than_1s_common_overlap')
    if not n or np.std(truth) < rule['minimum_true_std_m_s']:
        result['unknown_reasons'].append('truth_has_insufficient_excitation')
    if result['unknown_reasons']:
        return result
    target = truth[margin:n-margin]
    target = target-target.mean()
    if np.std(target) < rule['minimum_true_std_m_s']:
        result['unknown_reasons'].append('common_overlap_has_insufficient_excitation')
        return result
    scores = []
    for shift in range(-margin, margin+1):
        candidate = predicted[margin+shift:n-margin+shift]
        candidate = candidate-candidate.mean()
        denominator = np.linalg.norm(target)*np.linalg.norm(candidate)
        scores.append(None if denominator < 1e-15 else float(np.dot(target, candidate)/denominator))
    finite = [i for i, score in enumerate(scores) if score is not None]
    if not finite:
        result['unknown_reasons'].append('prediction_has_no_variation')
        return result
    best = max(finite, key=lambda i: (scores[i], -abs(i-margin)))
    shift, correlation = best-margin, scores[best]
    result.update(raw_best_lag_ms=shift*10, raw_best_correlation=correlation,
        raw_correlations=[dict(lag_ms=(i-margin)*10, correlation=score) for i, score in enumerate(scores)])
    near = [i for i in finite if scores[i] >= correlation-rule['near_optimal_correlation_gap']]
    result['near_optimal_peak_width_ms'] = (max(near)-min(near))*10
    result['identifiability_warning'] = 'Broad or repeated peak; do not interpret as a precise physical delay.' if result['near_optimal_peak_width_ms'] > rule['maximum_near_optimal_peak_width_ms'] else None
    if correlation < rule['minimum_peak_correlation']:
        result['unknown_reasons'].append('peak_correlation_below_0_6')
    if abs(shift) == margin:
        result['unknown_reasons'].append('best_lag_at_search_boundary')
    if result['near_optimal_peak_width_ms'] > rule['maximum_near_optimal_peak_width_ms']:
        result['unknown_reasons'].append('broad_peak')
    if not result['unknown_reasons']:
        result['lag_ms'] = shift*10
    return result


def measurement(predicted, truth, valid, mask):
    expected = int(mask.sum())
    observed = mask & valid & np.isfinite(truth) & np.isfinite(predicted)
    complete = bool(expected and observed.sum() == expected)
    result = dict(expected_samples=expected, observed_samples=int(observed.sum()), complete=complete,
                  metrics=None, lag=None)
    if not complete:
        result['lag'] = dict(lag_ms=None, raw_best_lag_ms=None, raw_best_correlation=None,
                             unknown_reasons=['incomplete_or_missing_phase'])
        return result
    error = predicted[mask]-truth[mask]
    result['metrics'] = dict(mae_m_s=float(abs(error).mean()), p95_absolute_error_m_s=float(np.percentile(abs(error), 95)),
        bias_m_s=float(error.mean()), abs_bias_m_s=float(abs(error.mean())), maximum_absolute_error_m_s=float(abs(error).max()))
    result['lag'] = lag_measurement(predicted[mask], truth[mask])
    return result


def gate(kind, rows):
    encoder, compensated = rows['encoder']['metrics'], rows['compensated']['metrics']
    if encoder is None or compensated is None:
        return dict(passed=False, failed_checks=['incomplete_encoder_or_compensated_phase'])
    checks = {}
    if kind == 'parking':
        spec = GATE_SPEC['parking']
        checks.update(mae_absolute=compensated['mae_m_s'] <= spec['mae_max_m_s'],
            p95_absolute=compensated['p95_absolute_error_m_s'] <= spec['p95_max_m_s'],
            bias_absolute=compensated['abs_bias_m_s'] <= spec['abs_bias_max_m_s'])
        if encoder['mae_m_s'] >= spec['relative_rule_encoder_mae_at_least_m_s']:
            checks.update(mae_relative=compensated['mae_m_s'] <= spec['maximum_encoder_mae_ratio']*encoder['mae_m_s'],
                p95_relative=compensated['p95_absolute_error_m_s'] <= spec['maximum_encoder_p95_ratio']*encoder['p95_absolute_error_m_s'])
    elif kind in ('cruise', 'initial'):
        spec = GATE_SPEC[kind]
        checks.update(mae_preserved=compensated['mae_m_s'] <= encoder['mae_m_s']+spec['maximum_mae_increase_vs_encoder_m_s'],
            p95_preserved=compensated['p95_absolute_error_m_s'] <= encoder['p95_absolute_error_m_s']+spec['maximum_p95_increase_vs_encoder_m_s'])
        if kind == 'initial':
            checks['peak_preserved'] = compensated['maximum_absolute_error_m_s'] <= encoder['maximum_absolute_error_m_s']+spec['maximum_peak_increase_vs_encoder_m_s']
    elif kind in ('start', 'brake'):
        spec = GATE_SPEC['transition']
        checks.update(mae_improved=compensated['mae_m_s'] <= spec['maximum_encoder_mae_ratio']*encoder['mae_m_s'],
            p95_not_worse=compensated['p95_absolute_error_m_s'] <= spec['maximum_encoder_p95_ratio']*encoder['p95_absolute_error_m_s'])
    else:
        raise ValueError(kind)
    return dict(passed=all(checks.values()), checks=checks, failed_checks=[key for key, value in checks.items() if not value])


def merged_segments(case):
    result = []
    for segment in case['segments']:
        row = dict(start=float(segment.get('start_s', segment.get('start'))),
            end=float(segment.get('end_s', segment.get('end'))), speed=float(segment['speed']),
            acceleration=float(segment.get('acceleration', case['acceleration'])))
        assert row['end'] > row['start'] and row['acceleration'] > 0.
        if result:
            assert abs(result[-1]['end']-row['start']) < 1e-9
        if result and row['speed'] == result[-1]['speed']:
            result[-1]['end'] = row['end']
        else:
            result.append(row)
    assert result[0]['start'] == 0. and result[0]['speed'] == 0. and result[-1]['end'] == 30.
    return result


def phase_specs(segments, times, command, valid):
    result = []
    initial_end = GATE_SPEC['initial']['end_s']
    for index, segment in enumerate(segments):
        start, end, speed = segment['start'], segment['end'], segment['speed']
        if index == 0:
            assert end >= initial_end
            result.append(dict(segment=index, kind='initial', start=0., end=initial_end, requested_speed=0.))
            result.append(dict(segment=index, kind='parking', start=initial_end, end=end, requested_speed=0.))
            continue
        previous = segments[index-1]['speed']
        nominal_end = start+abs(speed-previous)/segment['acceleration']
        in_segment = (times >= start-1e-9) & (times < end-1e-9) & valid
        reached = np.flatnonzero(in_segment & (abs(command-speed) <= GATE_SPEC['phase']['command_target_tolerance_m_s']))
        observed_end = None if not len(reached) else float(times[reached[0]])
        platform = None if observed_end is None else max(nominal_end, observed_end)+1.5
        transition_end = end if platform is None else min(end, platform)
        detail = dict(segment=index, requested_speed=speed, previous_requested_speed=previous,
            nominal_ramp_end_s=nominal_end, observed_command_target_reached_s=observed_end,
            platform_start_s=platform)
        result.append(dict(**detail, kind='brake' if speed == 0. else 'start', start=start, end=transition_end))
        result.append(dict(**detail, kind='parking' if speed == 0. else 'cruise',
            start=end if platform is None else min(end, platform), end=end))
    return result


def aggregate_phase(rows):
    result = dict(environments=len(rows), gate_passed_environments=sum(row['gate']['passed'] for row in rows), methods={})
    for method in METHODS:
        measures = [r['methods'][method] for r in rows]
        result['methods'][method] = dict(complete_environments=sum(m['complete'] for m in measures),
            metrics={key: describe([None if m['metrics'] is None else m['metrics'][key] for m in measures],
                                  'absolute' if key == 'bias_m_s' else 'maximum') for key in ERROR_KEYS},
            lag_ms=describe([m['lag']['lag_ms'] for m in measures], 'absolute'),
            raw_peak_correlation=describe([m['lag'].get('raw_best_correlation') for m in measures], 'minimum'),
            lag_unknown_reasons=dict(Counter(reason for m in measures for reason in m['lag']['unknown_reasons'])))
    return result


def score_run(context):
    with np.load(context['trace'], allow_pickle=False) as saved:
        # Scoring labels are opened only after the two runs' candidate arrays are frozen.
        encoder, truth, used = saved['encoder_m_s'], saved['target_com_m_s'], saved['used_vx_m_s']
    assert encoder.shape == truth.shape == (3000, 96, 3) and used.shape == (3000, 96)
    assert np.isfinite(encoder).all() and np.isfinite(truth).all() and np.isfinite(used).all()
    assert np.allclose(used[context['valid']], encoder[..., 0][context['valid']], rtol=0., atol=1e-7)
    observation, times, valid = context['observation'], context['times'], context['valid']
    predictions = dict(encoder=encoder[..., 0], wheel=.03*(observation[..., 15]-observation[..., 18])/.05,
                       compensated=context['candidate'])
    cases, ids_all, failures = [], [], Counter()
    for case in context['metadata']['cases']:
        segments = merged_segments(case)
        ids = case['env_ids']
        ids_all.extend(ids)
        environments = []
        for env in ids:
            specifications = phase_specs(segments, times, observation[:, env, 6]/2., valid[:, env])
            phases = []
            for spec in specifications:
                mask = (times >= spec['start']-1e-9) & (times < spec['end']-1e-9)
                methods = {method: measurement(predicted[:, env], truth[:, env, 0], valid[:, env], mask)
                           for method, predicted in predictions.items()}
                decision = gate(spec['kind'], methods)
                if not decision['passed']:
                    failures[spec['kind']] += 1
                command = observation[mask, env, 6]/2.
                phases.append(dict(**spec, methods=methods, gate=decision,
                    observed_actor_command_min_m_s=None if not len(command) else float(command.min()),
                    observed_actor_command_max_m_s=None if not len(command) else float(command.max())))
            environments.append(dict(env=env, baseline_episode_survived=bool(not context['done'][env]),
                episode_end_s=context['metadata']['episode_end'][env], phases=phases,
                gate_passed=not context['done'][env] and all(phase['gate']['passed'] for phase in phases)))
        summaries = {}
        for segment in range(len(segments)):
            for kind in ('initial', 'parking', 'cruise', 'start', 'brake'):
                rows = [phase for e in environments for phase in e['phases'] if phase['segment'] == segment and phase['kind'] == kind]
                if rows:
                    summaries[f'segment{segment}_{kind}'] = aggregate_phase(rows)
        worst = {}
        for method in METHODS:
            worst[method] = {}
            for key in ('mae_m_s', 'p95_absolute_error_m_s', 'abs_bias_m_s', 'maximum_absolute_error_m_s'):
                values = []
                for env in environments:
                    observations = [p['methods'][method]['metrics'][key] for p in env['phases']
                                    if p['methods'][method]['metrics'] is not None]
                    values.append(None if not observations else max(observations))
                worst[method][key] = describe(values)
        cases.append(dict(case=case['name'], speed=case['speed'], acceleration=case['acceleration'],
            merged_requested_segments=segments, environments=len(ids), per_environment=environments,
            segment_phase_summaries=summaries, worst_complete_phase_per_environment=worst,
            gate_passed=all(e['gate_passed'] for e in environments),
            gate_passed_environments=sum(e['gate_passed'] for e in environments)))
    assert sorted(ids_all) == list(range(96))
    return dict(schedule=context['metadata']['schedule'], seed=context['metadata']['seed'],
        input_files=context['inputs'], checkpoint_sha256=context['metadata']['checkpoint_sha256'],
        actor_state_sha256=context['metadata']['actor_state_sha256'], parameter_updates=0,
        baseline_survived=int((~context['done']).sum()), saved_fixed_arrays=context['fixed_arrays'],
        candidate_formula_spec=context['estimator'].formula_spec, candidate_invalid=context['invalid'],
        causal_prefix_invariance=context['prefix_invariance'], cases=cases,
        failed_gate_phase_counts=dict(failures),
        gate_passed=all(c['gate_passed'] for c in cases) and context['invalid']['invalid_samples'] == 0)


def main(args):
    import mujoco
    from scripts.agent.split.walk_sensor_velocity import SensorWheelVelocity
    assert args.discovery.resolve() != args.validation.resolve(), 'Separate recordings are required'
    candidate_source = ROOT/'scripts/agent/split/walk_sensor_velocity.py'
    geometry_source = ROOT/'sim2sim/chuanliantui_closed_adapter.py'
    source_migration = verify_candidate_source(candidate_source)
    assert digest(geometry_source) == EXPECTED_GEOMETRY_CODE_SHA256
    formula_lock_path = args.discovery.resolve().parent/'formula-lock.json'
    assert formula_lock_path == args.validation.resolve().parent/'formula-lock.json'
    formula_lock = json.loads(formula_lock_path.read_text())
    assert formula_lock['formula_sha256'] == EXPECTED_CANDIDATE_SHA256
    assert formula_lock['dependency_sha256'] == EXPECTED_GEOMETRY_CODE_SHA256
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = dict(gate_spec_sha256=stable_digest(GATE_SPEC), analyzer_sha256=digest(Path(__file__)),
        candidate_sha256=digest(candidate_source), candidate_source_migration=source_migration,
        geometry_code_sha256=digest(geometry_source),
        formula_lock_path=str(formula_lock_path), formula_lock_sha256=digest(formula_lock_path),
        gate_locked_before_loading_scoring_labels=True, gate_spec=GATE_SPEC)
    write_json(args.output/'gate-lock.json', lock)
    contexts = [read_context(args.discovery, 'discovery'), read_context(args.validation, 'validation')]
    assert contexts[0]['metadata']['checkpoint_sha256'] == contexts[1]['metadata']['checkpoint_sha256']
    assert contexts[0]['metadata']['actor_state_sha256'] == contexts[1]['metadata']['actor_state_sha256']
    for context in contexts:
        context['estimator'] = SensorWheelVelocity(mujoco.MjModel.from_binary_path(str(context['scene'])))
    geometry_hashes = [c['estimator'].formula_spec['geometry_sha256'] for c in contexts]
    assert geometry_hashes[0] == geometry_hashes[1], 'Different nominal geometry; no scoring performed'
    assert geometry_hashes[0] == formula_lock['formula_spec']['geometry_sha256'], 'Nominal geometry differs from preregistered formula lock'
    for context in contexts:
        role = context['metadata']['schedule']
        prediction, invalid = predict_with_invalid(context['estimator'], context['observation'],
            context['nominal'], context['valid'], role)
        context['candidate'], context['invalid'] = prediction, invalid
        fresh = SensorWheelVelocity(mujoco.MjModel.from_binary_path(str(context['scene'])))
        prefix, _ = predict_with_invalid(fresh, context['observation'][:2], context['nominal'], context['valid'][:2], role+' prefix')
        assert np.allclose(prefix, prediction[:2], rtol=0., atol=1e-12, equal_nan=True), 'Prediction depends on later inputs or batch context'
        context['prefix_invariance'] = dict(checked=True, prefix_policy_steps=2,
            output_identical_atol=1e-12, model_reinstantiated=True,
            interpretation='Fixed memoryless formula; a short fresh-instance prefix matches full-record prediction. Labels have not yet been loaded.')
        candidate_file = args.output/(role+'-candidate.npz')
        np.savez_compressed(candidate_file, candidate_vx_m_s=prediction,
            available=context['valid'] & np.isfinite(prediction), policy_time_s=context['times'])
        write_json(args.output/(role+'-prediction-receipt.json'), dict(
            candidate_file=str(candidate_file), candidate_sha256=digest(candidate_file),
            source_sha256=digest(candidate_source), geometry_sha256=context['estimator'].formula_spec['geometry_sha256'],
            scoring_labels_loaded=False, invalid=invalid))
    print('Both sensor-only prediction artifacts are frozen; scoring labels are now opened once.', flush=True)
    runs = [score_run(context) for context in contexts]
    assert verify_candidate_source(candidate_source) == source_migration
    assert digest(geometry_source) == EXPECTED_GEOMETRY_CODE_SHA256
    assert digest(Path(__file__)) == lock['analyzer_sha256']
    passed = all(run['gate_passed'] for run in runs)
    report = dict(verified=True, physics_steps=0, parameter_updates=0,
        gate_lock=lock, gate_spec_sha256=lock['gate_spec_sha256'], nominal_geometry_sha256=geometry_hashes[0],
        gate_passed=passed, closed_loop_entry_permitted_by_this_diagnostic=passed,
        decision='Eligible for the preregistered frozen closed-loop comparison; not task acceptance.' if passed else 'Reject closed-loop promotion of this frozen candidate; do not add filters or tune on these validation labels.',
        no_leakage=dict(estimator_inputs_only=['current Actor25 sensor packet', 'fixed PD nominal motor positions', 'fixed nominal geometry'],
            scoring_label='Same-policy-boundary root-body inertial COM vx in body frame, never passed to estimator.',
            label_arrays_loaded_after_both_predictions=True, fitted_parameters=0, normalization_fitted_on_data=False,
            no_privileged_qpos_or_passive_joint_inputs=True, no_contact_truth_gating=True,
            causal_prefix_invariance_checked=True, no_future_filter=True,
            separate_recordings=dict(discovery=str(args.discovery.resolve()), validation=str(args.validation.resolve()),
                seeds=[44, 45], schedules=['discovery', 'validation']),
            caveat='Different recordings and commands are nominal simulation validation, not an independent robot or randomized robustness guarantee. Environment replicas are not treated as independent statistics.'),
        timing='100Hz pre-action estimator packets and pre-action derived COM truth. Derived truth may lag raw sensors <=2ms. Lag is descriptive on a10ms grid, never used to align error scores.',
        aggregation='Per-environment, per-complete-segment gate; medians and worst values describe distributions. No overall mean controls promotion. Missing or failed episodes retain their denominator.',
        no_task_acceptance_claim=True, runs=runs)
    write_json(args.output/'results.json', report)
    print(json.dumps(dict(verified=True, results=str(args.output/'results.json'), gate_passed=passed,
        runs=[dict(schedule=r['schedule'], gate_passed=r['gate_passed'],
            invalid=r['candidate_invalid']['invalid_samples'], baseline_survived=r['baseline_survived'],
            failed_gate_phase_counts=r['failed_gate_phase_counts']) for r in runs]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
