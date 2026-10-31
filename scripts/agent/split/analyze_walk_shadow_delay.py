#!/usr/bin/env python3
"""Replay fixed grouped sensor delays through the frozen 2999 encoder only."""

import argparse
import ast
from collections import Counter
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.agent.split import analyze_walk_velocity_compensation as phase_helpers


ROOT = Path(__file__).resolve().parents[3]
CARD = ROOT/'artifacts/walk25-shadow-delay-20261031-r1/experiment-card.json'
EXPECTED_MODEL_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
COMBINATIONS = tuple(itertools.product((3, 5, 8), repeat=4))
METRICS = ('mae_m_s', 'p95_absolute_error_m_s', 'bias_m_s', 'abs_bias_m_s',
           'peak_absolute_error_m_s', 'error_band_1_3Hz_rms_m_s')
LAG_REASONS = ('less_than_1s_common_overlap', 'truth_has_insufficient_excitation',
    'common_overlap_has_insufficient_excitation', 'prediction_has_no_variation',
    'peak_correlation_below_0_6', 'best_lag_at_search_boundary', 'broad_peak',
    'incomplete_or_missing_phase')
GATE_SPEC = dict(
    formula='For group g: S[max(0,5*k-lag[g])]; native current commands6:9 and previous actions19:25; Actor25 scaling/reordering unchanged; five frames oldest to newest with reset backfill.',
    tuples=COMBINATIONS, weights='equal 1/81 on individual combination metrics, never mean predictions',
    initial_end_s=1.5, settle_after_native_and_nominal_ramp_s=1.5,
    parking=dict(mae_ratio_max=.8, abs_bias_ratio_max=.8, error_band_rms_ratio_max=.8,
        small_baseline_threshold_m_s=.001, small_baseline_absolute_allowance_m_s=.0005,
        p95_increase_max_m_s=0., minimum_complete_band_duration_s=3.),
    nonregression=dict(mae_increase_max_m_s=.002, p95_increase_max_m_s=.005,
        abs_bias_increase_max_m_s=.002, peak_increase_max_m_s=.020),
    reconstruction=dict(actor25_max_abs=1e-7, history125_max_abs=1e-7,
        encoder_m_s_max_abs=1e-5),
    coverage='All phases, environments, all 81 tuples, both runs; failed/missing remain failures. At least one complete >=3s parking phase per environment.',
    lag='Descriptive only, 10ms grid +/-150ms; unknown for <1s overlap, truth std<.005, correlation<.6, boundary or near-optimal peak >40ms. Never align scoring labels.',
    scope='Network-output input-delay effect only; no training, control changes, behavior improvement or compensation-spike causal claim.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def actor_frames(raw, native, lag):
    ticks = np.arange(len(native))*5
    sensors = np.concatenate([raw[np.maximum(0, ticks-delay), :, columns]
        for delay, columns in zip(lag, (slice(0, 3), slice(3, 6), slice(6, 12), slice(12, 18)))], axis=-1)
    # Commands and previous raw actions retain native current-policy values.
    frames = np.concatenate((sensors[..., :3]*np.float32(.25), sensors[..., 3:6],
        native[..., 6:9], sensors[..., 6:10],
        sensors[..., 12:18][..., (0, 1, 4, 2, 3, 5)]*np.float32(.05), native[..., 19:25]), axis=-1)
    return np.clip(frames, -100., 100.)


def histories(frames):
    ids = np.maximum(0, np.arange(len(frames))[:, None]+np.arange(-4, 1))
    return frames[ids].transpose(0, 2, 1, 3).reshape(len(frames), frames.shape[1], 125)


def extract_delay_class(path):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'GroupedSensorDelay')
    namespace = dict(torch=torch, SENSOR_SLICES=(slice(0, 3), slice(3, 6), slice(6, 12), slice(12, 18)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['GroupedSensorDelay']


def synthetic_check(delay_class, circular_class):
    # Compare vectorized indexing with actual native 500Hz push and history append.
    combinations = ((0, 0, 0, 0),)+COMBINATIONS
    batch = len(combinations)
    raw = np.arange(41*batch*18, dtype=np.float32).reshape(41, batch, 18)/np.float32(4096.)
    native = np.arange(9*batch*25, dtype=np.float32).reshape(9, batch, 25)/np.float32(8192.)
    delay = delay_class(torch.from_numpy(raw[0]), 8)
    delay.lags.copy_(torch.tensor(combinations))
    history = circular_class(5, batch, 'cpu')
    expected = np.stack([actor_frames(raw[:, env:env+1], native[:, env:env+1], lag)[:, 0]
                         for env, lag in enumerate(combinations)], axis=1)
    expected_history = histories(expected)
    last = 0
    frame_max, history_max = 0., 0.
    for tick in range(9):
        for sample in range(last+1, tick*5+1):
            delay.push(torch.from_numpy(raw[sample]))
        last = tick*5
        sensors = delay.read().numpy()
        current = np.concatenate((sensors[:, :3]*np.float32(.25), sensors[:, 3:6], native[tick, :, 6:9],
            sensors[:, 6:10], sensors[:, 12:18][:, (0, 1, 4, 2, 3, 5)]*np.float32(.05),
            native[tick, :, 19:25]), axis=-1).clip(-100., 100.)
        history.append(torch.from_numpy(current))
        frame_max = max(frame_max, float(abs(current-expected[tick]).max()))
        history_max = max(history_max, float(abs(history.buffer.numpy().reshape(batch, 125)-expected_history[tick]).max()))
    assert frame_max == history_max == 0.
    return dict(combinations=batch, physics_samples=41, policy_frames=9,
        actor25_max_abs=frame_max, history125_max_abs=history_max, passed=True)


def read_context(folder, role):
    folder = folder.resolve()
    meta_path, receipt_path, trace = folder/'encoder.json', folder/'result.json', folder/'encoder.npz'
    meta, receipt = json.loads(meta_path.read_text()), json.loads(receipt_path.read_text())
    assert receipt['verified'] is True and receipt['schedule'] == role
    assert meta['schedule'] == role and meta['seed'] == (46 if role == 'study' else 47)
    assert meta['checkpoint_sha256'] == EXPECTED_MODEL_SHA
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] and meta['encoder_unchanged']
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
    assert meta['feedback_mode'] == meta['velocity_source'] == 'encoder'
    assert meta['seconds'] == 30. and meta['dt'] == .002 and meta['policy_dt'] == .01
    assert not meta['native_observation_cfg']['noise'] and not meta['native_observation_cfg']['delay']
    with np.load(trace, allow_pickle=False) as saved:
        # No privileged target velocity is opened until both runs' predictions are sealed.
        raw, native, history = saved['raw_sensors'], saved['policy_observation'], saved['native_history']
        valid, times = saved['policy_valid'], saved['policy_time_s']
        original = saved['encoder_m_s']
        delayed = saved['native_delayed_sensors_before_policy']
        raw_policy = saved['raw_sensors_before_policy']
        source_lags, source_index = saved['read_source_lags'], saved['read_source_raw_index']
        sensor_valid = saved['raw_sensor_valid']
        final_live, reset_after = saved['qpos_live'][-1], saved['reset_after_policy']
    assert raw.shape == (15001, 96, 18) and native.shape == (3000, 96, 25)
    assert history.shape == (3000, 96, 125) and original.shape == (3000, 96, 3)
    assert valid.shape == (3000, 96) and valid.dtype == np.bool_
    assert np.array_equal(times, np.arange(3000)*.01)
    assert np.isfinite(raw).all() and np.isfinite(native).all() and np.isfinite(history).all()
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    assert np.array_equal(final_live, ~np.asarray(meta['episode_done'], dtype=bool))
    assert int(final_live.sum()) == meta['survived']
    assert np.array_equal(valid.sum(0), np.rint(np.asarray(meta['episode_end'])/.01).astype(int))
    assert not source_lags.any()
    expected_index = np.broadcast_to((np.arange(3000)*5)[:, None, None], source_index.shape)
    assert np.array_equal(source_index, expected_index)
    assert np.array_equal(raw_policy[valid], raw[::5][:-1][valid])
    assert np.array_equal(delayed[valid], raw[::5][:-1][valid])
    assert np.array_equal(sensor_valid[::5][:-1], valid)
    assert reset_after.shape == valid.shape
    checkpoint = Path(meta['checkpoint']).resolve()
    assert digest(checkpoint) == EXPECTED_MODEL_SHA
    runtime = folder/'frozen-runtime/mjlab_training/src/ct_mjlab/envs/chuanliantui/v14_observations.py'
    assert runtime.is_file()
    inputs = {str(path):digest(path) for path in (meta_path, receipt_path, trace, runtime, checkpoint)}
    for item in meta['runtime_dependency_sources'].values():
        assert digest(item['path']) == item['sha256']
        inputs[item['path']] = item['sha256']
    return dict(folder=folder, role=role, meta=meta, trace=trace, raw=raw, native=native,
        native_history=history, valid=valid, times=times, original=original,
        checkpoint=checkpoint, runtime=runtime, inputs=inputs)


@torch.inference_mode()
def predict(encoder, history, device):
    flat = history.reshape(-1, 125)
    output = np.empty((len(flat), 3), dtype=np.float32)
    for begin in range(0, len(flat), 8192):
        end = min(begin+8192, len(flat))
        output[begin:end] = (encoder(torch.from_numpy(flat[begin:end]).to(device))/2.).cpu().numpy()
    return output.reshape(*history.shape[:2], 3)


def predict_run(context, encoder, args):
    role, valid = context['role'], context['valid']
    path = args.output/f'{role}-predictions.npy'
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(82, 3000, 96, 3))
    zero_frames = actor_frames(context['raw'], context['native'], (0, 0, 0, 0))
    zero_history = histories(zero_frames)
    frame_error = float(abs(zero_frames[valid]-context['native'][valid]).max())
    history_error = float(abs(zero_history[valid]-context['native_history'][valid]).max())
    assert frame_error <= 1e-7 and history_error <= 1e-7, 'Reconstruction failure before truth scoring'
    output[0] = predict(encoder, zero_history, args.device)
    network_error = float(abs(output[0][valid]-context['original'][valid]).max())
    assert network_error <= 1e-5, 'Frozen encoder fails native replay before truth scoring'
    del zero_frames, zero_history
    for index, lag in enumerate(COMBINATIONS, start=1):
        frames = actor_frames(context['raw'], context['native'], lag)
        output[index] = predict(encoder, histories(frames), args.device)
        if index % 9 == 0:
            print(role, index, '/81 fixed tuples predicted', flush=True)
    output.flush()
    assert np.isfinite(output).all()
    del output
    context['predictions_path'] = path
    context['reconstruction'] = dict(actor25_max_abs=frame_error, history125_max_abs=history_error,
        encoder_m_s_max_abs=network_error, passed=True)
    # Prediction files are sealed before any privileged scoring labels are read.
    context['predictions_sha256'] = digest(path)
    write(args.output/f'{role}-prediction-receipt.json', dict(
        target_com_opened=False, path=str(path), sha256=context['predictions_sha256'],
        shape=[82, 3000, 96, 3], tuple_order=[[0, 0, 0, 0], *COMBINATIONS],
        reconstruction=context['reconstruction'], inputs=context['inputs']))


def merged_segments(case):
    result = []
    for segment in case['segments']:
        row = dict(start=float(segment['start']), end=float(segment['end']), speed=float(segment['speed']),
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


def phase_measurement(predictions, truth, valid, mask, kind):
    expected = int(mask.sum())
    complete = bool(expected and valid[mask].all() and np.isfinite(truth[mask]).all()
                    and np.isfinite(predictions[:, mask]).all())
    metrics = np.full((82, len(METRICS)), np.nan)
    # lag columns: accepted lag, raw best lag, peak correlation, near-optimal width, unknown reason bitmask.
    lags = np.full((82, 5), np.nan)
    if not complete:
        lags[:, 4] = 1 << LAG_REASONS.index('incomplete_or_missing_phase')
        return metrics, lags, dict(expected_samples=expected, observed_samples=int((mask & valid).sum()), complete=False)
    values, target = predictions[:, mask], truth[mask]
    error = values-target[None, :]
    absolute = abs(error)
    metrics[:, 0] = absolute.mean(axis=1)
    metrics[:, 1] = np.percentile(absolute, 95, axis=1)
    metrics[:, 2] = error.mean(axis=1)
    metrics[:, 3] = abs(metrics[:, 2])
    metrics[:, 4] = absolute.max(axis=1)
    if kind == 'parking' and expected >= 300:
        centered = error-error.mean(axis=1, keepdims=True)
        frequencies = np.fft.rfftfreq(expected, d=.01)
        spectrum = np.fft.rfft(centered, axis=1)
        spectrum[:, (frequencies < 1.) | (frequencies > 3.)] = 0.
        band = np.fft.irfft(spectrum, n=expected, axis=1)
        metrics[:, 5] = np.sqrt(np.mean(band**2, axis=1))
    for index in range(82):
        lag = phase_helpers.lag_measurement(values[index], target)
        # Old helper's derived-field uncertainty is not applicable to this refreshed pre-action capture.
        lags[index] = [np.nan if lag.get(key) is None else lag[key] for key in
            ('lag_ms', 'raw_best_lag_ms', 'raw_best_correlation', 'near_optimal_peak_width_ms')] + [
            sum(1 << LAG_REASONS.index(reason) for reason in lag['unknown_reasons'])]
    return metrics, lags, dict(expected_samples=expected, observed_samples=expected, complete=True)


def metric_dict(values):
    return {key:None if not np.isfinite(value) else float(value) for key,value in zip(METRICS, values)}


def phase_gates(metrics, complete):
    if not complete:
        return np.zeros(81, dtype=bool), dict(incomplete=81)
    base, delayed = metrics[0], metrics[1:]
    checks = np.stack((delayed[:, 0] <= base[0]+.002, delayed[:, 1] <= base[1]+.005,
        delayed[:, 3] <= base[3]+.002, delayed[:, 4] <= base[4]+.020), axis=-1)
    return checks.all(axis=1), {key:int((~checks[:, index]).sum()) for index,key in
        enumerate(('mae', 'p95', 'absolute_bias', 'peak'))}


def parking_gate(base, delayed):
    expected = np.mean(delayed, axis=0)
    expected[5] = np.sqrt(np.mean(delayed[:, 5]**2))
    checks = dict(mae=expected[0] <= .8*base[0], p95=expected[1] <= base[1])
    for key, index in (('absolute_bias', 3), ('error_band_1_3Hz_rms', 5)):
        checks[key] = bool(expected[index] <= (base[index]+.0005 if base[index] < .001 else .8*base[index]))
    return expected, {key:bool(value) for key,value in checks.items()}


def describe(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=len(values), minimum=None if not len(values) else float(values.min()),
        median=None if not len(values) else float(np.median(values)),
        maximum=None if not len(values) else float(values.max()))


def score_run(context, args):
    assert digest(context['predictions_path']) == context['predictions_sha256']
    predictions = np.load(context['predictions_path'], mmap_mode='r')[:, :, :, 0]
    with np.load(context['trace'], allow_pickle=False) as saved:
        truth = saved['target_com_m_s'][..., 0]
    assert truth.shape == (3000, 96) and np.isfinite(truth).all()
    rows, all_metrics, all_lags, passed_all = [], [], [], True
    failure_counts, lag_counts = Counter(), Counter()
    required_long_parking = np.zeros(96, dtype=np.int64)
    env_coverage = []
    for case_index, case in enumerate(context['meta']['cases']):
        segments = merged_segments(case)
        for env in case['env_ids']:
            env_coverage.append(env)
            specs = phase_helpers.phase_specs(segments, context['times'], context['native'][:, env, 6]/2., context['valid'][:, env])
            for spec in specs:
                mask = ((context['times'] >= spec['start']-1e-9) & (context['times'] < spec['end']-1e-9))
                metrics, lags, coverage = phase_measurement(predictions[:, :, env], truth[:, env],
                    context['valid'][:, env], mask, spec['kind'])
                nonregression, failed = phase_gates(metrics, coverage['complete'])
                failure_counts.update(failed)
                passed = bool(nonregression.all())
                parking = None
                if spec['kind'] == 'parking':
                    # Every complete parking phase gets MAE/bias/P95 checks; band only with >=3s.
                    if coverage['complete']:
                        expected, checks = parking_gate(metrics[0], metrics[1:])
                        if coverage['expected_samples'] < 300:
                            checks.pop('error_band_1_3Hz_rms')
                        else:
                            required_long_parking[env] += 1
                        parking = dict(expected_combination_metrics=metric_dict(expected), checks=checks,
                            passed=all(checks.values()), band_eligible=coverage['expected_samples'] >= 300)
                        passed &= parking['passed']
                        failure_counts.update('parking_'+key for key,value in checks.items() if not value)
                    else:
                        passed = False
                if not coverage['complete']:
                    failure_counts['incomplete_phase'] += 1
                passed_all &= passed
                for method_index in range(82):
                    for reason_index, reason in enumerate(LAG_REASONS):
                        if int(lags[method_index, 4]) & (1 << reason_index):
                            lag_counts[reason] += 1
                rows.append(dict(row=len(rows), case=case_index, case_name=case.get('name', str(case_index)),
                    env=env, phase=spec, coverage=coverage, passed=passed,
                    zero_delay_metrics=metric_dict(metrics[0]),
                    expected_combination_metrics=metric_dict(np.mean(metrics[1:], axis=0)),
                    worst_combination_metrics=metric_dict(np.nanmax(metrics[1:], axis=0)) if coverage['complete'] else None,
                    nonregression_failed_tuples=(np.flatnonzero(~nonregression)+1).tolist(),
                    nonregression_failures=failed, parking_gate=parking))
                all_metrics.append(metrics)
                all_lags.append(lags)
        print(context['role'], 'case', case_index, 'scored;', len(rows), 'phases', flush=True)
    assert sorted(env_coverage) == list(range(96))
    missing_long = np.flatnonzero(required_long_parking == 0).tolist()
    failed_episodes = np.flatnonzero(np.asarray(context['meta']['episode_done'], dtype=bool)).tolist()
    passed_all &= not missing_long and not failed_episodes
    if missing_long:
        failure_counts['no_complete_at_least_3s_parking_environment'] = len(missing_long)
    if failed_episodes:
        failure_counts['baseline_episode_failure'] = len(failed_episodes)
    metrics, lags = np.asarray(all_metrics), np.asarray(all_lags)
    score_path = args.output/f"{context['role']}-phase-metrics.npz"
    np.savez_compressed(score_path, metrics=metrics, lags=lags,
        metric_names=np.asarray(METRICS), lag_columns=np.asarray(['lag_ms', 'raw_best_lag_ms', 'peak_correlation', 'near_optimal_width_ms', 'unknown_reason_bitmask']),
        lag_reason_bits=np.asarray(LAG_REASONS), combinations=np.asarray([[0, 0, 0, 0], *COMBINATIONS]))
    summaries = {}
    for kind in ('initial', 'parking', 'cruise', 'start', 'brake'):
        indices = [index for index,row in enumerate(rows) if row['phase']['kind'] == kind]
        if not indices:
            continue
        data = metrics[indices]
        summaries[kind] = dict(phases=len(indices), complete=sum(rows[i]['coverage']['complete'] for i in indices),
            passed=sum(rows[i]['passed'] for i in indices),
            zero={name:describe(data[:, 0, j]) for j,name in enumerate(METRICS)},
            equal_weight_tuple_metric_expectation={name:describe(data[:, 1:, j].mean(axis=1)) for j,name in enumerate(METRICS)},
            all_delayed_combinations={name:describe(data[:, 1:, j]) for j,name in enumerate(METRICS)},
            zero_lag_ms=describe(lags[indices, 0, 0]), delayed_lag_ms=describe(lags[indices, 1:, 0]))
    return dict(role=context['role'], support=bool(passed_all), survived=context['meta']['survived'],
        missing_long_parking_envs=missing_long, failed_episode_envs=failed_episodes,
        inputs=context['inputs'], reconstruction=context['reconstruction'],
        predictions=dict(path=str(context['predictions_path']), sha256=context['predictions_sha256']),
        complete_tuple_metrics=dict(path=str(score_path), sha256=digest(score_path),
            axes='row from phases, method (0=zero,1..81=combination_order), metric/lag column; NaN means unavailable'),
        phase_summaries=summaries, failure_counts=dict(failure_counts), lag_unknown_reasons=dict(lag_counts), phases=rows)


def main(args):
    from mjlab.utils.buffers import CircularBuffer
    from rsl_rl.modules import MLP
    torch.set_num_threads(1)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    card = json.loads(CARD.read_text())
    assert card['parent_model_sha256'] == EXPECTED_MODEL_SHA
    assert card['combination_order'] == [list(row) for row in COMBINATIONS]
    assert card['judgment_not_user_task_acceptance']['all_combinations_nonregression'] == GATE_SPEC['nonregression']
    assert phase_helpers.GATE_SPEC['initial']['end_s'] == 1.5
    assert phase_helpers.GATE_SPEC['phase']['settle_after_ramp_s'] == 1.5
    contexts = [read_context(args.study, 'study'), read_context(args.validation, 'validation')]
    assert contexts[0]['meta']['seed'] != contexts[1]['meta']['seed']
    assert contexts[0]['inputs'][str(contexts[0]['runtime'])] == contexts[1]['inputs'][str(contexts[1]['runtime'])]
    checkpoint = contexts[0]['checkpoint']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['iter'] == 2999
    state = {key.removeprefix('encoder.'):value for key,value in saved['actor_state_dict'].items() if key.startswith('encoder.')}
    encoder = MLP(125, 3, (128, 64), 'elu').to(args.device)
    encoder.load_state_dict(state, strict=True)
    del saved
    encoder.eval().requires_grad_(False)
    before = {key:value.detach().cpu().clone() for key,value in encoder.state_dict().items()}
    encoder_sha = state_digest(before)
    sources = {str(Path(path).resolve()):digest(path) for path in
        (__file__, phase_helpers.__file__, inspect.getfile(MLP), inspect.getfile(CircularBuffer), CARD, contexts[0]['runtime'])}
    lock = dict(status='locked_before_any_target_com_scoring_or_shadow_inference', gate=GATE_SPEC,
        experiment_card=str(CARD), source_sha256=sources, checkpoint=str(checkpoint),
        checkpoint_sha256=EXPECTED_MODEL_SHA, encoder_state_sha256=encoder_sha,
        encoder_eval=True, requires_grad=False, target_com_opened=False,
        labels_never_shifted=True, prediction_method_order=[[0, 0, 0, 0], *COMBINATIONS],
        capture_inputs={context['role']:context['inputs'] for context in contexts})
    write(args.output/'gate-lock.json', lock)
    shutil.copy2(__file__, args.output/'source.py')
    native_delay = extract_delay_class(contexts[0]['runtime'])
    synthetic = synthetic_check(native_delay, CircularBuffer)
    write(args.output/'synthetic-reconstruction.json', synthetic)
    for context in contexts:
        predict_run(context, encoder, args)
    assert all(torch.equal(before[key], value.cpu()) for key,value in encoder.state_dict().items())
    assert state_digest(encoder.state_dict()) == encoder_sha
    write(args.output/'all-predictions-sealed.json', dict(target_com_opened=False,
        predictions={c['role']:c['predictions_sha256'] for c in contexts},
        encoder_state_sha256_before=encoder_sha, encoder_state_sha256_after=state_digest(encoder.state_dict())))
    runs = [score_run(context, args) for context in contexts]
    assert all(digest(path) == sha for path,sha in sources.items())
    assert all(digest(path) == sha for context in contexts for path,sha in context['inputs'].items())
    assert digest(checkpoint) == EXPECTED_MODEL_SHA
    assert state_digest(encoder.state_dict()) == encoder_sha
    result = dict(support=all(run['support'] for run in runs), gate=GATE_SPEC,
        gate_lock_sha256=digest(args.output/'gate-lock.json'), sources_unchanged=True,
        encoder_unchanged=True, encoder_state_sha256=encoder_sha, parameter_updates=0,
        shadow_control=False, label_alignment='Original current-policy COM label; no label shift or score alignment',
        target_timing_metadata={context['role']:context['meta']['target_com_scope'] for context in contexts},
        synthetic_reconstruction=synthetic, runs=runs,
        conclusion_scope=GATE_SPEC['scope'], height='Backward low height remains unresolved; unchanged.')
    write(args.output/'results.json', result)
    print(json.dumps(dict(support=result['support'], parameter_updates=0, shadow_control=False,
        reconstruction_passed=True, runs={run['role']:dict(support=run['support'], failures=run['failure_counts']) for run in runs}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    main(parser.parse_args())
