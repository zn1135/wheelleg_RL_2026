"""Independent acceptance v2: score saved 500 Hz records without running physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ct_mjlab.fudan_ppo_walk.protocol import manifest, samples
from .score import DT, score as score_v1

PROTOCOL = 'fudan-walk-stage1-v2'
MEAN_TICKS = 10
LOAD_BAD_TICKS = 50
LOW_SUPPORT_TICKS = 10
STARTUP_TICKS = 1000
STARTUP_CONTACT_RUN_TICKS = 10
STARTUP_CONTACT_TOTAL_TICKS = 25
SHAPES = dict(time=(), position=(3,), linear_body=(3,), linear_world=(3,),
    angular_body=(3,), roll_pitch_deg=(2,), wheel_support=(2,), base_contact=(),
    weight=(), terminated=(), numerical_failure=())


def acceptance_manifest():
    result = dict(protocol=PROTOCOL, input_matrix_protocol=manifest()['protocol'],
        input_matrix_sha256=manifest()['sha256'], sample_period_s=DT,
        interval='integer ticks, left closed and right open; terminal state excluded from force durations',
        mean_ticks=MEAN_TICKS, wheel_min_weight_fraction=.1, wheel_min_N=5.,
        total_weight_fraction=[.8, 1.2], joint_load_valid_fraction=.95,
        load_bad_run_fail_ticks=LOAD_BAD_TICKS, simultaneous_wheel_below_N=1.,
        simultaneous_low_run_fail_ticks=LOW_SUPPORT_TICKS,
        startup_ticks=STARTUP_TICKS, startup_contact_above_N=10.,
        startup_contact_run_max_ticks=STARTUP_CONTACT_RUN_TICKS,
        startup_contact_total_max_ticks=STARTUP_CONTACT_TOTAL_TICKS,
        post_startup_body_mean_max_N=1.,
        measurement_windows='tracking segment[start+2s,end); stop observation[settle_completed,settle_completed+5s)',
        body_measurement='all complete20ms windows wholly within[2s,episode_end), including command transitions',
        bad_window_run_duration='consecutive window-start tick count times2ms, not the span of overlapping raw windows',
        unchanged_tracking_limits={k:v for k,v in manifest()['limits'].items() if k != 'fall_base_contact_N'},
        final_second_tilt_max_deg=35., final_second_angular_xy_max_rad_s=.5,
        numerical_failure_or_termination_allowed=False,
        final_acceptance='requires_explicit_user_video_review')
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def tick(seconds):
    value = round(seconds / DT)
    if abs(seconds - value * DT) > 1e-8:
        raise ValueError(f'Time boundary is not a 2ms tick: {seconds}')
    return value


def runs(mask, start_tick=0):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    begins = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [dict(start_tick=int(lo + start_tick), end_tick=int(hi + start_tick),
        ticks=int(hi - lo), duration_ms=int(hi - lo) * 2) for lo, hi in zip(begins, ends)]


def run_summary(mask, start_tick=0):
    intervals = runs(mask, start_tick)
    return dict(total_ticks=int(np.count_nonzero(mask)),
        maximum_run_ticks=max((r['ticks'] for r in intervals), default=0), intervals=intervals)


def moving_mean(values):
    # 仅保留完整的10点窗口，不在段边缘补零或混入响应段。
    return np.lib.stride_tricks.sliding_window_view(values, MEAN_TICKS, axis=0).mean(axis=-1, dtype=np.float64)


def load_window(trace, lo, hi, name):
    support = trace['wheel_support'][lo:hi]
    weight = trace['weight'][lo:hi]
    result = dict(name=name, start_tick=lo, end_tick=hi, passed=False)
    if hi - lo < MEAN_TICKS:
        result['missing'] = 'No complete20ms window'
        return result
    averaged = moving_mean(support)
    average_weight = moving_mean(weight)
    conditions = dict(each_wheel=(averaged >= np.maximum(.1 * average_weight, 5.)[:, None]).all(axis=1),
        total_min=averaged.sum(axis=1) >= .8 * average_weight,
        total_max=averaged.sum(axis=1) <= 1.2 * average_weight)
    valid = np.logical_and.reduce(list(conditions.values()))
    bad = run_summary(~valid, lo)
    # 原始两轮同时失载仍单独否决，不能被平均值掩盖。
    low = run_summary((support < 1.).all(axis=1), lo)
    raw_valid = ((support >= np.maximum(.1 * weight, 5.)[:, None]).all(axis=1)
        & (support.sum(axis=1) >= .8 * weight) & (support.sum(axis=1) <= 1.2 * weight))
    valid_count = int(valid.sum())
    fraction_ok = 20 * valid_count >= 19 * len(valid)
    continuity_ok = bad['maximum_run_ticks'] < LOAD_BAD_TICKS
    low_ok = low['maximum_run_ticks'] < LOW_SUPPORT_TICKS
    ratio = support.sum(axis=1) / weight
    result.update(valid_windows=len(valid), joint_valid_windows=valid_count,
        joint_valid_fraction=valid_count / len(valid), fraction_passed=fraction_ok,
        bad_window_runs=bad, bad_run_passed=continuity_ok,
        simultaneous_low_raw=low, simultaneous_low_passed=low_ok,
        per_condition_valid_fraction={k:float(v.mean()) for k,v in conditions.items()},
        mean20_total_weight_ratio_min=float((averaged.sum(axis=1) / average_weight).min()),
        mean20_total_weight_ratio_max=float((averaged.sum(axis=1) / average_weight).max()),
        raw_diagnostics=dict(wheel_min_N=support.min(axis=0).tolist(), wheel_max_N=support.max(axis=0).tolist(),
            total_weight_ratio_min=float(ratio.min()), total_weight_ratio_max=float(ratio.max()),
            load_bad_runs=run_summary(~raw_valid, lo)),
        passed=bool(fraction_ok and continuity_ok and low_ok))
    return result


def score(sample, trace):
    expected = tick(sample['case']['duration']) + 1
    result = dict(protocol=PROTOCOL, sample_id=sample['id'], passed=False, complete=False,
        numerical_failure=False, failure_reasons=[], final_acceptance='pending_user_video_review')
    missing = [k for k in SHAPES if k not in trace]
    if missing:
        result.update(missing=missing, failure_reasons=['missing_fields'])
        return result
    arrays = {k:np.asarray(trace[k]) for k in SHAPES}
    for flag in ('numerical_failure','terminated'):
        if arrays[flag].dtype.kind in 'bifu':result[flag]=bool(arrays[flag].any())
    finite = all(np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_) for v in arrays.values())
    finite = finite and all(np.isfinite(v).all() for v in arrays.values())
    result['finite'] = bool(finite)
    if not finite:
        result.update(numerical_failure=True, failure_reasons=['nonfinite_or_nonnumeric'])
        return result
    invalid_shapes = {k:list(value.shape) for k,value in arrays.items() if value.shape != (expected, *SHAPES[k])}
    if invalid_shapes:
        result.update(invalid_shapes=invalid_shapes, expected_samples=expected,
            failure_reasons=['incomplete_or_invalid_shape']+(['numerical_failure'] if result['numerical_failure'] else []))
        return result
    canonical_time = np.arange(expected) * DT
    if not np.allclose(arrays['time'], canonical_time, rtol=0., atol=1e-7):
        result['failure_reasons'] = ['invalid_time_grid']
        return result
    if (arrays['weight'] <= 0.).any() or any(not np.isin(arrays[k], [0,1]).all() for k in ('terminated','numerical_failure')):
        result['failure_reasons'] = ['invalid_weight_or_flags']
        return result
    result.update(complete=True, observed_samples=expected, expected_samples=expected)
    legacy = score_v1(sample, arrays)
    # 沿用v1跟踪/停车公式，以规范tick时间选择同一测量段；不继承v1的联合否决。
    tracking = legacy if np.array_equal(arrays['time'], canonical_time) else score_v1(sample, dict(arrays, time=canonical_time))
    result.update(segments=tracking['segments'], v1_diagnostic=legacy,
        numerical_failure=legacy['numerical_failure'], terminated=legacy['terminated'])
    if 'stop' in tracking:
        result['stop'] = tracking['stop']
    reasons = result['failure_reasons']
    if result['numerical_failure']:reasons.append('numerical_failure')
    if result['terminated']:reasons.append('terminated')
    angles = np.abs(arrays['roll_pitch_deg'])
    if (angles > 60.).any():reasons.append('tilt_over60deg')
    for i,row in enumerate(tracking['segments']):
        if not row['passed']:reasons.append(f'tracking_segment_{i}')
    if 'stop' in tracking and not tracking['stop']['passed']:reasons.append('stop')

    end = expected - 1
    startup_end = min(STARTUP_TICKS, end)
    body = arrays['base_contact']
    startup = run_summary(body[:startup_end] > 10.)
    startup['passed'] = (startup['maximum_run_ticks'] <= STARTUP_CONTACT_RUN_TICKS
        and startup['total_ticks'] <= STARTUP_CONTACT_TOTAL_TICKS)
    if startup['maximum_run_ticks'] > STARTUP_CONTACT_RUN_TICKS:reasons.append('startup_contact_run')
    if startup['total_ticks'] > STARTUP_CONTACT_TOTAL_TICKS:reasons.append('startup_contact_total')
    if end - STARTUP_TICKS < MEAN_TICKS:
        post = dict(passed=False, missing='No complete post-startup20ms window')
    else:
        averaged_body = moving_mean(body[STARTUP_TICKS:end])
        post = dict(start_tick=STARTUP_TICKS, end_tick=end, valid_windows=len(averaged_body),
            maximum_mean_N=float(averaged_body.max()), passed=bool((averaged_body <= 1.).all()),
            bad_window_runs=run_summary(averaged_body > 1., STARTUP_TICKS))
    if not post['passed']:reasons.append('post_startup_body_mean')
    peak = int(body.argmax())
    result['body_contact'] = dict(startup_over10N=startup, post_startup_mean20=post,
        raw_peak_N=float(body[peak]), raw_peak_tick=peak,
        raw_over10N=run_summary(body[:end] > 10.))

    windows = [dict(name=f'tracking_segment_{i}', lo=tick(row['window'][0]), hi=tick(row['window'][1]))
        for i,row in enumerate(tracking['segments'])]
    if 'stop' in tracking and 'observation_window' in tracking['stop']:
        lo,hi = tracking['stop']['observation_window']
        windows.append(dict(name='stop_observation', lo=tick(lo), hi=tick(hi)))
    result['load_windows'] = [load_window(arrays, w['lo'], w['hi'], w['name']) for w in windows]
    for row in result['load_windows']:
        if not row['passed']:reasons.append('load_' + row['name'])
    last = slice(end - tick(1.), end)
    tilt = float(angles[last].max())
    angular = float(np.linalg.norm(arrays['angular_body'][last,:2], axis=1).max())
    result['final_second_attitude'] = dict(start_tick=end-tick(1.), end_tick=end,
        tilt_max_deg=tilt, angular_xy_max_rad_s=angular, passed=tilt <= 35. and angular <= .5)
    if not result['final_second_attitude']['passed']:reasons.append('final_second_attitude')
    result['passed'] = not reasons
    return result


def summarize_recording(directory):
    protocol_path = directory / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    mode = protocol['mode']
    items = protocol['samples']
    if mode not in ('diagnostic_not_formal_acceptance', 'formal_987_samples'):
        raise ValueError(f'Unsupported recording mode: {mode}')
    if mode == 'formal_987_samples' and items != samples():
        raise ValueError('Formal matrix must retain all987 frozen samples')
    if not items or len({s['id'] for s in items}) != len(items):
        raise ValueError('Recording requires nonempty unique sample IDs')
    rows = []
    for sample in items:
        path = directory / 'traces' / (sample['id'] + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as trace:row = score(sample, trace)
            row['trajectory_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            row = dict(protocol=PROTOCOL, sample_id=sample['id'], passed=False, complete=False,
                missing=['trajectory'], numerical_failure=False, failure_reasons=['missing_trajectory'])
        row.update(case=sample['case'], randomized=sample['randomized'])
        rows.append(row)
    groups = {}
    for row in rows:
        key = row['case']['category'] + (':random' if row['randomized'] else ':nominal')
        group = groups.setdefault(key, dict(total=0, complete=0, technical_passed=0))
        for field,value in [('total',1),('complete',row['complete']),('technical_passed',row['passed'])]:
            group[field] += int(value)
    for name,group in groups.items():
        required = .95 if name.endswith(':random') else 1.
        group.update(required_fraction=required,
            thresholds_met=group['complete'] == group['total'] and group['technical_passed'] / group['total'] >= required)
    return dict(protocol=PROTOCOL, acceptance_manifest=acceptance_manifest(), mode=mode,
        source_recording=str(directory), source_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        total=len(rows), complete=sum(r['complete'] for r in rows),
        technical_passed=sum(r['passed'] for r in rows), numerical_failures=sum(r['numerical_failure'] for r in rows),
        groups=groups, samples=rows, physics_steps=0, model_forwards=0, optimizer_updates=0,
        formal_matrix_thresholds_met=(all(g['thresholds_met'] for g in groups.values()) if mode == 'formal_987_samples' else None),
        final_acceptance='pending_user_video_review')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize_recording(args.recording)
    with args.output.open('x') as output:output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','acceptance_manifest','groups')}, indent=2))


if __name__ == '__main__':
    main()
