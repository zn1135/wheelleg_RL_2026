#!/usr/bin/env python3
"""Compare saved same-checkpoint walk feedback interventions on CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


FIXED_KEYS = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
PROTOCOL_KEYS = ('actual_profile', 'common_environment_profile', 'load', 'seed',
                 'seconds', 'warmup', 'height', 'dt', 'policy_dt', 'weight_N',
                 'layout', 'joint_names', 'cases', 'timing')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes() == right.tobytes())


def array_receipt(value, saved_name):
    return dict(saved_name=saved_name, shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def distribution(values):
    observed = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(observed_environments=len(observed), missing_environments=len(values) - len(observed),
        median=None if not len(observed) else float(np.median(observed)),
        p95=None if not len(observed) else float(np.percentile(observed, 95)))


def summarize(meta, values):
    assert values.ndim == 3 and np.isfinite(values).all(), 'Invalid trace'
    assert len(values) == round(meta['seconds'] / meta['dt'])
    times = (np.arange(len(values)) + 1) * meta['dt']
    ends = np.asarray(meta['episode_end'])
    assert ends.shape == (values.shape[1],) and np.isfinite(ends).all()
    assert ((ends > 0.) & (ends <= meta['seconds'])).all()

    def get(name):
        return values[..., slice(*meta['layout'][name])]

    valid_values = get('valid')[..., 0]
    assert ((valid_values == 0.) | (valid_values == 1.)).all()
    valid = valid_values > .5
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    assert np.array_equal(valid.sum(0), np.rint(ends / meta['dt']).astype(int))
    vx, height = get('true_velocity')[..., 0], get('height')[..., 0]
    joint_ids = [meta['joint_names'].index(n) for n in ('lf0', 'lf00', 'rf0', 'rf00')]
    motor = get('joint_q')[..., joint_ids]
    survive = ends >= meta['seconds'] - 1e-6
    final = times >= meta['seconds'] - 2. - 1e-9
    late = (times >= 15. - 1e-9) & (times <= 20. + 1e-9)
    assert meta['seconds'] >= 20. and final.any() and late.any()
    rows, all_ids = [], []
    for case in meta['cases']:
        ids = np.asarray(case['env_ids'])
        all_ids.extend(ids.tolist())
        settled_at = meta['warmup'] + abs(case['speed']) / case['acceleration'] + .5
        assert times[final][0] >= settled_at, 'Final 2s includes command ramp/settling'
        environments = []
        for env in ids:
            mask = final & valid[:, env]
            late_mask = late & valid[:, env]
            complete = bool(mask.sum() == final.sum())
            late_complete = bool(late_mask.sum() == late.sum())
            speed_mae = None if not complete else float(abs(vx[mask, env] - case['speed']).mean())
            height_mean = None if not complete else float(height[mask, env].mean())
            height_mae = None if not complete else float(abs(height[mask, env] - meta['height']).mean())
            # 完整15～20秒才计入摆幅对照，失败及缺样保留环境行并记null。
            peak = None if not late_complete else np.ptp(
                np.rad2deg(np.unwrap(motor[late_mask, env], axis=0)), axis=0).tolist()
            environments.append(dict(env=int(env), survived=bool(survive[env]),
                episode_end_s=float(ends[env]), final2s_live_samples=int(mask.sum()),
                final2s_complete=complete, final2s_vx_mae_m_s=speed_mae,
                final2s_height_mean_m=height_mean, final2s_height_mae_m=height_mae,
                survived_and_final2s_vx_mae_le_0_1=bool(survive[env] and complete and speed_mae <= .1),
                late15_20s_live_samples=int(late_mask.sum()), late15_20s_complete=late_complete,
                late15_20s_motor_peak_to_peak_deg=peak,
                late15_20s_motor_max_peak_to_peak_deg=None if peak is None else max(peak)))
        row = dict(case=case['name'], speed=case['speed'], acceleration=case['acceleration'],
            environments=len(ids), survived=int(survive[ids].sum()),
            survived_and_final2s_vx_mae_le_0_1=sum(
                e['survived_and_final2s_vx_mae_le_0_1'] for e in environments),
            final2s_complete_environments=sum(e['final2s_complete'] for e in environments),
            late15_20s_complete_environments=sum(e['late15_20s_complete'] for e in environments),
            per_environment=environments)
        for name in ('final2s_vx_mae_m_s', 'final2s_height_mean_m', 'final2s_height_mae_m',
                     'late15_20s_motor_max_peak_to_peak_deg'):
            row[name] = distribution([e[name] for e in environments])
        rows.append(row)
    assert sorted(all_ids) == list(range(values.shape[1]))
    return rows


def load_measurement(label, prefix):
    prefix = Path(prefix).resolve()
    meta_path, trace_path = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    meta = json.loads(meta_path.read_text())
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] is True
    assert meta.get('action_source', 'original') == 'original', 'Only velocity feedback may change'
    sha = meta['checkpoint_sha256']
    assert len(sha) == 64
    int(sha, 16)
    checkpoint = Path(meta['checkpoint']).resolve()
    checkpoint_verified = checkpoint.exists()
    if checkpoint_verified:
        assert digest(checkpoint) == sha, 'Actual checkpoint SHA differs from metadata'
    protocol = {key: meta[key] for key in PROTOCOL_KEYS}
    protocol['action_source'] = meta.get('action_source', 'original')
    with np.load(trace_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files if key != 'values'}
        assert all(np.isfinite(a).all() for a in arrays.values())
        state, receipts, missing = {}, {}, []
        for key in FIXED_KEYS:
            # 兼容历史产物的initial_*命名；缺项不由运行时声明补全。
            names = [n for n in ('fixed_' + key, 'initial_' + key) if n in arrays]
            if not names:
                missing.append(key)
                continue
            assert all(equal(arrays[names[0]], arrays[n]) for n in names)
            state[key] = arrays[names[0]]
            receipts[key] = array_receipt(state[key], names[0])
        for key in ('qpos', 'qvel'):
            if key in state and 'initial_' + key in arrays:
                assert equal(state[key], arrays['initial_' + key])
        rows = summarize(meta, saved['values'])
    result_path = prefix.parent / 'result.json'
    receipt = json.loads(result_path.read_text()) if result_path.exists() else {}
    run = dict(label=label, measurement=str(prefix), checkpoint=str(checkpoint),
        checkpoint_sha256=sha, actual_checkpoint_sha_verified=checkpoint_verified,
        velocity_source=meta['velocity_source'], parameter_updates=0, actor_unchanged=True,
        actor_invariance_evidence='Recorder metadata asserts tensor equality and zero updates; CPU comparison does not rerun actor.',
        saved_fixed_arrays=receipts, missing_fixed_arrays=missing,
        saved_initial_arrays={k: array_receipt(arrays['initial_' + k], 'initial_' + k)
                              for k in ('qpos', 'qvel') if 'initial_' + k in arrays},
        missing_initial_arrays=[k for k in ('qpos', 'qvel') if 'initial_' + k not in arrays],
        original_runtime_physics_equality_claim=receipt.get('initial_and_physical_states_equal'),
        input_files={str(p): digest(p) for p in (meta_path, trace_path)},
        missing_formal_balance_signals=[key for key in ('roll', 'pitch', 'angular_velocity_xy')
                                        if key not in meta['layout']], cases=rows)
    return run, protocol, state


def main(args):
    labels = [label for label, _ in args.measurement]
    assert len(labels) >= 2 and len(set(labels)) == len(labels)
    assert args.baseline in labels and not args.output.exists()
    runs, states, reference_protocol = [], [], None
    for label, prefix in args.measurement:
        run, protocol, state = load_measurement(label, prefix)
        if reference_protocol is None:
            reference_protocol = protocol
        else:
            assert protocol == reference_protocol, f'{label}: protocol differs beyond velocity_source'
            assert run['checkpoint_sha256'] == runs[0]['checkpoint_sha256'], 'Use one checkpoint per comparison'
            for key in FIXED_KEYS:
                for old in states:
                    if key in state and key in old:
                        assert equal(state[key], old[key]), f'{label}: saved {key} differs'
        runs.append(run)
        states.append(state)
    baseline = next(run for run in runs if run['label'] == args.baseline)
    for run in runs:
        lost_alive, lost_tracking = [], []
        for row, base in zip(run['cases'], baseline['cases'], strict=True):
            def passed(field, case):
                return {e['env'] for e in case['per_environment'] if e[field]}
            alive, old_alive = passed('survived', row), passed('survived', base)
            tracking = passed('survived_and_final2s_vx_mae_le_0_1', row)
            old_tracking = passed('survived_and_final2s_vx_mae_le_0_1', base)
            lost, lost_speed = sorted(old_alive - alive), sorted(old_tracking - tracking)
            row.update(lost_baseline_survivor_envs=lost,
                gained_survivor_envs=sorted(alive - old_alive),
                lost_baseline_tracking_envs=lost_speed,
                gained_tracking_envs=sorted(tracking - old_tracking))
            paired = [(e, b) for e, b in zip(row['per_environment'], base['per_environment'], strict=True)
                      if e['late15_20s_complete'] and b['late15_20s_complete']]
            row['paired_late15_20s_motor_max_peak_to_peak_delta_deg'] = distribution([
                e['late15_20s_motor_max_peak_to_peak_deg'] - b['late15_20s_motor_max_peak_to_peak_deg']
                for e, b in paired] + [None] * (row['environments'] - len(paired)))
            lost_alive.extend(lost)
            lost_tracking.extend(lost_speed)
        run.update(environments=sum(r['environments'] for r in run['cases']),
            survived=sum(r['survived'] for r in run['cases']),
            survived_and_final2s_vx_mae_le_0_1=sum(r['survived_and_final2s_vx_mae_le_0_1'] for r in run['cases']),
            lost_baseline_survivor_envs=lost_alive, lost_baseline_tracking_envs=lost_tracking,
            diagnostic_regression=bool(lost_alive or lost_tracking))
    all_fixed = not any(run['missing_fixed_arrays'] for run in runs)
    report = dict(verified=True, baseline=args.baseline,
        script=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)),
        invocation=[sys.executable] + sys.argv, parameter_updates=0,
        same_checkpoint_metadata_sha_verified=True,
        actual_checkpoint_sha_verified=all(run['actual_checkpoint_sha_verified'] for run in runs),
        protocol_except_velocity_source_equal=True,
        protocol=reference_protocol,
        initial_qpos_qvel_bitwise_equal=None if any(run['missing_initial_arrays'] for run in runs) else True,
        all_saved_common_fixed_arrays_bitwise_equal=True,
        all_eight_required_fixed_arrays_bitwise_equal=True if all_fixed else None,
        physical_array_limit=None if all_fixed else 'Missing saved physical arrays are unknown; runtime claims do not replace array equality.',
        motor_order=['lf0', 'lf00', 'rf0', 'rf00'],
        metric_rules=dict(denominator='All initial environments retained, including failures.',
            final2s='Metrics require all valid first-episode samples in final planned 2s; incomplete windows are null. Speed pass also requires full-duration survival and vx MAE<=0.1m/s.',
            motor='Each motor unwrapped peak-to-peak over complete 15–20s; maximum across four motors per environment. Incomplete window is null.',
            aggregate='Median/P95 across per-environment complete-window metrics; observed and missing counts are explicit. Missing is never zero.'),
        nondeployable_intervention=True,
        scope='Saved nominal fixed-request feedback diagnosis; oracle feedback is a simulator-only intervention, not a deployable model improvement.',
        formal_balance_acceptance_performed=False, full_task_acceptance_performed=False,
        runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, output=str(args.output.resolve()),
        physical_arrays_fully_verified=all_fixed,
        runs=[dict(label=r['label'], survived=r['survived'],
            tracking=r['survived_and_final2s_vx_mae_le_0_1'],
            lost_survivors=len(r['lost_baseline_survivor_envs']),
            lost_tracking=len(r['lost_baseline_tracking_envs'])) for r in runs]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', nargs=2, action='append', required=True,
                        metavar=('LABEL', 'PATH_PREFIX'))
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
