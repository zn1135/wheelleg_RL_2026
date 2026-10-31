#!/usr/bin/env python3
"""CPU-only comparison of saved walk height-command interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


FIXED_KEYS = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
PROTOCOL_KEYS = ('actual_profile', 'common_environment_profile', 'load', 'seed',
                 'seconds', 'warmup', 'dt', 'policy_dt', 'weight_N', 'layout',
                 'joint_names', 'timing', 'velocity_source', 'action_source')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes() == right.tobytes())


def distribution(values):
    observed = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(observed_environments=len(observed), missing_environments=len(values) - len(observed),
        mean=None if not len(observed) else float(observed.mean()),
        median=None if not len(observed) else float(np.median(observed)),
        p95=None if not len(observed) else float(np.percentile(observed, 95)))


def array_receipt(value, name):
    return dict(saved_name=name, shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def load_measurement(height_target, folder, label):
    prefix = folder / label
    meta_path, trace_path = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    meta = json.loads(meta_path.read_text())
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] is True
    assert abs(meta['height'] - height_target) <= 1e-9
    assert meta['seed'] == 43 and meta['seconds'] == 20. and meta['warmup'] == 2.
    assert meta['velocity_source'] == 'encoder' and meta['action_source'] == 'original'
    assert len(meta['cases']) == 1
    case = meta['cases'][0]
    assert case['name'] == label and case['speed'] == (0. if label == 'zero' else -1.)
    assert case['acceleration'] == 2. and case['env_ids'] == list(range(16))
    protocol = {key: meta[key] for key in PROTOCOL_KEYS}
    with np.load(trace_path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files if k != 'values'}
        assert all(np.isfinite(a).all() for a in arrays.values())
        state, state_receipts, missing = {}, {}, []
        for key in FIXED_KEYS:
            names = [k for k in ('fixed_' + key, 'initial_' + key) if k in arrays]
            if not names:
                missing.append(key)
                continue
            assert all(equal(arrays[names[0]], arrays[n]) for n in names)
            state[key] = arrays[names[0]]
            state_receipts[key] = array_receipt(state[key], names[0])
        values = saved['values']
    assert values.ndim == 3 and values.shape[:2] == (10000, 16) and np.isfinite(values).all()
    assert len(values) == round(meta['seconds'] / meta['dt'])

    def get(name):
        return values[..., slice(*meta['layout'][name])]

    times = (np.arange(len(values)) + 1) * meta['dt']
    valid_values = get('valid')[..., 0]
    assert ((valid_values == 0.) | (valid_values == 1.)).all()
    valid = valid_values > .5
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    ends = np.asarray(meta['episode_end'])
    assert ends.shape == (16,) and np.isfinite(ends).all()
    assert ((ends > 0.) & (ends <= meta['seconds'])).all()
    assert np.array_equal(valid.sum(0), np.rint(ends / meta['dt']).astype(int))
    command = get('applied_command')
    assert np.allclose(command[..., 2][valid], height_target, rtol=0., atol=1e-7)
    height, vx = get('height')[..., 0], get('true_velocity')[..., 0]
    lengths, targets, torque = get('leg_length'), get('leg_target'), get('actual_torque')
    assert lengths.shape[-1] == 2 and targets.shape[-1] == 4 and torque.shape[-1] == 6
    final = times >= meta['seconds'] - 2. - 1e-9
    assert times[final][0] >= meta['warmup'] + abs(case['speed']) / 2. + .5
    per_env = []
    for env, end in enumerate(ends):
        live = final & valid[:, env]
        complete = bool(live.sum() == final.sum())
        survived = bool(end >= meta['seconds'] - 1e-6)
        mae = None if not complete else float(abs(vx[live, env] - case['speed']).mean())
        per_env.append(dict(env=env, survived=survived, episode_end_s=float(end),
            final2s_live_samples=int(live.sum()), final2s_complete=complete,
            final2s_height_mean_m=None if not complete else float(height[live, env].mean()),
            final2s_height_mae_m=None if not complete else float(abs(height[live, env] - height_target).mean()),
            final2s_vx_mean_m_s=None if not complete else float(vx[live, env].mean()),
            final2s_vx_mae_m_s=mae,
            survived_and_final2s_vx_mae_le_0_1=bool(survived and complete and mae <= .1),
            final2s_leg_length_mean_m=None if not complete else lengths[live, env].mean(0).tolist(),
            final2s_leg_target_mean_rad=None if not complete else targets[live, env].mean(0).tolist(),
            final2s_actual_torque_abs_peak_Nm=None if not complete else abs(torque[live, env]).max(0).tolist(),
            valid_first_episode_actual_torque_abs_peak_Nm=abs(torque[valid[:, env], env]).max(0).tolist()))
    row = dict(label=label, height_target_m=height_target, speed_m_s=case['speed'],
        acceleration_m_s2=case['acceleration'], environments=16,
        survived=sum(e['survived'] for e in per_env),
        final2s_complete_environments=sum(e['final2s_complete'] for e in per_env),
        survived_and_final2s_vx_mae_le_0_1=sum(e['survived_and_final2s_vx_mae_le_0_1'] for e in per_env),
        per_environment=per_env)
    for key in ('final2s_height_mean_m', 'final2s_height_mae_m', 'final2s_vx_mean_m_s', 'final2s_vx_mae_m_s'):
        row[key] = distribution([e[key] for e in per_env])
    for key, channels in (('final2s_leg_length_mean_m', 2), ('final2s_leg_target_mean_rad', 4),
                          ('final2s_actual_torque_abs_peak_Nm', 6),
                          ('valid_first_episode_actual_torque_abs_peak_Nm', 6)):
        row[key] = [distribution([None if e[key] is None else e[key][i] for e in per_env])
                    for i in range(channels)]
    row.update(measurement=str(prefix), checkpoint=meta['checkpoint'],
        checkpoint_sha256=meta['checkpoint_sha256'], parameter_updates=0, actor_unchanged=True,
        actor_invariance_evidence='Recorder asserts tensor equality and zero updates; CPU analysis does not rerun actor.',
        saved_fixed_arrays=state_receipts, missing_fixed_arrays=missing,
        saved_initial_arrays={k: array_receipt(arrays['initial_' + k], 'initial_' + k)
                              for k in ('qpos', 'qvel') if 'initial_' + k in arrays},
        missing_initial_arrays=[k for k in ('qpos', 'qvel') if 'initial_' + k not in arrays],
        input_files={str(p): digest(p) for p in (meta_path, trace_path)},
        missing_formal_balance_signals=[k for k in ('roll', 'pitch', 'angular_velocity_xy')
                                        if k not in meta['layout']])
    return row, protocol, state, case


def response_fit(commands, actual):
    if any(v is None for v in actual):
        return dict(endpoint_slope_m_per_m=None, fitted_slope_m_per_m=None,
                    fitted_intercept_m=None, response_description='unknown_incomplete_window')
    x, y = np.asarray(commands), np.asarray(actual)
    endpoint = float((y[-1] - y[0]) / (x[-1] - x[0]))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    if np.ptp(y) <= 1e-6:
        description = 'No response resolved at 1e-6m reporting precision; not proof of globally ignored input.'
    elif 0. < endpoint < 1.:
        description = 'Positive endpoint response smaller than commanded height change; not complete input neglect.'
    elif endpoint >= 1.:
        description = 'Positive endpoint response at least as large as commanded height change.'
    else:
        description = 'Nonpositive endpoint response; inspect intermediate point and per-environment results.'
    return dict(endpoint_slope_m_per_m=endpoint, fitted_slope_m_per_m=float(slope),
        fitted_intercept_m=float(intercept), fitted_point_count=len(x),
        fitted_max_abs_residual_m=float(abs(residual).max()),
        actual_endpoint_change_m=float(y[-1] - y[0]),
        actual_height_nondecreasing=bool((np.diff(y) >= -1e-6).all()),
        response_description=description)


def main(args):
    measurements = sorted((float(height), Path(folder).resolve()) for height, folder in args.measurement)
    commands = [h for h, _ in measurements]
    assert len(commands) >= 3 and len(set(commands)) == len(commands)
    assert not args.output.exists(), 'Preserve historical output; choose a new JSON'
    runs, states, cases, protocol = [], [], {}, None
    checkpoint_sha, checkpoint_receipts = None, {}
    for height_target, folder in measurements:
        assert (folder / 'result.json').exists(), 'Measurement incomplete: missing final result.json'
        for label in ('zero', 'backward'):
            row, current, state, case = load_measurement(height_target, folder, label)
            if protocol is None:
                protocol = current
                checkpoint_sha = row['checkpoint_sha256']
                assert len(checkpoint_sha) == 64
                int(checkpoint_sha, 16)
            else:
                assert protocol == current, 'Protocol differs beyond height and fixed speed condition'
                assert row['checkpoint_sha256'] == checkpoint_sha, 'Different checkpoint'
            if label in cases:
                assert cases[label] == case, 'Case differs across requested heights'
            cases[label] = case
            for key in FIXED_KEYS:
                for previous in states:
                    if key in previous and key in state:
                        assert equal(previous[key], state[key]), f'{height_target}/{label}: {key} differs'
            checkpoint = Path(row['checkpoint']).resolve()
            if str(checkpoint) not in checkpoint_receipts:
                checkpoint_receipts[str(checkpoint)] = digest(checkpoint) if checkpoint.exists() else None
            assert checkpoint_receipts[str(checkpoint)] in (None, checkpoint_sha)
            runs.append(row)
            states.append(state)
    responses = []
    for label in ('zero', 'backward'):
        by_height = [next(r for r in runs if r['label'] == label and r['height_target_m'] == h) for h in commands]
        per_env = []
        for env in range(16):
            rows = [r['per_environment'][env] for r in by_height]
            actual = [r['final2s_height_mean_m'] if r['survived'] else None for r in rows]
            per_env.append(dict(env=env, all_heights_survived=all(r['survived'] for r in rows),
                all_final2s_complete=all(r['final2s_complete'] for r in rows),
                requested_heights_m=commands, actual_final2s_height_means_m=actual,
                **response_fit(commands, actual)))
        all_complete = all(r['survived'] == 16 and r['final2s_complete_environments'] == 16 for r in by_height)
        means = [r['final2s_height_mean_m']['mean'] for r in by_height]
        responses.append(dict(label=label, environments=16,
            complete_response_environments=sum(r['endpoint_slope_m_per_m'] is not None for r in per_env),
            per_environment=per_env, requested_heights_m=commands, actual_final2s_height_means_m=means,
            all_environment_mean_response=response_fit(commands, means if all_complete else [None] * len(commands)),
            endpoint_slope_m_per_m=distribution([r['endpoint_slope_m_per_m'] for r in per_env]),
            fitted_slope_m_per_m=distribution([r['fitted_slope_m_per_m'] for r in per_env])))
    all_fixed = not any(r['missing_fixed_arrays'] for r in runs)
    report = dict(verified=True, script=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__)),
        invocation=[sys.executable] + sys.argv, parameter_updates=0, checkpoint_sha256=checkpoint_sha,
        actual_checkpoint_files_sha256=checkpoint_receipts,
        actual_checkpoint_sha_verified=all(v is not None for v in checkpoint_receipts.values()),
        protocol_except_height_and_speed_condition_equal=True, protocol=protocol, cases=cases,
        all_saved_common_fixed_arrays_bitwise_equal=True,
        initial_qpos_qvel_bitwise_equal=None if any(r['missing_initial_arrays'] for r in runs) else True,
        all_eight_required_fixed_arrays_bitwise_equal=True if all_fixed else None,
        physical_array_limit=None if all_fixed else 'Missing saved arrays are unknown, not replaced by runtime claims.',
        leg_order=['left', 'right'], leg_target_order=['lf0', 'lf00', 'rf0', 'rf00'],
        actual_torque_order=['lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel'],
        metric_rules=dict(final2s='Complete valid first-episode samples in planned final 2s only; incomplete values null. All 16 initial environments remain in denominator.',
            velocity='vx is saved root-link true_velocity, matching task tracking; not encoder COM velocity.',
            slopes='Per-environment matched complete survivors; endpoint delta(actual)/delta(command), plus linear fit over every requested height. Unity means equal endpoint change, not low bias or task success.',
            torque='Absolute actual actuator torque maxima; final2s complete-window and all valid first-episode windows reported separately.'),
        formal_balance_acceptance_performed=False, full_task_acceptance_performed=False,
        scope='CPU-only nominal height-command diagnosis; descriptive response and tracking metrics do not establish reward causality, formal balance, randomized robustness or deployment success.',
        runs=runs, responses=responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, output=str(args.output.resolve()),
        physical_arrays_fully_verified=all_fixed,
        runs=[dict(label=r['label'], height=r['height_target_m'], survived=r['survived'],
                   height_mean=r['final2s_height_mean_m']['mean']) for r in runs],
        responses=[dict(label=r['label'], response=r['all_environment_mean_response']) for r in responses]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', nargs=2, action='append', required=True,
                        metavar=('HEIGHT_M', 'DIRECTORY'))
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
