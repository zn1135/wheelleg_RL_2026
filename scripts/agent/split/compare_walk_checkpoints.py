#!/usr/bin/env python3
"""Compare saved fixed-request walk traces on CPU; never simulate or learn."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIXED_KEYS = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
PROTOCOL_KEYS = ('common_environment_profile', 'seed', 'seconds', 'warmup', 'height',
                 'dt', 'policy_dt', 'weight_N', 'layout', 'joint_names', 'cases', 'timing')


def array_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes() == right.tobytes())


def array_digest(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def summarize(meta, values):
    def get(name):
        lo, hi = meta['layout'][name]
        return values[..., lo:hi]

    def describe(mask, height, tilt):
        samples = int(mask.sum())
        return dict(final2s_descriptive_live_samples=samples,
            final2s_height_mean_m=None if not samples else float(height[mask].mean()),
            final2s_height_mae_m=None if not samples else float(abs(height[mask] - meta['height']).mean()),
            final2s_tilt_mean_deg=None if not samples else float(tilt[mask].mean()),
            final2s_tilt_max_deg=None if not samples else float(tilt[mask].max()))

    assert values.ndim == 3 and np.isfinite(values).all()
    assert len(values) == round(meta['seconds'] / meta['dt'])
    ends = np.asarray(meta['episode_end'])
    assert ends.shape == (values.shape[1],) and np.isfinite(ends).all()
    assert ((ends > 0.) & (ends <= meta['seconds'])).all()
    valid_values = get('valid')[..., 0]
    assert ((valid_values == 0.) | (valid_values == 1.)).all()
    valid = valid_values > .5
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    times = (np.arange(len(values)) + 1) * meta['dt']
    # 提前终止后的复位数据不参与统计；失败环境仍保留在总分母。
    assert np.array_equal(valid.sum(0), np.rint(ends / meta['dt']).astype(int))
    survive = ends >= meta['seconds'] - 1e-6
    final_window = times >= meta['seconds'] - 2.
    expected_samples = int(final_window.sum())
    assert expected_samples > 0
    vx = get('true_velocity')[..., 0]
    height, tilt = get('height')[..., 0], get('tilt')[..., 0]
    rows, all_ids = [], []
    for case, existing in zip(meta['cases'], meta['summary']['cases'], strict=True):
        ids = np.asarray(case['env_ids'])
        assert existing['case'] == case['name']
        assert existing['speed'] == case['speed']
        assert existing['acceleration'] == case['acceleration']
        assert existing['environments'] == len(ids)
        assert existing['survived'] == int(survive[ids].sum())
        all_ids.extend(ids.tolist())
        full_at = meta['warmup'] + abs(case['speed']) / case['acceleration'] + .5
        full_mask = valid[:, ids] & (times[:, None] >= full_at)
        final_mask = full_mask & final_window[:, None] & survive[None, ids]
        assert existing['full_command_live_samples'] == int(full_mask.sum())
        actual_mean = None if not final_mask.any() else float(vx[:, ids][final_mask].mean())
        actual_mae = None if not final_mask.any() else float(
            abs(vx[:, ids] - case['speed'])[final_mask].mean())
        for key, actual in (('final2s_vx_survivors', actual_mean),
                            ('final2s_mae_survivors', actual_mae)):
            assert ((existing[key] is None and actual is None)
                    or (existing[key] is not None and actual is not None
                        and np.isclose(existing[key], actual, rtol=1e-6, atol=1e-7))), key
        environments = []
        for env_id in ids:
            mask = valid[:, env_id] & final_window & (times >= full_at)
            samples = int(mask.sum())
            mae = None if not samples else float(abs(vx[mask, env_id] - case['speed']).mean())
            complete = bool(survive[env_id] and samples == expected_samples)
            environments.append(dict(env=int(env_id), survived=bool(survive[env_id]),
                episode_end_s=float(ends[env_id]), final2s_live_samples=samples,
                final2s_complete=complete, final2s_mae_m_s=mae,
                survived_and_final2s_mae_le_0_1=bool(complete and mae <= .1),
                **describe(valid[:, env_id] & final_window, height[:, env_id], tilt[:, env_id])))
        rows.append(dict(case=case['name'], speed=case['speed'], acceleration=case['acceleration'],
            environments=len(ids), survived=int(survive[ids].sum()),
            survived_and_final2s_mae_le_0_1=sum(
                e['survived_and_final2s_mae_le_0_1'] for e in environments),
            **describe(valid[:, ids] & final_window[:, None], height[:, ids], tilt[:, ids]),
            original_summary=existing, per_environment=environments))
    assert sorted(all_ids) == list(range(values.shape[1]))
    return rows


def main(args):
    labels = [label for label, _ in args.measurement]
    assert len(set(labels)) == len(labels), 'Measurement labels must be unique'
    assert args.baseline in labels, 'Baseline label must be a measurement'
    assert not args.output.exists(), 'Use a new output JSON; preserve historical results'
    reference_meta, reference_arrays = None, {}
    runs, missing = [], {}
    for label, prefix_string in args.measurement:
        prefix = Path(prefix_string).resolve()
        meta = json.loads(prefix.with_suffix('.json').read_text())
        assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] is True
        assert len(meta['checkpoint_sha256']) == 64
        int(meta['checkpoint_sha256'], 16)
        protocol = {key: meta[key] for key in PROTOCOL_KEYS}
        protocol.update(velocity_source=meta['velocity_source'],
                        action_source=meta.get('action_source', 'original'))
        if reference_meta is None:
            reference_meta = protocol
        else:
            assert protocol == reference_meta, f'{label}: fixed-request protocol differs'
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files if key != 'values'}
            for key in ('initial_qpos', 'initial_qvel'):
                assert key in arrays, f'{label}: missing {key}'
            for key, value in arrays.items():
                assert np.isfinite(value).all(), f'{label}: nonfinite {key}'
            for fixed_key, initial_key in (('fixed_qpos', 'initial_qpos'),
                                            ('fixed_qvel', 'initial_qvel')):
                if fixed_key in arrays:
                    assert array_equal(arrays[fixed_key], arrays[initial_key])
            for key, value in arrays.items():
                if key in reference_arrays:
                    assert array_equal(reference_arrays[key], value), f'{label}: {key} differs'
                else:
                    reference_arrays[key] = value
            missing[label] = [f'fixed_{key}' for key in FIXED_KEYS if f'fixed_{key}' not in arrays]
            rows = summarize(meta, saved['values'])
        receipt_path = prefix.parent / 'result.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        runs.append(dict(label=label, measurement=str(prefix), checkpoint=meta['checkpoint'],
            checkpoint_sha256=meta['checkpoint_sha256'], actual_profile=meta['actual_profile'],
            parameter_updates=0, actor_unchanged=True,
            saved_state_arrays={key: array_digest(value) for key, value in arrays.items()},
            missing_fixed_arrays=missing[label],
            original_runtime_physics_equality_claim=receipt.get('initial_and_physical_states_equal'),
            missing_formal_balance_signals=[key for key in ('roll', 'pitch', 'angular_velocity_xy')
                if key not in meta['layout']], cases=rows))
    baseline = next(run for run in runs if run['label'] == args.baseline)
    for run in runs:
        total_lost, total_new, total_lost_tracking = [], [], []
        for row, base_row in zip(run['cases'], baseline['cases'], strict=True):
            def ids(key, case_row):
                return {e['env'] for e in case_row['per_environment'] if e[key]}
            alive, old_alive = ids('survived', row), ids('survived', base_row)
            tracking = ids('survived_and_final2s_mae_le_0_1', row)
            old_tracking = ids('survived_and_final2s_mae_le_0_1', base_row)
            lost, new = sorted(old_alive - alive), sorted(alive - old_alive)
            lost_tracking = sorted(old_tracking - tracking)
            row.update(lost_baseline_survivor_envs=lost, gained_survivor_envs=new,
                lost_baseline_survivor_count=len(lost), gained_survivor_count=len(new),
                lost_baseline_tracking_envs=lost_tracking,
                gained_tracking_envs=sorted(tracking - old_tracking))
            total_lost.extend(lost); total_new.extend(new); total_lost_tracking.extend(lost_tracking)
        run.update(survived=sum(r['survived'] for r in run['cases']),
            survived_and_final2s_mae_le_0_1=sum(r['survived_and_final2s_mae_le_0_1'] for r in run['cases']),
            lost_baseline_survivor_envs=total_lost, gained_survivor_envs=total_new,
            lost_baseline_tracking_envs=total_lost_tracking,
            diagnostic_regression=bool(total_lost or total_lost_tracking))
    report = dict(verified=True, parameter_updates=0, baseline=args.baseline,
        protocol=reference_meta, initial_qpos_qvel_bitwise_equal=True,
        all_saved_common_state_arrays_bitwise_equal=True,
        all_required_physical_arrays_bitwise_equal=None if any(missing.values()) else True,
        physical_array_limit=('Historical NPZ files omit physical arrays; runtime claims are recorded separately '
            'and do not replace saved-array equality.') if any(missing.values()) else None,
        final2s_speed_rule='Full-duration survival AND complete final 2s at full request AND per-environment vx MAE <=0.1m/s; failures remain in the total denominator.',
        final2s_descriptive_rule='Height/tilt use valid first-episode samples in the final 2s of the planned duration; case means are sample-weighted. Missing data are null. Descriptive values do not define task or balance acceptance.',
        scope='CPU-only saved fixed-request nominal diagnosis; not formal wheel-balance acceptance, full-task stability, randomized evaluation, CPU physics validation or deployment.',
        formal_balance_acceptance_performed=False, runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, output=str(args.output.resolve()),
        physical_arrays_fully_verified=report['all_required_physical_arrays_bitwise_equal'],
        runs=[dict(label=r['label'], survived=r['survived'],
            tracking=r['survived_and_final2s_mae_le_0_1'],
            lost_survivors=len(r['lost_baseline_survivor_envs']),
            diagnostic_regression=r['diagnostic_regression']) for r in runs]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', nargs=2, action='append', required=True,
                        metavar=('LABEL', 'PATH_PREFIX'))
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
