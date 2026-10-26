#!/usr/bin/env python3
"""Audit exact 500Hz FIFO/PD algebra, preview error and paired balance outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(args):
    report = json.loads((args.run/'result.json').read_text())
    with np.load(args.run/'substeps.npz') as trace:
        arrays = {key: trace[key] for key in trace.files}
    with np.load(args.run/'trace.npz') as trace:
        features = trace['features'].copy()
        initial = trace['initial_qpos'].copy()
        current_requests = trace['requested_torque'].copy()
    q, dq, target = arrays['q'], arrays['dq'], arrays['target']
    legs, wheels = [0, 1, 3, 4], [2, 5]
    reconstructed = np.zeros_like(dq)
    reconstructed[:, :, legs] = 60*(target[:, :, legs]-q[:, :, legs])-2*dq[:, :, legs]
    reconstructed[:, :, wheels] = .1*(target[:, :, wheels]-dq[:, :, wheels])
    limit = np.asarray([40, 40, 3.9, 40, 40, 3.9])
    reconstructed = np.clip(reconstructed, -limit, limit)
    reconstructed[arrays['hold']] = 0
    error = float(np.abs(reconstructed-arrays['actual']).max())
    assert error < 1e-4, error
    groups = []
    for leg_delay, wheel_delay in report['delay_pairs_s']:
        rows = [row for row in report['rows'] if row['leg_delay_s'] == leg_delay and row['wheel_delay_s'] == wheel_delay]
        ids = [row['env'] for row in rows]
        late = arrays['time'] >= 15.
        source = arrays['requested'][late][:, ids][:, :, wheels]
        age_error = .1*(arrays['predicted_dq'][late][:, ids][:, :, wheels]-dq[late][:, ids][:, :, wheels])
        actual = arrays['actual'][late][:, ids][:, :, wheels]
        identity_error = float(np.abs(np.clip(source+age_error, -3.9, 3.9)-actual).max())
        assert identity_error < 1e-4
        ticks = np.arange(len(dq))[:, None]-arrays['generation_tick'][:, ids, 1]
        first_application = (ticks == round(wheel_delay/.002)) & ~arrays['hold'][:, ids]
        pred_error = (arrays['predicted_dq'][:, ids][:, :, wheels]-dq[:, ids][:, :, wheels])[first_application]
        policy_tick = np.minimum(np.arange(len(dq))//5, len(current_requests)-1)
        settled_first = first_application & (arrays['time'][:, None] >= 2.)
        request_error = (arrays['requested'][:, ids][:, :, wheels]-current_requests[policy_tick][:, ids][:, :, wheels])[settled_first]
        gyro = np.linalg.norm(features[-100:, ids, 7:9], axis=-1)
        group = dict(leg_delay_s=leg_delay, wheel_delay_s=wheel_delay,
            balanced=sum(row['final_balance_v31'] for row in rows), count=len(rows),
            source_age_range_ms=[int(ticks[late].min())*2, int(ticks[late].max())*2],
            delayed_wheel_request_rms_nm=float(np.sqrt(np.square(source).mean())),
            speed_age_torque_term_rms_nm=float(np.sqrt(np.square(age_error).mean())),
            pd_algebra_error_nm=identity_error,
            first_application_wheel_velocity_prediction_rms_rad_s=float(np.sqrt(np.square(pred_error).mean())),
            first_application_request_vs_current_lqr_rms_nm=float(np.sqrt(np.square(request_error).mean())),
            last1s_gyro_xy_max_rad_s=float(gyro.max()),
            last5s_leg_speed_rms_range_rad_s=[min(row['last5s_leg_velocity_rms_rad_s'] for row in rows), max(row['last5s_leg_velocity_rms_rad_s'] for row in rows)],
            last5s_mean_speed_range_m_s=[min(row['last5s_mean_speed_m_s'] for row in rows), max(row['last5s_mean_speed_m_s'] for row in rows)])
        groups.append(group)
    output = dict(compensation=report.get('compensation', 'none'), groups=groups,
        optimizer_updates=0, pd_reconstruction_max_error_nm=error,
        substeps_sha256=digest(args.run/'substeps.npz'), trace_sha256=digest(args.run/'trace.npz'),
        balanced=sum(row['final_balance_v31'] for row in report['rows']), count=len(report['rows']),
        source_sha256=digest(__file__),
        scope='Exact nominal PD command decomposition; actuator_force separately stores physical motor force. Algebra alone is not proof of the sole oscillation cause. Prediction errors compare causal predictions with subsequently observed states, never feed future observations into the controller.')
    if args.baseline is not None:
        old = json.loads((args.baseline/'result.json').read_text())
        with np.load(args.baseline/'trace.npz') as trace:
            np.testing.assert_allclose(initial[:, 2:], trace['initial_qpos'][:, 2:], atol=1e-7)
        assert old['xml_sha256'] == report['xml_sha256']
        assert old['dataset_fingerprint'] == report['dataset_fingerprint']
        assert old['delay_pairs_s'] == report['delay_pairs_s']
        regressions = []
        for before, after in zip(old['rows'], report['rows'], strict=True):
            assert before['episode_id'] == after['episode_id']
            if before['final_balance_v31'] and not after['final_balance_v31']:
                regressions.append(after['env'])
        output['regressed_envs'] = regressions
        output['paired_balance_accepted'] = not regressions and output['balanced'] == output['count']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    assess(parser.parse_args())
