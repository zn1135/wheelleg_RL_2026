#!/usr/bin/env python3
"""Compare saved student-PD motions with matched LQR demonstrations; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion(root, motor_q, motor_dq, times):
    late = times > 15.
    legs = [0, 1, 3, 4]
    angles = motor_q[late][:, legs]
    centered = angles - angles.mean(0)
    power = np.abs(np.fft.rfft(centered*np.hanning(len(angles))[:, None], axis=0))**2
    frequencies = np.fft.rfftfreq(len(angles), .01)
    return dict(
        last5s_leg_velocity_rms_rad_s=float(np.sqrt(np.square(motor_dq[late][:, legs]).mean())),
        last5s_mean_speed_m_s=float(np.linalg.norm(root[late, 7:9], axis=-1).mean()),
        last5s_xy_excursion_m=float(np.linalg.norm(root[late, :2]-root[late, :2][0], axis=-1).max()),
        last5s_leg_angle_peak_frequency_hz=frequencies[np.argmax(power[1:], axis=0)+1].tolist(),
        last5s_leg_angle_power_fraction_ge10hz=float(power[frequencies >= 10].sum()/max(power.sum(), 1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'result.json').read_text())
    manifest = json.loads(args.dataset.read_text())
    assert report['compensation'] == 'none'
    if report['training_delay_grid']:
        assert report['delay_pairs_s'] == [[.002, .002], [.002, .004], [.004, .002], [.004, .004]]
    else:
        assert report['stress_delay_ms'] == 10
        assert report['delay_pairs_s'] == [[.002, .010], [.004, .010], [.010, .002], [.010, .004], [.010, .010]]
    assert digest(args.run/'trace.npz') == report['trace_sha256']
    with np.load(args.run/'substeps.npz') as high_rate:
        for key in ('q', 'dq', 'target', 'actual', 'actuator_force', 'post_dq'):
            assert np.isfinite(high_rate[key]).all(), key
        hold = high_rate['hold']
        np.testing.assert_array_equal(hold, np.broadcast_to(
            (high_rate['time'] < .2)[:, None], hold.shape))
        assert not high_rate['actual'][hold].any()
        assert not high_rate['actuator_force'][hold].any()
    episodes = {row['episode_id']: row for row in manifest['episodes']}
    rows, teachers, source_hashes = [], {}, {}
    with np.load(args.run/'trace.npz') as trace:
        for key in ('root', 'qpos', 'qvel', 'motor_q', 'motor_dq', 'torque', 'action'):
            assert np.isfinite(trace[key]).all(), key
        for row in report['rows']:
            assert row['full_20s'], 'Incomplete episodes require a separate failed-motion assessment'
            episode_id, index = row['episode_id'], row['env']
            if episode_id not in teachers:
                episode = episodes[episode_id]
                path = args.dataset.parent/episode['path']
                assert digest(path) == episode['sha256']
                source_hashes[episode_id] = digest(path)
                with np.load(path) as source:
                    teachers[episode_id] = {key: source[key].copy() for key in
                        ('time', 'root', 'motor_q', 'motor_dq', 'initial_qpos', 'initial_qvel')}
            teacher = teachers[episode_id]
            np.testing.assert_allclose(trace['time'], teacher['time'], atol=1e-6)
            np.testing.assert_allclose(trace['initial_qpos'][index, 2:], teacher['initial_qpos'][2:], atol=1e-6)
            np.testing.assert_allclose(trace['initial_qvel'][index], teacher['initial_qvel'], atol=1e-6)
            metrics = motion(trace['root'][:, index], trace['motor_q'][:, index], trace['motor_dq'][:, index], trace['time'])
            reference = motion(teacher['root'], teacher['motor_q'], teacher['motor_dq'], teacher['time'])
            rows.append(dict(env=index, episode_id=episode_id, leg_delay_s=row['leg_delay_s'],
                wheel_delay_s=row['wheel_delay_s'], strict_rise=row['strict_rise'],
                final_balance_v31=row['final_balance_v31'], student=metrics, teacher=reference,
                leg_velocity_rms_ratio_to_teacher=metrics['last5s_leg_velocity_rms_rad_s']/reference['last5s_leg_velocity_rms_rad_s']))
        worst = max(rows, key=lambda row: row['student']['last5s_leg_velocity_rms_rad_s'])
        teacher = teachers[worst['episode_id']]
        late = trace['time'] > 15.
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
        cases = [('Original LQR', teacher['root'], teacher['motor_dq'])]
        for pair in report['delay_pairs_s']:
            row = next(row for row in rows if row['episode_id'] == worst['episode_id']
                       and [row['leg_delay_s'], row['wheel_delay_s']] == pair)
            cases.append((f'PD {pair[0]*1000:.0f}/{pair[1]*1000:.0f} ms', trace['root'][:, row['env']], trace['motor_dq'][:, row['env']]))
        for label, root, dq in cases:
            axes[0].plot(trace['time'][late], dq[late, 0], label=label, linewidth=1)
            axes[1].plot(trace['time'][late], np.linalg.norm(root[late, 7:9], axis=-1), linewidth=1)
            axes[2].plot(trace['time'][late], np.linalg.norm(root[late, :2]-root[late, :2][0], axis=-1), linewidth=1)
        axes[0].set_ylabel('Leg lf0 velocity (rad/s)'); axes[0].legend(ncol=3)
        axes[1].set_ylabel('XY speed (m/s)'); axes[2].set_ylabel('XY excursion (m)')
        axes[2].set_xlabel('Time (s)')
        figure.suptitle(f'Matched initial pose: largest late leg-speed RMS in the {len(rows)}-case grid')
        figure.savefig(args.run/'motion-comparison.png', dpi=150)
        plt.close(figure)
    result = dict(rows=rows, count=len(rows), balanced=sum(row['final_balance_v31'] for row in rows),
        strict_rise=sum(row['strict_rise'] for row in rows), worst_leg_rms_env=worst['env'],
        trace_sha256=report['trace_sha256'], teacher_trace_sha256=source_hashes,
        substeps_sha256=digest(args.run/'substeps.npz'), hold020_motor_force_zero=True,
        assessor_sha256=digest(__file__), optimizer_updates=0,
        scope='Four existing initial poses, nominal physics, 20 seconds. Matched 100Hz late-motion metrics; >=10Hz angle power is diagnostic, not a smoothness pass threshold. Final balance does not certify parking or all-pose generalization.')
    (args.run/'motion-assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'teacher_trace_sha256')}, indent=2))


if __name__ == '__main__':
    main()
