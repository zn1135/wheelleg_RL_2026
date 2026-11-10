"""Quantify one complete real right-leg pose transition from paired raw records."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from scripts.agent.real2sim.control_leg_pose_gpu import inverse_pose, leg_pose


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steady-observation', type=Path, required=True)
    parser.add_argument('--exit-observation', type=Path, required=True)
    parser.add_argument('--user-motion-observation', help='Operator observation quoted from this trial; never inferred from telemetry')
    parser.add_argument('--motor-power-off-confirmed', action='store_true')
    args = parser.parse_args()
    audit = json.loads((args.run / 'report.json').read_text())
    assert len(audit['runs']) == 1 and audit['runs'][0]['reached_stage3']
    assert not audit['sequence_gap_count'] and not audit['crc_rejected']
    with np.load(args.run / 'paired-records.npz') as saved:
        rows = {kind: saved[f'kind{kind}'] for kind in (1, 3, 4, 5)}
    maps = {kind: {int(row[4]): row for row in values} for kind, values in rows.items()}
    ids = set.intersection(*(set(values) for values in maps.values()))
    ids = sorted(ids, key=lambda key: maps[5][key][14])
    control, state, geometry, target = [np.array([maps[kind][key] for key in ids]) for kind in (1, 3, 4, 5)]
    assert len(ids) == len(target) == len(rows[5])
    elapsed = target[:, 14].astype(np.int64)
    assert elapsed[0] == 0 and np.all(np.diff(elapsed) == 2)
    profile=int(target[0,20])&28
    assert profile in (8,16)
    assert not np.any(target[:, 5]) and np.all((target[:,20].astype(int)&31)==profile|1)
    assert np.all(control[:, 20].astype(int) & 32) and not np.any(control[:, 5])
    assert not np.any(control[:, [6, 7, 10, 11]])
    assert not np.any((target[:, 20].astype(int) >> 8) & 15)
    initial_l, initial_a, _ = leg_pose(state[:1, 6:10])
    times = elapsed / 1000
    phase = np.clip((times - 5) / 20, 0, 1)
    blend = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
    target_l = initial_l[0, 1] + blend * (.19 - initial_l[0, 1])
    goal_angle=initial_a[0,1] if profile==16 else np.pi/6
    target_a = initial_a[0, 1] + blend * (goal_angle - initial_a[0, 1])
    expected_fw = inverse_pose(target_l, target_a)
    target_error = float(np.max(abs(expected_fw - target[:, 8:10])))
    assert target_error < 3e-6
    actual_l = geometry[:, 12]
    actual_a = np.array([struct.unpack('<f', struct.pack('<I', int(value)))[0] for value in geometry[:, 14]])
    reconstructed_l, reconstructed_a, _ = leg_pose(state[:, 6:10])
    lparity = float(np.max(abs(actual_l - reconstructed_l[:, 1])) * 1000)
    aparity = float(np.degrees(np.max(abs(actual_a - reconstructed_a[:, 1]))))
    assert lparity < .01 and aparity < .01
    moving = (elapsed >= 5000) & (elapsed <= 25000)
    settled = elapsed >= elapsed[-1] - 3000
    after_goal=times>=25.
    within_band=after_goal&(abs(actual_l-.19)<=.001)&(abs(actual_a-goal_angle)<=np.deg2rad(.2))
    band_window=1501
    band_sum=np.r_[0,np.cumsum(within_band.astype(int))]
    band_starts=np.flatnonzero(band_sum[band_window:]-band_sum[:-band_window]==band_window)
    band_start_s=float(times[band_starts[0]]) if len(band_starts) else None
    steady = json.loads(args.steady_observation.read_text())
    exit_state = json.loads(args.exit_observation.read_text())
    assert steady['joint_capture']['stage'] == 3 and steady['joint_capture']['flags'] == profile|1 and not steady['fault']
    assert exit_state['strategy'] == 2 and not exit_state['fault'] and not any(exit_state['request_nm'])
    report = dict(source_window=audit['stream_window_sha256'],target_samples=len(ids),time_range_s=[float(times[0]),float(times[-1])],
                  profile='right-length-only-v1' if profile==16 else 'right-pose-v1',target_right_angle_deg=float(np.degrees(goal_angle)),
                  initial_right_length_mm=float(initial_l[0, 1] * 1000),initial_right_angle_deg=float(np.degrees(initial_a[0, 1])),
                  target_formula_max_error_rad=target_error,geometry_length_parity_max_mm=lparity,geometry_angle_parity_max_deg=aparity,
                  moving_length_rmse_mm=float(np.sqrt(np.mean((actual_l[moving] - target_l[moving]) ** 2)) * 1000),
                  moving_angle_rmse_deg=float(np.degrees(np.sqrt(np.mean((actual_a[moving] - target_a[moving]) ** 2)))),
                  moving_length_max_error_mm=float(np.max(abs(actual_l[moving]-target_l[moving]))*1000),
                  moving_angle_max_error_deg=float(np.degrees(np.max(abs(actual_a[moving]-target_a[moving])))),
                  full_window_angle_max_error_deg=float(np.degrees(np.max(abs(actual_a-target_a)))),
                  last3s_length_max_error_mm=float(np.max(abs(actual_l[settled] - .19)) * 1000),
                  last3s_angle_max_error_deg=float(np.degrees(np.max(abs(actual_a[settled] - goal_angle)))),
                  first_3s_within_1mm_and_point2deg_s=band_start_s,
                  settling_delay_after_target_s=band_start_s-25. if band_start_s is not None else None,
                  minimum_length_after_target_mm=float(np.min(actual_l[after_goal])*1000),
                  record_end_right_length_mm=float(actual_l[-1]*1000),record_end_right_angle_deg=float(np.degrees(actual_a[-1])),
                  max_right_motor_reported_speed_rad_s=float(np.max(abs(state[:, 12:14]))),
                  peak_right_request_nm=np.max(abs(control[:, 8:10]),axis=0).tolist(),left_and_wheel_requests_zero=True,
                  faults_zero=True,no_locked_or_saturated_records=True,
                  steady_right_length_mm=steady['lengths_m'][1] * 1000,steady_right_angle_deg=float(np.degrees(steady['angles_rad'][1])),
                  steady_elapsed_s=steady['joint_capture']['elapsed_ms'] / 1000,exit_requests_zero=True,
                  user_report=args.user_motion_observation,motor_power_off_confirmed=args.motor_power_off_confirmed,
                  hardware_commands=False,new_physics_steps=0,
                  scope='Single real closed-loop setpoint trial; not independent physical spring calibration or whole-range validation')
    np.savez_compressed(args.run / 'tracking.npz',time_s=times,target_length_m=target_l,target_angle_rad=target_a,
                        actual_length_m=actual_l,actual_angle_rad=actual_a,request_firmware_nm=control[:, 6:12],
                        measured_fw_q=state[:, 6:10],target_fw_q=target[:, 6:10])
    (args.run / 'tracking-report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].plot(times,actual_l*1000,label='Real solved length');axes[0].plot(times,target_l*1000,'--',label='Target')
    axes[1].plot(times,np.degrees(actual_a),label='Real solved angle');axes[1].plot(times,np.degrees(target_a),'--',label='Target')
    axes[2].plot(times,control[:,8],label='Right front request');axes[2].plot(times,control[:,9],label='Right rear request')
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Leg length (mm)');axes[1].set_ylabel('Leg angle (deg)');axes[2].set_ylabel('Request (Nm)');axes[2].set_xlabel('Time from engagement (s)')
    fig.tight_layout();fig.savefig(args.run/'tracking.png',dpi=140)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
