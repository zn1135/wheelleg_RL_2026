"""Audit recorded PD200 sim-to-real differences using frozen GPU evidence; no integration."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ct_mjlab.fudan_ppo_walk.policy import load_policy
from .analyze_usb_pd500 import decode, NAMES
from .replay_usb_pd_targets import sha, write

ORDER = ('lf', 'lr', 'lw', 'rf', 'rr', 'rw')
LEGS = [0, 1, 3, 4]
COL = {name: i for i, name in enumerate(NAMES)}


def signals(rows):
    get = lambda name: rows[:, COL[name]]
    obs = np.stack([get(f'obs_{i}') for i in range(25)], axis=1)
    g = obs[:, 3:6].copy()
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    angles = np.rad2deg(np.c_[np.arctan2(-g[:, 1], -g[:, 2]), np.arcsin(g[:, 0].clip(-1, 1))])
    return dict(obs=obs, command=obs[:, 6:9]/[2., .25, 5.], angles=angles,
                **{field: np.stack([get(field+'_'+name) for name in ORDER], axis=1)
                   for field in ('current', 'target', 'velocity_rad_s', 'pd_model_nm', 'request_firmware_nm')})


def motion(current, target, velocity, torque, angles, fs=200.):
    q = np.rad2deg(current[:, LEGS]); f, power = welch(q, fs=fs, axis=0, nperseg=min(len(q), round(2*fs)))
    df = f[1]-f[0]
    _, target_power = welch(np.rad2deg(target[:, LEGS]), fs=fs, axis=0, nperseg=min(len(q), round(2*fs)))
    _, wheel_power = welch(velocity[:, [2, 5]], fs=fs, axis=0, nperseg=min(len(q), round(2*fs)))
    spectrum_band = (f >= .5) & (f < 15)
    bands = {f'{lo:g}-{hi:g}Hz': np.sqrt(power[(f >= lo) & (f < hi)].sum(axis=0)*df).tolist()
             for lo, hi in ((.5, 3), (3, 10), (10, 50))}
    return dict(leg_peak_to_peak_deg=np.ptp(q, axis=0).tolist(),
                leg_error_mean_deg=np.rad2deg((target-current)[:, LEGS].mean(axis=0)).tolist(),
                leg_error_RMS_deg=np.rad2deg(np.sqrt(np.mean((target-current)[:, LEGS]**2, axis=0))).tolist(),
                leg_band_RMS_deg=bands,
                leg_target_band_3_10Hz_RMS_deg=np.sqrt(target_power[(f >= 3) & (f < 10)].sum(axis=0)*df).tolist(),
                leg_target_dominant_Hz=f[spectrum_band][np.argmax(target_power[spectrum_band], axis=0)].tolist(),
                wheel_speed_dominant_Hz=f[spectrum_band][np.argmax(wheel_power[spectrum_band], axis=0)].tolist(),
                spectrum_resolution_Hz=float(df),
                wheel_speed_RMS_rad_s=np.sqrt(np.mean(velocity[:, [2, 5]]**2, axis=0)).tolist(),
                wheel_speed_mean_rad_s=velocity[:, [2, 5]].mean(axis=0).tolist(),
                wheel_target_RMS_rad_s=np.sqrt(np.mean(target[:, [2, 5]]**2, axis=0)).tolist(),
                wheel_request_RMS_Nm=np.sqrt(np.mean(torque[:, [2, 5]]**2, axis=0)).tolist(),
                roll_pitch_mean_deg=angles.mean(axis=0).tolist(), roll_pitch_peak_to_peak_deg=np.ptp(angles, axis=0).tolist())


def audit(path, policy, metadata, output):
    x, sessions, decoding = decode(path)
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite telemetry: {path}')
    t = (x[:, 0]-x[0, 0])/1e6
    valid = (x[:, COL['pd_valid']] == 1) & (x[:, COL['obs_valid']] == 1) & (x[:, COL['warmup']] == 0)
    d = x[valid]; tv = t[valid]; s = signals(d); control = metadata['control']
    dt = np.diff(tv)
    if np.any(dt <= 0):
        raise ValueError(f'Nonmonotonic telemetry: {path}')
    report = dict(source=path.name, sha256=sha(path), rows=len(x), valid_rows=len(d), duration_s=float(t[-1]), decoding=decoding,
                  PD_dt_us_quantiles=np.quantile(dt*1e6, [0, .5, .95, 1]).tolist(),
                  sequence_gaps=int(np.count_nonzero(np.diff(d[:, 1]) != 1)),
                  board_drop_counter_increment=int(d[-1, COL['board_drops']]-d[0, COL['board_drops']]),
                  roll_pitch_min_deg=s['angles'].min(axis=0).tolist(), roll_pitch_max_deg=s['angles'].max(axis=0).tolist(),
                  observation_publish_to_PD_age_ms_quantiles=np.quantile(d[:, 0]/1000-d[:, COL['obs_publish_ms']], [0, .5, .95, 1]).tolist(),
                  command_min=s['command'].min(axis=0).tolist(), command_max=s['command'].max(axis=0).tolist(),
                  controller={}, zero_segments=[])
    report['constant_value_runs'] = {}
    for name, data in (('gyro_observation', s['obs'][:, :3]), ('gravity_observation', s['obs'][:, 3:6]),
                       ('leg_position_feedback', s['current'][:, LEGS])):
        changed = np.r_[True, np.any(data[1:] != data[:-1], axis=1) | (dt > .02)]
        starts = np.flatnonzero(changed); ends = np.r_[starts[1:], len(data)]
        lengths = tv[ends-1]-tv[starts]; longest = int(np.argmax(lengths))
        report['constant_value_runs'][name] = dict(initial_constant_s=float(lengths[0]),
            longest_constant_s=float(lengths[longest]), longest_start_end_s=[float(tv[starts[longest]]), float(tv[ends[longest]-1])],
            unique_values_first3s=int(len(np.unique(data[tv < tv[0]+3], axis=0))))
    for i, name in enumerate(ORDER):
        wheel = i in (2, 5); limit = 3.9 if wheel else 40.
        kp = control['wheel_stiffness'] if wheel else control['leg_stiffness']
        kd = control['wheel_damping'] if wheel else control['leg_damping']
        error = s['target'][:, i]-s['current'][:, i]; v = s['velocity_rad_s'][:, i]; torque = s['pd_model_nm'][:, i]
        expected = np.clip(kp*error-kd*v, -limit, limit); free = abs(torque) < limit-1e-4
        matrix = error[free, None] if wheel else np.c_[error[free], -v[free]]
        fit = np.linalg.lstsq(matrix, torque[free], rcond=None)[0]
        firmware = s['request_firmware_nm'][:, i]
        scale = float(np.dot(torque, firmware)/np.dot(torque, torque))
        report['controller'][name] = dict(fitted_PD=fit.tolist(), formula_max_error_Nm=float(abs(torque-expected).max()),
                                         saturation_fraction=float((~free).mean()), firmware_over_model_request=scale,
                                         firmware_mapping_max_error=float(abs(firmware-scale*torque).max()))
    unique = np.flatnonzero(np.r_[True, np.diff(d[:, COL['obs_sequence']]) != 0])
    obs = s['obs'][unique].astype('float32'); ud = d[unique]; ut = tv[unique]
    ids = np.arange(4, len(obs)-1)
    consecutive = (np.diff(ud[:, COL['obs_sequence']]) == 1) & (np.diff(ud[:, 2]) == 0)
    consecutive &= (np.diff(ut) < .03) & (sessions[valid][unique][1:] == sessions[valid][unique][:-1])
    ids = np.asarray([i for i in ids if consecutive[i-4:i+1].all()], dtype=int)
    history = np.stack([obs[i-4:i+1].reshape(-1) for i in ids])
    with torch.inference_mode():
        prediction, estimate = policy.act_inference(torch.from_numpy(obs[ids]), torch.from_numpy(history))
    prediction = prediction.numpy().clip(-100, 100); actual = obs[ids+1, 19:25]; difference = prediction-actual
    mapped = actual.astype('float64').copy()
    mapped[:, LEGS] = np.asarray(control['nominal_leg_positions'])+actual[:, LEGS]*control['leg_position_scale']
    mapped[:, [2, 5]] = np.clip(actual[:, [2, 5]]*control['wheel_velocity_scale'], -control['max_wheel_velocity'], control['max_wheel_velocity'])
    report['policy'] = dict(reconstructed_frames=len(ids), RMS_action_error=float(np.sqrt(np.mean(difference**2))),
                            max_action_error=float(abs(difference).max()),
                            matching_fraction=float(np.all(abs(difference) <= 1e-4+1e-4*abs(actual), axis=1).mean()),
                            target_mapping_max_error=abs(mapped-s['target'][unique[ids]]).max(axis=0).tolist(),
                            policy_dt_us_quantiles=np.quantile(np.diff(ut)*1e6, [0, .5, .95, 1]).tolist())
    # 观测与同次PD角的时间匹配仅作相对年龄诊断，不冒充完整链路延迟。
    obsq = obs[:, 9:13]+np.asarray(control['nominal_leg_positions'])
    lag_ms = np.arange(31); lag_errors = []
    use = (ut >= tv[0]+.04) & (ut <= tv[-1]-.04)
    for lag in lag_ms:
        interpolated = np.stack([np.interp(ut[use]-lag/1000, tv, s['current'][:, i]) for i in LEGS], axis=1)
        lag_errors.append(np.sqrt(np.mean((obsq[use]-interpolated)**2, axis=0)))
    lag_errors = np.asarray(lag_errors)
    report['relative_observation_alignment'] = dict(best_lag_ms_per_leg=lag_ms[np.argmin(lag_errors, axis=0)].tolist(),
        best_RMS_deg_per_leg=np.rad2deg(lag_errors.min(axis=0)).tolist(),
        limitation='Observation pose versus contemporaneous PD pose only; source sensor timestamp, actual inference completion and motor current arrival are absent.')
    zero = valid & (abs(x[:, COL['obs_6']]) < 1e-5) & (abs(x[:, COL['obs_7']]) < 1e-5)
    boundaries = np.r_[0, np.flatnonzero(np.diff(zero))+1, len(x)]
    windows = []
    for first, end in zip(boundaries[:-1], boundaries[1:]):
        duration = t[end-1]-t[first]
        if not zero[first] or duration < 1.2:
            continue
        ids0 = np.arange(first, end); ids0 = ids0[t[ids0] >= t[first]+1.]
        z = signals(x[ids0]); flat = bool(np.max(np.ptp(z['current'][:, LEGS], axis=0)) < 1e-6)
        item = dict(start_s=float(t[first]), end_s=float(t[end-1]), duration_s=float(duration), discard_first_s=1.,
                    command_height_min_max=[float(z['command'][:, 2].min()), float(z['command'][:, 2].max())],
                    frozen_joint_positions=flat,
                    **motion(z['current'], z['target'], z['velocity_rad_s'], z['pd_model_nm'], z['angles']))
        item['steady_comparison_eligible'] = bool(duration >= 3 and not flat and abs(z['angles']).max() <= 10
            and np.allclose(z['command'][:, 2], .22))
        report['zero_segments'].append(item)
        if item['steady_comparison_eligible']:
            windows.append((item, t[ids0], z))
    # 离线复算的估速只用于展示，不能作为实测机身速度。
    np.savez_compressed(output/(path.stem+'-policy.npz'), time_s=ut[ids], predicted_action=prediction,
                        recorded_next_previous_action=actual, estimated_velocity=estimate.numpy())
    return report, (tv, s), windows


def reference(root):
    protocol = json.loads((root/'gpu/protocol.json').read_text())
    with np.load(root/'motor-pd200.npz') as saved:
        values = saved['values'].copy(); times = saved['time'].copy()
        if saved['fields'].tolist() != ['target', 'actual', 'velocity', 'torque'] or saved['motor_order'].tolist() != list(ORDER):
            raise ValueError('GPU motor field ordering differs')
    reports = []
    for j, sample in enumerate(protocol['samples']):
        if sample['case']['category'] != 'zero':
            continue
        mask = times >= 25.; a = values[mask, j].reshape(-1, 4, 6)
        path = root/'gpu/traces'/(sample['id']+'.npz')
        with np.load(path) as trace:
            rp = np.stack([np.interp(times[mask], trace['time'], trace['roll_pitch_deg'][:, k]) for k in (0, 1)], axis=1)
        reports.append(dict(id=sample['id'], command_height_m=sample['case']['segments'][0]['command'][2],
                            window_s=[25., 30.], **motion(a[:, 1], a[:, 0], a[:, 2], a[:, 3], rp), trace_sha256=sha(path)))
    return dict(source=str(root.resolve()), motor_sha256=sha(root/'motor-pd200.npz'), samples=reports,
                limitation='Same frozen checkpoint, nominal GPU zero command; real initial state, support, bias and prior commands differ. Descriptive comparison, not paired causal proof.')


def plot(output, reports, traces, windows):
    fig, axes = plt.subplots(len(reports), 3, figsize=(15, 12), squeeze=False)
    for j, ((t, s), report) in enumerate(zip(traces, reports)):
        axes[j, 0].plot(t, s['command'][:, 0], label='vx m/s')
        axes[j, 0].plot(t, s['command'][:, 1], label='yaw rad/s')
        axes[j, 1].plot(t, s['angles'][:, 0], label='roll')
        axes[j, 1].plot(t, s['angles'][:, 1], label='pitch from observation')
        for k, name in zip((2, 5), ('left', 'right')):
            axes[j, 2].plot(t, s['velocity_rad_s'][:, k], label=name, lw=.6)
        for segment in report['zero_segments']:
            for ax in axes[j]:
                ax.axvspan(segment['start_s'], segment['end_s'], color='green', alpha=.08)
        axes[j, 0].set_ylabel(f'Part {j+1}')
        for ax in axes[j]:
            ax.grid(alpha=.2); ax.legend(fontsize=7, loc='upper right')
    for ax, title in zip(axes[0], ('Commands / shaded: zero command', 'Observation tilt (deg)', 'Measured wheel speed (rad/s)')):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel('Seconds within original file')
    fig.tight_layout(); fig.savefig(output/'overview.png', dpi=140); plt.close(fig)
    if windows:
        label, item, t, s = max(windows, key=lambda row: row[1]['duration_s'])
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        for ax, k in zip(axes.flat, range(6)):
            scale = 1. if k in (2, 5) else 180/np.pi
            actual = s['velocity_rad_s'][:, k] if k in (2, 5) else s['current'][:, k]
            ax.plot(t, actual*scale, label='Measured')
            ax.plot(t, s['target'][:, k]*scale, label='Target', alpha=.8)
            ax.set_title(ORDER[k]+(' (rad/s)' if k in (2, 5) else ' (deg)')); ax.grid(alpha=.2); ax.legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel('Seconds within original file')
        fig.suptitle(f'{label}, zero command, first 1s discarded; static leg error also exists in simulation')
        fig.tight_layout(); fig.savefig(output/'zero-command-tracking.png', dpi=150); plt.close(fig)


def plot_frequency(output, windows, gpu_root):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    groups = [(label.split('-')[2], s) for label, _, _, s in windows]
    protocol = json.loads((gpu_root/'gpu/protocol.json').read_text())
    index = next(i for i, s in enumerate(protocol['samples']) if s['id'] == 'zero_0_0_0.22__50001')
    with np.load(gpu_root/'motor-pd200.npz') as a:
        times = a['time'][a['time'] >= 25]; values = a['values'][a['time'] >= 25, index].reshape(-1, 4, 6)
    with np.load(gpu_root/'gpu/traces/zero_0_0_0.22__50001.npz') as a:
        angles = np.stack([np.interp(times, a['time'], a['roll_pitch_deg'][:, k]) for k in (0, 1)], axis=1)
    groups.append(('GPU 25-30s', dict(target=values[:, 0], velocity_rad_s=values[:, 2], angles=angles)))
    for label, data in groups:
        signals0 = (np.rad2deg(data['target'][:, 3]), data['velocity_rad_s'][:, 2], data['angles'][:, 1])
        for ax, signal in zip(axes, signals0):
            f, power = welch(signal, fs=200, nperseg=min(len(signal), 400))
            ax.semilogy(f, np.maximum(power, 1e-12), label=label, lw=1.2)
    for ax, title in zip(axes, ('Right-front leg TARGET (deg^2/Hz)', 'Left wheel measured speed ((rad/s)^2/Hz)', 'Pitch: real obs / GPU state (deg^2/Hz)')):
        ax.set_xlim(.5, 15); ax.set_title(title, fontsize=10); ax.set_xlabel('Hz'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same model and 0.22m zero command; different initial state/prior commands, not paired physics')
    fig.tight_layout(); fig.savefig(output/'frequency-gap.png', dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--logs', type=Path, required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpu-reference', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    gpu_plan = json.loads((args.gpu_reference/'plan.json').read_text())
    if gpu_plan['sha256'] != sha(args.checkpoint) or sha(gpu_plan['checkpoint']) != gpu_plan['sha256']:
        raise ValueError('GPU checkpoint differs')
    with redirect_stdout(io.StringIO()):
        policy, metadata = load_policy(args.checkpoint, 'cpu')
    reports = []; traces = []; windows = []
    for path in sorted(args.logs.glob('*.bin')):
        report, trace, found = audit(path, policy, metadata, args.output)
        reports.append(report); traces.append(trace)
        windows += [(path.stem, *item) for item in found]
    gpu = reference(args.gpu_reference)
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                  control=metadata['control'], control_timing=metadata['control_timing'], training_disturbances=metadata['disturbances'],
                  sessions=reports, GPU_reference=gpu, physics_steps=0, training_updates=0,
                  offline_policy_frames=sum(r['policy']['reconstructed_frames'] for r in reports),
                  limitations=['Requested torque is not measured output torque/current.',
                               'Body translation, body height, support/contact and raw unbiased IMU are not recorded.',
                               'Short mixed-command zero segments do not establish full30s parking acceptance.',
                               'Observation publish age is not total sensor/action latency.'])
    write(args.output/'summary.json', result)
    plot(args.output, reports, traces, windows)
    plot_frequency(args.output, windows, args.gpu_reference)
    print(json.dumps(dict(output=str(args.output), policy_frames=result['offline_policy_frames'],
                          max_action_error=max(r['policy']['max_action_error'] for r in reports),
                          steady_zero_windows=len(windows), GPU_reference=gpu['samples']), indent=2))


if __name__ == '__main__':
    main()
