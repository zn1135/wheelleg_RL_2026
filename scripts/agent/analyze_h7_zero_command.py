#!/usr/bin/env python3
"""Offline zero-command telemetry and policy sensitivity, without calibration.

Frozen-sensor policy iterations and substituted inputs are diagnostic counterfactuals,
not predictions of real motion. Tethered samples never enter a physical fit.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror

ORDER = np.array([0, 1, 3, 4, 2, 5])
GROUPS = ('actual', 'ground_flag_zero', 'gyro_zero', 'level_gravity', 'pitch_zero', 'roll_zero',
          'nominal_leg_pose', 'motor_velocity_zero', 'history_zero', 'wheel_history_zero')


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': np.mean(values, axis=0).tolist(), 'mean_abs': np.mean(abs(values), axis=0).tolist(),
            'min': np.min(values, axis=0).tolist(), 'max': np.max(values, axis=0).tolist()}


def substitute(obs, group):
    result = obs.copy()
    if group == 'ground_flag_zero': result[..., 35] = 0
    elif group == 'gyro_zero': result[..., 4:7] = 0
    elif group == 'level_gravity': result[..., 7:10] = [0, 0, -1]
    elif group == 'pitch_zero':
        result[..., 7] = 0
        result[..., 8:10] /= np.linalg.norm(result[..., 8:10], axis=-1, keepdims=True)
    elif group == 'roll_zero':
        result[..., 8] = 0
        result[..., 9] = -np.sqrt(np.clip(1-result[..., 7]**2, 0, 1))
    elif group == 'nominal_leg_pose': result[..., 10:14] = 0
    elif group == 'motor_velocity_zero': result[..., 16:22] = 0
    elif group == 'history_zero': result[..., 22:28] = 0
    elif group == 'wheel_history_zero': result[..., 26:28] = 0
    return result


def forward_target(action):
    # XML left/right wheel axes are +Y/-Y; unify rolling direction, without
    # converting wheel rotation into an unmeasured real body displacement.
    return .5 * (np.clip(10 * action[..., 2], -150, 150)
                 - np.clip(10 * action[..., 5], -150, 150))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--cpu-report', type=Path, required=True)
    p.add_argument('--cpu-snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    # Load the exact baseline implementation/assets; another chat can update
    # active XML or playback while this historical recording is being analyzed.
    global sim
    sys.path.insert(0, str(args.cpu_snapshot/'sim2sim'))
    module_path = args.cpu_snapshot/'sim2sim/mj_sim2sim_huananhu_multimode.py'
    spec = importlib.util.spec_from_file_location('zero_command_frozen_cpu', module_path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    for path in (Path(__file__), Path(sim.__file__), Path(mirror.__file__), ROOT/'sim2sim/wheel_response.py'):
        shutil.copy2(path, source/path.name)
    meta, _, config = mirror.load_session(args.capture_root/'session',
                         machine_config=args.capture_root/'jetson-source/machine_config.c')
    prior = json.loads((args.capture_root/'chronology/report.json').read_text())
    if prior['stream_sha256'] != meta['stream_sha256']:
        raise ValueError('Chronology does not belong to this raw recording')
    saved = np.load(args.capture_root/'chronology/frames.npz')
    f, rx, ticks = saved['frames'], saved['receive_s'], saved['board_tick_ms']
    policies = np.load(args.capture_root/'details-final/policy-all.npz')
    rows, obs = policies['rows'], policies['obs']
    if not np.array_equal(obs, f[rows, 57:96]):
        raise ValueError('Policy inputs differ from saved source frames')
    # Use only snapshots whose own input is active NORMAL, not stale RECOVER pairs.
    good = ((f[rows, 109].astype(int) & 33) == 33) & (obs[:, 28] == 1) & (obs[:, 36] == 1)
    good &= (f[rows, 154] == 3)
    rows, obs = rows[good], obs[good]
    if not len(obs) or not np.all(obs[:, [0, 1, 2]] == 0):
        raise ValueError('Expected nonempty zero-motion-command NORMAL samples')
    checkpoint = ROOT/'deployment/H7_RL_ros/models/model6848-20261021/model_6848.pt'
    actor = sim.load_actor(checkpoint)

    def infer(values):
        with torch.inference_mode():
            result = actor(torch.from_numpy(np.asarray(values, dtype=np.float32))).numpy()
        if not np.isfinite(result).all(): raise ValueError('Nonfinite policy output')
        return result

    actual = infer(obs)
    parity = float(abs(actual-f[rows, 96:102]).max())
    if parity > sim.ONNX_ACTION_ATOL: raise ValueError('Policy does not reproduce recording')
    intervals = ((20.4, 22), (22, 25), (25, 28), (28, 30), (30, 35), (35, 40), (50, 71.1))
    period_rows, sensitivity = [], []
    for lo, hi in intervals:
        mask = (rx >= lo) & (rx < hi) & ((f[:, 109].astype(int) & 33) == 33)
        measured = .5*(f[mask, 20]+f[mask, 21])
        target = .5*(f[mask, 114]-f[mask, 115])
        torque = .5*(f[mask, 136]+f[mask, 137])
        period_rows.append({'receive_interval_s': [lo, hi], 'frames': int(mask.sum()),
                            'measured_wheel_rad_s': stats(measured), 'target_wheel_rad_s': stats(target),
                            'target_positive_fraction': float(np.mean(target > .1)),
                            'request_per_wheel_firmware_nm': stats(torque)})
    predicted = {}
    for group in GROUPS:
        predicted[group] = infer(substitute(obs, group))
        for lo, hi in intervals:
            mask = (rx[rows] >= lo) & (rx[rows] < hi)
            sensitivity.append({'group': group, 'receive_interval_s': [lo, hi],
                                'pairs': int(mask.sum()),
                                'forward_wheel_target_rad_s': stats(forward_target(predicted[group][mask]))})
    np.savez_compressed(args.out/'substituted-policy.npz', rows=rows, obs=obs,
                        **{group: value for group, value in predicted.items()})

    # Keep physical sensor channels fixed and iterate only the documented previous
    # action slots. This is a network recurrence experiment, not physical time.
    representative = [int(np.argmin(abs(rx[rows]-at))) for at in (22., 25., 28., 35., 65.)]
    frozen, labels = [], []
    for index in representative:
        for group in ('actual', 'ground_flag_zero', 'gyro_zero', 'level_gravity',
                      'pitch_zero', 'roll_zero', 'nominal_leg_pose'):
            for history in ('recorded', 'zero'):
                value = substitute(obs[index], group)
                if history == 'zero': value[22:28] = 0
                frozen.append(value)
                labels.append({'row': int(rows[index]), 'receive_s': float(rx[rows[index]]),
                               'group': group, 'initial_history': history})
    initial = np.array(frozen)
    current = initial.copy()
    recurrent = []
    for iteration in range(251):
        output = infer(current)
        recurrent.append(forward_target(output))
        # The observation uses motor order; actor outputs use left,left,wheel,...
        current[:, 22:28] = np.clip(output[:, ORDER], -100, 100)
        np.testing.assert_array_equal(current[:, :22], initial[:, :22])
        np.testing.assert_array_equal(current[:, 28:], initial[:, 28:])
    recurrent = np.array(recurrent)
    np.savez_compressed(args.out/'frozen-recurrence.npz', initial_obs=initial,
                        forward_wheel_target_rad_s=recurrent)
    frozen_report = [{**label, 'wheel_target_iteration_0': float(recurrent[0, i]),
                      'wheel_target_iteration_50': float(recurrent[50, i]),
                      'wheel_target_iteration_250': float(recurrent[250, i])}
                     for i, label in enumerate(labels)]

    # Relative attitude changes can check gyro axes/units, but cannot determine
    # the absolute mechanical level or a static IMU mounting offset.
    quat = f[:, 22:26].astype(float)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    orientation_rate = (rotation[:-1].inv()*rotation[1:]).as_rotvec()/np.diff(ticks/1000)[:, None]
    gyro = .5*(f[:-1, 26:29]+f[1:, 26:29])
    moving = (rx[:-1] >= 19.65) & (rx[:-1] < 21.5) & (np.linalg.norm(orientation_rate, axis=1) > .05)
    gyro_report = {'samples': int(moving.sum()), 'receive_interval_s': [19.65, 21.5],
                   'axis_correlation': [float(np.corrcoef(orientation_rate[moving, k], gyro[moving, k])[0, 1]) for k in range(3)],
                   'mean_abs_error_rad_s': abs(orientation_rate[moving]-gyro[moving]).mean(axis=0).tolist(),
                   'scope': 'Relative rotation/gyro consistency only; no absolute mechanical calibration.'}

    # Compare the fixed ground-start input alone in an otherwise ordinary nominal
    # CPU start. No sensor, physical parameter, production input or firmware changes.
    cpu = json.loads(args.cpu_report.read_text())
    if cpu['checkpoint_sha256'] != identity(checkpoint)['sha256']:
        raise ValueError('Different baseline checkpoint')
    env = sim.Playback('normal', (0., 0., .27))
    if cpu['model_asset_sha256'] != identity(sim.MODEL_ASSET)['sha256']:
        raise ValueError('Different baseline physical model')
    if cpu['initial_pose'] != 'normal': raise ValueError('Expected nominal baseline')
    repeat = sim.Playback('normal', (0., 0., .27))
    repeated = []
    for _ in range(round(cpu['seconds']/(sim.DT*sim.DECIMATION))):
        action, torque = repeat.step(lambda value: infer(value[None])[0])
        repeated.append(repeat.metrics(action, torque))
    compared = ('time_s', 'x_m', 'y_m', 'vx_m_s', 'height_m')
    repeat_error = max(abs(a[key]-b[key]) for a, b in zip(repeated, cpu['trajectory']) for key in compared)
    if repeat_error > 1e-9:
        raise ValueError(f'Frozen baseline repeat diverges: {repeat_error}')
    traces, cpu_obs = [], []

    def controller(value):
        value = value.copy()
        value[35] = 1
        cpu_obs.append(value)
        return infer(value[None])[0]

    for _ in range(round(cpu['seconds']/(sim.DT*sim.DECIMATION))):
        action, torque = env.step(controller)
        traces.append(env.metrics(action, torque))
    (args.out/'cpu-ground-flag1.json').write_text(json.dumps(traces, indent=2)+'\n')
    np.savez_compressed(args.out/'cpu-ground-flag1-obs.npz', obs=np.asarray(cpu_obs))

    # Exercise the ordinary ground-reset/recovery path as well. Seed the existing
    # pose-bank sampler explicitly; the live viewer's default RNG is not fixed.
    ground = sim.Playback('normal', (0., 0., .27))
    ground.rng = np.random.default_rng(42)
    ground.reset('ground')
    ground_initial_q, ground_initial_v = ground.data.qpos.copy(), ground.data.qvel.copy()
    ground_obs, ground_trace = [], []

    def ground_controller(value):
        ground_obs.append(value.copy())
        return infer(value[None])[0]

    for _ in range(round(cpu['seconds']/(sim.DT*sim.DECIMATION))):
        action, torque = ground.step(ground_controller)
        ground_trace.append(ground.metrics(action, torque))
    ground_obs = np.asarray(ground_obs)
    np.savez_compressed(args.out/'cpu-ground-seed42.npz', initial_qpos=ground_initial_q,
                        initial_qvel=ground_initial_v, obs=ground_obs)
    (args.out/'cpu-ground-seed42.json').write_text(json.dumps(ground_trace, indent=2)+'\n')

    def cpu_stats(trajectory):
        after = [v for v in trajectory if v['time_s'] >= 2]
        xy = np.array([[v['x_m'], v['y_m']] for v in trajectory])
        settled_xy = np.array([[v['x_m'], v['y_m']] for v in after])
        return {'seconds': float(trajectory[-1]['time_s']),
                'net_displacement_first_sample_to_end_m': float(np.linalg.norm(xy[-1]-xy[0])),
                'net_displacement_2s_to_end_m': float(np.linalg.norm(settled_xy[-1]-settled_xy[0])),
                'path_length_2s_to_end_m': float(np.linalg.norm(np.diff(settled_xy, axis=0), axis=1).sum()),
                'vx_after_2s_m_s': stats([v['vx_m_s'] for v in after]),
                'base_contact_samples_after_2s': sum(v['base_contact'] for v in after),
                'both_wheel_contact_samples_after_2s': sum(v['both_wheels_contact'] for v in after)}

    report = {'complete': True, 'physical_calibration_accepted': False,
              'session_stream_sha256': meta['stream_sha256'], 'checkpoint': identity(checkpoint),
              'policy_pairs': len(rows), 'max_recorded_action_error': parity,
              'recording_intervals': period_rows, 'single_step_sensitivity': sensitivity,
              'frozen_sensor_recurrence': frozen_report, 'gyro_consistency': gyro_report,
              'cpu': {'normal_ground_flag0': cpu_stats(cpu['trajectory']),
                      'frozen_baseline_repeat_max_difference': repeat_error,
                      'normal_ground_flag1_input_override': cpu_stats(traces),
                      'natural_ground_start_seed42': {
                          **cpu_stats(ground_trace), 'seed': 42,
                          'first_active_normal_input_s': next((i*sim.DT*sim.DECIMATION
                              for i, value in enumerate(ground_obs) if value[28] == 1 and value[36] == 1), None),
                          'final_ground_start_flag': float(ground_obs[-1, 35])}},
              'counterfactual_scope': 'Substituted network inputs can be inconsistent physical states. Frozen recurrence does not evolve physics. CPU flag1 starts at the nominal pose, not the real prone pose.',
              'limitations': ['Tether onset remains unknown; no recording interval is certified free of tension.',
                              'Wheel rotation is not an independent real body velocity or displacement measurement.',
                              'Small attitude sensitivity does not establish an IMU zero error or authorize calibration changes.',
                              'No physical parameter fit, training, hardware commands or firmware modifications.'],
              'sources': [identity(v) for v in (args.capture_root/'session/meta.json',
                           args.capture_root/'chronology/frames.npz', args.capture_root/'details-final/policy-all.npz',
                           config, sim.MODEL_ASSET, sim.POSE_BANK, args.cpu_report,
                           args.cpu_snapshot/'manifest.json', *source.iterdir())]}
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for label, trajectory in (('CPU ground flag 0', cpu['trajectory']), ('CPU input ground flag 1', traces),
                               ('CPU full ground start, seed 42', ground_trace)):
        axes[0].plot([r['time_s'] for r in trajectory], [r['x_m'] for r in trajectory], label=label)
    axes[0].set_ylabel('CPU world x [m]')
    axes[0].set_xlabel('Simulation seconds; frozen baseline XML, no tether')
    for i, label in enumerate(labels):
        if abs(label['receive_s']-65) < .1 and label['group'] in ('actual', 'ground_flag_zero', 'level_gravity'):
            axes[1].plot(np.arange(len(recurrent)), recurrent[:, i],
                         '--' if label['initial_history'] == 'zero' else '-',
                         label=label['group']+'/'+label['initial_history'])
    axes[1].set_ylabel('Wheel target [rad/s]')
    axes[1].set_xlabel('Policy iterations with physical sensors frozen; not a motion prediction')
    axes[2].plot(rx[rows], forward_target(actual), label='recorded-input target')
    axes[2].plot(rx, .5*(f[:, 20]+f[:, 21]), label='measured wheel speed')
    axes[2].set_xlim(20.4, 40)
    axes[2].set_ylabel('Unified rolling direction [rad/s]')
    axes[2].set_xlabel('Recording receive seconds; rope-tension onset unknown')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out/'zero-command.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, trajectory in (('A: normal start, ground flag 0', cpu['trajectory']),
                              ('B: same start, ground flag input 1', traces),
                              ('C: full ground start, seed 42', ground_trace)):
        after = [r for r in trajectory if r['time_s'] >= 2]
        ax.plot([r['time_s'] for r in after], [r['x_m']-after[0]['x_m'] for r in after], label=label)
    ax.set_xlabel('Simulation time [s]; all commands vx=0, yaw=0, height=0.27 m')
    ax.set_ylabel('Forward displacement since t=2 s [m]')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out/'cpu-displacement.png', dpi=140)
    plt.close(fig)
    print(json.dumps({'policy_pairs': len(rows), 'gyro': gyro_report, 'cpu': report['cpu'],
                      'frozen_at_65s': [r for r in frozen_report if abs(r['receive_s']-65) < .1]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
