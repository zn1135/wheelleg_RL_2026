#!/usr/bin/env python3
"""Compare saved 6848 real telemetry and measured-start simulation; no dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MOTORS = ('lf0', 'lf00', 'rf0', 'rf00')


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': values.mean(axis=0).tolist(),
            'p95_minus_p05': (np.percentile(values, 95, axis=0)
                              - np.percentile(values, 5, axis=0)).tolist(),
            'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist()}


def pitch_degrees(quat):
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = quat.T
    return np.rad2deg(np.arcsin(np.clip(2 * (w * y - z * x), -1., 1.)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory')
    base = args.capture_root
    saved = np.load(base / 'chronology/frames.npz')
    f, ticks = saved['frames'], saved['board_tick_ms']
    prior_path = base / 'measured-start-dynamics-final/report.json'
    prior = json.loads(prior_path.read_text())
    for name, expected in prior['sources'].items():
        if name.endswith('chronology/frames.npz'):
            assert hashlib.sha256((base / 'chronology/frames.npz').read_bytes()).hexdigest() == expected
    manifest_path = base / 'frozen-baseline/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        source = base / 'frozen-baseline' / item['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    start = prior['first_nonzero_output_tick_ms']
    sim_path = base / 'measured-start-dynamics-final/original-limits.npz'
    sim = np.load(sim_path)
    model_path = base / 'frozen-baseline/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qadr = np.array([int(model.joint(n).qposadr[0]) for n in MOTORS])
    vadr = [int(model.joint(n).dofadr[0]) for n in ('lfwheel', 'rfwheel')]
    real_time = (ticks - start) / 1000.
    active = ((f[:, 152] == 0) & ((f[:, 109].astype(int) & 33) == 33))
    mask = active & (real_time >= 0) & (real_time <= 30.)
    real = {'time': real_time[mask], 'q': f[mask, 122:126],
            'target': f[mask, 110:114], 'pitch': pitch_degrees(f[mask, 22:26]),
            'wheel': .5 * (f[mask, 130] - f[mask, 131]),
            'wheel_target': .5 * (f[mask, 114] - f[mask, 115]),
            'phase': f[mask, 154]}
    nominal = (sim['qpos'][0, qadr] - sim['obs'][0, 10:14]).astype(np.float32)
    target = np.clip(nominal + .5 * sim['actions'][:, [0, 1, 3, 4]],
                     model.qpos0[qadr] - 3.14, model.qpos0[qadr] + 3.14)
    simulated = {'time': sim['time_s'][:-1], 'q': sim['qpos'][:-1, qadr],
                 'target': target, 'pitch': pitch_degrees(sim['qpos'][:-1, 3:7]),
                 'wheel': .5 * (sim['qvel'][:-1, vadr[0]] - sim['qvel'][:-1, vadr[1]]),
                 'wheel_target': .5 * (np.clip(10 * sim['actions'][:, 2], -150, 150)
                                      - np.clip(10 * sim['actions'][:, 5], -150, 150))}
    # 初始策略观测和物理角一致；仿真目标中心由这组已核对观测还原。
    initial_nominal = sim['qpos'][0, qadr] - sim['obs'][0, 10:14]
    np.testing.assert_allclose(initial_nominal, nominal, atol=1e-7)
    assert np.max(np.abs(sim['obs'][:, [0, 1, 2]])) == 0
    assert np.max(np.abs(sim['obs'][:, 3] / 5 - .27)) < 1e-6
    assert np.max(np.abs(f[mask, 57:60])) == 0
    assert np.max(np.abs(f[mask, 60] / 5 - .27)) < 1e-6

    periods = []
    for lo, hi in ((0., .5), (.5, 1.), (1., 2.), (2., 4.), (4., 6.), (6., 10.), (10., 20.)):
        period = {'seconds_after_first_nonzero_output': [lo, hi],
                  'rope_status': 'Unknown; cannot certify untethered interval'}
        for name, trace in [('real', real), ('simulation', simulated)]:
            selected = (trace['time'] >= lo) & (trace['time'] < hi)
            period[name] = {'samples': int(selected.sum()),
                            'leg_q_deg': stats(np.rad2deg(trace['q'][selected])),
                            'leg_target_deg': stats(np.rad2deg(trace['target'][selected])),
                            'leg_tracking_error_deg': stats(np.rad2deg(trace['target'][selected]
                                                                     - trace['q'][selected])),
                            'pitch_deg': stats(trace['pitch'][selected]),
                            'wheel_common_rad_s': stats(trace['wheel'][selected]),
                            'wheel_target_common_rad_s': stats(trace['wheel_target'][selected])}
        periods.append(period)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / 'aligned-signals.npz',
                        **{'real_' + k: v for k, v in real.items()},
                        **{'sim_' + k: v for k, v in simulated.items()})
    for label, bounds in [('startup', (0., 1.5)), ('post-start', (1., 6.))]:
        fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
        for j, motor in enumerate(MOTORS):
            for name, trace, color in [('Real', real, 'tab:orange'),
                                       ('Simulation', simulated, 'tab:blue')]:
                visible = (trace['time'] >= bounds[0]) & (trace['time'] <= bounds[1])
                axes[j].plot(trace['time'][visible], np.rad2deg(trace['q'][visible, j]), color=color,
                             lw=1., label=name + ' actual')
                axes[j].plot(trace['time'][visible], np.rad2deg(trace['target'][visible, j]), color=color,
                             lw=.8, ls='--', alpha=.6, label=name + ' target')
            axes[j].set_ylabel(motor + ' [deg]')
        for name, trace, color in [('Real', real, 'tab:orange'), ('Simulation', simulated, 'tab:blue')]:
            visible = (trace['time'] >= bounds[0]) & (trace['time'] <= bounds[1])
            axes[4].plot(trace['time'][visible], trace['pitch'][visible], color=color, label=name)
            axes[5].plot(trace['time'][visible], trace['wheel'][visible], color=color, label=name + ' measured')
            axes[5].plot(trace['time'][visible], trace['wheel_target'][visible], color=color, ls='--',
                         alpha=.6, label=name + ' target')
        axes[4].set_ylabel('Pitch [deg]')
        axes[5].set_ylabel('Common wheel [rad/s]')
        axes[-1].set_xlabel('H7-aligned seconds after first nonzero request; rope onset unknown')
        for ax in axes:
            ax.set_xlim(*bounds)
            ax.grid(alpha=.25)
            ax.legend(loc='upper right', fontsize=8, ncol=4)
        fig.suptitle('6848: saved real telemetry vs measured-start simulation\n'
                     'Same initial measured pose / height command; later posture differs; rope onset unknown')
        fig.tight_layout()
        fig.savefig(args.out / (label + '.png'), dpi=140)
        plt.close(fig)
    paths = [Path(__file__), base / 'chronology/frames.npz', prior_path, sim_path, manifest_path,
             model_path, base / 'session/meta.json', base / 'details-final/report.json']
    sources = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {'scope': 'Offline saved-signal comparison, no simulation stepping, training or hardware operation',
              'model': '6848, same checkpoint verified by source reports',
              'checkpoint_sha256': '4a96a4c216fed4cedc2153667defbc0ae7974b9a31eb857022a31b4658bb3001',
              'first_nonzero_board_ms': start, 'motor_order': MOTORS,
              'amplitude_definition': '95th percentile minus 5th percentile in each window; not peak-to-peak or geometric leg swing',
              'real_sampling': 'approx 100Hz H7 snapshots; no claim of control-rate phase accuracy',
              'sim_sampling': '50Hz saved states/actions',
              'limitations': ['Rope tension onset is unknown, so no steady window is certified untethered.',
                              'Matched initial measured pose; unmeasured root height/linear velocity were assumed in prior simulation.',
                              'Same height command does not imply same achieved height; later mean motor angles differ.',
                              'Closed-loop target/feedback correlations cannot establish which variable caused oscillation.',
                              'Initial saturated recovery response must not be treated as zero-speed steady oscillation.'],
              'periods': periods, 'sources': sources}
    shutil.copy2(__file__, args.out / Path(__file__).name)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'out': str(args.out), 'periods': periods[3:5]}), flush=True)


if __name__ == '__main__':
    main()
