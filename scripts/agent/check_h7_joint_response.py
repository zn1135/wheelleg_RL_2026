#!/usr/bin/env python3
"""Offline four-hip response screening with incomplete 100Hz input snapshots.

No fitting, hardware access, policy inference or training. This cannot accept a
physical calibration: held support and unsampled control inputs are unknown.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import re
import shutil
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_dynamics_huananhu as dyn
from h7_output_capture import CaptureParser

JOINTS = dyn.mirror.MOTORS[:4]
HORIZONS = (10, 50, 100)
CASES = ('lower_contact', 'zero_translation', 'floor_gap_2mm', 'next_snapshot')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(config):
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1]
    block = block.split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0].split('.rl = {', 1)[1]
    result = {}
    for name in ('nominal_pos', 'reference_pos', 'p_gains', 'd_gains'):
        text = re.search(r'\.' + name + r'\s*=\s*\{([^}]+)\}', block).group(1)
        result[name] = np.asarray([float(x.strip().rstrip('fF')) for x in text.split(',') if x.strip()],
                                  dtype=np.float32)
    for name in ('pos_scale', 'position_limit', 'leg_torque_limit'):
        text = re.search(r'\.' + name + r'\s*=\s*([^,]+)', block).group(1)
        result[name] = np.float32(dyn.numeric_expression(text))
    return result


def rollout(predictor, q, v, inputs):
    # Only this initial state is measured; no future states enter the loop.
    m, d, env = predictor.m, predictor.d, predictor.env
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_forward(m, d)
    qs, vs = [q.copy()], [v.copy()]
    for torque in inputs:
        d.ctrl[env.motor_ids] = torque[dyn.I2N]
        d.ctrl[env.spring_ids] = env._gas_spring_force(d.qpos[env.spring_q], d.qvel[env.spring_v])
        mujoco.mj_step(m, d)
        qs.append(d.qpos.copy())
        vs.append(d.qvel.copy())
    if max(d.warning.number) or not np.isfinite(qs).all() or not np.isfinite(vs).all():
        raise RuntimeError('MuJoCo warning or nonfinite prediction')
    return np.asarray(qs), np.asarray(vs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, default=ROOT / 'artifacts/real2sim-latest-gap-20261021/real2sim-20261021-103443')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--windows', type=int, default=64)
    args = p.parse_args()
    if args.windows < 1 or args.out.exists():
        p.error('positive --windows and a new --out directory required')
    args.out.mkdir(parents=True)
    s = dyn.Session(args.session)
    recorded = dyn.RecordedRL(s)
    capture = CaptureParser()
    packet_count = 0
    for event in dyn.mirror.iter_session_events(args.session):
        if event['type'] == 'data':
            packet_count += len(capture.feed(bytes.fromhex(event['data_hex'])))
        else:
            capture.buffer.clear()
    if packet_count:
        raise ValueError('Complete-output data found: use full-rate inputs instead of this screening path')
    cfg = parameters(s.config)
    normal = s.valid & ((s.f[:, 109].astype(int) & 63) == 61) & (s.f[:, 154] == 3)
    normal &= np.isfinite(s.f[:, 110:138]).all(axis=1)
    f = s.f[normal]
    targets = np.clip(cfg['nominal_pos'] + cfg['pos_scale'] * f[:, [116, 117, 119, 120]],
                      cfg['reference_pos'] - cfg['position_limit'],
                      cfg['reference_pos'] + cfg['position_limit'])
    expected = np.clip(cfg['p_gains'] * (f[:, 110:114] - f[:, 122:126])
                       - cfg['d_gains'] * f[:, 126:130],
                       -cfg['leg_torque_limit'], cfg['leg_torque_limit'])
    actual = s.torque[normal, :4]
    target_error = float(np.max(abs(targets - f[:, 110:114])))
    torque_error = float(np.max(abs(expected - actual)))
    if target_error > 1e-5 or torque_error > 1e-4:
        raise ValueError(f'Control snapshot mismatch: target={target_error}, torque={torque_error}')
    audit = {'normal_snapshots': len(f), 'target_max_difference_rad': target_error,
             'torque_max_difference_nm': torque_error, 'joints': {}}
    for k, name in enumerate(JOINTS):
        audit['joints'][name] = {
            'q_range_rad': [float(f[:, 122+k].min()), float(f[:, 122+k].max())],
            'velocity_range_rad_s': [float(f[:, 126+k].min()), float(f[:, 126+k].max())],
            'request_range_nm': [float(actual[:, k].min()), float(actual[:, k].max())],
            'driver_estimate_minus_request_nm': dyn.stats(f[:, 146+k] * s.sign[k] - actual[:, k]),
            'snapshot_request_saturation_count': int(np.sum(abs(actual[:, k]) >= cfg['leg_torque_limit'])),
        }
    predictor = dyn.BodyPredictor(dyn.sim.MODEL_ASSET, s.config)
    assert predictor.m.opt.timestep == .002
    possible, selection_rejected = [], Counter()
    for i in range(len(s.f)-10):
        # Select by recorded validity/time only, before seeing any model error.
        if not 5 <= s.rx[i] <= 25 or not s.continuous(i, i+10):
            continue
        if not normal[i:i+11].all() or not np.all(np.diff(s.output_ms[i:i+11]) == 10):
            continue
        if not possible or s.output_ms[i] - s.output_ms[possible[-1]] >= 200:
            possible.append(i)
    chosen = np.unique(np.round(np.linspace(0, len(possible)-1, min(args.windows, len(possible)))).astype(int))
    selected = [possible[k] for k in chosen]
    if not selected:
        raise ValueError('No eligible nonoverlapping 100ms windows')
    rows, traces, initialization = [], [], []
    repeat_error = None
    for i in selected:
        try:
            q0, v0, meta = predictor.measured_state(recorded, i)
        except ValueError as exc:
            selection_rejected[str(exc)] += 1
            initialization.append({'row': i, 'rejected': str(exc)})
            continue
        t0 = s.output_ms[i]
        times = t0 + np.arange(50)*2
        previous = np.searchsorted(s.output_ms, times, side='right')-1
        following = np.searchsorted(s.output_ms, times, side='left')
        for case in CASES:
            q, v = q0.copy(), v0.copy()
            q[2] += .002 if case == 'floor_gap_2mm' else -1e-6
            if case != 'zero_translation':
                v[:3] = meta['lower_velocity']
            # next_snapshot is deliberately noncausal edge sensitivity, never a
            # measured delay or a scored independent validation.
            ix = following if case == 'next_snapshot' else previous
            inputs = s.torque[ix].copy()
            pq, pv = rollout(predictor, q, v, inputs)
            if repeat_error is None:
                rq, rv = rollout(predictor, q, v, inputs)
                repeat_error = float(max(np.max(abs(pq-rq)), np.max(abs(pv-rv))))
                assert repeat_error == 0
            for h in HORIZONS:
                j, step = i+h//10, h//2
                for k, name in enumerate(JOINTS):
                    a = dyn.N2I[k]
                    predicted_q = pq[step, predictor.env.q[a]]
                    predicted_v = pv[step, predictor.env.v[a]]
                    actual_q, actual_v = s.f[j, 122+k], s.f[j, 126+k]
                    rows.append({'row': i, 'case': case, 'horizon_ms': h, 'joint': name,
                                 'q_error_deg': float(np.degrees(predicted_q-actual_q)),
                                 'v_error_rad_s': float(predicted_v-actual_v),
                                 'constant_velocity_q_error_deg': float(np.degrees(
                                     s.f[i, 122+k]+h/1000*s.f[i, 126+k]-actual_q)),
                                 'constant_velocity_v_error_rad_s': float(s.f[i, 126+k]-actual_v)})
            if case == 'lower_contact':
                for j in range(11):
                    for k, name in enumerate(JOINTS):
                        a = dyn.N2I[k]
                        traces.append({'row': i, 'time_ms': j*10, 'joint': name,
                                       'predicted_q_rad': float(pq[j*5, predictor.env.q[a]]),
                                       'actual_q_rad': float(s.f[i+j, 122+k]),
                                       'predicted_v_rad_s': float(pv[j*5, predictor.env.v[a]]),
                                       'actual_v_rad_s': float(s.f[i+j, 126+k])})
        initialization.append({'row': i, 'host_s': float(s.rx[i]), **meta})
    if not rows:
        raise ValueError('All selected initial states rejected')
    summaries = {}
    for case in CASES:
        summaries[case] = {}
        for h in HORIZONS:
            summaries[case][str(h)] = {}
            for name in JOINTS:
                group = [r for r in rows if r['case'] == case and r['horizon_ms'] == h and r['joint'] == name]
                summaries[case][str(h)][name] = {key: dyn.stats([r[key] for r in group]) for key in
                    ('q_error_deg', 'v_error_rad_s', 'constant_velocity_q_error_deg', 'constant_velocity_v_error_rad_s')}
    for name, data in (('predictions.csv', rows), ('traces.csv', traces)):
        with (args.out/name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    report = {'screening_completed': True, 'calibration_accepted': False,
              'session': s.summary, 'model': dyn.identity(dyn.sim.MODEL_ASSET),
              'recorded_model': s.meta['model'], 'machine_config': s.meta['machine_config'],
              'complete_output_packets_in_RL': packet_count, 'control_audit': audit,
              'physics_dt_s': .002, 'selected_rows': selected, 'initialization': initialization,
              'rejected_initialization': dict(selection_rejected),
              'evaluated_windows': len(rows)//len(CASES)//len(HORIZONS)//len(JOINTS),
              'repeat_max_state_difference': repeat_error, 'summaries': summaries,
              'limitations': ['100Hz gated request snapshots omit intervening 500Hz controls and DM CAN bytes',
                  'all known complete-output wheel bench sessions have zero hip requests',
                  'human support force, world translation and contact state unmeasured',
                  'driver torque estimate is not independently measured shaft torque',
                  'one reused whole RL session; no fitting or independent validation split',
                  'next_snapshot uses future requests solely as an input-edge sensitivity test',
                  'initial angular/passive states reconstructed with existing coordinate mapping',
                  'firmware boot/version and motor internal settings not fully attested by old session'],
              'no_hardware_or_training_or_model_changes': True}
    save(args.out/'report.json', report)
    paths = [Path(__file__), Path(dyn.__file__), Path(dyn.mirror.__file__), Path(dyn.sim.__file__),
             ROOT/'sim2sim/wheel_response.py', ROOT/'sim2sim/h7_drive_response.c', s.config,
             dyn.sim.MODEL_ASSET, args.session/'meta.json']
    source = args.out/'source'; source.mkdir()
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = source/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = digest(path)
    save(args.out/'manifest.json', {'source_sha256': hashes,
         'mujoco': mujoco.__version__, 'numpy': np.__version__, 'argv': sys.argv})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    first = traces[0]['row']
    for k, name in enumerate(JOINTS):
        trace = [r for r in traces if r['row'] == first and r['joint'] == name]
        for column, variable, unit in ((0, 'q', 'rad'), (1, 'v', 'rad/s')):
            ax = axes[k, column]
            key = 'q_rad' if variable == 'q' else 'v_rad_s'
            ax.plot([r['time_ms'] for r in trace], [r['actual_'+key] for r in trace], 'k', label='measured')
            ax.plot([r['time_ms'] for r in trace], [r['predicted_'+key] for r in trace], label='snapshot replay')
            ax.set_ylabel(f'{name} ({unit})'); ax.grid(alpha=.3)
    axes[0, 0].legend(); axes[0, 1].legend()
    axes[-1, 0].set_xlabel('ms from initialization'); axes[-1, 1].set_xlabel('ms from initialization')
    fig.suptitle('First selected window; free integration, incomplete inputs / unknown support')
    fig.tight_layout(); fig.savefig(args.out/'joint-response.png', dpi=140); plt.close(fig)
    print(json.dumps({'screening_completed': True, 'calibration_accepted': False,
                      'evaluated_windows': report['evaluated_windows'], 'control_audit': audit,
                      'report': str(args.out/'report.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
