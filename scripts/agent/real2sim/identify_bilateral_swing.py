"""Audit JRL2 swing trials and identify a command-referenced reduced swing model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from scripts.agent.real2sim.analyze_tilt_length import geometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_frames(path):
    """records.bin has 16-byte Jetson record envelopes around serial chunks."""
    raw = path.read_bytes()
    position, stream = 0, bytearray()
    while position < len(raw):
        if position + 16 > len(raw):
            raise ValueError('Incomplete Jetson envelope')
        kind, size, _ = struct.unpack_from('<IIQ', raw, position)
        position += 16
        if size > 4096 or position + size > len(raw):
            raise ValueError('Incomplete or invalid Jetson record')
        if kind == 1:
            stream.extend(raw[position:position + size])
        position += size
    frames, position = [], 0
    while position + 32 <= len(stream):
        magic, version, kind, size, _, trial, sequence, us = struct.unpack_from('<4sBBHQIIQ', stream, position)
        if magic != b'JRL2' or version != 2 or size > 348:
            raise ValueError('Invalid serial header')
        end = position + size + 36
        if end > len(stream):
            break
        packet = stream[position:end]
        if zlib.crc32(packet[:-4]) != struct.unpack_from('<I', packet, len(packet) - 4)[0]:
            raise ValueError('CRC mismatch')
        frames.append((kind, trial, sequence, us, bytes(packet[32:-4])))
        position = end
    return frames, len(stream) - position


def load_trials(directory):
    trials, audit = [], {}
    for side, prefix in [('left', '左'), ('right', '右')]:
        path, = directory.glob(prefix + '腿*.csv')
        binary = path.with_suffix('.bin')
        records, tail = raw_frames(binary)
        a = np.genfromtxt(path, delimiter=',', names=True)
        decoded = []
        for kind, trial, sequence, us, payload in records:
            if kind != 13:
                continue
            if len(payload) != 100:
                raise ValueError('Unknown swing frame layout')
            flags, drops, config, fault, front_age, rear_age, *values = struct.unpack('<6I19f', payload)
            decoded.append([us, sequence, trial, flags, drops, config, fault, front_age, rear_age, *values])
        expected = np.column_stack([a[name] for name in a.dtype.names])
        if not np.array_equal(expected, np.asarray(decoded)):
            raise ValueError('CSV is not an exact export of the CRC-checked raw frames')
        audit[side] = dict(csv_sha256=sha(path), bin_sha256=sha(binary), rows=len(a),
                           frame_types=dict(Counter(r[0] for r in records)), trailing_serial_bytes=tail,
                           csv_matches_binary=True, crc_errors=0)
        for trial in np.unique(a['trial']).astype(int):
            trial = int(trial)
            b = a[a['trial'] == trial]
            flags = b['flags'].astype(int)
            t = (b['h7_us'] - b['h7_us'][0]) * 1e-6
            q = np.column_stack([b['angle_rad_front'], b['angle_rad_rear']])
            v = np.column_stack([b['velocity_rad_s_front'], b['velocity_rad_s_rear']])
            tau = np.column_stack([b['submitted_nm_front'], b['submitted_nm_rear']])
            length, angle = geometry(q, .21, .25)
            theta = np.unwrap(b['swing_rad'])
            if np.max(abs(np.arctan2(np.sin(theta - angle), np.cos(theta - angle)))) > 1e-4:
                raise ValueError('Swing geometry does not match the existing H7 convention')
            if np.max(abs(length - b['length_m'])) > 2e-6:
                raise ValueError('Leg length does not match existing geometry')
            config = b['trial_config'].astype(int)
            if not (np.all((config & 255) == (1 if side == 'left' else 2))
                    and np.all((config >> 8 & 255) == trial) and np.all((config >> 16 & 255) == 1)):
                raise ValueError('Unexpected side/level/repetition')
            if (np.any(np.diff(b['sequence']) != 1) or np.any(np.diff(t) <= 0)
                    or np.any(b['fault']) or np.any(np.diff(b['board_drops']))
                    or np.any(flags & 8) or not np.all(flags & 7 == 7) or not np.any(flags & 64)):
                raise ValueError('Faulted or incomplete trial')
            active = (flags & 32) != 0
            steady = active & (b['swing_ref_rad_s'] >= trial - .001)
            on, off = np.flatnonzero(active)[[0, -1]]
            row = dict(side=side, level=trial, rows=len(b), duration_s=float(t[-1]),
                       dt_us_percentiles=np.percentile(np.diff(b['h7_us']), [0, 50, 100]).tolist(),
                       sequence_gaps=0, faults=0, board_drop_increase=0,
                       feedback_ages_all_zero=bool(not np.any(b['front_age_us']) and not np.any(b['rear_age_us'])),
                       active_start_s=float(t[on]), release_s=float(t[off + 1]),
                       steady_mean_speed_rad_s=float(b['swing_rate_rad_s'][steady].mean()),
                       steady_mae_rad_s=float(np.mean(abs(b['swing_rate_rad_s'][steady] - trial))),
                       length_mm_percentiles=(np.percentile(length, [0, 50, 100]) * 1000).tolist(),
                       geometry_max_length_error_mm=float(np.max(abs(length - b['length_m'])) * 1000),
                       geometry_max_angle_error_rad=float(np.max(abs(np.arctan2(np.sin(theta-angle), np.cos(theta-angle))))),
                       swing_command_identity_max_nm=float(np.max(abs(-tau.sum(1) - b['swing_nm']))),
                       front_rear_velocity_correlation=float(np.corrcoef(v.T)[0, 1]),
                       steady_mean_command_nm=float(np.mean(-tau[steady].sum(1))),
                       active_net_turns=float((theta[off] - theta[on]) / (2*np.pi)),
                       release_coast_angle_rad=float(theta[-1] - theta[off+1]))
            # 位置在运动方向连续；51ms局部多项式导数减少协议速度量化带来的惯量低估。
            fitted_speed = savgol_filter(-q.mean(1), 51, 3, deriv=1, delta=float(np.median(np.diff(t))))
            trials.append(dict(side=side, level=trial, t=t, q=q, v=v, tau=tau, theta=theta,
                               speed=fitted_speed, reported_speed=b['swing_rate_rad_s'], length=length, active=active,
                               steady=steady, reference=b['swing_ref_rad_s'], row=row))
    return trials, audit


def features(trial, harmonics):
    v, theta = trial['speed'], trial['theta']
    columns = [v, np.tanh(v / .1)]
    for order in range(1, harmonics + 1):
        columns.extend([np.sin(order * theta), np.cos(order * theta)])
    return np.column_stack(columns)


def equations(trial, harmonics, window):
    t = trial['t']
    # 积分形式免去直接对量化速度求加速度；每档等权，低速长记录不支配结果。
    x = features(trial, harmonics)
    integrals = cumulative_trapezoid(x, t, axis=0, initial=0)
    u = cumulative_trapezoid(-trial['tau'].sum(1), t, initial=0)
    stride = max(1, int(round(window / .001)))
    start = np.arange(0, len(t) - stride, max(1, stride // 4))
    end = start + stride
    moving = np.minimum(trial['speed'][start], trial['speed'][end]) > .2
    start, end = start[moving], end[moving]
    matrix = np.column_stack([trial['speed'][end] - trial['speed'][start], integrals[end] - integrals[start]])
    return matrix / window, (u[end] - u[start]) / window


def fit(trials, harmonics, window=.2):
    matrices, targets = [], []
    for tr in trials:
        x, y = equations(tr, harmonics, window)
        weight = 1 / np.sqrt(len(y))
        matrices.append(x * weight)
        targets.append(y * weight)
    x, y = np.vstack(matrices), np.concatenate(targets)
    scale = np.linalg.norm(x, axis=0)
    lower = [1e-5, 0, 0] + [-np.inf] * (2 * harmonics)
    upper = [10, 100, 100] + [np.inf] * (2 * harmonics)
    result = lsq_linear(x/scale, y, bounds=(np.asarray(lower)*scale, np.asarray(upper)*scale), tol=1e-12)
    if not result.success:
        raise ValueError(result.message)
    p = result.x / scale
    sv = np.linalg.svd(x/scale, compute_uv=False)
    return dict(coefficients=p.tolist(), harmonics=harmonics, integral_window_s=window,
                normalized_condition_number=float(sv[0]/sv[-1]), rank=int(np.linalg.matrix_rank(x/scale)),
                effective_inertia_kg_m2=float(p[0]), effective_damping_nm_s_rad=float(p[1]),
                positive_direction_offset_nm=float(p[2]),
                periodic_load_nm=p[3:].tolist(), active_bounds=result.active_mask.tolist())


def residual(trial, parameters):
    x, y = equations(trial, parameters['harmonics'], parameters['integral_window_s'])
    return float(np.sqrt(np.mean((x @ parameters['coefficients'] - y)**2)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trials, audit = load_trials(args.input)
    result = dict(input_audit=audit, trials=[tr['row'] for tr in trials], fitted={},
                  fitting_levels=[1,3,5], held_out_levels=[2,4],
                  scope='Command-referenced, positive-direction, near-constant-length effective swing model; not individual motor or spring identification',
                  user_fixture='side lying; nominal swing axis vertical',
                  limitations=['No calibrated shaft torque or torque gain', 'No reverse-direction trial',
                               'Only one start per side/speed', 'Feedback age fields all zero',
                               'No firmware hash or measured fixture tilt', 'No independent spring/axial excitation'],
                  sources={str(Path(__file__)):sha(Path(__file__))},
                  full_robot_acceptance=False, hardware_access=False, training_updates=0)
    for side in ('left', 'right'):
        training = [tr for tr in trials if tr['side'] == side and tr['level'] in (1,3,5)]
        selection = []
        for harmonic in (0,1,2):
            errors = []
            for omitted in (1,3,5):
                p = fit([tr for tr in training if tr['level'] != omitted], harmonic)
                errors.append(residual(next(tr for tr in training if tr['level'] == omitted), p))
            selection.append(dict(harmonics=harmonic, leave_one_speed_out_rmse_nm=errors,
                                  mean_rmse_nm=float(np.mean(errors))))
        chosen = min(selection, key=lambda r:r['mean_rmse_nm'])['harmonics']
        p = fit(training, chosen)
        p['model_selection_using_fitting_speeds_only'] = selection
        p['window_sensitivity'] = [fit(training, chosen, w) for w in (.1,.3)]
        p['raw_reported_velocity_sensitivity'] = fit([dict(tr, speed=tr['reported_speed']) for tr in training], chosen)
        p['per_level_integral_torque_rmse_nm'] = {tr['level']:residual(tr,p) for tr in trials if tr['side']==side}
        result['fitted'][side] = p
    (args.output/'identification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(fitted=result['fitted'],audit=audit), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
