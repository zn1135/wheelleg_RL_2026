"""Read a frozen prefix of a USB journal and compare leg swing with extension; no physics."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
from h7_output_capture import CaptureParser


def geometry(q, lu, lg):
    # 与板端一致：前髋加pi，取同一侧圆交点，不重设零点。
    angle = q + [np.pi, 0.]
    ab = lu * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    delta = ab[:, 1] - ab[:, 0]
    distance = np.linalg.norm(delta, axis=1)
    height = np.sqrt(lg * lg - distance * distance / 4)
    normal = np.column_stack((-delta[:, 1], delta[:, 0])) / distance[:, None]
    point = ab.mean(axis=1) + height[:, None] * normal
    return np.linalg.norm(point, axis=1), np.unwrap(np.pi / 2 - np.arctan2(point[:, 1], point[:, 0]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.session / 'meta.json').read_text())
    config = ROOT / meta['machine_config']['project_path']
    assert hashlib.sha256(config.read_bytes()).hexdigest() == meta['machine_config']['sha256']
    import re
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1].split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0]
    lu, lg = [float(re.search(r'\.' + name + r'\s*=\s*([\d.]+)f', block)[1]) for name in ('leg_lu', 'leg_lg')]
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.session / 'stream.jsonl'
    bound = source.stat().st_size
    decoder = CaptureParser()
    digest = hashlib.sha256()
    states, controls, samples = {}, {}, []
    consumed = 0
    with source.open('rb') as file:
        while consumed < bound:
            raw = file.readline(bound - consumed)
            if not raw.endswith(b'\n'):
                break
            consumed += len(raw)
            digest.update(raw)
            event = json.loads(raw)
            if event['type'] in ('connected', 'disconnected'):
                decoder.buffer.clear()
                states.clear()
                controls.clear()
            if event['type'] != 'data':
                continue
            for packet in decoder.feed(bytes.fromhex(event['data_hex'])):
                for row in packet['records']:
                    key = row[4]
                    if row[1] == 1:
                        controls[key] = row
                    elif row[1] == 3:
                        states[key] = row
                    elif row[1] == 4 and key in states and key in controls:
                        state, control = states.pop(key), controls.pop(key)
                        if state[2:4] != row[2:4] or row[20] & 7 != 7:
                            continue
                        ages = [*state[14:16], *struct.unpack('<2I', struct.pack('<4h', *state[16:20]))]
                        if max(ages) > 5000 or row[5]:
                            continue
                        right_angle = struct.unpack('<f', struct.pack('<I', row[14]))[0]
                        samples.append([row[3] / 1e9, *state[6:10], *row[11:13], row[13], right_angle,
                                        row[21], *control[6:12], row[20]])
                    if len(states) > 128:
                        states.pop(next(iter(states)))
                    if len(controls) > 128:
                        controls.pop(next(iter(controls)))
    a = np.asarray(samples)
    assert len(a), 'No fresh paired leg records'
    # 只汇报同一开机后的本次流；全程和最近一分钟分别列出，避免混淆不同动作。
    t = a[:, 0] - a[0, 0]
    passive = (a[:, 9] == 2) & (a[:, 16].astype(int) & 8 == 0) & np.all(a[:, 10:16] == 0, axis=1)
    report = dict(session=str(args.session), source_prefix_bytes=consumed,
                  source_prefix_sha256=digest.hexdigest(), paired_fresh_samples=len(a),
                  config=meta['machine_config'], rod_lengths_m=[lu, lg],
                  parser_rejections=decoder.rejected, physics_steps=0,
                  scope='Kinematics diagnostic of a frozen live-journal prefix; no independent physical distance measurement',
                  maximum_paired_time_gap_s=float(np.max(np.diff(t))) if len(t) > 1 else 0.,
                  max_abs_requested_torque_nm=np.max(abs(a[:, 10:16]), axis=0).tolist(),
                  windows={})
    traces = {}
    for name, mask in [('all', np.ones(len(a), dtype=bool)), ('last60s', t >= t[-1] - 60),
                       ('disabled_zero_request', passive)]:
        if not mask.any():
            report['windows'][name] = {'samples': 0}
            continue
        window = dict(samples=int(mask.sum()), time_range_s=[float(t[mask][0]), float(t[mask][-1])],
                      leg_mode_samples=int(np.sum(a[mask, 9] == 6)))
        for side, label in enumerate(('left', 'right')):
            q = a[:, 1 + side * 2:3 + side * 2]
            length, angle = geometry(q, lu, lg)
            half = (q[:, 0] + np.pi - q[:, 1]) / 2
            # 直接只用前后夹角算长度，独立核对二维交点实现。
            difference_length = abs(lu * np.cos(half) + np.sign(np.sin(half))
                                    * np.sqrt(lg * lg - lu * lu * np.sin(half) ** 2))
            diff = np.unwrap(q[:, 0] - q[:, 1])
            board = a[:, 5 + side]
            if name == 'all':
                traces[label + '_length_m'] = board
                traces[label + '_angle_rad'] = angle
                traces[label + '_motor_difference_rad'] = diff
            # 固定两关节夹角，仅保留共同转动，检验数学公式是否把摆动混入长度。
            common = q[mask].mean(axis=1)
            opening = diff[mask][0]
            constant_opening = np.column_stack((common + opening / 2, common - opening / 2))
            fixed_length, _ = geometry(constant_opening, lu, lg)
            error = (board - length)[mask] * 1000
            window[label] = dict(length_range_mm=(np.array([board[mask].min(), board[mask].max()]) * 1000).tolist(),
                                 length_span_mm=float(np.ptp(board[mask]) * 1000),
                                 angle_span_deg=float(np.rad2deg(np.ptp(angle[mask]))),
                                 motor_difference_span_deg=float(np.rad2deg(np.ptp(diff[mask]))),
                                 front_rear_spans_deg=np.rad2deg(np.ptp(q[mask], axis=0)).tolist(),
                                 board_recomputed_error_p95_mm=float(np.percentile(abs(error), 95)),
                                 board_recomputed_error_max_mm=float(np.max(abs(error))),
                                 difference_only_formula_error_max_mm=float(np.max(abs(difference_length[mask] - length[mask])) * 1000),
                                 constant_opening_length_span_mm=float(np.ptp(fixed_length) * 1000))
        report['windows'][name] = window
    np.savez_compressed(args.output / 'trace.npz', time_s=t, paired_records=a, passive=passive, **traces)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label in ('left', 'right'):
        axes[0].plot(t, traces[label + '_length_m'] * 1000, label=label)
        axes[1].plot(t, np.rad2deg(traces[label + '_angle_rad']), label=label)
        d = traces[label + '_motor_difference_rad']
        axes[2].plot(t, np.rad2deg(d - d[0]), label=label)
    for ax, title in zip(axes, ('Board leg length (mm)', 'Leg angle, unwrapped (deg)', 'Motor angle difference change (deg)')):
        ax.set_ylabel(title)
        ax.grid(alpha=.25)
        ax.legend()
    axes[2].set_xlabel('Time since first paired sample (s)')
    fig.tight_layout()
    fig.savefig(args.output / 'tilt-length.png', dpi=140)
    report['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
