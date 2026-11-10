"""Read recorded RC/leg-test inputs and outputs; no port or physics access."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--decoded', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    meta, _, _ = mirror.load_session(a.session)
    report = json.loads((a.decoded / 'report.json').read_text())
    if report['stream_sha256'] != meta['stream_sha256']:
        raise ValueError('Decoded report does not belong to the input stream')
    parser = mirror.FrameParser(); rc = []
    for e in mirror.iter_session_events(a.session):
        if e['type'] == 'data':
            for frame in parser.feed(bytes.fromhex(e['data_hex'])):
                if int(frame[152]) == 6:
                    rc.append((mirror.read_u32(frame[8:10]) / 1000., float(frame[91])))
    rc = np.asarray(rc)
    geometry = []; control = []
    with (a.decoded / 'outputs.csv').open() as f:
        for row in csv.DictReader(f):
            if row['strategy'] != '6':
                continue
            if row['kind'] == 'leg_geometry' and int(row['leg_valid_flags']) & 8:
                geometry.append([int(row['begin_ns']) / 1e9] + [float(row[k]) for k in (
                    'left_leg_length_m', 'right_leg_length_m',
                    'left_leg_test_target_m', 'right_leg_test_target_m')])
            elif row['kind'] == 'control':
                control.append([int(row['begin_ns']) / 1e9] + [float(row[f'request_{i}_firmware_nm']) for i in range(6)])
    geometry = np.asarray(geometry); control = np.asarray(control)
    if len(rc) < 2 or len(geometry) < 2 or len(control) < 2:
        raise ValueError('Insufficient active test records')
    moving = np.flatnonzero(np.abs(rc[:, 1]) > 1e-6)
    if not len(moving):
        raise ValueError('No nonzero dial input')
    origin = rc[moving[0], 0]
    active = rc[moving[-1], 0] - origin
    # Sampled RC snapshots provide an approximate integral, not exact1kHz command reconstruction.
    integral = float(np.sum(np.minimum(rc[:-1, 1], 0) * np.diff(rc[:, 0])))
    final_error = (geometry[-1, 1:3] - geometry[-1, 3:5]).tolist()
    result = dict(input_stream_sha256=meta['stream_sha256'],
        whole_stream_complete=report['complete_observed_span'],
        crc_errors=report['crc_or_schema_rejections'],
        queue_drop_increase=report['queue_drop_increase'],
        sequence_gap_count=len(report['event_sequence_gaps']),
        rc_nonzero_span_s=float(active), rc_peak_speed_m_s=float(np.max(np.abs(rc[:, 1]))),
        sampled_negative_rc_integral_m=integral,
        initial_targets_m=geometry[0, 3:5].tolist(), final_targets_m=geometry[-1, 3:5].tolist(),
        target_change_m=(geometry[-1, 3:5] - geometry[0, 3:5]).tolist(),
        final_lengths_m=geometry[-1, 1:3].tolist(), final_error_m=final_error,
        max_abs_requested_torque_Nm=np.max(np.abs(control[:, 1:]), axis=0).tolist(),
        post_dial_target_span_m=np.ptp(geometry[geometry[:, 0] > rc[moving[-1], 0] + .2, 3:5], axis=0).tolist(),
        scope='Recorded controller response; received motor values are not independent geometry or shaft-torque measurements; no causal fit or physics')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(rc[:, 0] - origin, rc[:, 1] * 1000.)
    axes[0].set_ylabel('Dial target rate (mm/s)')
    for i, side in enumerate(('Left', 'Right')):
        axes[1].plot(geometry[:, 0] - origin, geometry[:, 1+i] * 1000., label=side + ' measured')
        axes[1].plot(geometry[:, 0] - origin, geometry[:, 3+i] * 1000., '--', label=side + ' target')
    axes[1].set_ylabel('Leg length (mm)'); axes[1].legend()
    for i in range(4):axes[2].plot(control[:, 0] - origin, control[:, 1+i], label='Motor ' + str(i))
    axes[2].set_ylabel('Requested torque (Nm)'); axes[2].legend(ncol=4)
    axes[2].set_xlabel('Board time from first nonzero dial sample (s)')
    for ax in axes:ax.set_xlim(-1., 10.); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.output / 'response.png', dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
