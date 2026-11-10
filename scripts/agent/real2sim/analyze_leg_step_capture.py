"""Measure a recorded wheel-step gesture from a bounded live-journal slice; no hardware commands."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from h7_output_capture import CaptureParser


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--marker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    marker = json.loads(args.marker.read_text())
    path = args.session / 'stream.jsonl'
    bound = path.stat().st_size
    frames, outputs = mirror.FrameParser(), CaptureParser()
    dials, geometries, controls = [], [], []
    digest = hashlib.sha256()
    with path.open('rb') as source:
        start = marker['source_byte_begin']
        if start:
            source.seek(start - 1)
            if source.read(1) != b'\n':
                source.readline()
        begin = source.tell()
        while source.tell() < bound:
            position = source.tell()
            raw = source.readline(bound - source.tell())
            if not raw.endswith(b'\n'):
                source.seek(position)
                break
            digest.update(raw)
            event = json.loads(raw)
            if event['type'] != 'data':
                continue
            chunk = bytes.fromhex(event['data_hex'])
            for frame in frames.feed(chunk):
                t = mirror.read_u32(frame[8:10]) / 1000
                if t >= marker['before_board_s']:
                    dials.append([t, float(frame[91]), int(frame[152]), int(frame[2])])
            for packet in outputs.feed(chunk):
                for row in packet['records']:
                    t = row[3] / 1e9
                    if t < marker['before_board_s']:
                        continue
                    if row[1] == 4:
                        targets = struct.unpack('<2f', struct.pack('<4h', *row[16:20]))
                        geometries.append([t, *row[11:13], *targets, row[20], row[5]])
                    elif row[1] == 1:
                        controls.append([t, *row[6:12], row[5]])
        end = source.tell()
    dials, g, c = map(np.asarray, (dials, geometries, controls))
    assert len(dials) and len(g) and len(c)
    moving = np.flatnonzero(abs(dials[:, 1]) > 1e-7)
    assert len(moving), 'No observed dial gesture in this slice'
    groups = np.split(moving, np.flatnonzero((np.diff(moving) != 1) | (np.diff(dials[moving, 0]) > .03)) + 1)
    neutral_after = dials[(dials[:, 0] > dials[moving[-1], 0]) & (dials[:, 1] == 0)]
    before = np.array(marker['before_targets_m'])
    after = np.median(g[g[:, 0] >= g[-1, 0] - 1., 3:5], axis=0)
    before_actual = g[0, 1:3]
    after_actual = np.median(g[g[:, 0] >= g[-1, 0] - 1., 1:3], axis=0)
    delta = (after - before) * 1000
    report = dict(session=str(args.session), byte_begin=begin, byte_end=end, source_slice_sha256=digest.hexdigest(),
                  marker=marker, geometry_records=len(g), mirror_frames=len(dials),
                  gestures=[dict(start_board_s=float(dials[i[0], 0]), end_board_s=float(dials[i[-1], 0]),
                                 observed_duration_s=float(np.ptp(dials[i, 0])), direction='contract' if np.median(dials[i, 1]) < 0 else 'extend') for i in groups],
                  targets_before_m=before.tolist(), targets_after_m=after.tolist(), target_change_mm=delta.tolist(),
                  actual_before_m=before_actual.tolist(), actual_after_m=after_actual.tolist(),
                  actual_change_mm=((after_actual - before_actual) * 1000).tolist(),
                  single_contract_step_observed=bool(len(groups) == 1 and np.median(dials[moving, 1]) < 0 and np.max(abs(delta + 10)) < .02),
                  post_gesture_neutral_span_s=float(np.ptp(neutral_after[:, 0])) if len(neutral_after) else 0.,
                  maximum_abs_requested_torque_nm=np.max(abs(c[:, 1:7]), axis=0).tolist(),
                  zero_wheel_requests=bool(np.all(c[:, 5:7] == 0)),
                  faults_observed=bool(np.any(c[:, -1]) or np.any(g[:, -1]) or np.any(dials[:, -1])),
                  step_version_observed=bool(np.all(g[:, 5].astype(int) & 128)),
                  latest_recovering=bool(int(g[-1, 5]) & 32),
                  parser_candidate_rejections=dict(mirror=frames.rejected, output=outputs.rejected),
                  limitation='Observed gesture/targets only; not exact actual-length increment, complete CAN delivery or whole-run acceptance. Mirror candidate rejections in a mixed binary stream are not themselves lost-frame counts')
    np.savez_compressed(args.output / 'step-traces.npz', dials=dials, geometry=g, control=c)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
