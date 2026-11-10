"""Freeze first independent small/high PACE interval, preserving raw-prefix identity."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
from h7_output_capture import CaptureParser


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    decoder = CaptureParser()
    states, geometries, controls, targets = {}, {}, {}, []
    active = ended = completed = False
    digest = hashlib.sha256()
    consumed = 0
    previous = first_drop = last_drop = None
    gaps = []
    crc_start = 0
    crc_end = None
    can_failures = 0
    with (args.session / 'stream.jsonl').open('rb') as file:
        for line in file:
            if not line.endswith(b'\n'):
                break
            consumed += len(line)
            digest.update(line)
            event = json.loads(line)
            if event['type'] != 'data':
                continue
            for packet in decoder.feed(bytes.fromhex(event['data_hex'])):
                for row in packet['records']:
                    if active and row[1] == 1 and row[21] != 7:
                        ended = True
                        break
                    if row[1] == 5 and row[21] == 7 and row[20] & 1:
                        if not row[20] & 4:
                            raise ValueError('Excluded old capture profile; independent small/high data required')
                        if not active:
                            active = True
                            first_drop = packet['dropped']
                            crc_start = decoder.rejected
                        if row[22] == 2:
                            targets.append(row)
                    if active and not completed:
                        if row[1] == 2 and row[20] != 0:
                            can_failures += 1
                        if previous is not None and (row[0] - previous) & 0xffffffff != 1:
                            gaps.append([previous, row[0]])
                        previous = row[0]
                        last_drop = packet['dropped']
                    if row[1] == 5 and row[21] == 7 and row[22] == 3 and row[20] & 5 == 5 and not row[20] & 2 and not row[5]:
                        completed = True
                        crc_end = decoder.rejected
                    if row[21] == 7 and row[1] in (1, 3, 4):
                        {1: controls, 3: states, 4: geometries}[row[1]][row[4]] = row
                if ended:
                    break
            if ended:
                break
    if not targets:
        raise ValueError('No independent small/high target records')
    missing = [r[4] for r in targets if any(r[4] not in source for source in (controls, states, geometries))]
    if missing:
        raise ValueError('Missing paired controls/state/geometry: ' + str(missing[:10]))
    values = dict(id=[r[4] for r in targets], elapsed_ms=[r[14] for r in targets],
                  board_ns=[r[2] for r in targets], stage=[r[22] for r in targets],
                  axis=[r[23] for r in targets], target_flags=[r[20] for r in targets],
                  q_firmware=[states[r[4]][6:10] for r in targets],
                  v_firmware=[states[r[4]][10:14] for r in targets],
                  target_firmware=[r[6:10] for r in targets],
                  request_firmware=[controls[r[4]][6:10] for r in targets],
                  feedback_firmware=[geometries[r[4]][6:10] for r in targets],
                  kp=[r[10] for r in targets], kd=[r[11] for r in targets], limit=[r[12] for r in targets])
    a = {key: np.asarray(value) for key, value in values.items()}
    if not all(np.isfinite(value).all() for value in a.values()):
        raise ValueError('Nonfinite capture values')
    for target in targets:
        state, control, geometry = (source[target[4]] for source in (states, controls, geometries))
        ages = list(state[14:16]) + list(np.frombuffer(np.array(state[16:20], dtype='<i2').tobytes(), dtype='<u4'))
        if max(ages) > 5000 or geometry[20] & 7 != 7 or control[20] & 62 != 60 or any(control[10:12]):
            raise ValueError('Stale/gated/invalid state or nonzero wheel request')
    predicted = np.clip(a['kp'][:, None] * (a['target_firmware'] - a['q_firmware'])
                        - a['kd'][:, None] * a['v_firmware'], -a['limit'][:, None], a['limit'][:, None])
    pd_error = float(abs(predicted - a['request_firmware']).max())
    counts = [int(np.sum((a['stage'] == 2) & (a['axis'] == axis))) for axis in range(4)]
    blocked = any(r[20] & 2 or r[5] for r in targets)
    crc_count = (decoder.rejected if crc_end is None else crc_end) - crc_start
    quality = not (gaps or last_drop - first_drop or crc_count or can_failures or blocked or pd_error > .001)
    complete = quality and counts == [15000] * 4 and completed and ended
    report = dict(profile='small-high-v1', stream_prefix_bytes=consumed,
                  stream_prefix_sha256=digest.hexdigest(), target_points=len(targets),
                  first_elapsed_ms=int(a['elapsed_ms'][0]), last_elapsed_ms=int(a['elapsed_ms'][-1]),
                  record_sequence_gaps=gaps, queue_drop_increase_during_sweep=last_drop - first_drop,
                  crc_or_schema_rejections_during_sweep=crc_count,
                  all_target_ids_paired=True, pd_reconstruction_max_error_nm=pd_error,
                  fault_or_blocked=blocked, stage2_samples_by_axis=counts,
                  failed_CAN_enqueue_attempts_during_sweep=can_failures,
                  completed_center_hold_observed=completed, manual_exit_observed=ended,
                  target_flags=dict(Counter(int(r[20]) for r in targets)),
                  data_quality_passed=quality, complete_independent_sequence=complete,
                  scope='Selected interval only; source prefix frozen, no physics/parameter fit')
    np.savez_compressed(args.output / 'paired-inputs.npz', **a)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
