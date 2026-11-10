"""Freeze a raw-byte window and inspect actual right-pose state/lock events; no hardware."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
from h7_output_capture import CaptureParser
from scripts.agent.real2sim.inspect_pose_rc_transition import first_control_time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--after-observation', type=Path)
    parser.add_argument('--start-board-s', type=float)
    parser.add_argument('--end-board-s', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    stream = args.session / 'stream.jsonl'
    end = stream.stat().st_size
    if args.start_board_s is not None:
        low, high = 0, end
        while high-low>1048576:
            middle=(low+high)//2; value=first_control_time(stream,middle)
            if value is None: raise ValueError('Cannot locate start time')
            if value<args.start_board_s: low=middle
            else: high=middle
        start=max(0,low-1048576)
    elif args.after_observation is not None:
        start = max(0, json.loads(args.after_observation.read_text())['stream_bytes'] - 4194304)
    else: raise ValueError('Specify a start board time or observation')
    decoder, digest = CaptureParser(), hashlib.sha256()
    bundles, transitions = {}, []
    gaps, previous_seq, dropped = [], None, []
    latest_control = None
    with stream.open('rb') as file:
        file.seek(start)
        if start: file.readline()
        start = file.tell(); used_end = start
        while file.tell() < end:
            line = file.readline()
            if file.tell() > end or not line.endswith(b'\n'): break
            digest.update(line); used_end = file.tell()
            event = json.loads(line)
            if event['type'] != 'data': continue
            for packet in decoder.feed(bytes.fromhex(event['data_hex'])):
                dropped.append(packet['dropped'])
                for row in packet['records']:
                    if previous_seq is not None and (row[0] - previous_seq) & 0xffffffff != 1:
                        gaps.append([previous_seq, row[0]])
                    previous_seq = row[0]
                    if row[1] == 1:
                        latest_control = row
                        marker = (row[21], row[5], bool(row[20] & 2))
                        if not transitions or tuple(transitions[-1]['marker']) != marker:
                            transitions.append(dict(board_s=row[2] / 1e9, marker=marker, request_nm=list(row[6:12])))
                    if row[1] in (1, 3, 4, 5) and row[21] == 7:
                        bundles.setdefault(row[4], {})[row[1]] = row
            if args.end_board_s is not None and latest_control and latest_control[2]/1e9>args.end_board_s: break
    attempts = [group for group in bundles.values() if 5 in group]
    attempts.sort(key=lambda group: group[5][2])
    runs=[]
    for group in attempts:
        target=group[5]
        if not runs or target[14]<runs[-1][-1][5][14] or target[2]-runs[-1][-1][5][2]>10000000:
            runs.append([])
        runs[-1].append(group)

    def describe(group):
        target = group[5]
        result = dict(board_s=target[2] / 1e9, elapsed_ms=target[14], stage=target[22], flags=target[20],
                      running=bool(target[20] & 1), locked=bool(target[20] & 2), saturated_mask=(target[20] >> 8) & 15,
                      target_firmware_rad=list(target[6:10]), fault=target[5])
        if 1 in group:
            row = group[1]; result.update(request_nm=list(row[6:12]), control_flags=row[20], drive_permit=bool(row[20] & 32))
        if 3 in group:
            row = group[3]
            age = [row[14], row[15], *struct.unpack('<2I', struct.pack('<4h', *row[16:20]))]
            result.update(q_firmware_rad=list(row[6:10]), v_firmware_rad_s=list(row[10:14]), feedback_age_us=age)
            a, b = row[8] + np.pi, row[9]
            ax, ay, bx, by = .21 * np.cos(a), .21 * np.sin(a), .21 * np.cos(b), .21 * np.sin(b)
            dx, dy = bx - ax, by - ay; distance = np.hypot(dx, dy)
            height = np.sqrt(.25 ** 2 - distance ** 2 / 4)
            px, py = (ax + bx) / 2 - height * dy / distance, (ay + by) / 2 + height * dx / distance
            knee = -((np.arctan2(py - ay, px - ax) - a + np.pi) % (2 * np.pi) - np.pi + 1.6260029368)
            result.update(right_length_mm=float(np.hypot(px, py) * 1000), right_angle_deg=float(np.degrees(np.pi / 2 - np.arctan2(py, px))),
                          right_knee_model_rad=float(knee), age_over5ms=[i for i, value in enumerate(age) if value > 5000])
        if 4 in group: result['geometry_flags'] = group[4][20]
        return result

    locks = [i for i, group in enumerate(attempts) if group[5][20] & 2]
    selected = {0, max(0, len(attempts) - 1)}
    if locks: selected.update(range(max(0, locks[0] - 3), min(len(attempts), locks[0] + 3)))
    report = dict(stream_window_start_byte=start, stream_window_end_byte=used_end, stream_window_sha256=digest.hexdigest(),
                  strategy7_target_count=len(attempts), first_pose=describe(attempts[0]) if attempts else None,
                  last_pose=describe(attempts[-1]) if attempts else None,
                  first_lock=describe(attempts[locks[0]]) if locks else None,
                  around_first_lock=[describe(attempts[i]) for i in sorted(selected)] if attempts else [],
                  flags_histogram=dict(Counter(str(group[5][20]) for group in attempts)), transitions=transitions,
                  runs=[dict(start=describe(run[0]),end=describe(run[-1]),target_records=len(run),
                             reached_stage3=any(group[5][22]==3 for group in run),
                             any_locked=any(group[5][20]&2 for group in run)) for run in runs],
                  crc_rejected=decoder.rejected, sequence_gaps=gaps[:20], sequence_gap_count=len(gaps),
                  queue_drop_range=[min(dropped), max(dropped)] if dropped else None,
                  latest_control=dict(board_s=latest_control[2] / 1e9, strategy=latest_control[21], fault=latest_control[5], request_nm=list(latest_control[6:12])) if latest_control else None,
                  hardware_commands=False, new_physics_steps=0,
                  limitation='Lock reason is not transmitted explicitly; infer candidates only from paired state and actual source checks')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'paired-records.npz', **{f'kind{kind}': np.array([group[kind] for group in attempts if kind in group]) for kind in (1, 3, 4, 5)})
    print(json.dumps({key: report[key] for key in ('strategy7_target_count', 'first_pose', 'first_lock', 'last_pose', 'flags_histogram', 'transitions', 'sequence_gap_count')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
