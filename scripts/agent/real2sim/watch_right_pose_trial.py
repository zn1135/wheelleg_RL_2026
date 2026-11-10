"""Watch new raw recording events during a right-leg trial; never sends commands."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
from h7_output_capture import CaptureParser


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--start-byte', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=100.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.session / 'stream.jsonl'
    decoder = CaptureParser()
    groups, recent = {}, deque()
    seen, finished, reason = False, False, 'waiting_timeout'
    latest, last_print = None, -10000
    target_angle_deg = None
    started = time.monotonic()
    peak, left_peak, wheel_peak = 0., 0., 0.
    with path.open('rb') as file:
        file.seek(max(0, args.start_byte - 65536))
        if file.tell(): file.readline()
        while time.monotonic() - started < args.seconds and not finished:
            position = file.tell(); line = file.readline()
            if not line.endswith(b'\n'):
                file.seek(position); time.sleep(.03); continue
            event = json.loads(line)
            if event['type'] == 'stop': reason = 'recording_stopped'; break
            if event['type'] != 'data': continue
            for packet in decoder.feed(bytes.fromhex(event['data_hex'])):
                for row in packet['records']:
                    if row[1] == 1:
                        if seen and row[21] != 7:
                            reason, finished = 'mode_exited', True
                            print(json.dumps(dict(event=reason,board_s=row[2]/1e9,strategy=row[21],fault=row[5],request_nm=row[6:12])),flush=True)
                            break
                        if row[21] == 7:
                            seen = True
                            peak = max(peak, *(abs(value) for value in row[8:10]))
                            left_peak = max(left_peak, *(abs(value) for value in row[6:8]))
                            wheel_peak = max(wheel_peak, *(abs(value) for value in row[10:12]))
                    if row[1] not in (1,3,4,5) or row[21] != 7: continue
                    group = groups.setdefault(row[4], {})
                    group[row[1]] = row
                    if row[1] != 5 or 4 not in group: continue
                    geometry = group[4]
                    if target_angle_deg is None:
                        target_angle_deg=-.5*math.degrees(row[8]+row[9]) if row[20]&16 else 30.
                    latest = dict(board_s=row[2]/1e9,elapsed_ms=row[14],stage=row[22],flags=row[20],fault=row[5],
                                  right_length_mm=geometry[12]*1000,
                                  right_angle_deg=math.degrees(struct.unpack('<f',struct.pack('<I',geometry[14]))[0]),
                                  request_nm=list(group[1][6:12]) if 1 in group else None)
                    recent.append(latest)
                    while recent and latest['elapsed_ms']-recent[0]['elapsed_ms']>3000: recent.popleft()
                    if latest['elapsed_ms'] - last_print >= 1000:
                        print(json.dumps(latest),flush=True); last_print=latest['elapsed_ms']
                    if row[20]&2 or row[5]: reason,finished='locked_or_fault',True
                    elif row[22]==3 and row[14]>=28000: reason,finished='target_complete_and_3s_hold',True
                    if finished: break
                if finished: break
            # Keep pairing bounded while retaining all new records in the raw source.
            if len(groups)>6000:
                cutoff=max(groups)-2000
                groups={key:value for key,value in groups.items() if key>=cutoff}
        source_end=file.tell()
    report=dict(reason=reason,seen_strategy7=seen,latest=latest,
                peak_right_request_nm=peak,peak_left_request_nm=left_peak,peak_wheel_request_nm=wheel_peak,
                last_window_samples=len(recent),
                last_window_max_length_error_mm=max((abs(row['right_length_mm']-190) for row in recent),default=None),
                target_angle_deg=target_angle_deg,
                last_window_max_angle_error_deg=max((abs(row['right_angle_deg']-target_angle_deg) for row in recent),default=None),
                source_start_byte=args.start_byte,source_end_byte=source_end,hardware_commands=False,new_physics_steps=0,
                limitation='Live observation only; full paired-input and RC/continuity audit follows')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
