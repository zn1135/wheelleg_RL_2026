"""Locate a saved board-time transition and report RC samples around it; no hardware."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
from h7_output_capture import CaptureParser
import real2sim_huananhu as mirror


def first_control_time(stream, offset):
    parser = CaptureParser()
    with stream.open('rb') as file:
        file.seek(offset)
        if offset: file.readline()
        start = file.tell()
        while file.tell() - start < 262144:
            line = file.readline()
            if not line: break
            event = json.loads(line)
            if event['type'] != 'data': continue
            for packet in parser.feed(bytes.fromhex(event['data_hex'])):
                for row in packet['records']:
                    if row[1] == 1: return row[2] / 1e9
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--board-s', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stream = args.session / 'stream.jsonl'
    low, high = 0, stream.stat().st_size
    while high - low > 1048576:
        middle = (low + high) // 2
        value = first_control_time(stream, middle)
        if value is None: raise ValueError('Cannot locate a complete control packet')
        if value < args.board_s: low = middle
        else: high = middle
    frames, controls = [], []
    output_parser, frame_parser = CaptureParser(), mirror.FrameParser()
    start, end = max(0, low - 4194304), min(stream.stat().st_size, high + 4194304)
    with stream.open('rb') as file:
        file.seek(start)
        if start: file.readline()
        while file.tell() < end:
            line = file.readline()
            if not line or not line.endswith(b'\n'): break
            event = json.loads(line)
            if event['type'] != 'data': continue
            raw = bytes.fromhex(event['data_hex'])
            packets = output_parser.feed(raw)
            nearby = [row for packet in packets for row in packet['records'] if row[1] == 1 and abs(row[2] / 1e9 - args.board_s) < .15]
            for row in nearby:
                controls.append(dict(board_s=row[2] / 1e9,strategy=row[21],fault=row[5],flags=row[20],request_nm=list(row[6:12])))
            latest_time = next((row[2] / 1e9 for packet in packets for row in packet['records'] if row[1] == 1),None)
            for frame in frame_parser.feed(raw):
                frames.append(dict(nearest_control_s=latest_time,frame_board_s=mirror.read_u32(frame[8:10])/1000,
                                   s1=int(frame[89]),s2=int(frame[90]),strategy=int(frame[152]),rc_flags=int(frame[153]),
                                   fault=int(frame[2]),status=int(frame[3]),source_mask=int(frame[29]),
                                   wheel=float(frame[91]),arm=int(frame[92]),rc_age_ms=float(frame[145])))
    # Mirror and output packets may be in separate host reads; use monotonic 100Hz frame clock near the central switch.
    closest = min(range(len(frames)),key=lambda i:abs(frames[i]['frame_board_s'] - args.board_s))
    rows = frames[max(0,closest - 20):closest + 21]
    report = dict(window_byte_range=[start,end],target_board_s=args.board_s,rc_frames=rows,
                  control_transitions=[row for i,row in enumerate(controls) if i==0 or (row['strategy'],row['fault'],row['flags'])!=(controls[i-1]['strategy'],controls[i-1]['fault'],controls[i-1]['flags'])],
                  hardware_access=False,new_physics_steps=0)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
