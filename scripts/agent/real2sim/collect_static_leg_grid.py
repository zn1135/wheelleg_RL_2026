"""Observe the existing USB journal and save 10mm static points; never owns USB or commands motors."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import signal
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from h7_output_capture import CaptureParser
from scripts.agent.real2sim.analyze_tilt_length import geometry

STEP_MM, TOLERANCE_MM, HOLD_S = 10, 2., 5.
MIN_MM, MAX_MM = 180, 320


def available(f):
    return (f[152] == 6 and f[2] == 0 and int(f[109]) & 62 == 60 and abs(f[91]) < 1e-7
            and int(f[29]) & 31 == 31 and 0 <= f[138:142].min() and f[138:142].max() <= 5
            and 0 <= f[144] <= 20 and np.isfinite(f[np.r_[10:18, 22:26, 132:136, 146:150]]).all())


def candidate(frames, side, lu, lg, require_grid=True):
    if len(frames) < 400 or not all(available(f) for f in frames):
        return None
    ticks = np.array([mirror.read_u32(f[8:10]) for f in frames], dtype=np.int64)
    sequences = np.array([mirror.read_u32(f[6:8]) for f in frames], dtype=np.int64)
    if (ticks[-1] - ticks[0] < (HOLD_S - .1) * 1000 or np.any(np.diff(sequences) != 1)
            or np.any(np.diff(ticks) <= 0) or np.max(np.diff(ticks)) > 25):
        return None
    f = np.asarray(frames)
    if np.ptp(f[:, 10:14], axis=0).max() > .003 or abs(f[:, 14:18]).max() > .03:
        return None
    if np.ptp(f[:, 146:150], axis=0).max() > .2:
        return None
    quat = f[:, 22:26]
    if 2 * np.arccos(np.clip(abs(quat @ quat[len(f) // 2]), 0., 1.)).max() > np.deg2rad(.5):
        return None
    lengths, angles = geometry(f[:, 10 + 2 * side:12 + 2 * side], lu, lg)
    actual = float(np.median(lengths) * 1000)
    grid = int(round(actual / STEP_MM) * STEP_MM)
    if require_grid and (grid < MIN_MM or grid > MAX_MM or abs(actual - grid) > TOLERANCE_MM):
        return None
    return dict(side=('left', 'right')[side], grid_mm=grid if require_grid else None, actual_length_mm=actual,
                length_span_mm=float(np.ptp(lengths) * 1000), angle_deg=float(np.rad2deg(np.median(angles))),
                board_time_s=[float(ticks[0] / 1000), float(ticks[-1] / 1000)], frames=len(f),
                motor_angles_rad=np.median(f[:, 10:14], axis=0).tolist(),
                feedback_mean_nm=np.mean(f[:, 146:150], axis=0).tolist(),
                request_mean_nm=np.mean(f[:, 132:136], axis=0).tolist())


def selfcheck(path, output):
    base = np.load(path)['frames']
    f = np.tile(base, (2, 1))[:500].copy()
    for i, frame in enumerate(f):
        for start, value in ((6, 100 + i), (8, 10000 + i * 10)):
            frame[start:start + 2] = [value & 65535, value >> 16]
    assert candidate(f, 0, .21, .25)['grid_mm'] == 180
    assert candidate(f[:301], 0, .21, .25) is None
    for column, value in ((2, 1.), (138, 6.), (14, .05), (152, 2.), (91, .01)):
        changed = f.copy(); changed[-1, column] = value
        assert candidate(changed, 0, .21, .25) is None
    changed = f.copy(); changed[-1, 146] += .3
    assert candidate(changed, 0, .21, .25) is None
    changed = f.copy(); changed[-1, 6] += 1
    assert candidate(changed, 0, .21, .25) is None
    changed = f.copy(); changed[:, 10] -= .03
    assert candidate(changed, 0, .21, .25) is None
    assert candidate(changed, 0, .21, .25, require_grid=False) is not None
    changed = f.copy(); changed[:, 10] -= 1.1
    assert candidate(changed, 0, .21, .25) is None
    assert candidate(changed, 0, .21, .25, require_grid=False) is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(passed=True, cases=13, hardware_commands=0, physics_steps=0,
                                     scope='Post-decoder grid/hold/continuity/freshness/static gates on modified frozen frames')) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selfcheck-frames', type=Path)
    parser.add_argument('--actual-length', action='store_true', help='Record each stable target plateau at measured length, without integer-grid rejection')
    args = parser.parse_args()
    if args.selfcheck_frames:
        selfcheck(args.selfcheck_frames, args.output)
        return
    if not args.session:
        parser.error('--session is required for live collection')
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.session / 'meta.json').read_text())
    config = ROOT / meta['machine_config']['project_path']
    assert hashlib.sha256(config.read_bytes()).hexdigest() == meta['machine_config']['sha256']
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1].split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0]
    lu, lg = [float(re.search(r'\.' + n + r'\s*=\s*([\d.]+)f', block)[1]) for n in ('leg_lu', 'leg_lg')]
    source = args.session / 'stream.jsonl'
    start_byte = source.stat().st_size
    plan = dict(session=str(args.session.resolve()), source_start_byte=start_byte, model=meta['model'],
                config=meta['machine_config'], step_mm=STEP_MM, tolerance_mm=TOLERANCE_MM, hold_s=HOLD_S,
                range_mm=None if args.actual_length else [MIN_MM, MAX_MM], force_upper_bound_n=None,
                sampling_mode='measured_length_per_target_plateau' if args.actual_length else 'fixed_actual_length_grid',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                meaning='Actual length uses existing encoder geometry, not a ruler; samples are candidates, not adopted physical parameters')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    decoder, window = mirror.FrameParser(), deque(maxlen=800)
    output_decoder = CaptureParser()
    latest_geometry = None
    points, seen, direction = [], set(), 'initial'
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    status = dict(state='waiting', points=points, sampling_mode=plan['sampling_mode'], direction=direction)
    last_checked_tick = 0

    def publish():
        payload = dict(status, points=points, count_left=sum(p['side'] == 'left' for p in points),
                       count_right=sum(p['side'] == 'right' for p in points), updated_unix_s=time.time())
        temporary = args.output / 'status.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(args.output / 'status.json')

    with source.open('rb') as journal, (args.output / 'points.jsonl').open('w', buffering=1) as ledger:
        journal.seek(start_byte)
        # 开始时尾行若尚未写完，只丢弃这一行，绝不接管USB或修改记录。
        aligned = start_byte == 0
        if start_byte:
            journal.seek(start_byte - 1); aligned = journal.read(1) == b'\n'
        publish()
        last_publish = time.monotonic()
        while not stopped:
            position = journal.tell()
            line = journal.readline()
            if not line.endswith(b'\n'):
                journal.seek(position); time.sleep(.03)
                if time.monotonic() - last_publish >= 1.:
                    publish(); last_publish = time.monotonic()
                continue
            if not aligned:
                aligned = True
                continue
            event = json.loads(line)
            if event['type'] != 'data':
                if event['type'] in ('connected', 'disconnected', 'stop'):
                    window.clear(); last_checked_tick = 0; status['state'] = 'source_' + event['type']
                continue
            chunk = bytes.fromhex(event['data_hex'])
            for packet in output_decoder.feed(chunk):
                for row in packet['records']:
                    if row[1] == 4 and row[20]&8:
                        latest_geometry = (row[3]/1e9, struct.unpack('<2f',struct.pack('<4h',*row[16:20])),row[20])
            for frame in decoder.feed(chunk):
                tick = mirror.read_u32(frame[8:10])
                if abs(frame[91]) > 1e-7:
                    direction = 'extend' if frame[91] > 0 else 'contract'
                status['direction'] = direction
                if not available(frame):
                    window.clear(); status['state'] = 'waiting_for_static_leg_mode'
                    continue
                if window:
                    dt = tick - mirror.read_u32(window[-1][8:10])
                    ds = mirror.read_u32(frame[6:8]) - mirror.read_u32(window[-1][6:8])
                    if dt <= 0 or dt > 25 or ds != 1:
                        window.clear(); last_checked_tick = 0
                window.append(frame)
                while window and tick - mirror.read_u32(window[0][8:10]) > HOLD_S * 1000:
                    window.popleft()
                status['state'] = 'observing'
                status['board_s'] = tick / 1000
                if len(window) < 400 or tick - last_checked_tick < 200:
                    continue
                last_checked_tick = tick
                if args.actual_length and (latest_geometry is None or abs(latest_geometry[0]-tick/1000)>.05):
                    continue
                # 两侧分别归档10mm网格；不把另一条腿假定为相同实际长度。
                for side in (0, 1):
                    point = candidate(window, side, lu, lg, require_grid=not args.actual_length)
                    if point is None:
                        continue
                    key = (point['side'], round(latest_geometry[1][side]*1000,2) if args.actual_length else point['grid_mm'], direction)
                    if key in seen:
                        continue
                    name = f"point-{len(points) + 1:03d}.npz"
                    np.savez_compressed(args.output / name, frames=np.asarray(window))
                    point.update(direction=direction, data_file=name, source_byte_end=journal.tell())
                    if args.actual_length:
                        point['control_targets_m'] = list(latest_geometry[1])
                        point['geometry_flags'] = latest_geometry[2]
                    point['data_sha256'] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                    points.append(point); seen.add(key)
                    ledger.write(json.dumps(point) + '\n'); publish()
                    print(json.dumps(point), flush=True)
        status['state'] = 'stopped'; publish()


if __name__ == '__main__':
    main()
