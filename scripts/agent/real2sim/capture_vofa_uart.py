"""Passively record the eight-channel VOFA UART; no serial writes or motor control."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import serial

from scripts.agent.real2sim.view_h7_joint_limits import JointFrames


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--dtr', action='store_true', help='Assert the USB serial ready line for bridge compatibility; no serial writes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    decoder = JointFrames()
    samples, times = [], []
    started = time.monotonic_ns()
    with serial.Serial(port=None, baudrate=1152000, timeout=.03, exclusive=True) as connection:
        # 只读UART，不借DTR/RTS复位控制板。
        connection.dtr = args.dtr
        connection.rts = False
        connection.port = args.port
        connection.open()
        with (args.output / 'stream.bin').open('wb') as raw_file, (args.output / 'chunks.jsonl').open('w') as chunks:
            while (time.monotonic_ns() - started) / 1e9 < args.seconds:
                raw = connection.read(min(connection.in_waiting or 1, 4096))
                if not raw:
                    continue
                stamp = time.monotonic_ns()
                raw_file.write(raw)
                chunks.write(json.dumps(dict(host_monotonic_ns=stamp, bytes=len(raw))) + '\n')
                for values in decoder.feed(raw):
                    samples.append(values)
                    times.append(stamp)
    values = np.array(samples, dtype=np.float64).reshape(-1, 8)
    stamps = np.array(times, dtype=np.int64)
    np.savez_compressed(args.output / 'samples.npz', host_monotonic_ns=stamps, values=values)
    report = dict(port=args.port, baud=1152000, dtr=args.dtr, framing='8N1 JustFloat; four motors q_rad/feedback_tau_nm adjacent',
                  frames=len(values), duration_s=(time.monotonic_ns() - started) / 1e9,
                  host_frame_rate_hz=(len(values) - 1) / ((stamps[-1] - stamps[0]) / 1e9) if len(values) > 1 else None,
                  parser_discarded_bytes=decoder.discarded,
                  first_values=values[0].tolist() if len(values) else None,
                  last_values=values[-1].tolist() if len(values) else None,
                  stream_sha256=hashlib.sha256((args.output / 'stream.bin').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation='No source age/CRC/sequence in eight-channel frames; arrival rate is not motor feedback rate; zero physics/control writes')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not len(values):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
