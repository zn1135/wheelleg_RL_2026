"""Full synthetic120s wire export plus missing-pair/old-profile rejection; no hardware."""
import argparse
import binascii
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile

from scripts.agent.real2sim import prepare_pace_capture as prepare
from h7_output_capture import HEADER, RECORD


def record(sequence, kind, identifier, elapsed, stage=2, axis=0, profile=True):
    row = [0] * 27
    row[:6] = [sequence, kind, elapsed * 1000000 + 1000000, elapsed * 1000000 + 1010000, identifier, 0]
    row[20:24] = [60 if kind == 1 else 7, 7, stage, axis]
    if kind == 5:
        row[10:13] = [60., 2., 40.]
        row[14] = elapsed
        row[20] = 5 if profile else 1
    return row


def stream(path, rows):
    with path.open('w') as out:
        out.write(json.dumps({'type': 'connected', 't_ns': 0}) + '\n')
        batch = []
        packet_sequence = 0
        def flush():
            nonlocal packet_sequence, batch
            if not batch:
                return
            packet_sequence += 1
            data = HEADER.pack(b'H7OUT001', 1, 88, len(batch), 0, packet_sequence,
                               batch[-1][0], 0, len(batch), len(batch), 0)
            data += b''.join(RECORD.pack(*r) for r in batch)
            data += struct.pack('<H', binascii.crc_hqx(data, 0xffff))
            out.write(json.dumps({'type': 'data', 't_ns': packet_sequence, 'data_hex': data.hex()}) + '\n')
            batch = []
        for row in rows:
            batch.append(row)
            if len(batch) == 44:
                flush()
        flush()
        out.write(json.dumps({'type': 'stop', 't_ns': packet_sequence + 1}) + '\n')


def full_rows():
    sequence = 0
    for i in range(60001):
        for kind in (1, 3, 4, 5):
            sequence += 1
            yield record(sequence, kind, i, i * 2, 2 if i < 60000 else 3, i // 15000 if i < 60000 else 0)
    row = record(sequence + 1, 1, 60001, 120002)
    row[21] = 2
    yield row


def invoke(session, output):
    previous = sys.argv
    try:
        sys.argv = ['prepare', '--session', str(session), '--output', str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        session = root / 'session'
        session.mkdir()
        stream(session / 'stream.jsonl', full_rows())
        invoke(session, root / 'full')
        report = json.loads((root / 'full/report.json').read_text())
        assert report['complete_independent_sequence'] and report['target_points'] == 60000
        assert report['stage2_samples_by_axis'] == [15000] * 4
        for label, rows in (
                ('old', [record(i, kind, 0, 0, profile=False) for i, kind in enumerate((1, 3, 4, 5), 1)]),
                ('missing', [record(i, kind, 0, 0) for i, kind in enumerate((1, 4, 5), 1)])):
            stream(session / 'stream.jsonl', rows)
            try:
                invoke(session, root / label)
            except ValueError:
                pass
            else:
                raise AssertionError(label + ' data unexpectedly accepted')
    result = dict(passed=True, complete120s_and_final_hold_checked=True,
                  old_profile_rejected=True, missing_pair_rejected=True,
                  hardware_access=False, new_physics_steps=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        file.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
