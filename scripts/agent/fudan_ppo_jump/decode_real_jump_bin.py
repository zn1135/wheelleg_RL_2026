"""校验原始Jetson BIN和CSV，复用H7字段定义解码后交给Jump分析器。"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib

import numpy as np
from .probe_takeoff_work import ROOT, write_json

PHASES = {0: 'walk', 1: 'armed', 2: 'takeoff', 3: 'flight', 4: 'landing'}
NET_COLUMNS = ['jetson_time_us', 'sequence', 'phase', 'model', 'flags', 'board_time_us',
               *['obs_'+str(i) for i in range(25)], *['history_'+str(i) for i in range(125)],
               *['action_'+str(i) for i in range(6)], 'support_force_n', 'mean_leg_length_m']


def csv_text(columns, rows):
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def verify_csv(path, columns, rows):
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        assert next(reader) == columns
        supplied = list(reader)
    assert len(supplied) == len(rows)
    textual = [i for i, name in enumerate(columns) if name == 'phase']
    numeric = [i for i in range(len(columns)) if i not in textual]
    assert all(a[i] == b[i] for a, b in zip(supplied, rows) for i in textual)
    a = np.asarray([[r[i] for i in numeric] for r in supplied], dtype=float)
    b = np.asarray([[r[i] for i in numeric] for r in rows], dtype=float)
    assert np.array_equal(a, b), path
    return dict(rows=len(rows), all_fields_exact=True)


def run(binary, supplied_net, supplied_pd, sums, output):
    output.mkdir(parents=True, exist_ok=False)
    expected = {line.split(None, 1)[1]: line.split()[0] for line in sums.read_text().splitlines() if line.strip()}
    hashes = {}
    for path in (binary, supplied_net, supplied_pd):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected[path.name] == digest, path
        hashes[path.name] = digest
    source = ROOT/'deployment/H7_RL/tools/jetson/export_pd.py'
    spec = importlib.util.spec_from_file_location('h7_export_fields', source)
    fields = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fields)
    blob = binary.read_bytes()
    offset, rx, net_rows, events, counts = 0, [], [], [], {}
    while offset+16 <= len(blob):
        kind, size, host = struct.unpack_from('<IIQ', blob, offset)
        if offset+16+size > len(blob):
            break
        payload = blob[offset+16:offset+16+size]
        offset += 16+size
        counts[str(kind)] = counts.get(str(kind), 0)+1
        if kind == 1:
            rx.append(payload)
        elif kind == 15:
            assert size == 656
            seq, phase, model, flags, board, *values = struct.unpack('<4IQ158f', payload)
            net_rows.append([host, seq, PHASES[phase], model, flags, board, *values])
        elif kind == 4:
            events.append(dict(host_us=host, text=payload.decode('utf-8', errors='replace')))
    outer_tail = len(blob)-offset
    stream = b''.join(rx)
    offset, errors, skipped = 0, 0, 0
    pd_rows, stair_rows, frame_counts = [], [], {}
    while offset+36 <= len(stream):
        marker = stream.find(b'JRL2', offset)
        if marker < 0:
            break
        skipped += marker-offset
        offset = marker
        _, version, kind, size, session, reset, sequence, board = fields.FRAME_HEADER.unpack_from(stream, offset)
        if version != 2 or size > 348:
            errors += 1
            offset += 1
            continue
        end = offset+32+size+4
        if end > len(stream):
            break
        if zlib.crc32(stream[offset:end-4]) != struct.unpack_from('<I', stream, end-4)[0]:
            errors += 1
            offset += 1
            continue
        frame_counts[str(kind)] = frame_counts.get(str(kind), 0)+1
        if kind == 5:
            assert size == fields.PAYLOAD.size
            flags, fault, drops, obs_sequence, obs_ms, *values = fields.PAYLOAD.unpack_from(stream, offset+32)
            pd_rows.append([board, sequence, reset, flags, fault, drops, obs_sequence, obs_ms,
                            *[int(bool(flags & (1 << i))) for i in range(10)], *values])
        elif kind == 14:
            assert size == fields.STAIR_PAYLOAD.size
            flags, drops, fault, phase, event, condition, *values = fields.STAIR_PAYLOAD.unpack_from(stream, offset+32)
            stair_rows.append([board, sequence, reset, flags, drops, fault, phase,
                               event & 255, event >> 8, condition, *values])
        offset = end
    assert errors == 0
    checked = dict(model_input_actions=verify_csv(supplied_net, NET_COLUMNS, net_rows),
                   rl_pd=verify_csv(supplied_pd, fields.COLUMNS, pd_rows))
    obs = np.asarray([row[6:31] for row in net_rows], dtype=float)
    history = np.asarray([row[31:156] for row in net_rows], dtype=float).reshape(-1, 5, 25)
    sequences = np.asarray([row[1] for row in net_rows])
    models = np.asarray([row[3] for row in net_rows])
    state_columns = np.r_[0:6, 9:25]
    history_errors, speed_errors = [], []
    for i in np.flatnonzero(models == 5200):
        if i < 4 or not np.all(np.diff(sequences[i-4:i+1]) == 1):
            continue
        history_errors.append(float(abs(history[i, :, state_columns]-obs[i-4:i+1, state_columns].T).max()))
        expected_speed = obs[i-4:i+1, 6]*np.where(models[i-4:i+1] == 5200, 1., 1.5)
        speed_errors.append(float(abs(history[i, :, 6]-expected_speed).max()))
    checked['jump_history'] = dict(frames=len(history_errors),
        noncommand_max_error=max(history_errors, default=None),
        forward_scale_max_error=max(speed_errors, default=None))
    audit = dict(hashes=hashes, supplied_files_checked=3,
        unavailable_manifest_files=[name for name in expected if name not in hashes],
        outer_record_counts=counts, outer_trailing_bytes=outer_tail,
        valid_wire_frame_counts=frame_counts, crc_or_header_errors=errors,
        leading_or_resync_skipped_bytes=skipped, wire_trailing_bytes=len(stream)-offset,
        csv_cross_checks=checked, field_decoder_source=str(source), field_decoder_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_json(output/'decode-audit.json', audit)
    write_json(output/'events.json', events)
    # 固化同一BIN解码所得的输入，复用已有逐次分析并保留校验来源。
    bundle = output/'decoded-input.zip'
    folder = binary.stem
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(folder+'/'+binary.name, blob)
        archive.writestr('SHA256.txt', hashes[binary.name]+'  '+folder+'/'+binary.name+'\n')
        archive.writestr(folder+'/model-input-actions.csv', csv_text(NET_COLUMNS, net_rows))
        archive.writestr(folder+'/h7-pd.csv', csv_text(fields.COLUMNS, pd_rows))
        archive.writestr(folder+'/h7-stair.csv', csv_text(fields.STAIR_COLUMNS, stair_rows))
    from .analyze_real022 import run as analyze
    analyze(bundle, output/'analysis')
    report_path = output/'analysis/result.json'
    report = json.loads(report_path.read_text())
    report['decode_audit'] = '../decode-audit.json'
    report['scope'] = ('Original BIN hashes/framing and wire CRC checked. Both supplied CSV files '
                       'match the independently decoded BIN field by field. Contact and physical '
                       'jump height are unmeasured; motor feedback is the recorded estimate.')
    write_json(report_path, report)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model-csv', type=Path, required=True)
    parser.add_argument('--pd-csv', type=Path, required=True)
    parser.add_argument('--sums', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.binary, args.model_csv, args.pd_csv, args.sums, args.output)
