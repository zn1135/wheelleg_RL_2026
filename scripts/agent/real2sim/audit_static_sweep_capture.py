"""Audit saved stationary windows, source identities and same-pose load checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import collect_static_leg_grid as collect


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.run
    meta = json.loads((root / 'usb/meta.json').read_text())
    assert meta['complete'] and meta.get('error') is None
    plan = json.loads((root / 'points/plan.json').read_text())
    status = json.loads((root / 'points/status.json').read_text())
    assert status['state'] == 'stopped'
    assert digest(root / 'points/collect_static_leg_grid.py') == plan['source_sha256']
    config = collect.ROOT / plan['config']['project_path']
    model = collect.ROOT / plan['model']['project_path']
    assert digest(config) == plan['config']['sha256'] and digest(model) == plan['model']['sha256']
    import re
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1].split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0]
    lu, lg = [float(re.search(r'\.' + n + r'\s*=\s*([\d.]+)f', block)[1]) for n in ('leg_lu', 'leg_lg')]
    rows = [json.loads(line) for line in (root / 'points/points.jsonl').read_text().splitlines()]
    assert rows == status['points']
    maximum, duration, groups = {}, [], {}
    for row in rows:
        path = root / 'points' / row['data_file']
        assert digest(path) == row['data_sha256']
        frames = np.load(path)['frames']
        candidate = collect.candidate(frames, ('left', 'right').index(row['side']), lu, lg, require_grid=False)
        assert candidate is not None
        assert abs(candidate['actual_length_mm'] - row['actual_length_mm']) < 1e-9
        assert candidate['board_time_s'] == row['board_time_s']
        assert int(row['geometry_flags']) & 128 and int(row['geometry_flags']) & 8
        assert np.max(abs(frames[:, 136:138])) == 0
        peak = dict(position_span_rad=float(np.ptp(frames[:, 10:14], axis=0).max()),
                    speed_rad_s=float(abs(frames[:, 14:18]).max()),
                    feedback_torque_span_nm=float(np.ptp(frames[:, 146:150], axis=0).max()),
                    request_torque_nm=float(abs(frames[:, 132:136]).max()),
                    motor_age_ms=float(frames[:, 138:142].max()))
        for key, value in peak.items():
            maximum[key] = max(maximum.get(key, 0.), value)
        duration.append(candidate['board_time_s'][1] - candidate['board_time_s'][0])
        groups.setdefault(row['side'] + '_' + row['direction'], []).append(row['actual_length_mm'])
    loads = json.loads((root / 'static-loads/summary.json').read_text())
    checks = {}
    for path in loads['reports']:
        report = json.loads(Path(path).read_text())
        assert report['physics_steps'] == 0
        for key, value in report['verification'].items():
            checks[key] = max(checks.get(key, 0.), value)
    assert checks['closure_max_m'] < 1e-8
    assert checks['gravity_potential_derivative_error_nm'] < 1e-5
    assert checks['constraint_virtual_work_error'] < 1e-5
    assert checks['spring_distance_derivative_error_m_rad'] < 1e-7
    result = dict(passed=True, side_points=len(rows), unique_frame_payloads=len({r['data_sha256'] for r in rows}),
                  coverage={key: dict(points=len(values), range_mm=[min(values), max(values)]) for key, values in groups.items()},
                  window_duration_range_s=[min(duration), max(duration)], static_window_maxima=maximum,
                  same_pose_load_check_maxima=checks, final_usb_complete=True,
                  usb_stream_sha256=digest(root / 'usb/stream.jsonl'),
                  point_ledger_sha256=digest(root / 'points/points.jsonl'), physics_steps=0,
                  limitations=['Checks all accepted stationary windows, not whole moving-trajectory continuity',
                               'Encoder geometry and driver torque feedback are not independent ruler/shaft-torque measurements'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
