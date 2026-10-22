#!/usr/bin/env python3
"""Check actual no-output RL re-entry recordings, never command hardware.

An uninterrupted RL selection cannot prove re-entry correction. Match only
fresh policy pairs and near-stationary raw feedback with board timestamps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-report', type=Path,
                   help='Compare a separately recorded pose; gaps remain explicit.')
    a = p.parse_args()
    meta, _, config = mirror.load_session(a.session)
    parser, clock = mirror.FrameParser(), mirror.FrameClock()
    frames, receive = [], []
    for event in mirror.iter_session_events(a.session):
        if event['type'] == 'data':
            for f in parser.feed(bytes.fromhex(event['data_hex'])):
                if clock.accept(f):
                    frames.append(f); receive.append(event['t_ns'] / 1e9)
    f = np.asarray(frames)
    t = np.asarray(receive)
    ticks = np.array([mirror.read_u32(row[8:10]) for row in f])
    text = config.read_text().split('[MACHINE_ID_SMALL_WHEELLEG]')[0]
    def values(name):
        return np.asarray([float(x.strip().rstrip('f')) for x in
                           re.search(r'\.' + name + r'\s*=\s*\{([^}]+)\}', text).group(1).split(',')])
    sign, nominal, reference = map(values, ('motor_sign', 'nominal_pos', 'reference_pos'))
    period = float(np.float32(2) * np.float32(re.search(r'\.dm_pos_max\s*=\s*([\d.]+)f', text).group(1)))
    qraw = f[:, 10:14] * sign[:4]
    selected = f[:, 152] == 0
    starts = np.flatnonzero(selected & ~np.r_[False, selected[:-1]])
    episodes = []
    for start in starts:
        end = start
        while end + 1 < len(f) and selected[end + 1]:
            end += 1
        pairs = {}
        for j in range(start, end + 1):
            sample = mirror.read_u32(f[j, 50:52])
            if int(f[j, 47]) & 1 and ticks[start] <= sample <= ticks[end]:
                pairs.setdefault(mirror.read_u32(f[j, 48:50]), (j, sample))
        good = []
        for j, sample in pairs.values():
            i = int(np.abs(ticks - sample).argmin())
            # Exclude moving, unavailable or later poses: first two seconds only.
            if (ticks[i] - ticks[start] > 2000 or f[i, 2] != 0
                    or int(f[i, 29]) & 30 != 30 or np.max(np.abs(f[i, 14:18])) > .1):
                continue
            qobs = f[j, 67:71] + nominal
            good.append((i, qobs))
        ep = dict(start_s=float(t[start]), end_s=float(t[end]),
                  entry_observed=bool(start > 0 and not selected[start - 1]),
                  fresh_pairs=len(pairs), stationary_pairs=len(good))
        if good:
            raw = np.asarray([qraw[i] for i, _ in good])
            obs = np.asarray([q for _, q in good])
            turns = np.rint((raw - obs) / period)
            expected = np.floor((raw[0] - reference) / period + .5)
            residual = raw - obs - turns * period
            ep.update(raw_model_median_rad=np.median(raw, axis=0).tolist(),
                      policy_model_median_rad=np.median(obs, axis=0).tolist(),
                      expected_initial_turn_offset=expected.astype(int).tolist(),
                      observed_turn_offset=np.median(turns, axis=0).astype(int).tolist(),
                      max_raw_policy_residual_rad=float(np.max(np.abs(residual))),
                      correction_matches=bool(np.all(turns == expected) and np.max(np.abs(residual)) < .02))
        episodes.append(ep)
    comparisons = []
    sources = [(str(a.session), ep) for ep in episodes]
    reference_clean = True
    if a.reference_report:
        previous = json.loads(a.reference_report.read_text())
        if (previous['machine_config_sha256'] != meta['machine_config']['sha256']
                or previous['position_period_rad'] != period):
            raise ValueError('Reference configuration or position period differs')
        reference_clean = not any(previous[key] for key in
                                  ('engaged_frames', 'fault_frames', 'missing_submitted_frames', 'board_restarts'))
        sources = [(previous['session'], ep) for ep in previous['episodes']] + sources
    for i, (left_source, left) in enumerate(sources):
        for j in range(i + 1, len(sources)):
            right_source, right = sources[j]
            if right_source != str(a.session):
                continue
            if not left.get('stationary_pairs') or not right.get('stationary_pairs'):
                continue
            raw_delta = np.array(right['raw_model_median_rad']) - left['raw_model_median_rad']
            history_turns = np.rint(raw_delta / period)
            pose_residual = raw_delta - history_turns * period
            obs_delta = np.array(right['policy_model_median_rad']) - left['policy_model_median_rad']
            comparisons.append(dict(episodes=[i, j], history_turn_difference=history_turns.astype(int).tolist(),
                                    sessions=[left_source, right_source],
                                    continuous_recording=left_source == right_source,
                                    max_same_pose_residual_rad=float(np.max(np.abs(pose_residual))),
                                    max_policy_position_difference_rad=float(np.max(np.abs(obs_delta))),
                                    pass_same_pose_nonzero_history=bool(np.any(history_turns)
                                       and np.max(np.abs(pose_residual)) < .03
                                       and np.max(np.abs(obs_delta)) < .03
                                       and left.get('correction_matches') and right.get('correction_matches')
                                       and left['entry_observed'] and right['entry_observed'])))
    a.out.mkdir(parents=True, exist_ok=False)
    engaged = int(np.count_nonzero(f[:, 153].astype(int) & 4))
    faults = int(np.count_nonzero(f[:, 2]))
    passed = (any(x['pass_same_pose_nonzero_history'] for x in comparisons)
              and reference_clean and not engaged and not faults and not clock.missing and not clock.restarts)
    r = dict(status='observed_reentry_correction_passed' if passed else 'required_coverage_missing_or_check_failed',
             session=str(a.session), machine_config_sha256=meta['machine_config']['sha256'],
             position_period_rad=period, first_board_tick_ms=int(ticks[0]), last_board_tick_ms=int(ticks[-1]),
             reference_report=str(a.reference_report) if a.reference_report else None,
             reference_report_sha256=hashlib.sha256(a.reference_report.read_bytes()).hexdigest() if a.reference_report else None,
             stream_sha256=meta['stream_sha256'], episodes=episodes, comparisons=comparisons,
             frames=len(f), missing_submitted_frames=clock.missing, board_restarts=clock.restarts,
             engaged_frames=engaged, fault_frames=faults,
             analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             limits=['No measured shaft torque or independent pose calibration.',
                     '100Hz raw and 50Hz policy snapshots are asynchronous; comparison uses near-stationary input.',
                     'Passing applies only to tested motors/history/pose; not powered PD or prior feedback-loss acceptance.'])
    (a.out / 'report.json').write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
