#!/usr/bin/env python3
"""Replay recorded H7 obs39 through an explicit ONNX and compare paired action6."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_dynamics_huananhu as dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--machine-config', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=dynamics.mirror.MODEL)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    session = dynamics.Session(args.session, model=args.model, config=args.machine_config)
    evaluator = ReferenceEvaluator(onnx.load(str(args.onnx)))
    pairs = {}
    for row, frame in enumerate(session.f):
        if not int(frame[47]) & 1:
            continue
        key = (dynamics.mirror.read_u32(frame[48:50]), dynamics.mirror.read_u32(frame[52:54]))
        obs, action = frame[57:96], frame[96:102]
        if key in pairs:
            previous = pairs[key]
            np.testing.assert_array_equal(obs, previous['obs'])
            np.testing.assert_array_equal(action, previous['action'])
        else:
            pairs[key] = {'row': row, 'obs': obs.copy(), 'action': action.copy()}
    if not pairs:
        raise ValueError('No successful paired policy snapshots')
    checked = []
    for (sequence, tick), pair in pairs.items():
        prediction = np.asarray(evaluator.run(None, {'obs': pair['obs'][None]})[0])
        if prediction.shape != (1, 6) or not np.isfinite(prediction).all():
            raise ValueError('ONNX must produce one finite action6 for obs39')
        difference = abs(prediction[0] - pair['action'])
        checked.append({'policy_sequence': sequence, 'published_tick_ms': tick,
                        'first_row': pair['row'], 'max_action_difference': float(difference.max()),
                        'action_difference': difference.tolist()})
    maximum = max(p['max_action_difference'] for p in checked)
    tolerance = dynamics.sim.ONNX_ACTION_ATOL
    report = {'ok': maximum <= tolerance,
              'scope': 'recorded obs/action numerical parity only; not physics, timing or physical mapping acceptance',
              'session': session.summary, 'onnx': dynamics.identity(args.onnx),
              'unique_policy_pairs': len(checked), 'duplicate_snapshots_identical': True,
              'max_action_difference': maximum, 'action_atol': tolerance,
              'verified_actual_wheel_control_snapshots': session.wheel_response_verified,
              'pairs': checked, 'checker': dynamics.identity(Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'pairs'}, indent=2))
    if not report['ok']:
        raise AssertionError(f'Recorded H7 actions differ from explicit ONNX: {maximum} > {tolerance}')


if __name__ == '__main__':
    main()
