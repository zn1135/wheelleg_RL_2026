#!/usr/bin/env python3
"""Finite analytic checks for retimed H7 derivative, ramp and standup clocks."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ct_mjlab.teacher.mechanics import PD, Ramp
from ct_mjlab.teacher.standup import Standup, SWING, DONE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dt in (.001, .002, .01):
        pid = PD((1,), 'cpu', 2., 100., 10000., dt=dt)
        pid.step(torch.zeros(1), torch.zeros(1))
        output = pid.step(torch.zeros(1), torch.full((1,), dt))
        # 对单位斜率误差，D项在各周期下都应为100×1ms，而不是放大10倍。
        torch.testing.assert_close(output, torch.tensor([2*dt+.1]))
        ramp = Ramp((1,), 'cpu', dt=dt)
        value = ramp.step(torch.zeros(1), torch.ones(1), .2)
        torch.testing.assert_close(value, torch.tensor([.2*dt]))
        stand = Standup(1, 'cpu', dt=dt)
        stand.phase.fill_(SWING)
        for tick in range(round(.1/dt)):
            stand.step(torch.full((1, 2), .18), torch.zeros(1, 2), torch.zeros(1, 2),
                       torch.zeros(1), torch.ones(1), torch.ones(1, dtype=torch.bool))
            if stand.phase.item() == DONE:
                break
        handover = (tick+1)*dt
        assert abs(handover-.05) <= dt+1e-6
        rows.append(dict(dt=dt, pd_output=float(output), ramp_increment=float(value),
                         swing_stable_handover_s=handover))
    report = dict(passed=True, rows=rows, optimizer_updates=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Finite analytic derivative/ramp/50ms dwell checks; no physical behavior. The rollout separately asserts the first 200ms zero motor torque.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
