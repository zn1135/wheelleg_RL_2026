"""Check native CLI velocity commands and model-specific control ranges; no training."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [('walk_forward', ['--command', '3', '0', '.26'], (3., 0., .26), (3., 2.), 6838),
             ('walk_backward', ['--command', '-3', '0', '.26'], (-3., 0., .26), (3., 2.), 6838),
             ('walk_high_yaw', ['--command', '3', '10', '.26'], (0., 10., .26), (3., 2.), 6838),
             ('walk_spin', ['--spin'], (0., 10., .26), (3., 2.), 6838),
             ('step_defaults', ['--step-mode'], (0., 0., .35), (1., .5), 3200),
             ('step_upper', ['--step-mode', '--command', '2.5', '2', '.35'], (2.5, 2., .35), (1., .5), 3200)]
    results = []
    for name, flags, expected, keyboard, iteration in cases:
        folder = args.output / name
        command = [str(ROOT / 'scripts/rmuc2027.sh'), '--terrain', 'steps', '--viewer', 'none',
                   '--seconds', '.1', '--output', str(folder), *flags]
        with (args.output / (name + '.log')).open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ), stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        startup = json.loads((folder / 'startup.json').read_text())
        report = json.loads((folder / 'report.json').read_text())
        assert (startup['keyboard_speed'], startup['keyboard_yaw']) == keyboard
        assert startup['iterations'] == iteration and report['error'] is None
        with np.load(folder / 'trace.npz') as trace:
            assert len(trace['state']) == 10 and np.isfinite(trace['state']).all()
            np.testing.assert_allclose(trace['state'][:, 1:4], np.tile(expected, (10, 1)), atol=1e-7)
        results.append(dict(case=name, iterations=iteration, command=expected, keyboard=keyboard,
                            actual_command_verified=True, policy_steps=10, finite=True))
    bad = subprocess.run([str(ROOT / 'scripts/rmuc2027.sh'), '--terrain', 'steps', '--step-mode',
        '--spin', '--viewer', 'none', '--seconds', '.1'], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and '--spin requires' in bad.stderr
    (args.output / 'step-spin-rejected.log').write_text(bad.stdout + bad.stderr)
    result = dict(passed=True, cases=results, step_spin_rejected=True, optimizer_updates=0,
        formal_acceptance=False, scope='Requested command routing and finite native CLI checks; not achieved-speed or stair acceptance')
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=len(results), optimizer_updates=0)))


if __name__ == '__main__':
    main()
