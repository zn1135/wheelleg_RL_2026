"""Launch the four authorized remote experiments behind one explicit start gate.

Use the mjlab Python runtime. Creates a fresh campaign; never reuses run paths.
The coordinator checks ready.json hashes and memory before creating START.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'worker'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--group', choices=('baseline', 'wide_height', 'mode_reward', 'recover_timeout'))
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(release / 'mjlab_training/src'),
               MUJOCO_GL='disable', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('WHEELLEG_HUANANHU_INIT_CHECKPOINT', None)
    if args.command == 'worker':
        if args.group is None:
            parser.error('--group required for worker')
        command = [sys.executable, '-u', '-m', 'ct_mjlab.experiments.recovery_ablation',
                   'train', '--group', args.group, '--run-dir', str(args.campaign / args.group),
                   '--start-gate', str(args.campaign / 'START'), '--num-envs', str(args.num_envs),
                   '--iterations', str(args.iterations), '--seed', str(args.seed)]
        child = subprocess.Popen(command, cwd=release, env=env)
        (args.campaign / f'{args.group}.pid').write_text(str(child.pid) + '\n')
        code = child.wait()
        (args.campaign / f'{args.group}.exit-code').write_text(str(code) + '\n')
        sys.exit(code)
    args.campaign.mkdir(parents=True, exist_ok=False)
    pids = {}
    for group in ('baseline', 'wide_height', 'mode_reward', 'recover_timeout'):
        command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                   '--campaign', str(args.campaign), '--group', group,
                   '--num-envs', str(args.num_envs), '--iterations', str(args.iterations),
                   '--seed', str(args.seed)]
        with (args.campaign / f'{group}.console.log').open('xb') as output:
            process = subprocess.Popen(command, cwd=release, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        pids[group] = process.pid
    record = dict(release=str(release), campaign=str(args.campaign),
                  supervisor_pids=pids, num_envs=args.num_envs,
                  iterations=args.iterations, seed=args.seed, initial='identical random weights',
                  gate=str(args.campaign / 'START'))
    (args.campaign / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
