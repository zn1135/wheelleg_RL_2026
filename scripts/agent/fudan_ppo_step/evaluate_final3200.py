"""Evaluate the completed v7 checkpoint using its frozen training source; no learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / 'mjlab_training/.venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def check_source(source, manifest):
    files = json.loads(manifest.read_text())['files']
    for name, expected in files.items():
        assert sha(source / name) == expected, name
    return len(files)


def flat(args):
    import torch
    from unittest.mock import patch
    from scripts.agent.fudan_ppo_step.inspect_training import source_checks
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
    checks = source_checks(args.checkpoint.parent)
    inspect.MODEL = args.checkpoint
    inspect.RUN = args.checkpoint.parent
    # 直接复用原47例、初态与逐物理步PD审计，禁止优化器更新。
    with patch('torch.optim.Adam.step', side_effect=AssertionError('Evaluation cannot train')):
        inspect.run(args.output, checkpoint_audit=lambda _: checks)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    count = check_source(args.source, args.manifest)
    for name in ('acceptance.py', 'assess.py'):
        assert sha(ROOT / 'scripts/agent/motion' / name) == sha(args.source / 'scripts/agent/motion' / name)
    import torch
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert saved['iterations'] == 3200 and not saved['partial_update']
    assert saved['primary_updates'] == saved['encoder_updates'] == 64000
    assert saved['transitions'] == 629145600
    assert all(torch.isfinite(v).all() for v in saved['model_state_dict'].values())
    receipt = dict(checkpoint_sha256=sha(args.checkpoint), source_files_verified=count,
                   checkpoint_iteration=3200, std=saved['model_state_dict']['std'].tolist(),
                   source=str(args.source), optimizer_updates=0)
    write(args.output / 'input-receipt.json', receipt)
    env = dict(os.environ, PYTHONPATH=str(args.source / 'mjlab_training/src') + ':' + str(args.source),
               MUJOCO_GL='egl', OMP_NUM_THREADS='1')

    def execute(argv, label):
        with (args.output / (label + '.log')).open('x') as stream:
            subprocess.run([str(PYTHON), *argv], cwd=args.source, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, args.deadline - time.time()))

    # 主矩阵与200轮完全同工况；另一组与1300轮39例完全同工况。
    for name, heights, frictions in [('core', ['.02', '.05', '.20'], ['.4', '.8', '1.2']),
                                     ('nominal', ['.05', '.08', '.12', '.20'], ['.8'])]:
        execute(['-m', 'scripts.agent.fudan_ppo_step.diagnose_reverse', 'run',
                 '--checkpoint', str(args.checkpoint), '--output', str(args.output / name),
                 '--label', 'V7_3200', '--command-height', '.35', '--high-speed',
                 '--heights', *heights, '--speeds', '.3', '1', '2.5', '--angles', '0', '-10', '10',
                 '--directions', 'forward', '--frictions', *frictions, '--seconds', '20',
                 '--deadline', str(args.deadline)], name)
        execute(['-m', 'scripts.agent.fudan_ppo_step.diagnose_reverse', 'audit',
                 '--recording', str(args.output / name), '--output', str(args.output / ('audit-' + name))],
                'audit-' + name)
    execute([str(Path(__file__).resolve()), 'flat', '--checkpoint', str(args.checkpoint),
             '--output', str(args.output / 'flat47')], 'flat47')
    baseline = ROOT / 'artifacts/fudan_ppo_step/20261117-v7-effect200-01/local-baseline-acceptance/acceptance.json'
    with (args.output / 'flat-accept.log').open('x') as stream:
        subprocess.run([str(ROOT / 'scripts/mjlab.sh'), 'accept', '--task', 'fudan_ppo_walk',
                        '--recording', str(args.output / 'flat47/gpu'),
                        '--output', str(args.output / 'flat47/acceptance'), '--baseline', str(baseline)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert sha(args.checkpoint) == receipt['checkpoint_sha256']
    write(args.output / 'execution.json', dict(passed=True, checkpoint_unchanged=True,
          optimizer_updates=0, backend='mujoco_warp_gpu', source_files_verified=count,
          matrices=dict(core=90, nominal=39), flat_cases=47, full_task_accepted=False))
    print(json.dumps(dict(output=str(args.output), collection_and_audit_passed=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('run', 'flat'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.mode == 'run':
        args.source = args.source.resolve()
        args.manifest = args.manifest.resolve()
    {'run': run, 'flat': flat}[args.mode](args)


if __name__ == '__main__':
    main()
