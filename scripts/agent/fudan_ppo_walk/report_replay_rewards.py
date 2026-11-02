"""Reconstruct nine state-derived reward terms from existing diagnostic recordings."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnose_body_support import diagnose


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.run / 'source/ct_mjlab/fudan_ppo_walk'
    manifest = json.loads((args.run / 'source_sha256.json').read_text())
    for name in ('env.py', 'contract.py'):
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        assert digest == manifest['mjlab_training/src/ct_mjlab/fudan_ppo_walk/' + name]
    tree = ast.parse((source / 'contract.py').read_text())
    weights = next({item.arg: ast.literal_eval(item.value) for item in node.value.keywords}
        for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'WEIGHTS' for target in node.targets))
    protocol = json.loads((args.recording / 'protocol.json').read_text())
    rows = []
    for sample in protocol['samples']:
        path = args.recording / 'traces' / (sample['id'] + '.npz')
        # 停车取发零命令后3～8秒的固定窗口，只作奖励诊断。
        lo = 10. if sample['case']['category'].startswith('stop_') else 2.
        with np.load(path) as trace:
            row = diagnose(trace, sample['case'], lo, sample['case']['duration'])
        row.update(sample_id=sample['id'], case=sample['case'],
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    report = dict(physics_steps=0, optimizer_updates=0, model_forwards=0,
        scope='Replay reconstruction, not original per-term training logs or full16 reward',
        weights=weights, policy_dt=.01, per_term_clamp=[-.01, .01], samples=rows,
        full_reward_available=False, reconstructed_terms=9,
        unavailable_terms=rows[0]['reward']['unavailable_terms'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        if row['sample_id'] in ('diagnostic_0', 'diagnostic_5', 'diagnostic_15', 'diagnostic_19'):
            print(json.dumps(dict(sample=row['sample_id'], reward=row['reward'])))


if __name__ == '__main__':
    main()
