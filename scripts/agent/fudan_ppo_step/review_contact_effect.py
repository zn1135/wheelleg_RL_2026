"""Recheck frozen v7 GPU evidence and paired flat results; no physics or learning."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .diagnose_reverse import summarize
from .evaluate import score

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(old, new):
    """跨主机三角函数允许微小舍入差；布尔判定、字段和样本必须完全相同。"""
    if isinstance(old, dict):
        assert old.keys() == new.keys()
        return max((compare(old[key], new[key]) for key in old), default=0.)
    if isinstance(old, list):
        assert len(old) == len(new)
        return max((compare(a, b) for a, b in zip(old, new)), default=0.)
    if isinstance(old, float):
        assert math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-5), (old, new)
        return abs(old - new)
    assert old == new, (old, new)
    return 0.


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.input_manifest.read_text())['files']
    # 复算前确认评分实现仍是远程训练时封存的版本。
    sources = ['scripts/agent/fudan_ppo_step/diagnose_reverse.py',
               'scripts/agent/fudan_ppo_step/evaluate.py', 'scripts/agent/motion/assess.py']
    assert all(sha(ROOT / name) == expected[name] for name in sources)
    evidence = args.recording / 'evidence'
    protocols, metrics, roundoff = [], [], []
    for name, checkpoint in [('baseline', args.baseline), ('candidate', args.checkpoint)]:
        folder = evidence / name
        protocol = json.loads((folder / 'protocol.json').read_text())
        audit = json.loads((evidence / f'audit-{name}' / 'audit.json').read_text())
        assert audit['passed'] and protocol['optimizer_updates'] == 0
        assert protocol['all_networks_unchanged'] and protocol['all_finite']
        assert protocol['source_sha256'] == sha(checkpoint)
        assert sha(folder / 'source_diagnose_reverse.py') == protocol['script_sha256']
        step = protocol['recipe']['step']
        rows = []
        for case in protocol['cases']:
            path = folder / 'traces' / (case['id'] + '.npz')
            assert sha(path) == protocol['trace_sha256'][case['id']]
            with np.load(path, allow_pickle=False) as trace:
                row = summarize(case, trace)
                if case['height'] > 0:
                    row['task_score'] = score(dict(id=case['id'], mode='up', height=case['height']),
                        trace, duration=protocol['seconds'], max_approach_speed=protocol['max_approach_speed'],
                        allow_temporary_support=step.get('allow_temporary_support', False),
                        landing_clearance_m=step.get('landing_clearance_m', .45))
            rows.append(row)
        roundoff.append(compare(json.loads((folder / 'metrics.json').read_text()), rows))
        protocols.append(protocol)
        metrics.append(rows)
    assert protocols[0]['cases'] == protocols[1]['cases']
    assert protocols[0]['recipe'] == protocols[1]['recipe']
    assert protocols[0]['source_fingerprint'] == protocols[1]['source_fingerprint']
    differences = []
    for case in protocols[0]['cases']:
        with np.load(evidence / 'baseline/traces' / (case['id'] + '.npz')) as old, \
             np.load(evidence / 'candidate/traces' / (case['id'] + '.npz')) as new:
            delta = float(np.abs(old['initial'] - new['initial']).max())
        differences.append(dict(id=case['id'], height_m=case['height'], max_abs=delta,
                                within_tolerance=delta < 2e-6))
    groups = []
    for height in sorted({r['height'] for r in metrics[0] if r['height'] > 0}):
        group = dict(height_m=height)
        for name, rows in zip(('baseline', 'candidate'), metrics):
            selected = [r for r in rows if r['height'] == height]
            group[name] = dict(total=len(selected), landed=sum(r['landed'] for r in selected),
                task_passed=sum(r['task_score']['passed'] for r in selected),
                strict_parked=sum(r['parked'] for r in selected),
                online_parked=sum(r['stop_reason'] == 'parked' for r in selected))
        groups.append(group)
    flat = json.loads((args.recording / 'local-flat-acceptance/acceptance.json').read_text())
    baseline = json.loads((args.recording / 'local-baseline-acceptance/acceptance.json').read_text())
    for name, local in [('flat47', flat), ('baseline-flat47', baseline)]:
        remote = json.loads((evidence / name / 'acceptance/acceptance.json').read_text())
        roundoff.append(compare(remote['samples'], local['samples']))
        assert local['cohort_sha256'] == remote['cohort_sha256']
        runtime = json.loads((evidence / name / 'runtime-audit.json').read_text())
        assert runtime['passed'] and runtime['optimizer_updates'] == 0 and runtime['complete_traces'] == 47
    assert flat['total'] == baseline['total'] == 47
    assert flat['cohort_sha256'] == baseline['cohort_sha256']
    videos = []
    for folder in sorted(evidence.glob('video-*')):
        media = json.loads((folder / 'media.json').read_text())
        video = folder / 'replay-1x.mp4'
        assert sha(video) == media['video_sha256']
        with imageio.get_reader(video) as reader:
            assert reader.get_meta_data()['fps'] == media['fps']
            frames = sum(1 for _ in reader)
        assert frames == media['frames']
        videos.append(dict(directory=folder.name, case=media['case'], frames=frames,
                           duration_s=media['duration_s'], full_decode_passed=True))
    result = dict(evidence_verified=True, groups=groups, initial_tolerance=2e-6,
        numeric_recompute_tolerance=dict(relative=1e-6, absolute=1e-5,
            max_abs_difference=max(roundoff), discrete_decisions_exact=True),
        initial_differences=differences, all_initials_within_tolerance=all(r['within_tolerance'] for r in differences),
        flat_before=baseline['technical_passed'], flat_after=flat['technical_passed'],
        flat_total=47, flat_comparison=flat['comparison'], videos=videos,
        physics_steps=0, model_forwards=0, optimizer_updates=0, full_task_accepted=False,
        scope='Paired forward 20/50/200mm, three speeds/angles/frictions; not full stair matrix or arena acceptance.')
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(groups=groups, flat_before=result['flat_before'], flat_after=result['flat_after'],
                         all_initials_within_tolerance=result['all_initials_within_tolerance']), ensure_ascii=False))


if __name__ == '__main__':
    main()
