"""比较固定2m/s候选与4200的GPU跟速、停车和登台。"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import diagnose_reverse, evaluate_fixed200_turn_effect as evaluation
from .launch_training import ROOT, sha, write
from .prepare_fixed_speed import SOURCE, SOURCE_SHA


def straight_items(*_):
    rows = []
    for height in (.26, .35):
        name = f'stop_forward_v2_h{height:.2f}'
        segments = [dict(start=0., duration=7., command=[2., 0., height]),
                    dict(start=7., duration=8., command=[0., 0., height])]
        rows.append(dict(id=name, case=dict(id=name, category='stop_forward', segments=segments, duration=15.),
                         seed=50001, randomized=False))
    return rows


def stairs(checkpoint, path, deadline):
    # 使用4200验收的台阶布置，两模型在同一场景比较。
    original = diagnose_reverse.FudanStepCfg
    def old_layout():
        cfg = original()
        cfg.step.patch_size = (10., 4.)
        cfg.step.flat_forward_limit_m = 3.5
        return cfg
    path.mkdir()
    request = SimpleNamespace(output=path, checkpoint=checkpoint, command_height=.35,
        label=checkpoint.parent.name, seconds=20., deadline=deadline, heights=[.2],
        speeds=[.3, 1., 2., 2.5], angles=[0., -10., 10.], directions=['forward'],
        friction=.8, high_speed=True, frictions=[.4, .8, 1.2])
    with patch.object(diagnose_reverse, 'FudanStepCfg', old_layout):
        diagnose_reverse.run(request)
    audit = path/'audit'
    audit.mkdir()
    diagnose_reverse.audit(SimpleNamespace(recording=path, output=audit))


def run(output, *, audit_path=None, checkpoint=None):
    import torch
    from scripts.agent.motion.acceptance import canonical, compare_baseline
    if audit_path is not None:
        audit = json.loads(audit_path.read_text())
        assert audit['passed'] and audit['iterations'] == 5200
        checkpoint = Path(audit['checkpoint'])
        assert sha(checkpoint) == audit['checkpoint_sha256']
    checkpoint = checkpoint.resolve()
    digest = sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    tick = saved['iterations']
    step = saved['training_recipe']['step']
    assert 4200 < tick <= 5200 and not saved['partial_update']
    assert saved['num_envs'] == 4096 and saved['transitions'] == tick*4096*48
    assert saved['primary_updates'] == saved['encoder_updates'] == tick*20
    assert step['version'] == 'fixed200-straight-2mps-v10' and step['approach_speed'] == [2., 2.]
    assert sha(SOURCE) == SOURCE_SHA
    candidate = str(tick)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time()+1800
    models = (('4200', SOURCE), (candidate, checkpoint))
    for iteration, model in models:
        with patch.dict(evaluation.MODELS, {iteration: model}), patch.object(evaluation, 'turn_items', straight_items):
            evaluation.turns(SimpleNamespace(model=iteration, output=output/f'flat-{iteration}', deadline=deadline))
    for iteration, model in models:
        stairs(model, output/f'stair-{iteration}', deadline)
    straight = [json.loads((output/f'flat-{i}/acceptance.json').read_text()) for i in ('4200', candidate)]
    assert straight[0]['cohort_sha256'] == straight[1]['cohort_sha256'] and straight[0]['cohort_sha256']

    def stair_data(iteration):
        folder = output/f'stair-{iteration}'
        rows = {row['id']: row for row in json.loads((folder/'metrics.json').read_text()) if row['height'] == .2}
        return folder, rows
    (before, old), (after, new) = [stair_data(i) for i in ('4200', candidate)]
    assert old.keys() == new.keys() and len(old) == 36
    max_initial_difference = 0.
    for case in old:
        with np.load(before/'traces'/f'{case}.npz') as a, np.load(after/'traces'/f'{case}.npz') as b:
            max_initial_difference = max(max_initial_difference, float(abs(a['initial']-b['initial']).max()))
    assert max_initial_difference < 2e-6
    old_pass = {key for key, val in old.items() if val['task_score']['passed']}
    new_pass = {key for key, val in new.items() if val['task_score']['passed']}
    flat47 = output/'flat47'
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
    with patch.object(inspect, 'MODEL', checkpoint), patch.object(inspect, 'RUN', checkpoint.parent), \
         patch('torch.optim.Adam.step', side_effect=AssertionError('Evaluation cannot learn')):
        inspect.run(flat47, checkpoint_audit=lambda _: dict(sha256=digest, budget_verified=audit_path is not None))
    baseline = ROOT/'artifacts/fudan_ppo_step/20261118-v8-effect4200-01/flat47/acceptance/acceptance.json'
    flat_acceptance = compare_baseline(canonical(flat47/'gpu', 'fudan_ppo_walk'), json.loads(baseline.read_text()))
    write(flat47/'acceptance.json', flat_acceptance)
    def flat_metrics(iteration):
        folder = output/f'flat-{iteration}'
        rows = json.loads((folder/'metrics.json').read_text())
        for row in rows:
            with np.load(folder/'traces'/f"{row['id']}.npz") as trace:
                mask = (trace['time'] >= 2.) & (trace['time'] < 7.)
                row['actual_vx_mean'] = float(trace['linear_body'][mask, 0].mean())
        return rows
    assert sha(checkpoint) == digest
    write(output/'review.json', dict(checkpoint_sha256=digest, iterations=tick,
        final_budget_verified=audit_path is not None, optimizer_updates=0,
        flat_2mps={i: flat_metrics(i) for i in ('4200', candidate)},
        stair_by_speed={str(v): {i: sum(row['speed']==v and row['task_score']['passed'] for row in rows.values())
                                 for i, rows in (('4200', old), (candidate, new))}
                        for v in (.3, 1., 2., 2.5)},
        stair_lost=sorted(old_pass-new_pass), stair_gained=sorted(new_pass-old_pass),
        stair_initial_max_difference=max_initial_difference, flat47_passed=flat_acceptance['technical_passed'],
        flat47_lost=flat_acceptance['comparison']['regressed'], full_task_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--audit', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Evaluate a complete intermediate checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, audit_path=args.audit, checkpoint=args.checkpoint)
