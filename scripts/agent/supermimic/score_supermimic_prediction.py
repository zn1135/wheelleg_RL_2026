#!/usr/bin/env python3
"""Frozen CPU V13 forecasts on the original sealed V12 comparison windows."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from scripts.agent.supermimic import score_supermimic_motion_retrain as scoring
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, Learner, WORLD_LOSS_RECIPE
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.networks import RunningNorm

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT/'logs/mjlab/chuanliantui_supermimic_walk'
SEALED = ROOT/'artifacts/supermimic-v12-retrain-20261031-r1/score/predictions.pt'
SEALED_SHA = '5e8a4586d205955a11da3782903c0700e2bd97615ed33fad5f30f7b130f5c3db'
ENTRIES = (
    ('V12', '20261031_v12_motiononly_retrain_r1', None),
    ('A170', '20261031_v13_alternating_stage1', 170),
    ('A368', '20261031_v13_alternating_stage2', 368),
    ('A700', '20261101_v13_continuation_to700_r1', 700),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='One additional immutable V13 checkpoint')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    assert scoring.sha(SEALED) == SEALED_SHA
    receipt = json.loads((SEALED.parents[1]/'scoring-exit.json').read_text())
    assert receipt['returncode'] == 0 and receipt['predictions.pt_sha256'] == SEALED_SHA
    sealed = torch.load(SEALED, map_location='cpu', weights_only=False, mmap=True)
    windows = sealed['windows']
    assert len(windows) == 2 and all(len(w['plan']) == 360 for w in windows)
    source = ROOT/'mjlab_training/src/ct_mjlab'
    manifest = {str(p.relative_to(source)):scoring.sha(p) for p in source.rglob('*.py')}
    entries = [(label, LOGS/run/'final.pt', count) for label,run,count in ENTRIES]
    if args.checkpoint:
        extra = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        count = Learner._saved_optimizer_counts(extra)['actor']
        assert count not in [c for _,_,c in entries]
        entries.append((f'A{count}', args.checkpoint.resolve(), count))
    predictions, proofs, checks, results = [], [], [], []
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training allowed')))
        stack.enter_context(patch.object(RunningNorm, 'update', side_effect=AssertionError('No normalization update allowed')))
        for label, path, actor_count in entries:
            saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            digest = scoring.sha(path)
            run_manifest = json.loads((path.parent/'source_sha256.json').read_text())
            assert all(scoring.sha(path.parent/'snapshot/ct_mjlab'/name) == value for name,value in run_manifest.items())
            assert saved['asset_sha256'] == scoring.core.asset_digest()
            assert saved['config']['world_loss_recipe'] == WORLD_LOSS_RECIPE
            counts = Learner._saved_optimizer_counts(saved)
            if actor_count is None:
                assert saved['algorithm_contract'] == scoring.V12
                assert digest == receipt['final_checkpoint_sha256']
            else:
                assert saved['algorithm_contract'] == ALGORITHM_CONTRACT
                assert counts['actor'] == actor_count and counts == saved['optimizer_update_counts']
                assert run_manifest == manifest
            # The motion model and adapter are unchanged across V12 and V13.
            for name in ('supermimic/motion.py', 'supermimic/motion_adapter.py'):
                assert run_manifest[name] == manifest[name]
            world = WorldModel().cpu().eval().requires_grad_(False)
            world.load_state_dict(saved['world'], strict=True)
            before = scoring.core.state_digest(world)
            for index, window in enumerate(windows):
                free = scoring.predict(world, window, False)
                teacher = scoring.predict(world, window, True)
                assert all(scoring.bits(free[k][:,0], teacher[k][:,0]) for k in free)
                for mode,tensors in (('free',free), ('teacher',teacher)):
                    if label == 'V12':
                        old = next(p for p in sealed['predictions'] if p['world_label']=='final'
                            and p['recording_index']==index and p['mode']==mode)
                        # Reproduce the established baseline before trusting new comparisons.
                        assert all(scoring.bits(v,old['tensors'][k]) for k,v in tensors.items())
                    predictions.append(dict(world_label=label,recording_index=index,mode=mode,tensors=tensors))
                    results.append(dict(world_label=label,recording_index=index,mode=mode,
                        **scoring.summarize(tensors,window)))
                checks.append(dict(label=label,recording_index=index,h1_free_teacher_bitwise_equal=True,
                    sealed_baseline_bitwise_reproduced=True if label=='V12' else None))
            assert before == scoring.core.state_digest(world)
            assert scoring.sha(path) == digest
            proofs.append(dict(label=label,checkpoint=str(path),checkpoint_sha256=digest,
                optimizer_counts=counts,world_parameters_and_norm_unchanged=True,
                world_state_sha256=before,source_manifest_sha256=scoring.sha(path.parent/'source_sha256.json')))
            print(json.dumps(dict(event='scored',label=label,elapsed_s=time.monotonic()-started)),flush=True)
    artifact = args.output/'predictions.pt'
    torch.save(dict(windows=windows,predictions=predictions,worlds=proofs,checks=checks),artifact)
    assert scoring.sha(SEALED) == SEALED_SHA
    assert manifest == {str(p.relative_to(source)):scoring.sha(p) for p in source.rglob('*.py')}
    report = dict(worlds=proofs,checks=checks,results=results,source_manifest=manifest,
        input=str(SEALED),input_sha256=SEALED_SHA,predictions_sha256=scoring.sha(artifact),
        scorer_sha256=scoring.sha(__file__),elapsed_s=time.monotonic()-started,
        optimizer_updates=0,normalization_updates=0,actor_forward_calls=0,simulator_steps=0,
        scope='Same two historical V9-policy/V10-capture recordings as V12, 360 windows each. '
              'Motion prediction only; not A700 on-policy trajectories or a policy-gradient accuracy test. '
              'Historical reward labels unchanged and excluded from motion scores. '
              'Teacher means true state supplied each step, not a PPO policy.')
    (args.output/'score.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# V13世界模型：固定轨迹预测比较','',
        '720个历史固定窗口；CPU冻结前向，零训练、零统计更新、零新仿真。误差越低越好。',
        'V12预测逐位复现原封存结果；本测量不覆盖A700自身失稳轨迹。','',
        '| 模型 | 轨迹 | 预测步数 | 前向速度误差 m/s | 姿态误差 ° | 左轮速误差 rad/s | 右轮速误差 rad/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    fields=('LINK_vx_m_s','orientation_geodesic_degrees','left_wheel_velocity_rad_s','right_wheel_velocity_rad_s')
    for row in results:
        if row['mode'] != 'free': continue
        for h in (1,8,24):
            metrics=row['groups']['all'][str(h)]['metrics']
            values=['N/A' if metrics[k]['mae'] is None else f"{metrics[k]['mae']:.6f}" for k in fields]
            lines.append(f"| {row['world_label']} | {row['recording_index']} | {h} | "+' | '.join(values)+' |')
    lines+=['','1/8/24步分别对应10/80/240毫秒，前向速度标签比物理末状态早2毫秒。',
        '完整14项、单步真状态输入对照、分组及24拍误差曲线见score.json；不以预测误差代替行为验收。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='finished',output=str(args.output),elapsed_s=report['elapsed_s'])),flush=True)


if __name__ == '__main__':
    main()
