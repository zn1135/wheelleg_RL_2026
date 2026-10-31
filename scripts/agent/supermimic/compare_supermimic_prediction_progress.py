"""Compare A700/A1808/A2200 on identical windows with fixed A2200 loss scales."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, Learner, asset_digest
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.networks import RunningNorm
from scripts.agent.supermimic import score_supermimic_motion_retrain as scoring
from scripts.agent.supermimic.score_supermimic_prediction_details import overall_errors

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT/'artifacts'
BASE = ART/'supermimic-v13-a2200-prediction-20261101-r1'


def load_sealed(directory):
    report = json.loads((directory/'score.json').read_text())
    assert scoring.sha(directory/'predictions.pt') == report['predictions_sha256']
    return torch.load(directory/'predictions.pt', map_location='cpu', weights_only=False, mmap=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    latest = load_sealed(BASE/'historical')
    mid = load_sealed(ART/'supermimic-v13-current-prediction-20261101-r1')
    for a, b in zip(latest['windows'], mid['windows']):
        assert a['plan'] == b['plan'] and scoring.bits(a['mask'], b['mask'])
        assert all(scoring.bits(a['truth'][k], b['truth'][k]) for k in a['truth'])
    current_report = json.loads((BASE/'details-overall/details.json').read_text())
    current_path = BASE/'details-overall/current-predictions.pt'
    assert scoring.sha(current_path) == current_report['current_predictions_sha256']
    current = torch.load(current_path, map_location='cpu', weights_only=False, mmap=True)
    reference = WorldModel().cpu().eval().requires_grad_(False)
    reference_path = Path(current_report['checkpoint'])
    assert scoring.sha(reference_path) == current_report['checkpoint_sha256']
    reference.load_state_dict(torch.load(reference_path, map_location='cpu', weights_only=False, mmap=True)['world'], strict=True)
    rows, proofs, current_predictions = [], [], []
    source = ROOT/'mjlab_training/src/ct_mjlab'
    manifest = {str(p.relative_to(source)):scoring.sha(p) for p in source.rglob('*.py')}
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')))
        stack.enter_context(patch.object(RunningNorm, 'update', side_effect=AssertionError('No norm update')))
        for label in ('A700', 'A1808', 'A2200'):
            sealed = mid if label == 'A1808' else latest
            proof = next(w for w in sealed['worlds'] if w['label'] == label)
            checkpoint = Path(proof['checkpoint'])
            assert scoring.sha(checkpoint) == proof['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            assert saved['algorithm_contract'] == ALGORITHM_CONTRACT and saved['asset_sha256'] == asset_digest()
            assert Learner._saved_optimizer_counts(saved)['actor'] == int(label[1:])
            assert json.loads((checkpoint.parent/'source_sha256.json').read_text()) == manifest
            for name, sha in manifest.items():
                assert scoring.sha(checkpoint.parent/'snapshot/ct_mjlab'/name) == sha
            world = WorldModel().cpu().eval().requires_grad_(False)
            world.load_state_dict(saved['world'], strict=True)
            before = scoring.core.state_digest(world)
            for dataset, window in [('current_A2200_seed55', current['windows'][0])]+[(f'historical_{i}',w) for i,w in enumerate(latest['windows'])]:
                if dataset.startswith('current'):
                    pred = scoring.predict(world, window, False)
                    if label == 'A2200':
                        old = next(p['tensors'] for p in current['predictions'] if p['mode']=='free')
                        assert all(scoring.bits(pred[k], old[k]) for k in pred)
                    current_predictions.append(dict(label=label, tensors=pred))
                else:
                    index = int(dataset[-1])
                    pred = next(p['tensors'] for p in sealed['predictions'] if p['world_label']==label and p['mode']=='free' and p['recording_index']==index)
                rows.append(dict(model=label, dataset=dataset, overall=overall_errors(pred, window, reference)))
            assert scoring.core.state_digest(world) == before
            proofs.append(proof)
            print(json.dumps(dict(scored=label)), flush=True)
    torch.save(dict(window=current['windows'][0], predictions=current_predictions), args.output/'current-predictions.pt')
    report = dict(results=rows, checkpoints=proofs, fixed_scale_checkpoint_sha256=current_report['checkpoint_sha256'],
        fixed_delta_std=reference.delta_norm.std().tolist(), metric='Exact world_loss_terms, all46 coordinates and five substeps, normalized MSE averaged over horizon',
        source_manifest=manifest, scorer_sha256=scoring.sha(__file__), parameter_updates=0, normalization_updates=0,
        simulator_steps=0, actor_calls=0, A2200_current_predictions_bitwise_reproduced=True,
        predictions_sha256=scoring.sha(args.output/'current-predictions.pt'),
        limits='Current dataset comes from A2200 and may favor it; historical recordings provide separate coverage. No generalization or gradient-accuracy claim.')
    (args.output/'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 预测模型进步对照', '', '同一数据窗口，统一使用A2200归一化尺度；数值为整段240ms归一化均方误差，越低越好。',
        '当前轨迹由A2200策略采集，可能偏向A2200；另外保留两套历史轨迹分别比较。', '',
        '| 数据集 | A700 | A1808 | A2200 |', '|---|---:|---:|---:|']
    for dataset in ('current_A2200_seed55','historical_0','historical_1'):
        vals = [next(r for r in rows if r['model']==label and r['dataset']==dataset)['overall']['24']['normalized_mse'] for label in ('A700','A1808','A2200')]
        lines.append('| '+dataset+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', '零训练、零统计更新、零新仿真；A2200当前轨迹预测逐位复现先前结果。', '该指标不包含模型未输出的接触力/承重；不能代替控制行为验收。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(finished=True, results=rows)), flush=True)


if __name__ == '__main__':
    main()
