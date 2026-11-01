"""保留线的评分、运动判据与计数检查；不训练、不仿真。"""
import ast
import copy
import importlib
import json
import math
from pathlib import Path
import tarfile
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT/'scripts/archive/20261101-script-cleanup/source.tar.gz'


def original_functions(name, functions, namespace):
    with tarfile.open(ARCHIVE) as bundle:
        source = bundle.extractfile('scripts/agent/' + name + '.py').read().decode()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    assert len(nodes) == len(functions)
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
    return scope


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left.cpu().numpy(), right.cpu().numpy())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right, (left, right)


def rounding():
    from scripts.agent.supermimic.check_supermimic_motion_retrain import norm_count_rounding_proof as check
    old = original_functions('check_supermimic_motion_retrain_counts', ['norm_count_rounding_proof'], {'torch': torch})['norm_count_rounding_proof']
    cases = [(100., 500., 10, 64), (7154886., 35774412., 6992, 1024)]
    for count, delta, steps, samples in cases:
        args = (torch.tensor(count), torch.tensor(delta), steps, samples)
        same(old(*args), check(*args))
    rejected = 0
    for count, delta, steps, samples in [(100., 501., 10, 64), (7154886., 35874412., 6992, 1024),
                                         (100., float('nan'), 10, 64), (100., 500., -1, 64)]:
        for function in (old, check):
            try:
                function(torch.tensor(count), torch.tensor(delta), steps, samples)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError('坏计数被接受')
    assert rejected == 8
    return dict(exact_and_rounded_cases=2, rejected_cases=4, original_implementation_equal=True)


def scoring():
    from scripts.agent.supermimic import _motion_scoring as shared
    saved_path = ROOT / 'artifacts/supermimic-v12-retrain-20261031-r1/score/predictions.pt'
    saved = torch.load(saved_path, map_location='cpu', weights_only=False, mmap=True)
    reports = []
    for name in ('score_supermimic_motion', 'score_supermimic_motion_retrain'):
        module = importlib.import_module('scripts.agent.supermimic.' + name)
        old = original_functions(name, ['predict', 'summarize'], vars(module))
        # 使用实际保存的预测和窗口；保留两个版本不同的指标集合。
        selected = [p for p in saved['predictions'] if p['world_label'] == saved['predictions'][0]['world_label']]
        for entry in selected:
            w = saved['windows'][entry['recording_index']]
            same(old['summarize'](entry['tensors'], w), module.summarize(entry['tensors'], w))
        # 空掩码、非有限预测必须保持原有拒计／计数方式。
        entry = selected[0]; w = copy.deepcopy(saved['windows'][entry['recording_index']])
        w['mask'].zero_(); pred = dict(entry['tensors']); pred['path'] = pred['path'].clone()
        pred['path'][0, 0, 0, 0] = float('nan')
        same(old['summarize'](pred, w), module.summarize(pred, w))
        reports.append(dict(variant=name, saved_prediction_groups=len(selected)))
    # 以保存的真实窗口驱动原/新预测循环；回放适配器不学习、不积分物理。
    window = saved['windows'][0]
    small = dict(plan=window['plan'][:3], mask=window['mask'][:3].clone(),
                 truth={k: v[:3].clone() for k, v in window['truth'].items()})
    small['mask'][1, 8:] = False
    class RecordedWorld:
        adapter = SimpleNamespace(continuation=lambda path: torch.isfinite(path).all(dim=(-2, -1)).float())

        def step(self, state, parameters, action, command, context, previous, previous_previous, noise, return_path):
            # 输出依赖调用传入量，能发现历史动作、命令、噪声传递顺序变化。
            updated = state + action.sum(-1, keepdim=True) * .001 + command.sum(-1, keepdim=True) * .002
            updated = updated + previous.sum(-1, keepdim=True) * .003 + previous_previous.sum(-1, keepdim=True) * .004
            updated = updated + noise.sum(-1, keepdim=True) * .005
            path = updated[..., :64].unsqueeze(1).expand(-1, 5, -1)
            return updated, updated[..., :25], updated.sum(-1), updated[..., 0] > 1., updated[..., :31], path
    for teacher in (False, True):
        for batch in (1, 3):
            same(old['predict'](RecordedWorld(), small, teacher, batch),
                 shared.predict(RecordedWorld(), small, teacher, batch))
    return dict(source=str(saved_path.relative_to(ROOT)), variants=reports,
                prediction_loop_cases=4, empty_mask_and_nonfinite_checked=True,
                scope='保存的预测张量与预测循环等价；未重新评分模型或运行物理。')


def motion():
    """四项统一标准的边界、缺项、完整性与横向漂移检查。"""
    from scripts.agent.motion.assess import metrics_verdict, phase_verdict
    base = dict(vx_mae=.1, yaw_mae=.1, horizontal_peak=.05, pitch_peak=5.,
                complete=True, parking=True)
    assert metrics_verdict(**base)['passed']
    checked = ['inclusive_boundaries']
    for key in ('vx_mae', 'yaw_mae', 'horizontal_peak', 'pitch_peak'):
        assert not metrics_verdict(**{**base, key: np.nextafter(base[key], np.inf)})['passed']
        assert not metrics_verdict(**{**base, key: None})['passed']
        assert not metrics_verdict(**{**base, key: np.nan})['passed']
        assert not metrics_verdict(**{**base, key: -1.})['passed']
        checked.append(key + '_outside_missing_nan_negative')
    assert not metrics_verdict(**{**base, 'complete': False})['passed']
    assert metrics_verdict(**{**base, 'parking': False, 'horizontal_peak': None})['passed']
    data = dict(vx=np.array([-.11, .11]), yaw=np.zeros(2), horizontal_speed=np.zeros(2),
                pitch_deg=np.zeros(2), vx_target=0., yaw_target=0.,
                selected=np.ones(2, dtype=bool), valid=np.ones(2, dtype=bool),
                complete=True, parking=True)
    assert not phase_verdict(**data)['checks']['tracked_vx'], '正负误差不能抵消'
    zero = {**data, 'vx': np.zeros(2)}
    assert phase_verdict(**zero)['passed']
    assert not phase_verdict(**{**zero, 'horizontal_speed': np.array([0., .051])})['passed']
    assert not phase_verdict(**{**zero, 'valid': np.array([True, False])})['passed']
    assert not phase_verdict(**{**zero, 'selected': np.zeros(2, dtype=bool)})['passed']
    assert not phase_verdict(**{**zero, 'yaw': None})['passed']
    assert not phase_verdict(**{**zero, 'pitch_deg': np.array([0., np.nan])})['passed']
    assert not phase_verdict(**{**zero, 'command_vx': np.ones(2)})['passed']
    checked += ['absolute_not_signed_mean', 'lateral_peak', 'early_termination', 'empty_window',
                'missing_yaw', 'nonfinite_sample', 'command_delivery']
    return dict(cases=checked, training_updates=0, physics_steps=0)



def motion_saved():
    """保存轨迹与独立旧统计逐项比较；不加载策略或运行仿真。"""
    import math
    from scripts.agent.motion.assess import supermimic_phase, reevaluate
    directory = ROOT/'artifacts/supermimic-v13-final-seed43-20261031'
    meta = json.loads((directory/'summary.json').read_text())
    old = original_functions('assess_chuanliantui_supermimic', ['phase_metrics'],
                             dict(np=np, math=math, HEIGHT=.26))['phase_metrics']
    compared, maximum_error = 0, 0.
    with np.load(directory/'trajectory.npz', allow_pickle=False) as saved:
        values = saved['values']
        for row in meta['summary']['per_environment']:
            for name, phase in row['phases'].items():
                if name not in ('cruise', 'parking'):
                    continue
                args = (values, meta['layout'], row['env'], phase['start_s'], phase['end_s'],
                        phase['reference_vx_m_s'], meta['physics_dt_s'], meta['seconds'])
                previous = old(*args, yaw_reference=phase['reference_yaw_rad_s'])
                current = supermimic_phase(*args, yaw_reference=phase['reference_yaw_rad_s'])
                for new_key, old_key, stat in [('vx_mae', 'vx_error_m_s', 'mae'),
                        ('yaw_mae', 'yaw_error_rad_s', 'mae'),
                        ('horizontal_peak', 'horizontal_speed_m_s', 'max_abs'), ('pitch_peak', 'pitch_deg', 'max_abs')]:
                    before = None if previous[old_key] is None else previous[old_key][stat]
                    after = current['metrics'][new_key]
                    assert (before is None) == (after is None)
                    if before is not None:
                        maximum_error = max(maximum_error, abs(before-after))
                        assert np.isclose(before, after, rtol=1e-6, atol=1e-7), (name, new_key, before, after)
                assert current['checks']['complete'] == previous['complete']
                compared += 1
    return dict(supermimic_phases_compared=compared, maximum_metric_error=maximum_error, training_updates=0, physics_steps=0)



def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-only', action='store_true')
    parser.add_argument('--saved-motion', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(motion=motion())
    if not args.motion_only:
        result['motion_saved'] = motion_saved()
    if not args.motion_only and not args.saved_motion:
        result.update(rounding=rounding(), scoring=scoring())
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
