"""Frozen motion, geometry, observation and reward errors on saved recordings.

Current recordings are verified against their own contract. Historical motion
windows retain their original labels; reward comparisons use the current formula
on both predicted and true features, not the historical stored reward total.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from mjlab.utils.lab_api.math import quat_mul
from ct_mjlab.supermimic.analytic_reward import WEIGHTS, differentiable_reward
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, Learner, asset_digest
from ct_mjlab.supermimic.motion import WorldModel, physical47_from_full64
from ct_mjlab.supermimic.networks import RunningNorm
from ct_mjlab.supermimic.state import rotation_vector, unit_quat
from scripts.agent.supermimic import score_supermimic_motion_retrain as motion
from scripts.agent.supermimic import score_supermimic_physical_prediction as core

ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def euler(q):
    w, x, y, z = unit_quat(q).unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
        torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))), -1)


def arrays(pred, truth, world):
    result = motion.motion_arrays(pred, truth)
    p, t = pred['path'][..., -1, :], truth['path'][..., -1, :]
    pf, tf = pred['features'], truth['features']
    for i, axis in enumerate('xyz'):
        result[f'root_velocity_{axis}_m_s'] = p[..., 25+i], t[..., 25+i]
        result[f'root_angular_velocity_{axis}_rad_s'] = p[..., 28+i], t[..., 28+i]
        result[f'LINK_velocity_{axis}_m_s'] = pf[..., i], tf[..., i]
        result[f'LINK_angular_velocity_{axis}_rad_s'] = pf[..., 3+i], tf[..., 3+i]
        result[f'gravity_{axis}'] = pf[..., 6+i], tf[..., 6+i]
    pe, te = euler(p[..., 3:7]), euler(t[..., 3:7])
    # Wrap orientation differences across +/-pi before computing errors.
    error = torch.atan2(torch.sin(pe-te), torch.cos(pe-te))*180/math.pi
    for i, name in enumerate(('roll', 'pitch', 'yaw')):
        result[f'{name}_error_deg'] = error[..., i], torch.zeros_like(error[..., i])
    for i, name in enumerate(world.adapter.kin.joint_names[1:]):
        slide = 7+i in (11, 16)
        if 7+i not in (9, 14):
            scale = 1 if slide else 180/math.pi
            result[f'joint_{name}_position_'+('m' if slide else 'deg')] = p[..., 7+i]*scale, t[..., 7+i]*scale
        result[f'joint_{name}_velocity_'+('m_s' if slide else 'rad_s')] = p[..., 31+i], t[..., 31+i]
    result['reward_height_m'] = pf[..., 9], tf[..., 9]
    for i, side in enumerate(('left', 'right')):
        result[f'{side}_leg_swing_deg'] = pf[..., 10+i]*180/math.pi, tf[..., 10+i]*180/math.pi
    result['leg_swing_difference_deg'] = (pf[..., 10]-pf[..., 11])*180/math.pi, (tf[..., 10]-tf[..., 11])*180/math.pi
    result['leg_swing_abs_difference_deg'] = (pf[..., 10]-pf[..., 11]).abs()*180/math.pi, (tf[..., 10]-tf[..., 11]).abs()*180/math.pi
    lengths = []
    for path in (pred['path'], truth['path']):
        lagged = path[..., -2, :]
        pos, _, _, _ = world.adapter.kin(lagged[..., :25], lagged[..., 25:49])
        lengths.append((pos[..., world.adapter.wheels, :]-pos[..., world.adapter.hips, :]).norm(dim=-1))
    for i, side in enumerate(('left', 'right')):
        result[f'{side}_leg_length_m'] = lengths[0][..., i], lengths[1][..., i]
    for i, name in enumerate(('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')):
        # These are analytic PD estimates, not learned torque outputs.
        result[f'aux_estimated_torque_{name}_Nm'] = pf[..., 22+i], tf[..., 22+i]
    for i in range(25):
        result[f'actor_frame_{i:02d}_scaled'] = pred['frame'][..., i], truth['next_frame'][..., i]
    # A terminal environment frame may already be reset; exclude such labels.
    for i in range(25):
        pp, tt = result[f'actor_frame_{i:02d}_scaled']
        result[f'actor_frame_{i:02d}_scaled'] = pp.masked_fill(truth['done'], float('nan')), tt.masked_fill(truth['done'], float('nan'))
    pv, pr = differentiable_reward(pf, truth['action'], truth['previous'], truth['previous_previous'], truth['context'])
    tv, tr = differentiable_reward(tf, truth['action'], truth['previous'], truth['previous_previous'], truth['context'])
    for name, weight in WEIGHTS.items():
        factor = weight*truth['context'][..., 20]
        result['reward_term_'+name] = pv[name]*factor, tv[name]*factor
    result['reward_total_current_formula'] = pr, tr
    return result


def summarize(pred, window, world):
    pairs = arrays(pred, window['truth'], world)
    mask = window['mask']
    groups = {'all': torch.ones(len(mask), dtype=torch.bool)}
    for field in ('case', 'phase'):
        for value in sorted({p[field] for p in window['plan']}):
            groups[field+'/'+value] = torch.tensor([p[field] == value for p in window['plan']])
    result = {}
    for group, selected in groups.items():
        endpoints = {}
        for h in (1, 8, 24):
            valid = mask[selected, h-1]
            metrics = {}
            for name, (p, t) in pairs.items():
                pp, tt = p[selected, h-1], t[selected, h-1]
                vm = valid.clone()
                if name.startswith('actor_frame_'):
                    vm &= ~window['truth']['done'][selected, h-1]
                if pp.ndim > 1:
                    vm = vm[:, None].expand_as(pp)
                metrics[name] = core.stats(pp.flatten(), tt.flatten(), vm.flatten())
            endpoints[str(h)] = dict(complete_windows=int(valid.sum()), metrics=metrics)
        result[group] = endpoints
    return result


def overall_errors(pred, window, world):
    # Reuse the exact 46-coordinate, five-substep training loss; no extra weight.
    from types import SimpleNamespace
    terms = Learner.world_loss_terms(SimpleNamespace(world=world), window['truth']['path'], pred['path'])
    per_step = terms.double().mean(dim=(-1, -2))
    output = {}
    for h in (1, 8, 24):
        mask = window['mask'][:, :h]
        values = per_step[:, :h][mask]
        assert torch.isfinite(values).all()
        mse = float(values.mean())
        output[str(h)] = dict(normalized_mse=mse, normalized_rmse=math.sqrt(mse),
            valid_policy_steps=len(values), windows=int(mask[:, 0].sum()),
            duration_s=h*.01, includes_all_substeps=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['algorithm_contract'] == ALGORITHM_CONTRACT and saved['asset_sha256'] == asset_digest()
    counts = Learner._saved_optimizer_counts(saved)
    label = f"A{counts['actor']}"
    manifest = json.loads((checkpoint.parent/'source_sha256.json').read_text())
    for name, sha in manifest.items():
        assert core.digest(ROOT/'mjlab_training/src/ct_mjlab'/name) == sha
        assert core.digest(checkpoint.parent/'snapshot/ct_mjlab'/name) == sha
    checkpoint_sha = core.digest(checkpoint)
    meta = json.loads((args.recording/'record.json').read_text())
    receipt = json.loads((args.recording/'execution-exit.json').read_text())
    assert receipt['returncode'] == 0 and receipt['record_sha256'] == core.digest(args.recording/'record.json')
    assert meta['completed'] and meta['checkpoint_sha256'] == checkpoint_sha
    assert meta['algorithm_contract'] == ALGORITHM_CONTRACT and meta['asset_sha256'] == asset_digest()
    assert meta['parameter_updates'] == meta['normalization_updates'] == 0
    assert meta['model_state_sha256_before'] == meta['model_state_sha256_after']
    assert meta['transitions_sha256'] == core.digest(args.recording/'transitions.pt') == receipt['transitions_sha256']
    assert meta['initial_sha256'] == core.digest(args.recording/'initial.npz')
    assert meta['source_snapshot_sha256'] == core.digest(args.recording/'source-sha256.json')
    for name, sha in json.loads((args.recording/'source-sha256.json').read_text()).items():
        assert core.digest(args.recording/'snapshot'/name) == sha
    record = torch.load(args.recording/'transitions.pt', map_location='cpu', weights_only=False, mmap=True)
    window = core.make_windows(dict(data=record['data'], live=record['live_before'], proof=dict(metadata=meta)))
    historic_meta = json.loads((args.historical/'score.json').read_text())
    assert core.digest(args.historical/'predictions.pt') == historic_meta['predictions_sha256']
    historic = torch.load(args.historical/'predictions.pt', map_location='cpu', weights_only=False, mmap=True)
    assert next(w for w in historic['worlds'] if w['label'] == label)['checkpoint_sha256'] == checkpoint_sha
    world = WorldModel().cpu().eval().requires_grad_(False)
    world.load_state_dict(saved['world'], strict=True)
    before = core.state_digest(world)
    predictions, results = [], []
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')))
        stack.enter_context(patch.object(RunningNorm, 'update', side_effect=AssertionError('No norm update')))
        for teacher in (False, True):
            mode = 'teacher' if teacher else 'free'
            pred = motion.predict(world, window, teacher)
            predictions.append(dict(mode=mode, tensors=pred))
            if teacher:
                assert all(motion.bits(pred[k][:, 0], predictions[0]['tensors'][k][:, 0]) for k in pred)
            results.append(dict(dataset='current_A2200_seed55', mode=mode,
                overall=overall_errors(pred, window, world), groups=summarize(pred, window, world)))
            print(json.dumps(dict(scored='current', mode=mode)), flush=True)
        for index, old_window in enumerate(historic['windows']):
            for mode in ('free', 'teacher'):
                pred = next(p['tensors'] for p in historic['predictions'] if p['world_label'] == label and p['recording_index'] == index and p['mode'] == mode)
                results.append(dict(dataset=f'historical_{index}', mode=mode,
                    overall=overall_errors(pred, old_window, world), groups=summarize(pred, old_window, world)))
    assert core.state_digest(world) == before and core.digest(checkpoint) == checkpoint_sha
    torch.save(dict(windows=[window], predictions=predictions), args.output/'current-predictions.pt')
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, optimizer_counts=counts,
        recording_metadata=meta, source_manifest=manifest, world_unchanged=True,
        scoring_optimizer_updates=0, scoring_normalization_updates=0, results=results,
        current_predictions_sha256=core.digest(args.output/'current-predictions.pt'),
        notes=['H1/H8/H24 = 10/80/240ms physical endpoint; reward geometry is 2ms earlier.',
            'teacher: true state supplied every step; free: predicted state fed back. Both use recorded actions.',
            'Historical rewards recomputed with the current formula on both sides; stored historical totals are not compared.',
            'Force/contact/load and cumulative wheel angle are not model outputs and have no prediction error score.',
            'Torque is a PD approximation; reward, observations, geometry and failure are derived, not independent learned heads.',
            'New seed55/24-env recording is distinct from the seed43/96-env behavior acceptance.'])
    write(args.output/'details.json', report)
    lines = ['# A2200 全量预测误差', '', 'MAE：平均绝对误差；全部数值按各自单位，越小越好。完整偏差、RMSE、P95及最大误差见 details.json。',
        '', '本次新采24环境/seed55；历史两组各360窗口。所有模型与归一化冻结。',
        '1/8/24步对应10/80/240ms；teacher是每步重新输入真实状态；free为连续预测。',
        '轮承重/接触力/累计轮角不在模型输出中；力矩为解析近似，不能当作学习输出。', '']
    overall = next(r['overall'] for r in results if r['dataset']=='current_A2200_seed55' and r['mode']=='free')
    lines += [f"**当前整体误差：{overall['24']['normalized_mse']:.6g}**。按训练相同的归一化均方误差，汇总46个运动误差坐标、每拍5个物理子步、360窗口全部24拍（240ms）。越低越好，不是百分比。", '',
        '该整体指标包含机身与腿部姿态、位置及速度；不重复计入由同一状态推导的奖励或观测，也不含模型没有输出的承重/接触力。', '']
    for dataset in dict.fromkeys(r['dataset'] for r in results):
        free = next(r for r in results if r['dataset']==dataset and r['mode']=='free')['groups']['all']
        teacher = next(r for r in results if r['dataset']==dataset and r['mode']=='teacher')['groups']['all']
        lines += [f'## {dataset}', '', '| 指标（单位见名称） | 10ms MAE | 80ms MAE | 240ms MAE | 每步真状态输入，末步MAE | 240ms P95 |', '|---|---:|---:|---:|---:|---:|']
        for name in free['1']['metrics']:
            vals = [free[str(h)]['metrics'][name]['mae'] for h in (1,8,24)]
            vals += [teacher['24']['metrics'][name]['mae'], free['24']['metrics'][name]['p95_abs']]
            lines.append('| '+name+' | '+' | '.join('N/A' if v is None else f'{v:.6g}' for v in vals)+' |')
        lines.append('')
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(finished=True, output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
