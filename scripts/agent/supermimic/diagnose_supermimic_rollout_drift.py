#!/usr/bin/env python3
"""Frozen V10: true-state one-tick prediction versus a 24-tick rollout on recorded actions.

No environment, actor, critic, optimizer or normalization fitting is run.
The earlier sealed free rollout must be reproduced exactly before comparison.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from scripts.agent.supermimic import score_supermimic_physical_prediction as core


def teacher_predict(world, windows, batch):
    """Use the existing runner, replacing only its incoming full state each tick."""
    original_step = world.step
    first, tick, calls = 0, 0, 0

    def true_state_step(state, *args, **kwargs):
        nonlocal first, tick, calls
        count = len(state)
        # Reset physical state, sensor history and task memory to the same real tick.
        incoming = torch.where(windows['mask'][first:first+count, tick, None],
            windows['truth']['state'][first:first+count, tick], state)
        result = original_step(incoming, *args, **kwargs)
        tick += 1; calls += 1
        if tick == core.HORIZON:
            first += count; tick = 0
        return result

    with patch.object(world, 'step', side_effect=true_state_step):
        result = core.predict(world, windows, batch)
    expected = math.ceil(len(windows['plan'])/batch)*core.HORIZON
    if first != len(windows['plan']) or tick != 0 or calls != expected:
        raise AssertionError('Teacher replacement did not follow the native batch/tick order')
    return result, calls


def tensor_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.numpy().tobytes() == b.numpy().tobytes()


def paired_comparison(teacher, free):
    rows = []
    for left, right in zip(teacher['curve_24_policy_steps'], free['curve_24_policy_steps']):
        metrics = {}
        for name in left['metrics']:
            tm, fm = left['metrics'][name], right['metrics'][name]
            t, f = tm['mae'], fm['mae']
            same_pairs = (tm['finite_pairs'] == fm['finite_pairs'] == tm['available_samples']
                          and tm['available_samples'] == fm['available_samples'])
            metrics[name] = dict(teacher_mae=t, free_mae=f,
                identical_finite_pair_set=same_pairs,
                free_over_teacher=None if not same_pairs or t in (None, 0.) or f is None else f/t,
                teacher_zero_mae=t == 0., free_minus_teacher=None if t is None or f is None else f-t)
        rows.append(dict(h=round(left['endpoint_seconds']/core.DT),
            teacher_complete_windows=left['complete_windows'], free_complete_windows=right['complete_windows'],
            metrics=metrics))
    return dict(curve=rows, endpoints={str(h): rows[h-1] for h in core.ENDPOINTS},
        rule='Same target tick/mask; ratio of aggregate MAEs, not mean of per-window ratios. Difference is not a causal/additive decomposition.')


def physical_diagnostics(pred, windows):
    truth, mask = windows['truth'], windows['mask']
    p, t = pred['path'][..., -1, :], truth['path'][..., -1, :]
    groups = {'all': torch.ones(len(mask), dtype=torch.bool)}
    for case in sorted({item['case'] for item in windows['plan']}):
        groups['case/'+case] = torch.tensor([item['case'] == case for item in windows['plan']])
    def at_h(h, selected):
        available = mask[selected, h-1]
        def values(indices):
            pp, tt = p[selected, h-1][..., indices], t[selected, h-1][..., indices]
            return core.stats(pp.flatten(), tt.flatten(), available[:, None].expand_as(pp).flatten())
        # Two gas-spring coordinates slide in metres; the other joints rotate.
        hinge_q = [i for i in range(7, 25) if i not in (11, 16)]
        return dict(hinge_q_rad=values(hinge_q), hinge_v_rad_s=values([i+24 for i in hinge_q]),
            slider_q_m=values([11, 16]), slider_v_m_s=values([35, 40]),
            motor_effort_Nm=values(slice(49, 55)))
    result = {name: {str(h): at_h(h, selected) for h in core.ENDPOINTS} for name, selected in groups.items()}
    # Compare to carrying the measured previous force for one tick; no model involved.
    persistence = {}
    for h in core.ENDPOINTS:
        persistence[str(h)] = {side: core.stats(-truth['state'][:, h-1, index],
            -truth['path'][:, h-1, -1, index], mask[:, h-1])
            for side, index in (('left', 60), ('right', 63))}
    result['true_previous_force_persistence'] = persistence
    return result


def markdown(report):
    def number(value):
        return '—' if value is None else f'{value:.6f}'
    lines = ['# V10每拍真实状态与连续24拍预测对照', '',
        '固定最终world、两套原轨迹、原360窗与真实动作；不训练、不新仿真。MAE越小越准。', '',
        '每拍真状态：每10ms重新输入该拍真实完整状态，只预测下一拍；连续预测：仅起点是真状态，后续用自己的预测状态。H24两者比较同一真实目标时刻，但前者仍是该时刻的一拍预测。', '',
        '| 旧策略轨迹 | H | 速度MAE 真状态/连续 m/s | 左轮MAE 真状态/连续 N | 右轮MAE 真状态/连续 N | 右轮连续/真状态 |',
        '|---|---:|---:|---:|---:|---:|']
    for pair in report['pairs']:
        for h in core.ENDPOINTS:
            row = pair['comparison']['endpoints'][str(h)]['metrics']
            values = []
            for name in ('reward_link_vx_m_s','support_left_N','support_right_N'):
                item = row[name]; values.append(number(item['teacher_mae'])+'/'+number(item['free_mae']))
            ratio = row['support_right_N']['free_over_teacher']
            lines.append(f"| {pair['recording']} | {h} | "+' | '.join(values)+' | '+number(ratio)+' |')
    lines += ['', 'H1两模式全部输出逐位相同，连续分支全部输出与上次封存final预测逐位相同；权重和归一化前后不变。', '',
        'H1/H8/H24物理目标10/80/240ms，奖励LINK特征采样8/78/238ms。每个分支每份数据360个完整窗口，不按预测failure筛选。teacher的累计reward是24个条件单拍之和，不能当连续想象轨迹的回报。', '',
        'core_summary.json保留原分组/非有限数/奖励及失败统计；diagnostic.json另列关节状态误差、真实上一拍接触力保持基线及各拍比值。真状态/连续误差差值不能当作可相加的因果分解；对照不能直接验证训练梯度或机器人行为。', '',
        '本报告只诊断当前最终V10世界模型，不重新选择模型、不改变原联合验收失败结论。']
    return '\n'.join(lines)+'\n'


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    origin = args.trial.resolve()
    receipt = json.loads((origin/'scoring-exit.json').read_text())
    sealed_path, old_score_path = origin/'score/predictions.pt', origin/'score/score.json'
    if (receipt['returncode'] != 0 or core.digest(sealed_path) != receipt['predictions_sha256']
            or core.digest(old_score_path) != receipt['report_sha256']):
        raise ValueError('Previous sealed prediction/score receipt mismatch')
    old_score = json.loads(old_score_path.read_text())
    script = Path(__file__).resolve()
    base_script = Path(core.__file__).resolve()
    if core.digest(base_script) != old_score['score_script_sha256'] or args.batch != 64:
        raise ValueError('Require original scorer and batch64 for exact reproduction')
    current = core.current_training_sources()
    infos = core.contracts(core.command_cfg('walk'))
    old = torch.load(sealed_path, map_location='cpu', weights_only=False)
    final = [p for p in old['worlds'] if p['checkpoint_sha256'] == receipt['final_checkpoint_sha256']]
    if len(final) != 1 or final[0]['checkpoint_sha256'] != receipt['final_checkpoint_sha256']:
        raise ValueError('Require the actual previously scored final world')
    model = core.load_world('final', Path(final[0]['checkpoint']), current, infos)
    if model['proof']['world_state_sha256_before'] != final[0]['world_state_sha256_before']:
        raise ValueError('World/norm state differs from original frozen score')
    records = [core.load_recording(Path(p['directory']), current, infos) for p in old['recordings']]
    windows = [core.make_windows(record) for record in records]
    if len(records) != 2:
        raise ValueError('Exactly two original recordings required')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(script.relative_to(core.ROOT)): core.digest(script),
               str(base_script.relative_to(core.ROOT)): core.digest(base_script)}
    for record in records:
        sources.update(record['proof']['sources'])
    for name, sha in sources.items():
        dest = args.output/'snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(core.ROOT/name, dest)
        if core.digest(dest) != sha:
            raise AssertionError('Source snapshot mismatch')
    core.write(args.output/'source-sha256.json', sources)
    core.write(args.output/'protocol.json', dict(
        original_trial=str(origin), checkpoint=model['proof'], batch=args.batch, threads=2,
        intervention='At each tick replace incoming full254-state with true state; all other native recorded inputs identical.',
        horizons=core.ENDPOINTS, windows_per_recording=360,
        comparison='Same endpoint360windows and same mask; aggregate free MAE / aggregate teacher MAE; zero denominator reported.',
        checks_before_interpretation=['H1 all output fields bitwise identical', 'free all outputs reproduce earlier final exactly',
            'recorded windows bitwise identical', 'model/norm/checkpoint/source/data hashes unchanged'],
        optimizer_updates=0, normalization_updates=0, actor_forwards=0, critic_forwards=0, simulator_steps=0,
        scope='Diagnostic only; no training, policy feedback, gradient or behavior acceptance.'))
    predictions, checks, oracles = [], [], []
    with ExitStack() as guard:
        guard.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer update')))
        guard.enter_context(patch.object(core.RunningNorm, 'update', side_effect=AssertionError('No norm update')))
        for index, w in enumerate(windows):
            original = old['windows'][index]
            if (w['plan'] != original['plan'] or not tensor_equal(w['mask'], original['mask'])
                    or set(w['truth']) != set(original['truth'])
                    or not all(tensor_equal(value, original['truth'][name]) for name, value in w['truth'].items())):
                raise AssertionError('Reconstructed windows differ from sealed original')
            oracles.append(core.oracle_check(model['world'].adapter, records[index]))
            free = core.predict(model['world'], w, args.batch)
            reference = [p['tensors'] for p in old['predictions'] if p['world_label'] == 'final' and p['recording_index'] == index]
            if len(reference) != 1 or not all(tensor_equal(v, reference[0][k]) for k, v in free.items()):
                raise AssertionError('Continuous rollout differs from earlier sealed final')
            teacher, calls = teacher_predict(model['world'], w, args.batch)
            h1 = {name: tensor_equal(value[:, 0], free[name][:, 0]) for name, value in teacher.items()}
            if not all(h1.values()):
                raise AssertionError('Teacher/free H1 mismatch')
            checks.append(dict(recording=index, windows_equal=True, free_all_fields_exact_original=True,
                               h1_all_fields_exact=h1, teacher_world_step_calls=calls, free_world_step_calls=calls))
            for mode, tensors in (('teacher', teacher), ('free', free)):
                predictions.append(dict(recording_index=index, mode=mode, tensors=tensors))
    proof_after = core.state_digest(model['world'])
    if proof_after != model['proof']['world_state_sha256_before']:
        raise AssertionError('World/norm state changed')
    artifact = dict(schema='v10-true-state-versus-free-rollout-v1', windows=windows, predictions=predictions,
        world=model['proof'], recordings=[r['proof'] for r in records], oracles=oracles, checks=checks,
        original_predictions_sha256=core.digest(sealed_path), original_score_sha256=core.digest(old_score_path),
        original_trial=str(origin), term_names=core.TERMS)
    temp = args.output/'predictions.tmp'; sealed = args.output/'predictions.pt'
    torch.save(artifact, temp); temp.replace(sealed)
    sealed_sha = core.digest(sealed)
    del artifact, predictions, old, old_score, windows
    artifact = torch.load(sealed, map_location='cpu', weights_only=False)
    summaries, pairs = [], []
    for index, record in enumerate(records):
        selected = {p['mode']: p['tensors'] for p in artifact['predictions'] if p['recording_index'] == index}
        w = artifact['windows'][index]
        scores = {mode: core.summarize(pred, w) for mode, pred in selected.items()}
        for mode, summary in scores.items():
            summaries.append(dict(recording=str(record['directory']), mode=mode, **summary))
        pairs.append(dict(recording=record['directory'].name, comparison=paired_comparison(scores['teacher'], scores['free']),
            physical={mode: physical_diagnostics(pred, w) for mode, pred in selected.items()}))
    for record in records:
        for info in record['proof']['files'].values():
            if core.digest(Path(info['path'])) != info['sha256']:
                raise AssertionError('Recorded input changed during diagnostic')
    if (core.current_training_sources() != current or core.digest(sealed) != sealed_sha
            or core.digest(Path(model['proof']['checkpoint'])) != model['proof']['checkpoint_sha256']
            or core.digest(sealed_path) != artifact['original_predictions_sha256']
            or core.digest(old_score_path) != artifact['original_score_sha256']
            or any(core.digest(core.ROOT/name) != sha for name, sha in sources.items())):
        raise AssertionError('Original checkpoint/data/prediction/source changed')
    core.verified_training_sources(Path(model['proof']['checkpoint']), current)
    model['proof']['world_state_sha256_after'] = proof_after
    core.write(args.output/'core_summary.json', dict(results=summaries))
    report = dict(schema=artifact['schema'], original_trial=str(origin), world=model['proof'], checks=checks,
        oracles=oracles, pairs=pairs, predictions_sha256=sealed_sha, optimizer_steps=0, normalization_updates=0,
        actor_forward_calls=0, critic_forward_calls=0, simulator_steps=0, cpu_only=True,
        original_predictions_sha256=artifact['original_predictions_sha256'],
        source_manifest_sha256=core.digest(args.output/'source-sha256.json'),
        interpretation='Teacher H24 is a one-tick conditional forecast at tick24. Error difference is not additive causal attribution. No gradient/behavior claim.')
    core.write(args.output/'diagnostic.json', report)
    (args.output/'diagnostic.md').write_text(markdown(report))
    print(json.dumps(dict(output=str(args.output), pair_count=2, modes=2, world_step_calls=sum(
        c['teacher_world_step_calls']+c['free_world_step_calls'] for c in checks), predictions_sha256=sealed_sha,
        optimizer_steps=0, normalization_updates=0, simulator_steps=0), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
