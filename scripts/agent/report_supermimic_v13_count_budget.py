#!/usr/bin/env python3
"""CPU saved-behavior report for an update-count trial; never controls training."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_TOOL = ROOT / 'scripts/agent/report_supermimic_v13_continuation.py'
HISTORICAL_TOOL_SHA = 'b166f3c841dd09ff8b223d66c9de8cfc923c242de2f7fc0c45f97a53acc8f863'
with HISTORICAL_TOOL.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == HISTORICAL_TOOL_SHA
spec = importlib.util.spec_from_file_location('_historical_saved_behavior', HISTORICAL_TOOL)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
assert old.sha(HISTORICAL_TOOL) == HISTORICAL_TOOL_SHA
REFERENCES = dict(old.REFERENCES)
REFERENCES['A700'] = dict(
    directory=ROOT / 'artifacts/supermimic-v13-continuation-a700-seed43-20261101-r1',
    summary_sha256='e2a03ae9528fb40801d9d9d776faf88be200807d8ce099278e536c8ed8f06acd',
    checkpoint_sha256='48f909a2da93917b2b6b2e4c0bf74dee471b1295946b2b46e43dc09e1119d978')
PHASE_METRICS = ('actual_vx_mean_m_s', 'vx_mae_mean_m_s',
    'horizontal_speed_mean_m_s', 'horizontal_speed_peak_m_s',
    'actual_height_mean_m', 'height_abs_error_peak_m', 'pitch_abs_peak_deg',
    'motor_one_second_peak_to_peak_max_deg')


def behavior_summary(summary):
    result = old.behavior_summary(summary)
    for case in result['cases']:
        for phase in case['phases'].values():
            phase['full_case_complete'] = phase['complete_environments'] == phase['original_denominator'] == 16
            if not phase['full_case_complete']:
                # Preserve missing IDs/counts, but never fill a16-env result with survivors.
                for key in PHASE_METRICS:
                    phase[key] = None
    result['aggregation'] = ('All96 original environments and384 phases remain in the denominator. '
        'A case metric requires16 complete original environment windows; otherwise it is null. '
        'Parking mean is the mean horizontal-speed magnitude; peak is its actual maximum. '
        'Actual height is .26+height_error.mean; failures/reset episodes never fill missing windows.')
    return result


def describe(comparisons, numerical_failures):
    valid = bool(comparisons) and all(c['protocol']['passed'] for c in comparisons.values())
    lost = any(v['lost_count'] for c in comparisons.values() for v in c['original_success_preservation'].values())
    assessment = ('protocol_not_comparable' if not valid else
        'numerical_failures_observed' if numerical_failures else
        'measured_original_successes_lost' if lost else 'measured_original_successes_preserved')
    return dict(evaluation_valid=valid, behavior_assessment=assessment,
        training_control='report_only', observed_numerical_failures=numerical_failures)


def identity(label, summary, policy_seen, checkpoint_seen):
    actor_sha = summary['actor_sha256_before']
    assert actor_sha == summary['actor_sha256_after']
    previous_policy = policy_seen.get(actor_sha)
    previous_checkpoint = checkpoint_seen.get(summary['checkpoint_sha256'])
    policy_seen.setdefault(actor_sha, label)
    checkpoint_seen.setdefault(summary['checkpoint_sha256'], label)
    return dict(actor_sha256=actor_sha, same_policy_as=previous_policy,
        same_checkpoint_as=previous_checkpoint, distinct_new_policy_point=previous_policy is None)


def complete_pairs(reference, candidate):
    left = {r['env']: r for r in reference['summary']['per_environment']}
    return sum(left[r['env']]['phases'][p]['complete'] and value['complete']
        for r in candidate['summary']['per_environment'] for p, value in r['phases'].items())


def number(value):
    return 'N/A' if value is None else f'{value:.5f}'


def selfcheck():
    summary = old.read(REFERENCES['SM615-current13']['directory'] / 'summary.json')
    complete = behavior_summary(summary)
    assert complete['complete_phases'] == 384
    partial = copy.deepcopy(summary)
    partial['summary']['per_environment'][16]['phases']['cruise']['complete'] = False
    value = behavior_summary(partial)
    phase = value['cases'][1]['phases']['cruise']
    assert phase['complete_environments'] == 15 and phase['original_denominator'] == 16
    assert phase['incomplete_ids'] == [16] and all(phase[k] is None for k in PHASE_METRICS)
    assert value['complete_phases'] == 383
    for index, case in enumerate(complete['cases']):
        selected = summary['summary']['per_environment'][index * 16:(index + 1) * 16]
        p = case['phases']['parking']
        assert p['horizontal_speed_mean_m_s'] == sum(r['phases']['parking']['horizontal_speed_m_s']['mean'] for r in selected) / 16
        assert p['horizontal_speed_peak_m_s'] == max(r['phases']['parking']['horizontal_speed_m_s']['max_abs'] for r in selected)
    failed_summary = old.read(REFERENCES['A700']['directory'] / 'summary.json')
    failed = behavior_summary(failed_summary)
    assert failed['complete_phases'] == 160 and failed['survival_ids'] == failed['balance_ids'] == []
    assert complete_pairs(failed_summary, summary) == 160
    for case in failed['cases']:
        for name in ('cruise', 'parking'):
            if name in case['phases']:
                p = case['phases'][name]
                assert p['complete_environments'] == 0 and all(p[k] is None for k in PHASE_METRICS)
    fake = dict(protocol=dict(passed=True), original_success_preservation={'survival': dict(lost_count=96)})
    for n in (0, 1):
        d = describe({'reference': fake}, n)
        assert d['evaluation_valid'] and d['training_control'] == 'report_only'
        assert not {'early_stop_required', 'stop_reasons', 'decision'} & d.keys()
    policy_seen, checkpoint_seen = {}, {}
    assert identity('A1200', summary, policy_seen, checkpoint_seen)['distinct_new_policy_point']
    duplicate = identity('Final', summary, policy_seen, checkpoint_seen)
    assert not duplicate['distinct_new_policy_point'] and duplicate['same_checkpoint_as'] == duplicate['same_policy_as'] == 'A1200'
    changed_container = dict(summary, checkpoint_sha256='different_checkpoint_same_actor')
    assert not identity('Resaved', changed_container, policy_seen, checkpoint_seen)['distinct_new_policy_point']
    print(json.dumps(dict(all_checks_passed=True,
        scope='Synthetic15of16 masking/report-only semantics and repeat identity; actual saved baseline/A700 JSON only. No native comparison or behavior rerun.',
        partial_case_is_na=True, failed_A700_all_96_retained_in_denominator=True,
        lost_success_or_numerical_observation_never_requests_training_stop=True,
        same_actor_resave_is_not_a_new_policy_point=True,
        model_forwards=0, physics_steps=0, optimizer_updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=old.named, action='append', default=[], metavar='NAME=DIR')
    parser.add_argument('--evaluation-exit', type=old.named, action='append', default=[], metavar='NAME=JSON')
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--comparison-dir', type=Path)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if not args.candidate or args.output_prefix is None:
        parser.error('--candidate and --output-prefix are required')
    labels = [name for name, _ in args.candidate]
    assert len(set(labels)) == len(labels) and not set(labels) & REFERENCES.keys()
    assert len(dict(args.evaluation_exit)) == len(args.evaluation_exit)
    receipts = dict(args.evaluation_exit)
    assert set(receipts) <= set(labels)
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(prefix) + '.json'), Path(str(prefix) + '.md')
    assert not json_path.exists() and not md_path.exists()
    cache = args.comparison_dir.resolve() if args.comparison_dir else prefix.parent / (prefix.name + '-comparisons')
    assert old.sha(old.COMPARATOR) == old.COMPARATOR_SHA
    protected = {str(HISTORICAL_TOOL): HISTORICAL_TOOL_SHA, str(old.COMPARATOR): old.COMPARATOR_SHA}
    reference_summaries, reference_behavior = {}, {}
    policy_seen, checkpoint_seen = {}, {}
    for name, info in REFERENCES.items():
        path = info['directory'] / 'summary.json'
        assert old.sha(path) == info['summary_sha256']
        summary = old.read(path)
        assert summary['checkpoint_sha256'] == info['checkpoint_sha256']
        reference_summaries[name] = summary
        reference_behavior[name] = behavior_summary(summary)
        protected[str(path)] = old.sha(path)
        policy_seen.setdefault(summary['actor_sha256_before'], name)
        checkpoint_seen.setdefault(summary['checkpoint_sha256'], name)
    assert old.sha(old.HISTORICAL_AUDIT) == old.HISTORICAL_AUDIT_SHA
    historical = old.read(old.HISTORICAL_AUDIT)
    historical_receipt_path = old.HISTORICAL_AUDIT.with_name(old.HISTORICAL_AUDIT.stem + '-exit.json')
    historical_exit = old.read(historical_receipt_path)
    assert historical_exit['returncode'] == 0 and historical_exit['report_sha256'] == old.HISTORICAL_AUDIT_SHA
    assert historical['verdict'] == 'not_comparable'
    for key in ('survival', 'final_v3_1_balance'):
        retained = historical['original_success_preservation'][key]
        assert retained['reference_success_ids'] == retained['candidate_success_ids'] == list(range(96))
        assert retained['lost_count'] == 0
    protected[str(old.HISTORICAL_AUDIT)] = old.HISTORICAL_AUDIT_SHA
    protected[str(historical_receipt_path)] = old.sha(historical_receipt_path)
    candidates = []
    for label, directory in args.candidate:
        receipt_path = receipts.get(label, prefix.parent / (label + '-evaluation-exit.json'))
        receipt = old.read(receipt_path)
        assert receipt['returncode'] == 0 and not receipt.get('wall_cap_triggered') and not receipt.get('timeout')
        summary_path = directory / 'summary.json'
        summary = old.read(summary_path)
        command = receipt['command']
        assert Path(command[command.index('--output') + 1]).resolve() == directory
        assert Path(command[command.index('--checkpoint') + 1]).resolve() == Path(summary['checkpoint']).resolve()
        behavior = behavior_summary(summary)
        protected[str(summary_path)], protected[str(receipt_path)] = old.sha(summary_path), old.sha(receipt_path)
        comparisons, proofs = {}, {}
        for name, info in REFERENCES.items():
            # A fresh per-comparison CPU cap; no dependency on the historical trial deadline.
            result, proof = old.compare_saved(info['directory'], directory, cache, label + '-vs-' + name, time.time() + 47.)
            comparison_receipt = old.read(proof['receipt'])
            assert result['comparator_sha256'] == comparison_receipt['comparator_sha256'] == old.COMPARATOR_SHA
            assert result['counts']['environments'] == 96 and result['counts']['phase_pairs'] == 384
            assert result['counts']['common_complete_phase_pairs'] == complete_pairs(reference_summaries[name], summary)
            comparisons[name], proofs[name] = result, proof
            protected[proof['report']], protected[proof['receipt']] = proof['report_sha256'], proof['receipt_sha256']
        description = describe(comparisons, len(behavior['numerical_failure_ids']))
        candidates.append(dict(label=label, name=label, **description,
            directory=str(directory), checkpoint=summary['checkpoint'], checkpoint_sha256=summary['checkpoint_sha256'],
            checkpoint_iteration=summary['checkpoint_iteration'], checkpoint_transitions=summary['checkpoint_transitions'],
            summary_sha256=old.sha(summary_path), evaluation_exit=dict(path=str(receipt_path), sha256=old.sha(receipt_path), record=receipt),
            **identity(label, summary, policy_seen, checkpoint_seen),
            counts=dict(environments=96, planned_phases=384, complete_phases=behavior['complete_phases'],
                survived=len(behavior['survival_ids']), final_balanced=len(behavior['balance_ids']), numerical_failures=len(behavior['numerical_failure_ids'])),
            behavior=behavior, formal_comparisons=proofs,
            native_behavior_verdicts={name: value['verdict'] for name, value in comparisons.items()},
            retention_against_each_reference={name: value['original_success_preservation'] for name, value in comparisons.items()},
            protocol_against_each_reference={name: value['protocol'] for name, value in comparisons.items()}))
    result = dict(schema='v13-count-budget-saved-behavior-report-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        tool_sha256=old.sha(Path(__file__)), historical_helper_sha256=HISTORICAL_TOOL_SHA, comparator_sha256=old.COMPARATOR_SHA,
        training_control='report_only', training_scope='This report never stops or resumes training. Behavior failure/retention losses/direction/parking do not control the update-count experiment; actual training numerical/runtime errors are handled by its separate runner.',
        references={name: dict(info, directory=str(info['directory']), behavior=reference_behavior[name]) for name, info in REFERENCES.items()},
        candidates=candidates, distinct_new_policy_labels=[c['label'] for c in candidates if c['distinct_new_policy_point']],
        repeat_scope='Each Final remains a separate measured run. Identical complete actor hashes, including encoder/norm state, do not create another policy trend point; repeated simulations are not assumed bitwise equal.',
        historical_original615_scope=dict(report=str(old.HISTORICAL_AUDIT), sha256=old.HISTORICAL_AUDIT_SHA, formal_verdict='not_comparable',
            description='Historical18 versus current13 rewards remain not_comparable. The current13 SM615 actor baseline retains the historical original96 survival/balance IDs; no historical same-source candidate claim is made.'),
        protected_file_sha256=protected, full_goal_achieved=False, model_forwards=0, simulator_steps=0, optimizer_steps=0)
    lines = ['# V13更新次数预算：固定点行为报告', '',
        '仅描述训练后测量，不控制训练。摔倒、原成功丢失、方向或停车退化均不在此产生早停指令；数值/运行错误由独立训练器处理。', '',
        '每case保持原16个环境、每候选96例/384阶段。不足16个完整原回合窗口的整格指标为N/A；未用死前片段或重置后回合填补。停车mean是水平速度模长均值，peak是真实水平最大速度，实际高度=.26+height_error.mean。', '']
    for candidate in candidates:
        b = candidate['behavior']
        lines += ['## ' + candidate['label'], '',
            f"协议有效：{candidate['evaluation_valid']}；行为描述：{candidate['behavior_assessment']}；存活{len(b['survival_ids'])}/96、平衡{len(b['balance_ids'])}/96、数值异常{len(b['numerical_failure_ids'])}；完整阶段{b['complete_phases']}/384。",
            f"同策略参照/较早测量：{candidate['same_policy_as'] or '无'}；同checkpoint：{candidate['same_checkpoint_as'] or '无'}；新增不同策略测量点：{candidate['distinct_new_policy_point']}。", '',
            '| 参照 | 原存活成功/丢失 | 原平衡成功/丢失 | pitch丢失 | 协议/29数组 |', '|---|---|---|---|---|']
        for name, retained in candidate['retention_against_each_reference'].items():
            s, f = retained['survival'], retained['final_v3_1_balance']
            lines.append(f"| {name} | {s['reference_successes']}/{s['lost_count']} | {f['reference_successes']}/{f['lost_count']} | {retained['steady_pitch_abs_le_5deg']['lost_count']} | {candidate['protocol_against_each_reference'][name]['passed']} |")
        lines += ['', '| 模型/case | 巡航真vx/MAE m/s (n/16) | 停车mean/peak m/s (n/16) | 巡航/停车真高度 m | 巡航/停车高度最大误差 m | 存活/平衡 |', '|---|---|---|---|---|---|']
        for model, cases in [(name, reference_behavior[name]['cases']) for name in REFERENCES] + [(candidate['label'], b['cases'])]:
            for case in cases:
                c, p = case['phases'].get('cruise', {}), case['phases']['parking']
                lines.append(f"| {model}/{case['case']} | {number(c.get('actual_vx_mean_m_s'))}/{number(c.get('vx_mae_mean_m_s'))} ({c.get('complete_environments', 0)}/16) | {number(p['horizontal_speed_mean_m_s'])}/{number(p['horizontal_speed_peak_m_s'])} ({p['complete_environments']}/16) | {number(c.get('actual_height_mean_m'))}/{number(p['actual_height_mean_m'])} | {number(c.get('height_abs_error_peak_m'))}/{number(p['height_abs_error_peak_m'])} | {len(case['survival_ids'])}/{len(case['balance_ids'])} |")
        lines += ['', '完整原成功/失效ID、每个缺失阶段、29数组协议和正式比较路径保留在JSON。A700参照本身0/96存活/平衡，不把它当成96个成功样本；纯零请求没有巡航。', '']
    lines += ['Final即使同权重也独立列出，不作为额外训练趋势点；重复运行不保证逐帧一致。当前13项与历史18项奖励的比较保持not_comparable，完整任务未验收。', '',
        '下一步由本轮独立计数预算执行器安排既定保存点和最终评估；报告不因行为退化追加、停止或重启训练。最终结合各case完整窗口逐项判断，不从单个布尔量推断长期训练必然有效或无效。']
    assert all(old.sha(Path(path)) == digest for path, digest in protected.items())
    assert old.sha(Path(__file__)) == result['tool_sha256']
    old.write(json_path, result)
    with md_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(json_path), report_sha256=old.sha(json_path), training_control='report_only',
        candidates=[dict(label=c['label'], evaluation_valid=c['evaluation_valid'], counts=c['counts']) for c in candidates]), ensure_ascii=False))


if __name__ == '__main__':
    main()
