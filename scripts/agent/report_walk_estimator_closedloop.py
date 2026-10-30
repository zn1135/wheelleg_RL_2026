#!/usr/bin/env python3
"""Read-only three-seed report of sealed estimator-in-control comparisons."""

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


METRICS = (
    'world_horizontal_speed_max_m_s','world_horizontal_speed_p95_m_s',
    'world_horizontal_speed_over_0_05_duration_s','xy_displacement_norm_m','xy_max_distance_from_start_m',
    'vx_mae_m_s','vx_abs_error_p95_m_s','vx_abs_error_max_m_s','vx_mean_bias_m_s',
    'pitch_abs_max_deg','pitch_abs_p95_deg','pitch_abs_over_5_duration_s','pitch_abs_over_5_longest_s',
    'height_mean_m','height_mae_m')


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def describe(items):
    """Keep missing identities and worst identity; never impute failed phases."""
    observed=[(identity,float(value)) for identity,value in items if value is not None]
    missing=[identity for identity,value in items if value is None]
    if not observed:
        return dict(observed=0,missing=len(missing),missing_ids=missing,minimum=None,median=None,p95=None,
            maximum=None,worst_signed=None,worst_id=None)
    values=np.asarray([value for _,value in observed])
    worst=int(np.argmax(abs(values)))
    return dict(observed=len(values),missing=len(missing),missing_ids=missing,minimum=float(values.min()),
        median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max()),
        worst_signed=float(values[worst]),worst_id=observed[worst][0])


def direction(speed):
    return 'forward' if speed>0. else 'backward' if speed<0. else 'zero'


def stratum(phase):
    kind=phase['kind']
    if kind=='start':
        occurrence='first' if phase['segment']==1 else 'again'
        return f"start_{occurrence}_{direction(phase['requested_speed'])}"
    if kind=='brake':
        return 'brake_from_'+direction(phase['phase_detail']['previous_requested_speed'])
    return kind


def phase_record(seed,row,cases):
    base,candidate=row['baseline'],row['candidate']
    assert (base['env'],base['kind'],base['segment'],base['start_s'],base['end_s']) == (
        candidate['env'],candidate['kind'],candidate['segment'],candidate['start_s'],candidate['end_s'])
    env=base['env'];case=cases[env]
    record=dict(seed=seed,env=env,case=case['name'],acceleration=case['acceleration'],
        phase=base['kind'],stratum=stratum(base),segment=base['segment'],
        requested_speed=base['requested_speed'],requested_direction=direction(base['requested_speed']),
        start_s=base['start_s'],end_s=base['end_s'],checks=row['checks'],screening_passed=row['passed'],branches={})
    for name,value in (('parent',base),('candidate',candidate)):
        record['branches'][name]=dict(complete=value['complete'],expected_samples=value['expected_samples'],
            observed_samples=value['observed_samples'],metrics=value['metrics'],
            partial_observed_metrics=value['partial_observed_metrics'],pitch_recovery=value['pitch_recovery'],
            leg_window_max_p2p_p95_deg=value['leg_window_max_p2p_p95_deg'],
            leg_window_max_p2p_max_deg=value['leg_window_max_p2p_max_deg'],
            one_second_windows=value['one_second_windows'],sustained=value['sustained'])
    return record


def summarize_phases(rows):
    output={}
    for branch in ('parent','candidate'):
        output[branch]={}
        for metric in METRICS:
            output[branch][metric]=describe([
                (dict(env=row['env'],segment=row['segment'],start_s=row['start_s'],end_s=row['end_s']),
                 None if row['branches'][branch]['metrics'] is None else row['branches'][branch]['metrics'].get(metric))
                for row in rows])
        for metric in ('leg_window_max_p2p_p95_deg','leg_window_max_p2p_max_deg'):
            output[branch][metric]=describe([
                (dict(env=row['env'],segment=row['segment']),row['branches'][branch][metric]) for row in rows])
        output[branch]['incomplete_phases']=[dict(env=row['env'],segment=row['segment'],start_s=row['start_s'],end_s=row['end_s'])
            for row in rows if not row['branches'][branch]['complete']]
        output[branch]['sustained_envs']=sorted({row['env'] for row in rows if row['branches'][branch]['sustained']['detected']})
        output[branch]['sustained_coverage_insufficient']=[dict(env=row['env'],segment=row['segment']) for row in rows
            if not row['branches'][branch]['sustained']['coverage_sufficient']]
        output[branch]['partial_observed_phases']=[dict(env=row['env'],segment=row['segment'],metrics=row['branches'][branch]['partial_observed_metrics'])
            for row in rows if row['branches'][branch]['partial_observed_metrics'] is not None]
    output['screening_failed_phases']=[dict(env=row['env'],segment=row['segment'],failed_checks=[k for k,v in row['checks'].items() if not v])
        for row in rows if not row['screening_passed']]
    return output


def user_goals(rows,environments):
    parking=[row for row in rows if row['phase']=='parking']
    steady=[row for row in rows if row['phase'] in ('parking','cruise_forward','cruise_backward')]
    cruise=[row for row in rows if row['phase'] in ('cruise_forward','cruise_backward')]
    def failures(selected,check):
        result=[]
        for row in selected:
            branch=row['branches']['candidate'];metrics=branch['metrics']
            if metrics is None or not check(metrics):
                result.append(dict(env=row['env'],phase=row['phase'],segment=row['segment'],start_s=row['start_s'],
                    end_s=row['end_s'],missing=metrics is None,metrics=metrics))
        return result
    survival_losses=[row['env'] for row in environments if row['original_survived'] and not row['candidate_survived']]
    drift=failures(parking,lambda m:m['world_horizontal_speed_max_m_s']<=.05)
    pitch=failures(steady,lambda m:m['pitch_abs_max_deg']<=5.)
    proposal=failures(cruise,lambda m:m['vx_mae_m_s']<=.05 and m['vx_abs_error_p95_m_s']<=.10)
    swing=[dict(env=row['env'],phase=row['phase'],segment=row['segment'],sustained=row['branches']['candidate']['sustained'])
        for row in steady if row['branches']['candidate']['sustained']['detected']]
    insufficient=[dict(env=row['env'],phase=row['phase'],segment=row['segment']) for row in steady
        if not row['branches']['candidate']['sustained']['coverage_sufficient']]
    old_pitch_losses=[]
    old_drift_losses=[]
    old_no_swing_losses=[]
    transient_pitch_retention_failures=[]
    tracking_screening=[]
    proposal_losses=[]
    for row in steady:
        a,b=row['branches']['parent']['metrics'],row['branches']['candidate']['metrics']
        if a is not None and a['pitch_abs_max_deg']<=5. and (b is None or b['pitch_abs_max_deg']>5.):
            old_pitch_losses.append(dict(env=row['env'],phase=row['phase'],segment=row['segment']))
        if row['phase']=='parking' and a is not None and a['world_horizontal_speed_max_m_s']<=.05 and (
            b is None or b['world_horizontal_speed_max_m_s']>.05):
            old_drift_losses.append(dict(env=row['env'],phase=row['phase'],segment=row['segment'],
                start_s=row['start_s'],end_s=row['end_s'],parent_horizontal_peak_m_s=a['world_horizontal_speed_max_m_s'],
                candidate_horizontal_peak_m_s=None if b is None else b['world_horizontal_speed_max_m_s'],candidate_missing=b is None))
        parent_branch,candidate_branch=row['branches']['parent'],row['branches']['candidate']
        if parent_branch['complete'] and parent_branch['sustained']['coverage_sufficient'] and not parent_branch['sustained']['detected'] and candidate_branch['sustained']['detected']:
            old_no_swing_losses.append(dict(env=row['env'],phase=row['phase'],segment=row['segment'],
                start_s=row['start_s'],end_s=row['end_s'],parent_sustained=parent_branch['sustained'],
                candidate_sustained=candidate_branch['sustained']))
        if row['phase'].startswith('cruise'):
            if not row['checks'].get('tracking_mae',False) or not row['checks'].get('tracking_p95',False):
                tracking_screening.append(dict(env=row['env'],phase=row['phase'],segment=row['segment'],checks=row['checks']))
            if a is not None and a['vx_mae_m_s']<=.05 and a['vx_abs_error_p95_m_s']<=.1 and (
                b is None or b['vx_mae_m_s']>.05 or b['vx_abs_error_p95_m_s']>.1):
                proposal_losses.append(dict(env=row['env'],phase=row['phase'],segment=row['segment']))
    for row in rows:
        if row['phase'] not in ('start','brake') or not row['branches']['parent']['complete']:
            continue
        failed=[name for name in ('transient_pitch_peak','transient_pitch_total_over5','transient_pitch_longest_over5')
            if name in row['checks'] and not row['checks'][name]]
        if failed:
            transient_pitch_retention_failures.append(dict(env=row['env'],phase=row['phase'],segment=row['segment'],
                requested_speed=row['requested_speed'],stratum=row['stratum'],start_s=row['start_s'],end_s=row['end_s'],
                failed_checks=failed,parent_metrics=row['branches']['parent']['metrics'],
                candidate_metrics=row['branches']['candidate']['metrics']))
    return dict(confirmed_drift_peak_target_m_s=.05,drift_all_parking_passed=not drift,drift_failures=drift,
        drift_scope='Parent-locked steady parking windows only; initialization zero-command horizontal speeds remain separately reported, not silently exempted or called all-time drift compliance.',
        confirmed_steady_pitch_target_deg=5.,steady_pitch_all_passed=not pitch,steady_pitch_failures=pitch,
        actual_forward_backward_tracking='Per-direction MAE/P95 remain measured values; no new accuracy threshold confirmed.',
        tracking_accuracy_proposal=dict(confirmed=False,mae_max_m_s=.05,p95_max_m_s=.10,failures=proposal),
        sustained_swing_quantification=dict(final_goal_threshold_confirmed=False,observed_flagged_phases=swing,
            coverage_insufficient=insufficient,interpretation='Fixed experimental2deg/3windows/1deg reversals; absence of this flag alone does not prove visually acceptable behavior.'),
        transient_pitch='Initial/start/brake peaks, exceedance durations and recovery are separately retained. Absolute allowed transient duration is unconfirmed.',
        original_success_retention=dict(survival_lost_envs=survival_losses,steady_pitch_lost_phases=old_pitch_losses,
            parking_drift_success_lost_phases=old_drift_losses,
            covered_no_sustained_swing_success_lost_phases=old_no_swing_losses,
            transient_pitch_relative_tolerance_failures=transient_pitch_retention_failures,
            tracking_preservation_screening_failures=tracking_screening,tracking_proposal_success_losses=proposal_losses,
            regression_scope='Drift and no-swing losses require the complete parent phase to meet that same criterion first. Preexisting parent failures are not newly labeled regressions; relative transient failures use the unchanged comparator checks.'),
        final_task_complete_claim=False,height='Backward height remains independently unresolved.')


def seed_report(seed,comparison,path):
    assert comparison['verified'] and comparison['optimizer_updates']==0
    assert not comparison['network_inference'] and not comparison['simulation']
    baseline_meta_path=next(Path(p) for p in comparison['baseline_files'] if p.endswith('/encoder.json'))
    candidate_meta_path=next(Path(p) for p in comparison['candidate_files'] if p.endswith('/encoder.json'))
    base_meta,cand_meta=read(baseline_meta_path),read(candidate_meta_path)
    assert base_meta['seed']==cand_meta['seed']==seed
    assert base_meta['cases']==cand_meta['cases']
    assert base_meta['feedback_branch']=='parent' and cand_meta['feedback_branch']=='candidate'
    assert comparison['pairing']['initial_and_physical_arrays_byte_equal']
    assert comparison['pairing']['model_fields_except_encoder_byte_equal']
    sources={str(path):sha(path)}
    for files in (comparison['baseline_files'],comparison['candidate_files']):
        for file,expected in files.items():
            assert sha(file)==expected
            sources[file]=expected
    cases={env:case for case in base_meta['cases'] for env in case['env_ids']}
    rows=[phase_record(seed,row,cases) for row in comparison['per_phase']]
    strata={name:summarize_phases([row for row in rows if row['stratum']==name]) for name in sorted({row['stratum'] for row in rows})}
    phases={name:summarize_phases([row for row in rows if row['phase']==name]) for name in sorted({row['phase'] for row in rows})}
    per_env=[]
    for original in comparison['per_environment']:
        env=original['env'];parking=[row for row in rows if row['env']==env and row['phase']=='parking']
        item=dict(original)
        item['case']=cases[env]['name'];item['acceleration']=cases[env]['acceleration'];item['case_primary_speed']=cases[env]['speed']
        item['parking_legs']={}
        for branch in ('parent','candidate'):
            selected=[row['branches'][branch] for row in parking]
            complete_windows=[window for phase in selected for window in phase['one_second_windows'] if window['complete']]
            all_values=[window['four_motor_max_p2p_deg'] for window in complete_windows]
            item['parking_legs'][branch]=dict(complete_1s_windows=len(complete_windows),
                four_motor_max_p2p_p95_deg=None if not all_values else float(np.percentile(all_values,95)),
                four_motor_max_p2p_max_deg=None if not all_values else max(all_values),
                sustained_detected=any(phase['sustained']['detected'] for phase in selected),
                longest_flagged_span_s=max((phase['sustained']['longest_flagged_span_s'] for phase in selected),default=0.),
                sustained_coverage_insufficient_segments=[row['segment'] for row in parking if not row['branches'][branch]['sustained']['coverage_sufficient']],
                per_phase=[dict(segment=row['segment'],start_s=row['start_s'],end_s=row['end_s'],
                    p95_deg=row['branches'][branch]['leg_window_max_p2p_p95_deg'],max_deg=row['branches'][branch]['leg_window_max_p2p_max_deg'],
                    sustained=row['branches'][branch]['sustained']) for row in parking])
        per_env.append(item)
    survival=dict(parent_survived=sum(row['original_survived'] for row in per_env),candidate_survived=sum(row['candidate_survived'] for row in per_env),
        original_survival_lost_envs=[row['env'] for row in per_env if row['original_survived'] and not row['candidate_survived']],
        first_failures=[dict(env=row['env'],case=row['case'],parent=row['original_first_termination'],candidate=row['candidate_first_termination'])
            for row in per_env if not row['candidate_survived']])
    legs={branch:{metric:describe([(row['env'],row['parking_legs'][branch][metric]) for row in per_env])
        for metric in ('four_motor_max_p2p_p95_deg','four_motor_max_p2p_max_deg','longest_flagged_span_s')} for branch in ('parent','candidate')}
    for branch in legs:
        legs[branch]['sustained_envs']=[row['env'] for row in per_env if row['parking_legs'][branch]['sustained_detected']]
    return dict(seed=seed,survival=survival,existing_screening_passed=comparison['passed'],failure_counts=comparison['failure_counts'],
        pairing=comparison['pairing'],phase_summary=phases,direction_occurrence_summary=strata,per_environment=per_env,
        per_phase=rows,parking_leg_environment_summary=legs,
        second_backward_start=[row for row in rows if row['stratum']=='start_again_backward'],
        user_goals=user_goals(rows,per_env),sources=sources,not_applicable_by_env={str(row['env']):row['not_applicable'] for row in per_env})


def number(value):
    return '缺失' if value is None else f'{value:.5f}'


def main(args):
    paths={int(seed):Path(path).resolve() for seed,path in args.comparison}
    assert set(paths)=={43,48,49} and len(args.comparison)==3, 'Require all three paired seeds; do not select successful comparisons'
    offline_path=args.offline_result.resolve();offline=read(offline_path)
    assert offline['verified'] and not offline['gate']['passed'] and not offline['closed_loop_release']
    reports=[seed_report(seed,read(path),path) for seed,path in sorted(paths.items())]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(scope='Read-only report of existing paired closed-loop metrics. No network, simulation, fitting, new scoring windows or gate changes.',
        offline_conclusion=dict(passed=False,candidate_sha256=offline['candidate_sha256'],source=str(offline_path),source_sha256=sha(offline_path),
            failure_counts=offline['gate']['failure_counts'],unchanged=True),
        behavior_conclusion=dict(independently_authorized_diagnostic=True,uses_actual_candidate_feedback=True,
            existing_screening_all_three_passed=all(row['existing_screening_passed'] for row in reports),
            parent_survived=sum(row['survival']['parent_survived'] for row in reports),candidate_survived=sum(row['survival']['candidate_survived'] for row in reports),
            original_survival_losses=[dict(seed=row['seed'],env=env) for row in reports for env in row['survival']['original_survival_lost_envs']]),
        final_user_goals_conclusion=dict(stable_task_complete_claim=False,
            drift_all_three_parking_windows_passed=all(row['user_goals']['drift_all_parking_passed'] for row in reports),
            steady_pitch_all_three_passed=all(row['user_goals']['steady_pitch_all_passed'] for row in reports),
            unconfirmed='Tracking error tolerance, quantitative sustained-swing threshold and absolute transient-pitch allowed duration remain unconfirmed. Report experimental screening separately.',
            height='Backward low height remains independently unresolved.'),seeds=reports)
    with (output/'closedloop-summary.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    lines=['# 独立闭环诊断：三项结论分别保留','',
        '旧最终离线筛选仍未通过，原21项再次后退起步bias失败不撤销。本轮为用户另行授权的实际闭环诊断，不是旧离线门槛自动放行；没有训练或补训练数据。','',
        '下表报告每种子全部96环境及父窗口内的最坏值；P95先在单环境连续阶段计算，再列最坏环境/阶段。缺失与提前终止保留，不当作零误差。','',
        '| seed | 存活原→候选 | 停车水平峰值原→候选(m/s) | 停车水平P95最坏原→候选 | 每环境停车腿P95最坏原→候选(°) | 持续摆动环境原→候选 |',
        '|---|---:|---:|---:|---:|---:|']
    for report in reports:
        parking=report['phase_summary']['parking'];legs=report['parking_leg_environment_summary']
        vals=[]
        for metric in ('world_horizontal_speed_max_m_s','world_horizontal_speed_p95_m_s'):
            vals.append(number(parking['parent'][metric]['maximum'])+'→'+number(parking['candidate'][metric]['maximum']))
        leg=number(legs['parent']['four_motor_max_p2p_p95_deg']['maximum'])+'→'+number(legs['candidate']['four_motor_max_p2p_p95_deg']['maximum'])
        lines.append(f"| {report['seed']} | {report['survival']['parent_survived']}→{report['survival']['candidate_survived']} | {vals[0]} | {vals[1]} | {leg} | {len(legs['parent']['sustained_envs'])}→{len(legs['candidate']['sustained_envs'])} |")
    lines += ['', '| seed/阶段 | vx MAE最坏原→候选(m/s) | vx误差P95最坏原→候选 | pitch峰值最坏原→候选(°) | 候选缺失阶段数 |', '|---|---:|---:|---:|---:|']
    for report in reports:
        for kind,phase in report['direction_occurrence_summary'].items():
            values=[number(phase['parent'][metric]['maximum'])+'→'+number(phase['candidate'][metric]['maximum'])
                for metric in ('vx_mae_m_s','vx_abs_error_p95_m_s','pitch_abs_max_deg')]
            lines.append(f"| {report['seed']}/{kind} | {' | '.join(values)} | {len(phase['candidate']['incomplete_phases'])} |")
    lines += ['', '再次后退起步逐环境记录单独保存在JSON的`second_backward_start`；包括父窗口、请求/斜率、实际速度误差、pitch峰值/超角时长/回稳、原始每秒腿摆与持续标记。初始化、首次起步、再次起步及两方向刹车没有合并抵消。','',
        '| seed | 原存活样本损失env | 稳态pitch原成功损失数 | 原漂移成功损失数 | 原有覆盖无持续摆动成功损失数 | 起停pitch相对保留失败数 | 速度保留筛选失败数 |', '|---|---|---:|---:|---:|---:|---:|']
    for report in reports:
        goals=report['user_goals'];retained=goals['original_success_retention']
        lines.append(f"| {report['seed']} | {retained['survival_lost_envs']} | {len(retained['steady_pitch_lost_phases'])} | {len(retained['parking_drift_success_lost_phases'])} | {len(retained['covered_no_sustained_swing_success_lost_phases'])} | {len(retained['transient_pitch_relative_tolerance_failures'])} | {len(retained['tracking_preservation_screening_failures'])} |")
    lines += ['', '原成功损失只在父完整阶段原本满足相同门槛时列为新退化；父本来漂移超限或已持续摆动不列作新退化。原漂移成功、已有足够覆盖且无持续摆动的父阶段转坏，以及起停pitch既定相对容差失败的全部env/segment/时间区间和指标均保留在JSON，不只报计数。','']
    lines += ['', '既定实验筛选通过情况：'+str({row['seed']:row['existing_screening_passed'] for row in reports})+'。逐环境失败checks、缺失和首个终止原因在JSON完整保留；存活改善或平均值下降不能抵消原样本退化。','',
        '用户明确目标的零指令水平峰值≤0.05m/s和稳态pitch±5°单独统计。漂移通过字段只指父定义的稳态停车窗口；初始化零指令的世界水平速度峰值/P95另存阶段记录，不能据此宣称全程零指令合规。速度误差≤0.05/P95≤0.10以及2°/3窗/反向行程1°仍是量化建议；无持续标记不能单独证明肉眼摆动可接受。起停允许的绝对超角时长未确认，初始化/起停所有峰值、超角和回稳照实保留，不新增豁免。','',
        '此次结果只说明指定三组轨迹中的实际行为变化，不使旧离线失败改判，也不直接证明稳定完成最终任务或真机可用。后退欠高仍独立未解决。','']
    (output/'closedloop-summary.md').write_text('\n'.join(lines));shutil.copy2(__file__,output/'report-source.py')
    assert sha(offline_path)==result['offline_conclusion']['source_sha256']
    assert all(sha(path)==expected for report in reports for path,expected in report['sources'].items())
    print(json.dumps(dict(verified=True,paired_seeds=[43,48,49],offline_still_failed=True,
        behavior=result['behavior_conclusion'],output=str(output),new_inference=False,training=False),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',nargs=2,action='append',metavar=('SEED','JSON'),required=True)
    parser.add_argument('--offline-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
