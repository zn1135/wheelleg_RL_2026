#!/usr/bin/env python3
"""CPU saved-behavior comparisons; only numerical/survival/balance losses stop continuation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/supermimic/supermimic-v13-continuation-20261101-r1'
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
COMPARATOR=ROOT/'scripts/agent/compare_supermimic_behavior.py'
COMPARATOR_SHA='e88b119ac65b5fad0ad22ec3e51f94c997ff16ebe7dc9f5aa4ea6ba412942105'
REFERENCES={
 'SM615-current13':dict(directory=ROOT/'artifacts/supermimic/supermimic-v13-baseline615-current13-seed43-20261031',
  summary_sha256='bf127e2dae99f57aa1ab3f3efa39e247f9e02a045f4ba7f78bdc14da4f21d102',checkpoint_sha256='42b63440cc90e60bd5aa12ba35e473c153a7a91c90e34dd660d6cc25a3bb8063'),
 'A170':dict(directory=ROOT/'artifacts/supermimic/supermimic-v13-midpoint-seed43-20261031',
  summary_sha256='b277e7de1689f9f5bdc91f0ec6db0b9344d312d4b2ec10f6cb4977bccb0b7e1d',checkpoint_sha256='d306fe738b073ec600005330bc6c8929b19fc0e2086f979631c52cbd864c32dc'),
 'A368':dict(directory=ROOT/'artifacts/supermimic/supermimic-v13-final-seed43-20261031',
  summary_sha256='1936f767e6c12c64ccee39f53f98c74a1a0c399e074a462f7aacc4d1f4483757',checkpoint_sha256='088d4a4a7eebd3a11412ab0e62cdb640b064caee9333a4772b23f62ce54d8e22')}
HISTORICAL_AUDIT=ROOT/'artifacts/supermimic/supermimic-v13-alternating-20261031-r1/baseline-historical615-success-audit.json'
HISTORICAL_AUDIT_SHA='d81ac447e9392b5601a9e2a5d4a7cb8c1886f4f549cd969137ee4ad27670148f'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def named(value):
    if '=' not in value:raise argparse.ArgumentTypeError('Expected NAME=/absolute/path')
    name,path=value.split('=',1)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise argparse.ArgumentTypeError('NAME must contain only letters, digits, underscores or hyphens')
    return name,Path(path).resolve()


def early_decision(comparisons,numerical_failures):
    valid=bool(comparisons) and all(c['protocol']['passed'] for c in comparisons.values())
    losses={name:{kind:c['original_success_preservation'][kind] for kind in ('survival','final_v3_1_balance')}
        for name,c in comparisons.items()}
    lost=any(item['lost_count']>0 for row in losses.values() for item in row.values())
    stop=bool(numerical_failures or lost)
    reasons=[]
    if numerical_failures:reasons.append('numerical_failure')
    for kind in ('survival','final_v3_1_balance'):
        if any(row[kind]['lost_count']>0 for row in losses.values()):reasons.append('lost_original_'+kind+'_ids')
    return dict(evaluation_valid=valid,early_stop_required=True if stop else False if valid else None,
        stop_reasons=reasons,
        numerical_failure=numerical_failures>0,survival_or_balance_original_id_loss=lost,
        original_success_retention=losses,
        rule='Only numerical anomalies or losses of original survival/final-balance IDs trigger behavior early stop. Pitch, direction, speed, parking and height do not trigger early stop.',
        invalid_protocol_action='Pause for measurement/source/29-array verification; protocol failure is not a new behavioral threshold and cannot be treated as a valid pass.')


def behavior_summary(summary):
    rows=summary['summary']['per_environment'];assert {r['env'] for r in rows}==set(range(96)) and len(rows)==96
    assert summary['seed']==43 and summary['num_envs']==96 and summary['seconds']==30. and summary['height_m']==.26 and summary['conditions']=='nominal'
    assert summary['physics_dt_s']==.002 and summary['policy_dt_s']==.01 and summary['parameter_updates']==0
    assert summary['actor_parameters_unchanged'] and summary['source_unchanged']
    expected=[(v,a) for a in (.5,2.) for v in (0.,.5,-.5)]
    assert len(summary['cases'])==6
    missing=[];planned=0;complete_count=0;cases=[]
    for i,(case,(speed,accel)) in enumerate(zip(summary['cases'],expected)):
        assert case['speed']==speed and case['acceleration']==accel and case['yaw_rate']==0. and case['env_ids']==list(range(i*16,(i+1)*16))
        selected=[r for r in rows if r['env'] in case['env_ids']];phases={}
        expected_phases=('initial','parking') if speed==0. else ('initial','start','cruise','brake','parking')
        for r in selected:
            assert set(r['phases'])==set(expected_phases)
            for name,m in r['phases'].items():
                planned+=1;complete_count+=bool(m['complete'])
                if not m['complete']:missing.append(dict(env=r['env'],case=r['case'],phase=name,start_s=m['start_s'],end_s=m['end_s'],observed=m['observed_first_episode_finite_samples'],expected=m['expected_samples']))
        for name in expected_phases:
            valid=[r['phases'][name] for r in selected if r['phases'][name]['complete']]
            def mean(field,stat='mean'):
                values=[m[field][stat] for m in valid if m.get(field) is not None]
                return None if not values else sum(values)/len(values)
            def maximum(field,stat='max_abs'):
                values=[m[field][stat] for m in valid if m.get(field) is not None]
                return None if not values else max(values)
            vx=mean('vx_error_m_s');height=mean('height_error_m')
            reference=selected[0]['phases'][name]['reference_vx_m_s']
            legs=[m['motor_window_max_peak_to_peak_deg'] for m in valid if m.get('motor_window_max_peak_to_peak_deg') is not None]
            phases[name]=dict(complete_environments=len(valid),original_denominator=16,
                incomplete_ids=[r['env'] for r in selected if not r['phases'][name]['complete']],
                actual_vx_mean_m_s=None if vx is None else vx+reference,vx_mae_mean_m_s=mean('vx_error_m_s','mae'),
                horizontal_speed_mean_m_s=mean('horizontal_speed_m_s'),horizontal_speed_peak_m_s=maximum('horizontal_speed_m_s'),
                actual_height_mean_m=None if height is None else summary['height_m']+height,height_abs_error_peak_m=maximum('height_error_m'),
                pitch_abs_peak_deg=maximum('pitch_deg'),motor_one_second_peak_to_peak_max_deg=None if not legs else max(legs))
        cases.append(dict(case=case['name'],requested_vx_m_s=speed,acceleration_m_s2=accel,env_ids=case['env_ids'],
            survival_ids=[r['env'] for r in selected if r['survived']],balance_ids=[r['env'] for r in selected if r['final_balanced']],
            numerical_failure_ids=[r['env'] for r in selected if not r['numerical_ok']],phases=phases))
    assert planned==384
    assert summary['summary']['survived']==sum(r['survived'] for r in rows)
    assert summary['summary']['final_balanced']==sum(r['final_balanced'] for r in rows)
    numeric_ids=[r['env'] for r in rows if not r['numerical_ok']]
    assert len(numeric_ids)==summary['summary']['numerical_failures']
    return dict(planned_environments=96,planned_phases=384,complete_phases=complete_count,all_384_phases_complete=complete_count==384,
        missing_or_incomplete_phases=missing,numerical_failure_ids=numeric_ids,survival_ids=[r['env'] for r in rows if r['survived']],
        balance_ids=[r['env'] for r in rows if r['final_balanced']],cases=cases,
        aggregation='Each reported mean averages only complete environment phases, with original16 denominators and all missing IDs explicit. Parking peak is maximum of real world horizontal speed, never mean vx. Height is .26+mean real height_error.')


def compare_saved(reference,candidate,directory,label,deadline):
    directory.mkdir(parents=True,exist_ok=True);output=directory/(label+'.json');receipt_path=directory/(label+'-exit.json')
    if output.exists():
        receipt=read(receipt_path);assert receipt['returncode']==0 and receipt['report_sha256']==sha(output)
    else:
        assert not receipt_path.exists()
        cap=min(45.,deadline-time.time()-2.);assert cap>0,'CPU comparison deadline exhausted'
        command=[str(PYTHON),'-u',str(COMPARATOR),'--reference',str(reference),'--candidate',str(candidate),'--output',str(output),'--markdown',str(directory/(label+'.md'))]
        start=time.monotonic();receipt=dict(command=command,started_at_utc=datetime.now(timezone.utc).isoformat(),wall_cap_s=cap,comparator_sha256=COMPARATOR_SHA)
        with (directory/(label+'.log')).open('x') as log:
            try:
                run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=cap,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'mjlab_training/src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                receipt['returncode']=run.returncode
            except subprocess.TimeoutExpired:
                receipt.update(returncode=None,timeout=True)
        receipt.update(wall_elapsed_s=time.monotonic()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
        if output.exists():receipt['report_sha256']=sha(output)
        write(receipt_path,receipt)
        assert receipt['returncode']==0 and not receipt.get('timeout'),'Formal comparator did not successfully complete'
    report=read(output)
    assert report['input_sha256']['reference']['summary']==sha(reference/'summary.json') and report['input_sha256']['candidate']['summary']==sha(candidate/'summary.json')
    for role,measurement in (('reference',reference),('candidate',candidate)):
        summary=read(measurement/'summary.json');inputs=report['input_sha256'][role]
        assert inputs['trajectory']==summary['trajectory_sha256'] and inputs['scene']==summary['scene_sha256'] and inputs['checkpoint']==summary['checkpoint_sha256']
    assert len(report['protocol']['fixed_arrays'])==29
    return report,dict(report=str(output),report_sha256=sha(output),receipt=str(receipt_path),receipt_sha256=sha(receipt_path))


def selfcheck():
    c=dict(protocol=dict(passed=True),original_success_preservation={key:dict(lost_count=0) for key in ('survival','final_v3_1_balance','steady_pitch_abs_le_5deg')})
    c['original_success_preservation']['steady_pitch_abs_le_5deg']['lost_count']=160
    assert early_decision({'SM615':c},0)['early_stop_required'] is False
    assert early_decision({'SM615':c},1)['early_stop_required'] is True
    c['original_success_preservation']['final_v3_1_balance']['lost_count']=1
    assert early_decision({'SM615':c},0)['early_stop_required'] is True
    c['original_success_preservation']['final_v3_1_balance']['lost_count']=0;c['protocol']['passed']=False
    assert early_decision({'SM615':c},0)['early_stop_required'] is None
    print(json.dumps(dict(all_checks_passed=True,scope='Synthetic JSON stop-rule boundaries only; pitch loss deliberately does not stop',model_forwards=0,physics_steps=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=named,action='append',default=[],metavar='NAME=DIR')
    parser.add_argument('--evaluation-exit',type=named,action='append',default=[],metavar='NAME=JSON')
    parser.add_argument('--output-prefix',type=Path)
    parser.add_argument('--comparison-dir',type=Path,help='Reusable CPU comparison cache; hash-bound receipts required')
    parser.add_argument('--selfcheck',action='store_true');args=parser.parse_args()
    if args.selfcheck:selfcheck();return
    if not args.candidate or args.output_prefix is None:parser.error('--candidate and --output-prefix are required')
    assert len(dict(args.candidate))==len(args.candidate)
    prefix=args.output_prefix.resolve();prefix.parent.mkdir(parents=True,exist_ok=True)
    json_path=Path(str(prefix)+'.json');md_path=Path(str(prefix)+'.md');assert not json_path.exists() and not md_path.exists()
    comparison_dir=args.comparison_dir.resolve() if args.comparison_dir else prefix.parent/(prefix.name+'-comparisons')
    budget=read(ART/'budget.json');deadline=datetime.fromisoformat(budget['deadline_utc']).timestamp()
    assert time.time()<deadline and sha(COMPARATOR)==COMPARATOR_SHA
    receipts=dict(args.evaluation_exit);protected={};references={};reference_behavior={}
    for name,info in REFERENCES.items():
        p=info['directory']/'summary.json';assert sha(p)==info['summary_sha256'];summary=read(p)
        assert summary['checkpoint_sha256']==info['checkpoint_sha256'];references[name]=summary;reference_behavior[name]=behavior_summary(summary);protected[str(p)]=sha(p)
        assert len(reference_behavior[name]['survival_ids'])==len(reference_behavior[name]['balance_ids'])==96
    assert sha(HISTORICAL_AUDIT)==HISTORICAL_AUDIT_SHA
    historical=read(HISTORICAL_AUDIT);historical_exit=read(HISTORICAL_AUDIT.with_name(HISTORICAL_AUDIT.stem+'-exit.json'))
    assert historical_exit['returncode']==0 and historical_exit['report_sha256']==HISTORICAL_AUDIT_SHA and historical['verdict']=='not_comparable'
    for key in ('survival','final_v3_1_balance'):
        row=historical['original_success_preservation'][key]
        assert row['reference_successes']==row['retained']==96 and row['lost_count']==0
        assert row['reference_success_ids']==row['candidate_success_ids']==list(range(96))
    output_candidates=[]
    for name,directory in args.candidate:
        assert time.time()<deadline
        receipt_path=receipts.get(name,ART/(name+'-evaluation-exit.json'));receipt=read(receipt_path);assert receipt['returncode']==0
        summary_path=directory/'summary.json';summary=read(summary_path);behavior=behavior_summary(summary)
        protected[str(summary_path)]=sha(summary_path);protected[str(receipt_path)]=sha(receipt_path)
        comparisons={};proofs={}
        for ref_name,info in REFERENCES.items():
            report,proof=compare_saved(info['directory'],directory,comparison_dir,name+'-vs-'+ref_name,deadline)
            comparisons[ref_name]=report;proofs[ref_name]=proof
            assert report['counts']['environments']==96 and report['counts']['phase_pairs']==384
            assert report['counts']['common_complete_phase_pairs']==behavior['complete_phases']
            for kind in ('survival','final_v3_1_balance'):assert report['original_success_preservation'][kind]['reference_successes']==96
        decision=early_decision(comparisons,summary['summary']['numerical_failures'])
        output_candidates.append(dict(label=name,name=name,evaluation_valid=decision['evaluation_valid'],early_stop_required=decision['early_stop_required'],
            stop_reasons=decision['stop_reasons'],counts=dict(environments=96,planned_phases=384,complete_phases=behavior['complete_phases'],
                survived=len(behavior['survival_ids']),final_balanced=len(behavior['balance_ids']),numerical_failures=len(behavior['numerical_failure_ids'])),
            directory=str(directory),checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256'],
            checkpoint_iteration=summary['checkpoint_iteration'],checkpoint_transitions=summary['checkpoint_transitions'],
            summary_sha256=sha(summary_path),evaluation_exit=dict(path=str(receipt_path),sha256=sha(receipt_path),record=receipt),
            behavior=behavior,decision=decision,formal_comparisons=proofs,
            retention_against_each_reference={k:r['original_success_preservation'] for k,r in comparisons.items()},
            protocol_against_each_reference={k:r['protocol'] for k,r in comparisons.items()}))
    report=dict(schema='v13-continuation-saved-behavior-report-v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        tool_sha256=sha(Path(__file__)),comparator_sha256=COMPARATOR_SHA,budget_sha256=sha(ART/'budget.json'),deadline_utc=budget['deadline_utc'],
        references={k:dict(directory=str(REFERENCES[k]['directory']),summary_sha256=REFERENCES[k]['summary_sha256'],checkpoint_sha256=REFERENCES[k]['checkpoint_sha256'],behavior=reference_behavior[k]) for k in REFERENCES},
        candidates=output_candidates,historical_original615_scope=dict(report=str(HISTORICAL_AUDIT),sha256=HISTORICAL_AUDIT_SHA,formal_verdict='not_comparable',
            description='Historical18-term versus current13-term baseline remains not_comparable. Its original96 survival/balance IDs equal current13 baseline IDs; the native current13 comparisons retain that full denominator. No historical same-source candidate claim is made.'),
        early_stop_scope='Numerical anomalies and lost original survival/final-balance IDs only. Pitch/direction/parking/height are measured, never added to this trial early-stop rule.',
        full_goal_achieved=False,model_forwards=0,simulator_steps=0,optimizer_steps=0)
    lines=['# V13继续训练固定点行为比较','',
        '早停只看数值异常或原存活/末1秒平衡成功ID丢失。pitch、方向、速度、停车、高度和腿幅均报告，但不单独触发本轮早停；完整任务仍未验收。','',
        '停车水平速度mean为每个完整环境阶段的水平速度模长均值再平均；peak为所有完整环境阶段中的真实水平速度最大值。它不是有正负可抵消的vx均值。真高度=.26+height_error.mean。','']
    for candidate in output_candidates:
        d=candidate['decision'];b=candidate['behavior'];lines+=['## '+candidate['name'],'',
            f"测量有效：{d['evaluation_valid']}；应早停：{d['early_stop_required']}；存活{len(b['survival_ids'])}/96，平衡{len(b['balance_ids'])}/96，数值异常{len(b['numerical_failure_ids'])}；完整阶段{b['complete_phases']}/384。",'',
            '| 参照 | 存活丢失 | 平衡丢失 | pitch丢失（不早停） | 正式协议/29数组 |','|---|---:|---:|---:|---|']
        for name,retention in candidate['retention_against_each_reference'].items():
            lines.append(f"| {name} | {retention['survival']['lost_count']} | {retention['final_v3_1_balance']['lost_count']} | {retention['steady_pitch_abs_le_5deg']['lost_count']} | {candidate['protocol_against_each_reference'][name]['passed']} |")
        lines+=['','| 模型/case | cruise真vx/MAE m/s (n/16) | 停车水平mean/peak m/s (n/16) | cruise/停车真高度 m | cruise/停车高度最大误差 m | 存活/平衡 |','|---|---|---|---|---|---|']
        def number(v):return 'N/A' if v is None else f'{v:.5f}'
        for model,cases in [(n,reference_behavior[n]['cases']) for n in REFERENCES]+[(candidate['name'],b['cases'])]:
            for case in cases:
                c=case['phases'].get('cruise',{});p=case['phases']['parking']
                lines.append(f"| {model}/{case['case']} | {number(c.get('actual_vx_mean_m_s'))}/{number(c.get('vx_mae_mean_m_s'))} ({c.get('complete_environments',0)}/16) | {number(p['horizontal_speed_mean_m_s'])}/{number(p['horizontal_speed_peak_m_s'])} ({p['complete_environments']}/16) | {number(c.get('actual_height_mean_m'))}/{number(p['actual_height_mean_m'])} | {number(c.get('height_abs_error_peak_m'))}/{number(p['height_abs_error_peak_m'])} | {len(case['survival_ids'])}/{len(case['balance_ids'])} |")
        lines+=['',
            '完整成功ID、丢失ID、96个环境和384阶段中的缺失项保留在JSON；无完整阶段标N/A，不用失败后的重置回合补齐。纯零请求没有cruise，停车窗口为2～30秒。','']
    lines+=['旧615历史奖励源码不同时仍not_comparable；本工具没有把它升级为正式同源通过。','',
        '下一步按预声明规则：有数值异常或原存活/平衡ID丢失就停；仅方向/停车差仍按本轮既定固定点继续，不能事后把这些指标加入早停。若协议无效则先核测量，不能当作行为通过。所有后续训练仍受原总截止和最多3600秒学习预算约束。']
    assert all(sha(Path(path))==value for path,value in protected.items()) and sha(COMPARATOR)==COMPARATOR_SHA and time.time()<=deadline
    write(json_path,report)
    with md_path.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(json_path),report_sha256=sha(json_path),candidates=[dict(name=c['name'],**c['decision']) for c in output_candidates]),ensure_ascii=False))


if __name__=='__main__':main()
