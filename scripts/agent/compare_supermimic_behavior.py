#!/usr/bin/env python3
"""Compare paired saved walk measurements on CPU; never run a policy or physics.

Keep every original success in the denominator. Missing/incomplete candidate
episodes do not disappear through a common-survivor filter. Tracking error and
one-second motor amplitudes are descriptive deltas, not new acceptance gates.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIELDS = ('valid','height','true_velocity','root_linear_velocity_w','roll_pitch',
          'angular_velocity_xy','support','base_contact','joint_q')


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def array_record(array):
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def bitwise_equal(left,right):
    return left.shape==right.shape and left.dtype==right.dtype and left.tobytes()==right.tobytes()


def zero_yaw_evidence(directory,summary,manifest):
    """Only legacy, explicitly linear zero-yaw protocols may fill a missing array."""
    cases=summary['cases']
    linear=all(case.get('kind','linear')=='linear' and case.get('yaw_rate',0.)==0.
        and case.get('acceleration') is not None and case['acceleration']>0.
        and case['name']==f"v{case['speed']:g}-a{case['acceleration']:g}" for case in cases)
    if not linear:
        return None
    coverage=summary.get('coverage_facts',{})
    if coverage.get('case_kind')=='linear' and coverage.get('requested_yaw_rad_s')==[0.]:
        return dict(method='Explicit linear/zero-yaw coverage_facts',value=0.)
    for name,sha in manifest.items():
        if Path(name).name!='assess_chuanliantui_supermimic.py':
            continue
        path=directory/'snapshot'/name
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node,ast.FunctionDef) or node.name!='commands':
                continue
            # The zero call must be unconditional in the actual command function.
            for statement in node.body:
                if not isinstance(statement,ast.Expr) or not isinstance(statement.value,ast.Call):
                    continue
                call=statement.value.func
                if (isinstance(call,ast.Attribute) and call.attr=='zero_'
                    and isinstance(call.value,ast.Attribute) and call.value.attr=='tracking_yaw_rate'
                    and isinstance(call.value.value,ast.Name) and call.value.value.id=='state'):
                    return dict(method='Verified evaluator snapshot unconditionally zeroes requested yaw',
                                evaluator=name,evaluator_sha256=sha,value=0.)
    return None


def load_measurement(directory):
    directory=directory.resolve()
    path=directory/'summary.json'
    summary=json.loads(path.read_text())
    checks=[]
    paths=dict(summary=path,trajectory=directory/'trajectory.npz',scene=directory/'scene.mjb')
    hashes={name:digest(value) for name,value in paths.items()}
    for name in ('trajectory','scene'):
        if hashes[name]!=summary[f'{name}_sha256']:
            raise ValueError(f'{directory}: {name} hash differs from its summary')
    checkpoint=Path(summary['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint=directory/checkpoint
    if digest(checkpoint)!=summary['checkpoint_sha256']:
        raise ValueError(f'{directory}: checkpoint hash differs from its summary')
    paths['checkpoint']=checkpoint
    hashes['checkpoint']=summary['checkpoint_sha256']
    source_manifest=directory/'source-sha256.json'
    if digest(source_manifest)!=summary['source_snapshot_sha256']:
        raise ValueError(f'{directory}: source manifest hash differs from its summary')
    sources=json.loads(source_manifest.read_text())
    for name,sha in sources.items():
        if digest(directory/'snapshot'/name)!=sha:
            raise ValueError(f'{directory}: source snapshot changed: {name}')
    with np.load(paths['trajectory'],allow_pickle=False) as archive:
        fixed={name:archive[name].copy() for name in archive.files if name.startswith('fixed_')}
        requested=archive['requested_vx'].copy()
        yaw_evidence=None
        if 'requested_yaw' in archive:
            requested_yaw=archive['requested_yaw'].copy()
        else:
            yaw_evidence=zero_yaw_evidence(directory,summary,sources)
            if yaw_evidence is None:
                raise ValueError(f'{directory}: missing yaw requests without an explicit zero-yaw linear protocol')
            requested_yaw=np.zeros_like(requested)
        times=archive['time_s'].copy()
        policy_times=archive['policy_time_s'].copy()
        weight=archive['weight_N'].copy()
        values=archive['values']
        fields={name:values[...,slice(*summary['layout'][name])].copy() for name in FIELDS}
        finite=np.isfinite(values).all(-1)
        value_shape=list(values.shape)
        del values
    n=summary['num_envs'];dt=summary['physics_dt_s'];seconds=summary['seconds']
    if (len(times)!=round(seconds/dt) or requested.shape!=(len(policy_times),n)
        or requested_yaw.shape!=requested.shape or fields['valid'].shape!=(len(times),n,1)
        or not np.allclose(times,(np.arange(len(times))+1)*dt,atol=1e-8,rtol=0.)):
        raise ValueError(f'{directory}: inconsistent trace dimensions or timestamps')
    if not np.isfinite(requested).all() or not np.isfinite(requested_yaw).all():
        raise ValueError(f'{directory}: requested commands are not finite')
    valid=fields['valid'][...,0]>.5
    if np.any(np.diff(valid.astype(np.int8),axis=0)>0):
        raise ValueError(f'{directory}: validity reactivates after the first episode')
    rows={row['env']:row for row in summary['summary']['per_environment']}
    if set(rows)!=set(range(n)):
        raise ValueError(f'{directory}: per-environment result coverage differs')
    checks.extend(('measurement/checkpoint/source hashes verified','first-episode validity monotone'))
    return dict(directory=directory,summary=summary,paths=paths,hashes=hashes,sources=sources,
        fixed=fixed,requested=requested,requested_yaw=requested_yaw,yaw_evidence=yaw_evidence,
        times=times,policy_times=policy_times,weight=weight,fields=fields,finite=finite,
        valid=valid,rows=rows,checks=checks,value_shape=value_shape)


def case_protocol(record):
    summary=record['summary'];result={}
    for case in summary['cases']:
        # Missing legacy yaw is justified above using the saved zero-yaw evaluator.
        yaw=case.get('yaw_rate')
        if yaw is None:
            if record['yaw_evidence'] is None:
                raise ValueError('Case lacks yaw metadata without verified legacy zero-yaw evidence')
            yaw=0.
        row=dict(vx=case['speed'],yaw=yaw,acceleration=case.get('acceleration'),
            yaw_acceleration=case.get('yaw_acceleration'),height=case.get('height_m',summary['height_m']),
            kind=case.get('kind','linear'))
        for env in case['env_ids']:
            if env in result:
                raise ValueError('Case environment IDs overlap')
            result[env]=row
    if set(result)!=set(range(summary['num_envs'])):
        raise ValueError('Case protocol does not cover every environment exactly once')
    return result


def compare_protocol(reference,candidate):
    r,c=reference['summary'],candidate['summary']
    checks={name:r.get(name)==c.get(name) for name in ('task','seed','seconds','num_envs',
        'physics_dt_s','policy_dt_s','height_m','conditions','command_schedule','balance_criterion')}
    checks['per_environment_vx_yaw_acceleration_height']=case_protocol(reference)==case_protocol(candidate)
    checks['xml_sha256']=r['xml_sha256']==c['xml_sha256']
    checks['requested_vx_bitwise']=bitwise_equal(reference['requested'],candidate['requested'])
    checks['requested_yaw_bitwise']=bitwise_equal(reference['requested_yaw'],candidate['requested_yaw'])
    checks['policy_times_bitwise']=bitwise_equal(reference['policy_times'],candidate['policy_times'])
    checks['physics_times_bitwise']=bitwise_equal(reference['times'],candidate['times'])
    checks['weight_N_bitwise']=bitwise_equal(reference['weight'],candidate['weight'])
    checks['fixed_field_coverage']=set(reference['fixed'])==set(candidate['fixed'])
    arrays={}
    for name in sorted(set(reference['fixed'])|set(candidate['fixed'])):
        left,right=reference['fixed'].get(name),candidate['fixed'].get(name)
        equal=left is not None and right is not None and bitwise_equal(left,right)
        arrays[name]=dict(bitwise_equal=equal,reference=None if left is None else array_record(left),
                         candidate=None if right is None else array_record(right))
    checks['every_fixed_initial_and_physical_array_bitwise_equal']=bool(arrays) and all(v['bitwise_equal'] for v in arrays.values())
    shared_schema={name:(r['layout'][name][1]-r['layout'][name][0])==
        (c['layout'][name][1]-c['layout'][name][0]) for name in FIELDS}
    checks['shared_physical_field_dimensions']=all(shared_schema.values())
    checks['driven_leg_joint_order']=r['joint_names'][:4]==c['joint_names'][:4]==['lf0','lf00','rf0','rf00']
    source_names=set(reference['sources'])|set(candidate['sources'])
    differences=[dict(path=name,reference_sha256=reference['sources'].get(name),
        candidate_sha256=candidate['sources'].get(name)) for name in sorted(source_names)
        if reference['sources'].get(name)!=candidate['sources'].get(name)]
    environment=[row for row in differences if row['path'].startswith(
        ('mjlab_training/src/ct_mjlab/envs/','mjlab_training/src/ct_mjlab/smp/'))
        or row['path']=='mjlab_training/src/ct_mjlab/utils/task_registry.py']
    checks['shared_environment_implementation_sources']=not environment
    # Honor each measurement's original verification, without demanding a PPO world-model contract.
    checks['reference_original_training_source_verified']=r.get('training_source_verification',{}).get('verified') is True
    checks['candidate_original_training_source_verified']=c.get('training_source_verification',{}).get('verified') is True
    return dict(passed=all(checks.values()),checks=checks,fixed_arrays=arrays,
        shared_physical_fields=shared_schema,reference_only_fields=sorted(set(r['layout'])-set(c['layout'])),
        candidate_only_fields=sorted(set(c['layout'])-set(r['layout'])),
        legacy_zero_yaw_fill=dict(reference=reference['yaw_evidence'],candidate=candidate['yaw_evidence']),
        source_differences=differences,shared_environment_source_differences=environment,
        original_training_source_verification=dict(reference=r.get('training_source_verification'),
            candidate=c.get('training_source_verification')),
        original_training_physical_verification=dict(reference=r.get('training_physical_verification'),
            candidate=c.get('training_physical_verification')))


def stats(values):
    if not len(values):
        return None
    return dict(mean=float(np.mean(values)),mae=float(np.mean(abs(values))),
        p95_abs=float(np.quantile(abs(values),.95)),max_abs=float(np.max(abs(values))))


def phase_metrics(record,env,definition):
    start,end=definition['start_s'],definition['end_s'];dt=record['summary']['physics_dt_s']
    reference=definition['reference_vx_m_s'];yaw=definition.get('reference_yaw_rad_s',0.)
    times=record['times'];row=record['rows'][env]
    selected=(times>start+1e-9)&(times<=end+1e-9)
    mask=selected&record['valid'][:,env]&record['finite'][:,env]&(times<=row['episode_end_s']+1e-8)
    expected=max(0,round((end-start)/dt))
    complete=bool(expected>0 and int(mask.sum())==expected and end<=record['summary']['seconds']+1e-8)
    f=record['fields']
    vx=f['true_velocity'][:,env,0]-reference
    horizontal=np.linalg.norm(f['root_linear_velocity_w'][:,env,:2],axis=-1)
    pitch=f['roll_pitch'][:,env,1]
    motor=np.rad2deg(np.unwrap(f['joint_q'][:,env,:4],axis=0))
    windows=[]
    for second in range(math.ceil(start-1e-8),math.floor(end+1e-8)):
        window=(times>second+1e-9)&(times<=second+1.+1e-9)
        valid=bool(window.sum()==round(1./dt) and (record['valid'][:,env]&record['finite'][:,env])[window].all())
        amplitude=np.ptp(motor[window],axis=0) if valid else None
        windows.append(dict(start_s=second,complete=valid,
            motor_peak_to_peak_deg=None if amplitude is None else amplitude.tolist(),
            max_motor_peak_to_peak_deg=None if amplitude is None else float(amplitude.max())))
    amplitudes=[w['max_motor_peak_to_peak_deg'] for w in windows if w['complete']]
    return dict(start_s=start,end_s=end,reference_vx_m_s=reference,reference_yaw_rad_s=yaw,
        complete=complete,expected_samples=expected,observed_first_episode_finite_samples=int(mask.sum()),
        vx_error_m_s=stats(vx[mask]),horizontal_speed_m_s=stats(horizontal[mask]),pitch_deg=stats(pitch[mask]),
        steady_pitch_le_5deg=bool(abs(pitch[mask]).max()<=5.) if complete else False,
        pure_parking_horizontal_max_le_005=bool(horizontal[mask].max()<=.05)
            if complete and reference==0. and yaw==0. else False,
        motor_windows=windows,motor_window_max_peak_to_peak_deg=None if not amplitudes else max(amplitudes),
        motor_window_p95_peak_to_peak_deg=None if not amplitudes else float(np.quantile(amplitudes,.95)))


def behavior(record):
    f=record['fields'];seconds=record['summary']['seconds'];dt=record['summary']['physics_dt_s']
    rows={};mismatches=[]
    for env,source in record['rows'].items():
        finite=record['finite'][:,env];valid=record['valid'][:,env]
        numerical=bool(finite[valid].all())
        survived=bool(source['episode_end_s']>=seconds-1e-7
            and source['episode_end_reason'] in ('duration','timeout') and valid.all())
        support=f['support'][:,env];weight=record['weight'][env]
        balanced=(valid&finite&(support>=max(.1*weight,5.)).all(-1)
            &(support.sum(-1)>=.8*weight)&(support.sum(-1)<=1.2*weight)
            &(f['base_contact'][:,env,0]<=1.)&(abs(f['roll_pitch'][:,env]).max(-1)<=35.)
            &(np.linalg.norm(f['angular_velocity_xy'][:,env],axis=-1)<=.5))
        last_bad=np.flatnonzero(~balanced)
        hold=(len(balanced)-(int(last_bad[-1])+1 if len(last_bad) else 0))*dt
        final=bool(survived and numerical and hold>=1.-1e-7)
        phases={name:phase_metrics(record,env,definition) for name,definition in source['phases'].items()}
        row=dict(env=env,case=source['case'],survived=survived,numerical_ok=numerical,
            final_balanced=final,final_balance_hold_s=hold,episode_end_s=source['episode_end_s'],
            episode_end_reason=source['episode_end_reason'],phases=phases)
        for key in ('survived','numerical_ok','final_balanced'):
            if row[key] is not source[key]:
                mismatches.append(dict(env=env,field=key,summary=source[key],raw_recomputed=row[key]))
        for name,phase in phases.items():
            original=source['phases'][name]
            if phase['complete']!=original['complete']:
                mismatches.append(dict(env=env,phase=name,field='complete',summary=original['complete'],raw_recomputed=phase['complete']))
        rows[env]=row
    return rows,mismatches


def identities(items):
    return [dict(env=value[0],phase=value[1]) if isinstance(value,tuple) else value for value in sorted(items)]


def retention(original,current):
    return dict(reference_successes=len(original),candidate_successes=len(current),retained=len(original&current),
        lost_count=len(original-current),gained_count=len(current-original),
        lost=identities(original-current),gained=identities(current-original),
        reference_success_ids=identities(original),candidate_success_ids=identities(current))


def gates(rows):
    result=dict(survival=set(),final_v3_1_balance=set(),steady_pitch_abs_le_5deg=set(),
                pure_parking_horizontal_max_le_005=set())
    for env,row in rows.items():
        if row['survived']:result['survival'].add(env)
        if row['final_balanced']:result['final_v3_1_balance'].add(env)
        for phase,metrics in row['phases'].items():
            if phase not in ('cruise','parking') or not row['survived'] or not row['numerical_ok'] or not metrics['complete']:
                continue
            if metrics['steady_pitch_le_5deg']:result['steady_pitch_abs_le_5deg'].add((env,phase))
            if phase=='parking' and metrics['reference_vx_m_s']==0. and metrics['reference_yaw_rad_s']==0. and metrics['pure_parking_horizontal_max_le_005']:
                result['pure_parking_horizontal_max_le_005'].add((env,phase))
    return result


def delta_metrics(left,right):
    result={}
    for field in ('vx_error_m_s','horizontal_speed_m_s','pitch_deg'):
        result[field]={name:right[field][name]-left[field][name] for name in left[field]}
    for field in ('motor_window_max_peak_to_peak_deg','motor_window_p95_peak_to_peak_deg'):
        result[field]=None if left[field] is None or right[field] is None else right[field]-left[field]
    windows={row['start_s']:row for row in right['motor_windows']}
    result['motor_windows']=[]
    for row in left['motor_windows']:
        other=windows.get(row['start_s'])
        common=row['complete'] and other is not None and other['complete']
        result['motor_windows'].append(dict(start_s=row['start_s'],both_complete=common,
            candidate_minus_reference_motor_peak_to_peak_deg=None if not common else (
                np.asarray(other['motor_peak_to_peak_deg'])-row['motor_peak_to_peak_deg']).tolist()))
    return result


def compare(reference,candidate):
    protocol=compare_protocol(reference,candidate)
    r,rm=behavior(reference);c,cm=behavior(candidate)
    protocol['summary_recompute_mismatches']=dict(reference=rm,candidate=cm)
    protocol['checks']['summary_matches_raw_first_episode_recomputation']=not rm and not cm
    phases=[];missing=[];moving=[]
    cases=case_protocol(reference)
    for env in sorted(r):
        for name,left in r[env]['phases'].items():
            right=c.get(env,{}).get('phases',{}).get(name)
            definitions_equal=right is not None and all(left[key]==right[key] for key in (
                'start_s','end_s','reference_vx_m_s','reference_yaw_rad_s'))
            common=definitions_equal and left['complete'] and right['complete']
            item=dict(env=env,phase=name,reference_complete=left['complete'],
                candidate_complete=False if right is None else right['complete'],
                reference=left,candidate=right,both_complete=common,
                candidate_minus_reference=delta_metrics(left,right) if common else None)
            if not definitions_equal:
                protocol['checks']['all_phase_definitions_equal']=False
            if not common:
                reason=('candidate_phase_missing' if right is None else 'phase_definition_mismatch' if not definitions_equal
                    else 'candidate_incomplete_or_failed' if not right['complete'] else 'reference_incomplete_or_failed')
                missing.append(dict(env=env,phase=name,reason=reason))
            phases.append(item)
            if cases[env]['vx']!=0.:moving.append(item)
    protocol['checks'].setdefault('all_phase_definitions_equal',True)
    protocol['passed']=all(protocol['checks'].values())
    rg,cg=gates(r),gates(c)
    retained={name:retention(values,cg[name]) for name,values in rg.items()}
    lost=any(row['lost_count'] for row in retained.values())
    aggregate={}
    for phase in sorted({item['phase'] for item in phases}):
        common=[item for item in phases if item['phase']==phase and item['both_complete']]
        aggregate[phase]=dict(common_complete_phases=len(common),
            mean_reference_vx_mae_m_s=None if not common else float(np.mean([x['reference']['vx_error_m_s']['mae'] for x in common])),
            mean_candidate_vx_mae_m_s=None if not common else float(np.mean([x['candidate']['vx_error_m_s']['mae'] for x in common])),
            mean_candidate_minus_reference_vx_mae_m_s=None if not common else float(np.mean([x['candidate_minus_reference']['vx_error_m_s']['mae'] for x in common])),
            scope='Descriptive common-complete phase errors only; never the preservation denominator')
    missing_coverage=['One paired measurement does not prove full velocity/height/yaw coverage or independent seeds.',
        'No independent CPU physics/sim2sim or hardware result is established by this comparison.',
        'Numeric nonzero tracking accuracy, absolute transient pitch duration and motor oscillation cutoffs are unconfirmed.',
        'A representative video does not replace all-environment numerical behavior acceptance.']
    return dict(reference=str(reference['directory']),candidate=str(candidate['directory']),
        input_sha256=dict(reference=reference['hashes'],candidate=candidate['hashes']),protocol=protocol,
        verdict='not_comparable' if not protocol['passed'] else 'regressed_rejected' if lost else 'reference_successes_preserved_in_this_protocol',
        full_goal_achieved=False,original_success_preservation=retained,
        no_regression_in_measured_original_successes=bool(protocol['passed'] and not lost),
        original_denominator_note='All original successful environment/phase IDs remain in each criterion denominator; any candidate failure/incomplete record is not retained.',
        counts=dict(environments=len(r),reference_survived=len(rg['survival']),candidate_survived=len(cg['survival']),
            lost_original_survivors=len(rg['survival']-cg['survival']),reference_final_v3_1_balanced=len(rg['final_v3_1_balance']),
            candidate_final_v3_1_balanced=len(cg['final_v3_1_balance']),moving_environments=sum(row['vx']!=0. for row in cases.values()),
            phase_pairs=len(phases),common_complete_phase_pairs=sum(item['both_complete'] for item in phases),
            missing_or_incomplete_phase_pairs=len(missing)),
        missing_or_incomplete_phases=missing,common_complete_phase_error_summary=aggregate,
        moving_phase_descriptive_deltas=moving,all_phase_comparisons=phases,
        per_environment=[dict(env=env,case=r[env]['case'],reference={k:v for k,v in r[env].items() if k!='phases'},
            candidate={k:v for k,v in c[env].items() if k!='phases'}) for env in sorted(r)],
        confirmed_gates=dict(steady_pitch_abs_max_deg=5.,pure_parking_world_horizontal_speed_max_m_s=.05,
            final_balance='v3.1 final continuous1s: every wheel>=max(.1weight,5N); total.8..1.2weight; base<=1N; max|roll,pitch|<=35deg; xy omega norm<=.5rad/s'),
        descriptive_only=['Nonzero vx MAE/p95 and candidate-minus-reference deltas are not acceptance cutoffs.',
            'Driven-leg one-second peak-to-peak values/deltas have no user-confirmed 2-degree or other numeric gate.',
            'Common-complete phase averages do not include missing phases and cannot prove preservation.'],
        missing_full_goal_evidence=missing_coverage,physics_steps_executed=0,parameter_updates=0,
        comparator_sha256=digest(Path(__file__)))


def markdown(report):
    lines=[f"比较结果：{report['verdict']}。完整目标达成：否。",'',
        f"同初态与协议核对：{'通过' if report['protocol']['passed'] else '未通过'}；固定物理数组 {len(report['protocol']['fixed_arrays'])} 项。",'',
        '| 保留项目 | 原成功 | 新成功 | 保留 | 丢失 |','|---|---:|---:|---:|---:|']
    for name,row in report['original_success_preservation'].items():
        lines.append(f"| {name} | {row['reference_successes']} | {row['candidate_successes']} | {row['retained']} | {row['lost_count']} |")
    lines+=['',f"原移动环境 {report['counts']['moving_environments']} 个；共同完整阶段 {report['counts']['common_complete_phase_pairs']}；缺失/不完整阶段 {report['counts']['missing_or_incomplete_phase_pairs']}。",'',
        '失败和不完整样本保留在原成功分母中；MAE和腿幅仅描述变化，没有使用尚未确认的精度或2°摆幅作为门槛。','',
        '逐环境/阶段丢失ID、全部缺失阶段、原始阶段统计与差值、来源差异见JSON报告。','']
    for name,row in report['original_success_preservation'].items():
        lines += [f"{name} 丢失：{json.dumps(row['lost'],ensure_ascii=False)}",'']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New JSON report path')
    parser.add_argument('--markdown',type=Path,help='Optional new Markdown companion path')
    args=parser.parse_args()
    if args.output.exists() or (args.markdown is not None and args.markdown.exists()):
        parser.error('Output exists; use new report paths')
    reference=load_measurement(args.reference);candidate=load_measurement(args.candidate)
    report=compare(reference,candidate)
    for record in (reference,candidate):
        if any(digest(path)!=record['hashes'][name] for name,path in record['paths'].items()):
            raise AssertionError('An input artifact changed during comparison')
    write(args.output,report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True,exist_ok=True)
        with args.markdown.open('x') as stream:stream.write(markdown(report))
    print(json.dumps(dict(output=str(args.output),verdict=report['verdict'],protocol_passed=report['protocol']['passed'],
        counts=report['counts'],preservation={name:{key:row[key] for key in ('reference_successes','candidate_successes','retained','lost_count')}
            for name,row in report['original_success_preservation'].items()},full_goal_achieved=False),ensure_ascii=False))


if __name__=='__main__':
    main()
