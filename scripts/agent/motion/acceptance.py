"""One GPU acceptance report and one set of motion thresholds for all retained tasks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from .assess import STANDARD as MOTION_STANDARD,THRESHOLDS,phase_verdict,reevaluate
from ct_mjlab.validation import STANDARD

CAPABILITIES=('forward','backward','turn','parking')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(task,source,rows,coverage,backend_verified,*,rise=None,details=None):
    missing=[name for name in CAPABILITIES if name not in coverage]
    passed=sum(bool(row.get('passed')) for row in rows)
    return dict(standard=STANDARD,task=task,validation_backend='mujoco_warp_gpu',
        evaluator_sources={str(Path(__file__).resolve()):sha(__file__),
            str(Path(__file__).with_name('assess.py').resolve()):sha(Path(__file__).with_name('assess.py'))},
        backend_verified=bool(backend_verified),motion_standard=MOTION_STANDARD,thresholds=THRESHOLDS.copy(),
        source_recording=str(source),coverage=sorted(coverage),missing_capabilities=missing,
        total=len(rows),technical_passed=passed,samples=rows,rise=rise,
        technical_thresholds_met=bool(backend_verified and rows and passed==len(rows) and not missing),
        full_task_accepted=False,final_acceptance='pending_user_video_review',
        physics_steps=0,model_forwards=0,optimizer_updates=0,details=details)


def compare_baseline(result,baseline):
    if (result['standard']!=baseline.get('standard') or result['task']!=baseline.get('task')
            or not result.get('cohort_sha256') or result['cohort_sha256']!=baseline.get('cohort_sha256')):
        raise ValueError('Paired comparison requires the same unified standard, task and recorded initial conditions')
    old={r['sample_id']:r['passed'] for r in baseline['samples']}
    new={r['sample_id']:r['passed'] for r in result['samples']}
    if set(old)!=set(new):raise ValueError('Paired comparison sample IDs differ')
    lost=sorted(k for k in old if old[k] and not new[k])
    result['comparison']=dict(regressed=lost,retained=sorted(k for k in old if old[k] and new[k]),
        gained=sorted(k for k in old if not old[k] and new[k]),regression_rejected=bool(lost))
    result['technical_thresholds_met']=bool(result['technical_thresholds_met'] and not lost)
    return result


def canonical_sample(sample,trace):
    """Keep v2 integrity/contact/height gates; replace motion scoring with the shared standard."""
    from scripts.agent.fudan_ppo_walk.score_v2 import score as score_v2,DT,tick
    times=np.asarray(trace['time']) if 'time' in trace else np.empty(0)
    dt=float(times[1]-times[0]) if len(times)>1 else DT
    if not np.isclose(dt,.001,rtol=0,atol=1e-10) and not np.isclose(dt,DT,rtol=0,atol=1e-10):
        return dict(sample_id=sample['id'],passed=False,complete=False,failure_reasons=['unsupported_sample_period'])
    previous=score_v2(sample,trace,dt=dt)
    row=copy.deepcopy(previous);row['protocol']=STANDARD;row['legacy_v2_diagnostic']=previous
    row['motion_phases']=[]
    if not previous.get('complete') or not previous.get('finite'):
        row['passed']=False
        return row
    t=np.asarray(trace['time']);n=len(t);valid=np.ones(n,dtype=bool)
    complete=not previous.get('terminated') and not previous.get('numerical_failure')
    def phase(name,lo,hi,command,parking):
        chosen=np.zeros(n,dtype=bool);chosen[tick(lo,dt):tick(hi,dt)]=True
        result=phase_verdict(vx=trace['linear_body'][:,0],yaw=trace['angular_body'][:,2],
            horizontal_speed=np.linalg.norm(trace['linear_world'][:,:2],axis=1),
            pitch_deg=trace['roll_pitch_deg'][:,1],vx_target=command[0],yaw_target=command[1],
            selected=chosen,valid=valid,complete=complete,parking=parking)
        return dict(name=name,window=[lo,hi],**result)
    for i,segment in enumerate(previous['segments']):
        lo,hi=segment['window'];command=segment['command']
        row['motion_phases'].append(phase(f'segment_{i}',lo,hi,command,command[0]==command[1]==0))
    if 'stop' in previous:
        stop=previous['stop']
        if 'observation_window' in stop:
            lo,hi=stop['observation_window']
            row['motion_phases'].append(phase('parking',lo,hi,[0.,0.,0.],True))
        else:
            row['motion_phases'].append(dict(name='parking',passed=False,failed_checks=['no_complete_parking_window']))
    row['passed']=bool(previous['passed'] and row['motion_phases'] and all(p['passed'] for p in row['motion_phases']))
    row['failure_reasons']=row.get('failure_reasons',[])+[
        'unified_motion_'+p['name'] for p in row['motion_phases'] if not p['passed']]
    return row


def canonical(directory,task):
    from ct_mjlab.fudan_ppo_walk.protocol import samples as fudan_samples
    protocol_path=directory/'protocol.json';protocol=json.loads(protocol_path.read_text())
    execution=json.loads((directory/'execution.json').read_text()) if (directory/'execution.json').exists() else {}
    backend=protocol.get('validation_backend',execution.get('validation_backend'))
    if backend is None and execution.get('engine')=='mjlab':backend='mujoco_warp_gpu'
    if backend!='mujoco_warp_gpu':raise ValueError('GPU recording provenance required; CPU recordings cannot be final acceptance')
    items=protocol['samples'];mode=protocol.get('mode')
    if mode=='formal_987_samples' and (task!='fudan_ppo_walk' or items!=fudan_samples()):
        raise ValueError('The frozen987 matrix belongs to Fudan; do not relabel a different task')
    if not items or len({s['id'] for s in items})!=len(items):raise ValueError('Nonempty unique sample list required')
    rows=[];coverage=set()
    for sample in items:
        for segment in sample['case']['segments']:
            vx,yaw,_=segment['command']
            if vx>0:coverage.add('forward')
            if vx<0:coverage.add('backward')
            if yaw!=0:coverage.add('turn')
        if sample['case']['category'].startswith('stop_'):coverage.add('parking')
        path=directory/'traces'/(sample['id']+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:row=canonical_sample(sample,saved)
            row['trajectory_sha256']=sha(path)
        else:row=dict(sample_id=sample['id'],passed=False,complete=False,failure_reasons=['missing_trajectory'])
        row.update(case=sample['case'],randomized=sample['randomized']);rows.append(row)
    result=envelope(task,directory,rows,coverage,True,details=dict(recording_mode=mode,
        contact_height_protocol='fudan-walk-stage1-v2',source_protocol_sha256=sha(protocol_path),
        phase_windows='Original frozen2s response and5s measurement; original stop windows preserved'))
    groups={}
    for row in rows:
        name=row['case']['category']+(':random' if row['randomized'] else ':nominal')
        group=groups.setdefault(name,dict(total=0,passed=0,complete=0))
        group['total']+=1;group['passed']+=int(row['passed']);group['complete']+=int(row.get('complete',False))
    for name,g in groups.items():
        required=.95 if name.endswith(':random') else 1.
        g.update(required_fraction=required,passed_threshold=bool(g['complete']==g['total'] and g['passed']/g['total']>=required))
    result['groups']=groups
    # Compare decoded fixture arrays, not ZIP timestamps or model/output file names.
    cohort=hashlib.sha256(json.dumps(items,sort_keys=True).encode());all_initial=True
    for sample in items:
        initial=directory/'fixtures'/(sample['id']+'_initial.npz')
        if not initial.exists():all_initial=False;break
        with np.load(initial,allow_pickle=False) as data:
            for key in sorted(data.files):
                value=data[key];cohort.update(key.encode());cohort.update(str(value.dtype).encode())
                cohort.update(str(value.shape).encode());cohort.update(value.tobytes())
    result['cohort_sha256']=cohort.hexdigest() if all_initial else None
    if mode=='formal_task_samples' and not protocol.get('protocol'):
        raise ValueError('Task-specific formal matrix requires its frozen protocol identity')
    result['technical_thresholds_met']=bool(mode in ('formal_987_samples','formal_task_samples')
        and not result['missing_capabilities'] and all(g['passed_threshold'] for g in groups.values()))
    return result


def supermimic(directory):
    raw=json.loads((directory/'summary.json').read_text());motion=reevaluate(directory)
    if raw.get('validation_backend') not in (None,'mujoco_warp_gpu'):raise ValueError('Expected GPU recording')
    # A training-source receipt does not establish the device of a historical evaluation.
    gpu=raw.get('validation_backend')=='mujoco_warp_gpu'
    native={r['env']:r for r in raw['summary']['per_environment']};rows=[];coverage=set()
    for r in motion['rows']:
        original=native[r['env']];vx=original.get('speed_m_s',0.);yaw=original.get('yaw_rate_rad_s',0.)
        if vx>0:coverage.add('forward')
        if vx<0:coverage.add('backward')
        if yaw!=0:coverage.add('turn')
        if vx!=0 and 'parking' in r['phases']:coverage.add('parking')
        rows.append(dict(sample_id=str(r['env']),motion_phases=r['phases'],
            complete=original['survived'],final_balanced=original['final_balanced'],
            passed=bool(r['passed'] and original['final_balanced'])))
    return envelope('supermimic',directory,rows,coverage,gpu,details=dict(source=motion,
        safety_profile='Original final1s wheel balance; four shared motion thresholds; original declared windows retained'))


def bc(directory):
    raw=json.loads((directory/'result.json').read_text());path=directory/'trace.npz'
    if raw.get('validation_backend')!='mujoco_warp_gpu':raise ValueError('BC final acceptance requires a GPU-tagged recording')
    if sha(path)!=raw['trace_sha256']:raise ValueError('BC trace SHA differs')
    rows=[]
    with np.load(path,allow_pickle=False) as data:
        for old in raw['rows']:
            i=old['env'];selected=(data['time']>15.)&(data['time']<=20.)
            if 'command' not in data:
                motion=dict(passed=False,failed_checks=['missing_recorded_command'])
            else:
                command=data['command'][:,i];f=data['features'][:,i]
                gravity=f[:,1:4];pitch=np.rad2deg(np.arcsin(np.clip(gravity[:,0],-1.,1.)))
                motion=phase_verdict(vx=f[:,4],yaw=f[:,9],horizontal_speed=np.linalg.norm(data['qvel'][:,i,:2],axis=-1),
                    pitch_deg=pitch,vx_target=0.,yaw_target=0.,selected=selected,valid=data['alive'][:,i].astype(bool),
                    complete=old['full_20s'] and old['finite'],parking=True,command_vx=command[:,0],command_yaw=command[:,1])
            rows.append(dict(sample_id=str(i),motion_phases=dict(parking=motion),complete=old['full_20s'],
                passed=bool(motion['passed'] and old['final_balance_v31']),rise_passed=old['strict_rise']))
    return envelope('bc',directory,rows,{'parking'},True,rise=dict(status='separate',legacy_strict_rise=raw['strict_rise']),
        details=dict(trace_sha256=sha(path),scope='BC get-up/standing cohort only; no forward/backward/turn coverage'))


def smp(directory):
    path=directory if directory.is_file() else directory/'result.json'
    raw=json.loads(path.read_text())
    if raw.get('validation_backend')!='mujoco_warp_gpu':raise ValueError('SMP final acceptance requires a GPU-tagged recording')
    return envelope('smp',directory,[],set(),True,rise=dict(status='separate',diagnostic=raw),
        details=dict(scope='Recover-only evaluation has no complete motion trace; movement is not measured'))


def isaac(path):
    raw=json.loads(path.read_text());metadata=raw.get('metadata',{})
    if metadata.get('validation_backend')!='physx_gpu':raise ValueError('Isaac GPU provenance required')
    result=envelope('isaac',path,[],set(),True,rise=dict(status='separate',legacy_summary=raw.get('summary')),
        details=dict(source_task=metadata.get('task'),source_sha256=sha(path),
            scope='Legacy get-up/PD diagnostic; missing full motion phases cannot be accepted as walking'))
    result['validation_backend']='physx_gpu'
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--task',choices=('fudan_ppo_walk','supermimic','bc','smp','isaac'),required=True)
    parser.add_argument('--recording',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,help='Prior acceptance.json using the same standard and initial conditions')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if (args.recording/'protocol.json').exists():result=canonical(args.recording,args.task)
    else:
        if args.task=='fudan_ppo_walk':raise ValueError('Fudan recording requires protocol.json and traces/')
        result={'supermimic':supermimic,'bc':bc,'smp':smp,'isaac':isaac}[args.task](args.recording)
    if args.baseline:result=compare_baseline(result,json.loads(args.baseline.read_text()))
    args.output.mkdir(parents=True)
    (args.output/'acceptance.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (args.output/'README.md').write_text(f"# 统一GPU验收\n\n任务：{args.task}；标准：{STANDARD}。\n\n通过样本：{result['technical_passed']}/{result['total']}；缺测能力：{', '.join(result['missing_capabilities']) or '无'}。\n\n技术门槛达标：{result['technical_thresholds_met']}；最终待用户视频验收。起立单列，缺测不视作通过。\n\n[完整报告](acceptance.json)\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','details','rise')},ensure_ascii=False))


if __name__=='__main__':main()
