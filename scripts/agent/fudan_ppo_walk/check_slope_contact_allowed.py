"""斜坡允许触地规则检查与冻结5700的GPU复测；零训练更新。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import inspect_regions_stage as regional
from . import inspect_critic_scan_stage as inspect
from .inspect_randomization_4000 import write, sha
from .run_torque_ablation import run_command

ROOT=regional.ROOT
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
MODULE='scripts.agent.fudan_ppo_walk.inspect_critic_scan_stage'


def check(out,source):
    manifest=json.loads((source/'evidence-index.json').read_text())['files_sha256']
    tested=0
    for folder in ('terrain','terrain-height-extra'):
        root=inspect.terrain_path(source,'scan',folder,5700)
        for row in json.loads((root/'result.json').read_text())['rows']:
            if row['complete']:continue
            path=root/'traces'/(row['id']+'.npz')
            assert sha(path)==manifest[str(path.relative_to(source))]
            with np.load(path) as trace:
                sub=np.zeros((len(trace['time']),18))
                sub[:,12:14]=trace['roll_pitch_deg'];sub[:,16]=trace['base_contact']
            assert sub[:,16].max()>10 and abs(sub[:,12:14]).max()<60
            assert regional.route_failure(row,sub)==''
            tested+=1
    assert tested==19
    body=np.zeros((10,18));body[:,16]=300.
    assert regional.route_failure(dict(family='rough'),body)=='body_contact'
    body[:,13]=61.
    assert regional.route_failure(dict(family='up'),body)=='tilt_over60'
    assert regional.route_failure(dict(family='down'),body)=='tilt_over60'
    return dict(passed=True,old19_body_contact_events_no_longer_terminate_slopes=True,
        real_old_traces_checked=tested,rough_contact_still_fails=True,tilt_still_fails=True,
        training_collision_reward_unchanged_zero=True,physics_steps=0,optimizer_updates=0)


def report(out,source):
    rows=[];old=[];compared=0;differences=[]
    for folder in ('terrain','terrain-height-extra'):
        new=out/'scan'/folder/'5700'
        previous=inspect.terrain_path(source,'scan',folder,5700)
        current=json.loads((new/'result.json').read_text())
        rows+=current['rows'];old+=json.loads((previous/'result.json').read_text())['rows']
        protocol=json.loads((out/'scan'/(folder+'-protocol.json')).read_text())
        assert protocol['version']=='regional-route-diagnostic-slope-contact-allowed-v2'
        for path in sorted((new/'fixtures').glob('*_initial.npz'))+[new/'initial.npz']:
            with np.load(path) as a,np.load(previous/path.relative_to(new)) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:
                    compared+=1
                    if not np.array_equal(a[key],b[key]):
                        differences.append(dict(path=str(path.relative_to(new)),field=key,
                            max_difference=float(np.abs(a[key]-b[key]).max())))
        for r in current['rows']:
            assert r['body_contact_check_applied']==(r['family']=='rough')
            with np.load(new/'traces'/(r['id']+'.npz')) as d:
                assert all(np.isfinite(d[k]).all() for k in d.files)
                if r['complete']:assert abs(d['time'][-1]-r['duration'])<1e-8
                else:assert d['terminated'][-1]
    byid={r['id']:r for r in old}
    continued=[dict(id=r['id'],old_end_s=float(np.load(inspect.terrain_path(source,'scan',
        'terrain-height-extra' if '_h' in r['id'] else 'terrain',5700)/'traces'/(r['id']+'.npz'))['time'][-1]),
        new_complete=r['complete'],new_failure_reason=r['failure_reason'],new_reached=r['reached'],
        contact_peak_N=r['body_contact_peak_N']) for r in rows if not byid[r['id']]['complete']]
    summary={label:{family:dict(total=sum(r['family']==family for r in data),
        reached=sum(r['reached'] for r in data if r['family']==family),
        passed=sum(r['passed'] for r in data if r['family']==family)) for family in ('up','down','rough')}
        for label,data in [('old_strict',old),('slope_contact_allowed',rows)]}
    result=dict(passed=True,scope='Frozen same5700 policy under changed slope contact diagnostic rules',
        scores_are_not_training_improvement=True,summary=summary,previous19_contact_failures=continued,
        initial_arrays_compared=compared,all_initial_arrays_bitwise_equal=not differences,differences=differences,
        full_finite_traces=True,optimizer_updates=0,training_started=False)
    write(out/'comparison.json',result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'check-source.py')
    write(out/'rule-check.json',check(out,source))
    stages=('audit','terrain','height-sweep')
    try:
        for stage in stages:
            run_command(out,stage,[str(PYTHON),'-u','-m',MODULE,stage,'--arm','scan','--iteration','5700','--output',str(out)])
        result=report(out,source)
        write(out/'status.json',dict(status='completed',optimizer_updates=0,training_started=False,
            source_iteration=5700,finished_unix=time.time()))
        print(json.dumps(result['summary']),flush=True)
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False))
        raise


if __name__=='__main__':main()
