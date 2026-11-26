"""按主跳下降沿、实际承重及首个 WALK 控制步对齐每个 1 ms 接触样本。"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from . import attribute_contacts as probe
from . import evaluate, handoff, measure_flat


def timeline(trace, modes):
    start = int(np.flatnonzero(modes)[0])
    expected, confirmed = handoff.expected_landing_mode(trace,start)
    assert confirmed >= 0 and np.array_equal(expected,modes)
    event = max(measure_flat.events(trace,modes),key=lambda e:e['peak_lower_wheel_m'])
    assert event['landed']
    lift = round(1000*event['takeoff_s'])
    wheel = lift+round(1000*event['airtime_s'])
    end = wheel+1000
    assert end <= len(trace)
    switch = 10*(int(np.flatnonzero(modes)[-1])+1)
    assert switch == (confirmed+9)*10 and wheel <= switch < end
    apex = lift+trace[lift:wheel,6:8].min(1).argmax()
    other = np.flatnonzero(trace[apex:end,10]>1.)+apex
    return int(apex),wheel,switch,other


def run(output, reference, checkpoint):
    output.mkdir(parents=True,exist_ok=False)
    prev=json.loads((reference.parent/'report.json').read_text())
    assert sha(checkpoint)==prev['model_sha256']
    walk,jump,saved=evaluate.policies(checkpoint,'cuda:0')
    assert saved['iterations']==4200
    raw=output/'gpu'
    with patch.object(handoff,'LANDING',dict(handoff.LANDING,buffer_ticks=8)):
        probe.collect(raw,reference,None,(walk,jump),True,flat_reference=reference,landing_handoff=True)
        meta=json.loads((raw/'cases.json').read_text())
        ground=meta['ground_names']
        wheels=meta['wheel_names']
        bodies=meta['solver_body_names']
        assert set(wheels)=={'lfwheel','rfwheel'}
        records=[]
        for r in meta['cases']:
            key=r['case']
            with np.load(raw/(key+'.npz')) as d,np.load(reference/(key+'.npz')) as old:
                t=d['trace']
                apex,wheel,switch,other=timeline(t,d['mode'])
                _,ow,os,orig=timeline(old['trace'],old['mode'])
                assert other.size and orig.size
                hit=int(other[0])
                force=d['ground_force_N']
                pairs=d['solver_ground']
                # 不用求解器中零载荷的预接触代替 >1 N 的首次承重。
                raw_first=int(apex+np.flatnonzero(pairs[apex:wheel+1000].any(1))[0])
                first=min(wheel,hit)
                parts=probe.names_at(wheels,t[first,8:10]>.5)+probe.names_at(ground,force[first]>1.)
                nonwheel=probe.names_at(ground,force[hit]>1.)
                for tick,names in ((first,parts),(hit,nonwheel)):
                    assert all(pairs[tick,bodies.index(name)] for name in names)
                assert r['loaded_ground_without_raw_pair_samples']==0
                assert r['initial_exact']
                records.append(dict(case=key,wheel_ms=wheel,first_load='/'.join(parts),
                    raw_first_ms=raw_first,raw_first_parts='/'.join(probe.names_at(bodies,pairs[raw_first])),
                    nonwheel_ms=hit,first_nonwheel='/'.join(nonwheel),
                    all_nonwheel='/'.join(probe.names_at(ground,(force[apex:wheel+1000]>1.).any(0))),
                    walk_ms=switch,hit_after_wheel_ms=hit-wheel,hit_minus_walk_ms=hit-switch,
                    contact_ms_jump=int(np.sum(other<switch)),contact_ms_walk=int(np.sum(other>=switch)),
                    first_hit_vz=float(t[hit,19]),switch_vz=float(t[switch,19]),
                    original_hit_minus_walk_ms=int(orig[0])-os,original_hit_after_wheel_ms=int(orig[0])-ow,
                    original_jump_contact_ms=int((orig<os).sum()),original_walk_contact_ms=int((orig>=os).sum()),
                    mode_exact=r['mode_exact'],trace_exact=r['trace_exact']))
    with (output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    summary=dict(worlds=len(records),initial_exact=len(records),dt_ms=1,optimizer_updates=0,
        checkpoint_sha256=sha(checkpoint),walk_sha256=sha(evaluate.WALK),
        first_load=dict(Counter(r['first_load'] for r in records)),
        raw_first=dict(Counter(r['raw_first_parts'] for r in records)),
        first_nonwheel=dict(Counter(r['first_nonwheel'] for r in records)),
        all_nonwheel=dict(Counter(r['all_nonwheel'] for r in records)),
        onset_before_walk=sum(r['hit_minus_walk_ms']<0 for r in records),
        original_onset_before_walk=sum(r['original_hit_minus_walk_ms']<0 for r in records),
        trace_exact=sum(r['trace_exact'] for r in records),mode_exact=sum(r['mode_exact'] for r in records))
    for k in ('hit_after_wheel_ms','hit_minus_walk_ms','contact_ms_jump','contact_ms_walk','first_hit_vz','switch_vz',
              'original_hit_minus_walk_ms','original_hit_after_wheel_ms','original_jump_contact_ms','original_walk_contact_ms'):
        summary[k+'_min_median_max']=np.percentile([r[k] for r in records],[0,50,100]).tolist()
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('output','reference','checkpoint'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.reference,args.checkpoint)
