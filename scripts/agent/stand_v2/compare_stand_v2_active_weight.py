#!/usr/bin/env python3
"""After-training 3999/4999/new paired braking and separate zero-v3.1 report."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.agent.motion.assess import STANDARD
from scripts.agent.stand_v2.assess_chuanliantui_stand_v2 import summarize
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest
from scripts.agent.stand_v2.assess_stand_v2_speed_transients import SEGMENTS, segment_metrics, tracking_accuracy


def braking(rows, start, limit=None):
    rows=[r for r in rows if limit is None or abs(r['speed_m_s'])<=limit]
    segments=[next(s for s in r['segments'] if s['kind']=='stop' and s['start_s']==start) for r in rows]
    gated=[s for s in segments if s['complete_gate_and_live']]
    confirmed=[s for s in gated if s['stopped_within_window']]
    return dict(start_s=start,attempts=len(segments),passed=sum(s['passed'] for s in segments),
        full_gate_attempts=len(gated),full_gate_stop_confirmed=len(confirmed),
        full_gate_confirmation_rate=len(confirmed)/len(gated) if gated else None,
        all_attempts_pass_rate=sum(s['passed'] for s in segments)/len(segments) if segments else None,
        confirmed_times_s=[s['braking_confirmed_s'] for s in confirmed],
        confirmed_paths_m=[s['braking_path_m'] for s in confirmed])


def selfcheck():
    rows=[dict(speed_m_s=.5,segments=[dict(kind='stop',start_s=18.,complete_gate_and_live=True,
        stopped_within_window=True,passed=True,braking_confirmed_s=1.,braking_path_m=.2)]),
        dict(speed_m_s=-.5,segments=[dict(kind='stop',start_s=18.,complete_gate_and_live=True,
        stopped_within_window=False,passed=False,braking_confirmed_s=None,braking_path_m=.6)]),
        dict(speed_m_s=1.,segments=[dict(kind='stop',start_s=18.,complete_gate_and_live=False,
        stopped_within_window=False,passed=False,braking_confirmed_s=None,braking_path_m=None)])]
    d=braking(rows,18.)
    assert d['attempts']==3 and d['full_gate_attempts']==2 and d['full_gate_confirmation_rate']==.5
    assert braking(rows,18.,.5)['attempts']==2
    assert braking([],18.)['full_gate_confirmation_rate'] is None
    trace=np.zeros((1000,1,15))
    trace[:,:,:2]=50.
    trace[:,:,7]=.15  # 原v3.1允许慢移，不用静止速度否决零速平衡。
    trace[:,:,12:14]=1.
    trace[:,:,14]=-1.
    rollout=dict(episodes=[dict(ever_isaac_stood=True,survived_to_end=True)])
    assert summarize(trace,np.array([100.]),rollout,.002)[0]['passed']
    trace[-1,:,5]=.6
    assert not summarize(trace,np.array([100.]),rollout,.002)[0]['passed']
    print('Braking-count reducer selfcheck passed; no simulation or learning')


def main(args):
    accelerations=[]
    for acceleration in ('1','0.5'):
        loaded=[]
        for label in args.labels:
            folder=args.root/f'{label}-a{acceleration}'
            d=json.loads((folder/'result.json').read_text())
            assert d['parameter_updates']==d['gate_violations']==d['slope_violations']==0
            assert d['acceleration_m_s2']==float(acceleration)
            assert digest(d['checkpoint'])==d['checkpoint_sha256']
            initial=np.load(folder/'initial.npz')
            trajectory=np.load(folder/'trajectories.npz')
            # Independently recompute zero-v3.1 from the saved500Hz physical trace.
            p=trajectory['physical']; weight=trajectory['weight']
            old_zero=d['zero_balance_v31']['episodes']
            rollout=dict(episodes=[dict(ever_isaac_stood=r['ever_stood'],survived_to_end=r['full_20s']) for r in old_zero])
            n=len(old_zero)
            computed=summarize(p[:,:n],weight[:n],rollout,.002)
            assert [r['passed'] for r in computed]==[r['passed'] for r in old_zero]
            assert d['zero_balance_v31']['passed']==sum(r['passed'] for r in computed)
            loaded.append((label,d,initial,trajectory))
        try:
            _,reference,initial,trajectory=loaded[0]
            for label,d,x,t in loaded[1:]:
                for key in ('seed','num_envs','conditions','duration_s','physical_fields','thresholds'):
                    assert d[key]==reference[key],key
                assert set(x.files)==set(initial.files)
                assert all(np.array_equal(x[k],initial[k]) for k in x.files)
                assert np.array_equal(t['weight'],trajectory['weight'])
                assert np.array_equal(t['moving'],trajectory['moving'])
            models=[]
            for label,d,x,t in loaded:
                moving=d['moving']['episodes']
                # 从轨迹复算新增精度指标，也兼容旧评估产物没有新字段的情况。
                physical, command = t['physical'], t['command']
                accuracy_rows = [dict(speed_m_s=r['speed_m_s'], segments=[segment_metrics(
                    physical[round(a/.002):round(b/.002), r['env']],
                    command[round(a/.002):round(b/.002), r['env']],
                    r['speed_m_s']*multiplier, a, kind, .002)
                    for a,b,kind,multiplier in SEGMENTS]) for r in moving]
                for row in accuracy_rows:
                    row["passed"] = all(segment["passed"] for segment in row["segments"])
                historical_reported_passed = sum(row["passed"] for row in moving)
                moving = accuracy_rows
                models.append(dict(historical_reported_pipeline_passed=historical_reported_passed,
                    motion_standard=STANDARD, label=label,checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],
                    tracking_accuracy=tracking_accuracy(accuracy_rows),
                    zero_v31=dict(total=d['zero_balance_v31']['total'],passed=d['zero_balance_v31']['passed'],
                        ever_stood=sum(r['ever_stood'] for r in d['zero_balance_v31']['episodes'])),
                    within_vx05=dict(total=sum(abs(r['speed_m_s'])<=.5 for r in moving),
                        full_pipeline_passed=sum(r['passed'] and abs(r['speed_m_s'])<=.5 for r in moving),
                        braking=[braking(moving,s,.5) for s in (18.,36.)]),
                    all_speeds_braking=[braking(moving,s) for s in (18.,36.)]))
            accelerations.append(dict(acceleration_m_s2=float(acceleration),initial_arrays_equal=True,
                initial_array_count=len(initial.files),weight_equal=True,models=models))
        finally:
            for _,_,x,t in loaded:x.close();t.close()
    result=dict(verified=True,groups=accelerations,
        note='Same seed at two accelerations is not two independent seed replicates. Time/path among confirmed cases is conditional.',
        scope='Paired braking counts/confirmation rates and separate zero-v3.1; not deployment approval')
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--labels',nargs=3,default=('3999','4999','new'))
    p.add_argument('--selfcheck',action='store_true')
    args=p.parse_args()
    if args.selfcheck:selfcheck()
    else:
        if args.root is None or args.output is None:p.error('Need --root and --output')
        main(args)
