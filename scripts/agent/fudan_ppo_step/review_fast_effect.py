"""Bind paired high-speed GPU recordings, original walking regression and decoded videos; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--report-initial-mismatches',action='store_true',
        help='Retain and report initial-condition tolerance failures; never treats them as passed pairing')
    args=parser.parse_args();root=args.recording
    protocols=[json.loads((root/name/'protocol.json').read_text()) for name in ('baseline','candidate')]
    assert protocols[0]['cases']==protocols[1]['cases']
    for name,protocol in zip(('baseline','candidate'),protocols):
        audit_name='audit-baseline-final' if name=='baseline' else 'audit-candidate'
        assert json.loads((root/audit_name/'audit.json').read_text())['passed']
        assert protocol['high_speed'] and protocol['max_approach_speed']==3.
        assert protocol['command_height_m']==.35 and protocol['settle_s']==protocol['ramp_s']==0.
        assert protocol['all_networks_unchanged'] and protocol['optimizer_updates']==0
        assert sha(Path(protocol['source_checkpoint']))==protocol['source_sha256']
    difference=0.;ray_misses=[0,0];first_contacts=[];initial_differences=[]
    for case in protocols[0]['cases']:
        starts=[]
        for index,(name,protocol) in enumerate(zip(('baseline','candidate'),protocols)):
            path=root/name/'traces'/(case['id']+'.npz')
            assert sha(path)==protocol['trace_sha256'][case['id']]
            with np.load(path) as trace:
                starts.append(trace['initial'])
                valid=trace['valid'].astype(bool);n=int(valid.sum())
                assert n>0 and valid[:n].all() and not valid[n:].any()
                ray_misses[index]+=int(not trace['terrain_valid'][:n].all())
                if case['height']==.2 and case['angle']==0 and case['friction']==.8:
                    # 高速从t=0直接接近，首接触统计不能跳过原低速诊断的1s静止窗。
                    force=trace['force'][:n].reshape(-1,2);side=trace['side_sub'][:n].reshape(-1).astype(bool)
                    tt=(np.arange(len(force))+1)*.001;event=np.flatnonzero((force>1).any(-1)|side)
                    first=float(tt[event[0]]) if len(event) else None
                    v=np.repeat(trace['linear_world'][:n],10,axis=0)
                    window=(tt>=first-.2)&(tt<=first-.05) if first is not None else np.zeros(len(tt),bool)
                    first_contacts.append(dict(model=name,case=case['id'],command_speed_m_s=case['speed'],first_contact_s=first,
                        precontact_horizontal_speed_m_s=float(np.linalg.norm(v[window,:2],axis=-1).mean()) if window.any() else None,
                        nonwheel_peak_N=force.max(0).tolist()))
        delta=float(abs(starts[0]-starts[1]).max());difference=max(difference,delta)
        initial_differences.append(dict(case=case['id'],height=case['height'],max_abs_difference=delta,
                                       within_predeclared_tolerance=delta<2e-6))
    # 保留未达到初态比较门槛的样本，不放宽门槛；目标200mm另列。
    target_pairing=all(row['within_predeclared_tolerance'] for row in initial_differences if row['height']==.2)
    if not args.report_initial_mismatches:assert target_pairing
    metrics=[json.loads((root/name/'metrics.json').read_text()) for name in ('baseline','candidate')]
    groups=[]
    for height in sorted({r['height'] for r in metrics[0] if r['height']>0}):
        selected=[[r for r in rows if r['height']==height] for rows in metrics]
        groups.append(dict(height_m=height,cases=len(selected[0]),
            landed_before=sum(r['landed'] for r in selected[0]),landed_after=sum(r['landed'] for r in selected[1]),
            task_passed_before=sum(r['task_score']['passed'] for r in selected[0]),
            task_passed_after=sum(r['task_score']['passed'] for r in selected[1])))
    flat=json.loads((root/'flat47/acceptance/acceptance.json').read_text())
    assert flat['total']==47 and len(flat['comparison']['retained'])+len(flat['comparison']['regressed'])==47
    videos=[]
    for name in ('candidate','baseline'):
        folder=root/f'video-{name}-200mm-clear';path=folder/'replay-1x.mp4'
        media=json.loads((folder/'media.json').read_text());reader=imageio.get_reader(path)
        frames=sum(1 for _ in reader);reader.close()
        assert frames==media['frames'] and sha(path)==media['video_sha256']
        videos.append(dict(model=name,frames=frames,duration_s=frames/media['fps'],video_sha256=sha(path)))
    by_speed=[]
    for height in sorted({r['height'] for r in metrics[0] if r['height']>0}):
        for speed in sorted({r['speed'] for r in metrics[0] if r['height']==height}):
            selected=[[r for r in rows if r['height']==height and r['speed']==speed] for rows in metrics]
            by_speed.append(dict(height_m=height,command_speed_m_s=speed,cases=len(selected[0]),
                landed_before=sum(r['landed'] for r in selected[0]),landed_after=sum(r['landed'] for r in selected[1]),
                task_passed_before=sum(r['task_score']['passed'] for r in selected[0]),
                task_passed_after=sum(r['task_score']['passed'] for r in selected[1])))
    write(root/'review.json',dict(passed=True,groups=groups,by_speed=by_speed,first_contacts=first_contacts,
        initial_max_abs_difference=difference,missing_terrain_cases_before_after=ray_misses,
        initial_comparison_tolerance=2e-6,initial_differences=initial_differences,
        all_initials_within_tolerance=all(row['within_predeclared_tolerance'] for row in initial_differences),
        target_200mm_pairing_passed=target_pairing,
        initial_mismatches_reported_without_waiver=args.report_initial_mismatches,
        flat_passed=flat['technical_passed'],flat_total=47,flat_comparison=flat['comparison'],videos=videos,
        optimizer_updates=0,new_physics_steps=0,full_task_accepted=False,
        case_grid=dict(cases_per_model=len(protocols[0]['cases']),
            heights_m=sorted({c['height'] for c in protocols[0]['cases'] if c['height']>0}),
            command_speeds_m_s=sorted({c['speed'] for c in protocols[0]['cases']}),
            angles_deg=sorted({c['angle'] for c in protocols[0]['cases']}),
            frictions=sorted({c['friction'] for c in protocols[0]['cases']})),
        scope='Frozen direct-command forward platform cases in the recorded case_grid, .35m height, nominal parameters,15ms delays; not full randomized or low-speed acceptance.'))
    print(json.dumps(dict(groups=groups,first_contacts=first_contacts,flat_passed=flat['technical_passed'],videos=videos),ensure_ascii=False))


if __name__=='__main__':
    main()
