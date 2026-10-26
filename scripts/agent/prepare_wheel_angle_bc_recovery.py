#!/usr/bin/env python3
"""Keep full parking-qualified recoveries and separately screened early rise labels."""
import argparse,json,os
from pathlib import Path
import numpy as np
from prepare_bc_wheel_dataset import audit_sources,digest,write


def prepare(args):
    report=json.loads((args.source/'result.json').read_text())
    assert report['teacher_queries_only'] is False and not report['heldout_used']
    assert report['trace_sha256']==digest(args.source/'trace.npz')
    audit_sources(args.source,report)
    assert report.get('wheel_angles') and .2<=report['takeover_s']<2.
    with np.load(args.source/'trace.npz') as archive:arrays={k:archive[k] for k in archive.files}
    time=arrays['time'];rise_time=(time>report['takeover_s'])&(time<=2.)
    last=int(np.flatnonzero(time<=2.)[-1]);rise=[]
    for row in report['rows']:
        i=row['env'];q=row['quality']
        # 起立段需在2秒已进入平衡且连续轮支撑0.5秒，随后完整保持双轮平衡。
        good=bool(q['full'] and q['strict_rise'] and q['final_balance'] and q['finite'] and
            not q['body_contact'] and q['height']<=.03 and q['tilt']<=np.deg2rad(25) and
            arrays['teacher_phase'][last,i]==3 and arrays['balance_hold'][last,i]>=250 and
            not arrays['teacher_fault'][rise_time,i].any() and not arrays['target_clipped'][rise_time,i].any())
        row['rise_qualified']=good;rise.append(good)
    qualified=np.array([r['qualified'] for r in report['rows']])
    allowed=qualified[None,:]|((time<=2.)[:,None]&np.array(rise)[None,:])
    valid=(arrays['alive']&arrays['label_eligible']&arrays['teacher_selected']&allowed&
        (time>report['takeover_s'])[:,None])
    assert valid.sum()>arrays['label_valid'].sum()
    assert np.all(valid[arrays['label_valid']])
    original_labels_sha=digest(args.source/'trace.npz')
    arrays['label_valid']=valid
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'trace.npz',**arrays)
    for path in (args.source/'snapshot').rglob('*'):
        if path.is_file():
            target=args.output/'snapshot'/path.relative_to(args.source/'snapshot')
            target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
    for name in ('plan.json','env.yaml'):os.link(args.source/name,args.output/name)
    report.update(label_scope='qualified_full_recoveries_plus_rise_only_v1',
        valid_correction_frames=int(valid.sum()),trace_sha256=digest(args.output/'trace.npz'),
        rise_qualified=sum(rise),
        derived_from=dict(source=str(args.source),trace_sha256=original_labels_sha,
            result_sha256=digest(args.source/'result.json'),exporter_sha256=digest(Path(__file__))),
        rise_only_rule='2s balance stage and 0.5s continuous wheel support; full episode, strict rise, final balance, contact/height/tilt, and no early teacher fault or target clipping; only actions before 2s',
        note='Parking qualification and every state/action remain unchanged. Only label_valid adds screened early rise labels; no heldout or SMP positive trajectories imported')
    write(args.output/'result.json',report)
    print(json.dumps(dict(qualified_parking=report['qualified'],qualified_rise=report['rise_qualified'],
        valid_correction_frames=report['valid_correction_frames'],optimizer_updates=0),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args())
