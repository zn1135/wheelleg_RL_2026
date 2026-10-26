#!/usr/bin/env python3
"""Audit actual first-recovery wheel clearing in fresh teacher records; no training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from ct_mjlab.bc import (ActionDataset,check_handover_wheel_pairs,load_actor,check_aligned_position_pairs,
                        ALIGNED_LABEL_REFERENCE,LABEL_REFERENCE,check_label_reference)
from ct_mjlab import wheel_angles
from prepare_bc_wheel_dataset import audit_sources


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check(args):
    m=json.loads(args.dataset.read_text());report=json.loads((args.dataset.parent/'report.json').read_text())
    assert report['complete'] and report['optimizer_updates']==0
    assert m['observation_contract']==wheel_angles.HANDOVER_OBSERVATION_CONTRACT
    source=audit_sources(args.dataset.parent,m);rows=[];difference=0.
    aligned=m['label_reference']==ALIGNED_LABEL_REFERENCE
    for row in m['episodes']+report['rejected_episodes']:
        path=args.dataset.parent/row['path'];assert digest(path)==row['sha256']
        with np.load(path) as d:
            valid=d['alive'];angles=check_handover_wheel_pairs(d,d['source_motor_q'],d['observation'],valid)
            if aligned:check_aligned_position_pairs(d,valid)
            np.testing.assert_array_equal(d['action'],d['label'])
            np.testing.assert_array_equal(d['action'],d['applied_action'])
            assert not d['actuator_torque'][:20].any()
            np.testing.assert_array_equal(d['observation_time'],d['action_time'])
            difference=max(difference,float(np.max(np.abs(angles.clip(-100,100)[valid]-d['observation'][valid,14:16]))))
            latched=d['wheel_angle_handover_latched']
            events=np.flatnonzero(latched & ~np.concatenate(([False],latched[:-1])) & valid)
            assert len(events)<=1
            done=np.flatnonzero((d['phase']==3)&valid)
            if len(events):
                index=events[0];assert not d['observation'][index,14:16].any()
                assert np.max(np.abs(angles[index:][valid[index:]]))>0
                # 人为破坏参考必须被拒绝，避免只信文件里的观测。
                values={k:d[k].copy() for k in ('wheel_angle_origin','wheel_angle_handover_latched','wheel_angle_recovered')}
                values['wheel_angle_origin'][index:,0]+=.1
                try:check_handover_wheel_pairs(values,d['source_motor_q'],d['observation'],valid)
                except AssertionError:pass
                else:raise AssertionError('Corrupt wheel reference was accepted')
            rows.append(dict(env=row['env'],qualified=row['qualified'],reanchor_count=len(events),
                reanchor_time_s=float(d['observation_time'][events[0]]) if len(events) else None,
                teacher_done_time_s=float(d['action_time'][done[0]]) if len(done) else None))
            if aligned:
                lqr_events=np.flatnonzero(d['lqr_position_reference_event'] & valid)
                np.testing.assert_array_equal(events,lqr_events)
                rows[-1]['lqr_position_reset_time_s']=float(d['action_time'][lqr_events[0]]) if len(lqr_events) else None
                rows[-1]['wheel_lqr_reset_difference_s']=(float(d['action_time'][lqr_events[0]]-
                    d['observation_time'][events[0]]) if len(events) else None)
    data=ActionDataset([args.dataset],require_reference=True)
    actor,saved=load_actor(args.old_checkpoint)
    assert saved['dataset']['observation_contract']==wheel_angles.OBSERVATION_CONTRACT
    assert saved['dataset']['observation_contract']!=data.description['observation_contract']
    if aligned:
        try:check_label_reference([m,dict(m,label_reference=LABEL_REFERENCE)],[])
        except ValueError:pass
        else:raise AssertionError('Aligned and historical LQR labels must not mix')
    result=dict(passed=True,optimizer_updates=0,records=len(rows),reanchored=sum(r['reanchor_count'] for r in rows),
        max_observation_error_rad=difference,rows=rows,dataset=data.description,source_audit=source,
        old_checkpoint_sha256=digest(args.old_checkpoint),old_checkpoint_loads_as_v1=True,
        shared_position_reference_aligned=aligned,
        corrupt_reference_rejected=True,source_checker_sha256=digest(__file__),
        scope='Actual teacher20 interface precheck, not a new trained student or full parking acceptance. Shared recovered trigger is separate from internal LQR DONE.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='dataset'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','old-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
