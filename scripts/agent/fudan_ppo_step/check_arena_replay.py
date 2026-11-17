"""Offline check of heading and delivered commands in a finite original-arena replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.recording/'report.json').read_text())
    report=dict(passed=False,physics_steps=0,optimizer_updates=0,
                scope='Recorded heading/command delivery/finite states only; not climbing or GPU acceptance')
    try:
        assert meta['physics_engine']=='native_mujoco_cpu' and meta['error'] is None
        assert meta['ngeom']==57187
        schedule=meta['command_sequence'];assert schedule
        with np.load(a.recording/'trace.npz',allow_pickle=False) as data:
            state,qpos=data['state'],data['qpos']
            assert len(state)==len(qpos)==meta['policy_steps']==schedule[-1][0]
            assert np.isfinite(state).all() and np.isfinite(qpos).all()
            assert np.allclose(np.diff(state[:,0]),meta['timing']['policy_dt'],atol=1e-9,rtol=0)
            begin=0
            for end,command in schedule:
                assert np.allclose(state[begin:end,1:4],command,atol=1e-7,rtol=0)
                begin=end
            q=qpos[0,3:7]
            yaw=np.degrees(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
            error=(yaw-meta['spawn_yaw_deg']+180)%360-180
            assert abs(error)<1.
        report.update(passed=True,first_frame_yaw_deg=float(yaw),heading_error_deg=float(error),
            command_segments=len(schedule),recorded_policy_steps=meta['policy_steps'],
            checkpoint_sha256=meta['checkpoint_sha256'],
            trace_sha256=hashlib.sha256((a.recording/'trace.npz').read_bytes()).hexdigest())
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
