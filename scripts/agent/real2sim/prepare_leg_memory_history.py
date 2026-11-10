"""Freeze only pre-window measurements for causal elastic-memory initialization; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.reference/'report.json').read_text());session=Path(report['session'])
    meta=json.loads((session/'meta.json').read_text());config=Path(meta['machine_config']['path'])
    assert hashlib.sha256(config.read_bytes()).hexdigest()==report['config_sha256']
    start=report['window_start_board_s'];wire=replay.mirror.FrameParser();frames=[]
    for event in replay.mirror.iter_session_events(session):
        if event['type']!='data':continue
        for f in wire.feed(bytes.fromhex(event['data_hex'])):
            t=replay.mirror.read_u32(f[8:10])/1000
            if t>=start:continue
            if int(f[29])&31!=31 or f[31]<0 or f[31]>=5 or f[30]<0 or f[30]>=20:continue
            if not np.isfinite(f[np.r_[10:18,22:29,132:136]]).all():continue
            frames.append(np.r_[t,f[10:14],f[132:136],f[22:26]])
    raw=np.asarray(frames);assert len(raw)>=100,'No sufficient fresh pre-window history'
    raw=raw[np.r_[True,np.diff(raw[:,0])>0]]
    gaps=np.flatnonzero(np.diff(raw[:,0])>.05)
    if len(gaps):raw=raw[gaps[-1]+1:]
    assert len(raw)>=100,'Last source gap leaves too little reliable history'
    index=None
    for i in range(len(raw)-99):
        if raw[i+99,0]-raw[i,0]>.95 and np.max(np.ptp(raw[i:i+100,1:5],axis=0))<.003:
            index=i;break
    assert index is not None,'History has no stationary initialization interval'
    raw=raw[index:];sign,zero=replay.mirror.load_mapping(config)
    pose=replay.mirror.MirrorPose(args.reference/'fixed-base-model.xml',config)
    q=[]
    for row in raw:
        motor=(row[1:5]-zero)*sign[:4]
        assert all(pose._solve_leg(side,motor[side*2:side*2+2]) for side in (0,1)),'Past measured pose does not close'
        q.append(pose.data.qpos.copy())
    with np.load(args.reference/'replay.npz') as saved:
        q.append(saved['qpos'][0,0].copy())
    t=np.r_[raw[:,0],start];q=np.asarray(q)
    # 平滑只用于估计窗口开始前的速度，不使用窗口后的任何测量。
    velocity=np.gradient(q,t,axis=0)
    if len(q)>=11:velocity=savgol_filter(velocity,11,3,axis=0)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'history.npz',time_board_s=t,qpos=q,qvel=velocity,
                        initial_model_torque_nm=raw[0,5:9]*sign[:4],initial_base_quat=raw[0,9:13],
                        recording_sha256=np.array(report['stream_sha256']),source_model_sha256=np.array(report['source_model_sha256']),
                        config_sha256=np.array(report['config_sha256']))
    result=dict(scope='causal initial-memory state estimation input only; never a future leg-state forcing input',
                session=str(session),reference=str(args.reference),recording_sha256=report['stream_sha256'],source_model_sha256=report['source_model_sha256'],
                time_range_board_s=[float(t[0]),float(t[-1])],selected_window_start_board_s=start,samples=len(q),
                maximum_source_gap_s=float(np.max(np.diff(t))),future_measurements_used=False,environment_steps=0,
                history_sha256=hashlib.sha256((args.output/'history.npz').read_bytes()).hexdigest(),
                limitations=['Only the final contiguous fresh history is used',
                             'Internal memory before the first recorded stationary pose remains conditionally initialized',
                             'Measured-position quantization and derivative smoothing affect the observer'])
    assert t[-1]<=start and np.all(np.diff(t)>0)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
