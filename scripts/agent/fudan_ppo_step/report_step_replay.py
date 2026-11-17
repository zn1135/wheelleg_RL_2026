"""Bind an audited bounded step replay to its checkpoint and fully decoded videos."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.recording
    protocol=json.loads((root/'replay/protocol.json').read_text())
    audit=json.loads((root/'audit/audit.json').read_text())
    checkpoint=json.loads((root/'checkpoint-verification.json').read_text())
    receipt=json.loads((root/'input-receipt.json').read_text())
    rows=json.loads((root/'replay/metrics.json').read_text())
    assert audit['passed'] and checkpoint['passed'] and protocol['optimizer_updates']==0
    assert protocol['all_networks_unchanged'] and protocol['all_finite']
    assert sha(Path(protocol['source_checkpoint']))==protocol['source_sha256']==receipt['checkpoint_sha256']
    groups=[]
    for height in sorted({row['height'] for row in rows if row['height']>0}):
        selected=[row for row in rows if row['height']==height]
        groups.append(dict(height_m=height,total=len(selected),landed=sum(row['landed'] for row in selected),
            task_passed=sum(row['task_score']['passed'] for row in selected),
            strict_parked=sum(row['parked'] for row in selected)))
    videos=[]
    for folder in sorted(root.glob('video-*')):
        media=json.loads((folder/'media.json').read_text());path=folder/'replay-1x.mp4'
        assert sha(path)==media['video_sha256']
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps']==media['fps']
            frames=sum(1 for _ in reader)
        assert frames==media['frames']
        trace_path=root/'replay/traces'/(media['case']+'.npz')
        assert sha(trace_path)==media['trace_sha256']==protocol['trace_sha256'][media['case']]
        row=next(row for row in rows if row['id']==media['case'])
        with np.load(trace_path,allow_pickle=False) as trace:
            n=int(trace['valid'].sum());g=trace['gravity'][n-1]
            pitch=float(np.degrees(np.arcsin(np.clip(g[0],-1,1))))
            roll=float(np.degrees(np.arctan2(-g[1],-g[2])))
            stops=np.flatnonzero(abs(trace['command'][:n,:2]).max(-1)<1e-6)
            last=dict(wheel_x_relative_edge_m=(trace['wheel_pos'][n-1,:,0]-trace['origin'][0]-1.2).tolist(),
                wheel_z_m=trace['wheel_pos'][n-1,:,2].tolist(),
                wheel_ground_height_m=trace['support_height'][n-1].tolist(),
                normal_support=bool(trace['normal_support'][n-1]),roll_deg=roll,pitch_deg=pitch,
                horizontal_speed_m_s=float(np.linalg.norm(trace['linear_world'][n-1,:2])))
        videos.append(dict(directory=folder.name,case=media['case'],frames=frames,fps=media['fps'],
            duration_s=media['duration_s'],full_decode_passed=True,
            first_wheel_top_support_s=row['first_wheel_top_support_s'],
            stop_command_s=float(stops[0]*.01) if len(stops) else None,
            stable_stop_s=row['task_score']['stable_stop_s'],last=last))
    result=dict(evidence_verified=True,checkpoint_iteration=checkpoint['iteration'],
        checkpoint_sha256=receipt['checkpoint_sha256'],source_fingerprint=protocol['source_fingerprint'],
        backend=protocol['backend'],cases=len(rows),groups=groups,videos=videos,
        optimizer_updates=0,full_task_accepted=False,flat47_evaluated=False,
        scope='Bounded nominal forward replay at fixed friction/delay; not a full stair/flat regression or deployment acceptance.')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(groups=groups,videos=len(videos),evidence_verified=True)))


if __name__=='__main__':main()
