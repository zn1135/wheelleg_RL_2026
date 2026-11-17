"""Measure hip-to-knee direction in recorded standing poses; kinematics only, no simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    reference=Path('artifacts/fudan_reference_step/20261115-reverse-01/runs/matched200')
    local=Path('artifacts/fudan_ppo_step/20261115-effect500-01/candidate-h30')
    inputs=[('reference',reference,'forward_v0.30_a+00.npz','base_Link_del',
             [('lf0_Joint','lf1_Joint','l_wheel_Link'),('rf0_Joint','rf1_Joint','r_wheel_Link')]),
            ('local',local,'traces/forward_h0.200_v0.30_a+0.npz','robot/base_link',
             [('robot/lf0','robot/lf1','robot/lfwheel'),('robot/rf0','robot/rf1','robot/rfwheel')])]
    rows=[]
    for name,folder,file,base,sides in inputs:
        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));data=mujoco.MjData(model)
        with np.load(folder/file) as trace:
            data.qpos[:]=trace['initial'] if 'initial' in trace else trace['qpos'][0]
        mujoco.mj_kinematics(model,data)
        rotation=data.xmat[model.body(base).id].reshape(3,3)
        vectors=[]
        for hip,knee,wheel in sides:
            h=data.xanchor[model.joint(hip).id]
            vectors.append(dict(hip=hip,hip_to_knee_body_m=((data.xanchor[model.joint(knee).id]-h)@rotation).tolist(),
                                hip_to_wheel_body_m=((data.xpos[model.body(wheel).id]-h)@rotation).tolist()))
        rows.append(dict(model=name,model_sha256=sha(folder/'model.mjb'),trace=str(folder/file),
                         trace_sha256=sha(folder/file),vectors=vectors))
    opposite=all(a['hip_to_knee_body_m'][0]*b['hip_to_knee_body_m'][0]<0
                 for a,b in zip(rows[0]['vectors'],rows[1]['vectors']))
    assert opposite
    report=dict(standing_knee_fold_direction_opposite=True,rows=rows,physics_steps=0,optimizer_updates=0,
                scope='Recorded starting poses in each body frame; different poses/robots. No dynamical reachability proof.')
    with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
