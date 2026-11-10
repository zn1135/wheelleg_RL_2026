"""Build smooth empirical two-motor load tables versus length after nominal inverse dynamics."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scripts.agent.real2sim.control_leg_pose_gpu import leg_pose


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    with np.load('artifacts/length-response-fit/20261110-nonlinear-axial-seed-01/inverse.npz') as a:
        time=a['time_s'];q=a['qpos'];v=a['qvel'];required=a['required_motor_nm'];mask=time>1
    with np.load('artifacts/length-response-fit/20261109-inputs-02/inputs.npz') as a:
        mq=a['mq'].astype(int);mv=a['mv'].astype(int);sign=a['sign'];zero=a['zero']
    length=leg_pose(q[:,mq]*sign+zero)[0][:,1];knots=np.linspace(.187,.252,23)
    old=json.loads(Path('artifacts/length-response-fit/20261109-bristle-final-full-01/parameters.json').read_text())[0]
    profiles=[old]
    for dry in (.1,.5,1.):
        for damp in (0.,.5,2.):
            target=-required+dry*np.tanh(v[:,mv[2:]]/.01)+damp*v[:,mv[2:]]
            values=[]
            for x in knots:
                distance=abs(length-x);chosen=mask&(distance<.003)
                if chosen.sum()<3:
                    selected=np.argsort(np.where(mask,distance,np.inf))[:5]
                    chosen=np.zeros(len(length),dtype=bool);chosen[selected]=True
                values.append(np.median(target[chosen],axis=0))
            values=gaussian_filter1d(np.array(values),.7,axis=0,mode='nearest')
            profiles.append(dict(old,remove_baseline_right_load=True,bias_delta_nm=[0.,0.],
                length_stiffness_delta_nm_rad=0.,cross_stiffness_delta_nm_rad=0.,length_quadratic_delta_nm_rad2=0.,
                static_nm=[dry,dry],kinetic_nm=[dry,dry],damping_nms_rad=[damp,damp],seal_static_n=0.,seal_kinetic_n=0.,
                seal_damping_ns_m=0.,seal_friction_route='slider',right_total_length_knots_m=knots.tolist(),right_total_length_load_nm=values.tolist()))
    (args.output/'parameters.json').write_text(json.dumps(profiles,indent=2)+'\n')


if __name__=='__main__':main()
