"""Matched forward/backward approach to a20cm single platform; no learning."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path
from .replay import execute,write,sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.source/'source',a.output/'source')
    shutil.copy2(a.source/'source-manifest.json',a.output/'source-manifest.json')
    cases=[]
    for reverse in (False,True):
        for angle in (-20,-10,0,10,20):
            for speed in (.3,.5,1.,1.5,2.,2.5):
                cases.append(dict(id=f'{"reverse" if reverse else "forward"}_v{speed:.2f}_a{angle:+03d}',
                    speed=-speed if reverse else speed,angle=180+angle if reverse else angle,
                    spawn=[-.9,-.9*math.tan(math.radians(angle))],cmd_height=.27,
                    reverse=reverse,approach_angle_deg=angle))
    write(a.output/'cases.json',cases)
    write(a.output/'protocol.json',dict(height=.2,width=1.148,depth=2.09,
        explanation='Same world approach direction+X and initial position; reverse robot yaw+180deg and negate body velocity command',
        settled_s=1.,ramp_s=.5,physics_dt=.002,policy_dt=.004,optimizer_updates=0))
    execute(a.output,'matched200',cases,height=.2,width=1.148,seconds=10,deadline=time.time()+300)


if __name__=='__main__':main()
