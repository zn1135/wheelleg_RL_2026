"""Read-only interim GPU comparison against14679 while the authorized training continues."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command
from scripts.agent.fudan_ppo_walk.resume_output_gain_14679 import ROOT,PYTHON,LOG

BASE=ROOT/'artifacts/fudan_ppo_walk/20261112-output045-resume14179-500-01/evaluation'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();output=args.output.resolve()
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir()
    source_sha=sha(args.checkpoint);checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    number=checkpoint['iterations'];assert not checkpoint['partial_update']
    assert checkpoint['primary_updates']==checkpoint['encoder_updates']==number*20
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    model=output/'models'/f'model{number}.pt';shutil.copy2(args.checkpoint,model)
    assert sha(model)==sha(args.checkpoint)==source_sha
    manifest=json.loads((LOG/'source_sha256.json').read_text())
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/') or name.endswith('.xml'):assert sha(ROOT/name)==digest,name
    write(output/'checkpoint.json',dict(source=str(args.checkpoint.resolve()),sha256=source_sha,iteration=number,
        iterations_since14679=number-14679,partial_update=False,finite=True,optimizer_updates=0,
        scope='Interim frozen checkpoint; authorized3000-iteration training continues separately'))
    shutil.copy2(__file__,output/'inspection-launcher.py');py=str(PYTHON);after=output/f'model{number}';before=BASE/'model14679'
    run_command(output,'parking',[py,'-m','scripts.agent.fudan_ppo_walk.diagnose_parking_posture','--checkpoint',str(model),
        '--output',str(after),'--leg-p','60','--leg-d','2','--wheel-p','.045','--wheel-motor-gain','.45','--seconds','600'])
    for group in ('no_external','configured_external'):
        run_command(output,'accept-'+group,[str(ROOT/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_walk',
            '--recording',str(after/group),'--output',str(after/(group+'-acceptance')),
            '--baseline',str(before/(group+'-acceptance')/'acceptance.json')])
    zero=output/f'zero{number}';zero.mkdir();write(output/'profiles.json',[dict(name='output_gain045',armature_scale=2.,wheel_motor_gain=.45)])
    run_command(output,'zero',[py,'-m','scripts.agent.real2sim.probe_video_leg_parameters','--checkpoint',str(model),
        '--output',str(zero/'no_external'),'--leg-p','60','--leg-d','2','--wheel-p','.045','--standstill',
        '--duration-seconds','30','--profiles-json',str(output/'profiles.json'),'--seconds','600'])
    write(zero/'run.json',dict(checkpoint=str(model),sha256=source_sha,leg_p=60.,leg_d=2.,wheel_p=.045,wheel_motor_gain=.45))
    run_command(output,'accept-zero',[str(ROOT/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_walk','--recording',str(zero/'no_external'),
        '--output',str(zero/'no_external-acceptance'),'--baseline',str(BASE/'zero14679/no_external-acceptance/acceptance.json')])
    run_command(output,'leg-motion',[py,'-m','scripts.agent.fudan_ppo_walk.report_parking_leg_motion',
        '--recordings',str(before),str(after),'--output',str(output/'leg-motion')])
    run_command(output,'parking-comparison',[py,'-m','scripts.agent.fudan_ppo_walk.compare_parking_runs','--before',str(before),
        '--after',str(after),'--leg-motion',str(output/'leg-motion/leg-motion.json'),'--output',str(output/'comparison')])
    run_command(output,'zero-response',[py,'-m','scripts.agent.real2sim.report_wheel_motor_response',
        '--recordings',str(BASE/'zero14679/no_external'),str(zero/'no_external'),'--output',str(output/'zero-response')])
    yaw_rows=[]
    for n,folder in ((14679,BASE/'zero14679/no_external'),(number,zero/'no_external')):
        for s in json.loads((folder/'protocol.json').read_text())['samples']:
            with np.load(folder/'traces'/f'{s["id"]}.npz') as tr:
                w,x,y,z=tr['qpos'][:,3:7].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                yaw_rows.append(dict(model=n,sample_id=s['id'],yaw_total_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                    yaw_last5s_deg=float(np.rad2deg(yaw[-1]-yaw[-501]))))
    write(output/'zero-yaw-comparison.json',dict(rows=yaw_rows,physics_steps=0,policy_forwards=0))
    run_command(output,'zero-video',[py,'-m','scripts.agent.fudan_ppo_walk.render_parking_comparison',
        '--before',str(BASE/'zero14679'),'--after',str(zero),'--samples','output_gain045_h22','output_gain045_h26','output_gain045_h30',
        '--output',str(output/'zero-videos')])
    run_command(output,'parking-video',[py,'-m','scripts.agent.fudan_ppo_walk.render_parking_comparison',
        '--before',str(before),'--after',str(after),'--samples','parking_2','parking_5','parking_8','--output',str(output/'parking-videos')])
    write(output/'status.json',dict(status='complete',iteration=number,sha256=source_sha,new_GPU_trajectories=21,
        reused_baseline_trajectories=21,optimizer_updates=0,full_behavior_accepted=False))


if __name__=='__main__':main()
