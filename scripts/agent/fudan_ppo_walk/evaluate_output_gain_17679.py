"""Paired GPU parking/zero-command validation after the authorized output-gain continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command
from scripts.agent.fudan_ppo_walk.resume_output_gain_14679 import ROOT,PYTHON,MODEL,MODEL_SHA,LOG


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    experiment=a.experiment.resolve();audit=json.loads((experiment/'resume-audit.json').read_text())
    assert audit['passed'] and audit['final_iteration']==17679 and audit['additional_iterations']==3000
    assert sha(MODEL)==MODEL_SHA and sha(LOG/'model_final.pt')==audit['model_sha256']
    source_manifest=json.loads((LOG/'source_sha256.json').read_text())
    for name,digest in source_manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/') or name.endswith('.xml'):
            assert sha(ROOT/name)==digest,name
    output=experiment/'evaluation';output.mkdir(exist_ok=False);(output/'models').mkdir()
    shutil.copy2(__file__,output/'evaluation-launcher.py')
    helpers=[Path(__file__),*[ROOT/'scripts/agent'/name for name in (
        'fudan_ppo_walk/diagnose_parking_posture.py','fudan_ppo_walk/evaluate.py',
        'fudan_ppo_walk/compare_parking_runs.py','fudan_ppo_walk/render_parking_comparison.py',
        'fudan_ppo_walk/report_parking_leg_motion.py','fudan_ppo_walk/score_v2.py',
        'fudan_ppo_walk/run_torque_ablation.py','fudan_ppo_walk/resume_output_gain_14679.py',
        'real2sim/probe_video_leg_parameters.py','real2sim/report_wheel_motor_response.py',
        'real2sim/report_video_leg_reproduction.py','motion/acceptance.py','motion/assess.py')]]
    helper_hashes={}
    for source in helpers:
        relative=source.relative_to(ROOT);target=output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        helper_hashes[str(relative)]=sha(target)
    write(output/'source_sha256.json',helper_hashes)
    write(output/'profiles.json',[dict(name='output_gain045',armature_scale=2.,wheel_motor_gain=.45)])
    models=[]
    for number,source in ((14679,MODEL),(17679,LOG/'model_final.pt')):
        model=output/'models'/f'model{number}.pt';shutil.copy2(source,model);assert sha(model)==sha(source)
        models.append((number,model))
    py=str(PYTHON)
    for number,model in models:
        folder=output/f'model{number}'
        run_command(output,f'parking-{number}',[py,'-m','scripts.agent.fudan_ppo_walk.diagnose_parking_posture',
            '--checkpoint',str(model),'--output',str(folder),'--leg-p','60','--leg-d','2',
            '--wheel-p','.045','--wheel-motor-gain','.45','--seconds','360'])
        for group in ('no_external','configured_external'):
            cmd=[str(ROOT/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_walk',
                '--recording',str(folder/group),'--output',str(folder/(group+'-acceptance'))]
            if number==17679:cmd.extend(['--baseline',str(output/'model14679'/(group+'-acceptance')/'acceptance.json')])
            run_command(output,f'accept-{number}-{group}',cmd)
        # 零指令30秒单独采集，防止把制动后的暂态当成持续站立表现。
        zero=output/f'zero{number}';zero.mkdir()
        run_command(output,f'zero-{number}',[py,'-m','scripts.agent.real2sim.probe_video_leg_parameters',
            '--checkpoint',str(model),'--output',str(zero/'no_external'),'--leg-p','60','--leg-d','2','--wheel-p','.045',
            '--standstill','--duration-seconds','30','--profiles-json',str(output/'profiles.json'),'--seconds','240'])
        # 共用视频渲染器需要的模型来源；轨迹与物理夹具保留在原采集目录。
        write(zero/'run.json',dict(checkpoint=str(model),sha256=sha(model),leg_p=60.,leg_d=2.,wheel_p=.045,wheel_motor_gain=.45))
        cmd=[str(ROOT/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_walk',
            '--recording',str(zero/'no_external'),'--output',str(zero/'no_external-acceptance')]
        if number==17679:cmd.extend(['--baseline',str(output/'zero14679/no_external-acceptance/acceptance.json')])
        run_command(output,f'accept-zero-{number}',cmd)
    before=output/'model14679';after=output/'model17679'
    run_command(output,'leg-motion',[py,'-m','scripts.agent.fudan_ppo_walk.report_parking_leg_motion',
        '--recordings',str(before),str(after),'--output',str(output/'leg-motion')])
    run_command(output,'parking-comparison',[py,'-m','scripts.agent.fudan_ppo_walk.compare_parking_runs',
        '--before',str(before),'--after',str(after),'--leg-motion',str(output/'leg-motion/leg-motion.json'),
        '--output',str(output/'comparison')])
    run_command(output,'zero-response',[py,'-m','scripts.agent.real2sim.report_wheel_motor_response',
        '--recordings',str(output/'zero14679/no_external'),str(output/'zero17679/no_external'),'--output',str(output/'zero-response')])
    yaw_rows=[]
    for number,_ in models:
        folder=output/f'zero{number}/no_external'
        for sample in json.loads((folder/'protocol.json').read_text())['samples']:
            with np.load(folder/'traces'/f'{sample["id"]}.npz') as trace:
                w,x,y,z=trace['qpos'][:,3:7].T
                yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                yaw_rows.append(dict(model=number,sample_id=sample['id'],
                    yaw_total_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                    yaw_last5s_deg=float(np.rad2deg(yaw[-1]-yaw[-501])),
                    last5s_horizontal_displacement_m=float(np.linalg.norm(trace['position'][-1,:2]-trace['position'][-2501,:2]))))
    write(output/'zero-yaw-comparison.json',dict(rows=yaw_rows,physics_steps=0,policy_forwards=0))
    run_command(output,'parking-video',[py,'-m','scripts.agent.fudan_ppo_walk.render_parking_comparison',
        '--before',str(before),'--after',str(after),'--samples','parking_2','parking_5','parking_8','--output',str(output/'parking-videos')])
    run_command(output,'zero-video',[py,'-m','scripts.agent.fudan_ppo_walk.render_parking_comparison',
        '--before',str(output/'zero14679'),'--after',str(output/'zero17679'),
        '--samples','output_gain045_h22','output_gain045_h26','output_gain045_h30','--output',str(output/'zero-videos')])
    external=json.loads((after/'configured_external-acceptance/acceptance.json').read_text())
    fallen=[r['sample_id'] for r in external['samples'] if r.get('terminated') or 'tilt_over60deg' in r.get('failure_reasons',[])]
    if fallen:
        run_command(output,'external-failure-video',[py,'-m','scripts.agent.fudan_ppo_walk.render_parking_comparison',
            '--before',str(before),'--after',str(after),'--group','configured_external',
            '--samples',fallen[0],'--output',str(output/'external-failure-video')])
    write(output/'status.json',dict(status='complete',model_sha256={str(n):sha(m) for n,m in models},
        GPU_trajectories=42,training_updates=0,full_behavior_accepted=False))


if __name__=='__main__':main()
