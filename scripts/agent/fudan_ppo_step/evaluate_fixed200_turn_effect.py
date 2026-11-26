"""4200固定200mm配方的冻结GPU转向/登台对照，不执行学习。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
MODELS={
    '3200': ROOT/'artifacts/fudan_ppo_step/20261117-v7-final3200-01/training/model_final.pt',
    '4200': ROOT/'logs/mjlab/fudan_ppo_step/20261117_fixed200_turn_v8_4096_resume3200_to4200_01/model_final.pt',
}
RUN=MODELS['4200'].parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def turn_items(*_):
    rows=[]
    for height in (.26,.35):
        commands=[('spin',0.,yaw) for yaw in (-2.,-1.,-.5,.5,1.,2.)]
        commands += [('turn',vx,yaw) for vx in (-1.,1.) for yaw in (-1.,1.)]
        commands += [('zero',0.,0.)]
        for category,vx,yaw in commands:
            name=f'{category}_v{vx:+g}_y{yaw:+g}_h{height:.2f}'
            segments=[dict(start=0.,duration=7.,command=[vx,yaw,height])]
            duration=30. if category=='zero' else 15.
            if category=='zero':segments[0]['duration']=duration
            else:segments.append(dict(start=7.,duration=8.,command=[0.,0.,height]))
            rows.append(dict(id=name,case=dict(id=name,category=category,segments=segments,duration=duration),
                             seed=50001,randomized=False))
    return rows


def turns(args):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from scripts.agent.fudan_ppo_walk import evaluate
    from scripts.agent.motion.acceptance import canonical
    checkpoint=MODELS[args.model];digest=sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert not saved['partial_update'] and saved['iterations']==int(args.model)
    conditions=from_record(saved['disturbances'])
    assert not conditions.pushes and conditions.external_force_range==conditions.external_torque_range==(0.,0.)
    cfg=saved['control'];assert (cfg['leg_stiffness'],cfg['leg_damping'])==(20.,1.)
    original=evaluate.protocol_record
    def protocol(items,diagnostic):
        result=original(items,diagnostic)
        result.update(diagnostic_protocol='fixed200-turn-h26-h35-v1',
            checkpoint_sha256=sha(checkpoint),evaluation_script_sha256=sha(__file__),
            measurement='Tracking[2,7)s; commanded stop[7,15)s; zero[2,30)s',
            optimizer_updates=0)
        return result
    request=SimpleNamespace(checkpoint=checkpoint,output=args.output,deadline=args.deadline,
        diagnostic=True,diagnostic_height=None,batch_size=22,selection=None,
        leg_load_profile='legacy',response_parameters=None)
    torch.set_num_threads(1)
    with patch('torch.optim.Adam.step',side_effect=AssertionError('Evaluation cannot learn')),\
         patch.object(evaluate,'evaluation_items',turn_items),\
         patch.object(evaluate,'from_sim2sim_config',return_value=conditions),\
         patch.object(evaluate,'protocol_record',protocol),\
         patch.object(FudanWalkCfg.sim,'wheel_model','mechanical'):
        evaluate.gpu(request)
    result=canonical(args.output,'fudan_ppo_walk')
    write(args.output/'acceptance.json',result)
    rows=[]
    for sample in turn_items():
        with np.load(args.output/'traces'/(sample['id']+'.npz')) as trace:
            t=trace['time'];motion=(t>=2)&(t<7);tail=(t>=10)&(t<15)
            vx,yaw,height=sample['case']['segments'][0]['command']
            # 直接报告实际转速/位移和停车，不用奖励代替跟踪。
            rows.append(dict(id=sample['id'],height=height,vx=vx,yaw=yaw,
                actual_yaw_mean=float(trace['angular_body'][motion,2].mean()),
                yaw_mae=float(abs(trace['angular_body'][motion,2]-yaw).mean()),
                vx_mae=float(abs(trace['linear_body'][motion,0]-vx).mean()),
                height_mean=float(trace['position'][motion,2].mean()),
                pitch_peak_deg=float(abs(trace['roll_pitch_deg'][motion,1]).max()),
                stop_speed_peak=float(np.linalg.norm(trace['linear_world'][tail,:2],axis=-1).max()),
                trace_sha256=sha(args.output/'traces'/(sample['id']+'.npz')),
                **{'passed':next(r['passed'] for r in result['samples'] if r['sample_id']==sample['id'])}))
    write(args.output/'metrics.json',rows)
    assert sha(checkpoint)==digest
    write(args.output/'receipt.json',dict(passed=True,checkpoint_sha256=sha(checkpoint),
        model=int(args.model),optimizer_updates=0,all_traces_finite=all(
            r['complete'] and r['finite'] for r in result['samples']),
        complete_traces=len(rows),script_sha256=sha(__file__)))
    print(json.dumps(dict(model=args.model,total=len(rows),passed=sum(r['passed'] for r in rows))))


def stairs(args):
    from . import diagnose_reverse
    args.output.mkdir(parents=True,exist_ok=False)
    request=SimpleNamespace(output=args.output,checkpoint=MODELS[args.model],command_height=.35,
        label='STEP_'+args.model,seconds=20.,deadline=args.deadline,heights=[.2],speeds=[.3,1.,2.5],
        angles=[0.,-10.,10.],directions=['forward'],friction=.8,high_speed=True,frictions=[.4,.8,1.2])
    diagnose_reverse.run(request)
    audit=args.output/'audit';audit.mkdir()
    diagnose_reverse.audit(SimpleNamespace(recording=args.output,output=audit))


def run(args):
    from .inspect_training import source_checks
    checks=source_checks(RUN)
    audit=json.loads((ROOT/'artifacts/fudan_ppo_step/20261117-fixed200-turn-resume3200-4200-01/run/training-audit.json').read_text())
    assert audit['passed'] and audit['iterations']==4200 and sha(MODELS['4200'])==audit['checkpoint_sha256']
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'input-receipt.json',dict(checks=checks,training_audit=audit,
        checkpoints={key:sha(value) for key,value in MODELS.items()},optimizer_updates=0,
        script_sha256=sha(__file__),scope='Paired nominal turn/200mm GPU subset plus original flat47; not full acceptance'))
    (args.output/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    env=dict(os.environ,PYTHONPATH=str(ROOT/'mjlab_training/src')+':'+str(ROOT),MUJOCO_GL='egl')
    for model in ('4200','3200'):
        for mode in ('turns','stairs'):
            assert all(source_checks(RUN).values())
            with (args.output/f'{mode}-{model}.log').open('x') as stream:
                subprocess.run([str(ROOT/'mjlab_training/.venv/bin/python'),'-m',
                    'scripts.agent.fudan_ppo_step.evaluate_fixed200_turn_effect',mode,
                    '--model',model,'--output',str(args.output/f'{mode}-{model}'),
                    '--deadline',str(args.deadline)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                    check=True,timeout=max(1,args.deadline-time.time()))
            print(json.dumps(dict(completed=mode,model=model)),flush=True)
    assert all(sha(MODELS[key])==digest for key,digest in json.loads((args.output/'input-receipt.json').read_text())['checkpoints'].items())
    write(args.output/'execution.json',dict(passed=True,optimizer_updates=0,checkpoint_files_unchanged=True))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=('turns','stairs','run'))
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    if args.mode!='run' and args.model is None:p.error('--model is required')
    {'turns':turns,'stairs':stairs,'run':run}[args.mode](args)


if __name__=='__main__':main()
