#!/usr/bin/env python3
"""Zero-update control: preserve stable legacy39 control while ignoring wheel slots."""
import argparse,copy,json,shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor
from ct_mjlab.wheel_angles import WHEEL_ANGLES
from check_lqr_student_closed_loop import digest
import eval_bc
import assess_bc_closed_loop


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1)
    actor,saved=load_actor(args.checkpoint,args.device)
    old,_=load_actor(args.checkpoint,args.device)
    assert saved['format']=='ct_bc_actor39_pd6_v1' and actor.mlp[0].in_features==39
    before={k:v.clone() for k,v in actor.state_dict().items()}
    # 旧模型这两列的训练输入恒为零；将对应列置零，保持其余37通道的原函数。
    actor.mlp[0].weight[:,14:16]=0.
    data=ActionDataset([args.dataset],require_reference=True)
    assert data.description['wheel_angles']==WHEEL_ANGLES
    obs,_=data.sample('validation',2048,torch.Generator().manual_seed(714));obs=obs.to(args.device)
    legacy=obs.clone();legacy[:,14:16]=0.
    error=float((actor.mlp(obs)-old.mlp(legacy)).abs().max())
    torch.testing.assert_close(actor.mlp(obs),old.mlp(legacy),atol=2e-6,rtol=2e-6)
    changed=obs.clone();changed[:,14]=torch.linspace(-100,100,len(obs),device=args.device);changed[:,15]=-changed[:,14]
    torch.testing.assert_close(actor.mlp(obs),actor.mlp(changed),atol=0,rtol=0)
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=copy.deepcopy(saved);runtime['dataset']=data.description
    protocol=dict(diagnostic_only=True,optimizer_updates=0,checkpoint_written=False,
        original_checkpoint_sha256=digest(args.checkpoint),
        original_training_fingerprint=saved['dataset']['fingerprint'],
        runtime_dataset_fingerprint=data.description['fingerprint'],
        transform='In-memory first-layer columns14,15 set to zero; every other parameter unchanged',
        max_offline_action_difference=error,arbitrary_wheel_angles_have_no_action_effect=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    call=SimpleNamespace(dataset=args.dataset,corrections=[],checkpoint=args.checkpoint,pose_dataset=args.pose_dataset,
        device=args.device,split='validation',collect_corrections=False,full_precision=False,output=args.output/'evaluation')
    with patch.object(eval_bc,'load_actor',return_value=(actor,runtime)):eval_bc.run(call)
    result_path=args.output/'evaluation/result.json';report=json.loads(result_path.read_text())
    report.update(diagnostic_only=True,actor_transform=protocol['transform'],
        original_training_dataset_fingerprint=saved['dataset']['fingerprint'],
        dataset_fingerprint_scope='Runtime observation/pose manifest only; historical training data not replaced')
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    comparison=SimpleNamespace(run=args.output/'evaluation',previous=args.previous,checkpoint=args.checkpoint,
        dataset=args.dataset,expected_updates=saved['total_actor_updates'],output=args.output/'comparison.json')
    with patch.object(assess_bc_closed_loop,'load_actor',return_value=(actor,runtime)):assess_bc_closed_loop.assess(comparison)
    with np.load(args.output/'evaluation/trace.npz') as d:
        differences=[]
        for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
            current=torch.from_numpy(d['observation'][tick].copy()).to(args.device)
            previous=current.clone();previous[:,14:16]=0.
            torch.testing.assert_close(actor.mlp(current),old.mlp(previous),atol=2e-6,rtol=2e-6)
            differences.append(float((actor.mlp(current)-old.mlp(previous)).abs().max()))
        angles=d['observation'][...,14:16][d['alive']]
    for name,value in before.items():
        actual=actor.state_dict()[name]
        if name=='mlp.0.weight':
            keep=value.clone();keep[:,14:16]=0.;torch.testing.assert_close(actual,keep,atol=0,rtol=0)
        else:torch.testing.assert_close(actual,value,atol=0,rtol=0)
        torch.testing.assert_close(old.state_dict()[name],value,atol=0,rtol=0)
    assert digest(args.checkpoint)==protocol['original_checkpoint_sha256']
    root=Path(__file__).resolve().parents[2]
    sources={}
    files=list((root/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[Path(__file__),root/'scripts/agent/eval_bc.py',root/'scripts/agent/assess_bc_closed_loop.py']
    for file in files:
        name=file.relative_to(root);destination=args.output/'snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,destination);sources[str(name)]=digest(file)
    result=dict(completed=True,optimizer_updates=0,protocol=protocol,
        original_nonwheel_parameters_preserved=True,new_checkpoint_written=False,
        max_live_action_difference_from_legacy_zero_slot_actor=max(differences),
        wheel_angles_range_rad=[angles.min(0).tolist(),angles.max(0).tolist()],
        strict_rise=report['strict_rise'],final_balance=report['final_balance'],qualified=report['qualified'],
        num_envs=report['num_envs'],source_sha256=sources,
        interpretation='Only checks new observation plumbing with stable control retained; does not test learning wheel-angle dependence')
    (args.output/'check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pose-dataset','checkpoint','previous','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');check(p.parse_args())
