#!/usr/bin/env python3
"""Zero-update diagnostic: retain a legacy actor and append a zero input column."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.bc import ActionDataset, BRAKING_SAMPLING, LEGACY_SAMPLING, expand_actor, load_actor
import eval_bc
import assess_bc_closed_loop


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


@torch.inference_mode()
def diagnose(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    old,saved=load_actor(args.checkpoint,args.device)
    assert old.mlp[0].in_features==39
    sampling=BRAKING_SAMPLING if args.corrections else LEGACY_SAMPLING
    data=ActionDataset([args.dataset],args.corrections,action_sampling=sampling,require_reference=True)
    assert data.observation_dim in (40,41)
    actor=expand_actor(old,data.observation_dim,args.device)
    before=copy.deepcopy(saved['actor_state_dict']);state=actor.state_dict();padded=[]
    for name,value in before.items():
        if value.shape==state[name].shape:
            state[name].copy_(value)
        else:
            assert value.shape==(256,39) and state[name].shape==(256,data.observation_dim)
            state[name].zero_();state[name][:,:39].copy_(value);padded.append(name)
    assert len(padded)==1
    actor.load_state_dict(state,strict=True)
    generator=torch.Generator().manual_seed(714)
    inputs,target=data.sample('validation',2048,generator)
    inputs=inputs.to(args.device);target=target.to(args.device)
    expected=old.mlp(inputs[:,:39].contiguous());actual=actor.mlp(inputs)
    difference=float((expected-actual).abs().max())
    torch.testing.assert_close(expected,actual,atol=2e-5,rtol=2e-5)
    changed=inputs.clone();changed[:,39:]=torch.linspace(-100,100,len(changed),device=args.device)[:,None]
    torch.testing.assert_close(actual,actor.mlp(changed),atol=0,rtol=0)
    original_sha=digest(args.checkpoint)
    # 仅诊断时用新清单选择同一批姿态，不改旧模型训练来源或保存新PT。
    runtime=copy.deepcopy(saved);runtime['dataset']=data.description
    source=dict(checkpoint_sha256=original_sha,original_training_dataset_fingerprint=saved['dataset']['fingerprint'],
        runtime_pose_dataset_fingerprint=data.description['fingerprint'],optimizer_updates=0,
        transform=f'Copy every learned parameter; append {data.observation_dim-39} zero first-layer columns; extra features ignored',
        padded_parameter=padded[0],max_offline_action_difference=difference,
        arbitrary_wheel_travel_minus100_to100_has_no_action_effect=True,
        checkpoint_written=False,diagnostic_only=True,scope='No claim that training on wheel travel improves or degrades control')
    write(args.output/'protocol.json',source)
    env_args=SimpleNamespace(dataset=args.dataset,corrections=args.corrections,checkpoint=args.checkpoint,
        device=args.device,split='validation',collect_corrections=False,full_precision=False,output=args.output/'evaluation')
    with patch.object(eval_bc,'load_actor',return_value=(actor,runtime)):
        eval_bc.run(env_args)
    result_path=args.output/'evaluation/result.json';result=json.loads(result_path.read_text())
    result.update(diagnostic_only=True,actor_transform=source['transform'],
        original_training_dataset_fingerprint=saved['dataset']['fingerprint'],
        dataset_fingerprint_scope='Runtime pose/observation manifest, not historical actor training data')
    write(result_path,result)
    assess_args=SimpleNamespace(run=args.output/'evaluation',previous=args.previous,checkpoint=args.checkpoint,
        dataset=args.dataset,expected_updates=saved['total_actor_updates'],output=args.output/'comparison.json')
    with patch.object(assess_bc_closed_loop,'load_actor',return_value=(actor,runtime)):
        assess_bc_closed_loop.assess(assess_args)
    for name,value in before.items():
        current=actor.state_dict()[name]
        if name==padded[0]:
            torch.testing.assert_close(current[:,:39],value,atol=0,rtol=0)
            assert not current[:,39:].any()
        else:
            torch.testing.assert_close(current,value,atol=0,rtol=0)
    assert digest(args.checkpoint)==original_sha
    with np.load(args.output/'evaluation/trace.npz') as d:
        wheel=d['observation'][:,:,39][d['alive']]
        wheel_range=[float(wheel.min()),float(wheel.max())]
    root=Path(__file__).resolve().parents[2]
    sources={}
    paths=list((root/'mjlab_training/src/ct_mjlab').rglob('*.py'))+[Path(__file__),
        root/'scripts/agent/eval_bc.py',root/'scripts/agent/assess_bc_closed_loop.py']
    for path in paths:
        name=path.relative_to(root);destination=args.output/'snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination);sources[str(name)]=digest(path)
    outcome=dict(completed=True,optimizer_updates=0,source=source,wheel_observation_range_m=wheel_range,
        original_learned_parameters_preserved=True,new_checkpoint_written=False,
        strict_rise=result['strict_rise'],final_balance=result['final_balance'],qualified=result['qualified'],
        num_envs=result['num_envs'],source_sha256=sources,
        trace_sha256=digest(args.output/'evaluation/trace.npz'),
        interpretation='Tests observation plumbing with existing skill retained, not the effect of learning a displacement feature')
    write(args.output/'check.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--corrections',type=Path,action='append',default=[])
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');diagnose(p.parse_args())
