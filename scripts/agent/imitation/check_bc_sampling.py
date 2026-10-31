#!/usr/bin/env python3
"""Check braking sampling, legacy compatibility and gradients without optimizer updates."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,BRAKING_SAMPLING,BRAKING_WEIGHTS,load_actor


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(args):
    torch.set_num_threads(1)
    original=ActionDataset([args.dataset],args.corrections[:-1])
    data=ActionDataset([args.dataset],args.corrections,action_sampling=BRAKING_SAMPLING)
    actor,saved=load_actor(args.init,args.device)
    assert original.description['fingerprint']==saved['dataset']['fingerprint']
    assert len(data.classes['validation'])==2 and len(data.classes['train'])==5
    for a,b in zip(original.frames['validation'],data.frames['validation']):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(original.sample('validation',2048,torch.Generator().manual_seed(714)),
                   data.sample('validation',2048,torch.Generator().manual_seed(714))):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    indices=torch.cat(tuple(data.classes['train'].values()))
    assert len(indices)==len(indices.unique())==len(data.frames['train'][0])
    assert indices.min()==0 and indices.max()==len(indices)-1
    groups={s:{e['group_id'] for e in rows} for s,rows in data.motion.splits.items()}
    assert not groups['train']&groups['validation'] and len(groups['validation'])==10
    # 从实际返回的观测/目标反查类别，检查采样比例，不额外复刻采样器的随机步骤。
    rows=torch.cat(data.frames['train'],dim=-1).numpy()
    lookup={}
    for name,pool in data.classes['train'].items():
        for index in pool.numpy():
            key=hashlib.sha256(rows[index].tobytes()).digest()
            lookup.setdefault(key,set()).add(name)
    obs,target=data.sample('train',32768,torch.Generator().manual_seed(801))
    sampled=torch.cat((obs,target),dim=-1).numpy();counts=Counter();ambiguous=0
    for row in sampled:
        names=lookup[hashlib.sha256(row.tobytes()).digest()]
        if len(names)==1:counts[next(iter(names))]+=1
        else:ambiguous+=1
    assert ambiguous==0
    proportions={name:counts[name]/len(sampled) for name in BRAKING_WEIGHTS}
    assert all(abs(proportions[k]-v)<.015 for k,v in BRAKING_WEIGHTS.items())
    state={k:v.clone() for k,v in actor.state_dict().items()}
    x,y=(v[:256].to(args.device) for v in (obs,target));loss=(actor.mlp(x)-y).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in actor.mlp.parameters())
    for k,v in state.items():torch.testing.assert_close(actor.state_dict()[k],v,atol=0,rtol=0)
    result=dict(passed=True,optimizer_updates=0,sampling=BRAKING_SAMPLING,weights=BRAKING_WEIGHTS,
        class_frame_counts=data.description['class_frame_counts'],empirical_proportions=proportions,
        sampled_frames=32768,training_classes_disjoint_and_complete=True,heldout_tensors_and_sampling_unchanged=True,
        legacy_fingerprint_matches_original_checkpoint=True,group_disjoint=True,finite_gradient=True,
        initial_actor_updates=saved.get('total_actor_updates',saved['iteration']),initial_actor_sha256=digest(args.init),
        fingerprint=data.description['fingerprint'],description=data.description,checker_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='description'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',required=True);p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');check(p.parse_args())
