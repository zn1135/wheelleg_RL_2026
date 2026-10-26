#!/usr/bin/env python3
"""Check migration, train-only retained skills and the actual preserved sampler."""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor,LEGACY_SAMPLING,PRESERVE_SAMPLING,PRESERVE_WEIGHTS
from ct_mjlab.bc_preservation import PreservationDataset,migrate_wheel_actor
from ct_mjlab.scripts.train_bc import train
from prepare_bc_wheel_dataset import audit_sources,digest,write


def check(args):
    torch.set_num_threads(1)
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    base=ActionDataset([args.dataset],require_reference=True)
    data=ActionDataset([args.dataset],[args.braking],action_sampling=PRESERVE_SAMPLING,require_reference=True)
    report=json.loads((args.braking/'result.json').read_text());audit=audit_sources(args.braking,report)
    assert not report['teacher_queries_only'] and report['legacy_zero_wheel'] and not report['heldout_used']
    assert report['checkpoint_sha256']==digest(args.init) and report['trace_sha256']==digest(args.braking/'trace.npz')
    assert {r['group_id'] for r in report['rows']}==set(base.description['group_split']['train'])
    actor,saved=load_actor(args.init,'cuda:0');before={k:v.clone() for k,v in actor.state_dict().items()}
    actor=migrate_wheel_actor(actor,saved,data.description)
    reference,_=load_actor(args.init,'cuda:0')
    with torch.no_grad():
        x,_=base.sample('validation',2048,torch.Generator().manual_seed(714));x=x.cuda();old_x=x.clone();old_x[:,14:16]=0.
        difference=float(abs(actor.mlp(x)-reference.mlp(old_x)).max());assert difference<=2e-6
    for name,value in before.items():
        expected=value.clone()
        if name=='mlp.0.weight':expected[:,14:16]=0
        torch.testing.assert_close(actor.state_dict()[name],expected,atol=0,rtol=0)
    for source,augmented in zip(base.frames['validation'],data.frames['validation']):torch.testing.assert_close(source,augmented,atol=0,rtol=0)
    for source,augmented in zip(base.sample('validation',2048,torch.Generator().manual_seed(714)),
            data.sample('validation',2048,torch.Generator().manual_seed(714))):torch.testing.assert_close(source,augmented,atol=0,rtol=0)
    preservation=PreservationDataset(args.preservation,data.description,digest(args.init))
    generator=torch.Generator().manual_seed(901)
    rise,feedback=preservation.loss(actor,generator,'cuda:0')
    assert rise.item()<1e-7 and feedback.item()<1e-7
    weak,_=load_actor(args.weak,'cuda:0')
    weak_rise,weak_feedback=preservation.loss(weak,torch.Generator().manual_seed(901),'cuda:0')
    assert weak_feedback.item()>feedback.item()+.01
    shifted=copy.deepcopy(actor)
    with torch.no_grad():
        [m for m in shifted.mlp.modules() if isinstance(m,torch.nn.Linear)][-1].bias.add_(.1)
    shifted_rise,shifted_feedback=preservation.loss(shifted,torch.Generator().manual_seed(901),'cuda:0')
    assert abs(shifted_feedback.item()-feedback.item())<1e-7 and shifted_rise.item()>rise.item()+1e-6
    obs,target=data.sample('train',1024,torch.Generator().manual_seed(42))
    loss=((actor.mlp(obs.cuda())-target.cuda())/data.frames['train'][1].std(0,correction=0).clamp_min(.1).cuda()).square().mean()+rise+feedback
    loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    assert actor.mlp[0].weight.grad[:,14:16].abs().sum()>0
    # 旧接口仍不能在没有显式迁移参数时静默进入新轮角训练。
    try:
        train(SimpleNamespace(iterations=1,batch_size=2,save_every=1,seed=42,device='cpu',dataset=[args.dataset],
            corrections=[],action_sampling=LEGACY_SAMPLING,init=args.init,expand_observations=False,
            output=args.output.parent/'forbidden-implicit-migration',lr=.0001))
    except ValueError as error:
        assert 'observation contracts differ' in str(error);rejection=str(error)
    else:raise AssertionError('Implicit legacy migration must remain rejected')
    assert not (args.output.parent/'forbidden-implicit-migration').exists()
    # 从采样返回值反查实际类别，排除专家采样误混入原教师起立动作。
    lookup={}
    for name in PRESERVE_WEIGHTS:
        for row in torch.cat(tuple(x[data.classes['train'][name]] for x in data.frames['train']),dim=-1).numpy():
            lookup.setdefault(hashlib.sha256(row.tobytes()).digest(),set()).add(name)
    x,y=data.sample('train',8192,torch.Generator().manual_seed(803));counts={name:0 for name in PRESERVE_WEIGHTS};ambiguous=0
    for row in torch.cat((x,y),-1).numpy():
        names=lookup[hashlib.sha256(row.tobytes()).digest()]
        if len(names)==1:counts[next(iter(names))]+=1
        else:ambiguous+=1
    assert ambiguous==0 and all(abs(counts[k]/8192-PRESERVE_WEIGHTS[k])<.025 for k in counts)
    assert digest(args.init)==report['checkpoint_sha256']
    result=dict(passed=True,optimizer_updates=0,checkpoint_sha256=digest(args.init),
        migration_action_difference=difference,other_parameters_preserved=True,
        heldout_tensors_and_fixed_sample_unchanged=True,heldout_used_for_preservation=False,
        initial_preserve_rise_loss=rise.item(),initial_preserve_feedback_loss=feedback.item(),
        weak_model_feedback_loss=weak_feedback.item(),constant_action_offset_feedback_loss=shifted_feedback.item(),
        constant_action_offset_rise_loss=shifted_rise.item(),finite_combined_loss_and_gradients=True,
        new_wheel_input_columns_have_gradient=True,implicit_migration_rejected=rejection,
        empirical_expert_sampling={name:counts[name]/8192 for name in counts},
        teacher_rise_excluded_from_expert_sampler=True,dataset=data.description,preservation=preservation.description,
        braking_source_audit=audit,checker_sha256=digest(Path(__file__)),
        scope='Migration, train/heldout isolation, loss behavior, source integrity, actual sampling and backward only; no optimizer step or behavior claim')
    write(args.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('dataset','preservation')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','braking','preservation','init','weak','output'):p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
