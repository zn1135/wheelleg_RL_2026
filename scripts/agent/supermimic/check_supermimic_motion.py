#!/usr/bin/env python3
"""V12 motion-only zero-update checks or read-only frozen-run audit."""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.supermimic import __main__ as entry
from ct_mjlab.supermimic.learner import Learner, ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE
from ct_mjlab.supermimic.motion import physical47_from_full64, expand_motion47_to_full64, delta46, integrate47
from ct_mjlab.supermimic.motion_adapter import MOTION_AUXILIARY_RECIPE
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.state import StateAdapter, STATE_DIM, PARAM_DIM, PHYS_DIM, SUBSTEPS, MEMORY, NOISE, SENSORS, MASSES, SOLREF

ROOT=Path(__file__).resolve().parents[3]
V12='chuanliantui-supermimic-motion-only-pd-reward-soft-termination-v12'
PARENT=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v7_worldfit4_actortransfer3600/checkpoint_000615.pt'
DELETED=[9,14]+list(range(49,64))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def fixture(n=20,device='cpu'):
    adapter=StateAdapter().to(device);s=torch.zeros(n,STATE_DIM,device=device)
    s[:,:25]=adapter.kin.qpos0.float();s[:,2]=.3;s[:,3]=1.
    p=torch.zeros(n,PARAM_DIM,device=device);p[:,:6]=s[:,adapter.motor_q]
    p[:,6:12]=10.;p[:,12:18]=.5;p[:,18:24]=1.;p[:,MASSES]=1.;p[:,SOLREF]=p.new_tensor([.02,1.])
    s[:,SENSORS]=adapter.raw_sensors(s[:,:64],p)[:,None].expand(-1,9,-1).flatten(1)
    c=torch.zeros(n,21,device=device);c[:,2]=.3;c[:,6]=1.;c[:,19]=.04;c[:,20]=.01
    c[:,7:19:2]=-torch.inf;c[:,8:19:2]=torch.inf
    return s,p,c


def synthetic_buffer():
    s,p,c=fixture();adapter=StateAdapter();b=ReplayBuffer(24,20,'cpu');h=torch.zeros(20,125);prev=torch.zeros(20,6)
    for tick in range(16):
        a=torch.full((20,6),.02+.001*tick);path=s[:,None,:64].expand(-1,5,-1).clone()
        path[:,:,0]+=torch.arange(1,6)*.0001;path[:,:,25]=.01+.001*tick
        ns,frame,reward,failure,features=adapter.advance(s,p,path,torch.zeros(20,3),a,prev,prev*.5,c,s[:,NOISE])
        b.add(history=h,privileged=torch.zeros(20,71),action=a,next_frame=frame,next_privileged=torch.zeros(20,71),velocity=s[:,25:28]*2,
            reward=reward,done=torch.zeros(20,dtype=torch.bool),failure=failure,features=features,context=c,previous=prev,previous_previous=prev*.5,
            state=s,parameters=p,next_state=ns,path=path.flatten(-2))
        h=torch.cat((h[:,25:],frame),-1);s=ns;prev=a
    return b


def cpu_checks(parent):
    from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
    command=SimpleNamespace(cfg=command_cfg('walk'),checkpoint_state=lambda:{},load_checkpoint_state=lambda _:None)
    l=Learner('cpu',actor_lr=3e-5);l.init_actor(parent,command,task='chuanliantui_walk')
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    assert equal(l.actor.state_dict(),saved['actor'])
    assert all(not getattr(l,n).state for n in ('opt_actor','opt_encoder','opt_world','opt_critic'))
    s,p,c=fixture(4);w=l.world;a=torch.full((4,6),.03)
    assert w.adapter.motor_q[[2,5]].tolist()==[9,14]
    assert w.in_norm.mean.numel()==141 and w.delta_norm.mean.numel()==46 and w.net[-1].out_features==230
    changed=s.clone();changed[:,DELETED]=torch.nan;changed[:,MEMORY.start+1:MEMORY.stop]=12345.
    assert equal(w.inputs(s,p,a,a*.5,a*.25),w.inputs(changed,p,a,a*.5,a*.25))
    first=w.step(s,p,a,torch.zeros(4,3),c,a*.5,a*.25,s[:,NOISE],return_path=True)
    other=w.step(changed,p,a,torch.zeros(4,3),c,a*.5,a*.25,s[:,NOISE],return_path=True)
    assert all(equal(x,y) for x,y in zip(first,other))
    height_context=c.clone();height_context[:,2]=.24
    shifted=w.step(s,p,a,torch.zeros(4,3),height_context,a*.5,a*.25,s[:,NOISE],return_path=True)
    f=first[4]
    upright=((torch.deg2rad(f.new_tensor(40.))-torch.acos(-f[:,8].clamp(-1.,1.)))/torch.deg2rad(f.new_tensor(25.))).clamp(0.,1.)
    expected=(torch.exp(-(f[:,9]-c[:,2]).square()/.0025)-torch.exp(-(f[:,9]-.24).square()/.0025))*upright*c[:,20]
    torch.testing.assert_close(first[2]-shifted[2],expected,atol=1e-8,rtol=1e-5)
    assert torch.count_nonzero(first[-1][...,DELETED])==0 and torch.count_nonzero(first[0][:,MEMORY.start+1:MEMORY.stop])==0
    motion=physical47_from_full64(s[:,:64]);assert motion.shape==(4,47)
    restored=expand_motion47_to_full64(motion);assert torch.count_nonzero(restored[:,DELETED])==0
    d=torch.randn(4,46)*.001;roundtrip=delta46(motion,integrate47(motion,d))
    torch.testing.assert_close(roundtrip,d,atol=1e-6,rtol=1e-5)
    path=first[-1].detach().clone();path[:,-1,2]=torch.tensor([.17,.18,.19,.2]);path.requires_grad_()
    survival=w.adapter.continuation(path);g=torch.autograd.grad(survival.sum(),path)[0]
    assert torch.isfinite(g).all() and g[:,-1,2].abs().min()>0 and ((survival>0)&(survival<1)).all()
    original=path.detach().clone();original[...,DELETED]=float('nan')
    torch.testing.assert_close(w.adapter.continuation(original),survival.detach(),atol=0,rtol=0)
    b=synthetic_buffer();sample=b.sample(8,8);mutated=copy.deepcopy(sample)
    mutated['state'][...,DELETED]=float('nan');mutated['path']=mutated['path'].reshape(8,8,5,64)
    mutated['path'][...,DELETED]=float('nan');mutated['path']=mutated['path'].flatten(-2)
    w.observe(sample);norm=copy.deepcopy(w.state_dict());base_loss,_=l.world_loss(sample)
    new_loss,_=l.world_loss(mutated);torch.testing.assert_close(base_loss,new_loss,atol=0,rtol=0)
    base_loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in w.parameters())
    for module in (l.actor,l.world,l.critic):module.zero_grad(set_to_none=True)
    w.observe(mutated);w.load_state_dict(norm)
    before={n:copy.deepcopy(getattr(l,n).state_dict()) for n in ('actor','critic')}
    weights=[v.detach().clone() for v in w.parameters()]
    with patch.object(l.opt_world,'step') as steps:
        metrics=l.update(b,batch=8,window=8,horizon=24,policy=False,encoder=False)
        assert steps.call_count==4
    assert all(torch.equal(v,p) for v,p in zip(weights,w.parameters())) and all(equal(getattr(l,n).state_dict(),v) for n,v in before.items())
    assert all(not getattr(l,n).state for n in ('opt_actor','opt_encoder','opt_world','opt_critic'))
    # The replay mask includes the first true terminal and excludes following episodes.
    boundary=ReplayBuffer(12,20,'cpu')
    for tick in range(12):
        row={k:torch.zeros_like(v[0]) for k,v in boundary.data.items()};row['history'].fill_(tick);row['done'].fill_(tick==5);boundary.add(**row)
    for validation in (False,True):
        batch=boundary.sample(64,8,validation=validation);times=batch['history'][...,0]
        expected=torch.cumprod(torch.cat((torch.ones(64,1),(times[:,:-1]!=5).float()),1),1)
        assert torch.equal(batch['mask'],expected)
    return dict(network_input=141,learned_motion=47,learned_delta=46,network_output=230,deleted_full64_indices=DELETED,
        deleted_nan_perturbation_input_output_and_loss_invariant=True,deleted_outputs_zero=True,delta_roundtrip=True,
        explicit_height_tracking_times_upright_without_contact_factor=True,
        soft_survival_gradient_finite_nonzero=True,soft_survival_does_not_read_deleted_fields=True,real_episode_mask_oracle=True,
        full_actor_bitwise_615=True,four_optimizers_empty=True,mocked_world_step_calls=4,actual_optimizer_steps=0,metrics=metrics)


def integration(args):
    args.task,args.seed,args.num_envs='chuanliantui_walk',42,20
    env,_=entry.make_env(args)
    try:
        raw=env.unwrapped;l=Learner(args.device,actor_lr=3e-5);command=raw.command_manager.get_term('standup')
        l.init_actor(args.parent,command,task=args.task);before=copy.deepcopy(l.actor.state_dict())
        b=ReplayBuffer(max(64,args.steps),20,args.device);obs=env.get_observations();original=StateAdapter().to(args.device)
        maxima=dict(actor25=0.,history125=0.,raw_reward=0.,original_state=0.)
        def close(name,a,b):
            if a.numel():maxima[name]=max(maxima[name],float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
        resets=0
        for tick in range(args.steps):
            state=entry.capture_state(raw);history=obs['history'].clone();paths=[];method=raw.metrics_manager.compute_substep
            def observed():method();paths.append(entry.physical_snapshot(raw))
            with patch.object(raw.metrics_manager,'compute_substep',side_effect=observed):obs,reward,done=entry.collect(env,l,b,obs)
            row=(b.ptr-1)%b.capacity;data={k:v[row] for k,v in b.data.items()};assert len(paths)==5
            assert torch.equal(data['path'],torch.stack(paths,1).flatten(-2)) and torch.equal(data['state'],state)
            assert torch.equal(data['failure'],raw.reset_terminated.bool()) and torch.equal(data['done'],done.bool())
            close('raw_reward',data['reward'],reward)
            path=data['path'].reshape(20,5,64);valid=~done.bool();resets+=int(done.sum())
            real=original.advance(state,data['parameters'],path,obs['actor'][:,6:9],data['action'],data['previous'],data['previous_previous'],data['context'],data['next_state'][:,NOISE])
            close('raw_reward',real[2],reward);close('original_state',real[0],data['next_state'])
            motion=l.world.adapter.advance(state,data['parameters'],path,obs['actor'][:,6:9],data['action'],data['previous'],data['previous_previous'],data['context'],data['next_state'][:,NOISE])
            close('actor25',motion[1][valid],obs['actor'][valid]);close('history125',torch.cat((history[:,25:],motion[1]),-1)[valid],obs['history'][valid])
        sample=b.sample(16,8);l.world.observe(sample);loss,_=l.world_loss(sample);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in l.world.parameters())
        assert equal(l.actor.state_dict(),before)
        assert all(not getattr(l,n).state for n in ('opt_actor','opt_encoder','opt_world','opt_critic'))
        return dict(real_transitions=20*args.steps,physics_substeps=100*args.steps,real_resets=resets,maximum_errors=maxima,
            true_done_failure_reward_labels_unchanged=True,new_motion_actor25_history125_matches_nonreset=True,
            motion_world_backward_finite=True,full_actor_bitwise_frozen=True,optimizer_steps=0,
            scope='Finite real collection and tensor check; PD model reward is deliberately not asserted equal to real actual-torque reward; no behavior acceptance.')
    finally:env.close()


def audit(args):
    run=args.run.resolve();initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    parent=torch.load(args.parent,map_location='cpu',weights_only=False);config=json.loads((run/'config.json').read_text())
    assert sha(args.parent)=='50ee90a69e6f58344898277a77ff850d2504dbda31dd3ff4b8beb6473ef5105f'
    manifest=json.loads((run/'source_sha256.json').read_text());source=ROOT/'mjlab_training/src/ct_mjlab'
    assert all(sha(source/k)==v==sha(run/'snapshot/ct_mjlab'/k) for k,v in manifest.items())
    assert initial['algorithm_contract']==V12 and config['world_loss_recipe']==WORLD_LOSS_RECIPE and config['policy_warmup_updates']==1000000000
    assert equal(initial['actor'],parent['actor']) and all(not initial[n]['state'] for n in ('opt_actor','opt_encoder','opt_world','opt_critic'))
    paths=[run/'initial.pt']
    if args.checkpoint:paths.append(args.checkpoint.resolve())
    elif (run/'final.pt').exists():paths.append(run/'final.pt')
    else:
        numbered=sorted(run.glob('checkpoint_*.pt'))
        if numbered:paths.append(numbered[-1])
    points=[]
    for path in paths:
        assert path.parent==run;d=torch.load(path,map_location='cpu',weights_only=False)
        assert d['algorithm_contract']==V12 and d['config']==config and equal(d['actor'],initial['actor']) and equal(d['critic'],initial['critic'])
        for name in ('opt_actor','opt_encoder','opt_critic'):assert not d[name]['state'] and equal(d[name],initial[name])
        steps={int(v['step']) for v in d['opt_world']['state'].values()};expected=d['world_updates']
        assert steps==({expected} if expected else set()) and expected%4==0
        count=float(d['world']['in_norm.count']);assert float(d['world']['delta_norm.count'])==5*count
        points.append(dict(path=str(path),sha256=sha(path),world_updates=expected,world_norm_count=count,delta_norm_count=5*count,encoder_actor_critic_steps=[0,0,0]))
    completed=False
    if args.exit_record:
        receipt=json.loads(args.exit_record.read_text());assert receipt['returncode']==0 and (run/'final.pt').is_file()
        completed=True
    return dict(source_count=len(manifest),source_manifest_sha256=sha(run/'source_sha256.json'),checkpoints=points,
        full_actor_and_critic_bitwise_frozen=True,three_optimizers_empty=True,completion_verified=completed,
        scope='Read-only V12 frozen-function/source/optimizer evidence; not prediction or behavior acceptance.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=32)
    p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--run',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--exit-record',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.manual_seed(55);assert ALGORITHM_CONTRACT==V12
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Optimizer updates forbidden')):
        result=audit(args) if args.run else cpu_checks(args.parent) if args.device=='cpu' else integration(args)
    source=ROOT/'mjlab_training/src/ct_mjlab';hashes={str(f.relative_to(source)):sha(f) for f in sorted(source.rglob('*.py'))}
    report=dict(all_checks_passed=True,algorithm_contract=V12,world_loss_recipe=WORLD_LOSS_RECIPE,model_reward_recipe=MOTION_AUXILIARY_RECIPE,
        checker_sha256=sha(__file__),source_sha256=hashes,optimizer_steps=0,checks=result)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),all_checks_passed=True,optimizer_steps=0)))


if __name__=='__main__':main()
