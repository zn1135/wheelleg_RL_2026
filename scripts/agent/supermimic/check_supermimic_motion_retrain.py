#!/usr/bin/env python3
"""V12 thirteen-reward cold retraining checks and frozen-run evidence audit."""
import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.supermimic import __main__ as entry
from ct_mjlab.supermimic import learner as learner_module
from ct_mjlab.supermimic.analytic_reward import WEIGHTS
from ct_mjlab.supermimic.learner import Learner, ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE
from ct_mjlab.supermimic.motion import physical47_from_full64, expand_motion47_to_full64, delta46, integrate47
from ct_mjlab.supermimic.motion_adapter import MOTION_AUXILIARY_RECIPE
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.state import StateAdapter, STATE_DIM, PARAM_DIM, PHYS_DIM, SUBSTEPS, MEMORY, NOISE, SENSORS, MASSES, SOLREF

ROOT=Path(__file__).resolve().parents[3]
V12='chuanliantui-supermimic-motion-only-pd-reward-soft-termination-v12'
PARENT=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v7_worldfit4_actortransfer3600/checkpoint_000615.pt'
DELETED=[9,14]+list(range(49,64))
PARENT_SHA='50ee90a69e6f58344898277a77ff850d2504dbda31dd3ff4b8beb6473ef5105f'
REMOVED={'lin_vel_z','base_link_contact','wheels_airborne','action_rate','action_smooth'}
NEW_REWARD='split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-no-vz-contact-rate-smooth-v7'
OLD_REWARD='split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-v5'


def norm_count_rounding_proof(input_tensor, delta_tensor, steps, max_samples):
    # Sequential float32 additions: each rounding error is <= half the final ULP.
    assert input_tensor.dtype==delta_tensor.dtype==torch.float32
    count,delta=float(input_tensor),float(delta_tensor)
    assert type(steps) is int and steps>=0 and max_samples>0
    assert 0.<=count<2**24 and count.is_integer() and delta>=0. and delta.is_integer()
    assert count<=steps*max_samples
    ulp=float(torch.nextafter(delta_tensor,torch.full_like(delta_tensor,float('inf')))-delta_tensor)
    assert ulp>0. and torch.isfinite(delta_tensor)
    exact=5.*count
    bound=0. if exact<=2**24 else steps*.5*ulp
    error=delta-exact
    assert abs(error)<=bound, (count,delta,error,bound)
    return dict(actual_input_count=count,actual_delta_count=delta,exact_five_times_input=exact,
        difference=error,absolute_bound=bound,delta_float32_ulp=ulp,updates=steps,
        max_input_samples_per_update=max_samples,input_integer_additions_exact=True,
        below_2pow24_requires_exact=True,
        reason='Nonnegative float32 delta-count additions; each error <=0.5 ULP(final count), at most one addition per actual world Adam step.')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def transfer_proof(proof, parent, target_infos):
    source=parent['infos'];expected_weights={k:v for k,v in source['reward_weights'].items() if k not in REMOVED}
    assert len(source['reward_weights'])==18 and len(expected_weights)==len(WEIGHTS)==13
    assert source['reward_recipe']==OLD_REWARD and target_infos['reward_recipe']==NEW_REWARD
    assert expected_weights==target_infos['reward_weights']==WEIGHTS
    assert {k:v for k,v in source.items() if k not in ('reward_recipe','reward_weights')}=={
        k:v for k,v in target_infos.items() if k not in ('reward_recipe','reward_weights')}
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    expected=dict(kind='SM615_reward_only_v5_to_v7_actor_transfer',source_recipe=OLD_REWARD,target_recipe=NEW_REWARD,
        removed_reward_terms=sorted(REMOVED),source_reward_weights=source['reward_weights'],target_reward_weights=expected_weights,
        source_infos_sha256=digest(source),target_infos_sha256=digest(target_infos),all_nonreward_infos_equal=True,
        world_critic_optimizers_not_inherited=True)
    assert proof==expected
    return expected


def compatibility_checks(learner, parent_path, command, parent, provenance):
    assert sha(parent_path)==PARENT_SHA
    target=learner_module.contracts(command.cfg)
    proof=transfer_proof(provenance['reward_only_actor_transfer'],parent,target)
    names=('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic')
    def snapshot():
        return ({n:copy.deepcopy(getattr(learner,n).state_dict()) for n in names},torch.get_rng_state().clone())
    failures=[]
    for name,mutated in (('nonreward_infos',dict(target,control_contract=target['control_contract']+'-invalid')),
                         ('remaining_reward_weight',dict(target,reward_weights=dict(target['reward_weights'],tracking_lin_vel=target['reward_weights']['tracking_lin_vel']+.001)))):
        before,rng=snapshot()
        with patch.object(learner_module,'contracts',return_value=mutated):
            try:learner.init_actor(parent_path,command,task='chuanliantui_walk')
            except ValueError:pass
            else:raise AssertionError('Invalid reward transfer accepted: '+name)
        assert all(equal(getattr(learner,n).state_dict(),before[n]) for n in names) and torch.equal(torch.get_rng_state(),rng)
        failures.append(name)
    before,rng=snapshot()
    with patch.object(command,'load_checkpoint_state') as curriculum:
        try:learner.load(parent_path,command,inference=False)
        except ValueError:pass
        else:raise AssertionError('Old full checkpoint resume accepted')
        assert curriculum.call_count==0
    assert all(equal(getattr(learner,n).state_dict(),before[n]) for n in names) and torch.equal(torch.get_rng_state(),rng)
    return dict(real_sm615_actor_transfer=True,provenance=proof,rejected_without_any_state_or_rng_mutation=failures+['old_full_resume'],
        curriculum_loads=0,actual_optimizer_steps=0)


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
    l=Learner('cpu',actor_lr=3e-5);provenance=l.init_actor(parent,command,task='chuanliantui_walk')
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    compatibility=compatibility_checks(l,parent,command,saved,provenance)
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
        explicit_height_tracking_times_upright_without_contact_factor=True,reward_only_actor_compatibility=compatibility,
        soft_survival_gradient_finite_nonzero=True,soft_survival_does_not_read_deleted_fields=True,synthetic_terminal_mask_oracle=True,
        synthetic_terminal_tick=5,synthetic_actual_simulator_resets=0,
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
            original_actual_reward_term_count=len(WEIGHTS),original_actual_reward_recipe=NEW_REWARD,
            actual_reset_boundary_observed=resets>0,synthetic_done_boundary_is_separate_cpu_check=True,
            true_done_failure_reward_labels_unchanged=True,new_motion_actor25_history125_matches_nonreset=True,
            motion_world_backward_finite=True,full_actor_bitwise_frozen=True,optimizer_steps=0,
            scope='Finite real collection and tensor check; PD model reward is deliberately not asserted equal to real actual-torque reward; no behavior acceptance.')
    finally:env.close()


def audit(args):
    run=args.run.resolve();initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    parent=torch.load(args.parent,map_location='cpu',weights_only=False);config=json.loads((run/'config.json').read_text())
    assert sha(args.parent)==PARENT_SHA
    budget_path=args.budget_record or args.output.parent/'budget.json'
    budget=json.loads(budget_path.read_text());start=datetime.fromisoformat(budget['started_at_utc']);deadline=datetime.fromisoformat(budget['deadline_utc'])
    assert (deadline-start).total_seconds()==budget['total_wall_clock_budget_s']==3600
    assert budget['training_loop_target_s']==config['seconds']==2400 and budget['minimum_post_training_reserve_s']>=600
    assert budget['actor_source_sha256']==PARENT_SHA and (ROOT/budget['actor_source']).resolve()==args.parent.resolve()
    assert all(config.get(k)==v for k,v in dict(num_envs=128,seed=42,window=8,horizon=24,batch=128,buffer_steps=512,
        policy_warmup_updates=1000000000).items())
    assert config.get('resume') is None and Path(config['init_actor']).resolve()==args.parent.resolve()
    assert initial['config']==config and initial['initialization']==config['initialization']
    assert config['initialization']['parent_sha256']==PARENT_SHA and config['initialization']['inherited']==['actor']
    from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
    target_infos=learner_module.contracts(command_cfg('walk'))
    provenance=transfer_proof(config['initialization']['reward_only_actor_transfer'],parent,target_infos)
    assert initial['infos']==target_infos and initial['asset_sha256']==parent['asset_sha256']==learner_module.asset_digest()
    manifest=json.loads((run/'source_sha256.json').read_text());source=ROOT/'mjlab_training/src/ct_mjlab'
    actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    archived={str(p.relative_to(run/'snapshot/ct_mjlab')):sha(p) for p in (run/'snapshot/ct_mjlab').rglob('*.py')}
    assert len(manifest)==101 and actual==archived==manifest
    assert all(sha(source/k)==v==sha(run/'snapshot/ct_mjlab'/k) for k,v in manifest.items())
    assert initial['algorithm_contract']==V12 and config['world_loss_recipe']==WORLD_LOSS_RECIPE and config['policy_warmup_updates']==1000000000
    assert equal(initial['actor'],parent['actor']) and all(not initial[n]['state'] for n in ('opt_actor','opt_encoder','opt_world','opt_critic'))
    assert initial['world_updates']==initial['iteration']==initial['transitions']==0
    assert all(float(initial[n][key])==0. for n,key in (('world','in_norm.count'),('world','delta_norm.count'),('critic','norm.count')))
    assert initial['world']['in_norm.mean'].numel()==141 and initial['world']['delta_norm.mean'].numel()==46
    assert initial['world']['net.6.weight'].count_nonzero()==0 and initial['world']['net.6.bias'].count_nonzero()==0
    physics_path=run/'initial_physics.pt';assert config['initial_physics_sha256']==sha(physics_path)
    physics=torch.load(physics_path,map_location='cpu',weights_only=False)
    assert physics['schema']=='supermimic-initial-physics-v1' and physics['algorithm_contract']==V12
    assert (physics['task'],physics['seed'],physics['num_envs'],physics['policy_dt_s'],physics['physics_dt_s'])==('chuanliantui_walk',42,128,.01,.002)
    state,params=physics['state'],physics['parameters']
    assert state.shape==(128,STATE_DIM) and params.shape==(128,PARAM_DIM) and state.dtype==params.dtype==torch.float32
    assert torch.isfinite(state).all() and torch.isfinite(params).all()
    ids=torch.arange(128);nominal_ids=ids[::4];other=ids[ids%4!=0]
    assert equal(physics['train_env_ids'],ids[ids%10!=0]) and equal(physics['validation_env_ids'],ids[ids%10==0])
    geoms,solref,default=(physics[k] for k in ('robot_collision_geom_ids','robot_collision_solref','default_robot_collision_solref'))
    assert geoms.dtype==torch.long and geoms.ndim==1 and len(geoms)>0
    assert solref.shape==(128,len(geoms),2) and default.shape==(len(geoms),2)
    assert torch.equal(default,default.new_tensor([.02,1.]).expand_as(default))
    assert torch.equal(solref,params[:,None,SOLREF].expand_as(solref))
    observed=(solref==default[None]).all(dim=(-1,-2)).nonzero().flatten()
    assert equal(observed,nominal_ids) and equal(physics['nominal_contact_env_ids'],nominal_ids)
    from ct_mjlab.envs.chuanliantui.standup_randomization import restitution_to_solref
    restitution=physics['original_restitution'];assert restitution.shape==(128,) and ((restitution>=.6)&(restitution<=1.)).all()
    expected=restitution_to_solref(restitution)
    torch.testing.assert_close(params[other,SOLREF],expected[other],atol=1e-5,rtol=1e-6)
    assert physics['contact_training_recipe']==config['contact_training_recipe'] and physics['contact_encoding_assumptions']==config['contact_encoding_assumptions']
    assert physics['state_and_parameters_unchanged_during_capture'] and physics['common_step_counter_before']==physics['common_step_counter_after']
    assert physics['rng_unchanged_during_capture'] and equal(physics['torch_rng_before'],physics['torch_rng_after']) and equal(physics['cuda_rng_before'],physics['cuda_rng_after'])
    assert equal(physics['torch_rng_after'],initial['torch_rng']) and equal(physics['cuda_rng_after'],initial['cuda_rng'])
    lines=(run/'metrics.jsonl').read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):lines=lines[:-1]
    metrics=[json.loads(line) for line in lines if line.strip()];by_iteration={r['iteration']:r for r in metrics}
    assert not any(set(r)&{'actor','critic','actor_grad','imagination_eligible_fraction'} for r in metrics)
    protected={str(p):sha(p) for p in (args.parent,run/'initial.pt',run/'config.json',physics_path,budget_path)}
    paths=[run/'initial.pt']
    if args.checkpoint:paths.append(args.checkpoint.resolve())
    elif (run/'final.pt').exists():paths.append(run/'final.pt')
    else:
        numbered=sorted(run.glob('checkpoint_*.pt'))
        if numbered:paths.append(numbered[-1])
    points=[]
    for path in paths:
        assert path.parent==run;d=torch.load(path,map_location='cpu',weights_only=False)
        assert d['algorithm_contract']==V12 and d['config']==config and d['infos']==target_infos and d['initialization']==config['initialization']
        assert d['asset_sha256']==initial['asset_sha256'] and not d.get('interrupted',False)
        assert equal(d['actor'],initial['actor']) and equal(d['critic'],initial['critic'])
        for name in ('opt_actor','opt_encoder','opt_critic'):assert not d[name]['state'] and equal(d[name],initial[name])
        steps={int(v['step']) for v in d['opt_world']['state'].values()};expected=d['world_updates']
        assert steps==({expected} if expected else set()) and expected%4==0
        count=float(d['world']['in_norm.count']);actual_delta=float(d['world']['delta_norm.count'])
        rounding=norm_count_rounding_proof(d['world']['in_norm.count'],d['world']['delta_norm.count'],expected,config['batch']*config['window'])
        assert (count>0)==(expected>0)
        changed=[k for k,v in d['world'].items() if k.startswith('net.') and not equal(v,initial['world'][k])]
        assert bool(changed)==bool(expected)
        if d['iteration']:
            row=by_iteration[d['iteration']];assert row['world_updates']==expected and row['transitions']==d['transitions'] and row['updates_this_run']*4==expected
        for name in ('actor','world','critic'):
            assert all(not value.is_floating_point() or torch.isfinite(value).all() for value in d[name].values())
        protected[str(path)]=sha(path)
        points.append(dict(path=str(path),sha256=sha(path),iteration=d['iteration'],transitions=d['transitions'],world_updates=expected,
            actual_world_optimizer_steps=expected,world_norm_count=count,delta_norm_count=actual_delta,norm_count_rounding=rounding,encoder_actor_critic_steps=[0,0,0],changed_world_net_tensors=changed))
    completed=False
    if args.exit_record:
        receipt=json.loads(args.exit_record.read_text());assert receipt['returncode']==0 and receipt['final_checkpoint_exists'] and (run/'final.pt').is_file()
        assert Path(receipt['log_dir']).resolve()==run and receipt['final_sha256']==sha(run/'final.pt')
        assert start<=datetime.fromisoformat(receipt['started_at_utc'])<=datetime.fromisoformat(receipt['finished_at_utc'])<=deadline
        assert (deadline-datetime.fromisoformat(receipt['finished_at_utc'])).total_seconds()>=budget['minimum_post_training_reserve_s']
        assert receipt['last_metrics']['elapsed_s']>=config['seconds']
        assert receipt['last_metrics']==metrics[-1]
        completed=True
    assert all(sha(p)==value for p,value in protected.items())
    return dict(source_count=len(manifest),source_manifest_sha256=sha(run/'source_sha256.json'),checkpoints=points,
        reward_only_actor_transfer=provenance,initial_physics_sha256=sha(physics_path),budget_record_sha256=sha(budget_path),
        actual_materials=dict(nominal_environments=len(observed),original_direct_environments=len(other),
            train=dict(total=115,nominal=25,original_direct=90),validation=dict(total=13,nominal=7,original_direct=6)),
        initial_physics_and_capture_rng_verified=True,world_and_value_cold_at_initial=True,all_four_initial_optimizers_empty=True,
        full_actor_and_critic_bitwise_frozen=True,three_optimizers_empty=True,completion_verified=completed,
        scope='Read-only V12 frozen-function/source/optimizer evidence; not prediction or behavior acceptance.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=32)
    p.add_argument('--parent','--parent-actor',type=Path,default=PARENT);p.add_argument('--run',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--budget-record',type=Path);p.add_argument('--startup',action='store_true')
    p.add_argument('--exit-record',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.manual_seed(55);assert ALGORITHM_CONTRACT==V12
    assert len(WEIGHTS)==13 and not REMOVED&WEIGHTS.keys()
    assert MOTION_AUXILIARY_RECIPE['version']=='motion-only-pd-soft-termination-v2'
    assert MOTION_AUXILIARY_RECIPE['height_reward']=='exp(-(height-target)^2/.0025)*upright_gate; no_contact_factor'
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Optimizer updates forbidden')):
        result=audit(args) if args.run else cpu_checks(args.parent) if args.device=='cpu' else integration(args)
    source=ROOT/'mjlab_training/src/ct_mjlab';hashes={str(f.relative_to(source)):sha(f) for f in sorted(source.rglob('*.py'))}
    report=dict(all_checks_passed=True,algorithm_contract=V12,world_loss_recipe=WORLD_LOSS_RECIPE,model_reward_recipe=MOTION_AUXILIARY_RECIPE,
        checker_sha256=sha(__file__),source_sha256=hashes,optimizer_steps=0,checks=result)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),all_checks_passed=True,optimizer_steps=0)))


if __name__=='__main__':main()
