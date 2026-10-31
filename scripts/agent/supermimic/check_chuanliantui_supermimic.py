#!/usr/bin/env python3
"""V11 true contact mixture/five-substep path, BPTT, checkpoint and ONNX; zero updates."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from functools import wraps
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ct_mjlab.supermimic import __main__ as entry
from ct_mjlab.supermimic.__main__ import make_env, collect
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE, Learner, frozen
from ct_mjlab.supermimic.export import export
from ct_mjlab.supermimic.analytic_reward import FEATURE_DIM, CONTEXT_DIM, WEIGHTS, capture, differentiable_reward
from ct_mjlab.supermimic.networks import model_context
from ct_mjlab.supermimic.state import (STATE_DIM, PARAM_DIM, PHYS_DIM, SUBSTEPS, MEMORY, NOISE,
    SOLREF, capture_state, capture_parameters, physical_snapshot)

V10='chuanliantui-supermimic-canonical-physics-contact-kb-fixed-scale-nominal-mix-v10'
V11='chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'
V9='chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9'
V8='chuanliantui-supermimic-contact-target-cap5-worldfit4-goal-isolated-bounded-value-bptt-analytic-reward-v8'


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def buffer_check(device):
    b=ReplayBuffer(6,20,device)
    assert b.FIELDS['state']==STATE_DIM and b.FIELDS['parameters']==PARAM_DIM and b.FIELDS['path']==SUBSTEPS*PHYS_DIM
    for t in range(10):
        tr={key:torch.full_like(value[0],float(t)) for key,value in b.data.items()}
        tr['done'].fill_(t==7);tr['failure'].fill_(t==7)
        tr['privileged'][:,0]=torch.arange(20,device=device);b.add(**tr)
    for validation in (False,True):
        w=b.sample(256,4,validation=validation);times=w['history'][...,0]
        assert times.min()>=4 and times.max()<=9 and torch.all(times[:,1:]-times[:,:-1]==1)
        assert torch.all((w['privileged'][:,0,0].long()%10==0)==validation)
        expected=torch.cumprod(torch.cat((torch.ones(256,1,device=device),(times[:,:-1]!=7).float()),1),1)
        assert torch.equal(w['mask'],expected)


def norm_check(learner):
    assert not hasattr(learner.world,'feature_norm') and not hasattr(learner.world,'reward_norm')
    for norm in (learner.world.in_norm,learner.world.delta_norm,learner.critic.norm):
        assert all(torch.isfinite(v).all() for v in norm.buffers())
        assert torch.isfinite(norm.std()).all() and (norm.std()>0).all()


def context_check(context):
    assert context.shape[-1]==CONTEXT_DIM and not context.isnan().any()
    unlimited=context.isinf();assert not unlimited[...,:7].any() and not unlimited[...,19:].any()
    safe=model_context(context);assert torch.isfinite(safe).all() and (safe[unlimited]==0).all()
    torch.testing.assert_close(safe[~unlimited],context[~unlimited],atol=0,rtol=0)
    return int(unlimited.sum())


def finite_gradients(module):
    gradients=[p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    return sum(float(g.abs().sum()) for g in gradients)


def contact_startup_observer(records):
    original=entry.supermimic_nominal_contact
    @wraps(original)
    def observed(raw,env_ids):
        geoms=raw._standup_randomization.collision_geom_ids
        selected=torch.arange(raw.num_envs,device=raw.device)%4==0
        before=raw.sim.model.geom_solref.clone()
        noncontact={name:getattr(raw.sim.model,name).clone()
            for name in ('body_mass','body_inertia','body_ipos','geom_friction')}
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
        assert (before[:,geoms]<0).all(),'Expected the original randomization before coverage'
        original(raw,env_ids)
        after=raw.sim.model.geom_solref.clone();expected=before.clone()
        nominal=raw.sim.get_default_field('geom_solref')
        ids=selected.nonzero().flatten()
        expected[ids[:,None],geoms[None,:]]=nominal[geoms][None,:]
        torch.testing.assert_close(after,expected,atol=0,rtol=0)
        for name,value in noncontact.items():assert torch.equal(value,getattr(raw.sim.model,name).clone()),name
        assert torch.equal(cpu_rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        torch.testing.assert_close(after[ids[:,None],geoms[None,:]],
            after.new_tensor([.02,1.]).expand(len(ids),len(geoms),2),atol=0,rtol=0)
        records.append(dict(selected_environment_ids=ids.cpu().tolist(),
            unchanged_randomized_environment_ids=(~selected).nonzero().flatten().cpu().tolist(),
            original_direct_format_all_geometries=True,all_robot_collision_geometries_checked=len(geoms),
            before_raw_solref=before[:,geoms[0]].cpu().tolist(),after_raw_solref=after[:,geoms[0]].cpu().tolist(),
            all_other_geometries_unchanged=True,noncontact_mass_inertia_ipos_friction_unchanged=True,
            torch_cpu_and_cuda_rng_unchanged=True))
    return observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.steps<16:parser.error('At least16 steps required')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=False);assert ALGORITHM_CONTRACT==V11
    buffer_check(args.device)
    args.task,args.seed,args.num_envs='chuanliantui_walk',42,20
    contact_startup=[]
    with patch.object(entry,'supermimic_nominal_contact',contact_startup_observer(contact_startup)):
        env,cfg=make_env(args)
    assert len(contact_startup)==1
    try:
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg))
        learner=Learner(args.device);adapter=learner.world.adapter;raw=env.unwrapped
        contact_parameters=capture_parameters(raw)
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
        entry.supermimic_nominal_contact(raw,None)
        torch.testing.assert_close(capture_parameters(raw),contact_parameters,atol=0,rtol=0)
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        torch.testing.assert_close(contact_parameters[:,SOLREF],
            contact_parameters.new_tensor(contact_startup[0]['after_raw_solref']),atol=0,rtol=0)
        originals={name:[p.detach().clone() for p in getattr(learner,name).parameters()] for name in ('actor','world','critic')}
        b=ReplayBuffer(64,20,args.device);obs=env.get_observations()
        failures=resets=inf_context=nonreset_frames=0
        max_errors={name:0. for name in ('features','state','memory','actor25','history125','reward','terms','derived_reward')}
        reward_state=raw._split_walk_rewards;original_compute=reward_state.compute
        reward_records=[];substeps=[];parameter_records=[]
        def observed_compute(environment):
            if reward_state.step!=environment.common_step_counter:
                record=capture(environment,reward_state);record['state']=capture_state(environment)
                reward_records.append(record)
            return original_compute(environment)
        def close(name,left,right,*,atol=2e-5,rtol=2e-5):
            if left.numel():
                max_errors[name]=max(max_errors[name],float((left-right).abs().max()))
                torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
        with ExitStack() as guards:
            opt_names=('opt_world','opt_encoder','opt_actor','opt_critic')
            mocked=[guards.enter_context(patch.object(getattr(learner,name),'step')) for name in opt_names]
            for _ in range(args.steps):
                torch.testing.assert_close(obs['history'][:,-25:],obs['actor'],atol=0,rtol=0)
                manager=raw.action_manager;previous=manager.action.clone();previous_previous=manager.prev_action.clone()
                before={name:obs[name].clone() for name in ('history','critic','velocity_target')}
                before_state=capture_state(raw)
                reward_records.clear();substeps.clear();parameter_records.clear()
                original_substep=raw.metrics_manager.compute_substep
                def observed_substep():
                    original_substep()
                    if not substeps:parameter_records.append(capture_parameters(raw))
                    substeps.append(physical_snapshot(raw))
                with patch.object(reward_state,'compute',side_effect=observed_compute),patch.object(raw.metrics_manager,'compute_substep',side_effect=observed_substep):
                    obs,reward,done=collect(env,learner,b,obs)
                assert len(reward_records)==1 and len(substeps)==SUBSTEPS and len(parameter_records)==1
                expected=reward_records[0];record=reward_state.record;row=(b.ptr-1)%b.capacity
                path=torch.stack(substeps,1);params=b.data['parameters'][row]
                torch.testing.assert_close(params,parameter_records[0],atol=0,rtol=0)
                torch.testing.assert_close(params[:,SOLREF],contact_parameters[:,SOLREF],atol=0,rtol=0)
                torch.testing.assert_close(b.data['state'][row],before_state,atol=0,rtol=0)
                torch.testing.assert_close(b.data['path'][row],path.flatten(-2),atol=0,rtol=0)
                torch.testing.assert_close(b.data['next_state'][row],expected['state'],atol=0,rtol=0)
                for name in ('features','context','action','previous','previous_previous'):
                    torch.testing.assert_close(record[name],expected[name],atol=0,rtol=0)
                    torch.testing.assert_close(b.data[name][row],expected[name],atol=0,rtol=0)
                torch.testing.assert_close(record['previous'],previous,atol=0,rtol=0)
                torch.testing.assert_close(record['previous_previous'],previous_previous,atol=0,rtol=0)
                for field,name in (('history','history'),('privileged','critic'),('velocity','velocity_target')):
                    torch.testing.assert_close(b.data[field][row],before[name],atol=0,rtol=0)
                torch.testing.assert_close(b.data['next_frame'][row],obs['actor'],atol=0,rtol=0)
                torch.testing.assert_close(b.data['next_privileged'][row],obs['critic'],atol=0,rtol=0)
                torch.testing.assert_close(b.data['reward'][row],reward,atol=0,rtol=0)
                assert torch.equal(b.data['done'][row],done.bool()) and torch.equal(b.data['failure'][row],raw.reset_terminated.bool())
                inf_context+=context_check(record['context'])
                rebuilt=adapter.advance(before_state,params,path,obs['actor'][:,6:9],record['action'],previous,previous_previous,
                    record['context'],record['state'][:,NOISE])
                state,frame,derived_reward,failure,features=rebuilt
                close('state',state,record['state']);close('memory',state[:,MEMORY],record['state'][:,MEMORY])
                close('features',features,record['features'])
                close('derived_reward',derived_reward,reward)
                assert torch.equal(failure,raw.reset_terminated.bool()),'Hard failure from physical path/memory differs'
                # The path is pre-reset; terminal returned observations belong to a new episode.
                valid=~done.bool();nonreset_frames+=int(valid.sum())
                close('actor25',frame[valid],obs['actor'][valid])
                history=torch.cat((before['history'][:,25:],frame),-1)
                close('history125',history[valid],obs['history'][valid])
                # Actual actuator forces are reused, never replaced by a PD estimate.
                torch.testing.assert_close(features[:,22:28],path[:,-1,49:55],atol=0,rtol=0)
                terms,measured=differentiable_reward(record['features'],record['action'],previous,previous_previous,record['context'])
                assert terms.keys()==reward_state.values.keys()==WEIGHTS.keys()
                close('reward',measured,reward,atol=2e-7,rtol=2e-5)
                for name,value in terms.items():close('terms',value,reward_state.values[name],atol=2e-6,rtol=2e-5)
                derived_terms,_=differentiable_reward(features,record['action'],previous,previous_previous,record['context'])
                for name,value in derived_terms.items():close('terms',value,reward_state.values[name])
                failures+=int(raw.reset_terminated.sum());resets+=int(done.sum())
            assert all(mock.call_count==0 for mock in mocked)
            assert inf_context>0 and nonreset_frames>0
            w=b.sample(32,4);learner.world.observe(w);norm_check(learner)
            inputs=learner.world.inputs(w['state'],w['parameters'],w['action'],w['previous'],w['previous_previous'])
            assert torch.isfinite(inputs).all()
            world_loss,errors=learner.world_loss(w);world_loss.backward()
            world_grad=finite_gradients(learner.world);assert world_grad>0
            torch.testing.assert_close(world_loss,errors['dynamics'],atol=0,rtol=0)
            for module in (learner.actor,learner.world,learner.critic):module.zero_grad(set_to_none=True)
            starts=b.sample(32,8)
            with frozen(learner.world,learner.critic,learner.actor.encoder):
                actor_loss,states,parameters,targets,mask,contexts=learner.imagination(starts);actor_loss.backward()
            actor_grad=finite_gradients(learner.actor.mlp)
            if not mask.any():assert actor_grad==0.
            assert states.shape==(9,32,STATE_DIM) and parameters.shape==(9,32,PARAM_DIM)
            assert contexts.shape==(9,32,CONTEXT_DIM) and mask.shape==(8,32)
            assert torch.isfinite(targets).all() and torch.isfinite(learner.critic(states,parameters,contexts)).all()
            for module in (learner.world,learner.critic,learner.actor.encoder):assert all(p.grad is None for p in module.parameters())
            for module in (learner.actor,learner.world,learner.critic):module.zero_grad(set_to_none=True)
            metrics=learner.update(b,batch=16,window=4,horizon=8,policy=True)
            # Real replay may contain no eligible constant tracking/parking windows.
            calls=[mock.call_count for mock in mocked]
            expected_policy=int(metrics['imagination_eligible_fraction']>0)
            assert calls==[4,1,expected_policy,expected_policy]
            assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        norm_check(learner)
        for name,values in originals.items():assert all(torch.equal(a,b) for a,b in zip(values,getattr(learner,name).parameters()))
        assert all(not getattr(learner,name).state for name in opt_names)
        command=raw.command_manager.get_term('standup')
        config=dict(task=args.task,fixture_kind='untrained_zero_update_v11_integration',parameter_updates=0,
            contains_synthetic_optimizer_slots=False,world_steps_per_update=4,
            world_loss_recipe=WORLD_LOSS_RECIPE.copy(),
            contact_training_recipe=entry.CONTACT_TRAINING_RECIPE.copy(),
            contact_encoding_assumptions=raw._supermimic_contact_encoding,
            initialization=dict(kind='untrained_zero_update_v11_integration',fresh_from_zero=True))
        fixture=args.output/'zero_update_fixture.pt'
        learner.save(fixture,command,config=config,iteration=0,transitions=20*args.steps,world_updates=0)
        restored=Learner(args.device);restored.load(fixture,command)
        for name in ('actor','world','critic',*opt_names):assert equal(getattr(learner,name).state_dict(),getattr(restored,name).state_dict()),name
        # Separate fixture tests state restoration, not evidence of actual training.
        for name in opt_names:
            optimizer=getattr(restored,name)
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    optimizer.state[parameter]=dict(step=torch.tensor(28. if name=='opt_world' else 7.),
                        exp_avg=torch.full_like(parameter,.003),exp_avg_sq=torch.full_like(parameter,.004))
        synthetic=args.output/'synthetic_optimizer_restore_fixture.pt'
        synthetic_config=dict(config,fixture_kind='synthetic_optimizer_restore_only',contains_synthetic_optimizer_slots=True)
        restored.save(synthetic,command,config=synthetic_config,iteration=0,transitions=20*args.steps,world_updates=28)
        restored_again=Learner(args.device);restored_again.load(synthetic,command)
        for name in ('actor','world','critic',*opt_names):assert equal(getattr(restored,name).state_dict(),getattr(restored_again,name).state_dict()),name
        old_checks=[]
        for contract,version in ((V8,'v8'),(V9,'v9'),(V10,'v10')):
            wrong=torch.load(fixture,map_location='cpu',weights_only=False);wrong['algorithm_contract']=contract
            wrong['config']['fixture_kind']='synthetic_old_contract_rejection_only'
            old=args.output/f'incompatible_{version}_fullresume_fixture.pt';torch.save(wrong,old)
            before_load={name:{k:v.clone() for k,v in getattr(restored,name).state_dict().items()}
                for name in ('actor','world','critic')}
            try:restored.load(old,command)
            except ValueError:pass
            else:raise AssertionError(f'{version} fullresume accepted')
            assert all(equal(before_load[name],getattr(restored,name).state_dict()) for name in before_load)
            restored.load(old,command,inference=True)
            old_export=export(old,args.output/f'synthetic_{version}_actor_only.onnx')
            assert old_export['algorithm_contract']==contract
            old_checks.append(dict(contract=contract,fullresume_rejected_without_network_mutation=True,
                actor_only_accepted=True,actor_export_contract=old_export['algorithm_contract'],
                onnx_max_error=old_export['numeric_max_error']))
        exported=export(fixture,args.output/'zero_update_fixture.onnx')
        assert exported['algorithm_contract']==V11
        source=Path(__file__).resolve().parents[3]/'mjlab_training/src/ct_mjlab'
        shared={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob('*.py')) if not str(path.relative_to(source)).startswith('supermimic/')}
        from ct_mjlab.supermimic.learner import asset_digest
        report=dict(algorithm_contract=ALGORITHM_CONTRACT,task=args.task,num_envs=20,steps=args.steps,
            non_supermimic_shared_source_sha256=shared,shared_source_file_count=len(shared),actual_xml_sha256=asset_digest(),
            real_transitions=20*args.steps,physics_samples=20*args.steps*SUBSTEPS,state_dim=STATE_DIM,
            parameter_dim=PARAM_DIM,critic_input_dim=STATE_DIM+PARAM_DIM+CONTEXT_DIM,
            policy_dt_s=raw.step_dt,physics_dt_s=raw.physics_dt,optimizer_steps=0,network_parameters_unchanged=True,
            contact_startup=contact_startup[0],contact_recipe=entry.CONTACT_TRAINING_RECIPE,
            contact_encoding_assumptions=raw._supermimic_contact_encoding,
            contact_coverage_idempotent_and_all75_capture_parameters_unchanged=True,
            raw_solref_and_independent_true_path_aligned_every_transition=True,
            nominal_contact_transitions=5*args.steps,original_randomized_contact_transitions=15*args.steps,
            real_resets=resets,physical_failures=failures,nonreset_derived_actor25_history125_rows=nonreset_frames,
            maximum_reconstruction_errors=max_errors,all18_original_reward_terms_checked=True,
            five_substep_path_independently_recorded=True,pre_reset_canonical_state_matches=True,
            hard_failure_exact=True,actual_torque_channels_not_PD=True,delayed_noisy_actor_history_matches_nonreset=True,
            replay_wrap_and_episode_mask=True,heldout_env_partition=True,unlimited_context_inf_values=inf_context,
            world_loss_only_physical_dynamics=True,world_gradient_l1=world_grad,bptt_actor_gradient_l1=actor_grad,
            zero_update_optimizer_step_mock_calls=calls,checkpoint_roundtrip_all_four_optimizers=True,
            synthetic_optimizer_fixture=str(synthetic),synthetic_slots_separate_not_training=True,
            old_fullresume_rejection_checks=old_checks,v8_fullresume_rejected=True,v9_fullresume_rejected=True,v10_fullresume_rejected=True,
            known_v8_v9_v10_actor_only_inference_accepted=True,
            onnx_max_error=exported['numeric_max_error'],old_actor_export_keeps_original_contract=True,
            outside_scope='Tracking/parkingflags use recorded constant windows; not all command flags are closedloop. No training or behavior acceptance.')
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(report,indent=2,ensure_ascii=False))
    finally:env.close()


if __name__=='__main__':main()
