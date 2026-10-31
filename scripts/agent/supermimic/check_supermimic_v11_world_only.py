#!/usr/bin/env python3
"""Read-only CPU evidence audit for frozen-actor V11 world fitting, never training."""
# Adapted from check_supermimic_world_only.py; original SHA256 79c69b2f9a1cdb698ec9db78c16c3d193eb611eee8f58ee4f654aa9583a3e7d4
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

from scripts.agent.supermimic import check_supermimic_training as formal
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE
from ct_mjlab.supermimic.state import STATE_DIM,PARAM_DIM,PHYS_DIM,SUBSTEPS,SOLREF
from ct_mjlab.envs.chuanliantui.standup_randomization import restitution_to_solref

ROOT=Path(__file__).resolve().parents[3]
PARENT=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261031_v7_worldfit4_actortransfer3600/checkpoint_000615.pt'
PARENT_SHA='50ee90a69e6f58344898277a77ff850d2504dbda31dd3ff4b8beb6473ef5105f'
V10='chuanliantui-supermimic-canonical-physics-contact-kb-fixed-scale-nominal-mix-v10'
V11='chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'
RECIPE=dict(num_envs=128,seed=42,horizon=24,window=8,batch=128,buffer_steps=512,
            policy_warmup_updates=1000000000,actor_lr=3e-5)
FROZEN_OPTS=('opt_actor','opt_encoder','opt_critic')


def bitwise(first,second):
    if isinstance(first,torch.Tensor):
        return (isinstance(second,torch.Tensor) and first.dtype==second.dtype and first.shape==second.shape
            and first.detach().cpu().contiguous().numpy().tobytes()==second.detach().cpu().contiguous().numpy().tobytes())
    if isinstance(first,dict):
        return isinstance(second,dict) and first.keys()==second.keys() and all(bitwise(first[k],second[k]) for k in first)
    if isinstance(first,(tuple,list)):
        return isinstance(second,type(first)) and len(first)==len(second) and all(bitwise(a,b) for a,b in zip(first,second))
    return type(first)==type(second) and first==second


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def physics_checks(physics,initial,config,check):
    check('initial_physics_schema',physics['schema']=='supermimic-initial-physics-v1')
    check('initial_physics_identity',physics['algorithm_contract']==V11 and physics['task']==config['task']
        and physics['seed']==42 and physics['num_envs']==128 and physics['policy_dt_s']==.01 and physics['physics_dt_s']==.002)
    state,params=physics['state'],physics['parameters']
    check('initial_physics_shapes',state.shape==(128,STATE_DIM) and params.shape==(128,PARAM_DIM)
        and state.dtype==params.dtype==torch.float32 and state.device.type==params.device.type=='cpu')
    check('initial_physics_finite',not formal.finite_tree(physics),formal.finite_tree(physics))
    ids=torch.arange(128);expected_nominal=ids[ids%4==0]
    train,validation=ids[ids%10!=0],ids[ids%10==0]
    check('actual_replay_partition_ids',bitwise(physics['train_env_ids'],train) and bitwise(physics['validation_env_ids'],validation))
    geoms,solref,default=(physics[k] for k in ('robot_collision_geom_ids','robot_collision_solref','default_robot_collision_solref'))
    check('actual_collision_material_shapes',geoms.ndim==1 and len(geoms)>0 and geoms.dtype==torch.long
        and solref.shape==(128,len(geoms),2) and default.shape==(len(geoms),2))
    nominal=solref.new_tensor([.02,1.])
    check('actual_compiled_default_contact',torch.equal(default,nominal.expand_as(default)))
    check('all_robot_geometries_share_recorded_solref',torch.equal(solref,params[:,None,SOLREF].expand_as(solref)))
    observed=(solref==default[None]).all(dim=(-1,-2)).nonzero().flatten()
    check('actual_nominal_ids_exact_mod4',bitwise(observed,expected_nominal) and bitwise(physics['nominal_contact_env_ids'],observed))
    random_ids=ids[ids%4!=0];original=restitution_to_solref(physics['original_restitution'])
    check('original_restitution_range',physics['original_restitution'].shape==(128,)
        and ((physics['original_restitution']>=.6)&(physics['original_restitution']<=1.)).all())
    reference_error=float((params[random_ids,SOLREF]-original[random_ids]).abs().max())
    check('other_96_materials_match_original_randomization',
        (params[random_ids,SOLREF.start]==10000.*-1).all()
        and torch.isclose(params[random_ids,SOLREF],original[random_ids],atol=1e-5,rtol=1e-6).all(),
        dict(max_abs_error=reference_error,absolute_tolerance=1e-5,relative_tolerance=1e-6,
             reason='Original GPU log/sqrt reference is recomputed on CPU; raw recorded values are unchanged.'))
    check('snapshot_contact_recipe_and_assumptions',physics['contact_training_recipe']==config['contact_training_recipe']
        and physics['contact_encoding_assumptions']==config['contact_encoding_assumptions'])
    check('capture_did_not_step_or_change_state',physics['state_and_parameters_unchanged_during_capture'] is True
        and physics['common_step_counter_before']==physics['common_step_counter_after'])
    check('capture_rng_bitwise_unchanged',physics['rng_unchanged_during_capture'] is True
        and bitwise(physics['torch_rng_before'],physics['torch_rng_after'])
        and bitwise(physics['cuda_rng_before'],physics['cuda_rng_after']))
    check('initial_checkpoint_rng_follows_capture',bitwise(physics['torch_rng_after'],initial['torch_rng'])
        and bitwise(physics['cuda_rng_after'],initial['cuda_rng']))
    nominal_set=set(observed.tolist())
    return dict(nominal_solref_environments=len(observed),original_direct_environments=len(random_ids),
        nominal_environment_ids=observed.tolist(),original_direct_environment_ids=random_ids.tolist(),
        train=dict(total=len(train),nominal=sum(i in nominal_set for i in train.tolist()),original_direct=sum(i not in nominal_set for i in train.tolist())),
        validation=dict(total=len(validation),nominal=sum(i in nominal_set for i in validation.tolist()),original_direct=sum(i not in nominal_set for i in validation.tolist())),
        scope='Nominal SOLREF only; all other original physical randomization remains active. Random replay batches need not contain exactly25% nominal samples.')


def recipe_checks(config,check):
    check('v11_current_algorithm_contract',ALGORITHM_CONTRACT==V11)
    check('v11_world_only_recipe',all(config.get(k)==v for k,v in RECIPE.items()))
    check('v11_fixed_world_loss_recipe',config.get('world_loss_recipe')==WORLD_LOSS_RECIPE)


def audit(args,report):
    checks=report['checks']
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    run=args.run.resolve();config=json.loads((run/'config.json').read_text())
    recipe_checks(config,check)
    check('no_full_resume',config.get('resume') is None)
    check('parent_path_matches_config',Path(config['init_actor']).resolve()==args.parent_actor.resolve())
    protected={str(path):formal.sha256(path) for path in (args.parent_actor,run/'config.json',run/'initial_physics.pt',args.budget_record)}
    check('fixed_sm615_parent_sha256',protected[str(args.parent_actor)]==PARENT_SHA)
    parent=formal.load_checkpoint(args.parent_actor);initial=formal.load_checkpoint(run/'initial.pt')
    check('initial_full_actor_bitwise_sm615',bitwise(initial['actor'],parent['actor']))
    check('initial_world_and_value_cold_norms',all(float(initial[name][key])==0 for name,key in
        (('world','in_norm.count'),('world','delta_norm.count'),('critic','norm.count'))))
    check('initial_four_optimizers_empty',all(not initial[name]['state'] and formal.optimizer_steps(initial[name])==0 for name in formal.OPTIMIZERS))
    check('initial_world_counter_zero',initial['world_updates']==0 and initial['iteration']==0 and initial['transitions']==0)
    check('initial_physics_file_hash_matches_config',config.get('initial_physics_sha256')==protected[str(run/'initial_physics.pt')])
    physics=formal.load_checkpoint(run/'initial_physics.pt')
    report['actual_materials']=physics_checks(physics,initial,config,check)
    budget=json.loads(args.budget_record.read_text())
    check('budget_recipe_matches',budget['algorithm_contract']==V11
        and budget['world_loss_recipe']==WORLD_LOSS_RECIPE
        and budget['midpoint_threshold_s']==400
        and (ROOT/budget['actor_source']).resolve()==args.parent_actor.resolve()
        and budget['recipe']==RECIPE and budget['world_steps_per_update']==4
        and budget['training_loop_target_s']==config['seconds'] and budget['world_learning_rate']==.0003
        and budget['training_exploration_std']==.2)
    start=datetime.fromisoformat(budget['started_at_utc']);deadline=datetime.fromisoformat(budget['deadline_utc'])
    check('explicit_total_budget_timeline',(deadline-start).total_seconds()==budget['total_wall_clock_budget_s']==1800
        and budget['minimum_post_training_reserve_s']>=300)
    report['budget']=dict(path=str(args.budget_record.resolve()),sha256=formal.sha256(args.budget_record),contents=budget,
        scope='This audit covers fitting evidence; external capture/scoring receipts must separately finish inside the same total deadline.')
    lines=(run/'metrics.jsonl').read_bytes().splitlines(keepends=True)
    if args.startup and lines and not lines[-1].endswith(b'\n'):lines=lines[:-1]
    metrics=[json.loads(line) for line in lines if line.strip()]
    by_iteration={row['iteration']:row for row in metrics}
    candidates=sorted(run.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
    requested=args.checkpoint.resolve() if args.checkpoint else (run/'final.pt' if (run/'final.pt').is_file() else candidates[-1] if candidates else run/'initial.pt')
    check('selected_checkpoint_in_this_run',requested.parent==run and (requested.name in ('initial.pt','final.pt')
        or re.fullmatch(r'checkpoint_\d+\.pt',requested.name) is not None))
    if requested.name.startswith('checkpoint_') and args.checkpoint is not None:
        if args.selection_seconds is None:raise ValueError('--selection-seconds is required for a numbered selection')
        check('declared_midpoint_400_seconds',args.selection_seconds==budget['midpoint_threshold_s']==400)
        eligible=[p for p in candidates if by_iteration.get(int(p.stem.split('_')[1]),{}).get('elapsed_s',-1)>=args.selection_seconds]
        check('first_published_checkpoint_at_declared_time',bool(eligible) and eligible[0].resolve()==requested)
    selected_paths=[('initial',run/'initial.pt'),('selected',requested)]
    if (run/'final.pt').is_file() and requested.name!='final.pt':selected_paths.append(('final',run/'final.pt'))
    report['checkpoints']=[]
    for label,path in selected_paths:
        saved=initial if label=='initial' else formal.load_checkpoint(path)
        protected[str(path)]=formal.sha256(path)
        check(label+'_v11_config_identity',saved['algorithm_contract']==V11 and saved['config']==config
            and saved['asset_sha256']==initial['asset_sha256'] and saved['infos']==initial['infos'])
        check(label+'_all_networks_and_optimizers_finite',not formal.finite_tree({n:saved[n] for n in formal.NETWORKS+formal.OPTIMIZERS}))
        check(label+'_full_actor_bitwise_initial_and_sm615',bitwise(saved['actor'],initial['actor']) and bitwise(saved['actor'],parent['actor']))
        check(label+'_complete_critic_including_norm_bitwise_initial',bitwise(saved['critic'],initial['critic']))
        check(label+'_three_frozen_optimizers_bitwise_empty',all(not saved[n]['state'] and formal.optimizer_steps(saved[n])==0
            and bitwise(saved[n],initial[n]) for n in FROZEN_OPTS))
        steps=formal.optimizer_steps(saved['opt_world']);world_updates=saved['world_updates']
        check(label+'_actual_world_adam_steps',steps==world_updates and type(world_updates) is int
            and 0<=world_updates<RECIPE['policy_warmup_updates'] and world_updates%4==0)
        changed=formal.parameter_change(initial['world'],saved['world'])
        check(label+'_world_weights_change_iff_real_steps',bool(changed['changed_tensors'])==bool(steps))
        count=float(saved['world']['in_norm.count']);delta_count=float(saved['world']['delta_norm.count'])
        check(label+'_world_norm_counts_five_substeps',math.isfinite(count) and count>=0 and delta_count==SUBSTEPS*count
            and (count>0)==(steps>0))
        check(label+'_world_norm_state_changed_iff_real_steps',
            (not bitwise({k:v for k,v in saved['world'].items() if k.startswith(('in_norm.','delta_norm.'))},
                         {k:v for k,v in initial['world'].items() if k.startswith(('in_norm.','delta_norm.'))}))==bool(steps))
        if saved['iteration']>0:
            row=by_iteration.get(saved['iteration'])
            check(label+'_published_metric_counters',row is not None and row['world_updates']==world_updates
                and row['transitions']==saved['transitions'] and row['updates_this_run']*4==world_updates)
        else:row=None
        report['checkpoints'].append(dict(label=label,path=str(path),sha256=protected[str(path)],iteration=saved['iteration'],
            transitions=saved['transitions'],world_updates=world_updates,optimizer_steps=dict(world=steps,encoder=0,actor=0,critic=0),
            world_input_norm_count=count,world_delta_norm_count=delta_count,world_parameter_change=changed,
            actor_state_sha256=state_digest(saved['actor']),critic_state_sha256=state_digest(saved['critic']),
            matching_metric=row))
    check('no_policy_metrics_anywhere',not any(set(row)&{'actor','critic','actor_grad','imagination_eligible_fraction'} for row in metrics))
    # Reuse the V11-aware comprehensive auditor for source99, task/XML, counters, provenance and actual exit.
    base=dict(checks=[])
    formal.audit(SimpleNamespace(run=run,startup=args.startup,exit_record=args.exit_record),base)
    report['formal_training_audit']=base
    check('formal_training_audit_all_passed',base['all_checks_passed'],[c['name'] for c in base['checks'] if not c['passed']])
    check('exact_99_training_sources',base['source']['archived_file_count']==99)
    if not args.startup:
        check('completed_requires_real_final',(run/'final.pt').is_file() and base['completion_verified'])
        receipt=json.loads(args.exit_record.read_text())
        finish=datetime.fromisoformat(receipt['finished_at'])
        check('real_training_inside_total_budget',datetime.fromisoformat(receipt['started_at'])>=start and finish<=deadline)
        reserve=(deadline-finish).total_seconds()
        check('actual_post_training_reserve_at_least_300s',reserve>=budget['minimum_post_training_reserve_s'])
        report['budget']['post_training_remaining_s']=reserve
        protected[str(args.exit_record)]=formal.sha256(args.exit_record)
    for path,sha in protected.items():check('immutable_'+Path(path).name,formal.sha256(path)==sha)
    report['protected_file_sha256']=protected
    report['all_checks_passed']=all(row['passed'] for row in checks)
    report['completion_verified']=bool(not args.startup and report['all_checks_passed'])
    report['status']=('world_only_startup_frozen_evidence_verified' if args.startup else 'world_only_training_completed_actor_value_frozen') if report['all_checks_passed'] else 'world_only_evidence_rejected'


def selfcheck():
    from unittest.mock import patch
    from ct_mjlab.supermimic import __main__ as entry
    from ct_mjlab.supermimic.buffer import ReplayBuffer
    ids=torch.arange(128);geoms=torch.tensor([1,2]);default=torch.tensor([[.02,1.]]).expand(3,-1).clone()
    restitution=torch.full((128,),.8);solref=restitution_to_solref(restitution)[:,None].expand(-1,3,-1).clone()
    solref[ids[::4,None],geoms[None,:]]=default[geoms][None]
    state=torch.zeros(128,STATE_DIM);params=torch.zeros(128,PARAM_DIM);params[:,SOLREF]=solref[:,1]
    proof=dict(fixture='synthetic_only')
    raw=SimpleNamespace(num_envs=128,common_step_counter=0,step_dt=.01,physics_dt=.002,
        _standup_randomization=SimpleNamespace(collision_geom_ids=geoms,restitution=restitution),
        _supermimic_contact_encoding=proof,sim=SimpleNamespace(model=SimpleNamespace(geom_solref=solref),get_default_field=lambda _name:default))
    buffer=ReplayBuffer(1,128,'cpu')
    with patch.object(entry,'capture_state',return_value=state.clone()),patch.object(entry,'capture_parameters',return_value=params.clone()):
        physics=entry.capture_initial_physics(raw,buffer,task='chuanliantui_walk',seed=42)
    initial=dict(torch_rng=physics['torch_rng_after'],cuda_rng=physics['cuda_rng_after'])
    config=dict(RECIPE,task='chuanliantui_walk',world_loss_recipe=WORLD_LOSS_RECIPE.copy(),
        contact_training_recipe=entry.CONTACT_TRAINING_RECIPE.copy(),contact_encoding_assumptions=proof)
    rows=[]
    recipe_checks(config,lambda name,passed,detail=None:rows.append((name,bool(passed))))
    materials=physics_checks(physics,initial,config,lambda name,passed,detail=None:rows.append((name,bool(passed))))
    assert all(value for _,value in rows),rows
    assert materials['train']==dict(total=115,nominal=25,original_direct=90)
    assert materials['validation']==dict(total=13,nominal=7,original_direct=6)
    assert not bitwise(torch.tensor([-0.]),torch.tensor([0.]))
    corrupted=dict(physics,nominal_contact_env_ids=ids[1::4])
    bad=[];physics_checks(corrupted,initial,config,lambda name,passed,detail=None:bad.append((name,bool(passed))))
    assert ('actual_nominal_ids_exact_mod4',False) in bad
    old_physics=dict(physics,algorithm_contract=V10)
    old_rows=[];physics_checks(old_physics,initial,config,lambda name,passed,detail=None:old_rows.append((name,bool(passed))))
    assert ('initial_physics_identity',False) in old_rows
    missing=dict(config);missing.pop('world_loss_recipe')
    tampered=dict(config,world_loss_recipe=dict(WORLD_LOSS_RECIPE,wheel_vertical_force_scale_N=10.))
    for invalid in (missing,tampered):
        failures=[];recipe_checks(invalid,lambda name,passed,detail=None:failures.append((name,bool(passed))))
        assert ('v11_fixed_world_loss_recipe',False) in failures
    return dict(synthetic=True,actual_training_evidence=False,checks=len(rows),capture_helper_zero_rng_and_state_mutation=True,
        actual_helper_tested_without_simulator=True,material_ids_and_partitions_checked=True,wrong_ids_rejected=True,
        signed_zero_bitwise_difference_detected=True,v10_physics_contract_rejected=True,
        missing_world_loss_recipe_rejected=True,tampered_world_loss_recipe_rejected=True,
        optimizer_steps=0,physics_steps=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--selection-seconds',type=float);parser.add_argument('--parent-actor',type=Path,default=PARENT)
    parser.add_argument('--budget-record',type=Path);parser.add_argument('--exit-record',type=Path)
    parser.add_argument('--startup',action='store_true');parser.add_argument('--selfcheck',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not args.selfcheck and (args.run is None or args.budget_record is None or not args.startup and args.exit_record is None):
        parser.error('--run and --budget-record required; completed audit also requires --exit-record')
    torch.set_num_threads(1)
    report=dict(algorithm_contract=ALGORITHM_CONTRACT,world_loss_recipe=WORLD_LOSS_RECIPE.copy(),
        original_v10_checker_sha256='79c69b2f9a1cdb698ec9db78c16c3d193eb611eee8f58ee4f654aa9583a3e7d4',
        checker_sha256=formal.sha256(__file__),formal_checker_sha256=formal.sha256(formal.__file__),
        audited_at_utc=datetime.now(timezone.utc).isoformat(),checks=[],device='cpu',model_forward_calls=0,
        optimizer_steps_performed=0,simulator_steps_performed=0,
        scope='Frozen-function/fitting provenance evidence only; no prediction accuracy, policy improvement or task acceptance.')
    try:
        if args.selfcheck:report.update(status='synthetic_interface_selfcheck_passed',selfcheck=selfcheck(),all_checks_passed=True,completion_verified=False)
        else:audit(args,report)
    except Exception as error:
        report.update(status='world_only_evidence_rejected',all_checks_passed=False,completion_verified=False,
            error=dict(type=type(error).__name__,message=str(error)))
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(formal.json_safe(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),status=report['status'],checks=len(report['checks']),
        failures=[row['name'] for row in report['checks'] if not row['passed']],error=report.get('error'))))
    if not report['all_checks_passed']:raise SystemExit(2)


if __name__=='__main__':main()
