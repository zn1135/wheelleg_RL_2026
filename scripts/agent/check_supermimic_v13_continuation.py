#!/usr/bin/env python3
"""Finite CPU checks of a full V13 continuation source and completed runs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

# This checker never owns the training GPU or constructs a simulation environment.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from ct_mjlab.supermimic import learner as module
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.runner import contracts, validate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'artifacts/supermimic/supermimic-v13-continuation-20261101-r1'
COMPONENTS = ('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic')
COUNTERS = ('world','encoder','actor','critic')
LRS = dict(actor=3e-5,encoder=1e-3,world=3e-4,critic=3e-4)
RESUME_FIELDS = ('task','num_envs','seed','horizon','window','batch','buffer_steps',
    'policy_warmup_updates','actor_lr','world_steps_per_update','world_loss_recipe',
    'alternating_recipe','contact_training_recipe','contact_encoding_assumptions')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return (isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(value,b[key]) for key,value in a.items())
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def json_form(value):
    return json.loads(json.dumps(value,allow_nan=False))


def checkpoint(path):
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def configuration(saved):
    config = saved['config']
    assert saved['algorithm_contract']==module.ALGORITHM_CONTRACT
    assert saved['asset_sha256']==module.asset_digest()
    assert not saved.get('interrupted',False)
    assert config['world_steps_per_update']==module.WORLD_STEPS_PER_UPDATE==4
    assert config['world_loss_recipe']==module.WORLD_LOSS_RECIPE
    assert config['alternating_recipe']==module.ALTERNATING_RECIPE
    assert config['policy_warmup_updates']==256 and config['actor_lr']==LRS['actor']
    assert config['task']=='chuanliantui_walk' and config['mode']=='train'
    count_only = config.get('termination_mode')=='actor_update_count_only'
    if count_only:
        assert config['budget_kind']=='actual_actor_updates'
        assert config['seconds'] is None and config['hard_deadline_utc'] is None
    else:
        assert isinstance(config['seconds'],(int,float)) and math.isfinite(config['seconds']) and config['seconds']>0
    cfg = command_cfg('walk')
    assert saved['infos']==contracts(cfg)
    validate(dict(saved['infos'],standup_curriculum=saved['curriculum']),cfg)
    counts = Learner._saved_optimizer_counts(saved)
    assert counts==saved['optimizer_update_counts'] and counts['world']==saved['world_updates']
    assert counts['world']%4==0 and counts['actor']==counts['critic'] and counts['actor']<=counts['encoder']
    assert saved['training_phase']==Learner._phase_for_counts(counts,256)=='alternating'
    assert counts['actor']>=368 and counts['world']>=1728
    assert all(isinstance(value,torch.Tensor) and torch.isfinite(value).all()
        for name in ('actor','world','critic') for value in saved[name].values())
    optimizer_info = {}
    for name,prefix,network in (('actor','mlp.','actor'),('encoder','encoder.','actor'),
                                ('world','net.','world'),('critic','net.','critic')):
        optimizer = saved['opt_'+name]
        assert len(optimizer['param_groups'])==1
        group = optimizer['param_groups'][0]
        assert group['lr']==LRS[name]
        parameters = [(key,value) for key,value in saved[network].items() if key.startswith(prefix)]
        assert len(group['params'])==len(parameters) and set(optimizer['state'])==set(group['params'])
        for key,(parameter_name,parameter) in zip(group['params'],parameters):
            state = optimizer['state'][key]
            assert {'step','exp_avg','exp_avg_sq'}<=state.keys() and float(state['step'])==counts[name]
            for field in ('exp_avg','exp_avg_sq'):
                assert state[field].shape==parameter.shape and state[field].dtype==parameter.dtype
                assert torch.isfinite(state[field]).all()
        optimizer_info[name] = dict(actual_steps=counts[name],lr=group['lr'],parameter_tensors=len(parameters))
    norms = {}
    for name in ('world','critic'):
        for key,value in saved[name].items():
            if key.endswith('.count'):
                assert float(value)>0
                norms[name+'.'+key] = float(value)
    assert saved['world']['in_norm.mean'].numel()==141 and saved['world']['delta_norm.mean'].numel()==46
    assert saved['critic']['norm.mean'].numel()==324
    return dict(actual_optimizer_counts=counts,optimizers=optimizer_info,norm_counts=norms,
        phase=saved['training_phase'],iteration=saved['iteration'],transitions=saved['transitions'])


def source_manifest(run):
    manifest = json.loads((run/'source_sha256.json').read_text())
    frozen = json.loads((ART/'source-sha256.json').read_text())
    assert manifest==frozen and len(manifest)==101
    source = ROOT/'mjlab_training/src/ct_mjlab'
    snapshot = run/'snapshot/ct_mjlab'
    assert manifest=={str(path.relative_to(source)):sha(path) for path in source.rglob('*.py')}
    assert manifest=={str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*.py')}
    return dict(entries=len(manifest),manifest_sha256=sha(run/'source_sha256.json'),
        snapshot_current_and_frozen_manifest_identical=True)


def native_restore(path,saved):
    state = copy.deepcopy(saved['curriculum'])
    command = SimpleNamespace(cfg=command_cfg('walk'),checkpoint_state=lambda:copy.deepcopy(state),
        load_checkpoint_state=lambda value:state.update(copy.deepcopy(value)))
    learner = Learner('cpu',actor_lr=saved['config']['actor_lr'])
    calls = []
    def forbidden_forward(*_):
        calls.append('forward')
        raise AssertionError('A forward is forbidden during continuation restoration checks')
    handles = [getattr(learner,name).register_forward_pre_hook(forbidden_forward)
        for name in ('actor','world','critic')]
    try:
        restored = learner.load(path,command,expected_config={key:saved['config'][key] for key in RESUME_FIELDS})
        components = {name:equal(getattr(learner,name).state_dict(),saved[name]) for name in COMPONENTS}
        assert all(components.values()) and not calls
        assert equal(torch.get_rng_state(),saved['torch_rng'])
        assert equal(state,saved['curriculum'])
        assert learner.optimizer_update_counts()==saved['optimizer_update_counts']
        assert learner.training_phase(256)=='alternating'
        assert not torch.cuda.is_initialized()
    finally:
        for handle in handles:
            handle.remove()
    return dict(components_bitwise_restored=components,cpu_rng_restored=True,curriculum_restored=True,
        actual_optimizer_counts=learner.optimizer_update_counts(),phase=learner.training_phase(256),
        model_forwards=0,optimizer_updates=0,environments_created=0,cuda_initialized=False,
        cuda_rng_scope='CUDA restoration is structurally checked by Learner.load and prior formal checks; this CPU-only process does not restore a CUDA generator.')


def audit_source(path):
    budget = json.loads((ART/'budget.json').read_text())
    assert path==Path(budget['source_checkpoint']) and sha(path)==budget['source_checkpoint_sha256']
    saved = checkpoint(path)
    summary = configuration(saved)
    expected = budget.get('source_optimizer_counts',dict(world=1728,encoder=368,actor=368,critic=368))
    assert summary['actual_optimizer_counts']==expected
    assert expected['actor']==budget['source_actor_updates']
    old = ROOT/'artifacts/supermimic/supermimic-v13-alternating-20261031-r1'
    evidence = dict(training_exit=Path(budget.get('source_training_exit',old/'stage2-training-exit.json')),
        audit_exit=Path(budget.get('source_audit_exit',old/'stage2-audit-exit.json')),
        audit=Path(budget.get('source_audit',old/'stage2-audit.json')))
    train_exit = json.loads(evidence['training_exit'].read_text())
    audit_exit = json.loads(evidence['audit_exit'].read_text())
    audit = json.loads(evidence['audit'].read_text())
    assert train_exit['returncode']==audit_exit['returncode']==0 and audit['all_checks_passed']
    assert train_exit['final_sha256']==sha(path) and audit['checks']['completion_verified']
    assert Path(train_exit['log_dir'])==path.parent
    assert all(train_exit['last_metrics'][name+'_updates']==count for name,count in expected.items())
    if 'final_summary' in audit['checks']:
        assert audit['checks']['final_summary']['actual_optimizer_counts']==expected
    return dict(source_checkpoint=str(path),source_sha256=sha(path),summary=summary,
        source_manifest=source_manifest(path.parent),native_restore=native_restore(path,saved),
        source_completion_evidence={key:dict(path=str(value),sha256=sha(value)) for key,value in evidence.items()},
        full_resume_required=True,no_reinitialization_or_repeated_warmup=True)


def lineage(initial,parent_path):
    budget = json.loads((ART/'budget.json').read_text())
    cursor = initial['initialization']
    assert cursor['kind']=='resume' and Path(cursor['parent_checkpoint'])==parent_path
    points = []
    found_origin = False
    while cursor.get('kind')=='resume':
        path = Path(cursor['parent_checkpoint'])
        assert path.is_file() and sha(path)==cursor['parent_sha256']
        assert cursor['parent_algorithm_contract']==module.ALGORITHM_CONTRACT
        points.append(dict(path=str(path),sha256=cursor['parent_sha256']))
        found_origin |= (path==Path(budget['source_checkpoint'])
            and cursor['parent_sha256']==budget['source_checkpoint_sha256'])
        cursor = cursor['inherited_initialization']
        assert len(points)<=8
    assert found_origin and cursor['kind']=='actor_and_world_transfer_fresh_optimizers'
    assert cursor['actor_transfer']['parent_sha256']=='50ee90a69e6f58344898277a77ff850d2504dbda31dd3ff4b8beb6473ef5105f'
    assert cursor['world_transfer']['parent_sha256']=='3a06b3ece5d284d658cdff7b95ddf932fb81f40b26e97e59073e9fb7afe09646'
    return points


def audit_run(run,exit_record):
    budget = json.loads((ART/'budget.json').read_text())
    count_only = budget.get('termination_mode')=='actor_update_count_only'
    config = json.loads((run/'config.json').read_text())
    initial,final = checkpoint(run/'initial.pt'),checkpoint(run/'final.pt')
    assert json_form(initial['config'])==json_form(final['config'])==config
    parent_path = Path(config['resume']).resolve()
    parent = checkpoint(parent_path)
    parent_summary,initial_summary,final_summary = [configuration(value) for value in (parent,initial,final)]
    assert all(equal(initial[name],parent[name]) for name in COMPONENTS)
    assert all(equal(initial[key],parent[key]) for key in ('iteration','transitions','world_updates',
        'optimizer_update_counts','training_phase','infos','asset_sha256','algorithm_contract'))
    assert all(initial['config'][key]==parent['config'][key] for key in RESUME_FIELDS)
    assert initial['initialization']['inherited_initialization']==parent['initialization']
    assert not (run/'prewarm.pt').exists() and not (run/'postwarm.pt').exists()
    receipt = json.loads(exit_record.read_text())
    assert receipt['returncode']==0 and receipt['final_checkpoint_exists']
    assert Path(receipt['log_dir']).resolve()==run and receipt['final_sha256']==sha(run/'final.pt')
    limit = receipt['stop_actor_updates']
    assert type(limit) is int and limit in budget.get('milestone_actor_updates',(700,1100,1500,2200))
    if count_only:
        assert config['termination_mode']=='actor_update_count_only' and config['budget_kind']=='actual_actor_updates'
        assert limit==budget['maximum_actor_updates']==config['stop_actor_updates']
    assert math.isfinite(receipt['wall_elapsed_s']) and receipt['wall_elapsed_s']>=0
    metrics_path = run/'metrics.jsonl'
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    assert rows and rows[-1]==receipt['last_metrics']
    last = rows[-1]
    for name,count in final_summary['actual_optimizer_counts'].items():
        assert last[name+'_updates']==count and count>=initial_summary['actual_optimizer_counts'][name]
    assert final_summary['actual_optimizer_counts']['actor']<=limit
    assert last['iteration']==final['iteration'] and last['transitions']==final['transitions']
    assert last['training_phase']==final['training_phase']=='alternating'
    assert math.isfinite(last['elapsed_s']) and last['elapsed_s']>=0
    if not count_only:
        assert last['elapsed_s']<=config['seconds']+10.
    points = []
    previous = initial_summary['actual_optimizer_counts']
    paths = [run/'initial.pt',*sorted(run.glob('checkpoint_*.pt')),run/'final.pt']
    for path in paths:
        value = initial if path.name=='initial.pt' else final if path.name=='final.pt' else checkpoint(path)
        summary = configuration(value)
        actual = summary['actual_optimizer_counts']
        assert all(actual[name]>=previous[name] for name in COUNTERS)
        assert actual['actor']<=limit
        points.append(dict(path=str(path),sha256=sha(path),**summary))
        previous=actual
    milestones = []
    if count_only:
        assert final_summary['actual_optimizer_counts']['actor']==limit
        assert final_summary['actual_optimizer_counts']['actor']-initial_summary['actual_optimizer_counts']['actor']==budget['additional_actor_updates']
        requested = budget['save_during_training_actor_updates']
        assert config['save_actor_updates']==requested
        result_path = ART/('A'+str(limit)+'-training-result.json')
        result = json.loads(result_path.read_text())
        assert result['termination_mode']=='actor_update_count_only' and result['budget_kind']=='actual_actor_updates'
        assert result['stop_reason']=='actor_update_limit' and result['target_reached_exactly']
        assert result['final_sha256']==sha(run/'final.pt')
        assert result['final_optimizer_counts']==final_summary['actual_optimizer_counts']
        assert result['last_metrics']==last and result['actual_training_loop_s']==last['elapsed_s']
        records = result['saved_actor_milestones']
        assert [record['threshold_actor_updates'] for record in records]==requested
        for record,threshold in zip(records,requested):
            path = run/('actor_'+str(threshold).zfill(6)+'.pt')
            assert Path(record['path']).resolve()==path and sha(path)==record['sha256']
            value = checkpoint(path)
            summary = configuration(value)
            actual = summary['actual_optimizer_counts']
            assert threshold<=actual['actor']<=limit
            assert actual==record['optimizer_update_counts'] and actual['actor']==record['actual_actor_updates']
            assert value['iteration']==record['iteration'] and value['transitions']==record['transitions']
            assert json_form(value['config'])==config
            assert all(initial_summary['actual_optimizer_counts'][name]<=actual[name]
                <=final_summary['actual_optimizer_counts'][name] for name in COUNTERS)
            earlier = [point for point in points if Path(point['path']).stat().st_mtime_ns<path.stat().st_mtime_ns]
            assert all(point['actual_optimizer_counts']['actor']<threshold for point in earlier)
            milestones.append(dict(threshold_actor_updates=threshold,path=str(path),sha256=record['sha256'],
                **summary,first_published_checkpoint_at_or_above_threshold=True))
    manifest = source_manifest(run)
    assert source_manifest(parent_path.parent)['entries']==manifest['entries']
    chain = lineage(initial,parent_path)
    restore = native_restore(run/'final.pt',final)
    return dict(completion_verified=True,run=str(run),parent_checkpoint=str(parent_path),parent_sha256=sha(parent_path),
        initial_all_seven_components_bitwise_equal_parent=True,initial_metadata_and_counters_equal_parent=True,
        initial_summary=initial_summary,final_summary=final_summary,
        additional_optimizer_updates={name:final_summary['actual_optimizer_counts'][name]
            -initial_summary['actual_optimizer_counts'][name] for name in COUNTERS},
        new_transitions=final['transitions']-initial['transitions'],no_repeated_warmup_snapshots=True,
        source_manifest=manifest,source_checkpoint_hash_chain=chain,saved_points=points,
        stop_actor_updates=limit,actual_actor_updates=final_summary['actual_optimizer_counts']['actor'],
        exact_actor_target_reached=final_summary['actual_optimizer_counts']['actor']==limit,
        actual_training_loop_elapsed_s=last['elapsed_s'],config_seconds=config['seconds'],
        termination_mode='actor_update_count_only' if count_only else 'bounded_wall_time_and_actor_count',
        user_wall_time_limit_s=None if count_only else budget.get('maximum_training_loop_s'),
        saved_actor_milestones=milestones,
        prior_training_wall_s=receipt.get('training_wall_used_before_s'),native_final_restore=restore,
        last_metrics_and_final_actual_adam_counts_equal=True,
        rng_scope='Native CPU restore is checked. Run initial RNG may differ from parent because the unchanged main calls env.reset and rebuilds replay after loading.')


def main():
    global ART
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,default=ART)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source',type=Path)
    group.add_argument('--run',type=Path)
    parser.add_argument('--exit-record',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    ART = args.experiment_dir.resolve()
    if bool(args.run)!=bool(args.exit_record):
        parser.error('--run and --exit-record must be provided together')
    torch.set_num_threads(1)
    result = audit_source(args.source.resolve()) if args.source else audit_run(args.run.resolve(),args.exit_record.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(all_checks_passed=True,checker_sha256=sha(__file__),checks=result),stream,
            ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(all_checks_passed=True,output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    main()
