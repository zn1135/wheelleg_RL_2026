"""Read completed continuation checkpoints and reward events; no inference or physics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .check_resume import equal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    budget = json.loads((args.experiment/'budget.json').read_text())
    process = json.loads((args.experiment/'process-result.json').read_text())
    run = Path(budget['log_dir'])
    source = Path(budget['resume_checkpoint'])
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert process['returncode'] == 0 and sha(source) == budget['resume_checkpoint_sha256']
    parent = torch.load(source,map_location='cpu',weights_only=False)
    start = parent['iterations']
    additional = budget['additional_iterations']
    target = start + additional
    num_envs = budget['num_envs']
    new_transitions = additional * num_envs * 48
    new_updates = additional * 20
    total_primary = parent['primary_updates'] + new_updates
    total_encoder = parent['encoder_updates'] + new_updates
    source_steps = parent.get('policy_steps', parent['transitions']//budget['source_num_envs']+1)
    assert budget['start_iteration'] == start and budget['target_iteration'] == target
    initial = torch.load(run/'model_initial.pt',map_location='cpu',weights_only=False)
    final = torch.load(run/'model_final.pt',map_location='cpu',weights_only=False)
    report = json.loads((run/'training_report.json').read_text())
    assert initial['iterations'] == start and initial['primary_updates'] == parent['primary_updates']
    assert initial['encoder_updates'] == parent['encoder_updates']
    for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng'):
        equal(initial[key],parent[key])
    assert initial['num_envs'] == report['num_envs'] == num_envs and initial['policy_steps'] == source_steps
    assert report['stopped_by'] == 'iteration_budget' and report['iteration_budget'] == additional
    assert report['initial_iterations'] == start and report['iterations_added'] == additional
    assert report['iterations'] == report['target_iteration'] == final['iterations'] == target
    assert report['transitions_added'] == new_transitions
    assert report['transitions'] == final['transitions'] == parent['transitions'] + new_transitions
    assert report['primary_updates_added'] == report['encoder_updates_added'] == new_updates
    assert report['primary_updates'] == final['primary_updates'] == total_primary
    assert report['encoder_updates'] == final['encoder_updates'] == total_encoder
    assert final['policy_steps'] == source_steps + 1 + additional*48
    assert not final['partial_update'] and not report['partial_update']
    assert report['checkpoint_sha256'] == sha(run/'model_final.pt')
    restore=json.loads((run/'restore_report.json').read_text())
    if 'expected_reset_contract' in budget:
        target_reset=budget['expected_reset_contract']
        assert initial['reset_contract']==final['reset_contract']==report['reset_contract']==target_reset
        assert restore['source_reset_contract']==restore['target_reset_contract']==target_reset
        assert restore['history_migration'] is None
        assert initial['command_sampling']==final['command_sampling']==report['command_sampling']==budget['command_sampling']
        assert initial['disturbances']==final['disturbances']==report['disturbances']==budget['disturbances']
    for key, count in [('optimizer_state_dict',total_primary),('encoder_optimizer_state_dict',total_encoder)]:
        assert {int(state['step']) for state in final[key]['state'].values()} == {count}
    assert all(torch.isfinite(value).all() for value in final['model_state_dict'].values())
    assert (final['model_state_dict']['std'] > 0).all()
    assert any(not torch.equal(value,initial['model_state_dict'][key]) for key,value in final['model_state_dict'].items())
    events = EventAccumulator(str(run),size_guidance={'scalars':0}).Reload()
    tags = sorted(tag for tag in events.Tags()['scalars'] if tag.startswith('Rewards/'))
    assert len(tags) == 16
    series = {tag:events.Scalars(tag) for tag in tags}
    expected = list(range(start+1,target+1))
    for values in series.values():
        assert [value.step for value in values] == expected
        assert all(math.isfinite(value.value) for value in values)
    totals = events.Scalars('Train/reward')
    assert [value.step for value in totals] == expected
    maximum_sum_error = max(abs(value.value-sum(series[tag][i].value for tag in tags)) for i,value in enumerate(totals))
    assert maximum_sum_error < 1e-6
    checkpoints = []
    for iteration in range((start//100+1)*100,target+1,100):
        path = run/f'model_{iteration}.pt'
        saved = torch.load(path,map_location='cpu',weights_only=False)
        assert saved['iterations'] == iteration
        assert saved['primary_updates'] == saved['encoder_updates'] == iteration*20
        checkpoints.append(dict(iteration=iteration,sha256=sha(path)))
    source_manifest=json.loads((run/'source_sha256.json').read_text());verified_source=0
    prefix='mjlab_training/src/ct_mjlab/'
    for name,digest in source_manifest.items():
        if name.startswith(prefix):
            assert sha(run/'source/ct_mjlab'/name.removeprefix(prefix))==digest,name
            verified_source+=1
    if 'source_manifest' in budget:
        assert source_manifest==json.loads(Path(budget['source_manifest']).read_text())
    result = dict(passed=True,physics_steps=0,model_forwards=0,optimizer_updates=0,
        num_envs=num_envs,initial_iteration=start,additional_iterations=additional,final_iteration=target,
        additional_transitions=new_transitions,final_transitions=final['transitions'],
        additional_updates_each=new_updates,final_updates_each=total_primary,
        initial_learner_and_rng_identical_to_source=True,full_physics_state_restored=False,
        reward_tags=tags,logged_steps=[start+1,target],maximum_reward_sum_error=maximum_sum_error,
        model_sha256=sha(run/'model_final.pt'),checkpoints=checkpoints,training=report,
        source_files_verified=verified_source,restore=restore,
        behavior_accepted=False,cpu_launch_resolved=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('checkpoints','training')},indent=2))


if __name__ == '__main__':
    main()
