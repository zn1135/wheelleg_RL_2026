"""Audit an iteration-budget run, parameter identity and TensorBoard evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ct_mjlab.envs.chuanliantui.fitted_leg_load import parameter_record


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    budget = json.loads((args.experiment / 'budget.json').read_text())
    process = json.loads((args.experiment / 'train-execution.json').read_text())
    report = json.loads((args.run / 'training_report.json').read_text())
    additional = budget.get('train_iterations', budget.get('additional_iterations'))
    start, n = budget.get('start_iteration', 0), budget['num_envs']
    target = start + additional
    initial = torch.load(args.run / 'model_initial.pt', map_location='cpu', weights_only=False)
    final = torch.load(args.run / 'model_final.pt', map_location='cpu', weights_only=False)
    assert process['exit_code'] == 0
    assert report['iterations'] == target
    assert report['iterations_added'] == report['iteration_budget'] == additional
    assert report['initial_iterations'] == start and report['stopped_by'] == 'iteration_budget'
    assert report['transitions'] == initial['transitions'] + additional * n * 48
    assert report['primary_updates'] == report['encoder_updates'] == target * 20
    assert not report['partial_update'] and not final['partial_update']
    if start == 0:
        assert final['initialization'] == initial['initialization'] == 'random_from_scratch_seed_1'
        assert final['resume_from'] is initial['resume_from'] is None
        assert initial['transitions'] == 0
        assert not initial['optimizer_state_dict']['state'] and not initial['encoder_optimizer_state_dict']['state']
    else:
        assert final['initialization'] == initial['initialization'] == 'resumed_learning_state_fresh_environment'
        source = Path(budget['resume_checkpoint'])
        assert final['resume_from'] == initial['resume_from']
        assert initial['resume_from']['sha256'] == sha(source) == budget['resume_checkpoint_sha256']
        assert initial['resume_from']['iterations'] == start
        # 完整模型/两优化器/RNG逐值恢复另由audit_resume_run核对。
        assert initial['optimizer_state_dict']['state'] and initial['encoder_optimizer_state_dict']['state']
    assert initial['iterations'] == start
    assert initial['primary_updates'] == initial['encoder_updates'] == start * 20
    assert final['iterations'] == target and final['transitions'] == report['transitions']
    assert final['num_envs'] == initial['num_envs'] == n
    for key in ('reset_contract', 'config', 'disturbances', 'command_sampling', 'physical_leg_load'):
        assert initial[key] == final[key], key
    for optimizer in ('optimizer_state_dict', 'encoder_optimizer_state_dict'):
        steps = {int(state['step']) for state in final[optimizer]['state'].values()}
        assert steps == {target * 20}, (optimizer, steps)
    assert all(torch.isfinite(value).all() for value in final['model_state_dict'].values())
    assert any(not torch.equal(value, initial['model_state_dict'][name])
               for name, value in final['model_state_dict'].items())
    assert report['checkpoint_sha256'] == sha(args.run / 'model_final.pt')
    frozen = args.run / 'source/resources/robots/chuanliantui_new_1/mjlab/fitted_leg_load.json'
    assert final['physical_leg_load'] == parameter_record() == process['physical_leg_load']
    assert sha(frozen) == final['physical_leg_load']['sha256'] == budget['parameters_sha256']
    manifest = json.loads((args.run / 'source_sha256.json').read_text())
    prefix = 'mjlab_training/src/ct_mjlab/'
    source_checked = 0
    for name, digest in manifest.items():
        assert sha(Path(name)) == digest, 'Live source changed during training: ' + name
        if name.startswith(prefix):
            assert sha(args.run / 'source/ct_mjlab' / name.removeprefix(prefix)) == digest, name
            source_checked += 1
    checkpoints = []
    for path in sorted(args.run.glob('model_*.pt')):
        suffix = path.stem.removeprefix('model_')
        if suffix.isdigit():
            assert int(suffix) <= target
            checkpoints.append(dict(path=str(path), iteration=int(suffix), sha256=sha(path)))
    assert any(row['iteration'] == target for row in checkpoints)
    events = EventAccumulator(str(args.run), size_guidance={'scalars': 0}).Reload()
    rewards = json.loads((args.experiment / 'provenance/effective-config.json').read_text())['env']['rewards']['scales']
    scalar_tags = ['Train/reward'] + ['Rewards/' + name for name in rewards]
    expected_steps = list(range(start + 1, target + 1))
    values = {}
    for tag in scalar_tags:
        rows = events.Scalars(tag)
        assert [row.step for row in rows] == expected_steps, tag
        assert all(torch.isfinite(torch.tensor(row.value)) for row in rows), tag
        values[tag] = [row.value for row in rows]
    sum_error = max(abs(sum(values['Rewards/' + name][i] for name in rewards) - values['Train/reward'][i])
                    for i in range(additional))
    assert sum_error < 1e-6
    result = dict(passed=True, iterations=target, initial_iterations=start, additional_iterations=additional,
                  transitions=report['transitions'],
                  primary_updates=target * 20, encoder_updates=target * 20,
                  physical_leg_load=final['physical_leg_load'], source_files_verified=source_checked,
                  parameter_snapshot_sha256=sha(frozen), final_checkpoint_sha256=sha(args.run / 'model_final.pt'),
                  initial_checkpoint_sha256=sha(args.run / 'model_initial.pt'),
                  tensorboard_reward_tags=len(scalar_tags), tensorboard_steps=additional,
                  reward_sum_max_error=sum_error, checkpoints=checkpoints, process_exit_code=process['exit_code'],
                  scope='Counts, initialization or resume identity, unchanged source/parameter identity, finite weights and reward logs; not motion acceptance')
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoints'}, indent=2))


if __name__ == '__main__':
    main()
