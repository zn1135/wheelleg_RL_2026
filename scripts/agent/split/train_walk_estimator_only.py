#!/usr/bin/env python3
"""Budgeted offline supervision of the original encoder; never runs PPO or physics."""

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.modules import MLP

from scripts.agent.split import walk_estimator_offline as offline


PARENT_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
RECIPE = dict(seed=42, optimizer='Adam', new_optimizer_state=True, learning_rate=1e-4,
    betas=[.9, .999], eps=1e-8, weight_decay=0., batch_size=2048,
    gradient_norm_clip=.1, maximum_optimizer_updates=1000,
    milestones=[0, 250, 500, 750, 1000], only_candidate_update=1000,
    loss='mean((encoder(history125)-same_tick_COM_body_xyz*2)**2)',
    architecture=[125, 128, 64, 3], activation='elu',
    sampler='uniform six phases; then uniform eligible run/episode/segment/frame, with replacement',
    early_best_selection=False, performance_early_stop=False,
    control_changes=False, PPO_updates=0, noise_or_delay_augmentation=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(value):
    """Hash tensor trees independently of torch.save container timestamps."""
    result = hashlib.sha256()

    def update(item):
        result.update(type(item).__name__.encode()+b'\0')
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            result.update(str(array.dtype).encode())
            result.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            result.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str(item.dtype).encode())
            result.update(np.asarray(item.shape, dtype=np.int64).tobytes())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key:(type(key).__name__, repr(key))):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                update(child)
        else:
            result.update(repr(item).encode())
    update(value)
    return result.hexdigest()


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def frozen_parts(saved):
    return dict(
        actor_control_and_distribution={key:value for key,value in saved['actor_state_dict'].items()
                                        if not key.startswith('encoder.')},
        every_other_parent_field={key:value for key,value in saved.items() if key != 'actor_state_dict'})


def verify_frozen(saved, expected):
    observed = {key:tree_digest(value) for key,value in frozen_parts(saved).items()}
    assert observed == expected, 'Non-encoder parent state changed'
    return observed


def cpu_state(module):
    return {key:value.detach().cpu().clone() for key,value in module.state_dict().items()}


def finite_module(module):
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


@torch.inference_mode()
def predictions(encoder, split, device):
    encoder.eval()
    result = {}
    for run in split.runs:
        flat = run.history.reshape(-1, 125)
        predicted = np.empty((len(flat), 3), dtype=np.float32)
        for begin in range(0, len(flat), 8192):
            end = min(begin+8192, len(flat))
            batch = torch.from_numpy(np.array(flat[begin:end], copy=True)).to(device)
            predicted[begin:end] = (encoder(batch)/2.).cpu().numpy()
        assert np.isfinite(predicted).all(), f'Nonfinite predictions: {run.id}'
        result[run.id] = predicted.reshape(*run.history.shape[:2], 3)
    return result


def rng_state(rng):
    return dict(sample_numpy_bit_generator=copy.deepcopy(rng.bit_generator.state),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def sampling_counts(split, updates=None):
    counts = copy.deepcopy(split.sampling_counts)
    total = counts['total']
    assert total == sum(counts['buckets'].values()) == sum(counts['runs'].values())
    assert total == sum(sum(run_counts.values()) for run_counts in counts['bucket_runs'].values())
    for bucket, value in counts['buckets'].items():
        assert value == sum(counts['bucket_runs'][bucket].values())
    for run, value in counts['runs'].items():
        assert value == sum(run_counts[run] for run_counts in counts['bucket_runs'].values())
    if updates is not None:
        assert total == 2048*updates, 'Actual draws differ from successful-update budget'
    return counts


def save_checkpoint(output, step, encoder, optimizer, parent, frozen_hashes, rng, lock, train_split):
    verify_frozen(parent, frozen_hashes)
    assert finite_module(encoder), 'Nonfinite encoder parameter'
    state = cpu_state(encoder)
    paired = copy.deepcopy(parent)
    for key,value in state.items():
        paired['actor_state_dict']['encoder.'+key] = value
    # The paired model retains every parent field; offline metadata/Adam/RNG live beside it.
    verify_frozen(paired, frozen_hashes)
    path = output/f'paired-model-offline-update-{step:04d}.pt'
    assert not path.exists()
    torch.save(paired, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    verify_frozen(restored, frozen_hashes)
    assert tree_digest({key.removeprefix('encoder.'):value for key,value in
        restored['actor_state_dict'].items() if key.startswith('encoder.')}) == tree_digest(state)
    training_state = dict(actual_updates=step, parent_sha256=PARENT_SHA,
        encoder_state_dict=state, new_Adam_state_dict=copy.deepcopy(optimizer.state_dict()),
        sampling_and_torch_rng_state=rng_state(rng), recipe=RECIPE,
        training_lock_sha256=lock, paired_model_sha256=digest(path),
        sampling_counts=sampling_counts(train_split, step),
        only_terminal_candidate=step == 1000, original_PPO_iteration=2999,
        scope='Offline estimator state; not a resumed PPO iteration.')
    state_path = output/f'offline-training-state-update-{step:04d}.pt'
    assert not state_path.exists()
    torch.save(training_state, state_path)
    receipt = dict(update=step, path=str(path.resolve()), sha256=digest(path),
        encoder_sha256=tree_digest(state), frozen_parts_sha256=frozen_hashes,
        training_state_path=str(state_path.resolve()), training_state_sha256=digest(state_path),
        optimizer_state_sha256=tree_digest(training_state['new_Adam_state_dict']),
        sample_rng_state_sha256=tree_digest(training_state['sampling_and_torch_rng_state']),
        sampling_counts=training_state['sampling_counts'],
        original_PPO_iteration=restored['iter'], frozen_check_passed=True)
    write(output/f'checkpoint-{step:04d}.json', receipt)
    return receipt


def evaluate(encoder, splits, step, output, writer, parent_scores=None):
    scores, inferred = {}, {}
    for name, split in splits.items():
        inferred[name] = predictions(encoder, split, next(encoder.parameters()).device)
        if step == 0:
            parity = split.verify_parent_predictions(inferred[name], tolerance=1e-5)
            write(output/f'{name}-parent-replay.json', parity)
        scores[name] = offline.score_predictions(split, inferred[name],
            baseline=None if parent_scores is None else parent_scores[name])
        write(output/f'{name}-metrics-{step:04d}.json', scores[name])
        if writer is not None:
            for tag,value in offline.log_scalars(scores[name]).items():
                assert np.isfinite(value), f'Nonfinite metric scalar: {name}/{tag}'
                writer.add_scalar(f'{name}/{tag}', float(value), step)
        print(name, 'phase/axis evaluation saved at update', step, flush=True)
    if writer is not None:
        writer.flush()
    return scores, inferred


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    parent_path, data_card, execution_card = (path.resolve() for path in
        (args.parent, args.data_card, args.execution_card))
    completed, attempted = 0, 0
    encoder, optimizer, writer, rng, parent, journal = None, None, None, None, None, None
    splits, committed_sampling_counts = None, None
    try:
        authorization = json.loads(execution_card.read_text())
        assert authorization['authorized_encoder_updates'] == 1000, 'Explicit1000-update execution card required'
        assert authorization['parent_checkpoint_sha256'] == PARENT_SHA
        gate_sha = offline.validate_gate_card(authorization)
        assert digest(parent_path) == PARENT_SHA
        torch.set_num_threads(1)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        rng = np.random.default_rng(42)
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        assert parent['iter'] == 2999
        assert parent['infos']['experiment_profile_contract'] == 'chuanliantui-walk-physical25-v8'
        frozen_hashes = {key:tree_digest(value) for key,value in frozen_parts(parent).items()}
        state = {key.removeprefix('encoder.'):value for key,value in parent['actor_state_dict'].items()
                 if key.startswith('encoder.')}
        encoder = MLP(125, 3, (128, 64), 'elu').to(args.device)
        encoder.load_state_dict(state, strict=True)
        assert sum(parameter.numel() for parameter in encoder.parameters()) == 24579
        assert tree_digest(cpu_state(encoder)) == tree_digest(state)
        assert finite_module(encoder)
        encoder.requires_grad_(True)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4,
            betas=(.9, .999), eps=1e-8, weight_decay=0.)
        assert not optimizer.state, 'Offline Adam must start with new state'
        splits = offline.load_datasets(data_card)
        assert set(splits) == {'train', 'dev'}
        data_lock = {name:split.lock() for name,split in splits.items()}
        committed_sampling_counts = sampling_counts(splits['train'], 0)
        for split in splits.values():
            split.verify_files()
        sources = {str(Path(path).resolve()):digest(path) for path in
            (__file__, offline.__file__, inspect.getfile(MLP), data_card, execution_card)}
        helper = Path(__file__).resolve().with_name('analyze_walk_velocity_compensation.py')
        sources[str(helper)] = digest(helper)
        lock = dict(check_only=args.check_only, optimizer_updates_before_lock=0,
            recipe=RECIPE, parent_path=str(parent_path), parent_sha256=PARENT_SHA,
            offline_gate_sha256=gate_sha,
            parent_encoder_sha256=tree_digest(state), frozen_parts_sha256=frozen_hashes,
            sources_sha256=sources, datasets=data_lock, execution_authorization=authorization,
            versions=dict(torch=torch.__version__, numpy=np.__version__),
            only_trainable_component='Original single three-axis encoder',
            target_used_only_for_supervision_and_scoring=True, labels_shifted=False)
        write(args.output/'training-lock.json', lock)
        lock_sha = digest(args.output/'training-lock.json')
        shutil.copy2(__file__, args.output/'trainer-source.py')
        shutil.copy2(offline.__file__, args.output/'offline-library-source.py')
        baseline_scores, _ = evaluate(encoder, splits, 0, args.output, None)
        assert tree_digest(cpu_state(encoder)) == tree_digest(state), 'Parent replay changed encoder state'
        receipts = [save_checkpoint(args.output, 0, encoder, optimizer, parent, frozen_hashes, rng,
            lock_sha, splits['train'])]
        if args.check_only:
            assert all(digest(path) == sha for path,sha in sources.items())
            assert digest(parent_path) == PARENT_SHA
            verify_frozen(parent, frozen_hashes)
            for split in splits.values():
                split.verify_files()
            write(args.output/'result.json', dict(check_only=True, verified=True, optimizer_updates=0,
                actual_updates=0,
                parent_replay_passed=True, parent_and_source_sha256_locked=True,
                frozen_parts_sha256=frozen_hashes, checkpoints=receipts, training=False,
                sampling_counts=sampling_counts(splits['train'], 0),
                scope='Data/interface/source/frozen-state checks only; no optimizer step or behavior verdict.'))
            print('check-only complete; optimizer updates0', flush=True)
            return
        # A single execution card cannot grant another1000 steps through a new output path.
        claim_path = execution_card.parent/'training-execution-claim.json'
        claim = dict(execution_card=str(execution_card), execution_card_sha256=digest(execution_card),
            parent_sha256=PARENT_SHA, authorized_encoder_updates=1000,
            output=str(args.output), journal=str(args.output/'optimizer-steps.jsonl'),
            training_lock_sha256=lock_sha, actual_updates_at_claim=0,
            automatic_resume=False, automatic_retry=False)
        write(claim_path, claim)
        log_dir = args.tensorboard_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=False)
        writer = SummaryWriter(str(log_dir))
        journal = (args.output/'optimizer-steps.jsonl').open('x', buffering=1)
        for name, score in baseline_scores.items():
            for tag,value in offline.log_scalars(score).items():
                assert np.isfinite(value)
                writer.add_scalar(f'{name}/{tag}', float(value), 0)
        writer.flush()
        terminal_scores = None
        for update in range(1, 1001):
            encoder.train()
            history, target = offline.sample_batch(splits['train'], rng, 2048)
            assert history.shape == (2048, 125) and target.shape == (2048, 3)
            assert np.isfinite(history).all() and np.isfinite(target).all()
            history_t = torch.as_tensor(history, dtype=torch.float32, device=args.device)
            target_t = torch.as_tensor(target, dtype=torch.float32, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = (encoder(history_t)-target_t).square().mean()
            assert bool(torch.isfinite(loss)), 'Nonfinite supervised loss'
            loss.backward()
            assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                       for parameter in encoder.parameters()), 'Missing/nonfinite encoder gradient'
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), .1)
            assert bool(torch.isfinite(grad_norm)), 'Nonfinite encoder gradient norm'
            attempted = update
            optimizer.step()
            completed = update
            journal.write(json.dumps(dict(event='optimizer_step_succeeded', actual_updates=completed,
                attempted_updates=attempted, actual_sample_draws=splits['train'].sampling_counts['total'],
                scaled_xyz_MSE=float(loss.detach())), allow_nan=False)+'\n')
            journal.flush()
            os.fsync(journal.fileno())
            committed_sampling_counts = sampling_counts(splits['train'], completed)
            assert finite_module(encoder), 'Nonfinite parameter after optimizer step'
            verify_frozen(parent, frozen_hashes)
            steps = {int(value['step']) for value in optimizer.state.values()}
            assert steps == {update} and len(optimizer.state) == len(list(encoder.parameters()))
            writer.add_scalar('optimization/scaled_xyz_MSE', float(loss.detach()), update)
            writer.add_scalar('optimization/gradient_norm_before_clip', float(grad_norm), update)
            writer.add_scalar('optimization/learning_rate', optimizer.param_groups[0]['lr'], update)
            if update % 50 == 0:
                print('encoder update', update, '/1000; scaledxyz MSE', float(loss.detach()), flush=True)
                writer.flush()
            if update in (250, 500, 750, 1000):
                for split in splits.values():
                    split.verify_files()
                assert all(digest(path) == sha for path,sha in sources.items()), 'Source/card changed'
                assert digest(parent_path) == PARENT_SHA
                receipts.append(save_checkpoint(args.output, update, encoder, optimizer,
                    parent, frozen_hashes, rng, lock_sha, splits['train']))
                scores, _ = evaluate(encoder, splits, update, args.output, writer, baseline_scores)
                if update == 1000:
                    terminal_scores = scores
        assert completed == 1000 and terminal_scores is not None
        development_gate = offline.candidate_gate(terminal_scores['dev'], baseline_scores['dev'])
        write(args.output/'terminal-development-gate.json', development_gate)
        assert all(digest(path) == sha for path,sha in sources.items())
        assert digest(parent_path) == PARENT_SHA
        verify_frozen(parent, frozen_hashes)
        for split in splits.values():
            split.verify_files()
        result = dict(verified=True, check_only=False, optimizer_updates=1000,
            actual_updates=completed,
            sample_draws=2048000, budget_reached=True, automatic_extension=False,
            sampling_counts=sampling_counts(splits['train'], completed),
            execution_claim=str(claim_path), execution_claim_sha256=digest(claim_path),
            single_candidate_update=1000, checkpoints=receipts, parent_sha256=PARENT_SHA,
            candidate=receipts[-1], frozen_parts_sha256=frozen_hashes,
            development_gate=development_gate, tensorboard_dir=str(log_dir),
            training_lock_sha256=lock_sha, PPO_updates=0, production_changes=False,
            behavior_verified=False, height='Backward low height remains unresolved.',
            scope='Encoder supervision and offline development screening only; no closed-loop or final-test claim.')
        write(args.output/'result.json', result)
        print(json.dumps(dict(optimizer_updates=1000, budget_reached=True,
            candidate_path=receipts[-1]['path'], candidate_sha256=receipts[-1]['sha256'],
            development_gate=development_gate.get('passed', development_gate.get('support'))), ensure_ascii=False), flush=True)
    except BaseException as exc:
        actual_sampling_counts = None if splits is None else copy.deepcopy(splits['train'].sampling_counts)
        if encoder is not None and optimizer is not None and rng is not None:
            failure_path = args.output/'failure-training-state.pt'
            if not failure_path.exists():
                torch.save(dict(optimizer_updates_completed=completed, encoder=cpu_state(encoder),
                    actual_updates=completed, attempted_updates=attempted,
                    sampling_counts=actual_sampling_counts,
                    last_committed_sampling_counts=committed_sampling_counts,
                    optimizer=optimizer.state_dict(), rng=rng_state(rng)), failure_path)
        failure = dict(completed=False, optimizer_updates_completed=completed,
            actual_updates=completed, attempted_updates=attempted,
            sampling_counts=actual_sampling_counts,
            last_committed_sampling_counts=committed_sampling_counts,
            expected_draws_for_successful_updates=2048*completed,
            maximum_authorized_updates=1000, exception_type=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc(), automatic_retry=False)
        path = args.output/'failure.json'
        if not path.exists():
            write(path, failure)
        raise
    finally:
        if journal is not None:
            journal.flush()
            journal.close()
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--data-card', type=Path, required=True)
    parser.add_argument('--execution-card', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensorboard-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--check-only', action='store_true')
    run(parser.parse_args())
