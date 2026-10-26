#!/usr/bin/env python3
"""Assess a saved SMP prior and its training log; never updates model parameters."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.features import WINDOW
from ct_mjlab.smp.prior import MotionPrior, load_prior


def assess(args):
    prior,saved = load_prior(args.checkpoint,args.device)
    prior.freeze()
    data = MotionDataset(args.dataset)
    if saved['dataset']['fingerprint'] != data.fingerprint:
        raise ValueError('Checkpoint dataset or split disagrees with evaluation inputs')
    iteration = saved['iteration']
    if args.expected_updates and iteration != args.expected_updates:
        raise ValueError('Checkpoint iteration differs from authorized first-run budget')
    assert all(torch.isfinite(value).all() for value in prior.state_dict().values())
    assert (prior.feature_std >= .2).all() and int(prior.ema_step) == iteration
    optimizer_steps = sorted({int(state['step']) for state in saved['optimizer_state_dict']['state'].values()})
    assert optimizer_steps == [iteration], optimizer_steps
    events = EventAccumulator(str(args.checkpoint.parent),size_guidance={'scalars':0}).Reload()
    train = events.Scalars('SMP/train_l1')
    steps = [event.step for event in train]
    assert steps == list(range(1,iteration+1)), 'Training logs are incomplete or exceed first-run budget'
    values = np.asarray([event.value for event in train])
    assert np.isfinite(values).all()

    training_args = json.loads((args.checkpoint.parent/'arguments.json').read_text())
    torch.manual_seed(training_args['seed'])
    initial = MotionPrior(prior.config).to(args.device).freeze()
    initial.feature_mean.copy_(prior.feature_mean)
    initial.feature_std.copy_(prior.feature_std)
    generator = torch.Generator().manual_seed(917)
    windows,labels = [],[]
    for name,starts in data._class_starts['validation'].items():
        chosen = starts[torch.randint(len(starts),(64,),generator=generator)]
        windows.append(data._frames['validation'][chosen[:,None]+torch.arange(WINDOW)])
        labels.extend([name]*64)
    windows = torch.cat(windows).to(args.device)
    noise = torch.randn(len(windows),300,generator=generator).to(args.device)
    timestep = torch.arange(len(windows),device=args.device)%50
    score_noise = torch.randn(3,len(windows),300,generator=generator).to(args.device)
    with torch.no_grad():
        normalized = prior.normalize(windows).flatten(1)
        noised = prior.add_noise(normalized,noise,timestep)
        before = (initial.denoiser(noised,timestep)-noise).abs().mean(-1)
        after = (prior.denoiser(noised,timestep)-noise).abs().mean(-1)
        ema = (prior.ema_denoiser(noised,timestep)-noise).abs().mean(-1)
        errors = prior.score_errors(windows,noise=score_noise)
        scores = prior.score(windows,noise=score_noise)
    assert all(torch.isfinite(value).all() for value in (before,after,ema,errors,scores))
    classes = {}
    for name in sorted(set(labels)):
        mask = torch.tensor([label == name for label in labels],device=args.device)
        classes[name] = dict(windows=int(mask.sum()),initial_l1=float(before[mask].mean()),
            trained_l1=float(after[mask].mean()),ema_l1=float(ema[mask].mean()),
            sds_error=float(errors[mask].mean()))
    return dict(passed=True,iteration=iteration,optimizer_steps=optimizer_steps,training_events=len(train),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        dataset_fingerprint=data.fingerprint,classes=classes,
        train_first100=float(values[:100].mean()),train_last100=float(values[-100:].mean()),
        validation_initial_l1=float(before.mean()),validation_trained_l1=float(after.mean()),
        validation_ema_l1=float(ema.mean()),validation_improved=bool(after.mean()<before.mean()),
        validation_logged=[dict(step=e.step,value=e.value) for e in events.Scalars('SMP/validation_l1')],
        error_normalizer_count=int(prior.error_normalizer.count),assessment_optimizer_steps=0,
        scope='First-run budget/log/checkpoint and bounded heldout denoising checks; no PPO or behavior acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,action='append',required=True)
    parser.add_argument('--expected-updates',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(passed=False,assessment_optimizer_steps=0)
    try:
        report = assess(args)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
