#!/usr/bin/env python3
"""Measure score randomness on fixed late windows versus adjacent-state changes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ct_mjlab.smp.prior import load_prior
from mjlab.utils.torch import configure_torch_backends


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    configure_torch_backends(allow_tf32=False)
    with np.load(args.trace) as data:
        features = data['features'].copy()
    ticks = np.linspace(1510, 1990, 64).astype(int)
    envs = np.arange(64) % features.shape[1]
    windows = np.stack([features[t-9:t+1, e] for t, e in zip(ticks, envs)])
    following = np.stack([features[t-8:t+2, e] for t, e in zip(ticks, envs)])
    prior, _ = load_prior(args.prior, 'cuda:0')
    prior.freeze()
    # 与当前在线SMP完全相同的固定单位尺度。
    prior.error_normalizer.mean_abs.fill_(1.)
    x = torch.as_tensor(windows, device='cuda:0')
    y = torch.as_tensor(following, device='cuda:0')
    generator = torch.Generator(device='cuda:0').manual_seed(20261026)
    scores, differences = [], []
    for _ in range(args.draws):
        noise = torch.randn(3, len(x), 300, device='cuda:0', generator=generator)
        a = torch.exp(-6*prior.score_errors(x, noise=noise).mean(-1))
        b = torch.exp(-6*prior.score_errors(y, noise=noise).mean(-1))
        scores.append(a.cpu().numpy()); differences.append((b-a).cpu().numpy())
    scores, differences = np.asarray(scores), np.asarray(differences)
    velocity = features[ticks, envs, 4:6]; next_velocity = features[ticks+1, envs, 4:6]
    def speed_reward(v):
        error = np.square(v).sum(-1)
        return np.exp(-error/.02**2)+.2*(np.exp(-error/.2**2)-1.)
    delta = speed_reward(next_velocity)-speed_reward(velocity)
    report = dict(completed=True, optimizer_updates=0, windows=64, draws=args.draws,
        fixed_window_score_std_mean=float(scores.std(0, ddof=1).mean()),
        fixed_window_score_std_median=float(np.median(scores.std(0, ddof=1))),
        adjacent_window_common_noise_mean_abs_change=float(np.abs(differences.mean(0)).mean()),
        adjacent_velocity_reward_mean_abs_change=float(np.abs(delta).mean()),
        mean_score=float(scores.mean()),
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        prior_sha256=hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Reward rates before common dt; fixed windows isolate scoring randomness, common-noise adjacent windows isolate state dependence. Not policy-gradient causal attribution.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'prior', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--draws', type=int, default=64)
    run(parser.parse_args())
