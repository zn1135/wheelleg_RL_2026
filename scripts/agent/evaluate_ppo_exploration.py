#!/usr/bin/env python3
"""Original40 sampled-policy benchmark, changing wheel exploration only."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import evaluate_lqr_ppo


def run(args):
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    original = evaluate_lqr_ppo.task_registry.make_alg_runner
    means = []
    holder = {}

    def make_runner(*pos, **kw):
        runner, cfg = original(*pos, **kw)
        load = runner.load

        def sampled_load(*load_pos, **load_kw):
            result = load(*load_pos, **load_kw)
            actor = runner.alg._raw_actor
            std = torch.tensor([.001, .001, args.wheel_std, .001, .001, args.wheel_std], device=runner.device)
            with torch.no_grad():
                actor.distribution.std_param.copy_(std)
            holder['std'] = std.cpu().numpy()
            forward = actor.forward

            def sampled_forward(obs):
                action = forward(obs, stochastic_output=True)
                means.append(actor.output_distribution_params[0].cpu().numpy().copy())
                return action

            actor.forward = sampled_forward
            return result

        runner.load = sampled_load
        return runner, cfg

    args.kind = 'ppo'
    with patch.object(evaluate_lqr_ppo.task_registry, 'make_alg_runner', side_effect=make_runner), \
            patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')):
        evaluate_lqr_ppo.run(args)
    means = np.asarray(means)
    with np.load(args.output/'trace.npz') as data:
        innovation = (data['action']-means)/holder['std']
    empirical_std = innovation.std((0, 1))
    assert means.shape == (2000, 40, 6) and np.isfinite(innovation).all()
    np.testing.assert_allclose(empirical_std, np.ones(6), atol=.02, rtol=0)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == source_hash
    np.savez_compressed(args.output/'sampling.npz', mean=means, std=holder['std'])
    path = args.output/'result.json'
    report = json.loads(path.read_text())
    report.update(policy_sampling='stochastic Gaussian, frozen mean network',
        sampled_action_std=holder['std'].tolist(), empirical_normalized_std=empirical_std.tolist(),
        exploration_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Frozen mean network, sampled actions, original40 normal starts and2/4ms delays; wheel-only noise sweep, no parameter updates or checkpoint changes. Not deterministic policy acceptance.')
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'prior', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wheel-std', type=float, choices=(.005, .01, .02, .05), required=True)
    run(parser.parse_args())
