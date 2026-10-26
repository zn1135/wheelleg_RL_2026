#!/usr/bin/env python3
"""Recompute frozen wheel-pulse returns and actual PPO GAE directions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(delta, null):
    return dict(mean=float(delta.mean()), standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                null_mean_absolute=float(np.abs(null).mean()), positive_cases=int((delta > 0).sum()))


def assess(args):
    report = json.loads((args.run/'result.json').read_text())
    assert report['completed'] and report['optimizer_updates'] == 0
    assert max(report['initial_fork_max_difference'].values()) == 0
    for name, expected in report['hashes'].items(): assert digest(name) == expected
    with np.load(args.run/'wheel-pulse.npz') as data:
        reward = data['reward'].astype(np.float64)
        value = data['value'].astype(np.float64)
        initial = data['initial_value'].astype(np.float64)
        assert not data['done'].any()
    gamma = report['gamma']
    previous = np.concatenate((initial[None], value[:-1]))
    residual = reward+gamma*value-previous
    real = (reward*gamma**np.arange(500)[:, None]).sum(0).reshape(4, 40)
    actual_delta = real[2]-real[3]
    rows = []
    for horizon in (24, 100, 200, 500):
        for lam in (.95, .99, 1.):
            advantage = (residual[:horizon]*(gamma*lam)**np.arange(horizon)[:, None]).sum(0).reshape(4, 40)
            delta = advantage[2]-advantage[3]
            rows.append(dict(horizon_steps=horizon, gae_lambda=lam,
                **describe(delta, advantage[0]-advantage[1]),
                sign_agreement_with_observed5s=float((np.sign(delta)==np.sign(actual_delta)).mean())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, optimizer_updates=0, actual5s=describe(actual_delta, real[0]-real[1]),
                  gae_comparison=rows, pulse_sha256=digest(args.run/'wheel-pulse.npz'),
                  source_sha256=digest(__file__),
                  scope='First-action finite-difference direction only, at40 cached15s states. '
                        'GAE recomputed from frozen actual rewards/values with identical initial states. '
                        'No claim that one rollout/lambda change alone solves parking.')
    # 单独改变上一拍动作输入，检查价值预测是否直接受该记忆槽影响。
    state_file = Path(report['continuation_source']).parent/'state-at15s.pt'
    source = torch.load(state_file, map_location='cpu', weights_only=False)
    checkpoint = next(Path(p) for p in report['hashes'] if p.endswith('.pt') and 'prior_' not in p)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = saved['critic_state_dict']
    layer_keys = sorted([k[:-7] for k in weights if k.startswith('mlp.') and k.endswith('.weight')],
                        key=lambda k: int(k.split('.')[1]))

    def critic(inputs):
        x = inputs
        for i, key in enumerate(layer_keys):
            x = F.linear(x, weights[key+'.weight'], weights[key+'.bias'])
            if i < len(layer_keys)-1: x = F.elu(x)
        return x.squeeze(-1)

    inputs = source['observations']['critic'][:40]
    plus, minus = inputs.clone(), inputs.clone()
    plus[:, 26] += report['pulse_raw_action']; plus[:, 27] -= report['pulse_raw_action']
    minus[:, 26] -= report['pulse_raw_action']; minus[:, 27] += report['pulse_raw_action']
    with torch.inference_mode(): counterfactual = (critic(plus)-critic(minus)).numpy()
    result['previous_action_only_value_delta'] = dict(mean=float(counterfactual.mean()),
                                                     negative_cases=int((counterfactual < 0).sum()))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    assess(p.parse_args())
