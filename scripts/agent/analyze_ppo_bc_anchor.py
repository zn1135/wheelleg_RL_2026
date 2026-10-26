#!/usr/bin/env python3
"""Measure trained action changes on fixed recorded BC states; no optimization."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import load_actor


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def analyze(args):
    torch.set_num_threads(1)
    bc, _ = load_actor(args.bc, 'cpu'); policy, _ = load_actor(args.bc, 'cpu')
    state = torch.load(args.ppo, map_location='cpu', weights_only=False)
    policy.load_state_dict(state['actor_state_dict'])
    with np.load(args.trace) as d: obs = d['observation'].copy()
    sigma = state['actor_state_dict']['distribution.std_param'].numpy()
    result = {}
    for name, interval in (('rise', slice(20, 200)), ('late5s', slice(1500, 2000))):
        inputs = torch.from_numpy(obs[interval].reshape(-1, 39))
        change = policy.mlp(inputs)-bc.mlp(inputs)
        delta = change.numpy()
        result[name] = dict(action_rms_per_motor=np.sqrt(np.square(delta).mean(0)).tolist(),
            leg_target_rms_rad=float(np.sqrt(np.square(delta[:, [0, 1, 3, 4]]*.5).mean())),
            wheel_target_rms_rad_s=float(np.sqrt(np.square(delta[:, [2, 5]]*10).mean())),
            normalized_reference_loss=float(.5*np.square(delta/sigma).mean()))
    # 仅描述损失曲率；不能把损失值与原始奖励值直接比较成策略梯度因果。
    coefficients=(1./(12*sigma*sigma)).tolist()
    report = dict(completed=True, optimizer_updates=0, phases=result,
        reference_quadratic_coefficient_per_motor=coefficients, action_noise_sigma=sigma.tolist(),
        completed_iterations=state['bc_finetune']['completed_iterations'],
        bc_sha256=digest(args.bc), ppo_sha256=digest(args.ppo), trace_sha256=digest(args.trace),
        scope='Fixed BC observations, not closed-loop improvement. Reference term curvature describes strong/weak action anchoring; no per-term PPO gradient attribution.')
    args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bc', 'ppo', 'trace', 'output'): p.add_argument('--'+name, type=Path, required=True)
    analyze(p.parse_args())
