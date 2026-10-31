#!/usr/bin/env python3
"""Frozen-policy rollout audit: episode phases and long-return value estimates."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import configure_smp, checkpoint_bc_finetune, policy_observation_contract


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('chuanliantui')
    cfg.seed = train.seed = args.seed
    configure_smp(cfg, train, args.prior, policy_contract=policy_observation_contract(args.checkpoint),
                  bc_finetune=checkpoint_bc_finetune(args.checkpoint))
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=args.num_envs)
    try:
        raw = env.unwrapped
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        runner.load(str(args.checkpoint), load_cfg={'actor': True, 'critic': True})
        alg = runner.alg
        initial = {name: {k: v.clone() for k, v in model.state_dict().items()}
                   for name, model in (('actor', alg.actor), ('critic', alg.critic))}
        fields = {k: [] for k in ('age', 'reward', 'value', 'done', 'speed')}
        batches = []
        with torch.inference_mode(), patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No learning')):
            raw.reset(seed=args.seed)
            obs = env.get_observations()
            for batch in range(83):
                ages = []
                for tick in range(24):
                    age = raw.episode_length_buf.clone().float() * raw.step_dt
                    action = alg.act(obs)
                    value = alg.transition.values.squeeze(-1).clone()
                    obs, reward, done, extras = env.step(action)
                    alg.process_env_step(obs, reward, done, extras)
                    state = dict(age=age, reward=reward, value=value, done=done,
                                 speed=raw.scene['robot'].data.root_link_lin_vel_b[:, :2].norm(dim=-1))
                    for key, val in state.items():
                        fields[key].append(val.cpu().numpy().copy())
                    ages.append(age)
                alg.compute_returns(obs)
                age = torch.stack(ages)
                advantage = alg.storage.returns-alg.storage.values
                batches.append(dict(batch=batch, age_min=float(age.min()), age_max=float(age.max()),
                    inactive_fraction=float((age < .2).float().mean()),
                    rise_fraction=float(((age >= .2) & (age < 2)).float().mean()),
                    transition_fraction=float(((age >= 2) & (age < 10)).float().mean()),
                    balance_fraction=float((age >= 10).float().mean()),
                    advantage_mean=float(advantage.mean()), advantage_std=float(advantage.std()),
                    critic_mean=float(alg.storage.values.mean())))
                alg.storage.clear()
                if batch % 20 == 0:
                    print(f'batch={batch} age={batches[-1]["age_min"]:.2f}..{batches[-1]["age_max"]:.2f}s', flush=True)
            tail = alg.critic(obs).squeeze(-1).cpu().numpy().copy()
        arrays = {k: np.asarray(v) for k, v in fields.items()}
        target = np.empty_like(arrays['value'], dtype=np.float64)
        carry = tail.astype(np.float64)
        # 用冻结策略的后续真实奖励作长回报目标；末端bootstrap影响随折扣衰减。
        for tick in reversed(range(len(target))):
            carry = arrays['reward'][tick]+alg.gamma*(1-arrays['done'][tick])*carry
            target[tick] = carry
        summaries = {}
        for name, lo, hi in (('hold', 0, .2), ('rise', .2, 2), ('early_balance', 2, 5),
                             ('braking', 5, 10), ('late', 10, 15)):
            mask = (arrays['age'] >= lo) & (arrays['age'] < hi)
            prediction = arrays['value'][mask]; actual = target[mask]
            summaries[name] = dict(samples=int(mask.sum()), critic_mean=float(prediction.mean()),
                long_return_mean=float(actual.mean()), bias=float((prediction-actual).mean()),
                rmse=float(np.sqrt(np.square(prediction-actual).mean())))
        for name, model in (('actor', alg.actor), ('critic', alg.critic)):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[name][key], atol=0, rtol=0)
        assert not alg.optimizer.state
        np.savez_compressed(args.output/'rollout.npz', **arrays, long_return=target, bootstrap=tail)
        report = dict(completed=True, optimizer_updates=0, num_envs=args.num_envs, seed=args.seed,
            seconds=19.92, steps=1992, gamma=alg.gamma, gae_lambda=alg.lam,
            phase_batches=batches, value_error=summaries, resets=int(arrays['done'].sum()),
            checkpoint_sha256=digest(args.checkpoint), prior_sha256=digest(args.prior),
            source_sha256=digest(__file__), rollout_sha256=digest(args.output/'rollout.npz'),
            scope='Frozen stochastic policy on training reset distribution; critic compared with observed discounted rewards and final-value bootstrap. Not behavior acceptance or causal training ablation.')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(summaries, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    run(p.parse_args())
