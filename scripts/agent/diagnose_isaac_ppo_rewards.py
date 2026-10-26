#!/usr/bin/env python3
"""Replay a saved benchmark with exact per-term rewards; forbid learning."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import evaluate_lqr_ppo


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    original = evaluate_lqr_ppo.task_registry.make_env
    contributions, gates = [], []
    holder = {}

    def make_env(*pos, **kw):
        env, cfg = original(*pos, **kw)
        raw = env.unwrapped
        holder['names'] = list(raw.reward_manager.active_terms)
        step = env.step

        def measured_step(action):
            result = step(action)
            # 管理器存每秒加权奖励；乘一次dt才是本步实际送给PPO的奖励。
            values = raw.reward_manager._step_reward * raw.step_dt
            torch.testing.assert_close(values.sum(-1), result[1], atol=2e-6, rtol=2e-5)
            contributions.append(values.cpu().numpy().copy())
            gates.append(raw._isaac_standup_rewards.stood.cpu().numpy().copy())
            return result

        env.step = measured_step
        return env, cfg

    args.kind = 'ppo'
    with patch.object(evaluate_lqr_ppo.task_registry, 'make_env', side_effect=make_env), \
            patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Training forbidden')):
        evaluate_lqr_ppo.run(args)
    values = np.asarray(contributions)
    names = holder['names']
    assert values.shape == (2000, 40, len(names)) and np.isfinite(values).all()
    np.savez_compressed(args.output/'rewards.npz', contributions=values,
                        names=np.asarray(names), stood=np.asarray(gates))
    with np.load(args.expected_trace) as old, np.load(args.output/'trace.npz') as new:
        # GPU接触仿真复跑不保证逐位一致；初态必须一致，轨迹偏差单独报告。
        assert set(old.files) == set(new.files)
        for key in ('initial_qpos', 'initial_qvel'):
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
        assert new['alive'].all()
        speed = np.linalg.norm(new['features'][:, :, 4:6], axis=-1)
        exact = all(np.array_equal(new[key], old[key]) for key in old.files)
        differences = {key: float(np.abs(new[key]-old[key]).max())
                       for key in ('qpos', 'qvel', 'features', 'physical', 'action')}
    segments = {}
    for label, part in (('all20s', slice(0, 2000)), ('rise', slice(20, 200)),
                        ('2to5s', slice(200, 500)), ('5to10s', slice(500, 1000)),
                        ('10to15s', slice(1000, 1500)), ('late5s', slice(1500, 2000))):
        rates = values[part].astype(np.float64).mean((0, 1))/.01
        terms = dict(zip(names, map(float, rates)))
        segments[label] = dict(terms_per_second=terms, total_per_second=float(rates.sum()),
            mean_speed_m_s=float(speed[part].mean()), stood_fraction=float(np.asarray(gates)[part].mean()))
    report = dict(passed=True, optimizer_updates=0, initial_states_exactly_equal=True,
        trajectories_exactly_unchanged=exact, max_absolute_replay_differences=differences,
        checkpoint_sha256=digest(args.checkpoint), expected_trace_sha256=digest(args.expected_trace),
        rewards_sha256=digest(args.output/'rewards.npz'), source_sha256=digest(__file__),
        segments=segments,
        scope='Actual manager contributions in a fresh matched-initial-state replay; trajectories may vary. No per-term policy-gradient causal attribution.')
    (args.output/'reward-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'source', 'expected-trace', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
