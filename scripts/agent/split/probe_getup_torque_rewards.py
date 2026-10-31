#!/usr/bin/env python3
"""Measure pre-stand torque reward size with the frozen EMA80 actor; never learn."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split.runner import contracts
from ct_mjlab.envs.chuanliantui_stand_v2 import rewards as shared
from scripts.agent.split.check_getup_nominal_state import source_module
from scripts.agent.split.check_chuanliantui_two_policy import digest


@torch.inference_mode()
def probe(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_module(args.before_shared, 'ct_mjlab.envs.chuanliantui_stand_v2._torque_probe_before')
    cfg, agent = task_registry.get_cfgs('chuanliantui_getup', play=True)
    cfg.auto_reset, cfg.seed = False, 43
    env, _ = task_registry.make_env('chuanliantui_getup', env_cfg=cfg, train_cfg=agent,
        num_envs=4, device=args.device)
    raw = env.unwrapped
    samples = []
    live = torch.ones(4, dtype=torch.bool, device=args.device)
    first_stood, ended = [None]*4, [None]*4
    try:
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui_getup', train_cfg=agent, device=args.device)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = raw.command_manager.get_term('standup')
        normalize = lambda value: json.loads(json.dumps(value))
        # 跨奖励配方只读权重诊断，不伪造checkpoint或绕过正常续训/play的拒载保护。
        assert saved['infos']['reward_recipe'] == 'split-getup-torque2-full-strength-nominal-state-v3'
        assert all(normalize(saved['infos'].get(k)) == normalize(v)
            for k, v in contracts(state.cfg).items() if k != 'reward_recipe')
        runner.alg.actor.load_state_dict(saved['actor_state_dict'], strict=True)
        state.load_checkpoint_state(saved['infos']['standup_curriculum'])
        policy = runner.get_inference_policy(device=args.device)
        actor_before = {k:v.cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        obs, _ = raw.reset(seed=43)
        assert state.ground_start.all()
        initial_qpos = raw.sim.data.qpos.cpu().numpy().copy()
        with np.load(args.reference/'trajectory.npz') as reference:
            assert np.array_equal(initial_qpos, reference['qpos'][0])
        oracle = before.RewardState(raw)
        names = tuple(raw.cfg.rewards)
        weights = torch.tensor([raw.cfg.rewards[n].weight for n in names], device=args.device)
        compute = shared.RewardState.compute

        def measure(reward_state, current_env):
            fresh = reward_state.step != current_env.common_step_counter
            values = compute(reward_state, current_env)
            if fresh:
                oracle.step = -1
                previous = oracle.compute(current_env)
                assert all(torch.equal(values[n], previous[n]) for n in names if n != 'torques')
                assert torch.equal(values['torques'][state.has_stood], previous['torques'][state.has_stood])
                rates = torch.stack([values[n] for n in names], -1)*weights
                old_rates = torch.stack([previous[n] for n in names], -1)*weights
                samples.append(dict(rates=rates.clone(), old=old_rates.clone(),
                    pre=(live & state.reward_active & ~state.has_stood & ~state.numerical_failure).clone(),
                    active=(live & state.reward_active & ~state.numerical_failure).clone(),
                    torque=raw.sim.data.qfrc_actuator[:, reward_state.motor_vadr].clone()))
            return values

        with patch.object(shared.RewardState, 'compute', measure):
            for tick in range(args.steps):
                action = policy(obs)
                if agent.clip_actions is not None:
                    action = action.clamp(-agent.clip_actions, agent.clip_actions)
                obs, _, terminated, timeout, _ = raw.step(action)
                assert len(samples) == tick+1
                assert torch.allclose(raw.reward_manager._step_reward, samples[-1]['rates'], atol=1e-6)
                for i in range(4):
                    if live[i] and state.has_stood[i] and first_stood[i] is None:
                        first_stood[i] = (tick+1)*raw.step_dt
                    if live[i] and (terminated[i] or timeout[i]):
                        ended[i] = (tick+1)*raw.step_dt
                live &= ~(terminated | timeout)
                if tick+1 < args.steps and (terminated | timeout).any():
                    raw.reset(env_ids=(terminated | timeout).nonzero().flatten())
                    obs = raw.get_observations()
        assert all(torch.equal(v.cpu(), actor_before[k]) for k,v in runner.alg.actor.state_dict().items())
        array = lambda key: torch.stack([s[key] for s in samples]).cpu().numpy()
        rates, old_rates, pre, active, torque = (array(k) for k in ('rates','old','pre','active','torque'))
        assert np.isfinite(rates).all() and np.isfinite(old_rates).all()
        index = names.index('torques')
        rows = []
        for i in range(4):
            mask = pre[:, i]
            costs = rates[mask, i, index]
            positive = rates[mask, i].clip(min=0).sum()
            rows.append(dict(env=i, pre_stand_active_ticks=int(mask.sum()),
                mean_torque_penalty_rate=float(costs.mean()) if len(costs) else 0.,
                mean_torque_penalty_per_step=float(costs.mean()*raw.step_dt) if len(costs) else 0.,
                cumulative_torque_penalty=float(costs.sum()*raw.step_dt),
                positive_reward_sum=float(positive*raw.step_dt),
                torque_cost_to_positive_reward_ratio=float(-costs.sum()/positive) if positive else None,
                torque_cost_saturated_fraction=float((costs <= -1.+1e-6).mean()) if len(costs) else 0.,
                baseline_pre_stand_torque_penalty=float(old_rates[mask,i,index].sum()*raw.step_dt)))
        np.savez_compressed(args.output/'reward-trace.npz', weighted_rates=rates,
            baseline_weighted_rates=old_rates, pre_stand_first_episode=pre,
            active_first_episode=active, motor_torque_Nm=torque,
            reward_names=np.asarray(names), policy_dt_s=raw.step_dt, initial_qpos=initial_qpos)
        report = dict(verified=True, parameter_updates=0, explicit_frozen_actor_reward_probe=True,
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
            seed=43, num_envs=4, steps=args.steps, duration_s=args.steps*raw.step_dt,
            initial_qpos_matches_previous_recording=True, actor_unchanged=True,
            other_18_rewards_bitwise_equal_on_actual_states=True, actual_manager_rates_match=True,
            weight=-.0002, weighted_minimum_before_dt=-1., rows=rows,
            first_stood_s=first_stood, first_episode_end_s=ended,
            scope='Frozen baseline actor, nominal first episodes, reward contribution diagnostic only; no new-policy behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--before-shared', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Probe cannot train')):
        probe(parser.parse_args())
