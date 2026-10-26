#!/usr/bin/env python3
"""Check precision braking reward, transfer and real20env reset isolation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import configure_smp, policy_observation_contract, checkpoint_bc_finetune
from ct_mjlab.smp.finetune import PROFILE_ISAAC_V3, RECIPE_ISAAC_V3
from ct_mjlab.smp.isaac_standup_rewards import velocity_scores, RewardState
from check_ppo_finetune_resume import equal


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    command = torch.zeros(128, 3)
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(128, 3, generator=generator)*.2
    b = torch.randn(128, 3, generator=generator)*.2
    stood = torch.ones(128, dtype=torch.bool)
    def score(v, convex):
        x, y = velocity_scores(command, v, stood, tight=True, convex=convex)
        return x+.2*y
    ready = True
    # 凹得分/凸成本：相同平均速度下，增加波动不能增加精细制动得分。
    assert (score((a+b)/2, ready)+1e-6 >= (score(a, ready)+score(b, ready))/2).all()
    low = torch.zeros_like(a); low[:, 0] = .01
    high = torch.zeros_like(a); high[:, 0] = .03
    assert (score(low, ready) > score(high, ready)).all()
    zero_gate = velocity_scores(command, a, torch.zeros_like(stood), tight=True, convex=True)
    assert not zero_gate[0].any() and not zero_gate[1].any()
    old = velocity_scores(command, a, stood, tight=True)
    before = velocity_scores(command, a, stood, tight=True, convex=False)
    for x, y in zip(old, before):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    counterexample = {}
    for label, directory in (('401', args.run/'checkpoint400-evaluation'), ('601', args.run/'checkpoint600-evaluation')):
        with np.load(directory/'trace.npz') as data:
            v = torch.from_numpy(data['features'][1500:, :, 4:7].reshape(-1, 3).copy())
        c = torch.zeros_like(v); gate = torch.ones(len(v), dtype=torch.bool)
        x, y = velocity_scores(c, v, gate, tight=True)
        n, m = velocity_scores(c, v, gate, tight=True, convex=True)
        counterexample[label] = dict(speed=float(v[:, :2].norm(dim=-1).mean()),
            old_score=float((x+.2*y).mean()), new_precision_score=float((n+.2*m).mean()))
    assert counterexample['601']['speed'] > counterexample['401']['speed']
    assert counterexample['601']['old_score'] > counterexample['401']['old_score']
    assert counterexample['601']['new_precision_score'] < counterexample['401']['new_precision_score']
    cfg, train = task_registry.get_cfgs('chuanliantui')
    cfg.seed = train.seed = 42
    configure_smp(cfg, train, args.prior, policy_contract=policy_observation_contract(args.checkpoint),
                  bc_finetune='isaac-standup-v3')
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=20)
    try:
        raw = env.unwrapped
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        critic = copy.deepcopy(runner.alg.critic.state_dict())
        runner.initialize_from_ppo(args.checkpoint)
        source = torch.load(args.checkpoint, map_location=raw.device, weights_only=False)
        for key, value in source['actor_state_dict'].items():
            if key != 'distribution.std_param':
                torch.testing.assert_close(runner.alg.actor.state_dict()[key], value, atol=0, rtol=0)
        equal(critic, runner.alg.critic.state_dict())
        assert runner.alg.recipe == RECIPE_ISAAC_V3 and not runner.alg.optimizer.state
        initial = copy.deepcopy(runner.alg.save())
        legacy = RewardState(raw, tight_velocity=True)
        original = RewardState.compute
        compared = []
        minimum_primary = []
        def compare(self, owner):
            values = original(self, owner)
            if compared and compared[-1] == owner.common_step_counter:
                return values
            previous = original(legacy, owner)
            for key in values:
                if key not in ('tracking_lin_vel', 'tracking_lin_vel_enhance'):
                    torch.testing.assert_close(values[key], previous[key], atol=0, rtol=0)
            expected = velocity_scores(owner.command_manager.get_term('standup').command,
                owner.scene['robot'].data.root_link_lin_vel_b, self.stood, tight=True, convex=True)
            active = owner.command_manager.get_term('standup').policy_active
            for name, value in zip(('tracking_lin_vel', 'tracking_lin_vel_enhance'), expected):
                torch.testing.assert_close(values[name], value*active, atol=0, rtol=0)
            minimum_primary.append(values['tracking_lin_vel'].min().clone())
            compared.append(owner.common_step_counter)
            return values
        with torch.inference_mode(), patch.object(RewardState, 'compute', compare), \
                patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')):
            obs = env.get_observations()
            for tick in range(800):
                obs, reward, _, _ = env.step(runner.alg.actor(obs))
                assert torch.isfinite(reward).all()
                torch.testing.assert_close(raw.reward_manager._step_reward.sum(-1)*raw.step_dt, reward)
                if tick < 20:
                    assert not raw.action_manager.get_term('pd').torques.any()
            state = raw._isaac_standup_rewards
            assert state.convex_velocity and state.stood.all() and not hasattr(state, 'precision')
            ids = torch.arange(5, device=raw.device)
            raw.reset(env_ids=ids); legacy.reset(ids)
            assert not state.stood[:5].any() and state.stood[5:].all()
        minimum = float(torch.stack(minimum_primary).min())
        assert minimum < -1., 'Real velocity cost was clipped back to the old floor'
        equal(initial, runner.alg.save())
        path = args.output/'untrained-v3.pt'; runner.save(str(path))
        assert checkpoint_bc_finetune(path) == 'isaac-standup-v3'
        resumed, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        resumed.load(str(path)); equal(initial, resumed.alg.save())
        try:
            resumed.load(str(args.checkpoint))
        except ValueError:
            pass
        else:
            raise AssertionError('Cross-reward full resume accepted')
        report = dict(passed=True, optimizer_updates=0, profile=PROFILE_ISAAC_V3,
            speed_counterexample=counterexample, precision_score_concavity=True,
            before_precision_matches_v2=True, unchanged_other_reward_terms=17,
            physical_envs=20, policy_steps=800, compared_reward_calls=len(compared),
            original_stood_gate_reached=True, no_new_phase_state=True, partial_reset_isolated=True,
            actual_velocity_terms_match_convex_formula=True, minimum_primary_reward_rate=minimum,
            source_actor_mean_exact=True, fresh_critic_and_optimizer=True,
            checkpoint_roundtrip=True, cross_reward_resume_rejected=True,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            scope='Reward geometry, actual20env reward/control/reset/transfer; no policy training or behavior improvement proof.')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
