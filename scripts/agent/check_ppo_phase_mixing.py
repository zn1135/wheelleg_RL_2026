#!/usr/bin/env python3
"""Check real staggered starts, motor gating and checkpoint state without updates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.runner import DirectMotorOnPolicyRunner
from ct_mjlab.smp.ppo import configure_smp, checkpoint_bc_finetune, policy_observation_contract
from check_ppo_finetune_resume import equal


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('chuanliantui')
    cfg.seed = train.seed = 42
    configure_smp(cfg, train, args.prior, policy_contract=policy_observation_contract(args.checkpoint),
                  bc_finetune=checkpoint_bc_finetune(args.checkpoint))
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=20)
    try:
        raw = env.unwrapped
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        runner.load(str(args.checkpoint))
        before = copy.deepcopy(runner.alg.save())
        counter = raw.common_step_counter
        assert not hasattr(runner, 'phase_mixing')
        runner.enable_phase_mixing()
        dispatched = []

        def before_training(owner, num_learning_iterations, init_at_random_ep_len=False):
            assert owner is runner and num_learning_iterations == 1 and not init_at_random_ep_len
            assert owner._phase_mixing_ready
            assert raw.common_step_counter == counter
            assert raw.episode_length_buf.min() == 0 and raw.episode_length_buf.max() == 1900
            dispatched.append(True)

        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')), \
                patch.object(DirectMotorOnPolicyRunner, 'learn', before_training):
            runner.learn(1)
            runner.learn(1)  # 同一个活环境不能重复预运行。
        assert len(dispatched) == 2
        stats = runner.phase_mixing
        assert stats['prelude_runs'] == 1 and stats['prelude_control_steps_total'] == 2000
        assert stats['last_prelude']['phase_counts'] == [1, 1, 8, 10]
        assert stats['last_prelude']['target_age_steps'] == stats['last_prelude']['actual_age_steps']
        assert stats['last_prelude']['physical_failures'] == 0
        torch.testing.assert_close(raw._smp_reward.history.count,
                                   raw.episode_length_buf.clamp(max=10), atol=0, rtol=0)
        assert torch.equal(raw.command_manager.get_term('standup').policy_active,
                           raw.episode_length_buf >= 20)
        equal(before, runner.alg.save())
        torch.save(dict(qpos=raw.sim.data.qpos.cpu().clone(), qvel=raw.sim.data.qvel.cpu().clone(),
            age=raw.episode_length_buf.cpu().clone(),
            observations={k: v.cpu().clone() for k, v in env.get_observations().items()}), args.output/'phase-state.pt')
        runner.current_learning_iteration = runner.alg.completed_iterations-1
        runner.save(str(args.output/'untrained-phase.pt'))
        with torch.inference_mode(), patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')):
            obs = env.get_observations()
            for _ in range(24):
                ages = raw.episode_length_buf.clone()
                obs, _, _, _ = env.step(runner.alg.actor(obs, stochastic_output=True))
                assert not raw.action_manager.get_term('pd').torques[ages < 20].any()
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
        equal(before, runner.alg.save())
        resumed, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        resumed.load(str(args.output/'untrained-phase.pt'))
        equal(before, resumed.alg.save())
        assert resumed.phase_mixing == stats and not resumed._phase_mixing_ready
        resumed.load(str(args.checkpoint))
        assert not hasattr(resumed, 'phase_mixing')
        report = dict(passed=True, optimizer_updates=0, num_envs=20,
            true_physics_prelude_steps=2000, mixed_batch_steps=24,
            actor_critic_adam_reference_and_counters_unchanged=True,
            ground_hold_each_reset_checked=True, initial_reset_distribution_unchanged=True,
            sampling=stats, real_learn_dispatch_before_training=True,
            checkpoint_roundtrip=True, old_checkpoint_clears_sampler=True,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            scope='Real20env staggered physics and mixed24-step batch, no parameter updates; not trained policy acceptance.')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
