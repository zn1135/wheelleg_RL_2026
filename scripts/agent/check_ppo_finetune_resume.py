#!/usr/bin/env python3
"""Verify nonempty Adam/BC reference restoration and next iteration; never learns."""
import argparse, hashlib, json
from pathlib import Path
from unittest.mock import patch
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.runner import DirectMotorOnPolicyRunner
from ct_mjlab.smp.ppo import configure_smp, policy_observation_contract, checkpoint_smp_weight


def equal(left, right):
    if isinstance(left, torch.Tensor): torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): equal(a, b)
    else: assert left == right


def run(args):
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg, train = task_registry.get_cfgs('chuanliantui'); cfg.seed = train.seed = 42
    configure_smp(cfg, train, args.prior, policy_contract=policy_observation_contract(args.checkpoint),
                  reward_weight=checkpoint_smp_weight(args.checkpoint), bc_finetune=True)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=2)
    try:
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No learning')):
            runner.load(str(args.checkpoint))
            state = runner.alg.save()
            for name in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'bc_finetune'):
                equal(state[name], source[name])
            assert runner.alg.optimizer.state
            completed = source['bc_finetune']['completed_iterations']
            assert runner.alg.completed_iterations == completed >= 20
            assert env.unwrapped.common_step_counter == source['infos']['env_state']['common_step_counter']
            visited = []
            def before_sampling(owner, num_learning_iterations, init_at_random_ep_len=False):
                assert num_learning_iterations == 300 and not init_at_random_ep_len
                assert owner.current_learning_iteration == completed
                assert owner.alg.completed_iterations == completed
                visited.append(True)
            with patch.object(DirectMotorOnPolicyRunner, 'learn', before_sampling):
                runner.learn(num_learning_iterations=300)
            assert visited
            for name in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'bc_finetune'):
                equal(runner.alg.save()[name], source[name])
        result = dict(passed=True, optimizer_updates=0, nonempty_adam_restored=True,
            actor_critic_and_bc_reference_exact=True, completed_iterations=completed,
            next_iteration=runner.current_learning_iteration, warmup_not_restarted=True,
            learn_intercepted_before_sampling=True,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            scope='Exact saved tensors/moments/counters restored and next iteration dispatch checked; physical episodes and global RNG are restarted, not bitwise trajectory continuation.')
        (args.output/'check.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))
    finally: env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'output'): p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
