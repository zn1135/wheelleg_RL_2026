#!/usr/bin/env python3
"""Check wheel-only variance changes and saved recipe compatibility, no updates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import configure_smp, policy_observation_contract
from ct_mjlab.smp.finetune import ACTION_STD, RECIPE_ISAAC_V2, checked_recipe
from check_ppo_finetune_resume import equal


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('chuanliantui')
    cfg.seed = train.seed = 42
    configure_smp(cfg, train, args.prior, policy_contract=policy_observation_contract(args.checkpoint),
                  bc_finetune='isaac-standup-v2')
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=20)
    try:
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        runner.load(str(args.checkpoint))
        original = copy.deepcopy(runner.alg.save())
        with torch.inference_mode():
            obs = env.get_observations()
            mean = runner.alg.actor(obs).clone()
        paths = []
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No updates')):
            for wheel in (.01, .02, .05):
                runner.alg.set_wheel_std(wheel)
                state = runner.alg.save()
                assert state['bc_finetune']['recipe']['action_std'] == [.001, .001, wheel, .001, .001, wheel]
                for key in ('critic_state_dict', 'optimizer_state_dict'):
                    equal(original[key], state[key])
                for key in original['actor_state_dict']:
                    if key != 'distribution.std_param':
                        equal(original['actor_state_dict'][key], state['actor_state_dict'][key])
                for key in original['bc_finetune']:
                    if key != 'recipe':
                        equal(original['bc_finetune'][key], state['bc_finetune'][key])
                with torch.inference_mode():
                    torch.testing.assert_close(mean, runner.alg.actor(obs), atol=0, rtol=0)
                path = args.output/f'untrained-wheel-{wheel}.pt'
                runner.save(str(path)); paths.append(path)
            resumed, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
            resumed.load(str(paths[-1])); equal(runner.alg.save(), resumed.alg.save())
            assert resumed.alg.recipe['wheel_exploration_std'] == .05
            resumed.load(str(args.checkpoint)); equal(original, resumed.alg.save())
            assert 'wheel_exploration_std' not in resumed.alg.recipe
            bad = {**RECIPE_ISAAC_V2, 'wheel_exploration_std': .2,
                   'action_std': [.001, .001, .2, .001, .001, .2]}
            try:
                checked_recipe(bad, RECIPE_ISAAC_V2)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsupported wheel variance accepted')
        assert tuple(RECIPE_ISAAC_V2['action_std']) == ACTION_STD
        result = dict(passed=True, optimizer_updates=0, num_envs=20,
            mean_network_critic_adam_reference_and_counts_unchanged=True,
            fixed_wheel_std_checked=[.01, .02, .05], leg_std=.001,
            checkpoint_roundtrip=True, old_checkpoint_restores_original_recipe=True,
            unsupported_recipe_rejected=True,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            scope='Variance-only mutation and checkpoint roundtrip; no policy learning, no physics acceptance.')
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
