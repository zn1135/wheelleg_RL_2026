#!/usr/bin/env python3
"""Audit a first SMP PPO run's budget/checkpoint/logs; no optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ct_mjlab.smp.ppo import ADAPTIVE_STAND_PROFILE, LEGACY_PROFILE, PROFILE, WHEEL_PROFILE
from ct_mjlab.smp.finetune import (PROFILE as FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3,
                                  RECIPE, RECIPE_V2, RECIPE_ISAAC, RECIPE_ISAAC_V2, RECIPE_ISAAC_V3, checked_recipe)


def assess(args):
    start = getattr(args, 'start_iteration', 0)
    assert 0 <= start < args.expected_iterations
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    infos = state['infos']
    assert infos['experiment_profile_contract'] in (PROFILE, WHEEL_PROFILE, FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3, ADAPTIVE_STAND_PROFILE, LEGACY_PROFILE)
    assert infos['smp']['prior_sha256'] == hashlib.sha256(args.prior.read_bytes()).hexdigest()
    assert state['iter'] == args.expected_iterations-1, 'PPO checkpoint uses zero-based iteration indices'
    assert infos['env_state']['common_step_counter'] == args.expected_iterations*24
    steps = sorted({int(value['step']) for value in state['optimizer_state_dict']['state'].values()})
    if infos['experiment_profile_contract'] in (FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3):
        fine = state['bc_finetune']; opt = state['optimizer_state_dict']
        recipe = {FINETUNE_PROFILE: RECIPE, PROFILE_V2: RECIPE_V2,
                  PROFILE_ISAAC: RECIPE_ISAAC, PROFILE_ISAAC_V2: RECIPE_ISAAC_V2, PROFILE_ISAAC_V3: RECIPE_ISAAC_V3}[infos['experiment_profile_contract']]
        recipe = checked_recipe(fine['recipe'], recipe)
        assert fine['reference_ready']
        assert fine['completed_iterations'] == args.expected_iterations
        assert fine['critic_updates'] == args.expected_iterations*20
        assert 0 <= fine['actor_updates'] <= max(0, args.expected_iterations-recipe['critic_warmup_iterations'])*20
        for group, expected in zip(opt['param_groups'], (fine['actor_updates'], fine['critic_updates']), strict=True):
            for pid in group['params']:
                assert int(opt['state'].get(pid, {}).get('step', 0)) == expected
        assert torch.equal(state['actor_state_dict']['distribution.std_param'], torch.tensor(recipe['action_std']))
    else:
        assert steps == [args.expected_iterations*20], 'Expected five epochs x four mini-batches per iteration'
    for component in ('actor_state_dict', 'critic_state_dict'):
        assert all(torch.isfinite(value).all() for value in state[component].values())
    normalizer = infos['smp_normalizer']
    if infos['experiment_profile_contract'] in (PROFILE, WHEEL_PROFILE, FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3):
        assert infos['smp']['normalizer_mode'] == 'fixed-unit'
        assert infos['smp']['normalizer_scale'] == [1., 1., 1.]
        assert int(normalizer['count']) == 0 and not normalizer['pending_sum'].count_nonzero()
        assert torch.equal(normalizer['mean_abs'], torch.ones(3))
    else:
        assert int(normalizer['count']) == 1_000_000
    assert int(normalizer['pending_count']) == 0
    assert torch.isfinite(normalizer['mean_abs']).all() and (normalizer['mean_abs'] > 0).all()
    events = EventAccumulator(str(args.checkpoint.parent), size_guidance={'scalars': 0})
    events.Reload()
    summaries = {}
    last_logged_step = start-1
    for tag in events.Tags()['scalars']:
        all_records = events.Scalars(tag)
        records = [r for r in all_records if r.step < args.expected_iterations] if getattr(args, 'allow_later_logs', False) else all_records
        if tag.startswith('Loss/') or tag in ('SMP/reward', 'SMP/normalizer_count', 'SMP/valid_fraction'):
            assert [record.step for record in records] == list(range(start, args.expected_iterations)), tag
            assert all(np.isfinite(record.value) for record in records), tag
            last_logged_step = max(last_logged_step, max(r.step for r in all_records))
        if records and tag.startswith(('Loss/', 'SMP/', 'Train/', 'Metrics/', 'Episode_Reward/')):
            values = [record.value for record in records]
            summaries[tag] = dict(count=len(values), first=values[0], last=values[-1],
                                  first20_mean=float(np.mean(values[:20])),
                                  last20_mean=float(np.mean(values[-20:])))
    assert 'SMP/reward' in summaries and any(tag.startswith('Loss/') for tag in summaries)
    if infos['smp']['weight'] == 0.:
        assert all(record.value == 0. for record in events.Scalars('SMP/reward'))
    copied_prior = args.checkpoint.parent/'smp_prior.pt'
    assert hashlib.sha256(copied_prior.read_bytes()).hexdigest() == infos['smp']['prior_sha256']
    report = dict(passed=True, authorized_ppo_iterations=args.expected_iterations-start,
        total_completed_ppo_iterations=args.expected_iterations, start_iteration=start,
        checkpoint_iteration_index=state['iter'], checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        optimizer_steps=steps, policy_control_steps=infos['env_state']['common_step_counter'],
        prior_sha256=infos['smp']['prior_sha256'], prior_copy_unchanged=True,
        normalizer_count=int(normalizer['count']), normalizer_mean_abs=normalizer['mean_abs'].tolist(),
        smp_reward_weight=infos['smp']['weight'], bc_init=infos.get('bc_init'),
        ppo_init=infos.get('ppo_init'),
        metrics=summaries, scope='first-run budget/log/checkpoint audit, not behavior acceptance',
        assessment_optimizer_updates=0)
    if getattr(args, 'allow_later_logs', False):
        report.update(checkpoint_prefix_only=True, last_logged_iteration_index=last_logged_step,
                      logged_iterations_beyond_checkpoint=last_logged_step+1-args.expected_iterations)
    if 'phase_mixing' in infos:
        from ct_mjlab.smp.phase_sampling import CONTRACT
        sampling = infos['phase_mixing']
        assert sampling['contract'] == CONTRACT
        assert sampling['prelude_control_steps_total'] == 2000*sampling['prelude_runs']
        assert sampling['prelude_runs'] >= 1
        report['phase_mixing'] = sampling
        report['physical_control_steps_including_preludes'] = (
            infos['env_state']['common_step_counter']+sampling['prelude_control_steps_total'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key != 'metrics'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--expected-iterations', type=int, required=True)
    parser.add_argument('--start-iteration', type=int, default=0, help='Completed iterations before this resumed run')
    parser.add_argument('--allow-later-logs', action='store_true', help='Audit saved checkpoint prefix after an interrupted run; report later logs separately')
    parser.add_argument('--output', type=Path, required=True)
    assess(parser.parse_args())


if __name__ == '__main__':
    main()
