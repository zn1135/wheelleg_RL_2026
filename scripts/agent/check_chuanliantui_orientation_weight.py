"""Finite simulation and exact full-state migration checks; no PPO updates."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import torch

from ct_mjlab.experiments import getup_curriculum as base, getup_orientation as profile


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Incompatible checkpoint was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--migration-checkpoint', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert args.num_envs > 0 and args.steps > 0
    manifest = profile.manifest()
    assert profile.ORIENTATION_WEIGHTS == (-30., -30., -30.)
    original = base.manifest()
    for old, new in zip(original['weights'], manifest['weights']):
        assert {k: v for k, v in old.items() if k != 'orientation'} == {
            k: v for k, v in new.items() if k != 'orientation'}
    result = dict(profile=profile.INTERFACE, no_policy_updates=True, stages={},
                  only_orientation_weights_changed=True)
    env = profile.OrientationVecEnv(args.num_envs, device=args.device)
    try:
        raw = env.unwrapped
        current = base.serializable(profile.make_env_cfg(args.num_envs))
        previous = base.serializable(base.make_env_cfg(args.num_envs))
        previous['rewards']['orientation']['func'] = current['rewards']['orientation']['func']
        assert previous == current
        state = raw.command_manager.get_term('standup')
        state.cfg.curriculum_enabled = False
        cache = base.RewardState(raw)
        raw._getup_rewards = cache
        for stage, name in enumerate(base.STAGES):
            state.curriculum.stage = stage
            state._update_command(None)
            numerical = 0
            for _ in range(args.steps):
                obs, value, _, _ = env.step(torch.zeros(args.num_envs, 6, device=args.device))
                assert obs['actor'].shape == (args.num_envs, 35) and obs['critic'].shape == (args.num_envs, 78)
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(value).all()
                numerical += int(state.numerical_failure.sum())
            cache.step = -1
            values = cache.compute(raw)
            gravity = raw.scene['robot'].data.projected_gravity_b
            xy = gravity[:, :2].square().sum(-1)
            expected = .02 * xy + 2 * (gravity[:, 2] + 1).square() - torch.exp(-xy / .25).clamp(0., .1)
            expected.masked_fill_(state.numerical_failure, 0.)
            assert torch.allclose(values['orientation'], expected)
            assert torch.allclose(profile.reward(raw, name='orientation'), -30 * expected)
            total = sum(values[term] * weight for term, weight in profile.WEIGHTS[stage].items()) * raw.step_dt
            assert torch.allclose(raw.reward_manager.compute(raw.step_dt), total, atol=1e-5)
            assert numerical == 0
            result['stages'][name] = dict(steps=args.steps, numerical_failures=numerical,
                finite=True, old_orientation_formula_unchanged=True, reward_dt_applied_once=True)
    finally:
        env.close()

    if args.migration_checkpoint:
        saved = torch.load(args.migration_checkpoint, map_location='cpu', weights_only=False)
        profile.validate_checkpoint(saved, allow_orientation_weight=True)
        rejected(lambda: profile.validate_checkpoint(saved))
        corrupt = copy.deepcopy(saved)
        corrupt['infos']['interface']['module_sha256'] = 'unknown source'
        rejected(lambda: profile.validate_checkpoint(corrupt, allow_orientation_weight=True))
        env = profile.OrientationVecEnv(saved['infos']['environment']['scene']['num_envs'], device=args.device)
        try:
            with tempfile.TemporaryDirectory(prefix='orientation-migration-') as directory:
                runner = profile.OrientationRunner(env, base.runner_cfg(), directory, device=args.device)
                runner.load(str(args.migration_checkpoint), allow_orientation_weight=True)
                migrated_path = Path(directory) / 'model_migrated_initial.pt'
                runner.save(str(migrated_path))
                migrated = torch.load(migrated_path, map_location='cpu', weights_only=False)
                for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
                    assert equal_state(saved[key], migrated[key]), key
                assert migrated['infos']['getup_curriculum'] == saved['infos']['getup_curriculum']
                assert migrated['infos']['completed_updates'] == saved['infos']['completed_updates']
                assert runner.alg.learning_rate == saved['optimizer_state_dict']['param_groups'][0]['lr']
                state = env.unwrapped.command_manager.get_term('standup')
                effective_updates = state.training_iteration + env.unwrapped.common_step_counter // 24
                assert effective_updates == saved['infos']['completed_updates']
                env.unwrapped.action_manager.get_term('pd').apply_actions()
                assert (state.metrics['pull_up_budget_ratio'] == 0).all()
                assert (state.metrics['pull_up_force_n'] == 0).all()
                profile.validate_checkpoint(migrated)
                rejected(lambda: base.validate_checkpoint(migrated))
                runner.load(str(migrated_path))
                assert runner.current_learning_iteration == saved['infos']['completed_updates']
                result['migration'] = dict(num_envs=env.num_envs,
                    completed_updates=saved['infos']['completed_updates'],
                    actor_critic_adam_exact=True, curriculum_exact=True,
                    learning_rate=runner.alg.learning_rate,
                    training_iteration_offset=state.training_iteration,
                    common_step_counter=env.unwrapped.common_step_counter,
                    assistance_zero=True, rejects_implicit_v3=True,
                    rejects_unknown_source=True, rejects_old_profile_playback=True,
                    new_profile_roundtrip=True)
        finally:
            env.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print('ORIENTATION_VALIDATION ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
