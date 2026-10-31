#!/usr/bin/env python3
"""Recompute frozen refinement replay metrics and checkpoint provenance; no simulation."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.smp.finetune import ACTION_STD, RECIPE, PROFILE, RECIPE_V2, PROFILE_V2


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(args):
    report = json.loads((args.run/'result.json').read_text())
    recipe = RECIPE_V2 if report['recipe']['version'] == PROFILE_V2 else RECIPE
    profile = recipe['version']
    assert report['passed'] and report['optimizer_updates'] == 0 and report['recipe'] == recipe
    assert report['trace_sha256'] == digest(args.run/'trace.npz')
    for path, sha in report['immutable_sha256'].items(): assert digest(path) == sha
    cached = torch.load(args.source/'nominal_mean/initial-state.pt', map_location='cpu', weights_only=False)
    saved = torch.load(args.run/'untrained-finetune.pt', map_location='cpu', weights_only=False)
    assert saved['infos']['experiment_profile_contract'] == profile
    assert saved['infos']['bc_init']['sha256'] == digest(args.checkpoint)
    assert not saved['optimizer_state_dict']['state']
    assert saved['bc_finetune']['completed_iterations'] == 0
    assert saved['bc_finetune']['actor_updates'] == saved['bc_finetune']['critic_updates'] == 0
    assert saved['bc_finetune']['recipe'] == recipe and saved['bc_finetune']['reference_ready']
    # BC检查点键名由正式加载器统一，避免把训练优化器当作策略权重。
    from ct_mjlab.bc import load_actor
    actor, _ = load_actor(args.checkpoint, 'cpu')
    for key, value in actor.mlp.state_dict().items():
        torch.testing.assert_close(saved['actor_state_dict']['mlp.'+key], value, atol=0, rtol=0)
        torch.testing.assert_close(saved['bc_finetune']['reference_state_dict'][key], value, atol=0, rtol=0)
    torch.testing.assert_close(saved['actor_state_dict']['distribution.std_param'], torch.tensor(ACTION_STD), atol=0, rtol=0)
    with np.load(args.run/'trace.npz') as archive: d = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(d['initial_qpos'][:, 2:], cached['qpos'].numpy()[:, 2:])
    np.testing.assert_array_equal(d['initial_qvel'], cached['qvel'].numpy())
    noise_std = (d['action']-d['mean']).std((0, 1))
    np.testing.assert_allclose(noise_std, ACTION_STD, rtol=.025, atol=1e-6)
    assert d['alive'].shape == (2000, 40) and d['alive'].all()
    for row in report['rows']:
        i = row['env']; late = slice(1500, 2000)
        leg = float(np.sqrt(np.square(d['motor_dq'][late, i][:, [0, 1, 3, 4]]).mean()))
        speed = float(np.linalg.norm(d['features'][late, i, 4:6], axis=-1).mean())
        drift = float(np.linalg.norm(d['qpos'][late, i, :2]-d['qpos'][1500, i, :2], axis=-1).max())
        assert leg == row['leg_rms'] and speed == row['speed'] and drift == row['drift']
    rewards = {name: float(d['rewards'][1500:, :, i].mean()/.01) for i, name in enumerate(d['reward_names'])}
    assert 0. < rewards['zero_command_brake'] <= 2. and rewards['zero_command_position'] > 0.
    result = dict(passed=True, optimizer_updates=0, frozen_bc_means_and_reference_identical=True,
        empty_optimizer_and_zero_training_counters=True, all40_complete20s=True,
        strict_rise=report['strict_rise'], end_balance=report['end_balance'], smooth=report['smooth'],
        parking=report['qualified'], mean_leg_rms=report['mean_leg_rms'], mean_speed=report['mean_speed'],
        measured_noise_std=noise_std.tolist(), late_reward_per_second=rewards,
        checkpoint_sha256=digest(args.run/'untrained-finetune.pt'), trace_sha256=report['trace_sha256'],
        checker_sha256=digest(__file__),
        scope='Replay metrics and saved-state checks only; contact acceptance flags reuse runtime actual-weight probe. No optimization, new generalization or parking improvement.')
    (args.run/'assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'checkpoint'): p.add_argument('--'+name, type=Path, required=True)
    assess(p.parse_args())
