"""Native mjlab checks of recovery goals, horizon and explicit migration; no learning."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from ct_mjlab.experiments import huananhu_v14_recovery as profile
from ct_mjlab.experiments.recovery_goals import predicates, Curriculum


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def goal_checks():
    state = dict(gravity=torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.], [0., 0., -1.]]),
        wheel_support=torch.tensor([[20., 20.], [20., 20.], [20., 0.], [20., 20.]]),
        base_contact=torch.zeros(4), linear_velocity=torch.zeros(4, 3), angular_velocity=torch.zeros(4, 3),
        height=torch.tensor([.20, .30, .30, .30]), target_height=torch.full((4,), .30),
        valid=torch.ones(4, dtype=torch.bool))
    values = predicates(**state)
    assert values['righting'].tolist() == [True, False, True, True]
    assert values['wheel_balance'].tolist() == [True, False, False, True]
    assert values['height_tracking'].tolist() == [False, False, False, True]
    state['wheel_support'][3] = 0.  # clamp(-Fz,0): downward force gives no support
    assert not predicates(**state)['wheel_balance'][3]
    mode = profile.BalanceMode(4, 'cpu', .02)
    mode.reset(torch.arange(4), torch.ones(4, dtype=torch.bool))
    for step in range(25):
        mode.advance_balance(values['wheel_balance'], ~values['supported_upright'], step)
    assert mode.recover.tolist() == [False, True, True, False]
    curriculum = Curriculum()
    for step in range(4800):
        curriculum.advance(1., step, .02)
    assert curriculum.stage == 2 and len(curriculum.history) == 2
    other = Curriculum()
    other.load(curriculum.state_dict())
    assert other.state_dict() == curriculum.state_dict()
    previous = curriculum.stage_steps
    curriculum.advance(0., 4799, .02)
    assert curriculum.stage_steps == previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    goal_checks()
    env = profile.RecoveryVecEnv(num_envs=4096)
    try:
        raw = env.unwrapped
        state = raw.command_manager.get_term('standup')
        runner = profile.RecoveryRunner(env, profile.runner_cfg(), None, device=env.device)
        runner.load(str(args.checkpoint), migration=True)
        assert runner._completed_updates == 2000 and state.curriculum.stage == 0
        migrated = args.output / 'migration-no-learning.pt'
        runner.save(str(migrated))
        original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        saved = torch.load(migrated, map_location='cpu', weights_only=False)
        equality = {key: same(original[key], saved[key]) for key in (
            'actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')}
        assert all(equality.values()), equality
        assert runner.alg.learning_rate == float(original['optimizer_state_dict']['param_groups'][0]['lr'])
        state.curriculum.stage = 2
        runner.save(str(args.output / 'stage-roundtrip.pt'))
        state.curriculum.stage = 0
        runner.load(str(args.output / 'stage-roundtrip.pt'))
        assert state.curriculum.stage == 2 and runner._completed_updates == 2000
        state.curriculum.stage = 0
        # Contact and tilt in a finite ground state must not trigger the old 0.4s failure.
        ground = state.ground_start.clone()
        first_failure = torch.zeros(4096, dtype=torch.bool, device=env.device)
        for _ in range(64):
            obs, value, done, _ = env.step(torch.zeros(4096, 6, device=env.device))
            assert obs['actor'].shape == (4096, 35) and obs['critic'].shape == (4096, 78)
            assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(value).all()
            first_failure |= raw.reset_terminated
            assert not state.numerical_failure.any()
        assert not first_failure[ground].any()
        ids, _ = raw.scene['robot'].find_joints(profile.ACTUATED_JOINTS, preserve_order=True)
        adr = raw.scene['robot'].indexing.joint_v_adr[ids[0]]
        old = raw.sim.data.qvel[0, adr].clone()
        raw.sim.data.qvel[0, adr] = 1000.
        assert profile.failure(raw)[0] and state.numerical_failure[0]
        raw.sim.data.qvel[0, adr] = old
        profile.failure(raw)
        # An unfinished target at the horizon is a terminated failure, not a neutral timeout.
        raw.episode_length_buf[0] = 1000
        state.goal_times[0] = 0.
        assert profile.failure(raw)[0]
        result = dict(learning=False, num_envs=4096, steps=64, finite=True,
            goal_semantics=True, inverted_rejected=True, one_wheel_rejected=True,
            below_target_balance_accepted=True, numerical_immediate=True, unfinished_timeout_failed=True,
            ground_early_failures=int(first_failure[ground].sum()), migration_state_equal=equality,
            completed_updates=runner._completed_updates, learning_rate=runner.alg.learning_rate,
            curriculum_roundtrip=True, profile=profile.manifest())
        (args.output / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
        print('RECOVERY_HEIGHT_CHECK ' + json.dumps({k: v for k, v in result.items() if k != 'profile'}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
