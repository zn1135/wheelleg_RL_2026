"""Finite r15 checks for source rewards, batch transitions and checkpoint state.

No PPO updates. Run in the independent Python 3.12/mjlab environment.
"""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from ct_mjlab.experiments import getup_curriculum as getup
from ct_mjlab.experiments.pull_up_assistance import pull_up_force


def check_wheel_support(raw):
    # Direct support, one wheel airborne, downward reaction, and sub-threshold
    # contacts must not be conflated. Probe the complete promotion predicate.
    force = torch.zeros(4, 2, 3, device=raw.device)
    force[..., 2] = force.new_tensor(((-20., -30.), (-20., 0.), (20., 30.), (-4., -4.)))
    robot = SimpleNamespace(root_link_pos_w=force.new_tensor([[0., 0., .27]]).repeat(4, 1),
                            projected_gravity_b=force.new_tensor([[0., 0., -1.]]).repeat(4, 1))
    state = SimpleNamespace(command=force.new_tensor([[0., 0., .27]]).repeat(4, 1),
                            numerical_failure=torch.zeros(4, dtype=torch.bool, device=raw.device))
    action = SimpleNamespace(assistance_force=force.new_zeros(4), robot_weight=lambda: force.new_full((4,), 100.))
    probe = SimpleNamespace(scene={'robot': SimpleNamespace(data=robot),
                                  'wheel_contact': SimpleNamespace(data=SimpleNamespace(force=force)),
                                  'base_contact': SimpleNamespace(data=SimpleNamespace(force=force.new_zeros(4, 1, 3))),
                                  'unused': None},
                            command_manager=SimpleNamespace(get_term=lambda _: state),
                            action_manager=SimpleNamespace(get_term=lambda _: action))
    probe.scene = type('ProbeScene', (dict,), {'env_origins': force.new_zeros(4, 3)})(probe.scene)
    assert getup.wheel_support_force(probe).tolist() == [[20., 30.], [20., 0.], [0., 0.], [4., 4.]]
    assert getup.readiness(probe).tolist() == [True, False, False, False]
    robot.projected_gravity_b[0] = force.new_tensor((1., 0., 0.))
    assert not bool(getup.readiness(probe)[0])
    robot.projected_gravity_b[0] = force.new_tensor((0., 0., -1.))
    action.assistance_force[0] = 6.
    assert not bool(getup.readiness(probe)[0])

    # Exercise actual reward kernels, rather than only testing the helper.
    sensor_force = raw.scene['wheel_contact'].data.force
    original_force = sensor_force.clone()
    cache = getup.RewardState(raw)
    try:
        sensor_force.zero_(); sensor_force[..., 2] = -20.
        values = cache.compute(raw)
        valid = ~raw.command_manager.get_term('standup').numerical_failure
        assert (values['no_fly'][valid] == 0.).all()
        assert 'pose_bias' not in values
        expected_load = torch.exp(-(40. - raw.sim.model.body_mass[:, cache.robot.indexing.body_ids].sum(-1) * 10.).abs() * .05)
        assert torch.allclose(values['wheel_load'][valid], expected_load[valid])
        sensor_force[:, 1, 2] = 0.; cache.step = -1
        values = cache.compute(raw)
        assert (values['no_fly'][valid] == 1.).all()
        assert 'pose_bias' not in values
        sensor_force[..., 2] = 20.; cache.step = -1
        assert (cache.compute(raw)['no_fly'][valid] == 1.).all()
    finally:
        sensor_force.copy_(original_force)
    return dict(signed_upward_support=True, airborne_and_downward_rejected=True,
                actual_no_fly=True, pose_bias_removed=True, support_load_formula=True,
                promotion_contact_tilt_and_assistance_gates=True)


def check_assistance(env):
    raw = env.unwrapped
    action = raw.action_manager.get_term('pd')
    state = raw.command_manager.get_term('standup')
    disturbance = raw._huananhu_randomization
    episode_steps = raw.episode_length_buf.clone()
    action.apply_actions()
    assert (action.assistance_force[state.ground_start] == 0.).all()
    raw.episode_length_buf.fill_(action._ground_hold_steps)
    # Compose a tilted robot's existing local force/torque with world-up aid.
    disturbance.local_wrench[:] = torch.tensor((1., 2., 3., .1, .2, .3), device=raw.device)
    disturbance.apply_local_wrench()
    expected = raw.sim.data.xfrc_applied[:, action._base_id].clone()
    action.apply_actions()
    assisted = expected.clone()
    assisted[:, 2] += action.assistance_force
    assert torch.allclose(raw.sim.data.xfrc_applied[:, action._base_id], assisted)
    initial_force = action.assistance_force.clone()
    assert bool((initial_force > 0.).any())
    assert (initial_force <= action.robot_weight()).all()
    # Repeated physical substeps must not accumulate force or erase disturbances.
    for _ in range(20):
        action.apply_actions()
        assert torch.allclose(raw.sim.data.xfrc_applied[:, action._base_id], assisted)
    state.training_iteration = 1500
    action.apply_actions()
    assert torch.allclose(action.assistance_force, initial_force * .5, atol=1e-5)
    state.training_iteration = 3000
    action.apply_actions()
    assert (action.assistance_force == 0.).all()
    state.training_iteration = 0
    state.curriculum.stage = 2
    action.apply_actions()
    assert (action.assistance_force == 0.).all()
    state.curriculum.stage = 0
    action.cfg.assistance_enabled = False
    action.apply_actions()
    assert (action.assistance_force == 0.).all()
    assert torch.allclose(raw.sim.data.xfrc_applied[:, action._base_id], expected)
    action.cfg.assistance_enabled = True
    action.apply_actions()
    action.reset(torch.tensor([0], device=raw.device))
    assert action.assistance_force[0] == 0.
    assert (raw.sim.data.xfrc_applied[0, action._base_id] == 0.).all()
    disturbance.local_wrench.zero_()
    disturbance.apply_local_wrench()
    action.apply_actions()
    # Independent formula limits, including above-target and corrupt inputs.
    height = torch.tensor((0., .15, .30, .40, float('nan')), device=raw.device)
    target = torch.full_like(height, .30)
    force = pull_up_force(height, target, torch.full_like(height, 100.), 0.)
    assert torch.allclose(force, force.new_tensor((100., 50., 0., 0., 0.)))
    raw.episode_length_buf.copy_(episode_steps)
    return dict(world_up_at_com=True, disturbance_preserved=True, no_force_accumulation=True,
                ground_motor_hold_respected=True,
                maximum_one_weight=True, half_at_1500=True, zero_at_3000=True,
                walking_and_disabled_zero=True, selected_reset_cleared=True,
                observed_force_range_n=[float(initial_force.min()), float(initial_force.max())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--migration-checkpoint', type=Path)
    parser.add_argument('--pose-bias-migration-checkpoint', type=Path)
    args = parser.parse_args()
    assert args.num_envs > 0 and args.steps > 0
    # A whole-batch curriculum must not advance on one successful environment,
    # or before its minimum training time. Calls in one step must be idempotent.
    curriculum = getup.Curriculum()
    for step in range(1, 2501):
        assert not curriculum.advance(.5, step, .02)
    assert curriculum.stage == 0
    for step in range(2501, 2601):
        curriculum.advance(1., step, .02)
        assert not curriculum.advance(1., step, .02)
    assert curriculum.stage == 1
    for step in range(2601, 5001):
        curriculum.advance(1., step, .02)
    assert curriculum.stage == 2 and len(curriculum.history) == 2
    restored = getup.Curriculum()
    restored.load(curriculum.state_dict())
    assert restored.state_dict() == curriculum.state_dict()

    env = getup.GetupVecEnv(args.num_envs, device=args.device)
    result = dict(profile=getup.INTERFACE, stages={}, no_policy_updates=True,
                  curriculum_transition_and_restore=True)
    try:
        raw = env.unwrapped
        state = raw.command_manager.get_term('standup')
        assert all('pose_bias' not in weights for weights in getup.WEIGHTS)
        assert 'pose_bias' not in raw.cfg.rewards
        result['pose_bias'] = dict(removed_from_all_stages=True, removed_from_manager=True)
        if args.pose_bias_migration_checkpoint:
            legacy = torch.load(args.pose_bias_migration_checkpoint, map_location='cpu', weights_only=False)
            getup.validate_checkpoint(legacy, allow_pose_bias_fix=True)
            assert all('pose_bias' not in weights for weights in getup.WEIGHTS)
            try:
                getup.validate_checkpoint(legacy)
            except ValueError:
                pass
            else:
                raise AssertionError('Implicit pose bias migration accepted')
            legacy['infos']['interface']['module_sha256'] = 'unrecognized source'
            try:
                getup.validate_checkpoint(legacy, allow_pose_bias_fix=True)
            except ValueError:
                pass
            else:
                raise AssertionError('Unknown pose bias migration source accepted')
            result['explicit_pose_bias_migration_source_validation'] = True
        if args.migration_checkpoint:
            legacy = torch.load(args.migration_checkpoint, map_location='cpu', weights_only=False)
            getup.validate_checkpoint(legacy, allow_contact_fix=True)
            try:
                getup.validate_checkpoint(legacy)
            except ValueError:
                pass
            else:
                raise AssertionError('Implicit contact migration accepted')
            legacy['infos']['interface']['module_sha256'] = 'unrecognized source'
            try:
                getup.validate_checkpoint(legacy, allow_contact_fix=True)
            except ValueError:
                pass
            else:
                raise AssertionError('Unknown legacy source accepted')
            result['explicit_contact_migration_source_validation'] = True
        result['assistance'] = check_assistance(env)
        # Freeze stage promotion while probing each weight group independently.
        state.cfg.curriculum_enabled = False
        reward_cache = getup.RewardState(raw)
        raw._getup_rewards = reward_cache
        for stage, name in enumerate(getup.STAGES):
            state.curriculum.stage = stage
            state._update_command(None)
            numerical = 0
            speed_seen = False
            for _ in range(args.steps):
                obs, value, done, extra = env.step(torch.zeros(args.num_envs, 6, device=args.device))
                assert obs['actor'].shape == (args.num_envs, 35)
                assert obs['critic'].shape == (args.num_envs, 78)
                assert all(torch.isfinite(v).all() for v in obs.values())
                assert torch.isfinite(value).all()
                assert torch.allclose(state.command[:, 2], torch.full_like(state.command[:, 2], getup.HEIGHTS[stage]))
                if stage < 2:
                    assert (state.command[:, :2] == 0.).all()
                speed_seen |= bool((state.command[:, :2] != 0.).any())
                numerical += int(state.numerical_failure.sum())
            assert numerical == 0
            # env.step forwards kinematics before observations; refresh the
            # diagnostic cache so the formula comparison uses that same state.
            reward_cache.step = -1
            values = reward_cache.compute(raw)
            gravity = raw.scene['robot'].data.projected_gravity_b
            xy = gravity[:, :2].square().sum(-1)
            orientation_expected = .02 * xy + 2 * (gravity[:, 2] + 1).square() - torch.exp(-xy / .25).clamp(0., .1)
            assert torch.allclose(values['orientation'], orientation_expected)
            height = raw.scene['robot'].data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            assert torch.allclose(values['height'], torch.exp(-(height - state.command[:, 2]).abs() / .12))
            # The actual manager must multiply each selected weight and dt once.
            expected = sum(values[term] * weight for term, weight in getup.WEIGHTS[stage].items()) * raw.step_dt
            actual = raw.reward_manager.compute(raw.step_dt)
            assert torch.allclose(actual, expected, atol=1e-5)
            for term in ('linear_tracking', 'yaw_tracking'):
                if stage < 2:
                    assert (getup.reward(raw, name=term) == 0.).all()
            assert 'pose_bias' not in values
            if stage == 2:
                assert speed_seen
            if stage == 0:
                result['wheel_support'] = check_wheel_support(raw)
            result['stages'][name] = dict(steps=args.steps, finite=True, numerical_failures=numerical,
                                          zero_speed_commands=stage < 2)
        # Flips must be visible to the new orientation reward (old xy-only terms
        # cannot distinguish an upright base from an inverted one).
        assert 2 * (1 + 1) ** 2 - .1 > 2 * (-1 + 1) ** 2 - .1
        assert abs(float(torch.exp(torch.tensor(-.10 / .12))) - .4345982) < 1e-6
        state.curriculum.stage = 1
        state._update_command(None)
        state.curriculum.stage_steps = 222
        state.curriculum.ready_steps = 17
        state.curriculum.history = [dict(from_stage='righting', to_stage='standing', control_step=2400)]
        with tempfile.TemporaryDirectory(prefix='getup-check-') as directory:
            cfg = getup.runner_cfg()
            runner = getup.GetupRunner(env, cfg, directory, device=args.device)
            path = Path(directory) / 'model_initial.pt'
            runner.save(str(path))
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['infos']['completed_updates'] == 0
            assert getup.validate_checkpoint(saved)['getup_curriculum'] == state.curriculum.state_dict()
            state.curriculum = getup.Curriculum()
            runner.load(str(path))
            assert runner.current_learning_iteration == 0
            assert state.curriculum.state_dict() == saved['infos']['getup_curriculum']
            assert (state.command[:, 2] == .30).all()
            # Smoke-check the new frozen-stage playback path using the untouched
            # initial actor, not a trained-policy behaviour claim.
            getup.run(SimpleNamespace(command='play', checkpoint=path, num_envs=1,
                                      seed=42, device=args.device, seconds=.04, render=False))
            # Synthetic metadata probes resume at the middle of the force fade;
            # no optimizer step or policy learning is performed.
            saved['infos']['completed_updates'] = 1500
            saved['iter'] = 1499
            probe = Path(directory) / 'resume_counter_probe.pt'
            torch.save(saved, probe)
            runner.load(str(probe))
            assert runner.current_learning_iteration == 1500
            action = raw.action_manager.get_term('pd')
            action.apply_actions()
            height = raw.scene['robot'].data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            expected_force = pull_up_force(height, state.command[:, 2], action.robot_weight(),
                                          1500., stage=state.curriculum.stage)
            expected_force.masked_fill_(action._ground_hold(), 0.)
            assert torch.allclose(action.assistance_force, expected_force)
            assert torch.allclose(state.metrics['pull_up_budget_ratio'], torch.full_like(expected_force, .5))
            saved['infos']['interface']['interface'] = 'chuanliantui-mjlab-wheelleg-recover-r14'
            try:
                getup.validate_checkpoint(saved)
            except ValueError:
                pass
            else:
                raise AssertionError('Implicit historical resume was accepted')
        result.update(checkpoint_restore=True, rejects_historical_resume=True,
                      playback_smoke=True,
                      assistance_resume_counter=True,
                      wheel_spacing_m=reward_cache.wheel_spacing,
                      reward_dt_applied_once=True, source_height_and_orientation_formulas=True)
    finally:
        env.close()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print('GETUP_VALIDATION ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
