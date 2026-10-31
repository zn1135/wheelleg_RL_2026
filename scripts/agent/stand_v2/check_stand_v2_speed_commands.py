#!/usr/bin/env python3
"""Finite command/curriculum/resume checks plus frozen rollout; never learns."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand.mdp import StandCommand
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp, rewards, observations as observation_terms
from ct_mjlab.envs.chuanliantui_stand_v2.export import export
from ct_mjlab.envs.chuanliantui_stand_v2.runner import task_contracts
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest


def equal_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    elif torch.is_tensor(a):
        assert torch.equal(a, b)
    else:
        assert a == b


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand_v2')
    cfg.commands['standup'].post_stand_tracking = True
    agent.actor.distribution_cfg['std_range'] = (.05, 1e6)
    env, _ = task_registry.make_env('chuanliantui_stand_v2', env_cfg=cfg, train_cfg=agent,
        num_envs=128, seed=42, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui_stand_v2', train_cfg=agent, device=args.device)
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training')):
            runner.load(str(args.checkpoint), map_location=args.device)
            raw, s = env.unwrapped, env.unwrapped.command_manager.get_term('standup')
            before = [tensor_digest(m.state_dict()) for m in (runner.alg.actor, runner.alg.critic)]
            optimizers = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer, runner.alg.encoder_optimizer)]
            s.cfg.speed_curriculum = True
            ids = torch.arange(128, device=raw.device)
            s._resample_command(ids)
            assert .25 < float(s.tracking_episode.float().mean()) < .75
            assert ((s.tracking_acceleration >= .5) & (s.tracking_acceleration <= 2.)).all()
            # Controlled gate boundaries; physics gates are exercised below with frozen3999.
            s.has_stood[:] = s.observation_recovered[:] = True
            s.fallen.zero_()
            kinds = set()
            max_delta_error = 0.
            with patch.object(StandCommand, '_update_command'):
                for tick in range(3049):
                    previous = s.command[:, 0].clone()
                    raw.common_step_counter += 1
                    s._update_command(None)
                    if tick < 49:
                        assert not s.command[:, 0].any()
                    delta = (s.command[:, 0]-previous).abs()
                    max_delta_error = max(max_delta_error, float((delta-s.tracking_acceleration*.01).max()))
                    assert (delta <= s.tracking_acceleration*.01+2e-7).all()
                    if tick >= 49:
                        kinds.update(s.tracking_transition[s.tracking_episode].cpu().tolist())
                    command, ready = s.command.clone(), s.tracking_ready_steps.clone()
                    s._update_command(None)
                    assert torch.equal(command, s.command) and torch.equal(ready, s.tracking_ready_steps)
                    assert not s.command[~s.tracking_episode, :2].any()
                assert kinds == {0, 1, 2, 3}
                s.observation_recovered[:] = False
                raw.common_step_counter += 1
                s._update_command(None)
                assert not s.command[:, :2].any() and not s.tracking_ready_steps.any()
                s.observation_recovered[:] = True
                for _ in range(49):
                    raw.common_step_counter += 1
                    s._update_command(None)
                    assert not s.command[:, :2].any()
            # Zero-speed rewards and exact observations equal the old recipe at the same state.
            s.has_stood[:] = s.reward_active[:] = True
            s.numerical_failure.zero_()
            s._command[:, :2] = 0.
            port = rewards.RewardState(raw)
            def values(speed):
                s.cfg.speed_curriculum = speed
                port.step = -1
                return {k: v.clone() for k, v in port.compute(raw).items()}
            old, new = values(False), values(True)
            assert all(torch.equal(old[k], new[k]) for k in old)
            def observations(speed):
                s.cfg.speed_curriculum = speed
                return dict(actor=observation_terms.actor_observation(raw).clone(),
                            critic=observation_terms.critic_observation(raw).clone())
            old_obs, new_obs = observations(False), observations(True)
            assert all(torch.equal(old_obs[k], new_obs[k]) for k in old_obs)
            # Reward follows the ramped observation command, not hidden target.
            s._command[:, 0] = .2
            s.tracking_velocity[:] = 1.5
            port.step = -1
            first = {k: v.clone() for k, v in port.compute(raw).items()}
            s.tracking_velocity[:] = -.7
            port.step = -1
            assert all(torch.equal(first[k], v) for k, v in port.compute(raw).items())
            obs = dict(actor=observation_terms.actor_observation(raw))
            torch.testing.assert_close(obs['actor'][:, 6], torch.full((128,), .4, device=raw.device))
            # Failure cannot promote; successful high-speed segments advance both boundaries.
            s.speed_stage = s.speed_segments = s.speed_successes = 0
            for _ in range(2):
                s.segment_started[:] = True
                s.segment_stage[:] = 0
                s.segment_steps[:] = 49
                s.segment_error.zero_()
                s.segment_failed.zero_()
                s._finish_speed_segments(ids)
            assert s.speed_stage == s.speed_segments == s.speed_successes == 0
            for good, expected in ((False, 0), (True, 1), (True, 2)):
                for _ in range(2):
                    s.segment_started[:] = True
                    s.segment_stage[:] = s.speed_stage
                    s.segment_steps[:] = 50
                    s.segment_error[:] = 10. if good else 20.
                    s.segment_failed.zero_()
                    s._finish_speed_segments(ids)
                assert s.speed_stage == expected
            s.segment_started[:] = True
            s.segment_stage[:] = s.speed_stage
            s.segment_steps[:] = 100
            s.segment_error[:] = 0.
            s.segment_failed[:] = True
            s._finish_speed_segments(ids)
            assert s.speed_segments == 128 and s.speed_successes == 0
            saved_curriculum = copy.deepcopy(s.checkpoint_state())
            target = args.output/'interface-only-untrained-speed.pt'
            runner.save(str(target))
            s.speed_stage = s.speed_segments = s.speed_successes = 0
            runner.load(str(target), map_location=args.device)
            equal_tree(saved_curriculum, s.checkpoint_state())
            # Config mismatch and old contract must fail before parameter loading.
            s.cfg.tracking_acceleration_range = (.6, 2.)
            try:
                runner.load(str(target), map_location=args.device)
            except ValueError as e:
                assert 'configuration' in str(e)
            else:
                raise AssertionError('Changed slope silently loaded')
            s.cfg.tracking_acceleration_range = (.5, 2.)
            try:
                runner.load(str(args.checkpoint), map_location=args.device)
            except ValueError as e:
                assert 'profile' in str(e)
            else:
                raise AssertionError('Old recipe silently loaded')
            exported = export(target, args.output/'interface-only.onnx', samples=16)
            assert exported['command_contract'] == mdp.SPEED_COMMAND_CONTRACT
            # Partial reset cannot alter other environments' command state.
            s._resample_command(ids)
            other = {k: getattr(s, k)[3:].clone() for k in ('tracking_episode', 'tracking_acceleration',
                'tracking_velocity', 'tracking_interval_steps', 'tracking_ready_steps')}
            raw.reset(env_ids=ids[:3])
            for k, v in other.items():
                assert torch.equal(v, getattr(s, k)[3:])
            assert not s.command[:3, :2].any()
            # Infinite plane, no inherited multimode 20m out-of-patch termination.
            assert cfg.scene.terrain.terrain_type == 'plane'
            assert cfg.terminations['failure'].func is mdp.failure
            planes = (raw.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).nonzero()[0]
            assert len(planes) >= 1
            ray_data = mujoco.MjData(raw.sim.mj_model)
            mujoco.mj_forward(raw.sim.mj_model, ray_data)
            ray_geom = np.array([-1], dtype=np.int32)
            floor_distance = mujoco.mj_ray(raw.sim.mj_model, ray_data,
                np.array([60., 0., 1.]), np.array([0., 0., -1.]), None, True, -1, ray_geom)
            assert ray_geom[0] in planes and abs(floor_distance-1.) < 1e-6
            # Frozen40s with full training randomization, max-stage envelope for slope checks.
            raw.reset()
            s.speed_stage = 2
            s.speed_curriculum_frozen = True
            policy, obs = runner.get_inference_policy(device=args.device), env.get_observations()
            moving_steps = gate_violations = slope_violations = 0
            for tick in range(4000):
                previous = s.command[:, 0].clone()
                obs, reward, _, _ = env.step(policy(obs))
                active = s.tracking_active
                moving = s.command[:, 0] != 0.
                invalid = moving & (~s.has_stood | s.recover | s.fallen | ~s.tracking_episode | (s.tracking_ready_steps < 50))
                gate_violations += int(invalid.sum())
                slope_violations += int((active & ((s.command[:, 0]-previous).abs() > s.tracking_acceleration*.01+2e-7)).sum())
                moving_steps += int(moving.sum())
                assert not s.command[:, 1].any() and s.ground_start.all()
                assert (s.command[:, 0].abs() <= 1.5+1e-6).all()
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
                torch.testing.assert_close(obs['actor'][:, 6], s.command[:, 0]*2.)
                assert torch.equal(obs['history'][:, -32:], obs['actor'])
                if (tick+1) % 1000 == 0:
                    print(f'Frozen speed commands {(tick+1)/100:.0f}/40s', flush=True)
            assert moving_steps > 0 and gate_violations == slope_violations == 0
            assert before == [tensor_digest(m.state_dict()) for m in (runner.alg.actor, runner.alg.critic)]
            for old, optimizer in zip(optimizers, (runner.alg.optimizer, runner.alg.encoder_optimizer)):
                equal_tree(old, optimizer.state_dict())
            report = dict(passed=True, parameter_updates=0, checkpoint_sha256=digest(args.checkpoint),
                contracts=task_contracts(True, True), slope_violations=slope_violations,
                gate_violations=gate_violations, moving_env_steps=moving_steps, transitions=sorted(kinds),
                max_synthetic_slope_excess_m_s=max_delta_error, zero_rewards_equal=True,
                zero_observations_equal=True, reward_uses_ramped_command=True,
                promotion_boundaries_and_failure=True, curriculum_checkpoint_roundtrip=True,
                optimizer_roundtrip_exact=True, partial_reset_isolated=True, export=exported,
                plane_geoms=planes.tolist(), out_of_patch_termination=False,
                floor_present_at_60m=True,
                scope='Finite interface/command checks with frozen3999, not vx1.5 behavior approval')
            (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k != 'export'}, indent=2))
    finally:
        env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
