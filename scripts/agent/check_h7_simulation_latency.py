#!/usr/bin/env python3
"""Check training PD delays/references and measured playback timing; no learning."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output path')
    import mj_sim2sim_huananhu_multimode as cpu
    from ct_mjlab.envs.chuanliantui.multimode_env_cfg import make_multimode_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    playback = cpu.Playback()
    profile = playback.h7_timing
    paired = profile['paired_cache_to_first_pd_us']
    assert paired['count'] == 1017
    assert all(int(np.floor(paired[k] / 2000 + .5)) == 1 for k in ('min', 'mean', 'max'))
    assert not profile['simulation']['microsecond_jitter_modeled']
    configs = [make_multimode_env_cfg(play=play) for play in (False, True)]
    for cfg in configs:
        assert cfg.sim.mujoco.timestep == cpu.DT == .002
        assert cfg.decimation == cpu.DECIMATION == 5
    assert profile['simulation']['policy_dt_s'] == .01
    assert configs[0].actions['pd'].action_delay_s is None
    assert configs[0].actions['pd'].action_delay_choices_s == (.006, .008, .010)
    assert configs[0].actions['pd'].randomize_delays
    assert configs[0].actions['pd'].nominal_leg_offset_range == (-math.radians(5), math.radians(5))
    assert configs[1].actions['pd'].action_delay_s == .002
    assert not configs[1].actions['pd'].randomize_delays
    assert configs[1].actions['pd'].nominal_leg_offset_range == (0., 0.)
    # Sensor randomization is separate from the selected measured action path.
    assert configs[0].observations['actor'].terms['proprioception'].params['cfg'].delay
    assert not configs[1].observations['actor'].terms['proprioception'].params['cfg'].delay

    captured = []
    controls = playback.controls
    def record_controls(action):
        captured.append((float(playback.data.time), action.copy()))
        return controls(action)
    playback.controls = record_controls
    a, b = np.full(6, .01, dtype=np.float32), np.full(6, -.01, dtype=np.float32)
    playback.step(lambda obs: a)
    playback.step(lambda obs: b)
    for i, (_, action) in enumerate(captured):
        expected = np.zeros(6) if i == 0 else a if i <= cpu.DECIMATION else b
        np.testing.assert_allclose(action, expected, atol=1e-8)
    assert abs(captured[1][0] - .002) < 1e-12
    assert abs(captured[6][0] - .012) < 1e-12
    np.testing.assert_array_equal(playback.last_action, b)
    playback.reset()
    captured.clear()
    playback.step(lambda obs: a)
    np.testing.assert_array_equal(captured[0][1], np.zeros(6))

    cfg = configs[1]
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        assert env.physics_dt == .002 and env.step_dt == .01
        env.reset(seed=42)
        term = env.action_manager.get_term('pd')
        assert term._ground_hold_steps == 20  # 策略加速后，启动零力矩仍保持0.2秒。
        assert bool((term.action_delay_steps == 1).all())
        assert bool((term.wheel_action_delay_steps == 1).all())
        assert not bool(term.nominal_offset.any())
        env.command_manager.get_term('standup').ground_start[:] = False
        # Fixed state isolates action timing from motion; only one leg is excited.
        env.sim.data.qpos[:, term._motor_q_adr] = term.nominal_positions
        env.sim.data.qvel[:, term._motor_v_adr] = 0.
        action = torch.zeros((4, 6), device=env.device)
        action[:, 0] = .1
        term.reset()
        term.process_actions(action)
        term.apply_actions()
        torch.testing.assert_close(term._torques, torch.zeros_like(action), atol=1e-5, rtol=0)
        term.apply_actions()
        expected = torch.zeros_like(action); expected[:, 0] = 3.
        torch.testing.assert_close(term._torques, expected, atol=1e-5, rtol=0)
        # A partial reset must clear only those environments' queued actions.
        ids = torch.tensor([0, 2], device=env.device)
        term.reset(ids)
        term.process_actions(action * 2)
        term.apply_actions()
        expected[ids] = 0
        torch.testing.assert_close(term._torques, expected, atol=1e-5, rtol=0)
        term.apply_actions()
        expected[:, 0] = 6.
        torch.testing.assert_close(term._torques, expected, atol=1e-5, rtol=0)
        # Reset the real physics, then exercise managers and 39/82 observations.
        env.reset(seed=42)
        applied_steps = []
        apply_actions = term.apply_actions
        def count_apply_actions():
            applied_steps.append(1)
            return apply_actions()
        term.apply_actions = count_apply_actions
        with torch.inference_mode():
            for _ in range(32):
                obs, reward, _, _, _ = env.step(torch.zeros_like(action))
                assert obs['actor'].shape == (4, 39) and obs['critic'].shape == (4, 82)
                assert all(bool(torch.isfinite(obs[k]).all()) for k in ('actor', 'critic'))
                assert bool(torch.isfinite(reward).all())
        assert len(applied_steps) == 32 * 5
        result = dict(passed=True, paired_source_samples=1017, action_delay_steps=1,
                      mjlab_physics_dt_s=env.physics_dt, mjlab_policy_dt_s=env.step_dt,
                      mjlab_pd_substeps_per_action=len(applied_steps) // 32,
                      cpu_policy_dt_s=cpu.DT * cpu.DECIMATION,
                      simulated_action_delay_ms=2., microsecond_jitter_modeled=False,
                      cpu_first_action_applied_ms=2., cpu_second_action_applied_ms=12.,
                      cpu_reset=True, gpu_partial_reset=True, gpu_envs=4, finite_policy_steps=32,
                      scope='finite timing/physics checks only; no learning or behavior acceptance')
    finally:
        env.close()

    cfg = configs[0]
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        assert env.physics_dt == .002 and env.step_dt == .01
        env.reset(seed=42)
        term = env.action_manager.get_term('pd')
        offset = term.nominal_offset.clone()
        nominal = term.nominal_positions.clone()
        legs, wheels = term._leg_action_ids, term._wheel_action_ids
        assert bool((offset[:, legs].abs() <= math.radians(5)).all())
        assert bool((offset[:, legs] != 0.).all())
        assert torch.unique(offset[:, legs]).numel() == 16
        assert not bool(offset[:, wheels].any())
        base = nominal - offset
        torch.testing.assert_close(base[:, legs], torch.tensor(
            cfg.actions['pd'].nominal_leg_positions, device=env.device).expand(4, -1))
        # At the original pose, zero action must command the sampled PD offset.
        env.command_manager.get_term('standup').ground_start[:] = False
        env.sim.data.qpos[:, term._motor_q_adr] = base
        env.sim.data.qvel[:, term._motor_v_adr] = 0.
        term.process_actions(torch.zeros((4, 6), device=env.device))
        term.apply_actions()
        expected = torch.zeros_like(offset)
        expected[:, legs] = (cfg.actions['pd'].leg_stiffness * term.kp_scale[:, legs]
                             * offset[:, legs] * term.motor_strength[:, legs])
        torch.testing.assert_close(term._torques, expected, atol=5e-6, rtol=1e-5)
        from ct_mjlab.envs.chuanliantui.multimode_observations import get_state
        sensors = get_state(env)
        torch.testing.assert_close(sensors.raw_sensors()[:, 6:12],
                                   -offset[:, sensors.action_order], atol=1e-7, rtol=0)
        # At the shifted reference both the raw relative angles and zero-action PD vanish.
        env.sim.data.qpos[:, term._motor_q_adr] = nominal
        term.apply_actions()
        torch.testing.assert_close(term._torques, torch.zeros_like(offset), atol=1e-5, rtol=0)
        torch.testing.assert_close(sensors.raw_sensors()[:, 6:12], torch.zeros_like(offset))
        samples = []
        for _ in range(32):
            term.reset()
            samples.append(torch.stack((term.action_delay_steps.clone(),
                                        term.wheel_action_delay_steps.clone()), dim=1))
        sampled = torch.cat(samples)
        assert set(sampled[:, 0].tolist()) == set(sampled[:, 1].tolist()) == {3, 4, 5}
        assert bool((sampled[:, 0] != sampled[:, 1]).any())
        before = term.action_delay_steps.clone(), term.wheel_action_delay_steps.clone()
        term.reset(torch.tensor([0, 2], device=env.device))
        torch.testing.assert_close(term.nominal_offset, offset, rtol=0, atol=0)
        torch.testing.assert_close(term.nominal_positions, nominal, rtol=0, atol=0)
        for actual, previous in zip((term.action_delay_steps, term.wheel_action_delay_steps), before):
            torch.testing.assert_close(actual[[1, 3]], previous[[1, 3]])

        # Cover each discrete lag through PD at a fixed state, including old-action holding.
        term.reset()
        term.action_delay_steps[:] = torch.tensor([3, 4, 5, 3], device=env.device)
        term.wheel_action_delay_steps[:] = torch.tensor([5, 3, 4, 5], device=env.device)
        term.kp_scale.fill_(1.); term.kd_scale.fill_(1.); term.motor_strength.fill_(1.)
        env.command_manager.get_term('standup').ground_start[:] = False
        env.sim.data.qpos[:, term._motor_q_adr] = term.nominal_positions
        env.sim.data.qvel[:, term._motor_v_adr] = 0.
        action = torch.full((4, 6), .1, device=env.device)
        delays = term.action_delay_steps[:, None].expand_as(action).clone()
        delays[:, term._wheel_action_ids] = term.wheel_action_delay_steps[:, None]
        magnitude = torch.full_like(action, term.cfg.leg_stiffness * term.cfg.leg_position_scale * .1)
        magnitude[:, term._wheel_action_ids] = term.cfg.wheel_stiffness * term.cfg.wheel_velocity_scale * .1
        term.process_actions(action)
        for tick in range(6):
            term.apply_actions()
            expected = magnitude * (tick >= delays)
            torch.testing.assert_close(term._torques, expected, atol=1e-5, rtol=0)
        term.process_actions(action * 2)
        for tick in range(6):
            term.apply_actions()
            expected = magnitude * (1. + (tick >= delays).float())
            torch.testing.assert_close(term._torques, expected, atol=1e-5, rtol=0)
        # Real training managers/observations still run with random per-episode delays.
        env.reset(seed=42)
        with torch.inference_mode():
            for _ in range(32):
                obs, reward, _, _, _ = env.step(torch.zeros_like(action))
                assert obs['actor'].shape == (4, 39) and obs['critic'].shape == (4, 82)
                assert all(bool(torch.isfinite(obs[k]).all()) for k in ('actor', 'critic'))
                assert bool(torch.isfinite(reward).all())
        result.update(training_action_delay_ms=[6, 8, 10], training_sampling='per episode, independent leg/wheel groups',
                      training_nominal_offsets_rad=offset.cpu().tolist(), nominal_offset_pd_observation_consistent=True,
                      nominal_offset_fixed_across_resets=True, playback_nominal_offset_zero=True,
                      training_sampled_group_pairs=sampled.cpu().tolist(), training_first_use_and_hold=True,
                      training_partial_reset_sampling=True, training_finite_policy_steps=32)
    finally:
        env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'training_sampled_group_pairs'}, indent=2))


if __name__ == '__main__':
    main()
