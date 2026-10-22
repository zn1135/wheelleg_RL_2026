#!/usr/bin/env python3
"""Finite candidate drive-response checks; test tau is hypothetical, no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))


def reference(commands, tau, dt):
    # Independent closed-form convolution, including both signs and zero input.
    decay = np.exp(-dt / tau)
    return np.array([(1. - decay) * np.sum(
        commands[:i + 1] * decay ** np.arange(i, -1, -1)[:, None], axis=0)
        for i in range(len(commands))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-cpu', type=Path, help='Pre-change source snapshot for exact zero-tau comparison')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output path')
    import mj_sim2sim_huananhu_multimode as cpu
    from ct_mjlab.envs.chuanliantui.multimode_env_cfg import make_multimode_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    tau = .004  # Check-only value, never written to the shared configuration.
    profile = json.loads(cpu.MODEL_ASSET.with_name('h7_timing.json').read_text())
    assert profile['simulation']['motor_response_time_constant_s'] == 0.
    assert 'unmeasured' in profile['simulation']['motor_response_calibration_status']
    for play in (False, True):
        cfg = make_multimode_env_cfg(play=play)
        assert cfg.actions['pd'].motor_response_time_constant_s == 0.
        assert cfg.actions['pd'].action_delay_s == .002

    zero = cpu.Playback()
    baseline = None
    if args.baseline_cpu:
        old = types.ModuleType('motor_response_baseline')
        old.__file__ = cpu.__file__  # Resolve the same assets; source is the archived snapshot.
        exec(compile(args.baseline_cpu.read_text(), str(args.baseline_cpu), 'exec'), old.__dict__)
        baseline = old.Playback()
    for i in range(16):
        a = np.full(6, .01 * np.sin(i), dtype=np.float32)
        zero.step(lambda obs: a)
        np.testing.assert_array_equal(zero.data.ctrl[zero.motor_ids], zero._motor_response_torque)
        if baseline:
            baseline.step(lambda obs: a)
            for field in ('qpos', 'qvel', 'ctrl'):
                np.testing.assert_array_equal(getattr(zero.data, field), getattr(baseline.data, field))

    playback = cpu.Playback(motor_response_time_constant_s=tau)
    commands, applied = [], []
    controls, mj_step = playback.controls, cpu.mujoco.mj_step
    def fixed_state_controls(action):
        playback.data.qpos[playback.q] = playback.nominal
        playback.data.qvel[playback.v] = 0.
        command, spring = controls(action)
        commands.append(command.copy())
        return command, spring
    def capture_step(model, data):
        applied.append(data.ctrl[playback.motor_ids].copy())
        return mj_step(model, data)
    playback.controls = fixed_state_controls
    a = np.array([.1, -.1, .1, .05, -.05, -.1], dtype=np.float32)
    with patch.object(cpu.mujoco, 'mj_step', side_effect=capture_step):
        playback.step(lambda obs: a)
        playback.step(lambda obs: -a)
    expected = reference(np.array(commands), tau, cpu.DT)
    np.testing.assert_allclose(applied, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(applied[0], np.zeros(6))
    assert 0 < applied[1][0] < commands[1][0]
    playback.reset()
    np.testing.assert_array_equal(playback._motor_response_torque, np.zeros(6))
    playback._motor_response_torque.fill(2.)
    playback.ground_start = True
    playback.step(lambda obs: a)
    np.testing.assert_array_equal(playback.data.ctrl[playback.motor_ids], np.zeros(6))
    for invalid in (-.001, float('nan'), float('inf')):
        try:
            cpu.Playback(motor_response_time_constant_s=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid motor response tau accepted')

    cfg = make_multimode_env_cfg(play=True)
    cfg.scene.num_envs = 4
    cfg.actions['pd'].motor_response_time_constant_s = tau
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env.reset(seed=42)
        term = env.action_manager.get_term('pd')
        command_term = env.command_manager.get_term('standup')
        command_term.ground_start[:] = False
        env.sim.data.qpos[:, term._motor_q_adr] = term.nominal_positions
        env.sim.data.qvel[:, term._motor_v_adr] = 0.
        action = torch.tensor(a, device=env.device).repeat(4, 1)
        term.reset()
        gpu_commands, gpu_applied, submitted = [], [], []
        original_submit = term._entity.set_joint_effort_target
        def capture_submit(efforts, **kwargs):
            submitted.append(efforts.detach().cpu().numpy().copy())
            return original_submit(efforts, **kwargs)
        with patch.object(term._entity, 'set_joint_effort_target', side_effect=capture_submit):
            for value in (action, -action):
                term.process_actions(value)
                for _ in range(10):
                    term.apply_actions()
                    gpu_commands.append(term._torques[0].cpu().numpy().copy())
                    gpu_applied.append(term._motor_response_torque[0].cpu().numpy().copy())
        gpu_expected = reference(np.array(gpu_commands), tau, env.physics_dt)
        np.testing.assert_allclose(gpu_applied, gpu_expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(gpu_applied, applied, atol=1e-5, rtol=1e-5)
        # Zero velocity means the existing wheel speed factor is exactly one.
        np.testing.assert_allclose(np.array(submitted)[:, 0], gpu_expected, atol=2e-6, rtol=2e-6)
        before = term._motor_response_torque.clone()
        ids = torch.tensor([0, 2], device=env.device)
        term.reset(ids)
        torch.testing.assert_close(term._motor_response_torque[ids], torch.zeros_like(before[ids]))
        torch.testing.assert_close(term._motor_response_torque[[1, 3]], before[[1, 3]])
        term.process_actions(action)
        term.apply_actions()
        torch.testing.assert_close(term._motor_response_torque[ids], torch.zeros_like(before[ids]), atol=1e-5, rtol=0)
        command_term.ground_start[:] = True
        env.episode_length_buf[:] = 0
        term.apply_actions()
        torch.testing.assert_close(term._motor_response_torque, torch.zeros_like(before))
        env.reset(seed=42)
        with torch.inference_mode():
            for _ in range(32):
                obs, reward, _, _, _ = env.step(torch.zeros_like(action))
                assert obs['actor'].shape == (4, 39) and obs['critic'].shape == (4, 82)
                assert all(bool(torch.isfinite(obs[k]).all()) for k in ('actor', 'critic'))
                assert bool(torch.isfinite(reward).all())
        result = dict(passed=True, default_tau_s=0., candidate_tau_s=tau, tau_is_measured=False,
            cpu_gpu_response_max_difference=float(np.max(abs(np.array(gpu_applied) - applied))),
            cpu_convolution_max_error=float(np.max(abs(np.array(applied) - expected))),
            gpu_convolution_max_error=float(np.max(abs(np.array(gpu_applied) - gpu_expected))),
            zero_tau_exact_baseline_steps=16 if baseline else None,
            baseline_sha256=hashlib.sha256(args.baseline_cpu.read_bytes()).hexdigest() if baseline else None,
            partial_reset=True, disabled_gate_clears_response=True, invalid_tau_rejected=True,
            gpu_envs=4, finite_policy_steps=32,
            scope='finite candidate response and interface checks; no training, calibration or behavior acceptance')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
