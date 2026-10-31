#!/usr/bin/env python3
"""Check reset-relative wheel-angle slots in the 39D profile; no learning."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from mjlab.managers.event_manager import EventTermCfg
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations as observations
from ct_mjlab.envs.chuanliantui.huananhu_observations import HuananhuObservationState
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.wheel_angles import OBSERVATION_CONTRACT, WHEEL_ANGLES
from ct_mjlab.smp.ppo import SmpCommandCfg, reset_pose


@torch.inference_mode()
def check(args):
    xml_hash = hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    assert not cfg.observations['actor'].terms['proprioception'].params['cfg'].wheel_angle_displacement
    cfg.auto_reset = False
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.commands['standup'] = SmpCommandCfg()
    cfg.events['reset_pose'] = EventTermCfg(func=reset_pose, mode='reset')
    observations.configure_wheel_angle_observations(cfg)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                   num_envs=20, device=args.device)
    try:
        raw = env.unwrapped
        raw.reset(seed=42)
        state = observations.get_state(raw)
        assert state.cfg.wheel_angle_displacement and not state.cfg.noise and not state.cfg.delay
        ids = torch.arange(raw.num_envs, device=raw.device)
        adr = state._motor_q_adr[4:6]
        origin = state.wheel_angle_origin.clone()
        increments = torch.tensor([[2., -2.], [-2., 2.], [2., 2.], [0., 0.],
                                   [20*torch.pi, -20*torch.pi]], device=raw.device).repeat(4, 1)
        raw.sim.data.qpos[:, adr] = origin + increments
        raw.sim.forward()
        state.sample_substep()
        actor = state.actor()
        critic = state.critic()
        assert actor.shape == (20, 39) and critic.shape == (20, 82)
        torch.testing.assert_close(actor[:, 14:16], increments, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(critic[:, 14:16], increments, atol=2e-6, rtol=2e-6)
        # 独立调用旧传感器公式；其余37个actor通道和80个critic通道不变。
        saved_cfg = state.cfg
        history = state.delay.history.clone()
        state.cfg = replace(saved_cfg, wheel_angle_displacement=False)
        state.delay.history[:, :, 10:12] = 0.
        old_actor, old_critic = state.actor(), state.critic()
        state.cfg = saved_cfg
        state.delay.history.copy_(history)
        for current, old in ((actor, old_actor), (critic, old_critic)):
            keep = torch.ones(current.shape[-1], dtype=torch.bool, device=raw.device)
            keep[14:16] = False
            torch.testing.assert_close(current[:, keep], old[:, keep], atol=0, rtol=0)
        baseline = HuananhuObservationState.raw_sensors(state)
        assert not baseline[:, 10:12].any()
        torch.testing.assert_close(state.raw_sensors()[:, 16:18], baseline[:, 16:18], atol=0, rtol=0)
        # 观测读取不改物理、PD队列或轮角原点。
        action = raw.action_manager.get_term('pd')
        before = [x.clone() for x in (raw.sim.data.qpos, raw.sim.data.qvel,
                                      action._action_history, state.wheel_angle_origin)]
        torch.testing.assert_close(state.actor(), actor, atol=0, rtol=0)
        for current, old in zip((raw.sim.data.qpos, raw.sim.data.qvel,
                                 action._action_history, state.wheel_angle_origin), before):
            torch.testing.assert_close(current.clone(), old, atol=0, rtol=0)
        raw.sim.data.qpos[:, :2] += 10.
        raw.sim.forward()
        torch.testing.assert_close(state.actor(), actor, atol=0, rtol=0)
        # 两轮分别保留多圈位移，不取平均，也不按2π回绕；沿用位置槽±100限幅。
        raw.sim.data.qpos[:, adr] = origin + 125.
        state.sample_substep()
        assert torch.equal(state.actor()[:, 14:16], torch.full_like(increments, 100.))
        raw.sim.data.qpos[:, adr] = origin - 125.
        state.sample_substep()
        assert torch.equal(state.actor()[:, 14:16], torch.full_like(increments, -100.))
        # 新字段进入原位置组的历史，actor延迟/噪声与critic即时真值分别核对。
        state.cfg = replace(saved_cfg, noise=True, delay=True)
        state.delay.lags[:] = 0
        state.delay.lags[:, 2] = 2
        for tick in range(4):
            raw.sim.data.qpos[:, adr] = origin + tick + 1.
            state.sample_substep()
        state.noise[:] = 0.
        state.noise[:, 10:12] = state.noise.new_tensor([.01, -.02])
        delayed = state.delay.history[(state.delay.head-2) % state.delay.history.shape[0], :, 10:12]
        torch.testing.assert_close(state.actor()[:, 14:16], delayed+state.noise[:, 10:12], atol=0, rtol=0)
        torch.testing.assert_close(state.critic()[:, 14:16], torch.full_like(increments, 4.), atol=2e-6, rtol=2e-6)
        state.cfg = saved_cfg
        state.delay.lags[:] = 0
        state.sample_substep()
        keep_origin = state.wheel_angle_origin[1:].clone()
        keep_actor = state.actor()[1:].clone()
        keep_history = state.delay.history[:, 1:].clone()
        observations.reset_observations(raw, ids[:1], cfg=saved_cfg)
        assert not state.actor()[0, 14:16].any()
        assert not state.delay.history[:, 0, 10:12].any()
        torch.testing.assert_close(state.wheel_angle_origin[1:], keep_origin, atol=0, rtol=0)
        torch.testing.assert_close(state.actor()[1:], keep_actor, atol=0, rtol=0)
        torch.testing.assert_close(state.delay.history[:, 1:], keep_history, atol=0, rtol=0)
        # 真实20环境子步：每拍核对角差和轮速，无第40维或LQR速度旁路。
        raw.reset(seed=43)
        obs = raw.observation_manager.compute(update_history=True)
        assert obs['actor'].shape == (20, 39) and obs['critic'].shape == (20, 82)
        assert not obs['actor'][:, 14:16].any()
        for tick in range(50):
            expected = (state.wheel_angles()-state.wheel_angle_origin).clamp(-100., 100.)
            torch.testing.assert_close(obs['actor'][:, 14:16], expected, atol=0, rtol=0)
            torch.testing.assert_close(obs['critic'][:, 14:16], expected, atol=0, rtol=0)
            sensors = state.raw_sensors()
            torch.testing.assert_close(obs['actor'][:, 20:22], sensors[:, 16:18].clamp(-200., 200.)*.1, atol=0, rtol=0)
            assert obs['actor'].shape == (20, 39) and obs['critic'].shape == (20, 82)
            assert not hasattr(raw, '_bc_reference_observer')
            obs, _, _, _ = env.step(torch.zeros(20, 6, device=raw.device))
            assert all(torch.isfinite(value).all() for value in obs.values())
            if tick < 19:
                assert not raw.sim.data.qfrc_actuator[:, action._motor_v_adr].any()
        assert hashlib.sha256(MODEL_XML.read_bytes()).hexdigest() == xml_hash
        report = dict(passed=True, optimizer_updates=0, num_envs=20, duration_s=.5,
            observation_contract=OBSERVATION_CONTRACT, wheel_angles=WHEEL_ANGLES,
            actor_dim=39, critic_dim=82, forward_velocity_channel_removed=True,
            no_LQR_reference_observer_created=True, old_default_wheel_slots_still_zero=True,
            native_axes_forward_reverse_turn_multiturn=True, clipping_checked=True,
            other_actor37_and_critic80_unchanged=True, wheel_velocity_unchanged=True,
            position_group_delay_and_noise_checked=True, critic_uses_current_angles=True,
            repeated_reads_do_not_write_physics_or_pd_fifo=True,
            world_translation_invariant=True, partial_reset_and_delay_history_isolated=True,
            full_reset_zero=True, startup_actual_force_zero=True, xml_sha256=xml_hash,
            scope='Observation arithmetic and finite simulation only; no trained behavior, PPO integration or hardware acceptance')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    check(parser.parse_args())
