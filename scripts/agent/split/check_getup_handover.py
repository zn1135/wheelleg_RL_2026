#!/usr/bin/env python3
"""Check the actual getup PD handover at 2ms resolution without learning."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split.action import HANDOVER_DURATION_S
from ct_mjlab.envs.chuanliantui_split.runner import contracts, validate
from scripts.agent.split.check_chuanliantui_two_policy import rejected, digest


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    env, _ = task_registry.make_env('chuanliantui_getup', play=True,
        num_envs=4, seed=43, device=args.device)
    raw = env.unwrapped
    try:
        pd = raw.action_manager.get_term('pd')
        state = raw.command_manager.get_term('standup')
        legs, wheels = pd._leg_action_ids, pd._wheel_action_ids
        leg_q = pd._motor_q_adr[legs]
        assert HANDOVER_DURATION_S == .1 and raw.physics_dt == .002
        action = torch.full((4, 6), .2, device=raw.device)
        action[:, wheels] = torch.tensor([.3, -.2], device=raw.device)
        start = pd.nominal_positions[:, legs] + .25
        raw.sim.data.qpos[:, leg_q] = start
        raw.sim.data.qvel.zero_()
        state.ground_start.fill_(True)
        raw.episode_length_buf.fill_(19)
        pd.process_actions(action)
        for _ in range(5):
            pd.apply_actions()
            assert not pd.torques.any() and not pd._handover_started.any()
        assert not pd._handover_ticks.any()
        raw.episode_length_buf.fill_(20)
        trace = []
        for tick in range(56):
            if tick % 5 == 0:
                # 第六拍改变网络目标，插值终点必须跟随当前已延迟的动作。
                action[:, legs] = -.1 if tick >= 25 else .2
                pd.process_actions(action)
            fraction = min(tick*.002/.1, 1.)
            blend = fraction*fraction*(3.-2.*fraction)
            network = pd.nominal_positions[:, legs]+.5*pd._control_actions[:, legs]
            expected_target = start+blend*(network-start) if tick < 50 else network
            pd.apply_actions()
            expected_torque = 60.*(expected_target-raw.sim.data.qpos[:, leg_q])
            assert torch.allclose(pd.torques[:, legs], expected_torque, atol=2e-5)
            assert torch.allclose(pd.torques[:, wheels], action[:, wheels], atol=1e-6)
            assert torch.equal(pd.raw_action, action)
            assert torch.equal(pd._handover_start_angles, start)
            trace.append(dict(time_since_release_s=tick*.002, blend=blend,
                leg_torque_Nm=float(pd.torques[0, 0])))
        assert not trace[0]['leg_torque_Nm'] and pd._handover_ticks.eq(56).all()
        cached_ticks = pd._handover_ticks.clone()
        cached_start = pd._handover_start_angles.clone()
        pd.reset(torch.tensor([0], device=raw.device))
        assert not pd._handover_started[0] and pd._handover_ticks[0] == 0
        assert not pd._handover_start_angles[0].any()
        assert torch.equal(pd._handover_ticks[1:], cached_ticks[1:])
        assert torch.equal(pd._handover_start_angles[1:], cached_start[1:])
        # 局部复位后等待期不推进计时；重新释放必须读取新角度。
        raw.episode_length_buf[0] = 0
        pd.process_actions(action)
        pd.apply_actions()
        assert pd._handover_ticks[0] == 0 and not pd.torques[0].any()
        raw.sim.data.qpos[0, leg_q] = start[0]+.1
        raw.episode_length_buf[0] = 20
        pd.apply_actions()
        assert torch.equal(pd._handover_start_angles[0], start[0]+.1)
        assert not pd.torques[0, legs].any()
        # 动作仍走原2子步FIFO；接管自身在FIFO之后，不能再额外排队。
        pd.reset()
        raw.sim.data.qpos[:, leg_q] = pd.nominal_positions[:, legs]
        pd.action_delay_steps.fill_(2)
        action[:, legs] = .2
        pd.process_actions(action)
        for tick in range(5):
            pd.apply_actions()
            fraction = tick*.002/.1
            blend = fraction*fraction*(3.-2.*fraction)
            expected = 0. if tick < 2 else 4.8*blend
            assert torch.allclose(pd.torques[:, legs],
                torch.full((4, 4), expected, device=raw.device), atol=2e-5)
        # 位置差首次为零，但原速度阻尼继续存在。
        pd.reset()
        raw.sim.data.qvel[:, pd._motor_v_adr[legs]] = .2
        pd.process_actions(action)
        pd.apply_actions()
        assert torch.allclose(pd.torques[:, legs], torch.full((4, 4), -.4, device=raw.device))
        # 非地面起立不触发过渡，保持原PD行为。
        pd.reset()
        raw.sim.data.qvel.zero_()
        pd.action_delay_steps.zero_()
        state.ground_start.fill_(False)
        pd.process_actions(action)
        pd.apply_actions()
        assert not pd._handover_started.any()
        assert torch.allclose(pd.torques[:, legs], torch.full((4, 4), 4.8, device=raw.device), atol=2e-5)
        infos = contracts(state.cfg)
        infos['standup_curriculum'] = state.checkpoint_state()
        for key, value in (('handover_duration_s', .2), ('handover_dt_s', .01),
                           ('handover_curve', 'linear'), ('handover_start', 'nominal')):
            bad = copy.deepcopy(infos)
            bad['action_filter'][key] = value
            rejected(lambda: validate(bad, state.cfg))
        report = dict(verified=True, parameter_updates=0, duration_s=.1, physics_dt_s=.002,
            first_target_equals_measured=True, start_sampled_once=True,
            full_network_target_at_100ms=True, changing_network_target_checked=True,
            raw_actions_wheels_and_damping_preserved=True, original_fifo_preserved=True,
            partial_reset_and_new_episode_recapture_checked=True, non_ground_bypassed=True,
            metadata_mismatch_rejected=True, trace=trace, source_sha256=digest(__file__),
            scope='Actual PD with controlled state/actions; finite interface check, no behavior acceptance')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Check cannot train')):
        check(parser.parse_args())
