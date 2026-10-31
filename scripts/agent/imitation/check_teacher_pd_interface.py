#!/usr/bin/env python3
"""Check real teacher torque requests against the student's physical PD interface."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def check(args):
    from ct_mjlab.envs import task_registry
    from ct_mjlab.smp.dataset import MotionDataset
    dataset = MotionDataset(args.dataset)
    records = {key: [] for key in ('qpos', 'qvel', 'motor_q', 'motor_dq', 'torque', 'time')}
    for episode in dataset.episodes:
        if episode['behavior'] != 'standup':
            continue
        with np.load(episode['path']) as data:
            for key in records:
                records[key].append(data[key])
    values = {key: torch.as_tensor(np.concatenate(value), device=args.device).float()
              for key, value in records.items()}
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.actions['pd'].action_delay_s = 0.  # 只隔离数值接口；不声称验证了真实延迟下的动态复现。
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                    num_envs=20, device=args.device)
    try:
        raw = env.unwrapped
        action = raw.action_manager.get_term('pd')
        raw.episode_length_buf.fill_(1000)  # 先核对接管后的接口，再单列前0.2秒门控差异。
        legs, wheels = action._leg_action_ids, action._wheel_action_ids
        errors, targets_outside = [], []
        for start in range(0, len(values['torque']), 20):
            ids = torch.arange(start, min(start+20, len(values['torque'])), device=args.device)
            ids = torch.cat((ids, ids[-1:].expand(20-len(ids))))
            q, dq, torque = values['motor_q'][ids], values['motor_dq'][ids], values['torque'][ids]
            raw.sim.data.qpos.copy_(values['qpos'][ids])
            raw.sim.data.qvel.copy_(values['qvel'][ids])
            torch.testing.assert_close(raw.sim.data.qpos[:, action._motor_q_adr], q, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(raw.sim.data.qvel[:, action._motor_v_adr], dq, atol=2e-6, rtol=2e-6)
            target = q[:, legs]+(torque[:, legs]+action.cfg.leg_damping*dq[:, legs])/action.cfg.leg_stiffness
            commands = torch.zeros_like(q)
            commands[:, legs] = (target-action.nominal_positions[:, legs])/action.cfg.leg_position_scale
            commands[:, wheels] = (dq[:, wheels]+torque[:, wheels]/action.cfg.wheel_stiffness)/action.cfg.wheel_velocity_scale
            outside = (target-action.reference_positions[:, legs]).abs() > 3.14
            targets_outside.append(outside)
            action.process_actions(commands)
            action.apply_actions()
            errors.append((action.torques-torque).abs())
        errors = torch.cat(errors)[:len(values['torque'])]
        early = values['time'] <= .2
        report = dict(numeric_interface_passed=bool(errors.max() < 1e-3),
            samples=len(errors), max_requested_torque_error_nm=float(errors.max()),
            fraction_with_error_over_1e_minus3=float((errors.amax(-1) > 1e-3).float().mean()),
            unclipped_leg_targets_outside_range_fraction=float(torch.cat(targets_outside).float().mean()),
            teacher_nonzero_torque_fraction_first_02s=float((values['torque'][early].abs().amax(-1) > 1e-3).float().mean()),
            ppo_active_torque_first_02s='zero by ground_start_hold_s',
            dynamic_replay_verified=False, optimizer_updates=0,
            scope='same stored state, inverse-PD algebra then actual apply_actions; no optimizer or physics trajectory replay',
            timestamp_note='teacher torque drove the preceding 1ms step; inverse actions here are diagnostics, not behavior-cloning targets',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    check(parser.parse_args())


if __name__ == '__main__':
    main()
