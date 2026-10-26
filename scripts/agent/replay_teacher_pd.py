#!/usr/bin/env python3
"""Finite teacher-derived PD target replay through real student physics; no training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.ppo import SmpCommandCfg, inverted, reset_pose

ROOT = Path(__file__).resolve().parents[2]
DELAYS = (0., .002, .006, .008, .010)


@torch.inference_mode()
def replay(args):
    dataset = MotionDataset(args.dataset)
    if dataset.xml_sha256 != hashlib.sha256(MODEL_XML.read_bytes()).hexdigest():
        raise ValueError('Replay XML differs from the demonstrations')
    episodes = [episode for episode in dataset.episodes if episode['behavior'] == 'standup']
    if not episodes:
        raise ValueError('Replay requires qualified normal-ground standup episodes')
    records = []
    for episode in episodes:
        with np.load(episode['path']) as data:
            records.append({key: torch.as_tensor(data[key].copy(), device=args.device).float()
                            for key in ('initial_qpos', 'initial_qvel', 'motor_q', 'motor_dq',
                                        'torque', 'root', 'ground_height')})
    # 每个延迟组重放相同的完整初态，避免初态差异混入门控/延迟对照。
    cases = [(delay, index) for delay in DELAYS for index in range(len(records))]
    count = len(cases)
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.seed = 42
    cfg.auto_reset = False
    # 只复用SMP初态/命令/终止条件，不依赖或加载评分器权重。
    cfg.commands['standup'] = SmpCommandCfg()
    cfg.events['reset_pose'] = EventTermCfg(func=reset_pose, mode='reset')
    cfg.terminations['inverted'] = TerminationTermCfg(func=inverted)
    cfg.rewards = {}  # 只做物理回放，不加载/更新SMP评分统计或PPO。
    cfg.actions['pd'].action_delay_s = None
    cfg.actions['pd'].randomize_delays = False
    cfg.actions['pd'].action_delay_choices_s = DELAYS
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                    num_envs=count, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(optimizer_updates=0, policy_dt=.01, physics_dt=.002,
        teacher_physics_dt=.001, delay_s=list(DELAYS), num_episodes=len(episodes),
        xml_sha256=dataset.xml_sha256,
        dataset_fingerprint=dataset.fingerprint,
        teacher_ground_start_hold_s=dataset.ground_start_hold_s,
        scope='100Hz recorded reference targets with live 500Hz student PD feedback; not a closed-loop LQR controller or learned PPO',
        timing_limit='Teacher samples are post-step; their torque drove the preceding 1ms tick. First 10ms sample supplies the first target; phase error is not removed.',
        limitations='Nominal play physics; paired leg/wheel delays only, no motor randomization or observation noise. 100Hz contact samples cannot certify 1ms v3.1 balance.',
        cases=[])
    try:
        raw = env.unwrapped
        action = raw.action_manager.get_term('pd')
        assert raw.step_dt == .01 and raw.physics_dt == .002
        legs, wheels = action._leg_action_ids, action._wheel_action_ids
        reference_q = torch.stack([records[index]['motor_q'] for _, index in cases], 1)
        reference_dq = torch.stack([records[index]['motor_dq'] for _, index in cases], 1)
        reference_torque = torch.stack([records[index]['torque'] for _, index in cases], 1)
        reference_height = torch.stack([records[index]['root'][:, 2]-records[index]['ground_height']
                                        for _, index in cases], 1)
        targets = torch.zeros_like(reference_q)
        leg_target = reference_q[:, :, legs]+(
            reference_torque[:, :, legs]+action.cfg.leg_damping*reference_dq[:, :, legs]
        )/action.cfg.leg_stiffness
        targets[:, :, legs] = (leg_target-action.nominal_positions[:, legs])/action.cfg.leg_position_scale
        targets[:, :, wheels] = (reference_dq[:, :, wheels]+
            reference_torque[:, :, wheels]/action.cfg.wheel_stiffness)/action.cfg.wheel_velocity_scale
        weight = float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        steps = min(round(args.duration/raw.step_dt), len(targets))
        if steps < 1:
            raise ValueError('Replay duration must include at least one policy step')
        for hold in (.2, 0.):
            action.cfg.ground_start_hold_s = hold
            action._ground_hold_steps = math.ceil(hold/raw.step_dt-1e-9)
            raw.reset(seed=42)
            qpos = torch.stack([records[index]['initial_qpos'] for _, index in cases])
            qvel = torch.stack([records[index]['initial_qvel'] for _, index in cases])
            qpos[:, :2] = raw.scene.env_origins[:, :2]
            raw.sim.data.qpos.copy_(qpos)
            raw.sim.data.qvel.copy_(qvel)
            raw.sim.forward()
            initial_qpos = raw.sim.data.qpos.cpu().numpy().copy()
            delay_steps = torch.tensor([round(delay/raw.physics_dt) for delay, _ in cases],
                                      device=args.device)
            action.action_delay_steps.copy_(delay_steps)
            action.wheel_action_delay_steps.copy_(delay_steps)
            active_at_reset = raw.command_manager.get_term('standup').policy_active.cpu().tolist()
            live = torch.ones(count, dtype=torch.bool, device=args.device)
            buffers = {key: [] for key in ('qpos', 'torque', 'height', 'tilt', 'balance', 'alive', 'done')}
            for tick in range(steps):
                _, _, done, _ = env.step(torch.where(live[:, None], targets[tick], 0.))
                raw.sim.forward()
                robot = raw.scene['robot'].data
                height = robot.root_link_pos_w[:, 2]-raw.scene.env_origins[:, 2]
                g = robot.projected_gravity_b
                tilt = torch.maximum(torch.atan2(-g[:, 1], -g[:, 2]).abs(),
                                     torch.asin(g[:, 0].clamp(-1, 1)).abs())
                base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
                wheels_force = -raw.scene['wheel_contact'].data.force[:, :, 2]
                balance = ((tilt <= math.radians(35)) & (base <= 1.) &
                    (wheels_force >= max(.1*weight, 5.)).all(-1) &
                    (wheels_force.sum(-1) >= .8*weight) & (wheels_force.sum(-1) <= 1.2*weight) &
                    (robot.root_link_ang_vel_b[:, :2].norm(dim=-1) <= .5))
                values = dict(qpos=raw.sim.data.qpos, torque=action.torques, height=height,
                              tilt=tilt, balance=balance & live, alive=live, done=done.bool())
                for key, value in values.items():
                    buffers[key].append(value.cpu().numpy().copy())
                live &= ~done.bool()
                if not live.any():
                    break
                if done.any():
                    raw.reset(env_ids=done.nonzero().flatten())
            arrays = {key: np.asarray(value) for key, value in buffers.items()}
            name = 'immediate' if hold == 0. else 'hold-020'
            np.savez_compressed(args.output/f'{name}.npz', **arrays, initial_qpos=initial_qpos,
                                initial_qvel=qvel.cpu().numpy(), reference_height=reference_height.cpu().numpy(),
                                target_actions=targets.cpu().numpy())
            for index, (delay, episode) in enumerate(cases):
                mask = arrays['alive'][:, index]
                height = arrays['height'][:, index][mask]
                length = len(height)
                first_torque = arrays['torque'][0, index]
                final_balance = length >= 100 and arrays['balance'][length-100:length, index].all()
                report['cases'].append(dict(hold_s=hold, delay_s=delay,
                    episode_id=episodes[episode]['episode_id'],
                    active_at_reset=active_at_reset[index],
                    first_policy_step_torque_max_nm=float(np.abs(first_torque).max()),
                    duration_s=length*raw.step_dt, terminated=bool(arrays['done'][:length, index].any()),
                    finite=bool(np.isfinite(arrays['qpos'][:length, index]).all()),
                    max_height_m=float(height.max()), final_height_m=float(height[-1]),
                    height_mae_m=float(np.abs(height-reference_height[:length, index].cpu().numpy()).mean()),
                    final_sampled_balance_1s=bool(final_balance)))
        sources = [Path(__file__), ROOT/'mjlab_training/src/ct_mjlab/smp/ppo.py',
                   ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_action.py']
        report['sources'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sources}
        (args.output/'replay.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    replay(parser.parse_args())


if __name__ == '__main__':
    main()
