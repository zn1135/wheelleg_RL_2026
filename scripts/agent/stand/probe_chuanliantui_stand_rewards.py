#!/usr/bin/env python3
"""Read actual reward rates before first termination; no learning or acceptance."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand.mdp import supported_pose_mask
from scripts.agent.stand.assess_chuanliantui_stand import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-mode', choices=('near', 'ground'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand', play=True)
    cfg.commands['standup'].ground_start_probability = float(args.start_mode == 'ground')
    cfg.auto_reset = False
    wrapped, _ = task_registry.make_env('chuanliantui_stand', env_cfg=cfg,
        train_cfg=agent, num_envs=20, seed=42, device='cuda:0')
    env = wrapped.unwrapped
    try:
        runner, _ = task_registry.make_alg_runner(wrapped, 'chuanliantui_stand',
            train_cfg=agent, device='cuda:0')
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True,
                    map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        obs, _ = env.reset(seed=42)
        state = env.command_manager.get_term('standup')
        alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        rates, phases, live, actions, bodies, resets = [], [], [], [], [], []
        first_end = np.full(env.num_envs, np.nan)
        names = list(env.reward_manager.active_terms)
        with torch.inference_mode():
            for tick in range(500):
                # Record eligibility before step; rewards precede command.advance().
                phases.append(torch.stack((state.settle_required, state.recover,
                    state.fallen, state.quiet_ready, supported_pose_mask(env)), -1).cpu().numpy())
                live.append(alive.cpu().numpy().copy())
                action = policy(obs)
                obs, _, terminated, timeout, _ = env.step(action)
                # mjlab stores raw_value * weight, without the policy-dt factor.
                rate = env.reward_manager._step_reward.detach().cpu().numpy().copy()
                rates.append(rate)
                actions.append(action.cpu().numpy().copy())
                data = env.scene['robot'].data
                bodies.append(torch.cat((data.root_link_pos_w,
                    data.root_link_lin_vel_b, data.projected_gravity_b), -1).cpu().numpy().copy())
                ended = alive & (terminated | timeout)
                first_end[ended.cpu().numpy()] = (tick + 1) * env.step_dt
                alive &= ~(terminated | timeout)
                if not bool(alive.any()):
                    break
                ids = (terminated | timeout).nonzero(as_tuple=False).flatten()
                if len(ids):
                    # Freeze its first result; exclude all post-reset data from statistics.
                    resets.append({'step': tick + 1, 'env_ids': ids.cpu().tolist()})
                    env.reset(env_ids=ids)
                    obs = env.get_observations()
        rates, phases, live = np.asarray(rates), np.asarray(phases), np.asarray(live)
        assert rates.shape[-1] == len(names)
        assert np.isfinite(rates[live]).all()
        phase_names = ('settle_required', 'recover', 'fallen', 'quiet_ready', 'supported_pose')
        summaries = {}
        for label, mask in [('all_alive', live), ('recovery', live & phases[:, :, 0]),
                            ('standing', live & ~phases[:, :, 0] & phases[:, :, 4])]:
            selected = rates[mask]
            summaries[label] = {'environment_seconds': float(mask.sum() * env.step_dt),
                'weighted_rate_mean': dict(zip(names, selected.mean(0).tolist())) if len(selected) else {},
                'total_rate_mean': float(selected.sum(-1).mean()) if len(selected) else None}
        np.savez_compressed(args.output / 'reward-trace.npz', weighted_reward_rates=rates,
            reward_names=np.asarray(names), phase_before_step=phases,
            phase_names=np.asarray(phase_names), alive_before_step=live,
            action=np.asarray(actions), body_after_step=np.asarray(bodies), first_end_s=first_end,
            policy_time_s=(np.arange(len(rates)) + 1) * env.step_dt)
        report = {'scope': __doc__, 'checkpoint': identity(args.checkpoint),
            'source': identity(Path(__file__)), 'start_mode': args.start_mode,
            'num_envs': 20, 'seed': 42, 'physics_dt_s': env.physics_dt,
            'policy_dt_s': env.step_dt, 'steps': len(rates),
            'resets_after_first_result_frozen': resets,
            'terminated_environments': int(np.isfinite(first_end).sum()),
            'rate_units': 'raw reward * weight per second; sum * .01 gives policy-step reward',
            'summaries': summaries}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == '__main__':
    main()
