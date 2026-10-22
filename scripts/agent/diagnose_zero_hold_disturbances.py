#!/usr/bin/env python3
"""Measure actual zero-hold reward before/after training disturbances; no learning."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_rewards
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends


def run(args, arm):
    cfg, agent = task_registry.get_cfgs('chuanliantui', play=False)
    cfg.seed = args.seed
    cfg.scene.num_envs = 4 * args.envs_per_height
    cfg.auto_reset = False
    cfg.commands['standup'].zero_hold_probability = 1.
    cfg.commands['standup'].ground_start_probability = 1.
    cfg.events['reset_pose'].params['ground_probability'] = 1.
    cfg.commands['standup'].resampling_time_range = (1e9, 1e9)
    captures, events, live = [], [], None
    time_origin_step = 0
    original_reward = cfg.rewards['zero_command_hold'].func

    def capture_reward(env, *, name):
        state = env.command_manager.get_term('standup')
        robot = env.scene['robot'].data
        velocity = robot.root_link_lin_vel_b[:, :2]
        distance = (robot.root_link_pos_w[:, :2] - state.zero_hold_anchor_xy).norm(dim=-1)
        active = ((state.command[:, :2].abs() < .05).all(-1) & ~state.recover
                  & ~state.airborne & (state.jump_phase == 0)
                  & (robot.projected_gravity_b[:, 2] < -.9)
                  & (state.stable_time >= .5) & state.zero_hold_anchor_valid)
        vs = torch.exp(-velocity.square().sum(-1) / .005**2)
        ps = torch.exp(-distance.square() / .02**2)
        value = original_reward(env, name=name)
        torch.testing.assert_close(value, active.float() * vs * ps)
        captures.append({
            'time': float((env.common_step_counter - time_origin_step) * env.step_dt),
            'xy': robot.root_link_pos_w[:, :2].cpu().numpy().copy(),
            'speed': velocity.norm(dim=-1).cpu().numpy().copy(),
            'distance': distance.cpu().numpy().copy(),
            'velocity_score': vs.cpu().numpy().copy(),
            'position_score': ps.cpu().numpy().copy(),
            'active': active.cpu().numpy().copy(),
            'recover': state.recover.cpu().numpy().copy(),
            'anchor_valid': state.zero_hold_anchor_valid.cpu().numpy().copy(),
            'reward': value.cpu().numpy().copy(),
            'live': live.cpu().numpy().copy() if live is not None else np.ones(env.num_envs, bool),
        })
        return value

    cfg.rewards['zero_command_hold'].func = capture_reward

    def monitor_event(kind, original):
        def wrapped(env, env_ids, **kwargs):
            ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
            state = env._huananhu_randomization
            # 对照组仍消费相同随机数并保留事件时钟，仅撤销扰动的物理作用。
            before = (env.sim.data.qvel[ids].clone() if kind == 'velocity'
                      else state.local_wrench[ids].clone())
            original(env, env_ids, **kwargs)
            applied = (env.sim.data.qvel[ids, :2].clone() if kind == 'velocity'
                       else state.local_wrench[ids].clone())
            events.append({'time': float((env.common_step_counter - time_origin_step) * env.step_dt),
                           'kind': kind, 'ids': ids.cpu().tolist(),
                           'sampled': applied.cpu().tolist(), 'effect_enabled': arm == 'disturbed'})
            if arm == 'quiet_control':
                if kind == 'velocity':
                    env.sim.data.qvel[ids] = before
                else:
                    state.local_wrench[ids] = before
                    state.apply_local_wrench()
        return wrapped

    for key, kind in [('huananhu_velocity_push', 'velocity'), ('huananhu_local_wrench', 'wrench')]:
        term = cfg.events[key]
        term.func = monitor_event(kind, term.func)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = task_registry._task('chuanliantui').runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset(seed=args.seed)
        state = env.command_manager.get_term('standup')
        assert bool(state.ground_start.all())
        initial = env.sim.data.qpos.cpu().numpy().copy()
        initial_velocity = env.sim.data.qvel.cpu().numpy().copy()
        initial_physics = {name: getattr(env.sim.model, name).cpu().numpy().copy()
                           for name in ('body_mass', 'body_inertia', 'body_ipos',
                                        'geom_friction', 'dof_frictionloss', 'dof_damping')}
        time_origin_step = int(env.common_step_counter)
        height = torch.tensor([h for h in (.2165, .27, .34, .3664)
                               for _ in range(args.envs_per_height)], device=env.device)
        state.sampled[:, 2] = height
        state._command[:, 2] = height
        # 策略首次运行前刷新高度观测，零动作步骤同样保留在记录中。
        obs = env.observation_manager.compute(update_history=False)
        live = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        failed = torch.zeros_like(live)
        captures.clear()
        events.clear()
        with torch.inference_mode():
            for _ in range(round(20. / env.step_dt)):
                if not bool(live.any()):
                    break
                assert float(obs['actor'][:, :3].abs().max()) < 1e-6
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                failed |= live & terminated
                done = terminated | timeout
                live &= ~done
                if bool(done.any()) and bool(live.any()):
                    # 终止样本不再统计；显式复位仅为满足手动复位环境的步进契约。
                    ids = done.nonzero(as_tuple=False).flatten()
                    env.reset(env_ids=ids)
                    state.sampled[ids, 2] = height[ids]
                    state._command[ids, 2] = height[ids]
                    obs = env.observation_manager.compute(update_history=False)
        trace = {key: np.array([row[key] for row in captures]) for key in captures[0]}
        trace['height'] = height.cpu().numpy()
        trace['failed'] = failed.cpu().numpy()
        trace['initial_qpos'] = initial
        trace['initial_qvel'] = initial_velocity
        trace.update({'initial_' + name: value for name, value in initial_physics.items()})
        np.savez_compressed(args.out / (arm + '.npz'), **trace)
        (args.out / (arm + '-events.json')).write_text(json.dumps(events) + '\n')
        return trace, events
    finally:
        env.close()


def summarize(trace, events):
    time = trace['time'][:, None]
    first = np.full(len(trace['height']), np.inf)
    for event in events:
        first[event['ids']] = np.minimum(first[event['ids']], event['time'])
    # 事件在本步奖励之后执行，因此同时间戳的奖励仍归推扰前。
    before = (time >= 2.) & (time <= first[None, :])
    after = time > first[None, :]
    late = time >= first[None, :] + 2.
    masks = {'before_first_event': before, 'after_first_event': after,
             'two_seconds_after_first_event': late}

    def stats(mask):
        mask = mask & trace['live']
        active = mask & trace['active']
        slow = active & (trace['speed'] <= .005)
        def avg(key, valid):
            return float(trace[key][valid].mean()) if valid.any() else None
        return {'observed_samples': int(mask.sum()), 'active_samples': int(active.sum()),
                'active_seconds': float(active.sum() * .02),
                'gate_fraction': float(active.sum() / mask.sum()) if mask.any() else None,
                'active_reward_rate_mean': avg('reward', active),
                'velocity_score_mean': avg('velocity_score', active),
                'position_score_mean': avg('position_score', active),
                'anchor_distance_mean_m': avg('distance', active),
                'position_score_below_001_fraction': float((trace['position_score'][active] < .01).mean()) if active.any() else None,
                'slow_samples': int(slow.sum()), 'slow_seconds': float(slow.sum() * .02),
                'slow_reward_rate_mean': avg('reward', slow),
                'slow_anchor_distance_mean_m': avg('distance', slow),
                'slow_position_score_below_001_fraction': float((trace['position_score'][slow] < .01).mean()) if slow.any() else None}

    summary = {'failed_first_episodes': int(trace['failed'].sum()),
               'first_event_time_range_s': [float(first.min()), float(first.max())],
               'overall': {name: stats(mask) for name, mask in masks.items()},
               'by_height': {str(float(h)): {name: stats(mask & (trace['height'] == h)[None, :])
                   for name, mask in masks.items()} for h in np.unique(trace['height'])}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--envs-per-height', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists() or not args.checkpoint.is_file() or args.envs_per_height < 1:
        parser.error('Use a new output directory, an existing checkpoint and a positive environment count')
    args.out.mkdir(parents=True)
    configure_torch_backends()
    traces, summaries = {}, {}
    for arm in ('disturbed', 'quiet_control'):
        trace, events = run(args, arm)
        traces[arm] = trace
        summaries[arm] = summarize(trace, events)
        print(arm, json.dumps(summaries[arm]['overall']), flush=True)
    for name in traces['disturbed']:
        if name.startswith('initial_'):
            np.testing.assert_array_equal(traces['disturbed'][name], traces['quiet_control'][name])
    first_event = min(summaries[arm]['first_event_time_range_s'][0] for arm in summaries)
    before = traces['disturbed']['time'] < first_event
    pre_event_difference = float(np.abs(traces['disturbed']['xy'][before]
                                       - traces['quiet_control']['xy'][before]).max())
    np.testing.assert_allclose(traces['disturbed']['time'], np.arange(1, 1001) * .02)
    paths = [Path(__file__), args.checkpoint,
             *sorted((ROOT / 'mjlab_training/src/ct_mjlab/envs/chuanliantui').glob('*.py')),
             ROOT / 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml']
    sources = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for path in paths:
        if path != args.checkpoint:
            destination = args.out / 'sources' / path.resolve().relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    report = {'scope': 'No-learning deterministic-actor rollout in training config; not PPO exploration or hardware acceptance',
              'seed': args.seed, 'envs': 4 * args.envs_per_height,
              'duration_s': 20., 'ground_start_probability': 1., 'zero_hold_probability': 1.,
              'reward': 'unweighted zero_command_hold rate; training weight=3; step_dt=0.02',
              'slow_definition_m_s': .005, 'near_zero_position_score_threshold': .01,
              'initial_qpos_qvel_paired_equal': True,
              'initial_randomized_physics_paired_equal': True,
              'pre_first_global_event_xy_max_difference_m': pre_event_difference,
              'control_limitation': 'Independently initialized GPU rollouts may diverge numerically before events; compare aggregate distributions, not bitwise counterfactual trajectories',
              'quiet_control': 'Same timers and random draws; suppress only push velocity and persistent wrench effects',
              'sources': sources, 'arms': summaries}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'report': str(args.out / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
