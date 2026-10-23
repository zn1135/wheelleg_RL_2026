#!/usr/bin/env python3
"""Measure 20 s zero-command position holding after normal/ground starts; no learning."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'mjlab_training/src'))
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.huananhu_poses import sample_reset_state
from ct_mjlab.envs.chuanliantui.multimode_rewards import zero_command_hold
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import quat_apply_inverse


def reset_evaluation_pose(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state = env.command_manager.get_term('standup')
    ground = env_ids % 2 == 1
    state.ground_start[env_ids] = ground
    sample_reset_state(env, env_ids, ground)
    if hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards.step = -1


def verdict(*, duration, maximum_displacement, failed, maximum_tilt, height_error,
            wheel_support_fraction=1., base_contact_fraction=0.):
    return (duration >= 20.-1e-4 and maximum_displacement <= .05 and not failed
            and maximum_tilt <= 25. and height_error <= .03
            and wheel_support_fraction >= .95 and base_contact_fraction <= .01)


def check_arithmetic():
    n = 10
    command = torch.zeros(n, 3)
    velocity = torch.zeros(n, 3)
    velocity[:4, 0] = torch.tensor([0., .005, .014, .05])
    position_error = torch.zeros(n)
    position_error[4:7] = torch.tensor([.01, .02, .05])
    anchor_valid = torch.ones(n, dtype=torch.bool)
    command[4, 0], command[5, 1] = .1, .1
    recover, airborne, jump = (torch.zeros(n, dtype=torch.bool) for _ in range(3))
    recover[6], airborne[7], jump[8] = True, True, True
    stable = torch.ones(n, dtype=torch.bool); stable[9] = False
    gravity = torch.zeros(n, 3); gravity[:, 2] = -1.
    values = zero_command_hold(command=command, linear_velocity=velocity,
        position_error=position_error, anchor_valid=anchor_valid, recover=recover,
        airborne=airborne, jump=jump, gravity=gravity, stable=stable)
    assert values[0] == 1 and bool((torch.diff(values[:4]) < 0).all())
    assert bool((values[4:] == 0).all())
    assert verdict(duration=20., maximum_displacement=.049, failed=False,
                   maximum_tilt=20., height_error=.02)
    assert not verdict(duration=20., maximum_displacement=.06, failed=False,
                       maximum_tilt=20., height_error=.02)
    assert not verdict(duration=19., maximum_displacement=.0, failed=False,
                       maximum_tilt=0., height_error=.0)
    assert not verdict(duration=20., maximum_displacement=.0, failed=True,
                       maximum_tilt=0., height_error=.0)
    assert not verdict(duration=20., maximum_displacement=.0, failed=False,
                       maximum_tilt=0., height_error=.0, wheel_support_fraction=.5)
    position_errors = torch.tensor([0., .01, .02, .05])
    position_values = zero_command_hold(command=torch.zeros(4, 3), linear_velocity=torch.zeros(4, 3),
        position_error=position_errors, anchor_valid=torch.ones(4, dtype=torch.bool),
        recover=torch.zeros(4, dtype=torch.bool), airborne=torch.zeros(4, dtype=torch.bool),
        jump=torch.zeros(4, dtype=torch.bool), gravity=torch.tensor([[0., 0., -1.]]).expand(4, -1),
        stable=torch.ones(4, dtype=torch.bool))
    assert position_values[0] == 1 and bool((torch.diff(position_values) < 0).all())
    return {'zero_hold_reward_speeds_m_s': [0., .005, .014, .05],
            'reward_values': values[:4].tolist(), 'position_errors_m': position_errors.tolist(),
            'position_reward_values': position_values.tolist(),
            'recovery_jump_turn_gates': True,
            'displacement_and_duration_verdict': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heights', default='.2165,.27,.34,.3664')
    parser.add_argument('--envs-per-case', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arithmetic = check_arithmetic()
    if args.check_only:
        args.output.write_text(json.dumps(arithmetic, indent=2)+'\n')
        print(json.dumps(arithmetic)); return
    if args.checkpoint is None or not args.checkpoint.is_file():
        parser.error('--checkpoint must name an existing model_*.pt')
    heights = [float(x) for x in args.heights.split(',')]
    if args.envs_per_case < 1 or not heights or not all(0 < h < 1 for h in heights):
        parser.error('Invalid heights or envs-per-case')
    configure_torch_backends()
    cfg, agent = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.scene.num_envs = len(heights)*2*args.envs_per_case
    cfg.auto_reset = False
    cfg.seed = args.seed
    cfg.episode_length_s = 25.1
    cfg.commands['standup'].zero_hold_probability = 1.
    cfg.commands['standup'].resampling_time_range = (1e9, 1e9)
    cfg.events['reset_pose'] = EventTermCfg(func=reset_evaluation_pose, mode='reset')
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = task_registry._task('chuanliantui').runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset(seed=args.seed)
        state = env.command_manager.get_term('standup')
        robot = env.scene['robot'].data
        hip_ids, _ = env.scene['robot'].find_bodies(('lf0', 'rf0'), preserve_order=True)
        wheel_ids, _ = env.scene['robot'].find_bodies(('lfwheel', 'rfwheel'), preserve_order=True)
        n, dt = env.num_envs, env.step_dt
        height = torch.tensor([h for h in heights for _ in range(2*args.envs_per_case)], device=env.device)
        state.sampled[:, 2] = height; state._command[:, 2] = height
        # 刷新复位时缓存的高度观测；这一策略步计入起立等待，不计入保持窗口。
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(torch.zeros((n, 6), device=env.device))
        initial_ground = state.ground_start.clone()
        anchored = torch.zeros(n, dtype=torch.bool, device=env.device)
        failed = anchored.clone(); finished = anchored.clone()
        anchor = torch.zeros((n, 2), device=env.device)
        anchor_time = torch.full((n,), -1., device=env.device)
        observed = torch.zeros(n, device=env.device)
        displacement = observed.clone(); tilt_max = observed.clone(); height_max = observed.clone()
        support_time = observed.clone(); base_time = observed.clone()
        leg_sum = torch.zeros((n, 2), device=env.device)
        leg_min = torch.full_like(leg_sum, float('inf'))
        leg_max = torch.full_like(leg_sum, -float('inf'))
        weight = float(env.sim.mj_model.body_mass.sum()*np.linalg.norm(env.sim.mj_model.opt.gravity))
        positions, qposes, flags = [], [], []
        with torch.inference_mode():
            for step in range(round(25./dt)):
                actor_obs = obs['actor']
                active = ~failed & ~finished
                if float(abs(actor_obs[active, :3]).max()) > 1e-6:
                    raise ValueError('Zero command was overwritten')
                torch.testing.assert_close(actor_obs[active,3], height[active]*5)
                torch.testing.assert_close(actor_obs[active,35], initial_ground[active].float())
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                elapsed = (step+2)*dt
                live = ~failed & ~finished
                failed |= live & (terminated | timeout | state.numerical_failure)
                stable = (~state.recover & (state.stable_time >= .5) & ~state.airborne
                          & (state.jump_phase == 0))
                first = ~anchored & ~failed & stable & (elapsed >= 2.) & (elapsed <= 5.)
                anchor[first] = robot.root_link_pos_w[first, :2]
                anchor_time[first] = elapsed
                anchored |= first
                failed |= ~anchored & (elapsed >= 5.)
                measuring = anchored & ~failed & ~finished & ~first
                failed |= measuring & (state.recover | state.fallen)
                measuring &= ~failed
                delta = (robot.root_link_pos_w[:, :2]-anchor).norm(dim=-1)
                gravity = robot.projected_gravity_b
                roll = torch.atan2(-gravity[:,1], -gravity[:,2]).abs()
                pitch = torch.asin(gravity[:,0].clamp(-1.,1.)).abs()
                tilt = torch.rad2deg(torch.maximum(roll, pitch))
                observed += measuring.float()*dt
                wheel_force = env.scene['wheel_contact'].data.force.norm(dim=-1)
                supported = (wheel_force > 5.).all(-1) & (wheel_force.sum(-1) >= .7*weight)
                base_contact = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1) > 1.
                support_time += (measuring & supported).float()*dt
                base_time += (measuring & base_contact).float()*dt
                # 髋轮机体XZ平面距离，不将机身高度当作真实腿长。
                delta_w = robot.body_link_pos_w[:,wheel_ids]-robot.body_link_pos_w[:,hip_ids]
                quat = robot.root_link_quat_w[:,None,:].expand(-1,2,-1)
                delta_b = quat_apply_inverse(quat.reshape(-1,4), delta_w.reshape(-1,3)).reshape(n,2,3)
                lengths = delta_b[:,:,(0,2)].norm(dim=-1)
                leg_sum += torch.where(measuring[:,None], lengths*dt, 0.)
                leg_min = torch.where(measuring[:,None], torch.minimum(leg_min,lengths), leg_min)
                leg_max = torch.where(measuring[:,None], torch.maximum(leg_max,lengths), leg_max)
                displacement = torch.where(measuring, torch.maximum(displacement, delta), displacement)
                tilt_max = torch.where(measuring, torch.maximum(tilt_max, tilt), tilt_max)
                error = (state.height_above_ground-height).abs()
                height_max = torch.where(measuring, torch.maximum(height_max, error), height_max)
                finished |= observed >= 20.-1e-4
                positions.append(robot.root_link_pos_w.cpu().numpy().copy())
                qposes.append(env.sim.data.qpos.cpu().numpy().copy())
                flags.append(measuring.cpu().numpy())
                if bool((finished | failed).all()):
                    break
                reset_ids = (terminated | timeout).nonzero(as_tuple=False).flatten()
                if len(reset_ids):
                    # 失败结果已锁定；只复位终止环境，让其他样本继续测完。
                    env.reset(env_ids=reset_ids)
                    state.sampled[reset_ids, :2] = 0.
                    state._command[reset_ids, :2] = 0.
                    state.sampled[reset_ids, 2] = height[reset_ids]
                    state._command[reset_ids, 2] = height[reset_ids]
                    obs = env.get_observations()
        rows = []
        for i in range(n):
            fields = dict(duration=float(observed[i]), maximum_displacement=float(displacement[i]),
                          failed=bool(failed[i]), maximum_tilt=float(tilt_max[i]), height_error=float(height_max[i]),
                          wheel_support_fraction=float(support_time[i]/observed[i].clamp_min(dt)),
                          base_contact_fraction=float(base_time[i]/observed[i].clamp_min(dt)))
            rows.append({'env': i, 'height_command_m': float(height[i]),
                         'ground_start': bool(initial_ground[i]), 'anchor_s': float(anchor_time[i]),
                         'leg_xz_mean_m': (leg_sum[i]/observed[i].clamp_min(dt)).tolist(),
                         'leg_xz_min_m': leg_min[i].tolist() if observed[i] > 0 else None,
                         'leg_xz_max_m': leg_max[i].tolist() if observed[i] > 0 else None,
                         **fields, 'passed': verdict(**fields)})
        report = {'kind': 'zero-command hold after normal and ground starts; no training',
            'seed': args.seed, 'envs': n, 'passed': sum(r['passed'] for r in rows),
            'physics_dt_s': env.physics_dt, 'policy_dt_s': dt,
            'criteria': {'hold_s': 20., 'anchor_deadline_s': 5., 'settle_at_least_s': 2.,
                         'max_displacement_m': .05, 'max_tilt_deg': 25., 'max_height_error_m': .03,
                         'min_wheel_support_fraction': .95, 'wheel_weight_share': .7,
                         'max_base_contact_fraction': .01},
            'arithmetic': arithmetic, 'rows': rows,
            'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), args.checkpoint.resolve(),
                            ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_mdp.py',
                            ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_rewards.py',
                            ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_env_cfg.py',
                            ROOT/'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_mdp.py',
                            ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')}}
        np.savez_compressed(args.output.with_suffix('.npz'), position=np.array(positions),
                            qpos=np.array(qposes), measuring=np.array(flags), step_dt=dt)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'envs': n, 'passed': report['passed'], 'report': str(args.output)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
