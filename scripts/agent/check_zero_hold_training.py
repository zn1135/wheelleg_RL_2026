#!/usr/bin/env python3
"""Check zero-command sampling and survey low balanced poses; no learning."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'mjlab_training/src'))
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.huananhu_poses import _front_kinematics, _mesh_vertices
from ct_mjlab.envs.chuanliantui.model import load_model, solve_standing_joint_positions
from mjlab.envs import ManagerBasedRlEnv
from inspect_multimode_leg_length import standing_endpoint


def command_height_mapping():
    model = load_model()
    data = mujoco.MjData(model)
    points = []
    for length in (.18, .33):
        knee = brentq(lambda k: _front_kinematics(model, data, k)[0]-length,
                      *model.joint('lf1').range)
        point = standing_endpoint(model, knee)
        points.append({'requested_leg_xz_m': length, 'knee_rad': knee,
                       'base_height_m': point['base_reference_height_on_wheel_plane_m'],
                       'closure_m': point['closed_chain_residual_m']})
    return points


def low_pose_survey():
    model = load_model()
    data = mujoco.MjData(model)
    wheels = [model.body(n).id for n in ('lfwheel', 'rfwheel')]
    geoms = [g for g in range(model.ngeom) if model.geom_group[g] == 3]
    candidates = []
    for knee in np.linspace(*model.joint('lf1').range, 5):
        _, phi = _front_kinematics(model, data, float(knee))
        for delta in np.linspace(-1., 1., 21):
            try:
                positions = solve_standing_joint_positions(model, front_joint_targets={
                    'left': (phi+delta, knee), 'right': (-phi-delta, -knee)})
            except RuntimeError:
                continue
            data.qpos[:] = model.qpos0
            for name, value in positions.items():
                data.qpos[model.joint(name).qposadr[0]] = value
            if any(model.jnt_limited[j] and not model.jnt_range[j,0]-1e-7 <=
                   data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j,1]+1e-7
                   for j in range(model.njnt)):
                continue
            mujoco.mj_forward(model, data)
            relative = data.subtree_com[model.body('base_link').id]-data.xpos[wheels].mean(0)
            # 调整机身俯仰，使整体质心在轮轴正上方；不是动力学平衡验收。
            pitch = -np.arctan2(relative[0], relative[2])
            if abs(pitch) > np.deg2rad(25):
                continue
            data.qpos[3:7] = [np.cos(pitch/2), 0., np.sin(pitch/2), 0.]
            mujoco.mj_forward(model, data)
            bottom = min(_mesh_vertices(model, data, g)[:,2].min()
                         for g in geoms if model.geom_bodyid[g] in wheels)
            data.qpos[2] -= bottom
            mujoco.mj_forward(model, data)
            clearance = min(_mesh_vertices(model, data, g)[:,2].min()
                            for g in geoms if model.geom_bodyid[g] not in wheels)
            if clearance < -1e-6:
                continue
            candidates.append({'height_m': float(data.qpos[2]), 'pitch_deg': float(np.rad2deg(pitch)),
                               'knee_rad': float(knee), 'leg_rotation_rad': float(delta),
                               'nonwheel_clearance_m': float(clearance)})
    return {'scope': 'Sampled symmetric poses, COM above wheel axle, tilt<=25deg; not a proof of global reach.',
            'candidate_count': len(candidates),
            'lowest_sample': min(candidates, key=lambda x: x['height_m']) if candidates else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output path')
    cfg, _ = task_registry.get_cfgs('chuanliantui', play=False)
    default_probability = cfg.commands['standup'].zero_hold_probability
    cfg.scene.num_envs = 4
    cfg.commands['standup'].zero_hold_probability = 1.
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env.reset(seed=42)
        state = env.command_manager.get_term('standup')
        initial_ground_count = int(state.ground_start.sum())
        if cfg.commands['standup'].ground_start_probability == 1.:
            assert initial_ground_count == env.num_envs
        ids = torch.arange(4, device=env.device)
        state.ground_start[:] = True
        state.ground_command[:] = .7
        env.episode_length_buf[:] = 25  # 前1.5秒随机起立命令原来会覆盖静止命令。
        state._update_command(ids)
        assert bool((state.command[:,:2] == 0).all())
        state.zero_hold[:] = torch.tensor([True, False, True, False], device=env.device)
        state.sampled[:,:2] = .4
        state.standing[:] = False
        state.heading[:] = False
        state._update_command(ids)
        assert bool((state.command[state.zero_hold,:2] == 0).all())
        assert bool((state.command[~state.zero_hold,:2] != 0).all())
        selected = state.zero_hold.clone()
        state._resample_command(ids)
        torch.testing.assert_close(state.zero_hold, selected)
        state._update_command(ids)
        assert bool((state.command[selected,:2] == 0).all())
        # 锚点应只在零速回合稳定离开恢复后锁定，位置使用世界XY坐标。
        state.zero_hold[:] = True
        state.ground_start[:] = False
        state.getting_up[:] = False
        state.recovered[:] = True
        state.airborne[:] = False
        state.jump_phase[:] = 0
        state.stable_time[:] = .5
        state.zero_hold_anchor_valid[:] = False
        expected_anchor = env.scene['robot'].data.root_link_pos_w[:, :2].clone()
        state._latch_zero_hold_anchor(expected_anchor)
        assert bool(state.zero_hold_anchor_valid.all())
        torch.testing.assert_close(state.zero_hold_anchor_xy, expected_anchor)
        # 模拟局部复位：只允许被复位的回合重新选择零速标志。
        state.cfg.zero_hold_probability = 0.
        state._resetting = True
        state._resample_command(ids[2:])
        state._resetting = False
        assert bool(state.zero_hold[0]) and not bool(state.zero_hold[2:].any())
        state.zero_hold_anchor_valid[:] = True
        state.zero_hold_anchor_xy[:] = 1.
        state.reset(ids[2:])
        assert bool(state.zero_hold_anchor_valid[:2].all())
        assert not bool(state.zero_hold_anchor_valid[2:].any())
        assert bool((state.zero_hold_anchor_xy[2:] == 0).all())
        state._update_command(ids)
        with torch.inference_mode():
            obs, rewards, _, _, _ = env.step(torch.zeros((4,6), device=env.device))
        assert obs['actor'].shape == (4,39) and obs['critic'].shape == (4,82)
        assert bool(torch.isfinite(rewards).all())
        result = {'training_zero_hold_probability': default_probability,
                  'training_ground_start_probability': cfg.commands['standup'].ground_start_probability,
                  'initial_ground_start_count': initial_ground_count,
                  'ground_override_suppressed_only_for_zero_hold': True,
                  'resample_keeps_episode_selection': True, 'subset_reset_isolated': True,
                  'zero_hold_anchor_latches_after_recovery': True,
                  'zero_hold_anchor_subset_reset_isolated': True,
                  'actor_critic_dimensions': [39,82], 'finite_training_config_step': True}
    finally:
        env.close()
    result['low_pose_survey'] = low_pose_survey()
    result['requested_leg_length_to_height'] = command_height_mapping()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
