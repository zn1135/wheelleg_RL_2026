#!/usr/bin/env python3
"""Report static leg reach and height-command semantics; no training or rollout.

The leg length excludes the lateral axle offset, as in the CPU recovery-height
calculation. Wheel-on-floor heights assume a level base and vertical legs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))

from ct_mjlab.envs.chuanliantui.huananhu_mdp import HEIGHT_COMMAND_RANGE
from ct_mjlab.envs.chuanliantui.huananhu_poses import (
    _front_kinematics, _mesh_vertices, pose_length_range,
)
from ct_mjlab.envs.chuanliantui.model import (
    MODEL_XML, closed_chain_residual, load_model, solve_standing_joint_positions,
)


def standing_endpoint(model, knee):
    data = mujoco.MjData(model)
    _, phi = _front_kinematics(model, data, knee)
    positions = solve_standing_joint_positions(
        model, front_joint_targets={'left': (phi, knee), 'right': (-phi, -knee)})
    data.qpos[:] = model.qpos0
    for name, value in positions.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    residual = closed_chain_residual(model, data)
    assert residual < 1e-6, residual
    for name, value in positions.items():
        joint = model.joint(name)
        if joint.limited:
            assert joint.range[0] - 1e-6 <= value <= joint.range[1] + 1e-6, name

    # Put the lowest wheel mesh vertex at z=0; this is geometry, not load balance.
    wheels = {model.body(name).id for name in ('lfwheel', 'rfwheel')}
    collision_ids = [gid for gid in range(model.ngeom) if model.geom_group[gid] == 3]
    wheel_bottom = min(float(_mesh_vertices(model, data, gid)[:, 2].min())
                       for gid in collision_ids if model.geom_bodyid[gid] in wheels)
    data.qpos[2] -= wheel_bottom
    mujoco.mj_forward(model, data)
    sides = {}
    for side, hip, wheel in (('left', 'lf0', 'lfwheel'), ('right', 'rf0', 'rfwheel')):
        delta = data.body(wheel).xpos - data.xanchor[model.joint(hip).id]
        sides[side] = {
            'hip_wheel_xz_length_m': float(np.hypot(delta[0], delta[2])),
            'hip_wheel_3d_distance_m': float(np.linalg.norm(delta)),
            'hip_wheel_vertical_projection_m': float(-delta[2]),
        }
    nonwheel = [
        (float(_mesh_vertices(model, data, gid)[:, 2].min()),
         model.body(model.geom_bodyid[gid]).name)
        for gid in collision_ids if model.geom_bodyid[gid] not in wheels]
    lowest_z, lowest_body = min(nonwheel)
    return {
        'left_knee_rad': knee,
        'sides': sides,
        'base_reference_height_on_wheel_plane_m': float(data.qpos[2]),
        'lowest_nonwheel_mesh_clearance_m': lowest_z,
        'lowest_nonwheel_body': lowest_body,
        'closed_chain_residual_m': residual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output path to preserve previous measurements')
    model = load_model()
    bounds = model.joint('lf1').range
    # Exact mechanical stops; pose_length_range separately includes reset margins.
    endpoints = {name: standing_endpoint(model, float(knee))
                 for name, knee in zip(('minimum', 'maximum'), bounds, strict=True)}
    report = {
        'scope': 'static closed-chain geometry, not policy or hardware acceptance',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'source_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), MODEL_XML,
                         ROOT / 'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_poses.py',
                         ROOT / 'mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_mdp.py')},
        'height_command_definition': 'base reference height above terrain, not leg length',
        'height_command_range_m': HEIGHT_COMMAND_RANGE,
        'reset_xz_length_range_m': pose_length_range(model),
        'left_knee_limits_rad': bounds.tolist(),
        'right_knee_limits_rad': model.joint('rf1').range.tolist(),
        'vertical_standing_endpoints': endpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: point['sides'] for name, point in endpoints.items()}, indent=2))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
