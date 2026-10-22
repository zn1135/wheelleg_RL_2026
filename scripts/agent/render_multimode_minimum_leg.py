#!/usr/bin/env python3
"""Render the closed static pose at lf1=-0.77/rf1=+0.77, without stepping physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
import mujoco
import numpy as np

from ct_mjlab.envs.chuanliantui.model import (
    MODEL_XML, closed_chain_residual, load_model, load_spec, solve_standing_joint_positions,
)
from ct_mjlab.envs.chuanliantui.huananhu_poses import _front_kinematics
from inspect_multimode_leg_length import standing_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model()
    assert model.joint('lf1').range[0] == -.77
    assert model.joint('rf1').range[1] == .77
    measurement = standing_endpoint(model, -.77)
    data = mujoco.MjData(model)
    _, phi = _front_kinematics(model, data, -.77)
    positions = solve_standing_joint_positions(
        model, front_joint_targets={'left': (phi, -.77), 'right': (-phi, .77)})
    data.qpos[:] = model.qpos0
    for name, value in positions.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    data.qpos[2] = measurement['base_reference_height_on_wheel_plane_m']
    mujoco.mj_forward(model, data)
    assert data.qpos[model.joint('lf1').qposadr[0]] == -.77
    assert data.qpos[model.joint('rf1').qposadr[0]] == .77
    assert closed_chain_residual(model, data) < 1e-8
    np.save(args.output / 'qpos.npy', data.qpos)

    # Add a floor only to this rendering instance; leave the shared asset intact.
    spec = load_spec()
    floor = spec.worldbody.add_geom(name='measurement_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                     size=(10., 10., .1))
    floor.rgba = (.15, .18, .22, 1.)
    view = spec.compile()
    view_data = mujoco.MjData(view)
    view_data.qpos[:] = data.qpos
    mujoco.mj_forward(view, view_data)
    view.vis.global_.offwidth, view.vis.global_.offheight = 1000, 800
    view.light_castshadow[:] = False
    view.mat_reflectance[:] = 0.
    view.vis.headlight.ambient[:] = (.4, .4, .4)
    view.vis.headlight.diffuse[:] = (.7, .7, .7)
    for gid in range(view.ngeom):
        if view.geom_group[gid] != 2:
            continue
        name = view.body(view.geom_bodyid[gid]).name
        color = ((.18, .4, .72) if name.startswith(('lf', 'left'))
                 else (.82, .42, .12) if name.startswith(('rf', 'right')) else (.55, .57, .60))
        view.geom_matid[gid] = -1
        view.geom_rgba[gid] = (*color, 1.)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = .98, -13., 135.
    camera.lookat[:] = view_data.body('base_link').xpos
    camera.lookat[2] -= .035
    renderer = mujoco.Renderer(view, height=800, width=1000)
    try:
        renderer.update_scene(view_data, camera=camera, scene_option=option)
        picture = renderer.render().copy()
        # The side view uses the same pose, no policy or load-dependent settling.
        camera.azimuth, camera.elevation = 90., 0.
        renderer.update_scene(view_data, camera=camera, scene_option=option)
        side_picture = renderer.render().copy()
    finally:
        renderer.close()

    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font_path)
    plt.rcParams['font.family'] = FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(14, 10), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(1.1, 1.))
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(picture); ax.axis('off')
    ax.set_title('固定姿态：蓝色左腿，橙色右腿', fontsize=12)
    ax = fig.add_subplot(grid[1, 0]); ax.imshow(side_picture); ax.axis('off')
    ax.set_title('侧面观察，同一姿态；未进行物理步进', fontsize=12)
    for col, (side, prefix, color, knee_q) in enumerate((
        ('left', 'lf', '#2563b8', -.77), ('right', 'rf', '#d16b18', .77)), start=1):
        ax = fig.add_subplot(grid[col-1, 1])
        hip = data.xanchor[model.joint(prefix+'0').id].copy()
        knee = data.xanchor[model.joint(prefix+'1').id].copy()
        wheel = data.body(prefix+'wheel').xpos.copy()
        points = np.array([hip, knee, wheel])[:, (0, 2)] * 1000.
        points[:, 0] -= hip[0]*1000.
        name = '左腿' if side == 'left' else '右腿'
        actual = measurement['sides'][side]
        assert abs(np.linalg.norm(points[2]-points[0])/1000.
                   - actual['hip_wheel_xz_length_m']) < 1e-10
        # Solid lines follow the physical front links; dashed line is virtual leg length.
        ax.plot(points[:, 0], points[:, 1], '-o', color=color, linewidth=3, markersize=7)
        ax.plot(points[(0, 2), 0], points[(0, 2), 1], '--', color='#444444', linewidth=1.3)
        ax.add_patch(plt.Circle(points[2], .06*1000., color=color, alpha=.12))
        ax.axhline(0., color='#555555', linewidth=1.)
        ax.annotate('', xy=(50., points[0, 1]), xytext=(50., points[2, 1]),
                    arrowprops={'arrowstyle': '<->', 'color': '#333333', 'lw': 1.4})
        ax.text(58., points[:, 1].max()*.76,
                f"平面腿长\n{actual['hip_wheel_xz_length_m']*1000:.3f} mm", fontsize=11)
        for label, point, offset in zip(('髋轴', '膝轴', '轮轴'), points,
                                        ((-42, 10), (8, 10), (-42, -19)), strict=True):
            ax.annotate(label, point, xytext=offset, textcoords='offset points', fontsize=11)
        ax.set_title(f'{name} XZ 平面投影\n膝角 {knee_q:+.2f} rad', fontsize=13)
        ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax.set(xlim=(-230., 150.), ylim=(-15., 300.), xlabel='相对髋轴前后位置（mm）',
               ylabel='离地高度（mm）')
        ax.text(.02, .98, f"三维轴心直线距离：{actual['hip_wheel_3d_distance_m']*1000:.3f} mm",
                transform=ax.transAxes, va='top', fontsize=10)
        measurement[side+'_axis_positions_world_m'] = {
            'hip': hip.tolist(), 'knee': knee.tolist(), 'wheel': wheel.tolist()}
    fig.suptitle('最低腿长几何检查：lf1 = −0.77 rad，rf1 = +0.77 rad', fontsize=17)
    fig.savefig(args.output / 'minimum-leg-length.png', dpi=155)
    plt.close(fig)
    measurement.update({'joint_positions': positions,
                        'model_xml_sha256': hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),
                        'scope': 'static CAD geometry; no physics stepping, policy, training or hardware',
                        'base_orientation': 'level; virtual legs vertical; lowest wheel mesh on z=0'})
    (args.output / 'measurement.json').write_text(json.dumps(measurement, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(measurement['sides'], indent=2))


if __name__ == '__main__':
    main()
