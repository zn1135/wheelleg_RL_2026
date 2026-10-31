#!/usr/bin/env python3
"""Plot measured leg lengths and render an already recorded mjlab trajectory.

No policy inference, physics integration, training, or hardware access here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-rows', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.reference_rows.read_text())
    selected = [row for row in rows if row['command']['height_m'] == .1865]
    ids = [row['env'] for row in selected]
    means = np.array([[row['leg_length_xz_m'][side]['mean'] for side in ('left', 'right')]
                      for row in selected])
    with np.load(args.recording / 'trajectory.npz') as saved:
        qpos = saved['qpos']
        lengths = saved['leg_lengths'][:, ids]
        time = saved['time_s']
    window = time >= 2.
    replay_mean = lengths[window].mean(axis=(0, 1))
    reference_mean = means.mean(axis=0)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    colors = ('#2563b8', '#d16b18')
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout='constrained')
    ax = axes[0]
    ax.barh([1, 0], reference_mean, color=colors, height=.48, alpha=.8)
    for j, (name, y) in enumerate(zip(('左腿', '右腿'), (1, 0), strict=True)):
        ax.scatter(means[:, j], np.linspace(y-.17, y+.17, len(means)),
                   facecolor='white', edgecolor=colors[j], s=28, zorder=3)
        ax.text(reference_mean[j]+.002, y+.27, f'{name}平均 {reference_mean[j]:.4f} m', va='center')
    ax.axvline(.15, color='#555555', linestyle='--', label='目标腿长 0.15 m')
    ax.set(yticks=[1, 0], yticklabels=['左腿', '右腿'], xlim=(0, .245),
           xlabel='髋轴到轮轴的机体 XZ 平面腿长（m）',
           title='父模型 model_6848：原测量的 8 个零速样本，第 2～12 秒平均')
    ax.legend(loc='lower right'); ax.grid(axis='x', alpha=.15)
    ax = axes[1]
    for j, name in enumerate(('左腿', '右腿')):
        for i in range(len(ids)):
            ax.plot(time, lengths[:, i, j], color=colors[j], alpha=.12, linewidth=.7)
        ax.plot(time, lengths[:, :, j].mean(axis=1), color=colors[j], linewidth=2,
                label=f'{name}：本次回放的 8 环境均值')
    ax.axhline(.15, color='#555555', linestyle='--', label='目标 0.15 m')
    ax.axvspan(0, 2, color='#dddddd', alpha=.4)
    ax.set(xlim=(0, 12), xlabel='时间（s）；灰区为前 2 秒稳定窗口', ylabel='实际腿长（m）',
           title='同配置重新记录的过程；细线为单环境，粗线为平均')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=.15)
    fig.savefig(args.output / 'leg-length.png', dpi=170)
    plt.close(fig)

    # Use one representative recorded environment, never a synthetic mean pose.
    per_env_mean = lengths[window].mean(axis=0)
    chosen = int(np.argmin(np.linalg.norm(per_env_mean-reference_mean, axis=1)))
    env_id = ids[chosen]
    model = mujoco.MjModel.from_binary_path(str(args.recording / 'scene.mjb'))
    disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    model.light_castshadow[:] = False
    model.mat_reflectance[:] = 0.
    base = model.body('robot/base_link').id
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 2:
            continue
        name = model.body(model.geom_bodyid[gid]).name.rsplit('/', 1)[-1]
        rgb = ((.18, .40, .72) if name.startswith(('lf', 'left'))
               else (.82, .42, .12) if name.startswith(('rf', 'right')) else (.48, .50, .53))
        model.geom_matid[gid] = -1
        model.geom_rgba[gid] = (*rgb, 1.)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 1.05, -15., 135.
    video_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    renderer = mujoco.Renderer(model, height=600, width=800)
    try:
        with imageio.get_writer(args.output / 'parent-low-leg.mp4', fps=30,
                                codec='libx264', quality=8, macro_block_size=1) as writer:
            for frame in range(360):
                idx = min(len(time)-1, round(frame / 30. / .02))
                data.qpos[:] = qpos[idx, env_id]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.xpos[base]
                camera.lookat[2] -= .035
                renderer.update_scene(data, camera=camera, scene_option=option)
                canvas = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 800, 90), fill='#1c222a')
                draw.text((12, 8), f'model_6848 | RECORDED env {env_id} | t={time[idx]:.2f} s',
                          font=video_font, fill='white')
                draw.text((12, 36), f'Left XZ leg: {lengths[idx,chosen,0]:.4f} m',
                          font=video_font, fill='#8cbbff')
                draw.text((405, 36), f'Right XZ leg: {lengths[idx,chosen,1]:.4f} m',
                          font=video_font, fill='#ffb976')
                draw.text((12, 64), 'vx=0 | yaw=0 | base-height command=0.1865 m',
                          font=video_font, fill='white')
                if frame == 240:
                    canvas.save(args.output / 'robot-preview.png')
                writer.append_data(np.asarray(canvas))
    finally:
        renderer.close()
    summary = {'reference_mean_left_right_m': reference_mean.tolist(),
               'new_replay_mean_left_right_m': replay_mean.tolist(),
               'replay_minus_reference_m': (replay_mean-reference_mean).tolist(),
               'video_env': env_id, 'video_frames': 360, 'video_fps': 30,
               'scope': 'original 8-env mean chart + newly recorded representative replay; no training'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
