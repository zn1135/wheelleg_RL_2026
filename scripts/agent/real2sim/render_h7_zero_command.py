#!/usr/bin/env python3
"""Render the verified frozen seed42 ground-start example at original speed."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output/'render-report.json'
    report_path.write_text(json.dumps({'complete': False})+'\n')
    shutil.copy2(__file__, args.output/Path(__file__).name)
    study = args.capture_root/'zero-command-frozen-v2'
    snapshot = args.capture_root/'frozen-baseline'
    report = json.loads((study/'report.json').read_text())
    source_manifest = json.loads((snapshot/'manifest.json').read_text())
    for entry in source_manifest['files']:
        if digest(snapshot/entry['path']) != entry['sha256']:
            raise ValueError('Frozen source changed: '+entry['path'])
    sys.path.insert(0, str(snapshot/'sim2sim'))
    spec = importlib.util.spec_from_file_location('zero_video_frozen_cpu',
               snapshot/'sim2sim/mj_sim2sim_huananhu_multimode.py')
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    checkpoint = Path(report['checkpoint']['path'])
    if digest(checkpoint) != report['checkpoint']['sha256']:
        raise ValueError('Checkpoint changed')
    torch.set_num_threads(1)
    actor = sim.load_actor(checkpoint)
    env = sim.Playback('normal', (0., 0., .27))
    env.rng = np.random.default_rng(42)
    env.reset('ground')
    expected = np.load(study/'cpu-ground-seed42.npz')
    reference = json.loads((study/'cpu-ground-seed42.json').read_text())
    np.testing.assert_array_equal(env.data.qpos, expected['initial_qpos'])
    np.testing.assert_array_equal(env.data.qvel, expected['initial_qvel'])
    poses, velocities, metrics = [env.data.qpos.copy()], [env.data.qvel.copy()], []
    observation_error = 0.
    step = 0

    def controller(obs):
        nonlocal observation_error
        observation_error = max(observation_error, float(abs(obs-expected['obs'][step]).max()))
        with torch.inference_mode():
            return actor(torch.from_numpy(obs)[None]).numpy()[0]

    for step in range(len(reference)):
        action, torque = env.step(controller)
        metrics.append(env.metrics(action, torque))
        poses.append(env.data.qpos.copy())
        velocities.append(env.data.qvel.copy())
    keys = ('time_s', 'x_m', 'y_m', 'height_m', 'vx_m_s', 'vy_m_s', 'yaw_rate_rad_s')
    trajectory_error = max(abs(a[key]-b[key]) for a, b in zip(metrics, reference) for key in keys)
    if trajectory_error > 1e-9 or observation_error > 1e-7:
        raise ValueError(f'Replay differs from the cited run: trajectory={trajectory_error}, obs={observation_error}')
    poses, velocities = np.asarray(poses), np.asarray(velocities)
    np.savez_compressed(args.output/'replayed-states.npz', qpos=poses, qvel=velocities,
                        time_s=np.arange(len(poses))*sim.DT*sim.DECIMATION)
    print(f'Exact cited trajectory verified: max error {trajectory_error:g}; rendering 30 seconds.', flush=True)

    # Visual settings only; simulation above has already finished and been checked.
    model = env.model
    width, total_height, header, fps = 960, 640, 112, 25
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, total_height-header)
    for gid in range(model.ngeom):
        if gid == env.floor_geom:
            continue
        name = model.body(model.geom_bodyid[gid]).name
        color = ((.20, .47, .78) if name.startswith(('lf', 'left')) else
                 (.86, .47, .19) if name.startswith(('rf', 'right')) else (.58, .61, .65))
        model.geom_matid[gid] = -1
        model.geom_rgba[gid] = (*color, 1.)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.04, 0., .14]
    camera.distance, camera.azimuth, camera.elevation = 1.6, 90., -16.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 19)
    render_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=total_height-header)
    x_at_2 = float(poses[100, 0])
    movie = args.output/'model6848-ground-seed42-30s.mp4'
    frames = round(len(reference)*sim.DT*sim.DECIMATION*fps)
    try:
        with imageio.get_writer(movie, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(frames):
                index = round(frame/fps/(sim.DT*sim.DECIMATION))
                render_data.qpos[:], render_data.qvel[:] = poses[index], velocities[index]
                mujoco.mj_forward(model, render_data)
                renderer.update_scene(render_data, camera=camera, scene_option=option)
                # Fixed floor marks reveal translation without a tracking camera.
                for x in [*np.arange(-.6, .71, .1), x_at_2]:
                    is_ref = abs(x-x_at_2) < 1e-10
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([.0015 if is_ref else .0007, .32, .0005]),
                        np.array([x, 0., .0008]), np.eye(3).ravel(),
                        np.array([.9, .1, .1, 1.] if is_ref else [.65, .65, .65, .7]))
                    renderer.scene.ngeom += 1
                image = Image.fromarray(renderer.render().copy())
                canvas = Image.new('RGB', (width, total_height), '#18212b')
                canvas.paste(image, (0, header))
                draw = ImageDraw.Draw(canvas)
                obs = expected['obs'][min(index, len(reference)-1)]
                mode = '正常模式' if obs[28] else ('恢复起立' if obs[36] else '起立前保持')
                draw.text((12, 4), f'6848 冻结版本仿真  |  seed 42  |  原速 1倍  |  {frame/fps:.2f} 秒  |  {mode}', font=font, fill='white')
                draw.text((12, 31), '命令：前进 0 m/s，转向 0 rad/s，机身高度 0.27 m', font=font, fill='#a8cfff')
                value = f'{poses[index, 0]-x_at_2:+.3f} m' if index >= 100 else '起立阶段，尚未开始计量'
                draw.text((12, 58), '相对第2秒的前向位移：'+value, font=font, fill='#ffcc8a')
                draw.text((12, 85), '固定相机；地面刻度 10 cm；红线为第2秒机身位置；没有防护绳', font=font, fill='white')
                if frame in (0, 50, frames-1): canvas.save(args.output/f'preview-{frame:04d}.png')
                writer.append_data(np.asarray(canvas))
                if frame and frame % 250 == 0: print(f'Rendered {frame}/{frames} frames', flush=True)
    finally:
        renderer.close()
    summary = {'complete': True, 'video': str(movie), 'fps': fps, 'frames': frames,
               'duration_s': frames/fps, 'seed': 42, 'playback_speed': 1,
               'max_trajectory_difference': trajectory_error, 'max_observation_difference': observation_error,
               'forward_displacement_2_to_30_s_m': float(poses[-1, 0]-x_at_2),
               'reference_report_sha256': digest(study/'report.json'),
               'reference_trajectory_sha256': digest(study/'cpu-ground-seed42.json'),
               'snapshot_manifest_sha256': digest(snapshot/'manifest.json'),
               'renderer_source_sha256': digest(__file__), 'video_sha256': digest(movie),
               'scope': 'Exact replay of the cited frozen simulation; fixed camera and visual-only floor marks; no hardware.'}
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
