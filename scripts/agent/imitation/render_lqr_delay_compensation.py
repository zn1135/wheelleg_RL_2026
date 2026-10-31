#!/usr/bin/env python3
"""Render original teacher and paired before/after 10ms-delay states; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--last5-closeup', action='store_true')
    args = parser.parse_args()
    baseline = json.loads((args.baseline/'result.json').read_text())
    candidate = json.loads((args.candidate/'result.json').read_text())
    assert baseline['xml_sha256'] == candidate['xml_sha256'] == digest(MODEL_XML)
    row = next(row for row in baseline['rows'] if row['leg_delay_s'] == .006 and row['wheel_delay_s'] == .010)
    improved = next(item for item in candidate['rows'] if item['env'] == row['env'])
    assert improved['episode_id'] == row['episode_id']
    manifest = json.loads(args.dataset.read_text())
    episode = next(item for item in manifest['episodes'] if item['episode_id'] == row['episode_id'])
    teacher_path = args.dataset.parent/episode['path']
    assert digest(teacher_path) == episode['sha256']
    for directory, report in ((args.baseline, baseline), (args.candidate, candidate)):
        assert digest(directory/'trace.npz') == report['trace_sha256']
    with np.load(teacher_path) as teacher, np.load(args.baseline/'trace.npz') as old, np.load(args.candidate/'trace.npz') as new:
        time = teacher['time'].copy()
        np.testing.assert_allclose(time, old['time'], atol=1e-6)
        np.testing.assert_allclose(time, new['time'], atol=1e-6)
        i = row['env']
        poses = np.stack((teacher['qpos'], old['qpos'][:, i], new['qpos'][:, i]), 1)
        velocity = np.stack((teacher['qvel'], old['qvel'][:, i], new['qvel'][:, i]), 1)
        motor_dq = np.stack((teacher['motor_dq'], old['motor_dq'][:, i], new['motor_dq'][:, i]), 1)
        initial = np.stack((teacher['initial_qpos'], old['initial_qpos'][i], new['initial_qpos'][i]))
    np.testing.assert_allclose(initial[:, 2:], np.repeat(initial[:1, 2:], 3, axis=0), atol=1e-6)
    origin = initial[:, :2].copy()
    poses[:, :, :2] -= origin[None]
    initial[:, :2] -= origin
    titles = ['原LQR | 1ms直接力矩', '补偿前 | 学生PD | 腿/轮6/10ms', 'LQR预览补偿 | 学生PD | 腿/轮6/10ms']
    passed = [episode['quality']['final_balance_v31'], row['final_balance_v31'], improved['final_balance_v31']]
    env, _ = task_registry.make_env('chuanliantui', play=True, num_envs=1)
    renderer = None
    try:
        width, height, header, fps = 640, 448, 112, 60
        model = env.unwrapped.sim.mj_model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height-header)
        renderer = mujoco.Renderer(model, width=width, height=height-header)
        data, camera = mujoco.MjData(model), mujoco.MjvCamera()
        camera.lookat[:] = [(poses[:, :, 0].min()+poses[:, :, 0].max())/2, 0., .2]
        camera.distance, camera.azimuth, camera.elevation = 2.6, 100., -18.
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 17)
        start = 15. if args.last5_closeup else 0.
        frames = round((float(time[-1])-start)*fps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.output, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(frames):
                t = start+frame/fps
                index = -1 if frame == 0 and start == 0. else min(len(time)-1, max(0, round(t/.01)-1))
                canvas = Image.new('RGB', (3*width, height), '#18212b')
                draw = ImageDraw.Draw(canvas)
                for case in range(3):
                    if args.last5_closeup:
                        camera.lookat[:] = poses[time > 15., case, :3].mean(0)
                        camera.lookat[2], camera.distance = .2, 1.2
                    data.qpos[:] = initial[case] if index < 0 else poses[index, case]
                    data.qvel[:] = 0. if index < 0 else velocity[index, case]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    for x in np.arange(np.floor(poses[:, :, 0].min()*10)/10-.2, .31, .1):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([.0007, .36, .0005]), np.array([x, 0., .0008]),
                            np.eye(3).ravel(), np.array([.6, .6, .6, .7]))
                        renderer.scene.ngeom += 1
                    canvas.paste(Image.fromarray(renderer.render().copy()), (case*width, header))
                    left = case*width+10
                    status = '末段平衡通过' if passed[case] else '末段平衡未通过'
                    draw.text((left, 3), titles[case], font=font, fill='white')
                    draw.text((left, 29), f'{t:.2f}s | 原速 | {status}', font=font, fill='#a5e6aa' if passed[case] else '#ffb18a')
                    if index >= 0:
                        dq = motor_dq[max(0, index-49):index+1, case][:, [0, 1, 3, 4]]
                        rms = float(np.sqrt(np.square(dq).mean()))
                        speed = float(np.linalg.norm(velocity[index, case, :2]))
                        draw.text((left, 55), f'速度 {speed:.4f}m/s | 近0.5秒腿速RMS {rms:.4f}rad/s', font=font, fill='#a8cfff')
                    draw.text((left, 81), '同初态 | 前0.2s零主动出力 | 固定相机 | 地面10cm刻度', font=font, fill='white')
                if frame in (0, frames-1):
                    canvas.save(args.output.with_name(f'{args.output.stem}-{frame:04d}.png'))
                writer.append_data(np.asarray(canvas))
                if frame and frame % 300 == 0:
                    print(f'{frame}/{frames} frames', flush=True)
        with imageio.get_reader(args.output) as reader:
            assert reader.count_frames() == frames
            reader.get_data(0)
            reader.get_data(frames-1)
        sidecar = dict(video_sha256=digest(args.output), fps=fps, frames=frames, source_time_range_s=[start, float(time[-1])],
            baseline_trace_sha256=baseline['trace_sha256'], candidate_trace_sha256=candidate['trace_sha256'],
            teacher_sha256=digest(teacher_path), episode_id=row['episode_id'], env=row['env'],
            renderer_sha256=digest(__file__), scope='Recorded states only; no re-simulation or training; fixed camera; same initial state except world XY translation.')
        args.output.with_suffix('.json').write_text(json.dumps(sidecar, indent=2)+'\n')
        print(f'Saved and decoded: {args.output}', flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == '__main__':
    main()
