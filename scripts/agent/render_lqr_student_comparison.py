#!/usr/bin/env python3
"""Render matched recorded LQR/direct and live student-PD cases without resimulation."""
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
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--last5-closeup', action='store_true', help='Render a fixed-camera close-up of t=15..20s')
    parser.add_argument('--training-delays', action='store_true', help='Compare 2/2ms with the largest late leg RMS in the current training grid')
    parser.add_argument('--reference-run', type=Path, help='Matched 2/2ms run to compare against a separate 10ms stress run')
    parser.add_argument('--passed-stress', action='store_true', help='Select the largest late leg RMS among successful stress cases')
    parser.add_argument('--output-dir', type=Path, help='Separate directory for exported media')
    args = parser.parse_args()
    if args.passed_stress and args.reference_run is None:
        parser.error('--passed-stress requires --reference-run')
    output = args.output_dir or args.run
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run/'pd/result.json').read_text())
    assert digest(MODEL_XML) == report['xml_sha256']
    trace_path = args.run/'pd/trace.npz'
    assert digest(trace_path) == report['trace_sha256']
    reference_report, reference_path = report, trace_path
    if args.reference_run is not None:
        assert report['stress_delay_ms'] == 10
        reference_report = json.loads((args.reference_run/'pd/result.json').read_text())
        reference_path = args.reference_run/'pd/trace.npz'
        assert digest(reference_path) == reference_report['trace_sha256']
        assert report['xml_sha256'] == reference_report['xml_sha256']
        assert report['dataset_fingerprint'] == reference_report['dataset_fingerprint']
        candidates = [row for row in report['rows'] if not args.passed_stress or
                      (row['strict_rise'] and row['final_balance_v31'] and row['full_20s'] and row['finite'])]
        bad = max(candidates, key=lambda row: row['last5s_leg_velocity_rms_rad_s'])
        good = next(row for row in reference_report['rows'] if row['leg_delay_s'] == .002
                    and row['wheel_delay_s'] == .002 and row['episode_id'] == bad['episode_id'])
    elif args.training_delays:
        assert report['training_delay_grid']
        bad = max((row for row in report['rows']
                   if (row['leg_delay_s'], row['wheel_delay_s']) != (.002, .002)),
                  key=lambda row: row['last5s_leg_velocity_rms_rad_s'])
        good = next(row for row in report['rows'] if row['leg_delay_s'] == .002
                    and row['wheel_delay_s'] == .002 and row['episode_id'] == bad['episode_id'])
    else:
        good = next(row for row in report['rows'] if row['leg_delay_s'] == .002 and row['wheel_delay_s'] == .002)
        bad = next(row for row in report['rows'] if row['leg_delay_s'] == .006 and row['wheel_delay_s'] == .010
                   and row['episode_id'] == good['episode_id'])
    manifest = json.loads(args.dataset.read_text())
    episode = next(row for row in manifest['episodes'] if row['episode_id'] == good['episode_id'])
    teacher_path = args.dataset.parent/episode['path']
    assert digest(teacher_path) == episode['sha256']
    with np.load(teacher_path) as teacher, np.load(trace_path) as pd, np.load(reference_path) as reference:
        times = teacher['time'].copy()
        np.testing.assert_allclose(times, pd['time'], atol=1e-6)
        np.testing.assert_allclose(times, reference['time'], atol=1e-6)
        poses = np.stack((teacher['qpos'], reference['qpos'][:, good['env']], pd['qpos'][:, bad['env']]), 1)
        velocities = np.stack((teacher['qvel'], reference['qvel'][:, good['env']], pd['qvel'][:, bad['env']]), 1)
        initial = np.stack((teacher['initial_qpos'], reference['initial_qpos'][good['env']], pd['initial_qpos'][bad['env']]))
        initial_velocity = np.stack((teacher['initial_qvel'], reference['initial_qvel'][good['env']], pd['initial_qvel'][bad['env']]))
        motor_dq = np.stack((teacher['motor_dq'], reference['motor_dq'][:, good['env']], pd['motor_dq'][:, bad['env']]), 1)
    np.testing.assert_allclose(initial[:, 2:], np.repeat(initial[:1, 2:], 3, axis=0), atol=1e-6)
    np.testing.assert_allclose(initial_velocity, np.repeat(initial_velocity[:1], 3, axis=0), atol=1e-6)
    origins = initial[:, :2].copy()
    poses[:, :, :2] -= origins[None]
    initial[:, :2] -= origins
    titles = ['原LQR+VMC | 1ms直接力矩'] + [
        f'闭环LQR→学生PD | 腿/轮{row["leg_delay_s"]*1000:.0f}/{row["wheel_delay_s"]*1000:.0f}ms'
        for row in (good, bad)]
    statuses = ['末段平衡通过'] + [
        '末段平衡通过' if row['final_balance_v31'] else '末段平衡未通过' for row in (good, bad)]
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=1)
    renderer = None
    try:
        width, height, header, fps = 640, 448, 112, 60
        model = env.unwrapped.sim.mj_model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height-header)
        renderer = mujoco.Renderer(model, width=width, height=height-header)
        data, camera = mujoco.MjData(model), mujoco.MjvCamera()
        camera.lookat[:] = [(poses[:, :, 0].min()+poses[:, :, 0].max())/2, 0., .20]
        camera.distance, camera.azimuth, camera.elevation = 2.6, 100., -18.
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 17)
        start = 15. if args.last5_closeup else 0.
        stem = 'closed-loop-last5-closeup' if args.last5_closeup else 'closed-loop-comparison'
        movie = output/f'{stem}.mp4'
        frames = round((float(times[-1])-start)*fps)
        with imageio.get_writer(movie, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(frames):
                t = start+frame/fps
                index = -1 if frame == 0 and start == 0. else min(len(times)-1, max(0, round(t/.01)-1))
                canvas = Image.new('RGB', (3*width, height), '#18212b')
                draw = ImageDraw.Draw(canvas)
                for case in range(3):
                    if args.last5_closeup:
                        camera.lookat[:] = poses[times > 15., case, :3].mean(0)
                        camera.lookat[2] = .20
                        camera.distance = 1.2
                    data.qpos[:] = initial[case] if index < 0 else poses[index, case]
                    data.qvel[:] = initial_velocity[case] if index < 0 else velocities[index, case]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    for x in np.arange(np.floor(poses[:, :, 0].min()*10)/10-.2, .31, .1):
                        reference = abs(x) < 1e-7
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([.0015 if reference else .0007, .36, .0005]),
                            np.array([x, 0., .0008]), np.eye(3).ravel(),
                            np.array([.9, .1, .1, 1.] if reference else [.6, .6, .6, .7]))
                        renderer.scene.ngeom += 1
                    canvas.paste(Image.fromarray(renderer.render().copy()), (case*width, header))
                    left = case*width+10
                    draw.text((left, 3), titles[case], font=font, fill='white')
                    draw.text((left, 29), f'{t:.2f}s | 原速 | {statuses[case]}', font=font,
                              fill='#a5e6aa' if statuses[case] == '末段平衡通过' else '#ffb18a')
                    if index >= 0:
                        window = motor_dq[max(0, index-49):index+1, case][:, [0, 1, 3, 4]]
                        rms = float(np.sqrt(np.square(window).mean()))
                        speed = float(np.linalg.norm(velocities[index, case, :2]))
                        draw.text((left, 55), f'速度 {speed:.4f}m/s | 近0.5秒腿速RMS {rms:.4f}rad/s', font=font, fill='#a8cfff')
                    draw.text((left, 81), '同初态 | 前0.2s零主动出力 | 固定相机 | 地面10cm刻度', font=font, fill='white')
                if frame in (0, round(.6*fps), frames-1):
                    canvas.save(output/f'{stem}-{frame:04d}.png')
                writer.append_data(np.asarray(canvas))
                if frame and frame % 150 == 0:
                    print(f'{frame}/{frames} frames', flush=True)
        with imageio.get_reader(movie) as reader:
            assert reader.count_frames() == frames
            reader.get_data(0)
            reader.get_data(frames-1)
        metadata = dict(video_sha256=digest(movie), fps=fps, frames=frames, duration_s=frames/fps,
            source_time_range_s=[start, float(times[-1])],
            teacher_source_sha256=digest(teacher_path), pd_trace_sha256=report['trace_sha256'],
            reference_trace_sha256=reference_report['trace_sha256'],
            xml_sha256=report['xml_sha256'], renderer_sha256=digest(Path(__file__)),
            episode_id=good['episode_id'], pd_envs=[good['env'], bad['env']],
            selection=('Largest last5s leg RMS among successful 10ms stress cases; same-pose 2/2ms reference'
                       if args.passed_stress else
                       'Largest last5s leg RMS in the 10ms stress grid; same-pose 2/2ms reference'
                       if args.reference_run is not None else
                       'Largest last5s leg RMS among non-2/2ms training-grid rows; same initial pose in all columns'
                       if args.training_delays else 'Original 2/2ms and 6/10ms matched comparison'),
            scope='Render actual recorded states only. Identical initial pose/velocities except XY origin; no re-simulation or optimization. Labels are episode-level results, not per-frame acceptance.')
        sidecar = 'video-closeup.json' if args.last5_closeup else 'video.json'
        (output/sidecar).write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
        print(f'Saved and decoded: {movie}', flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == '__main__':
    main()
