#!/usr/bin/env python3
"""Render actual stand-task reset states or saved evaluation states; no learning."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts


def renderer_for(model, width=640, height=480):
    disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.light_castshadow[:] = False
    model.mat_reflectance[:] = 0.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 1.15, -18., 120.
    base = model.body('robot/base_link').id
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)

    def frame(qpos, label):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.xpos[base]
        camera.lookat[2] -= .025
        renderer.update_scene(data, camera=camera, scene_option=option)
        canvas = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, 28), fill='#19232c')
        draw.text((8, 8), label, fill='white')
        return canvas

    return renderer, frame


def precheck(args):
    from ct_mjlab.envs import task_registry

    env, _ = task_registry.make_env('chuanliantui_stand', play=True,
                                    num_envs=20, seed=42, device=args.device)
    try:
        raw = env.unwrapped
        raw.reset(seed=42)
        from scripts.agent.stand.check_chuanliantui_stand import runtime_pose_checks
        geometry = runtime_pose_checks(raw)
        ground = raw.command_manager.get_term('standup').ground_start.cpu().numpy().copy()
        if not ground.any() or ground.all():
            raise AssertionError('Mixed visual precheck must include both initial-state modes')
        qpos = raw.sim.data.qpos.detach().cpu().numpy().copy()
        mujoco.mj_saveModel(raw.sim.mj_model, str(args.output / 'scene.mjb'), None)
        np.savez_compressed(args.output / 'reset-states.npz', qpos=qpos,
                            qvel=raw.sim.data.qvel.cpu().numpy(), ground_start=ground,
                            env_origins=raw.scene.env_origins.cpu().numpy())
        renderer, frame = renderer_for(raw.sim.mj_model, 400, 300)
        try:
            sheet = Image.new('RGB', (2000, 1200), '#ffffff')
            for index, state in enumerate(qpos):
                mode = 'GROUND' if ground[index] else 'NEAR'
                sheet.paste(frame(state, f'env {index:02d} | {mode} | h_cmd=0.27 m'),
                            ((index % 5) * 400, (index // 5) * 300))
            sheet.save(args.output / 'reset-20-envs.png')
        finally:
            renderer.close()
        report = {'kind': '20 actual reset states; visual geometry precheck, not learned behavior',
                  'task': 'chuanliantui_stand', 'num_envs': 20,
                  'physics_dt_s': raw.physics_dt, 'policy_dt_s': raw.step_dt,
                  'qpos_shape': list(qpos.shape), 'image': 'reset-20-envs.png'}
        report.update(passed=True, ground_count=int(ground.sum()), near_count=int((~ground).sum()),
                      geometry=geometry, ground_source='original multimode reset plus ground-clearance correction')
        (args.output / 'precheck.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    finally:
        env.close()


def replay(args):
    import imageio.v2 as imageio

    model = mujoco.MjModel.from_binary_path(str(args.scene))
    with np.load(args.recording) as archive:
        qpos = archive['qpos']
        if qpos.ndim == 3:
            qpos = qpos[:, args.env_index]
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise ValueError('Expected [time, env, nq] or [time, nq] recorded qpos')
        time_key = next((key for key in ('policy_time_s', 'time', 't', 'time_s') if key in archive), None)
        times = (archive[time_key].reshape(-1) if time_key else
                 np.arange(len(qpos)) * args.record_dt)
    if len(times) != len(qpos):
        raise ValueError('Time and qpos lengths differ')
    start, end = float(times[0]), float(times[-1])
    label = 'DIAGNOSTIC'
    episode_path = args.recording.parent / 'episodes.json'
    if episode_path.is_file():
        episode = next(row for row in json.loads(episode_path.read_text())
                       if row['env'] == args.env_index)
        # Other worlds may keep running after this case fails and is reset.
        # Never show that reset as if it were a successful recovery.
        end = min(end, float(episode['ended_s']))
        label = 'PASSED' if episode['passed'] else 'FAILED'
    renderer, frame = renderer_for(model)
    video = args.output / 'recorded-rollout.mp4'
    try:
        with imageio.get_writer(video, fps=30, codec='libx264', quality=8,
                                macro_block_size=1) as writer:
            for frame_time in np.arange(start, end + 1e-8, 1 / 30):
                index = int(np.clip(np.searchsorted(times, frame_time), 0, len(times) - 1))
                canvas = frame(qpos[index], f'{label} env {args.env_index} | t={times[index]:.3f}s | h_cmd=0.27m')
                writer.append_data(np.asarray(canvas))
                if frame_time == start:
                    canvas.save(args.output / 'preview.png')
    finally:
        renderer.close()
    print(json.dumps({'video': str(video), 'source': str(args.recording),
                      'env': args.env_index, 'seconds': end - start, 'fps': 30,
                      'case_label': label, 'ends_at_case_result': episode_path.is_file(),
                      'scope': 'render saved trajectory only; no simulation or learning'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--env-index', type=int, default=0)
    parser.add_argument('--record-dt', type=float, default=.01)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; select a new directory')
    if not args.precheck and (args.recording is None or args.scene is None):
        parser.error('Replay needs --recording and --scene')
    args.output.mkdir(parents=True)
    precheck(args) if args.precheck else replay(args)


if __name__ == '__main__':
    main()
