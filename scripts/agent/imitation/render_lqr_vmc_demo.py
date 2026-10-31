#!/usr/bin/env python3
"""Render one accepted LQR+VMC demonstration from recorded states, without simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.teacher.env import make_teacher_env_cfg
from ct_mjlab.teacher.standup import PHASE_NAMES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--rejected', action='store_true', help='Render the indexed rejected episode from report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dataset.read_text())
    records = (json.loads((args.dataset.parent/'report.json').read_text())['rejected_episodes']
               if args.rejected else manifest['episodes'])
    episode = records[args.episode]
    source = args.dataset.parent/episode['path']
    assert digest(source) == episode['sha256']
    with np.load(source) as trace:
        metadata = json.loads(str(trace['metadata']))
        qpos, qvel = trace['qpos'].copy(), trace['qvel'].copy()
        initial, initial_velocity = trace['initial_qpos'].copy(), trace['initial_qvel'].copy()
        times, phases = trace['time'].copy(), trace['phase'].copy()
        active = (trace['control_active'].copy() if 'control_active' in trace else
                  times > metadata['ground_start_hold_s']+1e-8)
    assert metadata['episode_id'] == episode['episode_id']
    student = metadata.get('source_controller') == 'live_lqr_vmc_through_student_pd'
    controller = 'LQR→学生PD教师' if student else 'LQR+VMC 教师'
    timing = ('10ms目标/2ms PD | 腿/轮等待'
              f'{metadata["leg_delay_s"]*1000:.0f}/{metadata["wheel_delay_s"]*1000:.0f}ms'
              if student else '1ms控制/物理')
    balanced = episode['quality']['final_balance_v31']
    status = '末段平衡通过' if balanced else '末段平衡未通过'
    if student:
        status += ' | 筛选接收' if episode['accepted'] else ' | 筛选拒收'
    assert digest(MODEL_XML) == metadata['xml_sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 只统一显示原点；保存的位移、关节和姿态全部保留。
    origin = initial[:2].copy()
    qpos[:, :2] -= origin
    initial[:2] -= origin
    configure_torch_backends()
    cfg = make_teacher_env_cfg(initial=metadata['initial'], seed=metadata['seed'],
        velocity_integral=metadata['velocity_integral'], ground_start_hold_s=metadata['ground_start_hold_s'])
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    renderer = None
    try:
        model = env.sim.mj_model
        assert model.nq == qpos.shape[1]
        width, height, header, fps = 960, 640, 112, 60 if student else 30
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height-header)
        renderer = mujoco.Renderer(model, width=width, height=height-header)
        data, camera = mujoco.MjData(model), mujoco.MjvCamera()
        camera.lookat[:] = [(qpos[:, 0].min()+qpos[:, 0].max())/2, 0., .20]
        camera.distance, camera.azimuth, camera.elevation = 2.6, 100., -18.
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 21)
        frames = round(float(times[-1])*fps)
        indices = []
        with imageio.get_writer(args.output, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(frames):
                t = frame/fps
                index = -1 if frame == 0 else min(len(times)-1, max(0, round(t/.01)-1))
                indices.append(index)
                data.qpos[:] = initial if index < 0 else qpos[index]
                data.qvel[:] = initial_velocity if index < 0 else qvel[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                for x in np.arange(np.floor(qpos[:, 0].min()*10)/10-.2, .31, .1):
                    reference = abs(x) < 1e-7
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([.0015 if reference else .0007, .36, .0005]),
                        np.array([x, 0., .0008]), np.eye(3).ravel(),
                        np.array([.9, .1, .1, 1.] if reference else [.6, .6, .6, .7]))
                    renderer.scene.ngeom += 1
                canvas = Image.new('RGB', (width, height), '#18212b')
                canvas.paste(Image.fromarray(renderer.render().copy()), (0, header))
                draw = ImageDraw.Draw(canvas)
                phase = '等待0.2秒' if index < 0 or not active[index] else PHASE_NAMES[int(phases[index])]
                draw.text((12, 3), f'{controller} | seed {metadata["seed"]} env {metadata["env"]} | {t:.2f}s | 原速1倍', font=font, fill='white')
                draw.text((12, 30), f'腿长0.30m | {timing} | 阶段：{phase}', font=font, fill='#a8cfff')
                draw.text((12, 57), f'相对初态位移 {np.linalg.norm(data.qpos[:2]):.3f}m | {status}', font=font, fill='#ffcc8a')
                draw.text((12, 84), '固定相机 | 地面刻度10cm | 红线为初始位置 | 真实示范状态回放，没有重新模拟', font=font, fill='white')
                if frame in (0, 18, frames-1):
                    canvas.save(args.output.with_name(f'{args.output.stem}-{frame:04d}.png'))
                writer.append_data(np.asarray(canvas))
                if frame and frame % 150 == 0:
                    print(f'{frame}/{frames} frames', flush=True)
        with imageio.get_reader(args.output) as reader:
            assert reader.count_frames() == frames
            reader.get_data(0)
            reader.get_data(frames-1)
        report = dict(video=str(args.output), video_sha256=digest(args.output),
            source=str(source), source_sha256=digest(source), xml_sha256=metadata['xml_sha256'],
            renderer_sha256=digest(Path(__file__)), episode=episode, fps=fps, frames=frames,
            duration_s=frames/fps, frame_source_indices=indices, optimizer_updates=0,
            scope='Recorded teacher qpos/qvel rendering only; no policy or physics integration; fixed camera, translated XY origin.')
        args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f'Saved and decoded: {args.output}', flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == '__main__':
    main()
