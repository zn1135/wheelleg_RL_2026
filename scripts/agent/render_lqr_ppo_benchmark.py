#!/usr/bin/env python3
"""Uncut20s native-LQR / frozen-BC / trained-PPO recorded-state video."""
import argparse, hashlib, json
from pathlib import Path
import imageio.v2 as imageio
import mujoco, numpy as np
from PIL import Image, ImageDraw, ImageFont
from ct_mjlab.envs import task_registry


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(args):
    paths = (args.native, args.bc, args.ppo)
    reports = [json.loads((p/'result.json').read_text()) for p in paths]
    ids = (args.case % 10, args.case, args.case); clips = []
    for path, report, env in zip(paths, reports, ids):
        assert digest(path/'trace.npz') == report['trace_sha256']
        with np.load(path/'trace.npz') as d:
            q = d['qpos'][:, env].copy(); v = d['qvel'][:, env].copy()
            initial = d['initial_qpos'][env].copy(); alive = d['alive'][:, env].copy()
        q[:, :2] -= initial[:2]; initial[:2] = 0
        clips.append((q, v, initial, alive))
    for clip in clips: np.testing.assert_array_equal(clip[2], clips[0][2])
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=1)
    renderer = None
    try:
        model = env.unwrapped.sim.mj_model; model.vis.global_.offwidth = 640; model.vis.global_.offheight = 360
        renderer = mujoco.Renderer(model, width=640, height=360); data = mujoco.MjData(model)
        camera = mujoco.MjvCamera(); camera.azimuth = 100; camera.elevation = -18; camera.distance = 1.35
        option = mujoco.MjvOption(); option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 17)
        titles = ['原生LQR＋VMC · 1ms力矩控制', '训练前BC · 2ms PD', args.label+' · 2ms PD']
        movie = args.output/'comparison.mp4'
        with imageio.get_writer(movie, fps=60, codec='libx264', quality=8, macro_block_size=1,
                                ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(1200):
                t = frame/60; tick = max(0, round(t/.01)-1)
                canvas = Image.new('RGB', (1920, 460), '#18212b'); draw = ImageDraw.Draw(canvas)
                for i, (q, v, initial, alive) in enumerate(clips):
                    ended = not alive[min(tick, len(alive)-1)]
                    at = min(tick, max(0, int(alive.sum())-1)) if ended else tick
                    data.qpos[:] = initial if frame == 0 else q[at]
                    data.qvel[:] = 0 if frame == 0 or ended else v[at]
                    camera.lookat[:] = [data.qpos[0], data.qpos[1], .2]
                    mujoco.mj_forward(model, data); renderer.update_scene(data, camera=camera, scene_option=option)
                    # 跟随相机下仍保留世界地面10cm刻度，避免把滚动看成原地静止。
                    for x in np.arange(-2., 2.01, .1):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.array([.0007, .36, .0005]),
                            np.array([x, 0., .0008]), np.eye(3).ravel(), np.array([.6, .6, .6, .7]))
                        renderer.scene.ngeom += 1
                    canvas.paste(Image.fromarray(renderer.render()), (640*i, 100))
                    row = reports[i]['rows'][ids[i]]
                    draw.text((640*i+10, 7), titles[i], font=font, fill='white')
                    draw.text((640*i+10, 36), f't={t:.2f}s · 位移{np.linalg.norm(data.qpos[:2]):.3f}m · case{args.case}', font=font, fill='white')
                    status = '已终止，定格' if ended else f'起立{int(row["strict_rise"])} 平衡{int(row["final_balance_v31"])} 停车{int(row["qualified"])} · 地面10cm刻度'
                    draw.text((640*i+10, 65), status, font=font, fill='#ffaaaa' if ended else '#c7e4f3')
                writer.append_data(np.asarray(canvas))
        reader = imageio.get_reader(movie); assert reader.count_frames() == 1200
        imageio.imwrite(args.output/'poster.png', reader.get_data(600)); reader.close()
        (args.output/'video.json').write_text(json.dumps(dict(frames=1200, seconds=20, fps=60, case=args.case,
            video_sha256=digest(movie), trace_sha256=[r['trace_sha256'] for r in reports],
            source_sha256=digest(__file__), label=args.label,
            scope='Actual fixed-pose recorded states; native0s hold/1ms versus BC,PPO0.2s hold/2ms. Same XML; no resimulation or learning.'), indent=2)+'\n')
    finally:
        if renderer is not None: renderer.close()
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'bc', 'ppo', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', type=int, default=0); p.add_argument('--label', default='PPO微调100轮')
    render(p.parse_args())
