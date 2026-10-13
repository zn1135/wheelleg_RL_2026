"""Render recorded MuJoCo qpos into four-policy comparison videos; no simulation.

Use the same Linux runtime/version as the recorder. Set MUJOCO_GL=egl (or
osmesa) before launching. This script never reads or writes training logs.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', default=['baseline', 'wide_height', 'mode_reward', 'recover_timeout'])
    parser.add_argument('--width', type=int, default=320)
    args = parser.parse_args()
    root = args.input
    groups = tuple(args.groups)
    if args.width < 32 or not groups or any('/' in g or '\\' in g for g in groups):
        parser.error('Use valid group labels and width >=32')
    label_map = dict(baseline='A baseline', wide_height='B wide height',
                     mode_reward='C mode reward', recover_timeout='D recover timeout',
                     getup_r15='r15 | assistance OFF')
    labels = tuple(label_map.get(g, g) for g in groups)
    report_path = root / 'evaluation.json'
    target_height = json.loads(report_path.read_text()).get('command', {}).get('height', .30) if report_path.exists() else .30
    model = mujoco.MjModel.from_binary_path(str(root / 'scene.mjb'))
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot/base_link')
    if base < 0:
        raise ValueError('Robot base not found in recorded scene')
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.15
    camera.azimuth = 135
    camera.elevation = -20
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    width, height, header = args.width, 288, 32
    poster = Image.new('RGB', (len(groups) * width, 2 * (height + header)), 'white')
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for row, pose in enumerate(('normal', 'ground')):
            streams = [np.load(root / f'{g}-{pose}-qpos.npz') for g in groups]
            count = min(len(s['time_s']) for s in streams)
            if not all(np.array_equal(streams[0]['time_s'][:count], s['time_s'][:count]) for s in streams):
                raise ValueError('Representative replay timestamps differ')
            target = root / f'comparison-{pose}.mp4'
            if target.exists():
                raise FileExistsError(target)
            with imageio.get_writer(str(target), fps=10, codec='libx264',
                                    quality=8, macro_block_size=1) as writer:
                for frame in range(count):
                    canvas = Image.new('RGB', (len(groups) * width, height + header), '#eeeeee')
                    draw = ImageDraw.Draw(canvas)
                    for col, stream in enumerate(streams):
                        data.qpos[:] = stream['qpos'][frame]
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = data.xpos[base]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        canvas.paste(Image.fromarray(renderer.render().copy()), (col * width, header))
                        draw.text((col * width + 5, 3), f'{labels[col]} | {pose} | t={stream["time_s"][frame]:.1f}s', fill='black')
                        draw.text((col * width + 5, 17), f'sample 0 | height {data.xpos[base, 2]:.3f}m / target {target_height:.2f}m', fill='black')
                    writer.append_data(np.array(canvas))
                    if frame == count - 1:
                        poster.paste(canvas, (0, row * (height + header)))
            for stream in streams:
                stream.close()
            print('REPLAY_VIDEO ' + str(target), flush=True)
        poster.save(root / 'comparison-final.png')
        (root / 'render-record.json').write_text(json.dumps(dict(
            mujoco_version=mujoco.__version__, fps=10,
            source='recorded GPU rollout qpos; mj_forward only; no reintegration',
            representative_sample=0, policy_order=list(groups),
            poses=['normal', 'ground']), indent=2) + '\n')
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
