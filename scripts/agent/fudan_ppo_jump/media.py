"""只渲染已采集的 GPU 姿态。每 4 个 10 ms 策略步对应 25 fps 的一帧。"""
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def render(model, poses, output, *, grid=False):
    width, height = (224, 144) if grid else (640, 384)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 105., -12., 1.7
    with mujoco.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
            str(output), fps=25, codec='libx264', macro_block_size=None) as writer:
        for i, batch in enumerate(poses):
            frame = Image.new('RGB', (width*5,height*4) if grid else (width,height))
            for j, qpos in enumerate(batch):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]+[.15,0,.04]
                renderer.update_scene(data,camera=camera)
                tile = Image.fromarray(renderer.render())
                ImageDraw.Draw(tile).text((5,4), f'#{j+1}  {i/25:.2f}s' if grid else f'{i/25:.2f}s', fill='white')
                frame.paste(tile, ((j%5)*width,(j//5)*height))
            writer.append_data(np.asarray(frame))
