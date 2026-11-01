"""保留线录制轨迹的共用渲染；无旧稳站任务入口。"""
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
