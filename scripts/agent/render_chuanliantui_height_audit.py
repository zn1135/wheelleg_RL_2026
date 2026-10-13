"""Render solved wheel-ground poses; a geometry illustration, not a learned actor."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from ct_mjlab.asset import cpu_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input / 'xml-height-audit.json').read_text())
    m = cpu_model()
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.05
    camera.azimuth = 135.
    camera.elevation = -15.
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    canvas = Image.new('RGB', (1260, 420), 'white')
    renderer = mujoco.Renderer(m, height=360, width=420)
    try:
        for i, case in enumerate(report['cases']):
            target = case['target_height_m']
            d.qpos[:] = np.load(args.input / f'pose-{round(target*100):02d}.npy')
            mujoco.mj_forward(m, d)
            camera.lookat[:] = d.xpos[m.body('base_link').id]
            renderer.update_scene(d, camera=camera, scene_option=option)
            canvas.paste(Image.fromarray(renderer.render().copy()), (420*i, 60))
            draw = ImageDraw.Draw(canvas)
            draw.text((420*i+8, 6), f'Kinematic target height {target:.2f} m', fill='black')
            draw.text((420*i+8, 23), 'Wheels on ground; not a policy replay', fill='black')
            forces = case['force_equilibrium']['wheel_vertical_support_n']
            draw.text((420*i+8, 40), f'Static support L/R: {forces[0]:.1f}/{forces[1]:.1f} N', fill='black')
        canvas.save(args.input / 'height-reachability.png')
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
