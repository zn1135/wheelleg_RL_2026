"""Read-only 3D playback of a saved MuJoCo Warp leg-pose control experiment."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
import viser
from mjlab.viewer.viser.scene import MjlabViserScene


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8779)
    parser.add_argument('--snapshot', type=Path, help='Save initial/final MuJoCo views and exit')
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    assert json.loads((args.run / 'verification.json').read_text())['passed']
    trace = np.load(args.run / 'traces.npz')
    model = mujoco.MjModel.from_xml_path(str(args.run / 'model.xml'))
    data = mujoco.MjData(model)
    if args.snapshot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .66]
        camera.azimuth, camera.elevation, camera.distance = 90, -8, 1.1
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        with mujoco.Renderer(model, width=800, height=600) as renderer:
            for ax, index, title in zip(axes, (0, -1), ('Initial pose', 'Controlled pose')):
                data.qpos[:] = trace['qpos'][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                ax.imshow(renderer.render())
                length = trace['actual_length_m'][index] * 1000
                angle = np.degrees(trace['actual_angle_rad'][index])
                ax.set_title(f'{title}\nL/R: {length[0]:.2f}/{length[1]:.2f} mm; {angle[0]:.2f}/{angle[1]:.2f} deg')
                ax.axis('off')
        fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.84, wspace=.025)
        fig.savefig(args.snapshot, dpi=140, bbox_inches='tight', pad_inches=.15)
        plt.close(fig)
        return
    # 回放每10ms一帧；物理状态来自已完成的GPU积分。
    indices = np.arange(0, len(trace['time_s']), 5)
    positions, rotations = [], []
    for index in indices:
        data.qpos[:] = trace['qpos'][index]
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos.copy())
        rotations.append(data.xmat.reshape(-1, 3, 3).copy())
    for i in range(model.ngeom):
        if model.body(model.geom_bodyid[i]).name == 'base_link':
            model.geom_matid[i] = -1
            model.geom_rgba[i] = [.55, .60, .67, .3]
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='MuJoCo · 腿长与腿角控制')
    scene = MjlabViserScene(server, model, 1)
    scene.camera_tracking_enabled = False
    scene.create_scene_gui(camera_distance=1.7, camera_azimuth=90., camera_elevation=8., show_debug_viz_control=False)
    server.scene.add_grid('/floor', width=4., height=4., cell_size=.1, section_size=1., plane='xy')
    server.gui.add_markdown(f'**目标：{report["target_length_mm"]:.0f} mm / {report["target_angle_deg"]:.0f}°**\n\n机身固定；腿角相对机身竖直方向。')
    playing = server.gui.add_checkbox('播放', initial_value=False)
    slider = server.gui.add_slider('仿真时间 / 秒', min=0., max=16., step=.01, initial_value=16.)
    restart = server.gui.add_button('从头播放')
    table = server.gui.add_markdown('')
    server.gui.add_markdown('使用冻结的局部负载/摩擦/阻尼参数，关节PID控制加静态负载补偿。\n\n**这是GPU动力学结果回放，未连接USB或驱动实机。** 190mm/30°超出参数已验证姿态范围，模型内到位不代表真机已验证。')

    @restart.on_click
    def reset(_):
        slider.value = 0.
        playing.value = True

    @server.on_client_connect
    def connect(client):
        client.camera.position = (.65, 1.65, .9)
        client.camera.look_at = (0., 0., .65)
        client.camera.up_direction = (0., 0., 1.)

    last = time.monotonic()
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.get_port()}/', hardware_access=False, new_physics_steps=0)), flush=True)
    while True:
        now = time.monotonic()
        if playing.value and server.get_clients():
            slider.value = min(16., slider.value + min(now - last, .1))
            if slider.value >= 16.:
                playing.value = False
        last = now
        frame = min(round(slider.value / .01), len(indices) - 1)
        index = indices[frame]
        scene.update_from_arrays(np.asarray(positions[frame])[None], np.asarray(rotations[frame])[None],
                                 qpos=trace['qpos'][index][None], qvel=trace['qvel'][index][None],
                                 ctrl=np.zeros((1, model.nu)))
        length, angle = trace['actual_length_m'][index] * 1000, np.degrees(trace['actual_angle_rad'][index])
        target_l, target_a = trace['target_length_m'][index] * 1000, np.degrees(trace['target_angle_rad'][index])
        rows = ['| 项目 | 左腿 | 右腿 |', '|---|---:|---:|',
                f'| 当前长度 mm | {length[0]:.2f} | {length[1]:.2f} |',
                f'| 目标长度 mm | {target_l[0]:.2f} | {target_l[1]:.2f} |',
                f'| 当前腿角 ° | {angle[0]:.2f} | {angle[1]:.2f} |',
                f'| 目标腿角 ° | {target_a[0]:.2f} | {target_a[1]:.2f} |']
        torque = trace['request_nm'][index]
        rows += [f'| 前电机请求 N·m | {torque[0]:.2f} | {torque[2]:.2f} |',
                 f'| 后电机请求 N·m | {torque[1]:.2f} | {torque[3]:.2f} |']
        table.content = '\n'.join(rows)
        time.sleep(.03)


if __name__ == '__main__':
    main()
