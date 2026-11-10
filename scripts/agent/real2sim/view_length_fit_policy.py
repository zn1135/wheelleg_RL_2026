"""Read-only side-by-side playback of the8000 policy's original/fitted GPU traces."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser.scene import MjlabViserScene


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8781)
    parser.add_argument('--branch', default='candidate')
    parser.add_argument('--baseline', default='baseline')
    args = parser.parse_args()
    protocols = [json.loads((args.recording / name / 'protocol.json').read_text()) for name in (args.baseline, args.branch)]
    assert protocols[0]['samples'] == protocols[1]['samples']
    both = protocols[1].get('response_parameters',{}).get('sides')==['left','right'] or protocols[1].get('physical_leg_load',{}).get('mirrored_left',False)
    fitted_label = '双腿拟合' if both else '右腿拟合'
    model = mujoco.MjModel.from_binary_path(str(args.recording / args.baseline / 'fixtures/model.mjb'))
    data = mujoco.MjData(model)
    records = {}
    for sample in protocols[0]['samples']:
        case = sample['case']
        label = f'{sample["id"]} | {case["category"]} | {case["segments"][0]["command"]}'
        branches = []
        for name in (args.baseline, args.branch):
            with np.load(args.recording / name / 'traces' / (sample['id'] + '.npz')) as source:
                branches.append({k: source[k] for k in ('qpos', 'qpos_time', 'linear_body', 'roll_pitch_deg')})
        records[label] = (sample, branches)
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='8000轮 · 拟合参数GPU对照')
    scene = MjlabViserScene(server, model, 2)
    scene.camera_tracking_enabled = False
    scene.create_scene_gui(camera_distance=2.5, camera_azimuth=110., camera_elevation=20., show_debug_viz_control=False)
    server.scene.add_grid('/ground', width=60., height=60., cell_size=.1, section_size=1., plane='xy')
    selector = server.gui.add_dropdown('工况', options=list(records))
    play = server.gui.add_checkbox('播放', initial_value=True)
    follow = server.gui.add_checkbox('镜头跟随', initial_value=True)
    slider = server.gui.add_slider('时间 / 秒', min=0., max=15., step=.01, initial_value=0.)
    status = server.gui.add_markdown('')
    server.gui.add_markdown(f'**蓝标：原模型；橙标：{fitted_label}参数。**\n\n8000-v2＋停车2°参考，保存的GPU轨迹1×回放。两组仅沿画面横向错开放置；实际运动保留。左腿如启用则为右腿参数镜像，尚未独立标定；无USB或实机控制，这20例诊断不代表完整验收。')
    @server.on_client_connect
    def connect(client):
        client.camera.position=(2.2, -2.2, 1.5)
        client.camera.look_at=(0., 0., .3)
        client.camera.up_direction=(0., 0., 1.)
    previous = None
    last = time.monotonic()
    old_center = None
    while True:
        now = time.monotonic()
        sample, branches = records[selector.value]
        if previous != selector.value:
            slider.max = float(min(t['qpos_time'][-1] for t in branches))
            slider.value = 0.
            previous = selector.value
            old_center = None
        elif play.value and server.get_clients():
            slider.value = (slider.value + min(now-last, .1)) % slider.max
        last = now
        pp, rr, qq, roots = [], [], [], []
        cells = []
        for side, trace in enumerate(branches):
            i = min(round(slider.value / .01), len(trace['qpos'])-1)
            data.qpos[:] = trace['qpos'][i]
            mujoco.mj_kinematics(model, data)
            positions = data.xpos.copy()
            positions[:, 1] += (-.65 if side == 0 else .65)
            pp.append(positions)
            rr.append(data.xmat.reshape(model.nbody,3,3).copy())
            qq.append(data.qpos.copy())
            roots.append(data.qpos[:3] + np.array([0., -.65 if side == 0 else .65, 0.]))
            j = min(i*5, len(trace['linear_body'])-1)
            cells.append([trace['linear_body'][j,0], trace['roll_pitch_deg'][j,1], data.qpos[2]])
        scene.update_from_arrays(np.array(pp), np.array(rr), qpos=np.array(qq), qvel=np.zeros((2,model.nv)), ctrl=np.zeros((2,model.nu)))
        center = np.mean(roots, axis=0)
        if follow.value and old_center is not None:
            delta = center - old_center
            for client in server.get_clients().values():
                client.camera.position = np.asarray(client.camera.position) + delta
                client.camera.look_at = np.asarray(client.camera.look_at) + delta
        old_center = center
        for side, label in enumerate(('原模型', fitted_label)):
            server.scene.add_label(f'/label{side}', ('🔵 ' if side==0 else '🟠 ') + label,
                position=tuple(roots[side]+[0,0,.4]), font_screen_scale=1.3)
        command = sample['case']['segments'][0]['command']
        for segment in sample['case']['segments']:
            if slider.value >= segment['start']:command = segment['command']
        status.content = '\n'.join([f'指令 vx/yaw/h：{command}', f'| 项目 | 原模型 | {fitted_label} |', '|---|---:|---:|',
            f'| 前向速度 m/s | {cells[0][0]:.3f} | {cells[1][0]:.3f} |',
            f'| 俯仰角 ° | {cells[0][1]:.2f} | {cells[1][1]:.2f} |',
            f'| 机身高度 m | {cells[0][2]:.3f} | {cells[1][2]:.3f} |'])
        time.sleep(.03)


if __name__ == '__main__':main()
