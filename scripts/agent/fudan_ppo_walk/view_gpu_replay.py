"""Interactive playback of recorded GPU states; no physics steps or policy inference."""
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
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--branch', default='candidate')
    args = parser.parse_args()
    traces = {}; samples = {}
    names = {'zero': '原地停车', 'forward': '前进', 'backward': '后退',
             'spin': '原地转向', 'turn': '行驶转弯', 'stop_forward': '前进后停车', 'stop_backward': '后退后停车'}
    for height in ('h022', 'h026', 'h030'):
        folder = args.recording / height / args.branch
        protocol = json.loads((folder / 'protocol.json').read_text())
        for sample in protocol['samples']:
            case = sample['case']; cmd = case['segments'][0]['command']
            label = f'{cmd[2]:.2f}m | {names.get(case["category"],case["category"])} | vx={cmd[0]:+g}, yaw={cmd[1]:+g} | {sample["id"]}'
            with np.load(folder / 'traces' / (sample['id'] + '.npz')) as saved:
                traces[label] = {k: saved[k] for k in ('qpos', 'qpos_time', 'linear_body', 'linear_world', 'roll_pitch_deg')}
            samples[label] = sample
    iteration = int(args.branch.split('_')[0]) if args.branch.split('_')[0].isdigit() else 9000
    reference = protocol.get('target_pitch_deg',0.)
    title = f'{iteration}轮 · 停车参考{reference:+g}° · GPU轨迹回放'
    model = mujoco.MjModel.from_binary_path(str(args.recording / 'h030' / args.branch / 'fixtures/model.mjb'))
    data = mujoco.MjData(model)
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label=title)
    scene = MjlabViserScene(server, model, 1)
    scene.create_scene_gui(camera_distance=1.6, camera_azimuth=110., camera_elevation=18., show_debug_viz_control=False)
    initial = next(k for k in traces if k.startswith('0.30m') and samples[k]['case']['category'] == 'zero')
    with server.gui.add_folder(title, expand_by_default=True):
        selector = server.gui.add_dropdown('工况', options=list(traces), initial_value=initial)
        playing = server.gui.add_checkbox('播放', initial_value=True)
        follow = server.gui.add_checkbox('镜头跟随', initial_value=True)
        slider = server.gui.add_slider('时间 / 秒', min=0., max=15., step=.01, initial_value=0.)
        status = server.gui.add_markdown('')
        server.gui.add_markdown(f'**原始GPU轨迹，正常速度1×**\n\n可拖动旋转、滚轮缩放；可选{len(traces)}个工况。积分补偿关闭。')
    server.scene.add_grid('/world_reference', width=30., height=30., cell_size=.1, section_size=1., plane='xy')
    selected = None; position = 0.; last_slider = 0.; previous_root = None; last = time.monotonic(); last_status = 0.
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.get_port()}', trajectories=len(traces),
        checkpoint_iteration=iteration,parking_reference_deg=reference,physics_steps=0,policy_forwards=0,playback='saved GPU trajectories'), ensure_ascii=False), flush=True)
    try:
        while True:
            now = time.monotonic(); elapsed = now - last; last = now
            if selector.value != selected:
                selected = selector.value; position = 0.; last_slider = 0.; slider.value = 0.
            trace = traces[selected]; sample = samples[selected]; duration = float(trace['qpos_time'][-1])
            if slider.value != last_slider:
                position = min(slider.value, duration)
            elif playing.value:
                position = (position + min(elapsed, .1)) % duration
            tick = min(round(position / .01), len(trace['qpos']) - 1)
            data.qpos[:] = trace['qpos'][tick]
            # 只按保存的姿态计算显示位置，不调用mj_step或策略网络。
            mujoco.mj_forward(model, data)
            root = data.qpos[:3].copy()
            with server.atomic():
                scene.update_from_arrays(data.xpos[None], data.xmat.reshape(1, model.nbody, 3, 3),
                    qpos=data.qpos[None], qvel=data.qvel[None], ctrl=data.ctrl[None])
                if follow.value and previous_root is not None:
                    delta = root - previous_root
                    for client in server.get_clients().values():
                        client.camera.position = np.asarray(client.camera.position) + delta
                        client.camera.look_at = np.asarray(client.camera.look_at) + delta
                if now - last_status >= .1:
                    slider.value = round(tick * .01, 2); last_slider = slider.value
                    j = min(tick * 5, len(trace['linear_body']) - 1)
                    command = sample['case']['segments'][0]['command']
                    for segment in sample['case']['segments']:
                        if tick * .01 >= segment['start']:command = segment['command']
                    speed = np.linalg.norm(trace['linear_world'][j, :2])
                    status.content = (f'时间：**{tick*.01:.2f} / {duration:.2f}s**\n\n'
                        f'指令vx：**{command[0]:+.2f}m/s**；实际vx：**{trace["linear_body"][j,0]:+.3f}m/s**\n\n'
                        f'水平速度：**{speed:.3f}m/s**；pitch：**{trace["roll_pitch_deg"][j,1]:+.2f}°**\n\n'
                        f'世界X：**{root[0]:+.3f}m**')
                    last_status = now
            previous_root = root; time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
