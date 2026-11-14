"""Play recorded JRL2 PD telemetry as measured poses; no physics or hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import viser
from mjlab.viewer.viser.scene import MjlabViserScene

from .analyze_usb_pd500 import decode, NAMES
from .replay_usb_pd_targets import PoseSolver


def prepare(logs, output):
    output.mkdir(parents=True, exist_ok=False)
    solver = PoseSolver()
    columns = {name: i for i, name in enumerate(NAMES)}
    report = dict(source=str(logs.resolve()), physics_steps=0, policy_forwards=0,
                  assumptions=['XY fixed; height inferred from geometry, not measured',
                               'roll/pitch use recorded observation gravity; possible observation bias',
                               'yaw and wheel angles integrated; passive joints solved from current geometry'],
                  sessions=[])
    for number, path in enumerate(sorted(logs.glob('*.bin'))):
        rows, sessions, decoding = decode(path)
        times = (rows[:, 0] - rows[0, 0]) / 1e6
        if not np.isfinite(rows).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f'Nonfinite data or nonmonotonic timestamps: {path}')
        gravity = rows[:, [columns[f'obs_{i}'] for i in (3, 4, 5)]]
        valid = (rows[:, columns['obs_valid']] == 1) & (rows[:, columns['pd_valid']] == 1)
        valid &= np.linalg.norm(gravity, axis=1) > .5
        good = np.flatnonzero(valid)
        if not len(good):
            raise ValueError(f'No valid measured poses: {path}')
        roll = np.arctan2(-gravity[:, 1], -gravity[:, 2])
        pitch = np.arctan2(gravity[:, 0], np.hypot(gravity[:, 1], gravity[:, 2]))
        angular = rows[:, [columns[f'obs_{i}'] for i in (0, 1, 2)]] * 4
        yaw_rate = (np.sin(roll)*angular[:, 1] + np.cos(roll)*angular[:, 2]) / np.maximum(np.cos(pitch), .01)
        dt = np.diff(times)
        # 无效观测、复位或丢帧处不跨段积分。
        continuous = valid[1:] & valid[:-1] & (dt < .02)
        continuous &= (sessions[1:] == sessions[:-1]) & (np.diff(rows[:, columns['reset_count']]) == 0)
        yaw = np.r_[0., np.cumsum(.5*(yaw_rate[1:]+yaw_rate[:-1])*dt*continuous)]
        velocity = rows[:, [columns['velocity_rad_s_'+n] for n in ('lw', 'rw')]]
        wheels = np.vstack((np.zeros(2), np.cumsum(.5*(velocity[1:]+velocity[:-1])*dt[:, None]*continuous[:, None], axis=0)))
        # 只降低显示帧率，原200Hz遥测和时间戳保持不变。
        display_times = np.arange(0., times[-1], .04)
        indices = np.maximum(0, np.searchsorted(times, display_times, side='right')-1)
        poses = []; errors = []; held = int(good[0])
        for index in indices:
            if valid[index]:
                held = int(index)
            motors = [rows[held, columns['current_'+n]] if n not in ('lw', 'rw') else wheels[held, ('lw', 'rw').index(n)]
                      for n in ('lf', 'lr', 'lw', 'rf', 'rr', 'rw')]
            qpos, error = solver.solve(motors)
            quat = Rotation.from_euler('xyz', [roll[held], pitch[held], yaw[held]]).as_quat()[[3, 0, 1, 2]]
            poses.append(solver.ground(qpos, quat)); errors.append(error)
        name = f'part-{number+1:02d}.npz'
        np.savez_compressed(output/name, qpos=poses, time_s=display_times,
                            rows=rows[indices], valid=valid[indices], loop_error_m=errors)
        item = dict(file=name, source=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    rows=len(rows), duration_s=float(times[-1]), display_frames=len(poses),
                    invalid_rows=int((~valid).sum()), max_loop_error_m=float(max(errors)), decoding=decoding)
        report['sessions'].append(item)
        print(json.dumps(item), flush=True)
    if not report['sessions']:
        raise ValueError('No binary logs found')
    mujoco.mj_saveModel(solver.model, str(output/'display-model.mjb'))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def view(output, port):
    report = json.loads((output/'report.json').read_text())
    traces = {}
    for i, item in enumerate(report['sessions']):
        label = f'{i+1} · {item["source"].split("-")[2]} · {item["duration_s"]:.2f}秒'
        with np.load(output/item['file']) as saved:
            traces[label] = {key: saved[key].copy() for key in saved.files}
    model = mujoco.MjModel.from_binary_path(str(output/'display-model.mjb'))
    data = mujoco.MjData(model)
    server = viser.ViserServer(host='127.0.0.1', port=port, label='3000轮模型 · 真机日志回放')
    scene = MjlabViserScene(server, model, 1)
    scene.camera_tracking_enabled = False
    scene.create_scene_gui(camera_distance=1.6, camera_azimuth=115., camera_elevation=15., show_debug_viz_control=False)
    server.scene.add_grid('/floor', width=8., height=8., cell_size=.1, section_size=1., plane='xy')
    @server.on_client_connect
    def connect(client):
        previous = [None]
        @client.camera.on_update
        def resize(camera):
            size = (camera.image_width, camera.image_height)
            if size == previous[0]:
                return
            previous[0] = size
            pixels = camera.image_height/(2*np.tan(np.pi/6))
            distance = max(1.6, .65*pixels/max(180, camera.image_width-330))
            shift = -165*distance/pixels
            # 把机器人放在右侧控制面板之外，窄窗口也能完整观看。
            camera.position = (shift, distance, .9)
            camera.look_at = (shift, 0., .24)
            camera.up_direction = (0., 0., 1.)
    with server.gui.add_folder('实录回放', expand_by_default=True):
        selector = server.gui.add_dropdown('片段', options=list(traces))
        playing = server.gui.add_checkbox('播放', initial_value=True)
        speed = server.gui.add_dropdown('速度', options=['0.25×', '0.5×', '1×', '2×'], initial_value='1×')
        slider = server.gui.add_slider('时间 / 秒', min=0., max=report['sessions'][0]['duration_s'], step=.04, initial_value=0.)
        restart = server.gui.add_button('回到起点')
        status = server.gui.add_markdown('')
        server.gui.add_markdown('**真机关节实录姿态重建，原速1×。**\n\n机身XY固定，高度由几何推算；倾斜来自日志观测，航向和轮角由速度积分。无效观测时保持最近有效姿态。\n\n可旋转、缩放和拖动时间；到片段末尾自动暂停。')
    @restart.on_click
    def reset(_):
        slider.value = 0.
    selected = None; position = 0.; last_slider = 0.; last = time.monotonic(); last_status = 0.
    col = {name: i for i, name in enumerate(NAMES)}
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.get_port()}', sessions=len(traces), physics_steps=0)), flush=True)
    try:
        while True:
            now = time.monotonic(); elapsed = now-last; last = now
            trace = traces[selector.value]; duration = float(trace['time_s'][-1])
            if selected != selector.value:
                selected = selector.value; position = 0.; slider.value = 0.; last_slider = 0.; slider.max = duration
            if slider.value != last_slider:
                position = min(slider.value, duration)
            elif playing.value and server.get_clients():
                position = min(duration, position+min(elapsed, .1)*float(speed.value[:-1]))
                if position >= duration:
                    playing.value = False
            index = min(int(position/.04), len(trace['qpos'])-1)
            data.qpos[:] = trace['qpos'][index]
            # 仅更新显示几何，不积分、不加载策略或连接硬件。
            mujoco.mj_kinematics(model, data)
            scene.update_from_arrays(data.xpos[None], data.xmat.reshape(1, model.nbody, 3, 3),
                                     qpos=data.qpos[None], qvel=data.qvel[None], ctrl=data.ctrl[None])
            if now-last_status >= .1:
                row = trace['rows'][index]
                slider.value = float(trace['time_s'][index]); last_slider = slider.value; last_status = now
                status.content = (f'**{position:.2f} / {duration:.2f}秒** · '+('观测有效' if trace['valid'][index] else '观测无效：保持姿态')+
                    f'\n\n指令 vx={row[col["obs_6"]]/2:+.2f}m/s，yaw={row[col["obs_7"]]/.25:+.2f}rad/s，高度={row[col["obs_8"]]/5:.3f}m'+
                    f'\n\n左右轮速：{row[col["velocity_rad_s_lw"]]:+.2f} / {row[col["velocity_rad_s_rw"]]:+.2f}rad/s'+
                    f'\n\n闭链连接误差：{trace["loop_error_m"][index]*1000:.4f}mm')
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--logs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8786)
    args = parser.parse_args()
    if args.logs:
        prepare(args.logs, args.output)
    view(args.output, args.port)


if __name__ == '__main__':
    main()
