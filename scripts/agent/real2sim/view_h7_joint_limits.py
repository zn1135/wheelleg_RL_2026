"""Read eight-channel VOFA feedback and display closed MuJoCo poses, without stepping."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import struct
import sys
import threading
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror

TAIL = bytes.fromhex('0000807f')
CONFIG = ROOT / 'deployment/H7_RL_ros/imcalib/config/machine_config.c'
NAMES = mirror.MOTORS[:4] + mirror.PASSIVE[0] + mirror.PASSIVE[1]
LABELS = ('左腿收缩端', '左腿伸展端', '右腿收缩端', '右腿伸展端', '其他姿态')


class JointFrames:
    """Require two consecutive frame tails before accepting a new alignment."""
    def __init__(self):
        self.buffer = bytearray()
        self.aligned = False
        self.discarded = 0

    def feed(self, raw):
        self.buffer.extend(raw)
        frames = []
        while len(self.buffer) >= (36 if self.aligned else 72):
            if self.buffer[32:36] != TAIL or (not self.aligned and self.buffer[68:72] != TAIL):
                del self.buffer[0]
                self.discarded += 1
                self.aligned = False
                continue
            values = np.array(struct.unpack('<8f', self.buffer[:32]), dtype=np.float64)
            if not np.isfinite(values).all():
                del self.buffer[:36]
                self.discarded += 36
                self.aligned = False
                continue
            del self.buffer[:36]
            self.aligned = True
            frames.append(values)
        return frames


def solve(pose, values):
    # USB已经做过驱动标定，只应用现有的固件到模型映射一次。
    angles = pose.sign[:4] * (values[::2] - pose.zero)
    previous = pose.data.qpos.copy()
    valid = [pose._solve_leg(side, angles[side * 2:side * 2 + 2]) for side in (0, 1)]
    if not all(valid):
        pose.data.qpos[:] = previous
        mujoco.mj_kinematics(pose.model, pose.data)
        return None
    return {name: float(pose.data.qpos[pose.model.joint(name).qposadr[0]]) for name in NAMES}


def check(output):
    """Finite framing/mapping/closure checks; no hardware, physics or policy calls."""
    output.mkdir(parents=True, exist_ok=False)
    values = np.array([-.974696159, -.013187408, .498444557, -.039562225,
                       -.958397388, .013187408, .550504804, .013187408], dtype=np.float32)
    packet = struct.pack('<8f', *values) + TAIL
    decoder = JointFrames()
    decoded = []
    stream = b'noise' + packet * 7
    for i in range(0, len(stream), 11):
        decoded.extend(decoder.feed(stream[i:i + 11]))
    assert len(decoded) == 7 and decoder.discarded == 5
    assert all(np.array_equal(row, values) for row in decoded)
    decoder = JointFrames()
    damaged = packet[:35] + packet * 6
    recovered = decoder.feed(damaged)
    assert len(recovered) == 6 and decoder.discarded == 35
    invalid = struct.pack('<8f', float('nan'), *values[1:]) + TAIL
    assert len(JointFrames().feed(invalid + packet * 3)) == 3
    pose = mirror.MirrorPose(machine_config=CONFIG)
    pose.data.qpos[:7] = (0., 0., .8, 1., 0., 0., 0.)
    joints = solve(pose, values)
    assert joints is not None
    assert np.allclose([joints[n] for n in mirror.MOTORS[:4]],
                       pose.sign[:4] * (values[::2] - pose.zero), atol=1e-8)
    report = dict(ok=True, fragmented_frames=7, damaged_stream_recovered_frames=6,
                  nonfinite_rejected=True, maximum_closure_m=max(pose.residual_m),
                  model_joints_rad=joints, physics_steps=0, hardware_accessed=False,
                  scope='Framing and static closed-chain mapping only; not endpoint calibration')
    (output / 'check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', help='H7 USB CDC; close VOFA first')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--viewer-port', type=int, default=8778)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        check(args.output)
        return
    if not args.port:
        parser.error('--port required for live display')
    import serial
    import viser
    from mjlab.viewer.viser.scene import MjlabViserScene

    args.output.mkdir(parents=True, exist_ok=False)
    pose = mirror.MirrorPose(machine_config=CONFIG)
    pose.data.qpos[:7] = (0., 0., .8, 1., 0., 0., 0.)
    mujoco.mj_kinematics(pose.model, pose.data)
    sources = {}
    for name, path in [('model.xml', mirror.MODEL), ('machine_config.c', CONFIG),
                       ('viewer.py', Path(__file__)), ('mirror.py', Path(mirror.__file__))]:
        content = path.read_bytes()
        (args.output / name).write_bytes(content)
        sources[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest())
    (args.output / 'metadata.json').write_text(json.dumps(dict(
        sources=sources, port=args.port, channels='lf0 pos/torque, lf00 pos/torque, rf0 pos/torque, rf00 pos/torque',
        physics_steps=0, zero_verification='User reported passed; no offsets changed',
        limitation='Eight-channel stream has no source age, online bits, sequence or CRC. '
                   'A stationary value alone cannot establish fresh motor feedback. '
                   'Reconstructed passive angles depend on existing geometry and assembly branch.'), indent=2) + '\n')

    lock = threading.Lock()
    stopped = threading.Event()
    shared = dict(latest=None, count=0, error=None)
    decoder = JointFrames()
    serial_port = serial.Serial(args.port, 1152000, timeout=.1, exclusive=True)
    raw_file = (args.output / 'stream.bin').open('wb')
    samples_file = (args.output / 'samples.jsonl').open('w')
    solved_file = (args.output / 'poses.jsonl').open('w')
    marks_file = (args.output / 'marks.jsonl').open('w', buffering=1)
    started = time.monotonic()

    def read_serial():
        try:
            while not stopped.is_set():
                raw = serial_port.read(max(1, min(serial_port.in_waiting, 4096)))
                if not raw:
                    continue
                raw_file.write(raw)
                received = time.monotonic()
                for values in decoder.feed(raw):
                    with lock:
                        shared['count'] += 1
                        sample = dict(index=shared['count'], host_elapsed_s=received - started,
                                      firmware_positions_rad=values[::2].tolist(),
                                      feedback_torques_nm=values[1::2].tolist())
                        shared['latest'] = (received, values, sample)
                    samples_file.write(json.dumps(sample) + '\n')
        except Exception as error:
            with lock:
                shared['error'] = repr(error)
        finally:
            serial_port.close()

    reader = threading.Thread(target=read_serial, daemon=True)
    reader.start()
    server = viser.ViserServer(host='127.0.0.1', port=args.viewer_port, label='MuJoCo · 真机关节限位标定')
    server.gui.configure_theme(dark_mode=True)
    scene = MjlabViserScene(server, pose.model, 1)
    scene.create_scene_gui(camera_distance=1.5, camera_azimuth=110., camera_elevation=15., show_debug_viz_control=False)
    with server.gui.add_folder('真机姿态与端点记录', expand_by_default=True):
        server.gui.add_markdown('**机身固定显示，腿部跟随真机；不驱动电机。**\n\n先小幅摆动一侧，确认模型同向跟随，再逐侧记录收缩端、伸展端。\n\n零点按你已确认的结果使用，当前不会修改模型限位。')
        status = server.gui.add_text('反馈状态', initial_value='等待USB', disabled=True)
        motion = server.gui.add_text('四电机已观察到变化', initial_value='否 / 否 / 否 / 否', disabled=True)
        motor_table = server.gui.add_markdown('等待角度和力矩…')
        recording = server.gui.add_checkbox('记录活动范围', initial_value=False)
        endpoint = server.gui.add_dropdown('当前端点', options=LABELS, initial_value=LABELS[0])
        mark_button = server.gui.add_button('记录此端点')
        saved = server.gui.add_text('端点记录', initial_value='尚未记录', disabled=True)
        export_button = server.gui.add_button('导出当前测量')
        range_table = server.gui.add_markdown('尚未记录活动范围。')
        server.gui.add_markdown('保持至少1秒后记录，每个端点重复到达两次。**观察范围不能单独证明机械硬限位**；先核对实物到位与模型姿态。')

    state = dict(current=None, ranges={}, marks=[], initial=None, moved=np.zeros(4, dtype=bool), recent=[])

    def export():
        # 只导出观测值，不将有限样本的极值直接写成物理限位。
        report = dict(observed_ranges_rad=state['ranges'], endpoints=state['marks'],
                      frames=shared['count'], parser_discarded_bytes=decoder.discarded,
                      motor_motion_observed=state['moved'].tolist(), sources=sources,
                      physics_steps=0, source_model_modified=False, calibration_complete=False,
                      note='Needs physical endpoint confirmation and review before changing joint ranges')
        path = args.output / f'observations-{time.time_ns()}.json'
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        return path

    @mark_button.on_click
    def mark(_):
        with lock:
            current = state['current']
            if current is None or time.monotonic() - current['received'] > .2:
                saved.value = '未保存：反馈停更或闭链未解出'
                return
            recent = state['recent']
            if len(recent) < 10 or recent[-1][0] - recent[0][0] < .9:
                saved.value = '未保存：请保持至少1秒'
                return
            span = np.ptp(np.array([p[1] for p in recent]), axis=0)
            if np.max(span) > .003:
                saved.value = '未保存：角度仍在变化，请保持'
                return
            item = dict(label=endpoint.value, sample=current['sample'], joints_rad=current['joints'],
                        closure_m=current['closure'], holding_span_rad=span.tolist(),
                        motor_motion_observed=state['moved'].tolist())
            state['marks'].append(item)
            marks_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            saved.value = f"已记录 {len(state['marks'])} 个端点：{endpoint.value}"
            export()

    @export_button.on_click
    def save(_):
        with lock:
            saved.value = '已导出：' + export().name

    def stop(*_):
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps(dict(viewer_url=f'http://127.0.0.1:{args.viewer_port}', output=str(args.output)), ensure_ascii=False), flush=True)
    last_index = 0
    try:
        while not stopped.wait(.05):
            with lock:
                latest, error = shared['latest'], shared['error']
            if error:
                status.value = '串口异常：' + error
                continue
            if latest is None or time.monotonic() - latest[0] > .2:
                status.value = 'USB停更，暂停记录'
                continue
            received, values, sample = latest
            if sample['index'] == last_index:
                continue
            last_index = sample['index']
            joints = solve(pose, values)
            if joints is None:
                with lock:
                    state['current'] = None
                    state['recent'] = []
                status.value = '闭链未解出：请保持，核对实物与模型'
                continue
            model_warnings = []
            for name, q in joints.items():
                joint = pose.model.joint(name)
                if joint.limited[0] and not joint.range[0] <= q <= joint.range[1]:
                    model_warnings.append(name)
            status.value = ('已映射；超出现有限位：' + ','.join(model_warnings)) if model_warnings else '已映射；未超出现有限位'
            with lock:
                if state['initial'] is None:
                    state['initial'] = values[::2].copy()
                state['moved'] |= np.abs(values[::2] - state['initial']) > .005
                motion.value = ' / '.join('是' if x else '否' for x in state['moved'])
                state['current'] = dict(received=received, sample=sample, joints=joints, closure=list(pose.residual_m))
                state['recent'].append((received, values[::2].copy()))
                state['recent'] = [(t, q) for t, q in state['recent'] if received - t <= 1.1]
                if recording.value:
                    for name, q in joints.items():
                        previous = state['ranges'].get(name, [q, q])
                        state['ranges'][name] = [min(previous[0], q), max(previous[1], q)]
                ranges = dict(state['ranges'])
            solved_file.write(json.dumps(dict(sample=sample, joints_rad=joints, closure_m=pose.residual_m,
                                              recording=recording.value, outside_model_range=model_warnings)) + '\n')
            rows = ['| 电机 | 模型角度rad / ° | 反馈力矩N·m |', '|---|---:|---:|']
            for i, label in enumerate(('左前', '左后', '右前', '右后')):
                q = joints[mirror.MOTORS[i]]
                rows.append(f'| {label} | {q:.4f} / {np.rad2deg(q):.2f} | {values[2 * i + 1]:.3f} |')
            motor_table.content = '\n'.join(rows)
            if ranges:
                rows = ['| 模型关节 | 已记录范围rad | 当前限位rad |', '|---|---:|---:|']
                for name in NAMES:
                    joint = pose.model.joint(name)
                    current = f'{joint.range[0]:.3f} ~ {joint.range[1]:.3f}' if joint.limited[0] else '未设'
                    rows.append(f'| {name} | {ranges[name][0]:.3f} ~ {ranges[name][1]:.3f} | {current} |')
                range_table.content = '\n'.join(rows)
            scene.update_from_arrays(pose.data.xpos[None], pose.data.xmat.reshape(1, pose.model.nbody, 3, 3),
                                     qpos=pose.data.qpos[None], qvel=pose.data.qvel[None], ctrl=pose.data.ctrl[None])
    finally:
        stopped.set()
        reader.join(timeout=2.)
        with lock:
            export()
        for file in (raw_file, samples_file, solved_file, marks_file):
            file.close()
        server.stop()


if __name__ == '__main__':
    main()
