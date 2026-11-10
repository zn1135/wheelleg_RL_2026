"""Side-by-side playback of measured-angle reconstruction and saved MuJoCo Warp baselines."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser.scene import MjlabViserScene


MODES = {'共享模型 · 弹簧开 · 原始闭链': 1, '诊断 · 弹簧关 · 原始闭链': 2,
         '诊断 · 弹簧开 · 收紧闭链': 3, '诊断 · 弹簧关 · 收紧闭链': 4}


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--loop-diagnostic', type=Path, nargs='*', default=[])
    parser.add_argument('--response-fit', type=Path)
    parser.add_argument('--port', type=int, default=8778); args = parser.parse_args()
    manifest = json.loads((args.recording/'manifest.json').read_text()); traces = {}; reports = {}
    for item in manifest['sessions']:
        with np.load(args.recording/item['name']/'comparison.npz') as data:
            traces[item['label']] = {k: data[k].copy() for k in data.files if k != 'simulation_qvel'}
        reports[item['label']] = item
    model = mujoco.MjModel.from_xml_path(str(args.recording/'display-model.xml'))
    fixed = mujoco.MjModel.from_xml_path(str(args.recording/'original/measured-pose-template.xml'))
    if args.loop_diagnostic:
        modes = {'闭链修复诊断 · 弹簧开': 5, '闭链修复诊断 · 弹簧关': 6}
        modes.update(MODES)
        for label, trace in traces.items():
            session = reports[label]['name']
            paths = [p/session/'limits_fine-replay.npz' for p in args.loop_diagnostic]
            path, = [p for p in paths if p.exists()]
            with np.load(path) as data:
                q = data['qpos'][::10].copy(); gaps = data['max_pin_gap_m'][::10].copy()
                assert np.allclose(data['time_s'][::10], trace['time_s'])
                assert np.array_equal(data['submitted_model_torque_nm'], trace['submitted_model_torque_nm'])
            xpos = np.empty((len(q), 2, fixed.nbody, 3), dtype=np.float32)
            xmat = np.empty((len(q), 2, fixed.nbody, 3, 3), dtype=np.float32)
            data = mujoco.MjData(fixed)
            for i, row in enumerate(q):
                for j, state in enumerate(row):
                    data.qpos[:] = state; mujoco.mj_kinematics(fixed, data)
                    xpos[i, j] = data.xpos; xmat[i, j] = data.xmat.reshape(-1, 3, 3)
            for key, value in [('qpos', q), ('xpos', xpos), ('xmat', xmat)]:
                trace[key] = np.concatenate((trace[key], value), axis=1)
            trace['model_closure_error_m'] = np.concatenate((trace['model_closure_error_m'], gaps), axis=1)
    else: modes = MODES
    if args.response_fit:
        assert not (args.response_fit/'invalidated.json').exists(), 'This fit was invalidated; select a corrected experiment'
        fit_report=json.loads((args.response_fit/'report.json').read_text())
        length_fit = 'length_results' in fit_report
        pose_fit = 'pose_results' in fit_report
        prefix='腿长与倾角候选' if pose_fit else ('腿长优先候选' if length_fit else ('当前最优候选' if fit_report.get('solver_profile')=='ordered' else '未通过重拟合'))
        fit_offset = next(iter(traces.values()))['qpos'].shape[1]
        modes = {prefix+' · 同实录力矩': fit_offset, prefix+' · 同目标反馈': fit_offset+1, **modes}
        for label, trace in traces.items():
            directory = args.response_fit/('final-'+reports[label]['name'])
            trace['fit_metrics'] = json.loads((directory/'report.json').read_text())['results'][0]
            if length_fit:
                trace['length_metrics'] = fit_report['length_results'][reports[label]['name']]
                with np.load(directory/'leg-length.npz') as data:
                    assert np.allclose(data['time_s'][::10], trace['time_s'])
                    trace['measured_length_m'] = data['measured_length_m'][::10].copy()
                    trace['simulated_length_m'] = data['simulated_length_m'][::10].copy()
                    if pose_fit:
                        trace['measured_tilt_rad'] = data['measured_tilt_rad'][::10].copy()
                        trace['simulated_tilt_rad'] = data['simulated_tilt_rad'][::10].copy()
            candidate = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
            motor_ids = [candidate.joint(n).dofadr[0] for n in ('lf0', 'lf00', 'rf0', 'rf00')]
            assert np.array_equal(candidate.dof_armature[motor_ids], np.full(4, .015795)), 'Legacy motor inertia in fit'
            with np.load(directory/'trials.npz') as data:
                q = data['qpos'][::10].copy()
                assert np.allclose(data['time_s'][::10], trace['time_s'])
            xpos = np.empty((len(q), 2, fixed.nbody, 3), dtype=np.float32)
            xmat = np.empty((len(q), 2, fixed.nbody, 3, 3), dtype=np.float32)
            gaps = np.zeros((len(q), 2)); data = mujoco.MjData(fixed)
            pairs = [('left_rear_pin1', 'left_front_pin1'), ('left_rear_pin2', 'left_front_pin2'),
                     ('right_rear_pin1', 'right_front_pin1'), ('right_rear_pin2', 'right_front_pin2')]
            for i, row in enumerate(q):
                for j, state in enumerate(row):
                    data.qpos[:] = state; mujoco.mj_kinematics(fixed, data)
                    xpos[i, j] = data.xpos; xmat[i, j] = data.xmat.reshape(-1, 3, 3)
                    gaps[i, j] = max(np.linalg.norm(data.site(a).xpos-data.site(b).xpos) for a, b in pairs)
            for key, value in [('qpos', q), ('xpos', xpos), ('xmat', xmat)]:
                trace[key] = np.concatenate((trace[key], value), axis=1)
            trace['model_closure_error_m'] = np.concatenate((trace['model_closure_error_m'], gaps), axis=1)
    for i in range(model.ngeom):
        body = model.body(model.geom_bodyid[i]).name; model.geom_matid[i] = -1
        if body == 'base_link': model.geom_rgba[i] = [.45, .50, .58, .24]
        elif body in ('lf0', 'rf0'): model.geom_rgba[i] = [1., .55, .12, 1.]
        elif body in ('lf00', 'rf00'): model.geom_rgba[i] = [.12, .55, 1., 1.]
        elif 'spring1' in body: model.geom_rgba[i] = [.75, .15, .18, 1.]
        else: model.geom_rgba[i] = [.62, .67, .73, 1.]
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='实录与 MuJoCo 同输入对照')
    server.gui.configure_theme(dark_mode=True)
    scene = MjlabViserScene(server, model, 2); scene.camera_tracking_enabled = False
    scene.create_scene_gui(camera_distance=2.5, camera_azimuth=270., camera_elevation=12., show_debug_viz_control=False)
    server.scene.add_grid('/floor', width=6., height=6., cell_size=.1, section_size=1., plane='xy')
    with server.gui.add_folder('同步对照', expand_by_default=True):
        server.gui.add_markdown('**左：USB实录角度重建　｜　右：MuJoCo响应回放**')
        selector = server.gui.add_dropdown('实录片段', options=list(traces), initial_value=next(iter(traces)))
        mode = server.gui.add_dropdown('仿真对照', options=list(modes), initial_value=next(iter(modes)))
        playing = server.gui.add_checkbox('播放', initial_value=False)
        speed = server.gui.add_dropdown('速度', options=['0.25×', '0.5×', '1×', '2×'], initial_value='1×')
        slider = server.gui.add_slider('时间 / 秒', min=0., max=35.92, step=.01, initial_value=0.)
        restart = server.gui.add_button('回到起点')
        difference = server.gui.add_button('定位差异明显时刻')
        view = server.gui.add_dropdown('镜头', options=['机器人自身左侧', '机器人自身右侧', '斜侧观察'], initial_value='机器人自身左侧')
        status = server.gui.add_markdown('')
        table = server.gui.add_markdown('')
        server.gui.add_markdown('**左侧是角度数据重建，不是真机摄像视频。**\n\n现有实录为装有气弹簧的遥控腿长模式；不是后来描述的独立纯力矩测试。弹簧关闭项只用于诊断。\n\n橙色：长支路输入 lf0/rf0；蓝色：短摇臂输入 lf00/rf00。\n\n拖动可旋转，滚轮可缩放。不连接USB、不驱动电机；页面不做物理积分或策略推理。')
        if args.loop_diagnostic:
            server.gui.add_markdown('闭链修复项只收紧连接和限位约束，并以0.2ms子步积分；限位范围、弹簧、质量、摩擦和力矩输入保持。该项用于检查机构翻转，**响应仍未拟合通过**。')
        if args.response_fit:
            server.gui.add_markdown('重拟合候选使用同一组弹簧/关节阻力参数，固定闭链修正、质量和电机映射；原始收腿与往返联合选优，慢速/快速另作回归检查。**候选没有写回共享训练模型。**')
            if pose_fit:
                server.gui.add_markdown('**本轮同时拟合腿长与腿倾角，以轮轴位置误差选优。** 倾角相对机身竖直方向，非机身俯仰；同目标反馈另行检查。')
            elif length_fit:
                server.gui.add_markdown('**本轮按同实录力矩下的左右腿长误差选优。** 同目标反馈和四关节角度分别展示，腿长接近不代表关节姿态或动力学已经标定。')
            proof=args.response_fit/'replica-proof.json'
            if proof.exists():
                copy_error=float(np.max(json.loads(proof.read_text())['max_motor_difference_deg']))
                server.gui.add_markdown(f'**约束排序后，同输入24份复制的已存轨迹最大角差：{copy_error:.6g}°。** 当前为候选池综合角度误差最小的一组，实机响应仍有残差。')
            repeatability = args.response_fit/'repeatability/report.json'
            if repeatability.exists() and not json.loads(repeatability.read_text())['reproducible']:
                server.gui.add_markdown('**该候选重复性未通过：同批相同输入/初态的左后关节最大轨迹差约28.18°。当前页面仅展示未通过的拟合结果。**')
    position = 0.; selected = None; last_slider = 0.; last = time.monotonic(); last_status = 0.; selected_view = None

    def set_camera(client):
        width, height = (client.camera.image_width, client.camera.image_height) if client.camera.update_timestamp else (1240, 750)
        pixels_per_meter_at_unit_distance = height/(2*np.tan(np.pi/6))
        distance = max(2.1, 1.55*pixels_per_meter_at_unit_distance/max(220, width-340))
        shift = 170*distance/pixels_per_meter_at_unit_distance
        if view.value != '机器人自身右侧': shift = -shift
        camera = {'机器人自身左侧': (shift, distance, 1.05), '机器人自身右侧': (shift, -distance, 1.05), '斜侧观察': (shift+distance*.6, distance, 1.35)}[view.value]
        client.camera.position = camera; client.camera.look_at = (shift, 0., .66); client.camera.up_direction = (0., 0., 1.)

    @server.on_client_connect
    def connect(client):
        set_camera(client); previous_size = [None]
        @client.camera.on_update
        def resize(camera):
            size = (camera.image_width, camera.image_height)
            if size != previous_size[0]: previous_size[0] = size; set_camera(client)

    @restart.on_click
    def reset(_): slider.value = 0.

    @difference.on_click
    def jump(_):
        playing.value = False; slider.value = 3.2 if '8秒' in selector.value else 23.5

    view_indices = [model.joint(fixed.joint(j).name).qposadr[0] for j in range(fixed.njnt)]
    try:
        print(json.dumps(dict(url=f'http://127.0.0.1:{server.get_port()}/', source='saved MuJoCo GPU and measured pose records', physics_steps=0, hardware_access=False), ensure_ascii=False), flush=True)
        while True:
            now = time.monotonic(); elapsed = min(now-last, .1); last = now
            if selector.value != selected:
                selected = selector.value; position = 0.; slider.value = 0.; last_slider = 0.
            trace = traces[selected]; duration = float(trace['time_s'][-1]); slider.max = duration
            if slider.value != last_slider: position = min(slider.value, duration)
            elif playing.value and server.get_clients(): position = (position+elapsed*float(speed.value[:-1])) % duration
            index = min(round(position/.01), len(trace['time_s'])-1); branch = modes[mode.value]
            xpos = trace['xpos'][index, [0, branch]].copy(); xmat = trace['xmat'][index, [0, branch]].copy()
            # 相机从+Y看时画面横向与世界X相反，保持实录始终在画面左边。
            offset = -.48 if view.value == '机器人自身右侧' else .48
            xpos[0, :, 0] += offset; xpos[1, :, 0] -= offset
            q = np.tile(model.qpos0, (2, 1)); q[:, view_indices] = trace['qpos'][index, [0, branch]]
            q[:, :3] = xpos[:, 1]; quat = np.empty(4)
            for env in (0, 1):
                mujoco.mju_mat2Quat(quat, xmat[env, 1].ravel()); q[env, 3:7] = quat
            scene.update_from_arrays(xpos, xmat, qpos=q, qvel=np.zeros((2, model.nv)), ctrl=np.zeros((2, model.nu)))
            if view.value != selected_view:
                selected_view = view.value
                for client in server.get_clients().values(): set_camera(client)
            if now-last_status >= .1:
                slider.value = round(index*.01, 2); last_slider = slider.value; last_status = now
                sign = trace['sign'][:4]; zero = trace['zero']; mq = trace['motor_q_indices']
                measured = np.degrees(trace['measured_firmware_angles_rad'][index]-trace['measured_firmware_angles_rad'][0])
                simulated = np.degrees((trace['qpos'][index, branch, mq]-trace['qpos'][0, branch, mq])*sign)
                torque_index = min(index*10, len(trace['submitted_model_torque_nm'])-1)
                torque = trace['submitted_model_torque_nm'][torque_index, :4]*sign
                gap = float(trace['model_closure_error_m'][index, branch-1]*1000)
                status.content = f'**{index*.01:.2f} / {duration:.2f} 秒**　｜　{mode.value}\n\n仿真连接点间隙：**{gap:.3f} mm**' + ('　⚠ 已超过0.1mm' if gap>.1 else '')
                if args.response_fit and branch >= fit_offset:
                    key = 'recorded_torque' if branch == fit_offset else 'recorded_target'
                    if 'measured_tilt_rad' not in trace:
                        values = trace['fit_metrics'][key]['rmse_deg']
                        status.content += '\n\n本段四轴角度RMSE：'+ ' / '.join(f'{value:.2f}°' for value in values)
                    if 'length_metrics' in trace:
                        values = trace['length_metrics'][key]['rmse_mm']
                        status.content += '\n\n左右腿长RMSE：'+' / '.join(f'{value:.2f} mm' for value in values)
                        if 'measured_tilt_rad' in trace:
                            values = trace['length_metrics'][key]['tilt_rmse_deg']
                            status.content += '\n\n左右腿倾角RMSE：'+' / '.join(f'{value:.2f}°' for value in values)
                lines = ['关节角变化（相对录制起点）：', '', '| 电机 | 实录 | 仿真 | 记录请求力矩 |', '|---|---:|---:|---:|']
                for i, label in enumerate(['lf0 左长支路', 'lf00 左短摇臂', 'rf0 右长支路', 'rf00 右短摇臂']):
                    lines.append(f'| {label} | {measured[i]:+.1f}° | {simulated[i]:+.1f}° | {torque[i]:+.2f} Nm |')
                if args.response_fit and branch >= fit_offset and 'length_metrics' in trace:
                    real_length = trace['measured_length_m'][index]*1000
                    sim_length = trace['simulated_length_m'][index, branch-fit_offset]*1000
                    lines.extend(['', '| 腿长 | 实录 | 仿真 |', '|---|---:|---:|'])
                    for side, label in enumerate(['机器人左腿', '机器人右腿']):
                        lines.append(f'| {label} | {real_length[side]:.1f} mm | {sim_length[side]:.1f} mm |')
                    if 'measured_tilt_rad' in trace:
                        lines = ['| 状态 | 实录 | 仿真 |', '|---|---:|---:|']
                        rt = np.degrees(trace['measured_tilt_rad'][index])
                        st = np.degrees(trace['simulated_tilt_rad'][index, branch-fit_offset])
                        for side, label in enumerate(['左腿', '右腿']):
                            lines.extend([f'| {label}长度 | {real_length[side]:.1f} mm | {sim_length[side]:.1f} mm |',
                                          f'| {label}倾角 | {rt[side]:+.1f}° | {st[side]:+.1f}° |'])
                table.content = '\n'.join(lines)
            time.sleep(.025)
    except KeyboardInterrupt: pass
    finally: server.stop()


if __name__ == '__main__': main()
