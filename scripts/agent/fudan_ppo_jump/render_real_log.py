"""从实测IMU/关节反馈重建三次Jump视频，固定机身位置，零动力学积分。"""
import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from .analyze_real022 import read_csv, matrix, LEGS
from .probe_takeoff_work import ROOT, write_json
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.reset_geometry import mesh_vertices

sys.path.insert(0, str(ROOT/'sim2sim'))
from real2sim_huananhu import MirrorPose, MODEL, H7_PROJECT_CONFIGS

PHASE = {'walk': '行走', 'armed': '就绪 / WALK', 'takeoff': '起跳',
         'flight': '估计腾空', 'landing': '落地缓冲'}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def nearest(times, targets):
    right = np.searchsorted(times, targets).clip(0, len(times)-1)
    left = (right-1).clip(0)
    return np.where(abs(times[left]-targets) <= abs(times[right]-targets), left, right)


def reference_ground(scene, faded):
    """仅添加渲染地砖，接地与碰撞仍取原日志。"""
    opacity = .22 if faded else .72
    for x in range(-5, 5):
        for y in range(-5, 5):
            shade = .33 if (x+y) % 2 else .46
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                np.array([.2, .2, .001]), np.array([.4*x+.2, .4*y+.2, -.001]),
                np.eye(3).ravel(), np.array([shade, shade+.025, shade+.045, opacity]))
            scene.ngeom += 1


def load_complete_trials(path):
    """将完整阶段CSV的索引字段对应到已验证的回放字段。"""
    datasets = []
    episodes, details = [], []
    phase_names = ('walk', 'armed', 'takeoff', 'flight', 'landing')
    with zipfile.ZipFile(path) as archive:
        numbers = sorted(int(re.fullmatch(r'试次(\d+)-模型输入动作\.csv', n)[1])
                         for n in archive.namelist() if re.fullmatch(r'试次(\d+)-模型输入动作\.csv', n))
        for number in numbers:
            net = read_csv(archive, f'试次{number}-模型输入动作.csv')
            stair = read_csv(archive, f'试次{number}-H7反馈.csv')
            pd = read_csv(archive, f'试次{number}-H7控制.csv')
            net['board_time_us'] = net['board_us']
            net['phase'] = np.asarray([phase_names[int(x)] for x in net['phase']])
            for data in (stair, pd):
                data['h7_us'] = data['board_us']
                assert np.all(np.diff(data['h7_us']) > 0)
            for j, axis in enumerate('wxyz'):
                stair['quat_'+axis] = stair['quat_'+str(j)]
            for j, name in enumerate(LEGS):
                for prefix, source in (('angle_rad', 'hip_pos_rad'), ('velocity_rad_s', 'hip_vel_rad_s'), ('feedback_nm', 'hip_feedback_Nm')):
                    stair[prefix+'_'+name] = stair[source+'_'+str(j)]
                pd['current_'+name] = pd['current_'+str(j)]
            for j, side in enumerate(('left', 'right')):
                stair[side+'_length_m'] = stair[side+'_leg_0']
                stair['wheel_velocity_'+('l' if j == 0 else 'r')+'_rad_s'] = stair['wheel_vel_rad_s_'+str(j)]
            flags = pd['flags'].astype(np.int64)
            pd['pd_valid'], pd['drive_permit'] = (flags >> 3) & 1, (flags >> 7) & 1
            q = matrix(stair, ['quat_'+axis for axis in 'wxyz'])
            euler = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler('xyz')
            logged_euler = matrix(stair, ['euler_rad_'+str(j) for j in range(3)])
            assert abs((euler-logged_euler+np.pi) % (2*np.pi)-np.pi).max() < 1e-5
            stair['roll_rad'], stair['pitch_rad'] = euler[:, 0], euler[:, 1]
            jump = np.flatnonzero(net['model'] == '5200')
            start = float(net['board_time_us'][jump[0]])
            label = f'trial{number}'
            episodes.append(dict(label=label, request_sequence=int(net['sequence'][jump[0]]),
                start_board_us=start, capture_start_us=float(stair['h7_us'][0]),
                capture_end_us=float(stair['h7_us'][-1])))
            detail = dict(label=label)
            faults = np.flatnonzero(stair['fault'] != 0)
            if len(faults):
                j = int(faults[0])
                detail['feedback_freeze'] = []
                for name in LEGS:
                    triplet = matrix(stair, ['angle_rad_'+name, 'velocity_rad_s_'+name, 'feedback_nm_'+name])
                    k = j
                    while k > 0 and np.array_equal(triplet[k-1], triplet[j]):
                        k -= 1
                    detail['feedback_freeze'].append(dict(motor=name,
                        constant_feedback_from_ms=float((stair['h7_us'][k]-start)/1000),
                        constant_duration_before_fault_ms=float((stair['h7_us'][j]-stair['h7_us'][k])/1000)))
            details.append(detail)
            datasets.append((stair, pd, net))
    merged = [{key: np.concatenate([batch[i][key] for batch in datasets]) for key in datasets[0][i]} for i in range(3)]
    for data, time in zip(merged, ('h7_us', 'h7_us', 'board_time_us')):
        assert np.all(np.diff(data[time]) > 0), 'Trial timestamps must be disjoint and ordered'
    return *merged, dict(episodes=episodes), dict(real_episodes=details)


def run(recording, output, *, archive_path=None, height_estimate=None, ground=False):
    output.mkdir(parents=True, exist_ok=False)
    if archive_path is not None:
        stair, pd, net, report, timing = load_complete_trials(archive_path)
    else:
        report = json.loads((recording/'analysis02/result.json').read_text())
        timing = json.loads((recording/'timing/result.json').read_text())
        name = report['sessions'][0]['name']
        with zipfile.ZipFile(recording/'decoded-input.zip') as archive:
            stair = read_csv(archive, name+'/h7-stair.csv')
            pd = read_csv(archive, name+'/h7-pd.csv')
            net = read_csv(archive, name+'/model-input-actions.csv')
    estimated = None
    if height_estimate is not None:
        if archive_path is None:
            raise ValueError('Height estimates require the matching --archive input')
        with np.load(height_estimate) as saved:
            estimated = {key: saved[key] for key in saved.files}
        assert str(estimated['archive_sha256']) == sha(archive_path)
    times = stair['h7_us']
    quat = matrix(stair, ['quat_'+c for c in 'wxyz'])
    angles = matrix(stair, ['angle_rad_'+n for n in LEGS])
    wheel_velocity = matrix(stair, ['wheel_velocity_l_rad_s', 'wheel_velocity_r_rad_s'])
    pose = MirrorPose(MODEL, H7_PROJECT_CONFIGS['H7_ros'])
    # 使用现有标定映射，并与同一时刻PD记录中的模型角逐值核对。
    ids = nearest(times, pd['h7_us'])
    valid = (pd['pd_valid'] > 0) & (pd['drive_permit'] > 0) & (abs(times[ids]-pd['h7_us']) < 1000)
    expected = (angles[ids]-pose.zero)*pose.sign[:4]
    actual = matrix(pd, ['current_'+n for n in LEGS])
    mapping_error = float(abs(expected[valid]-actual[valid]).max())
    assert mapping_error < 1e-6
    model = pose.model
    model.vis.global_.offwidth = 560
    model.vis.global_.offheight = 432
    left_geoms, right_geoms = [], []
    for gid in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[gid])).name or ''
        if body.startswith(('lf', 'left_')):
            left_geoms.append(gid)
        if body.startswith(('rf', 'right_')):
            right_geoms.append(gid)
    for geoms, color in ((left_geoms, [.18, .65, .95, 1]), (right_geoms, [1., .56, .16, 1])):
        model.geom_matid[geoms] = -1
        model.geom_rgba[geoms] = color
    colors = model.geom_rgba.copy()
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = 1.35, -10.
    camera.lookat[:] = [0., 0., .43]
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    big = ImageFont.truetype(FONT, 28)
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 19)
    frames = {'normal': 0, 'slow': 0}
    all_poses, all_times, all_cases, closure = [], [], [], []
    cases = []
    length_errors = []
    wheel_ids = [model.body(n).id for n in ('lfwheel', 'rfwheel')]
    hip_ids = [model.joint(n).id for n in ('lf0', 'rf0')]
    base_id = model.body('base_link').id
    wheel_collision_geoms = [i for i in range(model.ngeom)
        if int(model.geom_bodyid[i]) in wheel_ids and (model.geom_contype[i] or model.geom_conaffinity[i])]
    with mujoco.Renderer(model, width=560, height=432) as renderer, \
         imageio.get_writer(output/'normal.mp4', fps=50, codec='libx264', quality=8) as normal, \
         imageio.get_writer(output/'slow-4x.mp4', fps=50, codec='libx264', quality=8) as slow:
        for case_index, episode in enumerate(report['episodes']):
            start = episode['start_board_us']
            if 'capture_start_us' in episode:
                steps = round((episode['capture_end_us']-episode['capture_start_us'])/5000.)
                grid = episode['capture_start_us']+np.arange(steps+1)*5000.
            else:
                end = start+episode['next_policy_after_ms']*1000
                grid = np.arange(start-250000, end+300000, 5000.)
            source_ids = nearest(times, grid)
            assert abs(times[source_ids]-grid).max() < 6000
            net_ids = (np.searchsorted(net['board_time_us'], grid, side='right')-1).clip(0)
            displacement = np.zeros(len(grid))
            vertical_speed = np.zeros(len(grid))
            camera.lookat[:] = [0., 0., .43]
            camera.distance = 1.35
            if estimated is not None:
                prefix = f'trial{case_index+1}_'
                estimate_time = estimated[prefix+'board_us']
                displacement = np.interp(grid, estimate_time, estimated[prefix+'height_m'])
                vertical_speed = np.interp(grid, estimate_time, estimated[prefix+'velocity_m_s'])
                # 固定每段镜头范围，同时保留正负积分漂移，避免逐帧追随抵消上下运动。
                low, high = .55+displacement.min()-.45, .55+displacement.max()+.25
                camera.lookat[2] = (low+high)/2
                camera.distance = max(1.35, (high-low)/(2*np.tan(np.radians(model.vis.global_.fovy/2)))*1.15)
            initial = Rotation.from_quat(quat[source_ids[0], [1, 2, 3, 0]])
            yaw0 = initial.as_euler('xyz')[2]
            heading = Rotation.from_euler('z', -yaw0)
            model_angles = (angles[source_ids]-pose.zero)*pose.sign[:4]
            wheels = wheel_velocity[source_ids]*pose.sign[4:]
            wheel_q = np.vstack((np.zeros(2), np.cumsum(.5*(wheels[:-1]+wheels[1:])*.005, axis=0)))
            base_height = .55
            ground_reference = None
            if ground:
                # 将触发前最近一个采样的较低轮底对齐z=0，每段地面随后保持固定。
                reference = max(0, int(np.searchsorted(grid, start, side='right'))-1)
                pose.data.qpos[:7] = [0., 0., .55+displacement[reference], 1., 0., 0., 0.]
                assert all(pose._solve_leg(side, model_angles[reference, side*2:side*2+2]) for side in (0, 1))
                orientation = Rotation.from_quat(quat[source_ids[reference], [1, 2, 3, 0]])
                pose.data.qpos[3:7] = (heading*orientation).as_quat()[[3, 0, 1, 2]]
                pose.data.qpos[pose.wheel_q] = wheel_q[reference]
                mujoco.mj_kinematics(model, pose.data)
                floor_z = min(float(mesh_vertices(model, pose.data, gid)[:, 2].min()) for gid in wheel_collision_geoms)
                base_height -= floor_z
                camera.lookat[2] -= floor_z
                ground_reference = dict(time_relative_s=float((grid[reference]-start)/1e6),
                                        initial_root_height_m=base_height, reference_frame=reference)
            stale_intervals = []
            detail = next(x for x in timing['real_episodes'] if x['label'] == episode['label'])
            for feedback in detail.get('feedback_freeze', []):
                if feedback['constant_duration_before_fault_ms'] < 20:
                    continue
                motor = feedback['motor']
                stale_start = start+feedback['constant_feedback_from_ms']*1000
                triplet = matrix(stair, ['angle_rad_'+motor, 'velocity_rad_s_'+motor, 'feedback_nm_'+motor])
                k = int(nearest(times, np.array([stale_start]))[0])
                j = k+1
                while j < len(times) and np.array_equal(triplet[j], triplet[k]):
                    j += 1
                stale_end = times[j] if j < len(times) else times[-1]+5000
                stale_intervals.append(dict(motor=motor, start=stale_start, end=stale_end))
            warnings = failures = 0
            ground_clearances = []
            largest_gap = abs(stair['left_length_m'][source_ids]-stair['right_length_m'][source_ids])
            preview_frame = int(largest_gap.argmax())
            for k, (at, source_id, net_id) in enumerate(zip(grid, source_ids, net_ids)):
                pose.data.qpos[:7] = [0., 0., .55, 1., 0., 0., 0.]
                solved = [pose._solve_leg(side, model_angles[k, side*2:side*2+2]) for side in (0, 1)]
                failures += int(not all(solved))
                warnings += int(any(pose.status[i] == 'MODEL RANGE WARNING' for i in (1, 2)))
                measured_rotation = Rotation.from_quat(quat[source_id, [1, 2, 3, 0]])
                pose.data.qpos[2] = base_height+displacement[k]
                pose.data.qpos[3:7] = (heading*measured_rotation).as_quat()[[3, 0, 1, 2]]
                pose.data.qpos[pose.wheel_q] = wheel_q[k]
                mujoco.mj_kinematics(model, pose.data)
                mujoco.mj_camlight(model, pose.data)
                local = (pose.data.xpos[wheel_ids]-pose.data.xanchor[hip_ids]) @ pose.data.xmat[base_id].reshape(3, 3)
                reconstructed_length = np.linalg.norm(local[:, (0, 2)], axis=1)
                recorded_length = np.array([stair[n+'_length_m'][source_id] for n in ('left', 'right')])
                measured = recorded_length > 0
                length_errors.extend(abs(reconstructed_length[measured]-recorded_length[measured]).tolist())
                below_ground = False
                if ground:
                    clearance = min(float(mesh_vertices(model, pose.data, gid)[:, 2].min()) for gid in wheel_collision_geoms)
                    ground_clearances.append(clearance)
                    below_ground = clearance < -.005
                frozen = [item['motor'] for item in stale_intervals if item['start'] <= at < item['end']]
                model.geom_rgba[:] = colors
                if any(name in ('lf', 'lr') for name in frozen):
                    model.geom_rgba[left_geoms, 3] = .4
                if any(name in ('rf', 'rr') for name in frozen):
                    model.geom_rgba[right_geoms, 3] = .4
                canvas = Image.new('RGB', (1120, 656), '#17212c')
                for column, (azimuth, label) in enumerate(((0., '正面'), (90., '侧面'))):
                    camera.azimuth = azimuth
                    renderer.update_scene(pose.data, camera=camera, scene_option=option)
                    if ground:
                        reference_ground(renderer.scene, below_ground)
                    canvas.paste(Image.fromarray(renderer.render()), (column*560, 136))
                    ImageDraw.Draw(canvas).text((column*560+12, 144), label, font=font, fill='white', stroke_width=1, stroke_fill='black')
                    if estimated is not None and not ground:
                        # 显示触发时IMU高度的参考线，该线不代表地面。
                        delta = .55-camera.lookat[2]
                        elevation = np.radians(camera.elevation)
                        focal = 216/np.tan(np.radians(model.vis.global_.fovy/2))
                        y = int(216-focal*delta*np.cos(elevation)/(camera.distance-delta*np.sin(elevation)))
                        if 0 <= y < 432:
                            overlay = ImageDraw.Draw(canvas)
                            for x in range(column*560, (column+1)*560, 20):
                                overlay.line((x, 136+y, x+10, 136+y), fill='#8c9aa8', width=1)
                            overlay.text((column*560+12, 136+y+4), 'Δh=0（触发时）', font=small, fill='#c8d8e6', stroke_width=1, stroke_fill='black')
                draw = ImageDraw.Draw(canvas)
                draw.text((14, 8), f'实机日志动作回放  {case_index+1}/{len(report["episodes"])}  |  请求 {episode["request_sequence"]}', font=big, fill='white')
                model_id = net['model'][net_id]
                speed = net['obs_6'][net_id]/(3 if model_id == '5200' else 2)
                phase = PHASE[net['phase'][net_id]]
                if at-net['board_time_us'][net_id] > 30000:
                    phase = '推理记录缺口 / 末次'+phase
                draw.text((14, 51), f'{phase}  |  时间 {(at-start)/1e6:+.3f}s  |  前进指令 {speed:.2f}m/s  |  机高命令 {net["obs_8"][net_id]/5:.2f}m', font=font, fill='#d9e6ef')
                position_note = '机身位置固定'
                if estimated is not None:
                    position_note = f'IMU估算 Δh {displacement[k]*100:+.1f}cm  /  v_z {vertical_speed[k]:+.2f}m/s'
                draw.text((14, 94), position_note+'  ·  0.25×慢放  ·  蓝左 / 橙右', font=small, fill='#a8d5ec')
                lengths = [stair[n+'_length_m'][source_id] for n in ('left', 'right')]
                left = f'{lengths[0]*100:.1f}cm' if lengths[0] > 0 else '无效'
                right = f'{lengths[1]*100:.1f}cm' if lengths[1] > 0 else '无效'
                # 按四元数定义横滚，避免导出字段的pitch/roll标签互换。
                roll = np.degrees(measured_rotation.as_euler('xyz')[0])
                fault = int(stair['fault'][source_id])
                draw.text((14, 579), f'日志腿长  左 {left} / 右 {right}  |  侧倾 {roll:+.1f}°  |  故障码 {fault}', font=font, fill='white')
                status = '记录姿态与关节角重建；机身平移和离地高度未测量'
                if estimated is not None:
                    status = '高度是IMU测点相对触发时的双积分估算，含漂移；参考线不代表地面'
                    if model_id != '5200' and at > start:
                        status = '已返回WALK：保留原始积分漂移，此段高度低可信'
                if ground:
                    status = '地面按触发前轮底对齐；竖直位移来自IMU积分估算'
                    if below_ground:
                        status = f'估算轮底低于参考地面 {-clearance*100:.1f}cm：积分漂移，地面淡化显示'
                if frozen:
                    names = {'lf': '左前', 'lr': '左后', 'rf': '右前', 'rr': '右后'}
                    status = '、'.join(names[n] for n in frozen)+'反馈冻结：半透明侧含陈旧角度，实际姿态未知'
                elif fault:
                    status = '故障期间：显示日志反馈，实际姿态需现场观测核对'
                elif not all(solved):
                    status = '闭链重建失败：对应侧保留上一显示姿态'
                elif any(pose.status[i] == 'MODEL RANGE WARNING' for i in (1, 2)):
                    status = '记录姿态超出模型关节限位；按日志几何展示'
                draw.text((14, 617), status, font=small, fill='#ffba87' if frozen or fault else '#c4d2dc')
                slow.append_data(np.asarray(canvas));frames['slow'] += 1
                if k % 4 == 0:
                    normal_frame = canvas.copy()
                    d = ImageDraw.Draw(normal_frame)
                    d.rectangle((0, 90, 1120, 130), fill='#17212c')
                    d.text((14, 94), position_note+'  ·  1×正常速度  ·  蓝左 / 橙右', font=small, fill='#a8d5ec')
                    normal.append_data(np.asarray(normal_frame));frames['normal'] += 1
                if k == preview_frame:
                    canvas.save(output/f'case{case_index+1}-preview.png')
                all_poses.append(pose.data.qpos.copy());all_times.append(at);all_cases.append(case_index)
                closure.append(pose.residual_m.copy())
            cases.append(dict(request_sequence=episode['request_sequence'], frames=len(grid),
                ground_reference=ground_reference,
                estimated_wheel_clearance_range_m=[float(min(ground_clearances)), float(max(ground_clearances))] if ground else None,
                below_reference_ground_frames=int(np.count_nonzero(np.asarray(ground_clearances) < -.005)) if ground else 0,
                rendered_relative_height_range_m=[float(displacement.min()), float(displacement.max())],
                source_interval_relative_s=[float((grid[0]-start)/1e6), float((grid[-1]-start)/1e6)],
                closure_failed_frames=failures, model_limit_warning_frames=warnings,
                stale_feedback_intervals=[dict(motor=item['motor'], start_s=float((item['start']-start)/1e6),
                                               end_s=float((item['end']-start)/1e6)) for item in stale_intervals]))
            print('Rendered', episode['request_sequence'], len(grid), 'frames; closure failures', failures, flush=True)
    np.savez_compressed(output/'displayed-poses.npz', qpos=np.asarray(all_poses), board_us=all_times,
                        case=all_cases, closure_error_m=np.asarray(closure, dtype=float))
    assert max(length_errors) < .0001
    videos = []
    for name, key, speed in [('normal.mp4', 'normal', 1.), ('slow-4x.mp4', 'slow', .25)]:
        reader = imageio.get_reader(output/name)
        metadata = reader.get_meta_data()
        count = sum(1 for _ in reader)
        reader.close()
        assert count == frames[key] and metadata['fps'] == 50
        videos.append(dict(file=name, frames=count, fps=50, duration_s=count/50, speed=speed, full_decode_passed=True))
    source = archive_path if archive_path is not None else recording
    write_json(output/'report.json', dict(source=str(source.resolve()), videos=videos, cases=cases,
        archive_sha256=sha(archive_path) if archive_path is not None else None,
        height_estimate_source=str(height_estimate.resolve()) if height_estimate is not None else None,
        height_estimate_sha256=sha(height_estimate) if height_estimate is not None else None,
        attitude_label_source='Roll about model X is derived from recorded wxyz quaternion.',
        ground_rendered=ground, ground_has_physics=False,
        mapping_check_pairs=int(valid.sum()), model_angle_mapping_max_error_rad=mapping_error,
        recorded_length_geometry_max_error_m=max(length_errors),
        model_sha256=sha(MODEL), mapping_source=str(H7_PROJECT_CONFIGS['H7_ros']),
        physics_steps=0, policy_forwards=0, training_updates=0,
        assumptions=[('Root XY fixed; root Z is the per-case display offset plus integrated IMU vertical displacement; sensor lever arm is unmeasured.'
                      if estimated is not None else 'Root XYZ fixed at 0,0,0.55m for display; no recorded translation or jump height.'),
            'Recorded quaternion used; initial yaw removed per clip for a consistent camera.',
            'Passive joints solved from linkage geometry; model-range warnings retained.',
            'Wheel angle integrated from recorded speed with arbitrary zero initial angle.',
            'Nearest measured sample at 5ms display cadence; frozen motor feedback explicitly marked.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--recording', type=Path)
    inputs.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height-estimate', type=Path)
    parser.add_argument('--ground', action='store_true')
    args = parser.parse_args()
    run(args.recording, args.output, archive_path=args.archive, height_estimate=args.height_estimate, ground=args.ground)
