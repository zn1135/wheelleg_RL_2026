"""Render every failed final-stage case from saved GPU poses, without physics steps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((root/'checkpoint-audit.json').read_text())
    checkpoint = root/'checkpoint/model_5000.pt'
    assert audit['iteration'] == 5000 and sha(checkpoint) == audit['checkpoint_sha256']
    diagnostic = {r['sample_id']: r for r in json.loads(
        (root/'high-yaw/5000/attitude-diagnostic.json').read_text())['rows']}
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 20)
    title_font = ImageFont.truetype(font_path, 25)
    rows = []
    reel = args.output/'all_failures_1x.mp4'
    chapter = 0
    forward_calls = 0
    with imageio.get_writer(reel, fps=50, codec='libx264', quality=8) as combined:
        for recording, acceptance in (
            (root/'nominal/gpu', root/'nominal/acceptance/acceptance.json'),
            (root/'high-yaw/5000', root/'high-yaw/5000/acceptance/acceptance.json'),
        ):
            scores = json.loads(acceptance.read_text())
            protocol = json.loads((recording/'protocol.json').read_text())
            samples = {s['id']: s for s in protocol['samples']}
            model_path = recording/'fixtures/model.mjb'
            model = mujoco.MjModel.from_binary_path(str(model_path))
            disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth = 768
            model.vis.global_.offheight = 448
            data = mujoco.MjData(model)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.azimuth = 1.5, -12., 90.
            option = mujoco.MjvOption()
            option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
            with mujoco.Renderer(model, width=768, height=448) as renderer:
                for score in scores['samples']:
                    if score['passed']:
                        continue
                    sid = score['sample_id']
                    case = samples[sid]['case']
                    command = case['segments'][0]['command']
                    duration = case['duration']
                    assert duration == 7 and not score['v1_diagnostic']['fell']
                    trace_path = recording/'traces'/(sid+'.npz')
                    with np.load(trace_path) as saved:
                        trace = {k: saved[k].copy() for k in saved.files}
                    phase = score['motion_phases'][0]
                    yaw_mae = phase['metrics']['yaw_mae']
                    xy_peak = score['v1_diagnostic']['last_second_measurements']['angular_xy_max']
                    # 直接核对原评分窗口；渲染不重标通过，也不推进物理。
                    selected = (trace['time'] >= 2) & (trace['time'] < 7)
                    assert abs(np.abs(trace['angular_body'][selected, 2]-command[1]).mean()-yaw_mae) < 1e-6
                    times = np.arange(round(duration*50))/50
                    pose_indices = np.searchsorted(trace['qpos_time'], times-1e-9)
                    assert np.allclose(trace['qpos_time'][pose_indices], times, atol=1e-6)
                    output = args.output/(sid+'.mp4')
                    with imageio.get_writer(output, fps=50, codec='libx264', quality=8) as writer:
                        for t, i in zip(times, pose_indices):
                            tick = int(np.searchsorted(trace['time'], t-1e-9))
                            data.qpos[:] = trace['qpos'][i]
                            mujoco.mj_forward(model, data)
                            forward_calls += 1
                            cam.lookat[:] = data.qpos[:3]
                            renderer.update_scene(data, camera=cam, scene_option=option)
                            canvas = Image.new('RGB', (768, 704), '#18212b')
                            canvas.paste(Image.fromarray(renderer.render().copy()), (0, 192))
                            draw = ImageDraw.Draw(canvas)
                            draw.text((16, 6), f'5000轮失败回放 {len(rows)+1}/6 | 正常1倍速 | {t:.2f}/7.00秒', font=title_font, fill='white')
                            draw.text((16, 45), f'目标 vx {command[0]:g} m/s   yaw {command[1]:+g} rad/s   高度 {command[2]:.2f} m', font=font, fill='white')
                            draw.text((16, 78), f'实测 vx {trace["linear_body"][tick, 0]:+.3f} m/s   yaw {trace["angular_body"][tick, 2]:+.3f} rad/s', font=font, fill='white')
                            draw.text((16, 112), f'2～7秒 yaw平均绝对误差：{yaw_mae:.4f} / 门槛0.1000 rad/s', font=font, fill='#ff8e8e' if yaw_mae > .1 else '#a9edb0')
                            draw.text((16, 145), f'6～7秒 机体系XY角速度峰：{xy_peak:.4f} / 门槛0.5000 rad/s', font=font, fill='#ff8e8e' if xy_peak > .5 else '#a9edb0')
                            label = '响应阶段0～2秒' if t < 2 else '速度评分阶段2～7秒'
                            if t >= 6:
                                label += '；同时检查末1秒姿态'
                            draw.text((16, 644), label, font=font, fill='white')
                            note = '完整GPU轨迹；仅离线渲染；未改验收结果'
                            if sid in diagnostic:
                                value = diagnostic[sid]['gravity_perpendicular_last1_peak_rad_s']
                                note = f'去除竖直转动后的姿态变化峰 {value:.3f} rad/s（诊断值，未重评分）'
                            draw.text((16, 674), note, font=font, fill='#c8d5df')
                            frame = np.asarray(canvas)
                            writer.append_data(frame)
                            combined.append_data(frame)
                            if abs(t-6) < 1e-8:
                                canvas.save(output.with_suffix('.png'))
                    row = dict(sample_id=sid, command=command, failure_reasons=score['failure_reasons'],
                        yaw_mae_rad_s=yaw_mae, body_xy_last1_peak_rad_s=xy_peak,
                        chapter_start_s=chapter, duration_s=duration, fps=50, expected_frames=len(times),
                        video=str(output.resolve()), video_sha256=sha(output),
                        trajectory=str(trace_path), trajectory_sha256=sha(trace_path),
                        model=str(model_path), model_sha256=sha(model_path),
                        acceptance=str(acceptance), acceptance_sha256=sha(acceptance))
                    rows.append(row)
                    chapter += duration
                    print(f'Rendered {sid}', flush=True)
    assert len(rows) == 6
    # 全量解码核对帧数与速度；整段保留每例0～7秒。
    for path, expected in [(Path(r['video']), r['expected_frames']) for r in rows]+[(reel, 2100)]:
        with imageio.get_reader(path) as reader:
            fps = reader.get_meta_data()['fps']
            count = sum(1 for _ in reader)
        assert fps == 50 and count == expected, (path, fps, count)
    index = dict(checkpoint=str(checkpoint), checkpoint_sha256=audit['checkpoint_sha256'],
        playback_speed=1, fps=50, duration_s=chapter, physics_steps=0, optimizer_updates=0,
        rendering_forward_calls=forward_calls, full_decode_verified=True,
        reel_sha256=sha(reel), cases=rows)
    (args.output/'video-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'render-source.py')
    lines = ['# 5000轮：全部6个失败工况', '', '每例完整7秒，正常1倍速；合辑42秒。红字为超标项，数值为完整评分窗口统计。均未摔倒，原评分保持。', '',
        '[播放全部失败工况](all_failures_1x.mp4)', '', '| 合辑起点 | vx / yaw / 高度 | yaw误差（≤0.1） | XY角速度峰（≤0.5） | 回放 |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["chapter_start_s"]:g}秒 | {r["command"]} | {r["yaw_mae_rad_s"]:.5f} | {r["body_xy_last1_peak_rad_s"]:.5f} | [{r["sample_id"]}]({r["sample_id"]}.mp4) |')
    lines += ['', '高速反向三例的XY角速度含机身倾斜造成的竖直偏航投影；视频底部列出的去投影值仅作诊断，未将其改判为通过。', '',
        '推荐先核对高速姿态评分的坐标定义，再单独诊断真实yaw误差；当前结果不足以支持直接追加训练。', '',
        '来源、模型和视频SHA256、完整解码检查见video-index.json。只有读取轨迹及mj_forward渲染，无物理积分、策略推理、训练或新验收。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Verified six clips and 42-second reel', flush=True)


if __name__ == '__main__':
    main()
