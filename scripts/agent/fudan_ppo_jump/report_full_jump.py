"""比较第二跳的实拍与从切入 Jump 开始的 GPU 闭环轨迹。"""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial.transform import Rotation

from .analyze_real022 import matrix, read_csv
from .compare_landing_lead import ALIGN, ARCHIVE
from .measure_flat import intervals
from .probe_takeoff_work import ROOT, sha, write_json

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABELS = {'walk': '原时刻切 WALK', 'wait': '等待后切 WALK', 'recorded': '实机动作输入'}


def metrics(trace, pitch, switch_ms, done_ms, walk_ms):
    # 初态来自实机，允许首跳早于常规复位后 200 ms 的统计门槛。
    wheels = (trace[:, 8:10] > .5).any(1)
    body = trace[:, 10] > 1.
    flight = next(((a, b) for a, b in intervals(~wheels)
                   if a >= 20 and wheels[:a].any() and b-a >= 40 and trace[a:b, 6:8].min(1).max() > .02), None)
    lift, land, height = None, None, None
    if flight is not None:
        a, b = flight
        impact = np.flatnonzero(body[a:b])
        end = a + int(impact[0]) if len(impact) else b
        lift = int(a)
        land = int(b) if b < len(trace) and not len(impact) else None
        if end > a:
            height = float(trace[a:end, 6:8].min(1).max()*100)
    first_body = np.flatnonzero(body)
    window = slice(walk_ms//10, (walk_ms+250)//10+1)
    return dict(first_liftoff_ms=lift, first_wheel_touchdown_ms=land, lower_wheel_peak_cm=height,
                walk_switch_ms=round(walk_ms+float(switch_ms)),
                positive_pitch_peak_250_deg=float(pitch[window].max()),
                absolute_pitch_peak_250_deg=float(abs(pitch[window]).max()),
                first_nonwheel_contact_ms=int(first_body[0]) if len(first_body) else None,
                nonwheel_contact_ms=int(body.sum()),
                fall_termination_ms=round(walk_ms+float(done_ms)) if np.isfinite(done_ms) else None)


def replay(out, q, model, case_indices, rows, video_path, offset, walk_s, labels=LABELS, *, clock_start_ms=0):
    font = ImageFont.truetype(FONT, 19)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    yaw = Rotation.from_quat(q[0, 0, [4, 5, 6, 3]]).as_euler('xyz', degrees=True)[2]
    camera.azimuth, camera.elevation, camera.distance = yaw+90, -11, 1.45
    target = out/'full-jump-slow.mp4'
    with imageio.get_reader(video_path) as source, mujoco.Renderer(model, height=288, width=352) as renderer, \
            imageio.get_writer(target, fps=25, codec='libx264', quality=8) as writer:
        source_fps = source.get_meta_data()['fps']
        assert abs(source_fps-30) < .01
        for tick, poses in enumerate(q):
            at = tick*.01
            canvas = Image.new('RGB', (1056, 384), '#192737')
            draw = ImageDraw.Draw(canvas)
            frame = Image.fromarray(source.get_data(round((offset+at)*source_fps)))
            canvas.paste(ImageOps.pad(frame, (352,288), color='black', method=Image.Resampling.LANCZOS), (0,44))
            draw.text((8,10), '实机第二跳', font=font, fill='white')
            draw.text((8,339), 'WALK' if at >= walk_s else 'Jump', font=font, fill='#bcd8ed')
            for col, (arm, index) in enumerate(case_indices.items(), start=1):
                pose = poses[index]
                data.qpos[:] = pose
                mujoco.mj_forward(model,data)
                camera.lookat[:] = [pose[0], pose[1], .36]
                renderer.update_scene(data,camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (352*col,44))
                draw.text((352*col+8,10), 'GPU · '+labels[arm], font=font, fill='white')
                switched = tick*10 >= rows[index]['walk_switch_ms']
                draw.text((352*col+8,339), 'WALK' if switched else 'Jump', font=font, fill='#bcd8ed')
            draw.text((8,361), f'触发后 {clock_start_ms+tick*10:4d} ms   ·   0.25×慢放   ·   实拍按动作事件对齐', font=font, fill='white')
            writer.append_data(np.asarray(canvas))
    with imageio.get_reader(target) as reader:
        assert reader.get_meta_data()['fps']==25 and sum(1 for _ in reader)==len(q)


def run(out):
    cap = json.loads((out/'capture.json').read_text())
    assert cap['full_jump'] and cap['archive_sha']==sha(ARCHIVE)
    alignment = json.loads(ALIGN.read_text())
    assert alignment['archive_sha256']==cap['archive_sha']
    second = next(t for t in alignment['trials'] if t['trial']==2)
    lift_range = np.array(second['takeoff_bracket_frames'])/30-second['video_offset_s']
    land_range = np.array(second['touchdown_bracket_frames'])/30-second['video_offset_s']
    walk_ms = round(cap['real_walk_s']*1000)
    with np.load(out/'gpu.npz') as raw:
        q, trace = raw['q'], raw['trace']
        pitch = Rotation.from_quat(q[..., [4,5,6,3]].reshape(-1,4)).as_euler('xyz',degrees=True)[:,1].reshape(q.shape[:2])
        rows = [dict(**case, **metrics(trace[:,j], pitch[:,j], raw['switch_ms'][j], raw['done_ms'][j], walk_ms))
                for j,case in enumerate(cap['cases'])]
        for j in range(0,len(rows),3):
            for k in (j+1,j+2):
                assert all(np.array_equal(raw[key][j],raw[key][k]) for key in ('q0','v0','initial_obs'))
        assert trace.shape==(cap['steps']*10,len(rows),23) and np.isfinite(trace).all()
    with (out/'cases.csv').open('x') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    case_indices = {arm:next(j for j,c in enumerate(rows) if c['scenario']=='imu' and c['arm']==arm) for arm in LABELS}
    with zipfile.ZipFile(ARCHIVE) as z:
        s=read_csv(z,'试次2-H7反馈.csv')
    t=s['rel_s'];real_pitch=Rotation.from_quat(matrix(s,[f'quat_{j}' for j in (1,2,3,0)])).as_euler('xyz',degrees=True)[:,1]
    real_window=(t>=cap['real_walk_s'])&(t<=cap['real_walk_s']+.25)
    write_json(out/'report.json',dict(real_liftoff_range_ms=(1000*lift_range).tolist(),real_touchdown_range_ms=(1000*land_range).tolist(),
        real_walk_ms=walk_ms,real_positive_pitch_peak_250_deg=float(real_pitch[real_window].max()),
        nominal={arm:rows[j] for arm,j in case_indices.items()},case_count=len(rows),paired_initial_states_exact=True))
    plt.rcParams.update({'font.family':['Noto Sans CJK SC'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    axes[0].plot(t,real_pitch,color='black',label='实机姿态')
    for arm in ('walk','wait'):
        j=case_indices[arm]
        axes[0].plot(np.arange(len(q))*.01,pitch[:,j],label=LABELS[arm])
        axes[1].plot(np.arange(len(trace))*.001,trace[:,j,6:8].min(1)*100,label=LABELS[arm])
    for ax in axes:
        ax.axvspan(*land_range,alpha=.14,color='#b65523',label='实拍触地窗口');ax.axvline(walk_ms/1000,color='grey',ls=':',label='实机切 WALK')
        ax.legend(ncol=2,fontsize=9);ax.grid(alpha=.2);ax.set_xlim(0,1.30)
    axes[0].set_ylabel('俯仰 / °');axes[1].set_ylabel('GPU 较低轮底高度 / cm');axes[1].set_xlabel('切入 Jump 后 / s')
    fig.savefig(out/'full-jump.png',dpi=145);plt.close(fig)
    video_path = ROOT/'artifacts/fudan_ppo_jump/20261125-ground-replay-01/source/real-video-1.mp4'
    assert sha(video_path)==alignment['video_sha256']
    replay(out,q,mujoco.MjModel.from_binary_path(str(out/'model.mjb')),
           {arm:case_indices[arm] for arm in ('walk','wait')},rows,video_path,second['video_offset_s'],cap['real_walk_s'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
