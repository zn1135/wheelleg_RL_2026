"""对比空中轮速干预、首次触轮和 WALK 交接后的实机与仿真轨迹。"""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .analyze_real022 import matrix, read_csv
from .compare_landing_lead import ALIGN, ARCHIVE
from .probe_takeoff_work import ROOT, sha, write_json
from .report_full_jump import metrics, replay

LABELS = {
    'walk': '原策略', 'onset_input': '只替换反转初期输入',
    'right_wheel_log': '复用右轮目标', 'both_wheels_log': '复用双轮目标',
    'right_wheel_log_wait': '复用右轮目标并等待', 'both_wheels_log_wait': '复用双轮目标并等待',
}


def pitch_degrees(q):
    return Rotation.from_quat(q[..., [4, 5, 6, 3]].reshape(-1, 4)).as_euler('xyz', degrees=True)[:, 1].reshape(q.shape[:-1])


def run(out):
    cap = json.loads((out/'capture.json').read_text())
    alignment = json.loads(ALIGN.read_text())
    assert cap['wheel_study'] and cap['archive_sha'] == alignment['archive_sha256'] == sha(ARCHIVE)
    second = next(trial for trial in alignment['trials'] if trial['trial'] == 2)
    touch_window = 1000*(np.asarray(second['touchdown_bracket_frames'])/alignment['fps']-second['video_offset_s'])
    walk_ms = round(cap['real_walk_s']*1000)
    with zipfile.ZipFile(ARCHIVE) as archive:
        inference, feedback, control = [read_csv(archive, f'试次2-{name}.csv') for name in ('模型输入动作', 'H7反馈', 'H7控制')]
    jump_start = np.flatnonzero(inference['model']=='5200')[0]
    real_t = feedback['rel_s']*1000
    real_pd_t = control['rel_s']*1000
    real_pitch = Rotation.from_quat(matrix(feedback,[f'quat_{j}' for j in (1,2,3,0)])).as_euler('xyz',degrees=True)[:,1]
    target_logged = matrix(inference, ['action_2','action_5'])[jump_start:jump_start+cap['steps']]*10
    model = mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
    right_wheel = model.joint('robot/rfwheel').dofadr[0]
    rows = []
    with np.load(out/'gpu.npz') as raw:
        q, trace = raw['q'], raw['trace']
        speeds = raw['v'][:, :, right_wheel]
        targets = (raw['action'][:, :, [2,5]]*10).clip(-150,150)
        intervention = raw['intervention']
        pitch = pitch_degrees(q)
        assert trace.shape == (cap['steps']*10,len(cap['cases']),23) and np.isfinite(trace).all()
        for j,case in enumerate(cap['cases']):
            base = next(i for i,c in enumerate(cap['cases']) if c['scenario']==case['scenario'] and c['arm']=='walk')
            assert all(np.array_equal(raw[key][base],raw[key][j]) for key in ('q0','v0','initial_obs'))
            row = dict(**case,**metrics(trace[:,j],pitch[:,j],raw['switch_ms'][j],raw['done_ms'][j],walk_ms))
            applied = np.flatnonzero(intervention[:,j])
            row['intervention_ticks'] = len(applied)
            arm = case['arm']
            if arm=='walk':
                assert not len(applied)
            else:
                assert len(applied)>0 and not raw['mode'][applied,j].any()
                assert (applied*10 >= row['first_liftoff_ms']).all() and (applied*10 < row['first_wheel_touchdown_ms']).all()
                if arm.startswith('both_wheels_log'):
                    np.testing.assert_allclose(targets[applied,j],target_logged[applied].clip(-150,150),atol=2e-5,rtol=0)
                if arm.startswith('right_wheel_log'):
                    np.testing.assert_allclose(targets[applied,j,1],target_logged[applied,1].clip(-150,150),atol=2e-5,rtol=0)
            row['right_target_180_rad_s'] = float(targets[18,j,1])
            row['right_speed_300_rad_s'] = float(speeds[30,j])
            # 取触地前最后一个策略拍，排除接触冲击造成的轮速跳变。
            pre_touch = (row['first_wheel_touchdown_ms']-1)//10
            row['right_speed_before_touchdown_rad_s'] = float(speeds[pre_touch,j])
            first_wheel = trace[row['first_wheel_touchdown_ms'],j,8:10] > .5
            row['first_wheel'] = 'both' if first_wheel.all() else ('left' if first_wheel[0] else 'right')
            before_handoff = trace[:row['walk_switch_ms'],j]
            row['nonwheel_contact_before_walk_ms'] = int((before_handoff[:,10]>1).sum())
            own_switch = row['walk_switch_ms']//10
            row['absolute_pitch_peak_250_after_own_switch_deg'] = float(abs(pitch[own_switch:own_switch+26,j]).max())
            rows.append(row)
    with (out/'cases.csv').open('x') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    nominal_indices = {row['arm']:j for j,row in enumerate(rows) if row['scenario']=='imu'}
    real_peak = float(abs(real_pitch[(real_t>=walk_ms)&(real_t<=walk_ms+250)]).max())
    real_before = control['velocity_5'][np.flatnonzero(real_pd_t<touch_window[0])[-1]]
    report = dict(real=dict(right_target_180_rad_s=float(target_logged[18,1]),
                            right_speed_300_rad_s=float(np.interp(300,real_pd_t,control['velocity_5'])),
                            right_speed_before_touchdown_rad_s=float(real_before), touch_window_ms=touch_window.tolist(),
                            absolute_pitch_peak_250_deg=real_peak),
                  nominal={arm:rows[j] for arm,j in nominal_indices.items()}, case_count=len(rows),
                  pairs_verified=True, logged_targets_verified=True)
    write_json(out/'report.json',report)

    plt.rcParams.update({'font.family':['Noto Sans CJK SC'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    clock=np.arange(len(q))*10
    axes[0].plot(clock,target_logged[:,1],color='black',label='实机')
    axes[1].plot(real_pd_t,control['velocity_5'],color='black',label='实机')
    axes[2].plot(real_t,real_pitch,color='black',label='实机')
    for arm in ('walk','onset_input','right_wheel_log','right_wheel_log_wait'):
        j=nominal_indices[arm]
        for ax,values in zip(axes,(targets[:,:,1],speeds,pitch)):
            ax.plot(clock,values[:,j],label=LABELS[arm])
    for ax,label in zip(axes,('右轮目标 / rad/s','右轮实测或仿真速度 / rad/s','俯仰 / °')):
        ax.set_ylabel(label);ax.set_xlim(100,1050);ax.grid(alpha=.2)
        ax.axvspan(*touch_window,alpha=.14,color='#b65523',label='实拍触地窗口');ax.axvline(walk_ms,color='grey',ls=':',label='实机切 WALK')
    axes[0].legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel('切入 Jump 后 / ms')
    fig.savefig(out/'flight-wheel.png',dpi=145);plt.close(fig)
    video_path=ROOT/'artifacts/fudan_ppo_jump/20261125-ground-replay-01/source/real-video-1.mp4'
    assert sha(video_path)==alignment['video_sha256']
    replay(out,q,model,{arm:nominal_indices[arm] for arm in ('walk','right_wheel_log')},
           rows,video_path,second['video_offset_s'],cap['real_walk_s'],LABELS)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
