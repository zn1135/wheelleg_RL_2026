"""第二跳实测机身姿态初始化后的 GPU 姿态与触地对照。"""
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
from .report_full_jump import replay

LABELS = {'walk':'5200 / 6838 闭环','recorded':'回放实机动作','recorded_torque':'回放实机力矩请求'}
PREVIOUS = ROOT/'artifacts/fudan_ppo_jump/20261125-fulljump-swing-02'


def body_angles(q):
    return Rotation.from_quat(q[..., [4,5,6,3]].reshape(-1,4)).as_euler('xyz',degrees=True).reshape(*q.shape[:-1],3)


def run(out):
    cap = json.loads((out/'capture.json').read_text()); alignment = json.loads(ALIGN.read_text())
    assert cap['attitude_study'] and cap['archive_sha']==alignment['archive_sha256']==sha(ARCHIVE)
    second=next(t for t in alignment['trials'] if t['trial']==2)
    window=1000*(np.array(second['touchdown_bracket_frames'])/alignment['fps']-second['video_offset_s'])
    walk_ms=round(cap['real_walk_s']*1000)
    with zipfile.ZipFile(ARCHIVE) as archive:
        s=read_csv(archive,'试次2-H7反馈.csv')
    time=s['rel_s']*1000
    real_angles=Rotation.from_quat(matrix(s,[f'quat_{j}' for j in (1,2,3,0)])).as_euler('xyz',degrees=True)
    with np.load(out/'gpu.npz') as raw:
        q=raw['q'];trace=raw['trace'];angles=body_angles(q);rows=[]
        assert np.isfinite(q).all() and np.isfinite(trace).all()
        for j,case in enumerate(cap['cases']):
            start=walk_ms+case['offset_ms'];ticks=start+np.arange(len(q))*10; substeps=start+np.arange(len(trace))
            expected=np.array([np.interp(start,time,real_angles[:,axis]) for axis in (0,1)])
            init_error=float(abs(expected-angles[0,j,:2]).max());assert init_error<.02
            matching=[k for k,c in enumerate(cap['cases']) if c['offset_ms']==case['offset_ms'] and c['scenario']==case['scenario']]
            assert all(np.array_equal(raw[key][k],raw[key][j]) for k in matching for key in ('q0','v0','initial_obs'))
            wheel=trace[:,j,8:10]>.5; hit=np.flatnonzero(wheel.any(1)); assert len(hit)>0
            land=int(substeps[hit[0]])
            # 触地角精度受10ms姿态网格限制，载荷事件为1ms。
            impact=[float(np.interp(land,ticks,angles[:,j,k])) for k in (0,1)]
            done=raw['done_ms'][j]
            done_abs=int(walk_ms+done) if np.isfinite(done) else None
            mask=(ticks>=walk_ms)&(ticks<=walk_ms+250)
            if done_abs is not None:mask&=ticks<=done_abs
            before=(substeps>=start)&(substeps<=walk_ms+250)
            prior = trace[before,j,10]>1
            values={f'{angle}_{ms}ms_deg':float(np.interp(ms,ticks,angles[:,j,axis])) for ms in (550,610,710,810,960) for axis,angle in enumerate(('roll','pitch'))}
            rows.append(dict(**case,start_ms=start,init_roll_deg=float(angles[0,j,0]),init_pitch_deg=float(angles[0,j,1]),init_angle_error_deg=init_error,
                         contact_ms=land,first_wheel='both' if wheel[hit[0]].all() else ('left' if wheel[hit[0],0] else 'right'),
                         contact_roll_deg=impact[0],contact_pitch_deg=impact[1],
                         pitch_post_walk_max_deg=float(angles[mask,j,1].max()) if mask.any() else None,
                         pitch_post_walk_min_deg=float(angles[mask,j,1].min()) if mask.any() else None,
                         nonwheel_contact_through_960_ms=int(prior.sum()),fall_termination_ms=done_abs,**values))
        nominal={r['arm']:j for j,r in enumerate(rows) if r['start_ms']==520 and r['scenario']=='imu'}
    with (out/'cases.csv').open('x') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    real_at={str(ms):[float(np.interp(ms,time,real_angles[:,axis])) for axis in (0,1)] for ms in (320,460,520,550,610,710,810,960)}
    real_mask=(time>=walk_ms)&(time<=walk_ms+250)
    summary=dict(real=dict(roll_pitch_by_ms=real_at,touch_window_ms=window.tolist(),pitch_max_post_walk_250_deg=float(real_angles[real_mask,1].max())),
                 nominal_520ms={arm:rows[j] for arm,j in nominal.items()},case_count=len(rows),paired_initial_states_exact=True,
                 nominal_other_starts=[r for r in rows if r['scenario']=='imu' and r['start_ms']!=520],
                 logged_torque_max_error_Nm=cap['logged_torque_max_error_Nm'],optimizer_updates=0)
    write_json(out/'report.json',summary)
    with np.load(PREVIOUS/'gpu.npz') as previous:
        before=body_angles(previous['q'][:,1]); clock=np.arange(len(before))*10
    plt.rcParams.update({'font.family':['Noto Sans CJK SC'],'axes.unicode_minus':False})
    fig,axs=plt.subplots(2,1,figsize=(10,6.5),sharex=True,constrained_layout=True)
    for axis,ax in enumerate(axs):
        ax.plot(time,real_angles[:,axis],color='black',lw=2,label='实机 IMU')
        ax.plot(clock,before[:,axis],color='#a6a6a6',ls='--',label='上一轮只复用摆角目标')
        for arm,j in nominal.items():
            start=rows[j]['start_ms'];ticks=start+np.arange(len(q))*10
            cutoff=(ticks<=rows[j]['fall_termination_ms']) if rows[j]['fall_termination_ms'] is not None else np.ones(len(ticks),bool)
            ax.plot(ticks[cutoff],angles[cutoff,j,axis],label=LABELS[arm])
        ax.axvspan(*window,alpha=.15,color='tab:orange');ax.axvline(walk_ms,color='grey',ls=':');ax.axvline(520,color='grey',ls='--')
        ax.set_ylabel(('横滚 / °','俯仰 / °')[axis]);ax.grid(alpha=.2);ax.set_xlim(350,1000)
    axs[0].legend(ncol=2,fontsize=9);axs[-1].set_xlabel('切入 Jump 后 / ms，阴影为实拍触地窗口，点线为切回 WALK')
    fig.savefig(out/'attitude.png',dpi=145);plt.close(fig)
    video_path=ROOT/'artifacts/fudan_ppo_jump/20261125-ground-replay-01/source/real-video-1.mp4'
    assert sha(video_path)==alignment['video_sha256']
    indexes={arm:nominal[arm] for arm in ('walk','recorded_torque')}
    # 录像从520ms处开始，转换到该片段的时间原点。
    video_rows=[dict(walk_switch_ms=walk_ms-520) for _ in rows]
    replay(out,q[:79],mujoco.MjModel.from_binary_path(str(out/'model.mjb')),indexes,video_rows,video_path,second['video_offset_s']+.52,cap['real_walk_s']-.52,LABELS,clock_start_ms=520)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out)
