"""全程 GPU 摆角目标对照的时序、落地统计和并排慢放。"""
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

LABELS = {'right_wheel_log':'原摆角', 'logged_swing':'复用实机摆角目标',
          'center_swing':'触地前摆回身下', 'logged_then_center':'先后摆，再摆回身下'}
DIRECTION = np.array([-180/np.pi,180/np.pi])


def swing(hips):
    return hips.reshape(*hips.shape[:-1],2,2).mean(-1)*DIRECTION


def run(out):
    cap=json.loads((out/'capture.json').read_text());alignment=json.loads(ALIGN.read_text())
    assert cap['swing_study'] and cap['archive_sha']==alignment['archive_sha256']==sha(ARCHIVE)
    real = next(trial for trial in alignment['trials'] if trial['trial']==2)
    touch = 1000*(np.array(real['touchdown_bracket_frames'])/alignment['fps']-real['video_offset_s'])
    walk_ms = round(cap['real_walk_s']*1000)
    with zipfile.ZipFile(ARCHIVE) as archive:
        pd, feedback, net = [read_csv(archive,'试次2-'+name+'.csv') for name in ('H7控制','H7反馈','模型输入动作')]
    real_time = feedback['rel_s']*1000
    control_time = pd['rel_s']*1000
    real_angle = np.degrees(matrix(feedback,['left_leg_1','right_leg_1']))
    real_target = swing(matrix(pd,[f'target_{i}' for i in range(4)]))
    real_from_hip = swing(matrix(pd,[f'current_{i}' for i in range(4)]))
    mask = (control_time>=250)&(control_time<=touch[0])
    pairing = np.abs(control_time[mask,None]-real_time).argmin(1)
    coordinate_error = float(abs(real_from_hip[mask]-real_angle[pairing]).max())
    assert coordinate_error < .1
    early = np.argmin(abs(control_time[:,None]-[280,350,500]),axis=0)
    real_samples = [{'ms':float(control_time[i]),'target_deg':real_target[i].tolist(),'actual_deg':real_from_hip[i].tolist()} for i in early]
    orient = matrix(feedback,[f'quat_{j}' for j in (1,2,3,0)])
    real_pitch = Rotation.from_quat(orient).as_euler('xyz',degrees=True)[:,1]
    model=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
    hip_ids=[model.joint('robot/'+n).qposadr[0] for n in ('lf0','lf00','rf0','rf00')]
    wheel_id=model.joint('robot/rfwheel').dofadr[0]
    with np.load(out/'gpu.npz') as raw,np.load(out/'swing-targets.npz') as interventions:
        q,trace=raw['q'],raw['trace'];clock=np.arange(len(q))*10
        pitch=Rotation.from_quat(q[..., [4,5,6,3]].reshape(-1,4)).as_euler('xyz',degrees=True)[:,1].reshape(q.shape[:2])
        leg_angle = swing(q[:,:,hip_ids]); targets = swing(interventions['applied']); active = interventions['active']
        before, after = interventions['before'], interventions['applied']
        extension_error = float(abs(np.diff(before.reshape(*before.shape[:2],2,2),axis=-1)-np.diff(after.reshape(*after.shape[:2],2,2),axis=-1)).max())
        assert extension_error<1e-6 and np.array_equal(before[~active],after[~active])
        assert not raw['mode'][active].any() and np.isfinite(trace).all()
        jump_idx = np.flatnonzero(net['model']=='5200')[0]
        wheel_tape = net['action_5'][jump_idx:jump_idx+len(q)].clip(-15,15)
        rows=[]
        for j,case in enumerate(cap['cases']):
            base=next(i for i,c in enumerate(cap['cases']) if c['scenario']==case['scenario'] and c['arm']=='right_wheel_log')
            assert all(np.array_equal(raw[key][base],raw[key][j]) for key in ('q0','v0','initial_obs'))
            wheel_active=raw['intervention'][:,j]
            np.testing.assert_allclose(raw['action'][wheel_active,j,5],wheel_tape[wheel_active],atol=1e-6,rtol=0)
            row=dict(**case,**metrics(trace[:,j],pitch[:,j],raw['switch_ms'][j],raw['done_ms'][j],walk_ms))
            land=row['first_wheel_touchdown_ms']; assert land is not None
            applied = clock[active[:,j]]
            # 策略拍开始时只能看到上一拍载荷，首次承重可与末次干预落在同一毫秒。
            assert (applied <= land).all()
            pre=(land-1)//10
            hit=trace[land,j,8:10]>.5
            pitch_post_walk=pitch[walk_ms//10:(walk_ms+250)//10+1,j]
            first_120=pitch[(clock>=land)&(clock<=land+120),j]
            row.update(first_wheel='both' if hit.all() else ('left' if hit[0] else 'right'),
                       swing_before_touchdown_deg=leg_angle[pre,j].tolist(),
                       target_before_touchdown_deg=targets[pre,j].tolist(),
                       right_speed_before_touchdown_rad_s=float(raw['v'][pre,j,wheel_id]),
                       signed_peak_after_walk_deg=float(pitch_post_walk[np.argmax(abs(pitch_post_walk))]),
                       pitch_touch_first_120_deg=[float(first_120.min()),float(first_120.max())],
                       swing_control_window_ms=applied[[0,-1]].tolist() if len(applied) else None)
            rows.append(row)
    with (out/'cases.csv').open('x') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    nominal={c['arm']:j for j,c in enumerate(rows) if c['scenario']=='imu'}
    write_json(out/'report.json',dict(real_samples=real_samples,real_touchdown_ms=touch.tolist(),
        nominal={arm:rows[i] for arm,i in nominal.items()},cases=rows,control_feedback_swing_match_deg=coordinate_error,
        extension_difference_error_rad=extension_error,paired_and_wheel_tape_verified=True,optimizer_updates=0))
    plt.rcParams.update({'font.family':['Noto Sans CJK SC'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for j,ax in enumerate(axes[:2]):
        ax.plot(control_time,real_target[:,j],color='black',ls='--',label='实机摆角目标')
        ax.plot(real_time,real_angle[:,j],color='black',label='实机摆角反馈')
    axes[2].plot(real_time,real_pitch,color='black',label='实机俯仰')
    for arm,j in nominal.items():
        for side,ax in enumerate(axes[:2]):ax.plot(clock,leg_angle[:,j,side],label=LABELS[arm])
        axes[2].plot(clock,pitch[:,j],label=LABELS[arm])
    for ax,label in zip(axes,('左腿摆角 / °','右腿摆角 / °','机身俯仰 / °')):
        ax.set_ylabel(label);ax.set_xlim(250,1050);ax.grid(alpha=.2);ax.axvspan(*touch,alpha=.12,color='tab:orange');ax.axvline(walk_ms,color='grey',ls=':')
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('切入 Jump 后 / ms，阴影为实拍触地窗口，虚线为切回 WALK')
    fig.savefig(out/'swing-and-landing.png',dpi=145);plt.close(fig)
    source=ROOT/'artifacts/fudan_ppo_jump/20261125-ground-replay-01/source/real-video-1.mp4'
    assert sha(source)==alignment['video_sha256']
    replay(out,q,model,{arm:nominal[arm] for arm in ('logged_swing','logged_then_center')},rows,source,real['video_offset_s'],cap['real_walk_s'],LABELS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out)
