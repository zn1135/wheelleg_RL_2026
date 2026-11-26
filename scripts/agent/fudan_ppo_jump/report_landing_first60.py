"""对齐第二跳的实机控制拍和1ms GPU记录，复核接触参数与姿态。"""
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
from .report_landing_attitude import body_angles


def pitch_rate(angles, omega):
    roll = np.deg2rad(angles[..., 0])
    return omega[..., 1]*np.cos(roll)-omega[..., 2]*np.sin(roll)


def load(out, digest):
    cap = json.loads((out/'capture.json').read_text())
    assert cap['archive_sha']==digest and cap['contact_profiles'] and cap['optimizer_updates']==0
    assert cap['dt_s']==.001 and cap['pd_ms']==5 and cap['policy_ms']==10
    with np.load(out/'gpu.npz') as gpu, np.load(out/'first-contact.npz') as fine:
        raw = {key:gpu[key] for key in ('q','v','trace','q0','v0','done_ms','initial_obs')}
        q, v, torque, wheel_force = (fine[key] for key in ('q','v','torque','wheel_force'))
    assert all(np.isfinite(a).all() for a in (q,v,torque,wheel_force,raw['trace']))
    np.testing.assert_array_equal(q[9::10][:-1],raw['q'][1:])
    np.testing.assert_array_equal(v[9::10][:-1],raw['v'][1:])
    np.testing.assert_array_equal(abs(wheel_force[...,2])>1,raw['trace'][...,8:10]>.5)
    assert (abs(torque)<=np.asarray([40,40,3.9,40,40,3.9])+1e-5).all()
    assert cap['logged_torque_max_error_Nm']<1e-4
    cases = {}
    for i, info in enumerate(cap['cases']):
        start = round(cap['real_walk_s']*1000)+info['offset_ms']
        key = (start,info['scenario'],info['arm'])
        if info['arm']=='walk':
            matching = [j for j,c in enumerate(cap['cases']) if c['offset_ms']==info['offset_ms']]
            for field in ('q0','v0','initial_obs'):
                np.testing.assert_array_equal(raw[field][matching],np.repeat(raw[field][i:i+1],len(matching),axis=0))
        angles = body_angles(q[:,i])
        cases[key] = dict(index=i,t=start+np.arange(1,len(q)+1),q=q[:,i],angles=angles,
                          rate=pitch_rate(angles,v[:,i,3:6]),v=v[:,i],torque=torque[:,i],
                          touch=raw['trace'][:,i,8:10]>.5,other_force=raw['trace'][:,i,10],
                          done=raw['done_ms'][i]+round(cap['real_walk_s']*1000))
    return cap, raw['q'], cases


def csv_rows(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_rows(cases, real_t, real_angles, left, tag):
    rows = []
    for (start,profile,arm), c in cases.items():
        t, angles = c['t'], c['angles']
        contacts = [t[mask][0].item() if mask.any() else None for mask in c['touch'].T]
        left_ms = contacts[0]
        speed = c['v'][:,left]
        compared = (t[:-5]>=left_ms-4)&(t[:-5]<=left_ms+12) if left_ms and start==520 else None
        window = (t>=left_ms)&(t<=left_ms+60) if left_ms and start==520 else None
        post = (t>=710)&(t<=960)&(np.isnan(c['done']) | (t<=c['done']))
        values = {f'{name}_{at}_deg': np.interp(at,t,angles[:,axis]).item() if t[0]<=at<=t[-1] else None
                  for at in (550,560,610,710,810,900) for axis,name in enumerate(('roll','pitch'))}
        rows.append(dict(batch=tag,start_ms=start,profile=profile,arm=arm,left_contact_ms=left_ms,right_contact_ms=contacts[1],
                         left_max_5ms_drop_rad_s=float((speed[:-5]-speed[5:])[compared].max()) if compared is not None else None,
                         pitch_rmse_first60_deg=float(np.sqrt(np.mean((angles[window,1]-np.interp(t[window],real_t,real_angles[:,1]))**2))) if window is not None else None,
                         pitch_peak_710_960_deg=float(angles[post,1].max()),pitch_peak_time_ms=int(t[post][angles[post,1].argmax()]),
                         leg_or_body_contact_ms=int(((t<=960)&(c['other_force']>1)).sum()),
                         fall_ms=None if np.isnan(c['done']) else int(c['done']),**values))
    return rows


def legs(model, q):
    d = mujoco.MjData(model)
    wheels = [model.body('robot/'+n).id for n in ('lfwheel','rfwheel')]
    hips = [model.joint('robot/'+n).id for n in ('lf0','rf0')]
    measured = np.empty((len(q),2,2))
    for i, pose in enumerate(q):
        d.qpos[:] = pose
        mujoco.mj_kinematics(model,d)
        local = (d.xpos[wheels]-d.xanchor[hips]) @ d.xmat[model.body('robot/base_link').id].reshape(3,3)
        measured[i,:,0]=np.linalg.norm(local[:,[0,2]],axis=1)
        measured[i,:,1]=np.rad2deg(np.arctan2(local[:,0],-local[:,2]))
    return measured


def report(out, refinement):
    alignment = json.loads(ALIGN.read_text())
    digest = sha(ARCHIVE)
    assert digest==alignment['archive_sha256']
    second = next(t for t in alignment['trials'] if t['trial']==2)
    video_window = 1000*(np.asarray(second['touchdown_bracket_frames'])/alignment['fps']-second['video_offset_s'])
    with zipfile.ZipFile(ARCHIVE) as archive:
        imu, pd, policy = [read_csv(archive,'试次2-'+name+'.csv') for name in ('H7反馈','H7控制','模型输入动作')]
    ct, st = pd['rel_s']*1000, imu['rel_s']*1000
    rotation = Rotation.from_quat(matrix(imu,[f'quat_{j}' for j in (1,2,3,0)]))
    angles = rotation.as_euler('xyz',degrees=True)
    rate = pitch_rate(angles,matrix(imu,[f'gyro_rad_s_{j}' for j in range(3)]))
    acc = rotation.apply(matrix(imu,[f'acc_g_{j}' for j in range(3)])*[1,-1,-1])[:,2]
    controls = np.flatnonzero((ct>=video_window[0]-5)&(ct<=video_window[1]))
    k = controls[np.argmin(np.diff(pd['velocity_4'])[controls])]
    impact = ct[k:k+2]
    assert (np.diff(impact)<5.1).all() and impact[0]>=video_window[0] and impact[1]<=video_window[1]
    pick = np.flatnonzero((ct>=impact[0]-5.1)&(ct<=impact[0]+65.1))
    interpolated = matrix(imu,['left_leg_0','right_leg_0','left_leg_1','right_leg_1'])

    cap, gpu_q, cases = load(out,digest)
    _, _, extra = load(refinement,digest)
    base = cases[(520,'imu','walk')]
    direct = cases[(520,'imu','recorded_torque')]
    for key in (('imu','walk'),('mu025','walk'),('imu','recorded_torque')):
        a, b = cases[(520,*key)],extra[(520,*key)]
        early = a['t']<=618
        np.testing.assert_allclose(a['angles'][early],b['angles'][early],rtol=0,atol=.001)
    model = mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
    left,right = [int(model.joint('robot/'+name).dofadr[0]) for name in ('lfwheel','rfwheel')]
    arrays = dict(real=(ct,np.column_stack((pd['velocity_4'],pd['velocity_5'],pd['pd_Nm_4'],pd['pd_Nm_5'])),
                        st,np.column_stack((angles[:,:2],rate,acc,interpolated[:,:2],np.rad2deg(interpolated[:,2:])))))
    for name,c in (('closed',base),('torque_replay',direct)):
        leg = legs(model,c['q'][:120])
        arrays[name] = (c['t'],np.column_stack((c['v'][:,[left,right]],c['torque'][:,[2,5]])),
                        c['t'][:120],np.column_stack((c['angles'][:120,:2],c['rate'][:120],np.full(120,np.nan),
                                                      leg[:,:,0],leg[:,:,1])))
    samples = []
    for u in ct[pick]:
        row = dict(Jump_ms=u,ms_from_last_preimpact_control=u-impact[0])
        for name,(t,raw,s,obs) in arrays.items():
            near = np.argmin(abs(s-u))
            row[f'{name}_imu_or_physics_ms']=s[near].item()
            for label,values in zip(('L_rad_s','R_rad_s','L_Nm','R_Nm'),raw.T):
                row[f'{name}_{label}']=np.interp(u,t,values)
            for label,value in zip(('roll_deg','pitch_deg','pitch_rate_rad_s','acc_z_g','L_m','R_m','L_swing_deg','R_swing_deg'),obs[near]):
                if np.isfinite(value):
                    row[f'{name}_{label}']=value
        samples.append(row)
    csv_rows(out/'first60.csv',samples)
    rows = case_rows(cases,st,angles,left,'contact')+case_rows(extra,st,angles,left,'refinement')
    csv_rows(out/'cases.csv',rows)
    peak_window = (st>=710)&(st<=960)
    policy_rows = [dict(Jump_ms=float(policy['rel_s'][j]*1000),action_2=float(policy['action_2'][j]))
                   for j in np.flatnonzero(abs(policy['rel_s']-.55)<.011)]
    write_json(out/'report.json',dict(archive_sha=digest,real_contact_video_ms=video_window.tolist(),
        real_left_impact_proxy_ms=impact.tolist(),real_left_drop_rad_s=float(pd['velocity_4'][k]-pd['velocity_4'][k+1]),
        real_right_delta_rad_s=float(pd['velocity_5'][k+1]-pd['velocity_5'][k]),
        real_left_request_Nm=pd['pd_Nm_4'][k:k+2].tolist(),first60=samples,
        real_left_policy_actions_540_560=policy_rows,
        real_pitch_by_ms_deg={str(at):float(np.interp(at,st,angles[:,1])) for at in (550,560,610,710,810,900)},
        real_pitch_peak_710_960_deg=float(angles[peak_window,1].max()),
        real_pitch_peak_time_ms=float(st[peak_window][angles[peak_window,1].argmax()]),
        zero_optimizer_updates=True,cases=rows))

    plt.rcParams.update({'font.family':['Noto Sans CJK SC'],'axes.unicode_minus':False})
    fig, axs = plt.subplots(4,2,figsize=(13,11),constrained_layout=True)
    panels = [(ct,pd['velocity_4'],'左轮速度（rad/s）'), (ct,pd['velocity_5'],'右轮速度（rad/s）'),
              (st,rate,'俯仰角速度（rad/s）'), (st,angles[:,0],'横滚 / °'),
              (st,angles[:,1],'俯仰 / °'), (st,angles[:,1],'俯仰 / °'),
              (st,np.rad2deg(interpolated[:,2]),'左腿摆角 / °'), (st,interpolated[:,0]*100,'左腿长度 / cm')]
    for i,(ax,(t,y,label)) in enumerate(zip(axs.flat,panels)):
        end = 960 if i==5 else 618
        window = (t>=534)&(t<=end)
        ax.plot(t[window],y[window],'k-' if i==5 else 'k.-',label='实机',lw=1.7,ms=4)
        ax.set_ylabel(label)
        ax.axvspan(*impact,color='#e6a556',alpha=.24)
        ax.set_xlim(534,end)
        ax.grid(alpha=.22)
    plotted = [(base,'名义闭环','#3478bb'),
               (direct,'名义接触 + 实机力矩请求','#8e6bad'),
               (cases[(520,'mu100_contact05_inertia_half','walk')],'较硬接触 + 较高摩擦 + 减轮惯量','#bb6944'),
               (cases[(520,'mu025','walk')],'低摩擦反事实','#20845c'),
               (extra[(520,'mu025_contact05_inertia_half','walk')],'低摩擦 + 较硬接触 + 减轮惯量','#509132')]
    for c,label,color in plotted:
        for i,(ax,y) in enumerate(zip(axs.flat,(c['v'][:,left],c['v'][:,right],c['rate'],c['angles'][:,0],c['angles'][:,1],c['angles'][:,1]))):
            if i==5 and c is direct:
                continue
            t = c['t']
            window = (t>=534)&(t<=(618 if i<5 else 960))
            ax.plot(t[window],y[window],label=label,color=color,ls='--' if c is direct else '-',lw=1.2)
        t = c['t'][:120]
        length, swing = legs(model,c['q'][:120])[:,0].T
        window = (t>=534)&(t<=618)
        for ax,y in zip(axs[3],(swing,length*100)):
            ax.plot(t[window],y[window],color=color,ls='--' if c is direct else '-',lw=1.2)
    for key,label,style in [(('imu','walk'),'610 ms实机状态起跑，名义摩擦',':'),(('mu025','walk'),'610 ms实机状态起跑，低摩擦','--')]:
        c = extra[(610,*key)]
        window = c['t']<=960
        axs[2,1].plot(c['t'][window],c['angles'][window,1],style,label=label,color='#168a92',lw=1.5)
    axs[2,1].axvline(710,color='#999999',ls=':')
    axs[0,0].legend(fontsize=8)
    axs[2,1].legend(fontsize=7,loc='upper left')
    for ax in axs[3]:
        ax.set_xlabel('切入 Jump 后 / ms，阴影为实机左轮减速区间')
    fig.savefig(out/'first60.png',dpi=150)
    plt.close(fig)
    video = ROOT/'artifacts/fudan_ppo_jump/20261125-ground-replay-01/source/real-video-1.mp4'
    assert sha(video)==alignment['video_sha256']
    indexes = {'base':base['index'],'mu025':cases[(520,'mu025','walk')]['index']}
    rendered_rows = [dict(walk_switch_ms=190) for _ in cap['cases']]
    replay(out,gpu_q[:45],model,indexes,rendered_rows,video,second['video_offset_s']+.52,
           cap['real_walk_s']-.52,{'base':'名义闭环','mu025':'低摩擦复现'},clock_start_ms=520)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('refinement',type=Path)
    args = p.parse_args()
    report(args.out,args.refinement)
