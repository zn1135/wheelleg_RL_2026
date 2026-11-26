"""只读追踪原6838高腿长录制的起振过程；不运行物理、训练或硬件控制。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import detrend, periodogram, find_peaks

from . import analyze_high_length_6838 as prior

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'artifacts/real2sim/20261117-high-length6838-01'
RUN = ROOT / 'logs/mjlab/fudan_ppo_walk/20261116_scan_slope5to20_mu08_resume5838_to6838_01'
MODEL = ROOT / 'artifacts/fudan_ppo_walk/models/20261116_model6838_regions_scan_slope5to20_mu08/model_final.pt'
MODEL_SHA = '8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
PHYSICAL = list(dict.fromkeys([*prior.PHYSICAL, 'acc_x_g', 'acc_y_g', 'acc_z_g',
    'left_length_rate_m_s', 'right_length_rate_m_s',
    *[f'feedback_nm_{m}' for m in prior.LEGS], *[f'angle_rad_{m}' for m in prior.LEGS]]))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def trailing_rms(value, count=40):
    # 只使用当前及过去0.2秒，不用双向滤波把起振后的运动泄漏到起振前。
    value = np.asarray(value, float)
    c = np.r_[0., np.cumsum(value)]
    c2 = np.r_[0., np.cumsum(value*value)]
    n = np.minimum(np.arange(1, len(value)+1), count)
    first = np.arange(len(value))+1-n
    mean = (c[1:]-c[first])/n
    return np.sqrt(np.maximum((c2[1:]-c2[first])/n-mean*mean, 0.))


def first_time(t, mask):
    ids = np.flatnonzero(mask)
    return None if not len(ids) else float(t[ids[0]])


def growth_time(t, rms, start, threshold):
    hot = (t >= start) & (rms > threshold)
    begins=np.flatnonzero(np.r_[hot[0],hot[1:] & ~hot[:-1]])
    ends=np.flatnonzero(np.r_[hot[:-1] & ~hot[1:],hot[-1]])+1
    for i,j in zip(begins,ends):
        if j-i >= 20 and rms[i:j].max() >= .45:
            return float(t[i])
    return None


def spectral(value, dt=.005):
    f, p = periodogram(detrend(value, axis=0), fs=1/dt, axis=0)
    take = (f >= 8) & (f <= 14)
    return dict(peak_Hz=float(f[take][np.argmax(p[take])]),
        band_rms=float(np.sqrt(np.trapezoid(p[take], f[take]))))


def load_sessions(out):
    index = json.loads((SOURCE/'input-index.json').read_text())
    assert prior.sha(Path(index['source'])) == index['sha256']
    hashes = {r['name']: r['sha256'] for r in index['files']}
    folder = SOURCE/'raw/高腿长抖腿/6838高腿长完整录制'
    sessions, audits = [], []
    for path in sorted(folder.glob('complete-cycle*.csv')):
        relative = str(path.relative_to(SOURCE/'raw'))
        assert prior.sha(path) == hashes[relative]
        pd_rows, pd_sources, ph_rows, sources = [], [], [], set()
        with path.open() as f:
            for row in csv.DictReader(f):
                sources.add(row['source'])
                if row['frame'] == 'pd':
                    pd_rows.append([float(row[k]) for k in prior.NAMES])
                    pd_sources.append(row['source'])
                elif row['frame'] == 'physical':
                    ph_rows.append([float(row[k]) for k in PHYSICAL])
        # 一个完整周期可能跨录制BIN轮转，逐行按source+板端时间核对，不能假设单文件。
        lookups, decoding = {}, {}
        for source in sorted(sources):
            raw = folder/source
            assert prior.sha(raw) == hashes[str(raw.relative_to(SOURCE/'raw'))]
            decoded, _, audit = prior.decode(raw)
            assert audit['crc_or_header_errors'] == 0
            lookups[source] = {int(row[0]): row for row in decoded}
            decoding[source] = dict(sha256=prior.sha(raw), **audit)
        pd, ph = np.asarray(pd_rows), np.asarray(ph_rows)
        assert np.array_equal(pd, np.asarray([lookups[source][int(row[0])] for source,row in zip(pd_sources,pd)]))
        arrival_order = dict(pd_backward_steps=int(np.sum(np.diff(pd[:,0])<=0)),
                             physical_backward_steps=int(np.sum(np.diff(ph[:,0])<=0)))
        d = prior.columns(pd); active = prior.valid(d)
        pd = pd[active]
        # CSV跨BIN按到达顺序交错；原始值核验后，按同一板端时钟重排，不当作控制时钟回跳。
        pd = pd[np.argsort(pd[:,0],kind='stable')]
        ph = ph[np.argsort(ph[:,0],kind='stable')]
        d = prior.columns(pd)
        assert len(np.unique(d['reset_count'])) == 1
        assert np.all(np.diff(pd[:, 0]) > 0) and np.all(np.diff(ph[:, 0]) > 0)
        # 物理遥测比同拍PD早几微秒，不能用独立窗口切片后按行号拼接。
        right = np.searchsorted(ph[:, 0], pd[:, 0]).clip(0, len(ph)-1)
        left = (right-1).clip(0, len(ph)-1)
        match = np.where(abs(ph[left, 0]-pd[:, 0]) <= abs(ph[right, 0]-pd[:, 0]), left, right)
        error = ph[match, 0]-pd[:, 0]
        assert abs(error).max() <= 50
        aligned = ph[match]
        assert np.isfinite(pd).all() and np.isfinite(aligned).all()
        assert np.all(np.diff(d['sequence']) == 1)
        first = np.r_[True, np.diff(d['obs_sequence']) != 0]
        assert np.all(np.diff(d['obs_sequence'][first]) == 1)
        identifier = path.stem
        np.savez_compressed(out/(identifier+'.npz'), pd=pd, physical=aligned, physical_h7_us=ph[:, 0],
            physical_original=ph, physical_names=np.array(PHYSICAL), pd_names=np.array(prior.NAMES))
        sessions.append(dict(id=identifier, pd=pd, physical=aligned, ph_raw=ph))
        audits.append(dict(id=identifier, csv_sha256=prior.sha(path),
            rows=len(pd), reset=int(d['reset_count'][0]), raw_decode=decoding, pd_matches_raw=True,
            original_csv_arrival_order=arrival_order, analysis_order='strictly increasing H7 timestamp; no duplicate timestamps',
            physical_pairing_error_us=np.quantile(error, [0,.5,1]).tolist(),
            unique_physical_pairing=len(np.unique(match)) == len(match)))
    assert len(sessions) == 5
    write(out/'input-audit.json', dict(zip_sha256=index['sha256'], sessions=audits,
        physical_source='Original physical CSV, verified against original ZIP per-file hashes; PD independently redecoded from CRC-checked BIN',
        physics_steps=0, hardware_operations=0))
    return sessions


def summarize(session):
    d = prior.columns(session['pd']); ph = prior.columns(session['physical'], PHYSICAL)
    t = d['h7_us']/1e6; height = d['obs_8']/5
    gx = ph['gyro_x_rad_s']; env = trailing_rms(gx)
    first_raise = first_time(t, height > .225)
    assert first_raise is not None
    onset = growth_time(t, env, first_raise, .15)
    assert onset is not None, session['id']
    n = int(np.argmin(abs(t-onset)))
    peak_window = (t >= onset+.2) & (t <= min(onset+1.2, t[-1]))
    before = (t >= onset-.5) & (t <= onset+1.)
    obs_first = np.r_[True, np.diff(d['obs_sequence']) != 0]
    after = (t >= onset-.5)
    sat = np.stack([abs(d[f'pd_model_nm_{m}']) >= (3.9-1e-3 if m in ('lw','rw') else 40.-1e-3) for m in prior.MOTORS], 1)
    lengths = np.c_[ph['left_length_m'], ph['right_length_m']]
    acc = np.sqrt(sum(ph['acc_'+k+'_g']**2 for k in ('x','y','z')))
    rising=np.flatnonzero(np.diff(height)>1e-5)+1
    ramps=np.split(rising,np.flatnonzero(np.diff(t[rising])>.03)+1)
    recent=next(run for run in reversed(ramps) if t[run[-1]]<=onset)
    rows = []
    for start in np.arange(t[0], t[-1]-.20, .10):
        take = (t >= start) & (t < start+.20)
        rows.append(dict(time_end_s=float(start+.20), height_mean=float(height[take].mean()),
            length_mean_m=lengths[take].mean(0).tolist(), gyroX_rms=float(np.std(gx[take])),
            gyroX_peak=float(abs(gx[take]).max()), accel_norm_peak_g=float(acc[take].max())))
    return dict(id=session['id'], reset=int(d['reset_count'][0]), time_h7_s=[float(t[0]),float(t[-1])],
        first_height_rise_s=first_raise, first_height_030_s=first_time(t,height>=.2999),
        detected_growth_s=onset, detection_thresholds_s={str(v):growth_time(t,env,first_raise,v) for v in (.10,.15,.30)},
        latest_rising_height_ramp_s=[float(t[recent[0]]),float(t[recent[-1]])],
        growth_after_latest_height_ramp_ms=float((onset-t[recent[-1]])*1000),
        at_detection=dict(height=float(height[n]), left_right_length_m=lengths[n].tolist(),
            left_right_length_rate_m_s=[float(ph[k][n]) for k in ('left_length_rate_m_s','right_length_rate_m_s')],
            gyroX_trailing_rms=float(env[n])), spectrum_after_growth=spectral(gx[peak_window]),
        first_motor_saturation_near_growth_s={m:first_time(t, after & sat[:,i]) for i,m in enumerate(prior.MOTORS)},
        onset_window=dict(range_s=[onset-.5,onset+1.], flags=np.unique(d['flags'][before]).astype(int).tolist(),
            faults=np.unique(d['fault'][before]).astype(int).tolist(),
            resets=np.unique(d['reset_count'][before]).astype(int).tolist(),
            board_drop_increment=float(np.ptp(d['board_drops'][before])),
            pd_dt_us=np.quantile(np.diff(d['h7_us'][before]),[0,.5,1]).tolist(),
            obs_dt_us=np.quantile(np.diff(d['h7_us'][before & obs_first]),[0,.5,1]).tolist(),
            accel_norm_range_g=[float(acc[before].min()),float(acc[before].max())]),
        vx_yaw_command_abs_max=np.max(abs(np.c_[d['obs_6']/2,d['obs_7']/.25]),0).tolist(),
        causal_window_series=rows)


def focus_events(session, summary):
    d = prior.columns(session['pd']); ph = prior.columns(session['physical'], PHYSICAL)
    t = d['h7_us']/1e6; h=d['obs_8']/5
    pre = (t>=376.593242)&(t<379.8)
    take=(t>=379.8)&(t<381.2)
    falling = first_time(t,take&(h<.2999))
    minimum=float(h[take].min()); low_ids=np.flatnonzero(take&(h<=minimum+1e-5))
    low_start, low_end=float(t[low_ids[0]]),float(t[low_ids[-1]])
    rise=first_time(t,(t>low_end)&(h>minimum+.0005))
    high=first_time(t,(t>low_end)&(h>=.2999))
    length=np.c_[ph['left_length_m'],ph['right_length_m']]
    mean_length=length.mean(-1)
    moving=(t>=low_start)&(t<381.2)
    j=np.flatnonzero(moving)[np.argmin(mean_length[moving])]
    growth=summary['detected_growth_s']
    stable_h=(t>=summary['first_height_030_s'])&(t<falling)
    q_steps=np.max(abs(np.diff(np.stack([d['target_'+m] for m in prior.LEGS],1),axis=0)),axis=1)
    pd_steps_t=t[1:]
    early=(t>=falling)&(t<growth)
    late=(t>=growth)&(t<growth+.5)
    peaks,_=find_peaks(ph['gyro_x_rad_s'],prominence=.05,distance=6)
    peaks=peaks[(t[peaks]>=380.4)&(t[peaks]<=381.35)]
    return dict(height_fall_s=falling, minimum_height=minimum, low_command_window_s=[low_start,low_end],
        height_rise_s=rise,height_return_030_s=high,
        actual_mean_leg_min=dict(time_s=float(t[j]),length_m=float(mean_length[j]),left_right_m=length[j].tolist()),
        growth_s=growth,growth_minus_rise_ms=1000*(growth-rise),growth_minus_high_command_ms=1000*(growth-high),
        first_high_quiet_window_s=[summary['first_height_030_s'],falling],
        first_high_gyroX_rms=float(np.std(ph['gyro_x_rad_s'][stable_h])),
        pre_dip_gyroX_rms=float(np.std(ph['gyro_x_rad_s'][pre])),
        pre_dip_target_step_p99_rad=float(np.quantile(q_steps[(pd_steps_t>=376.6)&(pd_steps_t<379.8)],.99)),
        before_growth_target_step_max_rad=float(q_steps[(pd_steps_t>=falling)&(pd_steps_t<growth)].max()),
        before_growth_mean_leg_range_m=[float(mean_length[early].min()),float(mean_length[early].max())],
        before_growth_accel_norm_peak_g=float(np.sqrt(sum(ph['acc_'+k+'_g'][early]**2 for k in ('x','y','z'))).max()),
        next_half_second_gyroX_rms=float(np.std(ph['gyro_x_rad_s'][late])),
        raw_gyroX_positive_peaks=[dict(time_s=float(t[i]),value=float(ph['gyro_x_rad_s'][i]),
            previous_peak_interval_ms=None if k==0 else float((t[i]-t[peaks[k-1]])*1000)) for k,i in enumerate(peaks)],
        notice='Growth detection uses a trailing200ms window and100ms persistence; times are detection bounds, not unique physical first-cause timestamps.')


def policy_check(out, sessions):
    import torch
    from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen
    frozen.RUN=RUN
    for name,digest in json.loads((RUN/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert prior.sha(RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest
    sys.meta_path.insert(0,frozen.FrozenTrainingSource())
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    assert prior.sha(MODEL)==MODEL_SHA
    torch.set_num_threads(1)
    policy,metadata=load_policy(MODEL,'cuda:0')
    policy.eval().requires_grad_(False)
    original={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    receipt=[];local=[]
    with torch.inference_mode():
        for session in sessions:
            d,start,ids,obs,current,history=prior.policy_inputs(session['pd'])
            a=torch.tensor(current,device='cuda:0');b=torch.tensor(history,device='cuda:0')
            prediction,_=policy.act_inference(a,b)
            target=np.stack([d['target_'+m][start][ids] for m in prior.ORDER],1)
            predicted=prediction.cpu().numpy().copy()
            predicted[:,[0,1,3,4]]=metadata['control']['nominal_leg_positions']+.5*predicted[:,[0,1,3,4]]
            predicted[:,[2,5]]*=10
            error=float(abs(predicted-target).max());assert error<1e-4
            receipt.append(dict(id=session['id'],rows=len(a),target_max_error_rad_or_rad_s=error,
                leg_target_error_rad=float(abs(predicted[:,[0,1,3,4]]-target[:,[0,1,3,4]]).max()),
                wheel_target_error_rad_s=float(abs(predicted[:,[2,5]]-target[:,[2,5]]).max())))
            if int(d['reset_count'][0])!=10:
                continue
            times=d['h7_us'][start][ids]/1e6
            # 独立量出同一网络在起振前后的局部输入敏感度，不把它当闭环控制效果。
            delta=.01
            high=a.clone();low=a.clone();high[:,0]+=delta*.25;low[:,0]-=delta*.25
            yp,_=policy.act_inference(high,b);ym,_=policy.act_inference(low,b)
            jac=((yp-ym)*.5/(2*delta))[:,[0,1,3,4]].cpu().numpy()
            held=a.clone();hb=b.clone().reshape(-1,5,25);held[:,8]=1.5;hb[:,:,8]=1.5
            held_result,_=policy.act_inference(held,hb.reshape(-1,125))
            held_target=held_result.cpu().numpy()[:,[0,1,3,4]]*.5+metadata['control']['nominal_leg_positions']
            np.savez_compressed(out/'focus-policy-check.npz',time=times,target=predicted,
                height_held_030_leg_target=held_target,current_gyroX_target_derivative=jac)
            for label,lo,hi in [('stable',376.6,379.8),('lowering',380.19,380.60),
                                ('extending',380.60,381.10),('early_growth',381.10,381.60),('strong',382.,383.)]:
                mask=(times>=lo)&(times<hi)
                local.append(dict(window=label,time_h7_s=[lo,hi],frames=int(mask.sum()),
                    gyroX_derivative_rms_per_leg=np.sqrt(np.mean(jac[mask]**2,axis=0)).tolist(),
                    height_held_target_difference_rms_per_leg=np.sqrt(np.mean((held_target[mask]-predicted[mask][:,[0,1,3,4]])**2,axis=0)).tolist()))
    assert all(torch.equal(v.cpu(),original[k]) for k,v in policy.state_dict().items())
    write(out/'policy-audit.json',dict(passed=True,checkpoint_sha256=MODEL_SHA,rows=receipt,
        local_sensitivity=local,optimizer_updates=0,physics_steps=0,source=frozen.LOADED,
        interpretation='Recorded-state open-loop recomputation only; holding command input does not remove the already-recorded real motion.'))


def plot(out, sessions, summaries, events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    focus=next(s for s in sessions if s['id']=='complete-cycle-162432')
    d=prior.columns(focus['pd']);ph=prior.columns(focus['physical'],PHYSICAL);t=d['h7_us']/1e6
    origin=events['height_fall_s'];x=t-origin
    fig,axes=plt.subplots(6,1,figsize=(13,12),sharex=True)
    axes[0].plot(x,d['obs_8']/5,label='RL height command');axes[0].plot(x,ph['left_length_m'],label='Measured left length')
    axes[0].plot(x,ph['right_length_m'],label='Measured right length');axes[0].set_ylabel('Command / length (m)')
    axes[1].plot(x,ph['left_length_rate_m_s'],label='Left');axes[1].plot(x,ph['right_length_rate_m_s'],label='Right');axes[1].set_ylabel('Leg length rate (m/s)')
    axes[2].plot(x,ph['gyro_x_rad_s'],lw=.7,label='Physical gyroX');axes[2].plot(x,trailing_rms(ph['gyro_x_rad_s']),lw=1.6,label='Causal past200ms RMS');axes[2].axhline(.15,color='grey',ls=':');axes[2].set_ylabel('gyroX (rad/s)')
    axes[3].plot(x,d['target_lr'],label='LR target');axes[3].plot(x,d['current_lr'],label='LR measured');axes[3].set_ylabel('LR angle (rad)')
    axes[4].plot(x,d['pd_model_nm_lr'],lw=.7,label='LR PD request');axes[4].plot(x,ph['feedback_nm_lr'],lw=.7,label='LR feedback');axes[4].set_ylabel('Torque (Nm)')
    axes[5].plot(x,np.maximum(abs(d['pd_model_nm_lw']),abs(d['pd_model_nm_rw'])),label='Wheel torque max');axes[5].axhline(3.9,color='grey',ls=':');axes[5].set_ylabel('Wheel request (Nm)')
    for ax in axes:
        for name,color in [('height_fall_s','#2196f3'),('height_rise_s','#ff9800'),('height_return_030_s','#4caf50'),('growth_s','#e91e63')]:
            ax.axvline(events[name]-origin,color=color,ls='--',alpha=.7)
        ax.legend(fontsize=8,ncol=3,loc='upper left');ax.grid(alpha=.2);ax.set_xlim(-.5,2.2)
    axes[-1].set_xlabel(f'Seconds from RL height command falling (H7 {origin:.6f}s)')
    fig.suptitle('Original6838 real log | raw signals / causal trailing window | dashed: height down / up / .30 / growth\nDetection order does not uniquely identify physical cause; no acausal filtering or new hardware motion')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'focus-onset.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(12,12))
    for ax,s,row in zip(axes,sessions,summaries):
        d=prior.columns(s['pd']);ph=prior.columns(s['physical'],PHYSICAL);tt=d['h7_us']/1e6
        ax.plot(tt-tt[0],ph['gyro_x_rad_s'],lw=.6,label='Physical gyroX')
        ax.plot(tt-tt[0],trailing_rms(ph['gyro_x_rad_s']),lw=1.5,label='Causal RMS')
        other=ax.twinx();other.plot(tt-tt[0],d['obs_8']/5,color='black',alpha=.6,label='RL height');other.set_ylim(.20,.32);other.set_ylabel('Height cmd (m)')
        ax.axvline(row['detected_growth_s']-tt[0],color='#e91e63',ls='--');ax.set_title(s['id']);ax.set_ylabel('gyroX rad/s');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Seconds from valid RL start in each session')
    fig.suptitle('All five recorded RL sessions | no duplicated overlapping CSV counting')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'all-session-onsets.png',dpi=140);plt.close(fig)


def review(out):
    report=json.loads((out/'onsets.json').read_text())
    checks=json.loads((out/'input-audit.json').read_text())
    model=json.loads((out/'policy-audit.json').read_text())
    for name,digest in json.loads((out/'evidence-sha256.json').read_text()).items():
        assert prior.sha(out/name)==digest,name
    rows=[]
    for summary in report['sessions']:
        with np.load(out/(summary['id']+'.npz')) as z:
            d=prior.columns(z['pd']);ph=prior.columns(z['physical'],PHYSICAL)
        t=d['h7_us']/1e6;rise,end=summary['latest_rising_height_ramp_s'];growth=summary['detected_growth_s']
        before=(t>=rise)&(t<growth)
        acc=np.sqrt(sum(ph['acc_'+k+'_g']**2 for k in ('x','y','z')))
        mean=(ph['left_length_m']+ph['right_length_m'])/2
        j=np.flatnonzero(before)[np.argmax(acc[before])]
        k=np.flatnonzero(before)[np.argmax(mean[before])]
        sat=[v for v in summary['first_motor_saturation_near_growth_s'].values() if v is not None]
        rows.append(dict(id=summary['id'],height_ramp_s=[rise,end],detected_growth_s=growth,
            growth_delay_from_command_end_ms=(growth-end)*1000,
            maximum_mean_leg_before_detected_growth=dict(time_s=float(t[k]),length_m=float(mean[k])),
            accel_norm_peak_before_detected_growth=dict(time_s=float(t[j]),value_g=float(acc[j])),
            first_saturation_minus_detected_growth_s=min(sat)-growth if sat else None,
            note='Maxima are before the detection threshold, not necessarily before every small precursor oscillation.'))
    focus=report['focus']
    with np.load(out/'complete-cycle-162432.npz') as z:
        d=prior.columns(z['pd']);ph=prior.columns(z['physical'],PHYSICAL)
    t=d['h7_us']/1e6
    pks=focus['raw_gyroX_positive_peaks']
    after=next(r for r in pks if r['time_s']>focus['height_return_030_s'])
    before=pks[pks.index(after)-1]
    i=int(np.argmin(abs(t-after['time_s'])))
    precursor=dict(previous_peak=before,first_positive_peak_after_return=after,
        amplitude_ratio=after['value']/before['value'],
        mean_geometric_leg_length_at_post_return_peak_m=float((ph['left_length_m'][i]+ph['right_length_m'][i])/2),
        left_right_length_rate_at_post_return_peak_m_s=[float(ph[k][i]) for k in ('left_length_rate_m_s','right_length_rate_m_s')],
        interpretation='Small approximately10-11Hz cycles exist while lowering/low hold. The first larger peak occurs during extension; not proof of a hard end-stop or an independently identified resonance.')
    stable=next(r for r in model['local_sensitivity'] if r['window']=='stable')
    early=next(r for r in model['local_sensitivity'] if r['window']=='early_growth')
    ratios=(np.array(early['gyroX_derivative_rms_per_leg'])/stable['gyroX_derivative_rms_per_leg']).tolist()
    write(out/'onset-review.json',dict(rows=rows,focus_precursor=precursor,
        current_gyroX_local_gain_early_growth_div_stable=ratios,
        PD_raw_and_timestamp_audit_passed=all(r['pd_matches_raw'] and r['unique_physical_pairing'] for r in checks['sessions']),
        total_policy_replayed=sum(r['rows'] for r in model['rows']),
        max_leg_target_error_rad=max(r['leg_target_error_rad'] for r in model['rows']),
        physics_steps=0,optimizer_updates=0,hardware_operations=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin=focus['height_fall_s'];x=t-origin
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    axes[0].plot(x,d['obs_8']/5,label='RL height command')
    axes[0].plot(x,(ph['left_length_m']+ph['right_length_m'])/2,label='Mean measured geometric leg length')
    axes[0].set_ylabel('Command / leg length (m)')
    axes[1].plot(x,ph['gyro_x_rad_s'],lw=1,label='Physical gyroX (raw)')
    for r in pks:
        axes[1].scatter(r['time_s']-origin,r['value'],s=18,color='#e91e63')
    axes[1].set_ylabel('gyroX (rad/s)');axes[1].set_ylim(-.6,.6)
    axes[2].plot(x,d['target_lr'],label='LR requested target')
    axes[2].plot(x,d['current_lr'],label='LR measured angle');axes[2].set_ylabel('LR angle (rad)')
    for ax in axes:
        ax.axvline(focus['height_rise_s']-origin,color='#ff9800',ls='--',label='Height rising')
        ax.axvline(focus['height_return_030_s']-origin,color='#4caf50',ls='--',label='Height reaches .30')
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);ax.set_xlim(.15,1.15)
    axes[-1].set_xlabel('Seconds from first decreasing RL height command')
    fig.suptitle('Small cycles precede re-extension; larger cycles grow while the legs are still extending\nOriginal recorded samples only; observation does not identify one unique hardware root cause')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'precursor-zoom.png',dpi=140);plt.close(fig)
    low_height_control(out)
    shutil.copy2(__file__,out/'review-source.py')
    write(out/'review-evidence-sha256.json',{str(f.relative_to(out)):prior.sha(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(dict(rows=rows,focus_precursor=precursor,gyro_gain_ratios=ratios),ensure_ascii=False),flush=True)


def low_height_control(out):
    summaries=json.loads((out/'onsets.json').read_text())['sessions']
    rows=[]
    for s in summaries:
        with np.load(out/(s['id']+'.npz')) as z:
            d=prior.columns(z['pd']);ph=prior.columns(z['physical'],PHYSICAL)
        t=d['h7_us']/1e6;h=d['obs_8']/5;g=ph['gyro_x_rad_s'];rms=trailing_rms(g)
        low=(h<.225)&(t<s['first_height_rise_s']-.05)
        candidates=np.flatnonzero(low&(t>=t[0]+.2))
        i=candidates[np.argmax(rms[candidates])]
        late=low&(t>=s['first_height_rise_s']-.25)
        f,p=periodogram(g[low],fs=200.);band=(f>=5)&(f<=20)
        rows.append(dict(id=s['id'],low_height_command_range_m=[float(h[low].min()),float(h[low].max())],
            low_window_h7_s=[float(t[low][0]),float(t[low][-1])],
            maximum_trailing200ms_gyroX_rms_rad_s=float(rms[i]),maximum_time_h7_s=float(t[i]),
            final200ms_gyroX_rms_rad_s=float(np.std(g[late])),final_window_rows=int(late.sum()),
            low_5to20_spectrum_peak_Hz=float(f[band][np.argmax(p[band])]),
            note='A decaying startup transient at low height is present; high-height motion is not the first occurrence of all10-11Hz energy.'))
        if s['id']=='complete-cycle-162501':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            x=t-t[0];fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
            axes[0].plot(x,h,label='RL height command');axes[0].plot(x,(ph['left_length_m']+ph['right_length_m'])/2,label='Mean measured geometric leg length')
            axes[0].set_ylabel('Command / leg length (m)')
            axes[1].plot(x,g,label='Physical gyroX, raw',lw=.8);axes[1].set_ylabel('gyroX (rad/s)')
            axes[2].plot(x,rms,label='Causal past200ms RMS');axes[2].set_ylabel('gyroX RMS (rad/s)')
            for ax in axes:
                ax.axvline(s['first_height_rise_s']-t[0],ls='--',color='black',label='Height starts rising')
                ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
            axes[-1].set_xlabel('Seconds from valid RL start | complete-cycle-162501')
            fig.suptitle('Same real6838 session: low-height transient decays; high-height transient grows\nRecorded closed-loop evidence, not an independent identification of a mechanical resonance')
            fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'low-vs-high.png',dpi=140);plt.close(fig)
    write(out/'low-height-control.json',dict(sessions=rows,physics_steps=0,optimizer_updates=0,
        interpretation='Similar-frequency startup motion can decay at low height. This supports a posture/state-dependent closed-loop stability problem, without identifying gas spring, motor, IMU or contact as the unique cause.'))
    print(json.dumps(rows,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review',action='store_true',help='只读已有采集/重算，生成起振复核与放大图')
    args=p.parse_args()
    if args.review:
        review(args.output)
        return
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(source=str(SOURCE),analysis='Five sessions, original raw paired frames, causal onset and frozen policy local sensitivity',
        onset=dict(trailing_window_ms=200,persistence_ms=100,threshold_rad_s=.15,
            sensitivity_thresholds=[.10,.15,.30],future_confirmation='Same above-threshold excursion subsequently reaches.45; no artificial750ms confirmation deadline; the RMS itself uses no future samples'),
        nearest_physical_frame_max_error_us=50,physics_steps=0,optimizer_updates=0,hardware_operations=0,
        limitations=['No independent contact/load/IMU truth', 'Timing order and closed-loop correlations do not prove a unique physical cause']))
    shutil.copy2(__file__,out/'analysis-source.py')
    sessions=load_sessions(out)
    summaries=[summarize(s) for s in sessions]
    k=next(i for i,s in enumerate(sessions) if s['id']=='complete-cycle-162432')
    events=focus_events(sessions[k],summaries[k])
    write(out/'onsets.json',dict(sessions=summaries,focus=events,physics_steps=0,optimizer_updates=0))
    policy_check(out,sessions)
    plot(out,sessions,summaries,events)
    write(out/'evidence-sha256.json',{str(f.relative_to(out)):prior.sha(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(dict(sessions=[{k:r[k] for k in ('id','reset','first_height_rise_s','first_height_030_s','detected_growth_s','at_detection','spectrum_after_growth','first_motor_saturation_near_growth_s')} for r in summaries],focus=events),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
