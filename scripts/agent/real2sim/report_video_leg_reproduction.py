"""Compare saved GPU oscillations with manually checked image-plane tracks; no physics."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import detrend,find_peaks


def shape(time,angle):
    dt=float(np.median(np.diff(time)))
    peaks,_=find_peaks(angle,prominence=max(float(np.ptp(angle))*.15,.1),distance=max(1,round(.3/dt)))
    f=np.fft.rfftfreq(len(angle),dt)
    power=abs(np.fft.rfft(detrend(angle)*np.hanning(len(angle))))**2
    valid=(f>=.4)&(f<=5.)
    return dict(peak_to_peak_deg=float(np.ptp(angle)),peak_times_s=time[peaks].tolist(),
        median_peak_period_s=float(np.median(np.diff(time[peaks]))) if len(peaks)>1 else None,
        dominant_hz=float(f[valid][np.argmax(power[valid])]),frequency_resolution_hz=1/(len(angle)*dt))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    rows=[];video=[];plots={};checks=[];root=args.experiment
    for name in ('video-8-10','video-14-16'):
        folder=root/name;tr=np.loadtxt(folder/'tracking.csv',delimiter=',',skiprows=1)
        metadata=json.loads((folder/'measurement.json').read_text())
        video.append(dict(window_s=metadata['window_s'],features=shape(tr[:,0],tr[:,5]),
            minimum_template_score=metadata['minimum_template_score'],video_sha256=metadata['video_sha256']))
        plots[name]=(tr[:,0]-tr[0,0],tr[:,5]-tr[:,5].mean())
    for name in ('probe','response-probe','hold-probe'):
        folder=root/name;protocol=json.loads((folder/'protocol.json').read_text())
        execution=json.loads((folder/'execution.json').read_text());assert execution['planned']==execution['complete_traces']
        mapping=json.loads((folder/'fixtures/mapping.json').read_text());qi=np.asarray(mapping['motor_q'])[[0,1,3,4]]
        model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));data=mujoco.MjData(model)
        hip=model.joint('robot/rf0').id;wheel=model.body('robot/rfwheel').id
        references={};initial_differences={}
        for sample in protocol['samples']:
            sid=sample['id'];height=sample['case']['segments'][0]['command'][2];end=sample['case']['duration']
            with np.load(folder/'fixtures'/f'{sid}_initial.npz') as fixture:
                common={k:fixture[k].copy() for k in fixture.files if k!='dof_armature'}
            if height not in references:references[height]=common
            for key in common:
                error=float(np.max(abs(common[key]-references[height][key])))
                initial_differences[key]=max(initial_differences.get(key,0.),error)
                # 原复位中float32零位相减会有1ULP差异，保留实际偏差而不称逐位一致。
                assert error<=(1e-7 if key in ('qpos','nominal') else 0.)
            assert not np.any(common['episode_wrench_body'])
            with np.load(folder/'traces'/f'{sid}.npz') as tr:
                assert len(tr['qpos'])==round(end*100)+1 and np.isfinite(tr['qpos']).all()
                assert np.isfinite(tr['linear_world']).all() and np.isfinite(tr['roll_pitch_deg']).all()
                # 同时保留原15秒窗口和长时间零指令末段，防止把慢衰减当持续振荡。
                for stop in sorted(set((15.,end))):
                    mask=(tr['qpos_time']>=stop-5)&(tr['qpos_time']<stop);time=tr['qpos_time'][mask];q=tr['qpos'][mask]
                    angle=[]
                    for state in q:
                        data.qpos[:]=state;mujoco.mj_kinematics(model,data)
                        vector=data.xpos[wheel]-data.xanchor[hip]
                        w,x,y,z=state[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                        forward=vector[0]*np.cos(yaw)+vector[1]*np.sin(yaw)
                        angle.append(np.rad2deg(np.arctan2(forward,-vector[2])))
                    angle=np.asarray(angle);joints=np.rad2deg(q[:,qi]);mm=(tr['time']>=stop-5)&(tr['time']<stop)
                    features=shape(time,angle)
                    rows.append(dict(recording=name,sample_id=sid,candidate=sample['candidate'],height_m=height,
                        window_s=[stop-5,stop],side_proxy=features,
                        side_proxy_first2s_pp_deg=float(np.ptp(angle[:200])),side_proxy_last2s_pp_deg=float(np.ptp(angle[-200:])),
                        joint_max_pp_deg=float(np.ptp(joints,axis=0).max()),
                        last_second_joint_max_pp_deg=float(np.ptp(joints[-100:],axis=0).max()),
                        speed_peak_m_s=float(np.linalg.norm(tr['linear_world'][mm,:2],axis=1).max()),
                        pitch_abs_peak_deg=float(abs(tr['roll_pitch_deg'][mm,1]).max()),
                        trace_sha256=hashlib.sha256((folder/'traces'/f'{sid}.npz').read_bytes()).hexdigest()))
                    if name=='hold-probe' and stop==end:plots[sid]=(time-time[0],angle-angle.mean())
        checks.append(dict(recording=name,complete_finite_traces=execution['complete_traces'],
            common_initial_arrays_exact=all(v==0 for v in initial_differences.values()),common_initial_max_difference=initial_differences,
            no_external_wrench=True,wall_s=execution['wall_s']))
    # 500Hz实际提交力矩与100Hz所存末子步请求配对，验证一阶响应实际施加。
    folder=root/'response-probe';profiles=json.loads((folder/'protocol.json').read_text())['samples']
    with np.load(folder/'applied-torque-500hz.npz') as applied,np.load(folder/'control-traces.npz') as requested:
        values=applied['values'];previous=values[3::5];actual=values[4::5];desired=requested['values'][...,2]
        ms=np.array([s['candidate'].get('torque_time_constant_ms',0.) for s in profiles],dtype=np.float32)
        alpha=-np.expm1(-np.float32(.002)/np.maximum(ms*.001,1e-12))
        expected=previous+alpha[None,:,None]*(desired-previous);expected[:,ms==0]=desired[:,ms==0]
        error=float(abs(actual-expected).max());assert error<1e-4
        checks.append(dict(torque_lag_last_substep_max_error_nm=error,checked_substeps=len(actual),
            zero_lag_bypass_exact=bool(np.array_equal(actual[:,ms==0],desired[:,ms==0]))))
    result=dict(video=video,rows=rows,checks=checks,physics_steps=0,policy_updates=0,
        limitations='Video tracks are uncalibrated image-plane proxies with two short windows; simulation proxy removes yaw. Different projection/height and unknown pre-video command history prevent parameter identification or exact angular matching.')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,10))
    for name in ('video-8-10','video-14-16'):axes[0].plot(*plots[name],label=name)
    axes[0].set_title('Real video: image-plane angle, two zero-command windows')
    for ax,height in zip(axes[1:],(22,26,30)):
        for name in ('baseline','armature_x4','armature_x6'):ax.plot(*plots[f'{name}_h{height}'],label=name)
        ax.set_title(f'GPU: h={height/100:.2f}m, final5s of30s run, zero command since7s')
    for ax in axes:ax.set_ylabel('Angle - mean (deg)');ax.set_xlabel('Time within window (s)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'oscillation.png',dpi=150)
    print(json.dumps(dict(video=video,checks=checks),indent=2))


if __name__=='__main__':main()
