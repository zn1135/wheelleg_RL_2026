"""Compare500Hz motor feedback experiments; no physics or policy inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import detrend


def spectrum(values,dt):
    centered=detrend(values.astype(float),axis=0,type='linear')
    power=np.square(abs(np.fft.rfft(centered*np.hanning(len(centered))[:,None],axis=0))).sum(1)
    f=np.fft.rfftfreq(len(centered),dt);valid=f>=.4
    return dict(peak_hz=float(f[np.flatnonzero(valid)[np.argmax(power[valid])]]),
        fraction_above5hz=float(power[f>=5].sum()/power[valid].sum()),
        rms=float(np.sqrt(np.mean((values-values.mean(0))**2))))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recordings',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];noise_reference=None;checks=[]
    for folder in args.recordings:
        run=json.loads((folder/'probe-run.json').read_text())
        assert run['zero_delay_PD_max_error_nm']<=1e-5
        with np.load(folder/'pd-500hz.npz') as d:
            time=d['time'];mask=(time>=10)&(time<15);dt=float(np.median(np.diff(time)));ids=d['sample_ids'].tolist()
            assert abs(dt-.002)<1e-12 and mask.sum()==2500
            signal=d['values'][mask].copy()
        vf=folder/'velocity-feedback-500hz.npz';feedback=None
        if vf.exists():
            with np.load(vf) as d:feedback=d['values'][mask].copy()
        for i,sid in enumerate(ids):
            target=signal[:,i,:,0];actual=signal[:,i,:,1];torque=signal[:,i,:,2]
            velocity=feedback[:,i,:,0] if feedback is not None else np.gradient(actual,dt,axis=0)
            row=dict(checkpoint=run['checkpoint'],leg_p=run['leg_p'],leg_d=run['leg_d'],sample_id=sid,
                sample_hz=500,window_s=[10,15],actual_angle_peak_to_peak_deg=float(np.rad2deg(np.ptp(actual,axis=0)).max()),
                last_second_angle_peak_to_peak_deg=float(np.rad2deg(np.ptp(actual[-500:],axis=0)).max()),
                target_angle=spectrum(target,dt),actual_angle=spectrum(actual,dt),actual_velocity=spectrum(velocity,dt),requested_torque=spectrum(torque,dt),
                velocity_source='true500Hz qvel' if feedback is not None else '500Hz position finite difference')
            if feedback is not None and 'velnoise' in sid:
                error=float(np.abs(feedback[:,i,:,1]-feedback[:,i,:,0]-feedback[:,i,:,2]).max())
                assert error<1e-5
                row['feedback_identity_max_error_rad_s']=error
                row['injected_noise_std_rad_s']=float(feedback[:,i,:,2].std())
                if sid=='armature_x2_velnoise_0.2_h26':
                    if noise_reference is None:noise_reference=feedback[:,i,:,2].copy()
                    checks.append(dict(checkpoint=run['checkpoint'],shared_noise_exact=np.array_equal(noise_reference,feedback[:,i,:,2])))
            rows.append(row)
    assert all(c['shared_noise_exact'] for c in checks)
    result=dict(rows=rows,shared_noise_checks=checks,physics_steps=0,model_forwards=0,
        interpretation='5Hz descriptive band only; physical feedback hypotheses, not measured real-robot noise/delay. Same x2 armature across policies.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 500Hz速度反馈对照','', '| 模型 | 条件，0.26m | 腿速主频Hz | 腿速≥5Hz能量占比 | 请求力矩波动RMS Nm | 末1秒关节峰峰值° |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['sample_id'] not in ('armature_x2_h26','armature_x2_velnoise_0.2_h26','armature_x2_veldelay_6ms_h26'):continue
        lines.append(f'| {Path(r["checkpoint"]).stem} | {r["sample_id"]} | {r["actual_velocity"]["peak_hz"]:.2f} | {r["actual_velocity"]["fraction_above5hz"]:.4f} | {r["requested_torque"]["rms"]:.3f} | {r["last_second_angle_peak_to_peak_deg"]:.3f} |')
    lines+=['','没有将噪声或延迟写入训练/H7。当前只是共同参数候选的仿真检验，不能凭文字描述认定同时匹配三个实机现象。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
