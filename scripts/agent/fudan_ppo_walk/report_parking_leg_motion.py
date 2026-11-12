"""Measure actual leg-joint oscillation from saved parking traces, without simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LEGS=(0,1,3,4)
NAMES=('lf0','lf00','rf0','rf00')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--recordings',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];plot_data=[]
    for root in args.recordings:
        run=json.loads((root/'run.json').read_text())
        label=Path(run['checkpoint']).stem+f' P{run["leg_p"]:g}/D{run["leg_d"]:g}'
        for group in ('no_external','configured_external'):
            folder=root/group;protocol=json.loads((folder/'protocol.json').read_text())
            mapping=json.loads((folder/'fixtures/mapping.json').read_text())
            indices=np.asarray(mapping['motor_q'])[list(LEGS)]
            for sample in protocol['samples']:
                path=folder/'traces'/(sample['id']+'.npz')
                with np.load(path) as d:
                    end=sample['case']['duration'];t=d['qpos_time'];mask=(t>=end-5)&(t<end)
                    # 实际电机角度，单位度；最大减最小为一次往返覆盖的角度范围。
                    q=np.rad2deg(d['qpos'][mask][:,indices]);ts=t[mask]
                    if len(ts)!=500:raise ValueError('Incomplete five-second joint window')
                    peak_to_peak=np.ptp(q,axis=0);j=int(np.argmax(peak_to_peak))
                    # 先去除线性漂移；频谱仅提示主要重复频率，不作为振荡因果证明。
                    trend=np.polyval(np.polyfit(ts,q[:,j],1),ts)
                    spectrum=np.abs(np.fft.rfft((q[:,j]-trend)*np.hanning(len(q))))**2
                    frequencies=np.fft.rfftfreq(len(q),.01);valid=(frequencies>=.4)&(frequencies<=20)
                    dominant=float(frequencies[valid][np.argmax(spectrum[valid])])
                    last=q[ts>=end-1]
                    row=dict(model=label,group=group,sample_id=sample['id'],case=sample['case']['id'],
                        window_s=[end-5,end],joint_names=NAMES,joint_peak_to_peak_deg=peak_to_peak.tolist(),
                        max_peak_to_peak_deg=float(peak_to_peak[j]),largest_joint=NAMES[j],
                        last_second_peak_to_peak_deg=np.ptp(last,axis=0).tolist(),
                        last_second_max_peak_to_peak_deg=float(np.ptp(last,axis=0).max()),
                        dominant_frequency_hz=dominant,frequency_resolution_hz=.2,
                        per_second_max_peak_to_peak_deg=[float(np.ptp(q[(ts>=end-5+k)&(ts<end-4+k)],axis=0).max()) for k in range(5)],
                        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    rows.append(row)
                    if group=='no_external' and sample['id']=='parking_5':plot_data.append((label,ts,q))
    (args.output/'leg-motion.json').write_text(json.dumps(dict(rows=rows,physics_steps=0,model_forwards=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual joint motion only; commanded targets and torque not recorded'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(plot_data),1,figsize=(11,2.8*len(plot_data)),sharex=True,sharey=True,squeeze=False)
    for ax,(label,t,q) in zip(axes[:,0],plot_data):
        for j,name in enumerate(NAMES):ax.plot(t,q[:,j]-q[:,j].mean(),label=name,lw=1)
        ax.set_title(label+' / no external / forward stop h0.26');ax.set_ylabel('Joint angle - mean (deg)');ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
    axes[-1,0].set_xlabel('Time (s); zero command since 7s')
    fig.tight_layout();fig.savefig(args.output/'leg-oscillation.png',dpi=150)


if __name__=='__main__':main()
