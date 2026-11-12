"""Offline target/actual/torque spectrum from the lastPDsubstep100Hz records."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import detrend


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recordings',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];plots=[]
    for folder in args.recordings:
        run=json.loads((folder/'probe-run.json').read_text())
        with np.load(folder/'control-traces.npz') as d:
            t=d['time'];dt=float(np.median(np.diff(t)));mask=(t>=10)&(t<15);freq=np.fft.rfftfreq(int(mask.sum()),dt)
            for i,sid in enumerate(d['sample_ids'].tolist()):
                if not sid.startswith(('baseline_','armature_x2_')):continue
                channels={};spectrum=[]
                for col,name in enumerate(('target_angle','actual_angle','requested_torque')):
                    values=d['values'][mask,i,:,col].astype(float)
                    values=detrend(values,axis=0,type='linear');power=np.square(abs(np.fft.rfft(values*np.hanning(len(values))[:,None],axis=0))).sum(1)
                    valid=freq>=.4;total=float(power[valid].sum());ordered=np.flatnonzero(valid)[np.argsort(power[valid])[-3:]][::-1]
                    channels[name]=dict(peak_hz=float(freq[ordered[0]]),top3_hz=freq[ordered].tolist(),
                        fraction_above5hz=float(power[freq>=5].sum()/total),fraction_above10hz=float(power[freq>=10].sum()/total))
                    spectrum.append(power/total)
                rows.append(dict(checkpoint=run['checkpoint'],sample_id=sid,channels=channels,window_s=[10,15],
                    sample_hz=1/dt,nyquist_hz=.5/dt,resolution_hz=1/(mask.sum()*dt)))
                if sid in ('baseline_h26','armature_x2_h26'):plots.append((Path(run['checkpoint']).stem,sid,freq,spectrum))
    (args.output/'spectrum.json').write_text(json.dumps(dict(rows=rows,physics_steps=0,model_forwards=0,
        method='Linear detrend, Hann window, summed4-joint spectral power; fractions normalized above0.4Hz',
        limitations='Control recorded at100Hz from lastPDsubstep; cannot establish full500Hz torque spectrum or real-robot frequency; above5Hz is a descriptive band, not a pass/fail rule',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(plots),1,figsize=(10,2.7*len(plots)),squeeze=False)
    for ax,(model,sid,freq,powers) in zip(axes[:,0],plots):
        for name,power in zip(('target angle','actual angle','requested torque'),powers):ax.semilogy(freq,np.maximum(power,1e-10),label=name)
        ax.set_xlim(.4,25);ax.set_ylim(1e-7,1);ax.set_title(model+' / '+sid);ax.set_xlabel('Hz');ax.set_ylabel('Normalized power');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'spectrum.png',dpi=150)


if __name__=='__main__':main()
