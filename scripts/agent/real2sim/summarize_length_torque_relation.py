"""Descriptive smooth torque/length trends, not a unique passive-load identification."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    with np.load('artifacts/length-response-fit/20261109-inputs-02/inputs.npz') as a:
        mask=a['time_s']<=35;l=a['real_length'][mask,1]*1000;tau=a['recorded_tau'][mask,2:]*a['sign'][2:]
    rows=[]
    for start in np.arange(np.floor(l.min()),np.ceil(l.max()),1.):
        selected=(l>=start)&(l<start+1)
        if selected.sum()<5:continue
        rows.append([float(l[selected].mean()),*np.median(tau[selected],axis=0),
            *np.quantile(tau[selected],.1,axis=0),*np.quantile(tau[selected],.9,axis=0),int(selected.sum())])
    rows=np.array(rows);x=np.linspace(rows[0,0],rows[-1,0],301)
    medians=gaussian_filter1d(rows[:,1:3],1.2,axis=0,mode='nearest')
    fitted=PchipInterpolator(rows[:,0],medians,axis=0)(x)
    assert np.all(fitted.min(0)>=tau.min(0)-1e-6) and np.all(fitted.max(0)<=tau.max(0)+1e-6)
    np.savetxt(args.output/'average-torque-length.csv',np.column_stack((x,fitted)),delimiter=',',
        header='actual_length_mm,front_request_nm,rear_request_nm',comments='')
    np.savetxt(args.output/'bins.csv',rows,delimiter=',',
        header='length_mm,front_median,rear_median,front_p10,rear_p10,front_p90,rear_p90,count',comments='')
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for j,ax in enumerate(axes):
        ax.scatter(l[::4],tau[::4,j],s=4,alpha=.12,label='Recorded samples')
        ax.fill_between(rows[:,0],rows[:,3+j],rows[:,5+j],alpha=.15,label='10-90% spread in1mm bins')
        ax.plot(x,fitted[:,j],color='black',lw=1.5,label='Smooth average trend')
        ax.set_ylabel(('Front' if j==0 else 'Rear')+' request [Nm]');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_title('One action: length does not uniquely determine motor torque')
    axes[1].set_xlabel('Actual right-leg length [mm]');fig.tight_layout();fig.savefig(args.output/'torque-vs-length.png',dpi=150)
    report=dict(scope='Mixed moving/holding empirical request trend; not static equilibrium or measured spring force',
        seconds=35,samples=len(l),bins=len(rows),method='1mm-bin medians, Gaussian smoothing1.2 bins, shape-preserving cubic interpolation',
        peak_request_not_exceeded=True,physics_steps=0,policy_training_updates=0)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
