"""Measured500Hz leg-motion/target/velocity/request spectra; mixed commands stay labelled."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.signal import welch,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .replay_usb_pd_targets import sha,write


def describe(signal,scale=1.):
    f,p=welch(signal*scale,fs=500,nperseg=2000,noverlap=1500,detrend='linear')
    peaks,_=find_peaks(p);peaks=[i for i in peaks if .5<=f[i]<=100]
    peaks=sorted(peaks,key=lambda i:p[i],reverse=True)[:5]
    bands={f'{lo:g}-{hi:g}Hz':float(np.sqrt(p[(f>=lo)&(f<hi)].sum()*(f[1]-f[0]))) for lo,hi in ((.5,3),(3,10),(10,50),(50,100))}
    return dict(band_RMS=bands,top_peaks_Hz=[float(f[i]) for i in peaks]),f,p


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=a.experiment/'rl_valid.csv'
    with source.open() as stream:reader=csv.reader(stream);names=next(reader);data=np.asarray(list(reader),dtype=float)
    columns={n:data[:,i] for i,n in enumerate(names)};rows=[];spectra={}
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
    for j,name in enumerate(('lf','lr','rf','rr')):
        values={}
        for field,scale in (('current',180/np.pi),('target',180/np.pi),('velocity_rad_s',1.),('pd_model_nm',1.)):
            # 排除首1秒启动，但不把余下有行驶指令的记录冒充静止段。
            result,f,psd=describe(columns[field+'_'+name][500:],scale);values[field]=result;spectra[name+'_'+field]=psd
            if field in ('current','target'):axes[0,j].semilogy(f,psd,label='Measured angle' if field=='current' else 'Target angle',lw=.9)
            if field=='pd_model_nm':axes[1,j].semilogy(f,psd,label='Requested torque',lw=.9)
        rows.append(dict(joint=name,fields=values));axes[0,j].set_title(name)
    for ax in axes.flat:ax.set_xlim(.5,50);ax.axvspan(3,10,color='orange',alpha=.12);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in axes[0]:ax.set_ylabel('Angle PSD (deg^2/Hz)')
    for ax in axes[1]:ax.set_ylabel('Request PSD (Nm^2/Hz)');ax.set_xlabel('Frequency (Hz)')
    fig.suptitle('Real model3000 record, t>=1s, mixed driving commands / shaded3-10Hz\n100Hz target updates include hold harmonics; request torque is not measured shaft torque')
    fig.tight_layout();fig.savefig(out/'spectra.png',dpi=140);plt.close(fig)
    np.savez_compressed(out/'spectra.npz',frequency_Hz=f,**spectra)
    comparisons=[]
    for label,name in (('baseline','baseline'),('wheel045','wheel_gain_0.45'),('action30','action_30ms')):
        path=a.experiment/'oscillation-probe-01/traces'/(name+'.npz')
        with np.load(path) as tr:
            comparisons.append(dict(profile=label,scope='simulation20-30s zero command, different command condition from real record',joints=[dict(joint=n,**describe(tr['motor_q'][10000:15000,motor],180/np.pi)[0]) for n,motor in zip(('lf','lr','rf','rr'),(0,1,3,4))]))
    write(out/'summary.json',dict(real_rows=rows,simulated_comparisons=comparisons,source_sha256=sha(source),
        scope='Offline spectra only; contains driving commands and does not establish stationary shaking frequency. Frequency bands describe angle motion separately from feedback velocity and torque request.',
        fs_Hz=500,window_start_s=1,frequency_resolution_Hz=.25,physics_steps=0,training_updates=0))
    shutil.copy2(__file__,out/'analysis-source.py');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
