"""Summarize frozen wheel-mechanics comparison without new physics or fitting."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .replay_usb_pd_targets import write,sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    root=a.experiment;run=root/'wheel-mechanics-validation-01';out=run/'review';out.mkdir(exist_ok=False)
    result=json.loads((run/'result.json').read_text());plan=json.loads((run/'plan.json').read_text());parameters=json.loads((run/'actual-wheel-parameters.json').read_text())
    original=json.loads(Path('resources/robots/chuanliantui_new_1/mjlab/wheel_mechanical.json').read_text())
    source=root/'mujoco-target-replay-01/inputs.npz'
    with np.load(source) as f:measured=f['current'][:,[0,1,3,4]].copy();real_wheel=f['velocity'][:,[2,5]].copy()
    qids=np.asarray(json.loads((root/'mujoco-target-replay-01/gpu-report.json').read_text())['motor_qpos_indices'])[[0,1,3,4]]
    with np.load(run/'traces.npz') as f:starts=f['starts'].copy();q=f['qpos'].copy();v=f['motor_velocity'].copy()
    n=len(starts);ticks=starts[None,:]+np.arange(101)[:,None];ratios=[];maxima=[];actual=[]
    for j,c in enumerate(plan['profiles']):
        physical={}
        for field in ('armature','damping','frictionloss'):
            data=np.asarray(parameters[field])[j*n:(j+1)*n]
            expected=np.asarray(original[field])*c['parameters'].get(field+'_scale',1.)
            assert np.allclose(data,expected,rtol=1e-6,atol=1e-8)
            physical[field]=data[0].tolist()
        actual.append(dict(profile=c,physical=physical))
        error=np.rad2deg(q[:,j*n:(j+1)*n][:,:,qids]-measured[ticks]);k,w,axis=np.unravel_index(abs(error[1:]).argmax(),error[1:].shape);k+=1
        maxima.append(dict(profile=c,max_joint_error_deg=float(abs(error[k,w,axis])),joint=['lf0','lf00','rf0','rf00'][axis],record_time_s=float(ticks[k,w]*.002)))
    for r in result['rows']:
        c=r['profile']
        if c['parameters']['name']=='baseline':continue
        base=next(row for row in result['rows'] if row['profile']['mode']==c['mode'] and row['profile']['initial']==c['initial'] and row['profile']['parameters']['name']=='baseline')
        for part in ('first16s','last8s'):
            ratio={key:r[part][key]/base[part][key] for key in ('wheel_RMS_rad_s','leg_RMS_deg','gravity_RMS_deg','pitch_RMS_deg','roll_RMS_deg')}
            ratios.append(dict(mode=c['mode'],initial=c['initial'],part=part,ratios=ratio,
                diagnostic_pass=ratio['wheel_RMS_rad_s']<=.9 and max(ratio[k] for k in ratio if k!='wheel_RMS_rad_s')<=1.05,
                all_metrics_improved=all(value<=1 for value in ratio.values())))
    report=dict(scope='Frozen candidate across four initial-state assumptions and both logged-torque/PD-target modes; already inspected single real session, not independent hardware validation',
        rows=result['rows'],ratios=ratios,maximum_joint_errors=maxima,actual_parameters=actual,
        diagnostic_passed=sum(r['diagnostic_pass'] for r in ratios),strict_all_improved=sum(r['all_metrics_improved'] for r in ratios),comparisons=len(ratios),
        wheel_improvement_percent_range=[100*(1-max(r['ratios']['wheel_RMS_rad_s'] for r in ratios)),100*(1-min(r['ratios']['wheel_RMS_rad_s'] for r in ratios))],
        new_physics_steps=0,training_updates=0,adopted=False,source_sha256={str(path):sha(path) for path in (run/'result.json',run/'traces.npz',source)})
    write(out/'summary.json',report);shutil.copy2(__file__,out/'source.py')
    window=int(np.flatnonzero(starts==8500)[0]);time=17+np.arange(101)*.002
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for axis,(ax,label,motor) in enumerate(zip(axes,['Left wheel','Right wheel'],[2,5])):
        ax.plot(time,real_wheel[8500:8601,axis],color='black',lw=2,label='Measured')
        for name,color,text in [('baseline','#008bc4','Original'),('J0.75_F2_D1','#e16d18','Armature x0.75 / friction x2')]:
            j=next(j for j,c in enumerate(plan['profiles']) if c['mode']=='recorded_torque' and c['initial']=='rolling' and c['parameters']['name']==name)
            ax.plot(time,v[:,j*n+window,motor],color=color,lw=1.6,label=text)
        ax.set_title(label);ax.set_ylabel('Wheel speed (rad/s)');ax.grid(alpha=.25);ax.legend(fontsize=9,ncol=3)
    axes[-1].set_xlabel('Time in valid log (s)');fig.suptitle('Recorded torque / frozen mechanical candidate / 17.0-17.2 s\nNo torque scaling, added delay or response lag; original wheel-axis signs',fontsize=11)
    fig.tight_layout();fig.savefig(out/'wheel-window.png',dpi=150);plt.close(fig)
    print(json.dumps({key:report[key] for key in ('diagnostic_passed','strict_all_improved','comparisons','wheel_improvement_percent_range')},indent=2))
    print(json.dumps([r for r in maxima if r['profile']['initial']=='rolling'],indent=2))


if __name__=='__main__':main()
