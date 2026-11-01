"""Reproduce early-collapse timeline and same-history crossover statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(mean=float(x.mean()),mae=float(abs(x).mean()),min=float(x.min()),max=float(x.max()))


def load(directory):
    s=json.loads((directory/'summary.json').read_text())
    assert sha(directory/'trajectory.npz')==s['trajectory_sha256']
    with np.load(directory/'trajectory.npz') as z:
        # Only the original first 1.09 seconds are compared; never reset tails.
        t=z['time_s'];m=t<=1.09+1e-9
        values=z['values'][m].copy();t=t[m]
        fixed={k:z[k].copy() for k in z.files if k.startswith('fixed_')}
        weight=z['weight_N'].copy()
    return dict(summary=s,values=values,t=t,fixed=fixed,weight=weight,directory=str(directory))


def series(record,key):
    return record['values'][:,:,slice(*record['summary']['layout'][key])]


def timeline(record):
    times=record['t']; height=series(record,'height')[:,:,0]
    rows=[]
    for tm in (.002,.1,.2,.4,.6,.7,.8,1.,1.09):
        i=int(np.argmin(abs(times-tm)))
        row=dict(time_s=float(times[i]))
        for name in ('height','roll_pitch','swing','leg_length','raw_action','leg_target','support','base_contact','encoder_velocity','true_velocity_before','down_ticks'):
            v=series(record,name)[i]
            row[name]=dict(mean=v.mean(0).tolist(),min=v.min(0).tolist(),max=v.max(0).tolist())
        rows.append(row)
    events={}
    force=series(record,'support');weight=record['weight']
    loaded=(force>=np.maximum(.1*weight,5)[None,:,None]).all(-1)&(force.sum(-1)>=.8*weight[None,:])
    masks=dict(height_below_026=height<.26,height_below_022=height<.22,height_below_018=height<=.18,
        both_wheels_under_5N=(force<5).all(-1),base_contact_over_10N=series(record,'base_contact')[:,:,0]>10,
        roll_over_10deg=abs(series(record,'roll_pitch')[:,:,0])>10,
        pitch_over_10deg=abs(series(record,'roll_pitch')[:,:,1])>10)
    for name,m in masks.items():
        m=m&(times[:,None]>=.2)
        first=[float(times[np.flatnonzero(m[:,e])[0]]) if m[:,e].any() else None for e in range(96)]
        found=[v for v in first if v is not None]
        events[name]=dict(count=len(found),earliest=min(found) if found else None,latest=max(found) if found else None,per_environment=first)
    final=times>=1.-1e-9
    conditions=dict(underloaded=~loaded,low_height=height<=.18,base_contact=series(record,'base_contact')[:,:,0]>10,
        excessive_tilt=series(record,'tilt')[:,:,0]>40)
    return dict(samples=rows,events=events,armed_interval_conditions={name:dict(any_env_count=int(m[final].any(0).sum()),all_env_count=int(m[final].all(0).sum())) for name,m in conditions.items()},
        estimator_error=stats(series(record,'encoder_velocity')-series(record,'true_velocity_before')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnosis',type=Path,required=True);args=p.parse_args()
    out=args.diagnosis/'analysis';out.mkdir(exist_ok=False)
    original={n:load(ROOT/f'artifacts/supermimic-v13-r3-a{n}-seed43-20261101') for n in (4200,5200)}
    assert all(np.array_equal(v,original[5200]['fixed'][k]) for k,v in original[4200]['fixed'].items())
    diagnostic=json.loads((args.diagnosis/'result.json').read_text())
    comparisons={};cross={}
    for run in diagnostic['runs']:
        directory=Path(run['output']);replay=load(directory);label=run['label']
        assert all(np.array_equal(v,original[4200]['fixed'][k]) for k,v in replay['fixed'].items())
        if label in ('controller4200-encoder4200','controller5200-encoder5200'):
            n=4200 if '4200' in label else 5200
            comparisons[label]={key:float(np.max(abs(series(replay,key)-series(original[n],key)))) for key in ('raw_action','roll_pitch','height','encoder_velocity')}
        assert sha(directory/'crossover.pt')==run['crossover_sha256']
        c=torch.load(directory/'crossover.pt',map_location='cpu',weights_only=False)
        a=c['actions'].numpy();h=c['history'].numpy();e=c['encoder_velocity'].numpy()
        # Capture is before physics at t=0, .01...; only common pre-failure rows.
        end=min(100,len(h));a=a[:end];e=e[:end]
        cross[label]=dict(first_history_sha256=hashlib.sha256(h[0].tobytes()).hexdigest(),
            first_actions_mean=a[0].mean(0).tolist(),first_encoder_mean=e[0].mean(0).tolist(),
            controller_effect_old_encoder=stats(a[:, :,2]-a[:,:,0]),
            encoder_effect_old_controller=stats(a[:,:,3]-a[:,:,0]),
            controller_effect_new_encoder=stats(a[:,:,1]-a[:,:,3]),
            encoder_effect_new_controller=stats(a[:,:,1]-a[:,:,2]),
            action_order=c['action_order'])
    assert len({v['first_history_sha256'] for v in cross.values()})==1
    report=dict(original_initial_arrays_equal=True,diagnostic_initial_arrays_equal_original=True,
        first_observation_history_identical_all_runs=True,baseline_reproduction_max_abs=comparisons,
        original={str(k):timeline(v) for k,v in original.items()},crossover=diagnostic,
        same_history_action_comparisons=cross,script_sha256=sha(Path(__file__)),
        scope='Read-only saved trajectories plus frozen 2s crossover. Descriptive thresholds identify events; they do not change environment termination.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
    for n,r in original.items():
        t=r['t'];axes[0].plot(t,series(r,'height')[:,:,0].mean(1),label=f'A{n}')
        axes[1].plot(t,series(r,'roll_pitch')[:,:,1].mean(1),label=f'A{n} pitch')
        axes[1].plot(t,series(r,'roll_pitch')[:,:,0].mean(1),ls='--',label=f'A{n} roll')
        axes[2].plot(t,series(r,'leg_target').mean(1)[:,0],label=f'A{n} left lf0')
        axes[2].plot(t,series(r,'leg_target').mean(1)[:,2],ls='--',label=f'A{n} right rf0')
        axes[3].plot(t,series(r,'encoder_velocity')[:,:,0].mean(1),label=f'A{n} estimated vx')
        axes[3].plot(t,series(r,'true_velocity_before')[:,:,0].mean(1),ls='--',label=f'A{n} true vx')
    axes[0].axhline(.18,color='red',ls=':');labels=['Root height (m)','Body angles (deg)','Leg target (rad)','COM vx (m/s)']
    for ax,label in zip(axes,labels):ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time (s)');fig.suptitle('A4200 vs A5200: mean of all 96 first episodes');fig.tight_layout();fig.savefig(out/'timeline.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(output=str(out),baseline_reproduction=comparisons,runs=diagnostic['runs']),ensure_ascii=False))


if __name__=='__main__':main()
