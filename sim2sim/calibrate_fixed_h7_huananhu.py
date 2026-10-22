#!/usr/bin/env python3
"""Offline session-separated H7 command-response study. Never accesses hardware."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter
import mujoco
from mujoco.rollout import rollout
import numpy as np
from scipy.optimize import least_squares
import real2sim_dynamics_huananhu as d

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/fixed-h7-calibration-20261021'
TRAIN = ['artifacts/wheel-bench-test-20261021/wheel-bench-pulses-20261021',
         'artifacts/wheel-bench-1p70-test-20261021/wheel-bench-1p70-pulses-20261021']
VALID = 'artifacts/wheel-bench-3p00-300ms-speed30-test-20261021/wheel-bench-3p00-speed30-pulses-20261021'
RL = 'artifacts/real2sim-latest-gap-20261021/real2sim-20261021-103443'
# Broad conditional response bounds: J below .020 at 3Nm permits 150rad/s²;
# losses cover observed .05Nm stiction and high-speed coasting, not CAD uncertainty.
UPPER = np.array([.020, .08, .8])
SCALE = np.array([.004, .004, .12])

def save(path, obj):
    d.save_json(path, obj)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path, rows):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def bench_data():
    sessions = [d.Session(ROOT/p) for p in TRAIN+[VALID]]
    pulses = []
    for s in sessions:
        ps, rejected, _ = d.pulses_from_session(s)
        if rejected: raise ValueError(rejected)
        for p in ps:
            p.session_name = s.directory.name
        pulses.append(ps)
    return sessions, pulses[0]+pulses[1], pulses[2]

def fast(p, params, edges=None):
    f = p.fixture; f.parameters(params)
    mujoco.mj_resetData(f.m, f.d)
    f.d.qpos[f.q], f.d.qvel[f.v] = p.q0, p.v0
    initial = np.empty(mujoco.mj_stateSize(f.m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(f.m, f.d, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ctrl = np.zeros((len(p.grid)-1, f.m.nu)); ctrl[:, f.a] = p.inputs('request', edges)
    states, _ = rollout(f.m, f.d, initial, ctrl)
    states = np.vstack([initial, states[0]])
    return states[:, 1:1+f.m.nq][:, f.q], states[:, 1+f.m.nq:1+f.m.nq+f.m.nv][:, f.v]

def fit_group(pulses, adjustable, budget):
    params = np.zeros((2, 3)); info = []
    for side in (0, 1):
        ps = [p for p in pulses if p.side == side]
        free = [0, 1, 2] if adjustable else [1, 2]
        counter, best_cost, best, trace = 0, float('inf'), None, []
        class BudgetReached(Exception): pass
        limit = budget
        def objective(x):
            nonlocal counter, best_cost, best
            if counter >= limit: raise BudgetReached()
            values = np.zeros((2, 3)); values[side, free] = x
            error = np.concatenate([(p.sampled(fast(p, values)[1])-p.actual_v)[p.fit_mask]
                                    /np.sqrt(p.fit_mask.sum()) for p in ps])
            counter += 1; cost = float(error@error)
            if cost < best_cost: best_cost, best = cost, np.array(x).copy()
            if counter % 60 == 0:
                print('fit', 'free' if adjustable else 'fixed', side, counter, best_cost, flush=True)
            trace.append({'evaluation': counter, 'cost': cost, 'x': np.asarray(x).tolist()})
            return error
        # Same two starts, loss, pulses and actual MuJoCo objective-call budget.
        for number, initial in enumerate((np.array([.004, .003, .12]), np.array([.001, .015, .3]))):
            limit = min((number+1)*(budget//2), budget)
            try:
                least_squares(objective, initial[free], bounds=(np.zeros(len(free)), UPPER[free]),
                              x_scale=SCALE[free], max_nfev=budget, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            except BudgetReached: pass
        limit = budget
        rng = np.random.default_rng(2031+side)
        while counter < budget:
            x = np.clip(best+rng.normal(size=len(free))*SCALE[free]*.08, 1e-12, UPPER[free]-1e-12)
            objective(x)
        params[side, free] = best
        info.append({'side': d.mirror.WHEELS[side], 'actual_objective_calls':counter,
                     'best_cost':best_cost, 'trace':trace, 'bound_hit':(best>UPPER[free]*.99).tolist()})
    return params, info

def bench_rows(pulses, models, split, edge=None):
    rows=[]
    for p in pulses:
        for name, values in models.items():
            q, v=fast(p, values, edge(p) if edge else None); pred=p.sampled(v)
            for j,t in enumerate(p.time):
                phase = ('acceleration' if p.on+.005 <= t < p.off else
                         'deceleration' if t > p.off_interval[1]+.01 else 'edge_or_initial')
                speed=abs(p.actual_v[j]); band=next(label for lo,hi,label in
                    [(0,3,'0-3'),(3,10,'3-10'),(10,20,'10-20'),(20,40,'20-40'),(40,np.inf,'40+')] if lo<=speed<hi)
                rows.append({'split':split,'session':p.session_name,'pulse':p.number,'group':p.group,
                             'model':name,'time_s':float(t),'phase':phase,'speed_band_rad_s':band,
                             'actual_v':float(p.actual_v[j]),'predicted_v':float(pred[j]),
                             'error_v':float(pred[j]-p.actual_v[j])})
    return rows

def summarize(rows, keys, column='error_v'):
    groups={}
    for row in rows: groups.setdefault(tuple(row[k] for k in keys),[]).append(row[column])
    return [{**dict(zip(keys,k)),'samples':len(v),**d.stats(v)} for k,v in groups.items()]

def smoke(out):
    sessions, train, valid=bench_data(); p=valid[0]; params=np.array([[.004,.004,.12],[.004,.004,.08]])
    a=p.predict(params,'request');b=fast(p,params)
    err=max(float(np.max(abs(x-y))) for x,y in zip(a,b))
    if err>1e-9: raise ValueError(f'rollout differs from reference mj_step: {err}')
    fixture=p.fixture; before=fixture.fixed_positions.copy();fast(p,params)
    fixed=float(np.max(abs(fixture.d.xpos[1:]-before[1:])))
    # Wheel origins stay fixed; orientation is expected to rotate.
    if fixed>1e-9: raise ValueError('fixture moved')
    rl=d.Session(ROOT/RL); rec=d.RecordedRL(rl); pred=d.BodyPredictor(rl.model_path,rl.config)
    for i in range(len(rl.f)-10):
        if rl.continuous(i,i+10) and np.all((rl.f[i:i+11,109].astype(int)&63)==61):
            try: q,v,meta=pred.measured_state(rec,i)
            except ValueError: continue
            v[:3]=meta['lower_velocity'];pq,pv=torque_predict(pred,q,v,rl,i,'zoh','lower_contact')
            assert np.isfinite(pq).all() and np.isfinite(pv).all();break
    else: raise ValueError('No usable RL state')
    save(out/'smoke.json',{'passed':True,'rollout_vs_mj_step_max_abs':err,'fixture_origin_motion_m':fixed,
                          'rl_row':i,'rl_steps':len(pq)-1,'fit_pulses':len(train),'validation_pulses':len(valid),
                          'independent_session_split':set(TRAIN).isdisjoint({VALID,RL})})

def fit(out, budget):
    sessions, train, valid=bench_data()
    split={'fit_sessions':TRAIN,'validation_session':VALID,'rl_locked_diagnostic_session':RL,
           'overlapping_windows_independent':False,'fit_pulses':len(train),'validation_pulses':len(valid),
           'stream_hashes':{str(s.directory.relative_to(ROOT)):s.meta['stream_sha256'] for s in sessions},
           'locked_before_fit':True,'validation_used_for_selection':False}
    save(out/'split.json',split)
    models={'original':np.zeros((2,3))};details={}
    for name,adjustable in [('fixed_armature',False),('free_armature',True)]:
        values,info=fit_group(train,adjustable,budget);models[name]=values
        details[name]={'additional_parameters':values.tolist(),'diagnostics':info}
        d.export_candidate(sessions[0].model_path,out/(name+'.xml'),values)
        print(name,values.tolist(),flush=True)
    rows=bench_rows(train,models,'fit')+bench_rows(valid,models,'validation')
    write_csv(out/'bench-predictions.csv',rows)
    edges=[]
    for j in range(4):
        edge=lambda p:(float(p.on_interval[j//2]),float(p.off_interval[j%2]))
        rr=bench_rows(valid,models,'validation',edge)
        for r in summarize(rr,['model','group']): r['corner']=j;edges.append(r)
    save(out/'fit.json',{'models':details,'parameter_order':d.FIELDS,'wheel_order':d.mirror.WHEELS,
                        'upper_bounds':UPPER.tolist(),'lower_bounds':[0,0,0],
                        'budget_per_wheel_per_group':budget,'loss':'equal-weight pulse mean-square interior/coast speed',
                        'timing_corner_metrics':edges,
                        'fit_speed_max_rad_s':{n:float(max(abs(p.actual_v).max() for p in train if p.side==i)) for i,n in enumerate(d.mirror.WHEELS)},
                        'summary':summarize(rows,['split','model','group']),
                        'phase_speed_summary':summarize(rows,['split','model','group','phase','speed_band_rad_s']),
                        'automatic_adoption':False})
    plot_bench(out,rows)

def torque_predict(predictor,q,v,s,index,method,case):
    # Use the same logged post-gate torque requests for every candidate. No policy/PD rerun.
    for name,value in predictor.contact.items(): getattr(predictor.m,'geom_'+name)[:]=value
    if case=='friction_055':
        for gs in predictor.wheel_geoms: predictor.m.geom_friction[gs,0]=.55
    m,dat=predictor.m,predictor.d
    mujoco.mj_resetData(m,dat);dat.qpos[:],dat.qvel[:]=q,v;mujoco.mj_forward(m,dat)
    t=s.output_ms[index]+np.arange(50)*2
    if method=='linear':
        torque=np.column_stack([np.interp(t,s.output_ms,s.torque[:,k]) for k in range(6)])
    else:
        ids=np.searchsorted(s.output_ms,t,side='right')-1;torque=s.torque[ids]
    if method=='quantized':
        scale=d.current_scale(s.config)
        torque=torque.copy();torque[:,4:]=np.trunc(torque[:,4:]/scale)*scale
    qs,vs=[q.copy()],[v.copy()]
    for u in torque:
        spring=predictor.env._gas_spring_force(dat.qpos[predictor.env.spring_q],dat.qvel[predictor.env.spring_v])
        dat.ctrl[predictor.env.motor_ids]=u[d.I2N]
        dat.ctrl[predictor.env.spring_ids]=spring
        mujoco.mj_step(m,dat);qs.append(dat.qpos.copy());vs.append(dat.qvel.copy())
    if max(dat.warning.number) or not np.isfinite(qs).all() or not np.isfinite(vs).all():raise ValueError('unstable physics')
    return np.asarray(qs),np.asarray(vs)

def rl_run(out,stride,small=False):
    s=d.Session(ROOT/RL); rec=d.RecordedRL(s)
    paths={'original':s.model_path,**{n:out/(n+'.xml') for n in ('fixed_armature','free_armature')}}
    predictors={n:d.BodyPredictor(p,s.config) for n,p in paths.items()}
    first=predictors['original'];rows=[];rejected=Counter();initials=[];saved={};n=0
    cases=[('lower_contact','zoh','lower',-1e-6),('zero_translation','zoh','zero',-1e-6),
           ('floor_plus_1mm','zoh','lower',.001),('friction_055','zoh','lower',-1e-6),
           ('linear_torque','linear','lower',-1e-6),('pitch_plus_01deg','zoh','lower',-1e-6),
           ('quantized_wheel_torque','quantized','lower',-1e-6),('initial_wheel_plus_01','zoh','lower',-1e-6)]
    for i in range(0,len(s.f)-10,stride):
        if not s.continuous(i,i+10) or not np.all((s.f[i:i+11,109].astype(int)&63)==61) or not np.all(s.f[i:i+11,154]==3) or not np.all(np.diff(s.output_ms[i:i+11])==10):
            rejected['invalid_or_not_normal_applied']+=1;continue
        try:q,v,meta=first.measured_state(rec,i)
        except ValueError as e:rejected[str(e).split(':')[0]]+=1;continue
        n+=1;initials.append({'row':i,**meta});saved[f'q_{i}']=q;saved[f'v_{i}']=v
        for case,method,translation,gap in cases:
            iq,iv=q.copy(),v.copy();iq[2]+=gap
            if translation=='lower':iv[:3]=meta['lower_velocity']
            if case=='pitch_plus_01deg':
                rot=np.array([0.,np.deg2rad(.1),0.]);mujoco.mju_quatIntegrate(iq[3:7],rot,1.)
            if case=='initial_wheel_plus_01':iv[first.env.v[d.sim.WHEELS]]+=.1
            for name,pred in predictors.items():
                pq,pv=torque_predict(pred,iq,iv,s,i,method,case)
                saved[f'{i}_{case}_{name}_q']=pq;saved[f'{i}_{case}_{name}_v']=pv
                for h in (10,50,100):
                    j=i+h//10;step=h//2
                    wheel=pv[step,pred.env.v][d.N2I][4:]-s.f[j,130:132]
                    rows.append({'row':i,'case':case,'model':name,'horizon_ms':h,
                                 'left_error':float(wheel[0]),'right_error':float(wheel[1]),
                                 'pitch_error_deg':float(np.rad2deg(d.pitch(pq[step,3:7])-d.pitch(rec.attitude(s.output_ms[i]+h)))),
                                 'initial_speed_max':float(max(abs(s.f[i,130:132]))),
                                 'constant_left_error':float(s.f[i,130]-s.f[j,130]),
                                 'constant_right_error':float(s.f[i,131]-s.f[j,131])})
        if n%20==0:print('RL windows',n,'row',i,flush=True)
        if small and n==2:break
    write_csv(out/'rl-predictions.csv',rows);np.savez_compressed(out/'rl-states.npz',**saved)
    metrics={c:summarize(rows,['case','model','horizon_ms'],c) for c in ('left_error','right_error','pitch_error_deg')}
    save(out/'rl.json',{'session':s.summary,'mode':s.wheel_response_mode,'verified_original_P_snapshots':s.wheel_response_verified,
                        'windows':n,'stride':stride,'initialization':initials,'rejected':dict(rejected),'metrics':metrics,
                        'input':'recorded post-gate 100Hz torque snapshots; 2ms zero-order hold, linear sensitivity',
                        'support_force':'unknown human support; no external wrench invented',
                        'translation':'unmeasured, lower-wheel no-slip estimate and zero sensitivity',
                        'independent_replicates':1,'automatic_adoption':False})
    plot_rl(out,rows)

def plot_bench(out,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
    for p,ax in zip(range(1,9),axes.flat):
        rr=[r for r in rows if r['split']=='validation' and r['pulse']==p]
        for name in ('original','fixed_armature','free_armature'):
            x=[r for r in rr if r['model']==name];ax.plot([r['time_s'] for r in x],[r['predicted_v'] for r in x],label=name)
        x=[r for r in rr if r['model']=='original'];ax.plot([r['time_s'] for r in x],[r['actual_v'] for r in x],'k--',label='recorded')
        ax.set_title(f'Validation pulse {p} {x[0]["group"]}');ax.set_ylabel('rad/s');ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'bench-validation.png',dpi=140);plt.close(fig)

def plot_rl(out,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for col,ax in zip(('left_error','right_error','pitch_error_deg'),axes):
        for name in ('original','fixed_armature','free_armature'):
            vals=[]
            for h in (10,50,100):
                rr=[r[col] for r in rows if r['case']=='lower_contact' and r['model']==name and r['horizon_ms']==h]
                vals.append(np.mean(abs(np.array(rr))))
            ax.plot([10,50,100],vals,'o-',label=name)
        ax.set_title(col);ax.set_xlabel('ms');ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'rl-errors.png',dpi=140);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['smoke','fit','rl'])
    ap.add_argument('--out',type=Path,default=BASE/'study');ap.add_argument('--budget',type=int,default=240)
    ap.add_argument('--stride',type=int,default=20);ap.add_argument('--small',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    if args.phase=='smoke':smoke(args.out)
    elif args.phase=='fit':fit(args.out,args.budget)
    else:rl_run(args.out,args.stride,args.small)
    save(args.out/(args.phase+'-run.json'),{'complete':True,'elapsed_s':time.time()-start,'script_sha256':digest(__file__),
        'command':sys.argv,'mujoco':mujoco.__version__,'numpy':np.__version__})
if __name__=='__main__': main()
