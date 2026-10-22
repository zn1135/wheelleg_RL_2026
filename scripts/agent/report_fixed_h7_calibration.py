#!/usr/bin/env python3
"""Summarize all saved outputs; no physics, firmware or parameter fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/fixed-h7-calibration-20261021'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=BASE/'study')
    ap.add_argument('--rounding',type=Path,default=BASE/'rounding-final-isolated');a=ap.parse_args()
    fit=json.loads((a.out/'fit.json').read_text());rl=json.loads((a.out/'rl.json').read_text())
    rows=list(csv.DictReader((a.out/'bench-predictions.csv').open()))
    with (a.out/'bench-phase-speed-metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(fit['phase_speed_summary'][0]));w.writeheader();w.writerows(fit['phase_speed_summary'])
    def metric(model,h,col,case='lower_contact'):
        return next(r['mae'] for r in rl['metrics'][col] if r['case']==case and r['model']==model and r['horizon_ms']==h)
    guard=[]
    for model in fit['models']:
        for h in (10,50,100):
            guard.append({'model':model,'horizon_ms':h,
                          'both_wheel_mae_improved':all(metric(model,h,c)<metric('original',h,c) for c in ('left_error','right_error')),
                          'pitch_mae_not_worse':metric(model,h,'pitch_error_deg')<=metric('original',h,'pitch_error_deg'),
                          'pitch_relative_change':metric(model,h,'pitch_error_deg')/metric('original',h,'pitch_error_deg')-1})
    sensitivity=[]
    for col,values in rl['metrics'].items():
        for model in ('original','fixed_armature','free_armature'):
            for h in (10,50,100):
                rs=[r for r in values if r['model']==model and r['horizon_ms']==h]
                sensitivity.append({'metric':col,'model':model,'horizon_ms':h,'min_mae':min(r['mae'] for r in rs),'max_mae':max(r['mae'] for r in rs)})
    rlrows=list(csv.DictReader((a.out/'rl-predictions.csv').open()))
    continuation=[]
    for h in (10,50,100):
        rs=[r for r in rlrows if r['model']=='original' and r['case']=='lower_contact' and int(r['horizon_ms'])==h]
        continuation.append({'horizon_ms':h,**{c:float(np.mean([abs(float(r[c])) for r in rs])) for c in ('constant_left_error','constant_right_error')}})
    report={'calibration_passed':False,'reason':'10/50ms pitch regression, speed extrapolation, sparse actuation telemetry and supported RL data',
            'constant_velocity_baseline_mae_rad_s':continuation,
            'regression_guard':guard,'sensitivity':sensitivity,'rl_windows':rl['windows'],
            'not_independent_replicates':'136 nonoverlapping starts in one supported RL session',
            'parameter_meaning':'equivalent commanded-torque-to-response, not independently identified physical inertia or friction',
            'training_sync_proposal':None,'official_model_changed':False,'H7_changed_by_this_task':False}
    (a.out/'conclusion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['| 时域 | 模型 | 左轮 MAE rad/s | 右轮 MAE rad/s | 俯仰 MAE ° |','|---|---|---:|---:|---:|']
    for h in (10,50,100):
        for model in ('original','fixed_armature','free_armature'):
            lines.append(f'| {h} ms | {model} | {metric(model,h,"left_error"):.3f} | {metric(model,h,"right_error"):.3f} | {metric(model,h,"pitch_error_deg"):.3f} |')
    (a.out/'rl-metrics.md').write_text('\n'.join(lines)+'\n')
    # All models remain in the overview. A second view magnifies measured speeds.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,11),sharex=True)
    for pulse,ax in zip(range(1,9),axes.flat):
        rr=[r for r in rows if r['split']=='validation' and int(r['pulse'])==pulse]
        for model in ('fixed_armature','free_armature'):
            rs=[r for r in rr if r['model']==model]
            ax.plot([float(r['time_s']) for r in rs],[float(r['predicted_v']) for r in rs],label=model)
        rs=[r for r in rr if r['model']=='original']
        ax.plot([float(r['time_s']) for r in rs],[float(r['actual_v']) for r in rs],'k--',label='recorded')
        ax.set_title(f'Pulse {pulse} {rs[0]["group"]}');ax.set_ylim(-50,50);ax.set_ylabel('rad/s');ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.out/'bench-validation-zoom.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for group,ax in zip(('lfwheel+','lfwheel-','rfwheel+','rfwheel-'),axes.flat):
        bands=['0-3','3-10','10-20','20-40','40+'];data=np.full((2,len(bands)),np.nan)
        for pi,phase in enumerate(('acceleration','deceleration')):
            for bi,band in enumerate(bands):
                r=next((r for r in fit['phase_speed_summary'] if r['split']=='validation' and r['model']=='free_armature' and r['group']==group and r['phase']==phase and r['speed_band_rad_s']==band),None)
                if r:
                    data[pi,bi]=r['rmse'];ax.text(bi,pi,f'{r["rmse"]:.1f}\nn={r["samples"]}',ha='center',va='center',fontsize=8)
        im=ax.imshow(data,vmin=0,vmax=40,cmap='YlOrRd');ax.set_xticks(range(5),bands);ax.set_yticks([0,1],['acceleration','deceleration']);ax.set_title(group)
    fig.suptitle('Free-armature validation RMSE (rad/s), by measured speed band; blank = no data')
    fig.tight_layout();fig.savefig(a.out/'phase-speed-validation.png',dpi=140);plt.close(fig)
    files=[ROOT/'sim2sim/calibrate_fixed_h7_huananhu.py',ROOT/'scripts/agent/check_fixed_h7_calibration.py',
           ROOT/'scripts/agent/check_fixed_h7_rounding.py',Path(__file__),ROOT/'sim2sim/real2sim_dynamics_huananhu.py',
           ROOT/'sim2sim/mj_sim2sim_huananhu_multimode.py',ROOT/'sim2sim/real2sim_huananhu.py',
           ROOT/'resources/robots/chuanliantui_new_1/multimode_model3023/chuanliantui.xml',
           ROOT/'deployment/H7_RL_ros/imcalib/config/machine_config.c',
           ROOT/'exports/chuanliantui_multimode/model5849-h7-20261017/model_5849_opset13.onnx',
           ROOT/'scripts/agent/reproduce_fixed_h7_calibration.sh',
           ROOT/'docs/diagnostics/20261021-fixed-h7-real2sim.md']
    files+=list(a.out.glob('*.xml'))+[p for p in a.out.glob('*.json') if p.name!='result-manifest.json']
    files+=list(a.out.glob('*.csv'))+list(a.out.glob('*.npz'))+list(a.out.glob('*.png'))
    files+=list(a.rounding.glob('*.json'))+list(a.rounding.glob('*.npz'))+list(a.rounding.glob('*.png'))
    manifest={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
              'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (a.out/'result-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'calibration_passed':False,'endpoint_check':json.loads((a.out/'verification.json').read_text())['passed'],'rl_windows':rl['windows']}))
if __name__=='__main__':main()
