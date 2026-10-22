#!/usr/bin/env python3
"""Verify saved measured-input predictions, splits and protected model fields."""
import argparse
import csv
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sim2sim'))
import calibrate_fixed_h7_huananhu as c
import real2sim_dynamics_huananhu as d


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=c.BASE/'study');a=p.parse_args()
    split=json.loads((a.out/'split.json').read_text());fit=json.loads((a.out/'fit.json').read_text())
    assert not set(split['fit_sessions'])&{split['validation_session'],split['rl_locked_diagnostic_session']}
    assert len(set(split['stream_hashes'].values()))==3
    for name,model in fit['models'].items():
        assert all(x['actual_objective_calls']==fit['budget_per_wheel_per_group'] for x in model['diagnostics'])
    assert np.all(np.array(fit['models']['fixed_armature']['additional_parameters'])[:,0]==0)
    sessions,train,valid=c.bench_data();compiled=[]
    for name in fit['models']:
        m=mujoco.MjModel.from_xml_path(str(a.out/(name+'.xml')))
        original=mujoco.MjModel.from_xml_path(str(sessions[0].model_path))
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_axis','jnt_range','actuator_gear','geom_friction'):
            np.testing.assert_allclose(getattr(m,field),getattr(original,field),rtol=1e-12,atol=1e-12)
        wheel=[m.joint(n).dofadr[0] for n in d.mirror.WHEELS]
        for field in d.FIELDS:
            np.testing.assert_array_equal(np.delete(getattr(m,'dof_'+field),wheel),np.delete(getattr(original,'dof_'+field),wheel))
        compiled.append(name)
    # Reintegrate held-out fixture data, independent of the exported CSV's predictions.
    rows=list(csv.DictReader((a.out/'bench-predictions.csv').open()));max_bench=0.
    for name,params in [('original',np.zeros((2,3)))]+[(n,np.array(v['additional_parameters'])) for n,v in fit['models'].items()]:
        pulse=valid[0];q,v=c.fast(pulse,params)
        rr=[r for r in rows if r['split']=='validation' and int(r['pulse'])==1 and r['model']==name]
        max_bench=max(max_bench,float(np.max(abs(pulse.sampled(v)-np.array([float(r['predicted_v']) for r in rr])))))
    # Recompute every RL endpoint directly from saved states and raw measured targets.
    s=d.Session(ROOT/c.RL);rec=d.RecordedRL(s);states=np.load(a.out/'rl-states.npz')
    rr=list(csv.DictReader((a.out/'rl-predictions.csv').open()));max_endpoint=0.
    pred=d.BodyPredictor(s.model_path,s.config);v_index=pred.env.v[d.N2I][4:]
    for r in rr:
        i,h=int(r['row']),int(r['horizon_ms']);key=f'{i}_{r["case"]}_{r["model"]}'
        err=states[key+'_v'][h//2,v_index]-s.f[i+h//10,130:132]
        pitch=float(np.rad2deg(d.pitch(states[key+'_q'][h//2,3:7])-d.pitch(rec.attitude(s.output_ms[i]+h))))
        max_endpoint=max(max_endpoint,float(np.max(abs(err-[float(r['left_error']),float(r['right_error'])]))),abs(pitch-float(r['pitch_error_deg'])))
    assert max_bench<1e-10 and max_endpoint<1e-10
    i=int(rr[0]['row']);q,v=states[f'q_{i}'].copy(),states[f'v_{i}'].copy()
    meta=next(x for x in json.loads((a.out/'rl.json').read_text())['initialization'] if x['row']==i)
    q[2]-=1e-6;v[:3]=meta['lower_velocity']
    qp,vp=c.torque_predict(pred,q,v,s,i,'zoh','lower_contact')
    max_replay=max(float(np.max(abs(qp-states[f'{i}_lower_contact_original_q']))),float(np.max(abs(vp-states[f'{i}_lower_contact_original_v']))))
    assert max_replay<1e-9
    # Future output measurements may be corrupted without affecting the prediction.
    before=s.f[i+1:,10:29].copy();pd_before=s.f[i+1:,122:132].copy()
    s.f[i+1:,10:29]=10000;s.f[i+1:,122:132]=10000
    qa,va=c.torque_predict(pred,q,v,s,i,'zoh','lower_contact')
    s.f[i+1:,10:29]=before;s.f[i+1:,122:132]=pd_before
    assert np.array_equal(qa,qp) and np.array_equal(va,vp)
    f=s.f;mask=(f[:,109].astype(int)&63)==61
    expected=np.clip(.1*(f[mask,114:116]-f[mask,130:132]),-3.9,3.9)
    torque_error=float(np.max(abs(expected-s.torque[mask,4:])))
    target=np.clip(f[mask,116:122][:,[2,5]]*10,-150,150)
    target_error=float(np.max(abs(target-f[mask,114:116])))
    report={'passed':True,'scope':'finite offline software/data checks, not physical calibration acceptance',
            'models_protected':compiled,'bench_reintegration_max_abs':max_bench,'rl_endpoints_recomputed':len(rr),
            'rl_endpoint_max_abs':max_endpoint,'rl_reintegration_max_abs':max_replay,
            'future_measurements_not_read':True,'recorded_P_torque_max_error_Nm':torque_error,
            'recorded_action_target_max_error_rad_s':target_error,'verified_output_snapshots':int(mask.sum()),
            'mujoco_version':mujoco.__version__}
    assert torque_error<1e-4 and target_error<1e-4
    d.save_json(a.out/'verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
