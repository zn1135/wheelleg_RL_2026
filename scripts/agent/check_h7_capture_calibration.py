#!/usr/bin/env python3
"""Finite checks of the new complete-input session fit; no hardware or training."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'sim2sim'))
import calibrate_fixed_h7_huananhu as original_fit
import real2sim_dynamics_huananhu as dynamics

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--fit-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 split=json.loads((a.fit_root/'split.json').read_text());report=json.loads((a.fit_root/'report.json').read_text())
 assert set(split['fit_sessions']).isdisjoint(split['validation_sessions'])
 assert split['fit_sessions']==['fit-complete'] and split['validation_sessions']==['validation-complete']
 assert len(set(split['stream_sha256'].values()))==2
 hashes={}
 for name in split['fit_sessions']+split['validation_sessions']:
  meta=json.loads((a.data_root/name/'meta.json').read_text());audit=json.loads((a.data_root/(name+'-analysis')/'summary.json').read_text())
  digest=hashlib.sha256((a.data_root/name/'stream.jsonl').read_bytes()).hexdigest();assert digest==meta['stream_sha256']==split['stream_sha256'][name]
  assert meta['complete'] and audit['input_capture_pass'] and audit['complete_observed_span']
  assert not audit['event_sequence_gaps'] and not audit['partial_packets_at_connection_end'] and audit['queue_drop_increase']==0
  assert len(audit['pulses'])==4 and {(q['side'],q['command_nm']) for q in audit['pulses']}=={('left',-3.),('left',3.),('right',-3.),('right',3.)}
  assert meta['firmware_manifest']['expected_build']['bench_profile']=='fixed300_no_early_stop_v1'
  hashes[name]=digest
 for group,model in report['models'].items():
  params=np.asarray(model['additional_parameters']);assert params.shape==(2,3) and np.isfinite(params).all()
  assert (params>=np.asarray(report['lower_bounds'])).all() and (params<=np.asarray(report['upper_bounds'])).all()
  if group=='fixed_armature':assert not params[:,0].any()
  for info in model['diagnostics']:
   assert info['objective_calls']==report['budget_per_wheel_per_group']
   assert [r['call'] for r in info['trace']]==list(range(1,info['objective_calls']+1))
 # Independent compiled-field comparison of the exported full-body XML files.
 manifest=json.loads((ROOT/'artifacts/h7-bench-fixed300-20261021/firmware-machine-0.json').read_text())
 for name,digest in manifest['protected_sha256'].items():
  path=ROOT/name
  if name.endswith('multimode_model3023/chuanliantui.xml') and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
   path=ROOT/'artifacts/model-versions'/digest/'source/model.xml'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
 model_key='resources/robots/chuanliantui_new_1/multimode_model3023/chuanliantui.xml'
 model_path=ROOT/model_key
 if hashlib.sha256(model_path.read_bytes()).hexdigest()!=manifest['protected_sha256'][model_key]:
  model_path=ROOT/'artifacts/model-versions'/manifest['protected_sha256'][model_key]/'source/model.xml'
 baseline=mujoco.MjModel.from_xml_path(str(model_path))
 fields=('body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat','jnt_range','jnt_axis','eq_data','eq_solref','eq_solimp','geom_friction','geom_contype','geom_conaffinity','actuator_gear','actuator_gainprm','actuator_ctrlrange')
 wheels=[baseline.joint(name).dofadr[0] for name in dynamics.mirror.WHEELS]
 for group in report['models']:
  candidate=mujoco.MjModel.from_xml_path(str(a.fit_root/(group+'.xml')))
  for field in fields:np.testing.assert_allclose(getattr(candidate,field),getattr(baseline,field),rtol=1e-12,atol=1e-12)
  for field in dynamics.FIELDS:np.testing.assert_array_equal(np.delete(getattr(candidate,'dof_'+field),wheels),np.delete(getattr(baseline,'dof_'+field),wheels))
 # Reconstruct held-out pulses from original raw metadata/CSV and integrate again.
 spec=importlib.util.spec_from_file_location('new_capture_fit',a.data_root/'fit_capture.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 s,pulses=module.load('validation-complete');rows=list(csv.DictReader((a.fit_root/'predictions.csv').open()));max_error=0.;checked=0
 models={'original':np.zeros((2,3)),**{name:np.asarray(x['additional_parameters']) for name,x in report['models'].items()}}
 for pulse in pulses:
  mask=pulse.acc|pulse.dec
  for name,params in models.items():
   q,v=original_fit.fast(pulse,params);expected=pulse.sampled(v)[mask]
   saved=[float(r['predicted_rad_s']) for r in rows if r['split']=='validation' and r['model']==name and int(r['pulse'])==pulse.info['number']]
   assert len(saved)==len(expected);max_error=max(max_error,float(np.max(abs(expected-saved))));checked+=len(saved)
 assert max_error<1e-9
 pulse=pulses[0];q1,v1=original_fit.fast(pulse,models['free_armature']);measured=pulse.actual_v.copy();pulse.actual_v[:]=1e8
 q2,v2=original_fit.fast(pulse,models['free_armature']);pulse.actual_v[:]=measured
 assert np.array_equal(q1,q2) and np.array_equal(v1,v2)
 # Parameter perturbations characterize sensitivity without refitting/selecting on validation.
 sensitivity=[]
 for field in (0,1):
  for scale in (.99,1.01):
   params=models['free_armature'].copy();params[:,field]*=scale
   evaluated=module.predictions(pulses,{'perturbed':params},'validation_diagnostic')
   sensitivity.append({'parameter':dynamics.FIELDS[field],'multiplier':scale,'stats':dynamics.stats([r['error_rad_s'] for r in evaluated])})
 rl_verification=None
 if (a.fit_root/'rl.json').exists():
  rl_session=dynamics.Session(ROOT/original_fit.RL);recorded=dynamics.RecordedRL(rl_session)
  stored=np.load(a.fit_root/'rl-states.npz');rlrows=list(csv.DictReader((a.fit_root/'rl-predictions.csv').open()))
  body=dynamics.BodyPredictor(rl_session.model_path,rl_session.config);wheel_v=body.env.v[dynamics.N2I][4:]
  grouped={}
  for row in rlrows:grouped.setdefault((int(row['row']),row['case'],row['model']),[]).append(row)
  endpoint_max=0.;endpoint_count=0
  for (i,case,name),items in grouped.items():
   key=f'{i}_{case}_{name}';q=stored[key+'_q'];v=stored[key+'_v']
   for row in items:
    h=int(row['horizon_ms']);err=v[h//2,wheel_v]-rl_session.f[i+h//10,130:132]
    pitch=float(np.rad2deg(dynamics.pitch(q[h//2,3:7])-dynamics.pitch(recorded.attitude(rl_session.output_ms[i]+h))))
    endpoint_max=max(endpoint_max,float(np.max(abs(err-[float(row['left_error']),float(row['right_error'])]))),abs(pitch-float(row['pitch_error_deg'])));endpoint_count+=1
  assert endpoint_max<1e-9
  initialization=json.loads((a.fit_root/'rl.json').read_text())['initialization'];i=initialization[0]['row'];q=stored[f'q_{i}'].copy();v=stored[f'v_{i}'].copy();q[2]-=1e-6;v[:3]=initialization[0]['lower_velocity'];replay_max=0.
  for name in models:
   model_path=rl_session.model_path if name=='original' else a.fit_root/(name+'.xml')
   predictor=dynamics.BodyPredictor(model_path,rl_session.config)
   pq,pv=original_fit.torque_predict(predictor,q,v,rl_session,i,'zoh','lower_contact')
   replay_max=max(replay_max,float(np.max(abs(pq-stored[f'{i}_lower_contact_{name}_q']))),float(np.max(abs(pv-stored[f'{i}_lower_contact_{name}_v']))))
  assert replay_max<1e-9
  rl_verification={'endpoints_recomputed':endpoint_count,'max_saved_endpoint_difference':endpoint_max,'one_window_each_model_reintegration_max_difference':replay_max,'scope':'historical100Hz-request/unknown-support diagnostic only; not new complete-input RL validation'}
 result={'passed':True,'scope':'finite raw-session/hash/budget/XML/free-integration checks; not full-body or physical parameter acceptance','session_hashes':hashes,'validation_predictions_recomputed':checked,'max_saved_prediction_difference':max_error,'future_measured_velocities_do_not_affect_prediction':True,'repeated_identical_prediction_bitwise_equal':True,'compiled_protected_fields':list(fields),'nonwheel_dof_fields_unchanged':True,'physical_config_and_recorded_XML_hashes_verified':True,'recorded_XML':str(model_path),'parameter_1pct_sensitivity':sensitivity,'pitch_acceptance':False,'historical_RL_numeric_verification':rl_verification,'mujoco_version':mujoco.__version__}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
