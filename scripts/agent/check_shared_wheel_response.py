#!/usr/bin/env python3
"""Check promoted wheel response, train/native equivalence and recorded replay.

Finite interface/numerical checks only. No hardware, policy, learning or RL
trajectory comparison. Historical input recordings retain their frozen model.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sim2sim'))
sys.path.insert(0,str(ROOT/'mjlab_training/src'))
import probe_h7_drive_response as probe
from wheel_response import prepare_wheel_response, wheel_gain
from ct_mjlab.envs.chuanliantui import model as training_model
from ct_mjlab.envs.chuanliantui.huananhu_action import wheel_response_input


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    active=training_model.MODEL_XML
    selected=ROOT/'artifacts/h7-wheel-best-20261021/selected'
    parameters=json.loads((selected/'parameters.json').read_text())
    native=mujoco.MjModel.from_xml_path(str(active));prepare_wheel_response(native)
    gpu_source=training_model.load_spec().compile()
    reference=mujoco.MjModel.from_xml_path(str(selected/'wheel_response.xml'))
    before=ROOT/'artifacts/model-versions/1ea213314d8fd686841678d28246fbb588d53a5586e08c22ea977e95c0cb4ec5/source/model.xml'
    original=mujoco.MjModel.from_xml_path(str(before))
    fields=('body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat',
            'geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity',
            'mesh_vert','mesh_face','jnt_axis','jnt_pos','jnt_range','qpos0','eq_data','eq_solref','eq_solimp',
            'actuator_ctrlrange','actuator_gear')
    for field in fields:np.testing.assert_array_equal(getattr(native,field),getattr(original,field))
    wheels=[native.joint(n).dofadr[0] for n in probe.dynamics.mirror.WHEELS]
    acts=[native.actuator(n+'_motor').id for n in probe.dynamics.mirror.WHEELS]
    for field in ('dof_armature','dof_damping','dof_frictionloss','actuator_gaintype','actuator_gainprm'):
        np.testing.assert_array_equal(getattr(native,field),getattr(reference,field))
        ids=wheels if field.startswith('dof_') else acts
        np.testing.assert_array_equal(np.delete(getattr(native,field),ids,axis=0),
                                      np.delete(getattr(original,field),ids,axis=0))
    np.testing.assert_array_equal(gpu_source.actuator_gainprm,native.actuator_gainprm)
    assert np.all(gpu_source.actuator_gaintype[acts]==mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(native.actuator_gaintype[acts]==mujoco.mjtGain.mjGAIN_USER)
    # Existing board/calibration/weights must remain exact. The active XML change
    # is explicitly authorized and is checked above instead of its old hash.
    firmware=json.loads((ROOT/'artifacts/h7-bench-fixed300-20261021/firmware-machine-0.json').read_text())
    for name,digest in {**firmware['source_sha256'],**firmware['protected_sha256']}.items():
        if name.endswith('multimode_model3023/chuanliantui.xml'):continue
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    gain=training_model.wheel_response_parameters(gpu_source)
    velocity=np.array([[w,-w] for w in [-80.,-63.,-30.,-1.,0.,1.,30.,63.,80.]])
    command=np.array([[u,-u] for u in [-3.9,-3.,-.1,0.,.1,3.,3.9]])
    uu=np.repeat(command,len(velocity),axis=0);ww=np.tile(velocity,(len(command),1))
    actual=wheel_response_input(torch.tensor(uu),torch.tensor(ww),torch.tensor(gain)).numpy()*gain[:,0]
    expected=uu*gain[:,0]*np.maximum(0,1-np.maximum(0,np.sign(uu)*ww)/gain[:,1])**gain[:,2]
    formula_error=float(np.max(abs(actual-expected)));assert formula_error<1e-12
    # Explicitly cover zero input and the maximum low-speed gain once, not G².
    assert not np.any(actual[uu==0])
    np.testing.assert_allclose(actual[(uu[:,0]==3.9)&(ww[:,0]==0),0],3.9*gain[0,0])
    module=probe.load_data_module();session,pulses=module.load('validation-complete')
    assert session.model_path.resolve()==before.resolve(),'old recording did not load its exact archived model'
    saved=list(csv.DictReader((ROOT/'artifacts/h7-wheel-best-20261021/study/predictions.csv').open()))
    errors=[];max_difference=0.;rows=[]
    for pulse in pulses:
        # Only initialize once. All fitted properties now come from active XML,
        # so no candidate parameters or future states are injected in the loop.
        pulse.fixture=probe.dynamics.Fixture(active,pulse.full_qpos,dt=pulse.dt)
        q,v=probe.previous.fast(pulse,np.zeros((2,3)))
        pred=pulse.sampled(v)
        old=[float(r['predicted_rad_s']) for r in saved if r['split']=='validation'
             and r['model']=='joint_sharp' and int(r['pulse'])==pulse.info['number']]
        assert len(old)==len(pred)
        max_difference=max(max_difference,float(np.max(abs(pred-old))))
        mask=pulse.acc|pulse.dec;errors.extend((pred-pulse.actual_v)[mask])
        for i in range(len(pred)):
            rows.append({'pulse':pulse.info['number'],'time_from_on_s':float(pulse.time[i]-pulse.on),
                         'predicted_rad_s':float(pred[i]),'actual_rad_s':float(pulse.actual_v[i])})
    assert max_difference<1e-8,max_difference
    metrics=probe.dynamics.stats(errors)
    target=json.loads((ROOT/'artifacts/h7-wheel-best-20261021/selection.json').read_text())['validation_metrics']['rmse']
    assert abs(metrics['rmse']-target)<1e-8
    probe.previous.write_csv(args.out/'active-xml-predictions.csv',rows)
    _,coarse_pulses=module.load('validation-complete',dt=.002)
    coarse_errors=[]
    for pulse in coarse_pulses:
        pulse.fixture=probe.dynamics.Fixture(active,pulse.full_qpos,dt=pulse.dt)
        _,v=probe.previous.fast(pulse,np.zeros((2,3)))
        coarse_errors.extend((pulse.sampled(v)-pulse.actual_v)[pulse.acc|pulse.dec])
    coarse_metrics=probe.dynamics.stats(coarse_errors)
    assert np.isfinite(list(coarse_metrics.values())).all()
    result={'passed':True,'scope':__doc__,'active_xml':str(active.relative_to(ROOT)),
            'active_xml_sha256':hashlib.sha256(active.read_bytes()).hexdigest(),
            'protected_fields_unchanged':list(fields),'H7_configuration_and_weights_unchanged':True,
            'native_vs_training_force_formula_max_difference':formula_error,'force_probes':int(actual.size),
            'gain_applied_once':True,'GPU_spec_has_no_user_gain':True,
            'historical_recording_model_hash_preserved':True,'active_xml_replay_max_difference':max_difference,
            'active_xml_replay_metrics':metrics,'no_training_or_hardware':True}
    result['native_2ms_replay_metrics']=coarse_metrics
    result['native_2ms_scope']='native float64 at training timestep with2ms ZOH input grid; not a GPU float32 trajectory comparison or measured delay'
    probe.save(args.out/'report.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
