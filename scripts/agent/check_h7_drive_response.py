#!/usr/bin/env python3
"""Finite offline checks for continuous drive/coast candidates, not acceptance."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sim2sim'))
import probe_h7_drive_response as probe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    out = args.out
    metadata = json.loads((out/'parameters.json').read_text())
    models = metadata['models']
    native = out/'check-native';native.mkdir(exist_ok=True)
    library = probe.install_gain(native)
    try:
        firmware = json.loads((ROOT/'artifacts/h7-bench-fixed300-20261021/firmware-machine-0.json').read_text())
        for name,digest in {**firmware['source_sha256'],**firmware['protected_sha256']}.items():
            path = ROOT/name
            if name.endswith('multimode_model3023/chuanliantui.xml') and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                path = ROOT/'artifacts/model-versions'/digest/'source/model.xml'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest,name
        module = probe.load_data_module()
        s,pulses = module.load('validation-complete')
        baseline = mujoco.MjModel.from_xml_path(str(s.model_path))
        wheels = [baseline.joint(n).dofadr[0] for n in probe.dynamics.mirror.WHEELS]
        actuators = [baseline.actuator(n+'_motor').id for n in probe.dynamics.mirror.WHEELS]
        fields = ('body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat',
                  'jnt_axis','jnt_range','eq_data','eq_solref','eq_solimp','geom_pos','geom_quat',
                  'geom_friction','geom_contype','geom_conaffinity','actuator_ctrlrange','actuator_gear')
        orange = np.asarray(json.loads((probe.OLD/'report.json').read_text())['models']['free_armature']['additional_parameters'])
        for name,item in models.items():
            candidate = mujoco.MjModel.from_xml_path(str(out/(name+'.xml')))
            for field in fields:
                np.testing.assert_allclose(getattr(candidate,field),getattr(baseline,field),atol=1e-12,rtol=1e-12)
            for field in probe.dynamics.FIELDS:
                np.testing.assert_array_equal(np.delete(getattr(candidate,'dof_'+field),wheels),np.delete(getattr(baseline,'dof_'+field),wheels))
            for field in ('actuator_gaintype','actuator_gainprm','actuator_biasprm','actuator_biastype'):
                np.testing.assert_array_equal(np.delete(getattr(candidate,field),actuators,axis=0),np.delete(getattr(baseline,field),actuators,axis=0))
            np.testing.assert_array_equal(np.asarray(item['additional_parameters'])[:,0],orange[:,0])
            for stage in item['optimization'].values():
                for trace in stage:assert trace['calls'] == metadata['budget_each_stage_per_wheel']
        saved = list(csv.DictReader((out/'predictions.csv').open()))
        max_difference,checked = 0.,0
        for pulse in pulses:
            for name,item in models.items():
                losses,gain = np.asarray(item['additional_parameters']),np.asarray(item['drive_gain'])
                q,v = probe.bench(pulse,losses,gain)
                expected = pulse.sampled(v)
                got = [float(r['predicted_rad_s']) for r in saved if r['split']=='validation' and r['model']==name and int(r['pulse'])==pulse.info['number']]
                assert len(got) == len(expected)
                max_difference = max(max_difference,float(np.max(abs(expected-got))));checked += len(got)
                q2,v2 = probe.bench(pulse,losses,gain)
                assert np.array_equal(q,q2) and np.array_equal(v,v2)
        assert max_difference < 1e-9
        perturbations = []
        for name,item in models.items():
            for column in (0,1):
                for multiplier in (.99,1.01):
                    changed = {**item,'drive_gain':np.asarray(item['drive_gain']).copy()}
                    changed['drive_gain'][:,column] *= multiplier
                    rows = probe.predict_rows(module,pulses,{name:changed},'validation_diagnostic')
                    error = [r['error_rad_s'] for r in rows if r['phase'] != 'edge_or_initial']
                    perturbations.append({'model':name,'parameter':metadata['gain_order'][column],
                                          'multiplier':multiplier,'stats':probe.dynamics.stats(error)})
        result = {'passed':True,'scope':'finite source/XML/continuous integration/repeat checks; not calibration acceptance',
                  'firmware_sources_and_recorded_protected_hashes_verified':True,'candidate_protected_fields':list(fields),
                  'same_inertia_as_orange_both_candidates':True,'validation_predictions_recomputed':checked,
                  'max_saved_difference':max_difference,'repeated_identical_prediction_bitwise_equal':True,
                  'drive_parameter_1pct_sensitivity':perturbations}
        probe.save(out/'verification.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        mujoco.set_mjcb_act_gain(None)


if __name__ == '__main__':
    main()
