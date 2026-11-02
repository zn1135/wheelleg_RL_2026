"""Read paired native CPU diagnostics and their existing GPU baseline; no new physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--gpu-score',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.experiment
    budget=json.loads((root/'budget.json').read_text());source=Path(budget['source_recording'])
    protocols={name:json.loads((path/'protocol.json').read_text()) for name,path in [('gpu',source),('cpu2ms',root/'cpu2ms'),('cpu05ms',root/'cpu05ms')]}
    items=protocols['gpu']['samples']
    assert len(items)==15 and all(p['samples']==items for p in protocols.values())
    assert all(p['mode']=='diagnostic_not_formal_acceptance' for p in protocols.values())
    scores={'gpu':json.loads(args.gpu_score.read_text()),**{name:json.loads((root/name/'score-v2.json').read_text()) for name in ('cpu2ms','cpu05ms')}}
    lookup={name:{r['sample_id']:r for r in report['samples']} for name,report in scores.items()}
    assert all(report['protocol']=='fudan-walk-stage1-v2' for report in scores.values())
    assert sha(Path(budget['checkpoint']))==budget['checkpoint_sha256']
    commands=json.loads((root/'commands.json').read_text())
    for name,substeps in [('cpu2ms',1),('cpu05ms',4)]:
        command=next(c for c in commands if c[c.index('--output')+1]==str(root/name))
        assert command[command.index('--fixtures')+1]==str(source/'fixtures')
        assert command[command.index('--integration-substeps')+1]==str(substeps)
    rows=[];inputs={};measurements={name:[] for name in protocols}
    for sample in items:
        sid=sample['id'];row=dict(sample_id=sid,case=sample['case'],engines={})
        initial=None
        for name,directory in [('gpu',source),('cpu2ms',root/'cpu2ms'),('cpu05ms',root/'cpu05ms')]:
            path=directory/'traces'/(sid+'.npz');inputs[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as trace:
                assert np.isfinite(trace['qpos']).all()
                if initial is None:initial=trace['qpos'][0]
                else:assert np.array_equal(trace['qpos'][0],initial),(sid,name)
                mask=(trace['time']>=2.)&(trace['time']<7.)
                h=trace['position'][mask,2];contact=trace['base_contact'];peak=int(contact.argmax())
                value=dict(complete=lookup[name][sid]['complete'],technical_passed=lookup[name][sid]['passed'],
                    numerical_failure=lookup[name][sid]['numerical_failure'],
                    height_mean_m=float(h.mean()),height_mae_m=float(np.abs(h-.30).mean()),
                    actual_vx_mean_m_s=float(trace['linear_body'][mask,0].mean()),
                    vx_mae_m_s=float(np.abs(trace['linear_body'][mask,0]-sample['case']['segments'][0]['command'][0]).mean()),
                    yaw_mae_rad_s=float(np.abs(trace['angular_body'][mask,2]-sample['case']['segments'][0]['command'][1]).mean()),
                    measurement_pitch_max_deg=float(np.abs(trace['roll_pitch_deg'][mask,1]).max()),
                    whole_pitch_max_deg=float(np.abs(trace['roll_pitch_deg'][:,1]).max()),
                    whole_height_max_m=float(trace['position'][:,2].max()),
                    body_force_peak_N=float(contact[peak]),body_peak_tick=peak,
                    whole_vertical_velocity_max_m_s=float(np.abs(trace['linear_world'][:,2]).max()),
                    failure_reasons=lookup[name][sid]['failure_reasons'])
                if 'stop' in lookup[name][sid]:value['stop']=lookup[name][sid]['stop']
                value['startup_contact']=lookup[name][sid]['body_contact']['startup_over10N']
                row['engines'][name]=value;measurements[name].append(value)
        rows.append(row)
    fixture_paths=[*sorted((source/'fixtures').glob('*_initial.npz')),source/'fixtures/model.mjb',source/'fixtures/mapping.json']
    inputs.update({str(path):sha(path) for path in fixture_paths})
    summaries={name:dict(total=len(values),complete=sum(v['complete'] for v in values),
        technical_passed=sum(v['technical_passed'] for v in values),numerical_failures=sum(v['numerical_failure'] for v in values),
        height_mean_min_m=min(v['height_mean_m'] for v in values),height_mean_max_m=max(v['height_mean_m'] for v in values),
        whole_height_max_m=max(v['whole_height_max_m'] for v in values),
        whole_pitch_max_deg=max(v['whole_pitch_max_deg'] for v in values),
        measurement_pitch_max_deg=max(v['measurement_pitch_max_deg'] for v in values),
        samples_with_body_contact_over10N=[s['sample_id'] for s in rows if s['engines'][name]['body_force_peak_N']>10.]) for name,values in measurements.items()}
    result=dict(scope='15 paired nominal fixed0.30m diagnostics; not full987, randomization or hardware acceptance',
        checkpoint_sha256=budget['checkpoint_sha256'],input_history_contract=json.loads((source/'fixtures/mapping.json').read_text())['reset_contract'],
        same_initial_qpos=True,same_exported_fixture_source=True,integration_dt_s=dict(cpu2ms=.002,cpu05ms=.0005),
        pd_dt_s=.002,policy_dt_s=.01,optimizer_updates=0,additional_physics_steps=0,additional_model_forwards=0,
        gpu_baseline_reused=True,summaries=summaries,samples=rows,inputs_sha256=inputs,
        final_acceptance='pending_user_video_review')
    with args.output.open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
