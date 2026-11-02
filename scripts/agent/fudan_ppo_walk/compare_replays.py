"""Compare saved GPU diagnostic cases and verify their frozen initial parameters."""
import argparse
import json
from pathlib import Path

import numpy as np


def measure(trace, sample):
    case = sample['case']
    start = 10. if case['category'].startswith('stop_') else 2.
    end = case['duration']
    mask = (trace['time'] >= start - 1e-8) & (trace['time'] < end - 1e-8)
    command = case['segments'][-1]['command']
    body = trace['base_contact'][mask]
    wheels = trace['wheel_support'][mask].sum(-1) / trace['weight'][mask]
    return dict(window_s=[start, end], body_contact_mean_N=float(body.mean()),
        body_contact_peak_N=float(body.max()), body_contact_over1N_fraction=float((body > 1.).mean()),
        whole_trajectory_body_contact_peak_N=float(trace['base_contact'].max()),
        wheels_weight_ratio_mean=float(wheels.mean()), wheels_weight_ratio_min=float(wheels.min()),
        wheels_weight_ratio_max=float(wheels.max()),
        vx_mean_m_s=float(trace['linear_body'][mask,0].mean()),
        vx_mae_m_s=float(np.abs(trace['linear_body'][mask,0]-command[0]).mean()),
        yaw_mae_rad_s=float(np.abs(trace['angular_body'][mask,2]-command[1]).mean()),
        height_mean_m=float(trace['position'][mask,2].mean()),
        height_mae_m=float(np.abs(trace['position'][mask,2]-command[2]).mean()),
        pitch_max_abs_deg=float(np.abs(trace['roll_pitch_deg'][mask,1]).max()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocols = [json.loads((folder/'protocol.json').read_text()) for folder in (args.before,args.after)]
    assert protocols[0]['samples'] == protocols[1]['samples']
    scores = [{row['sample_id']:row for row in json.loads((folder/'diagnostic_summary.json').read_text())['samples']}
        for folder in (args.before,args.after)]
    rows = []
    for sample in protocols[0]['samples']:
        name = sample['id']
        with np.load(args.before/'fixtures'/(name+'_initial.npz')) as left, np.load(args.after/'fixtures'/(name+'_initial.npz')) as right:
            assert set(left.files) == set(right.files)
            for key in left.files:
                assert np.array_equal(left[key],right[key]), (name,key)
        measurements = []
        for folder, score in zip((args.before,args.after),scores):
            with np.load(folder/'traces'/(name+'.npz')) as trace:
                assert all(np.isfinite(trace[key]).all() for key in trace.files)
                value = measure(trace,sample)
            value.update(technical_pass=score[name]['passed'],complete=score[name]['complete'])
            measurements.append(value)
        rows.append(dict(id=name,case=sample['case'],before=measurements[0],after=measurements[1]))
    report = dict(mode='20_nominal_diagnostics_not_formal987_acceptance',
        initial_parameters_bit_identical=True,physics_steps=0,model_forwards=0,optimizer_updates=0,
        samples=rows,before_passed=sum(row['before']['technical_pass'] for row in rows),
        after_passed=sum(row['after']['technical_pass'] for row in rows),
        retained_passed_ids=[row['id'] for row in rows if row['before']['technical_pass'] and row['after']['technical_pass']],
        regressed_ids=[row['id'] for row in rows if row['before']['technical_pass'] and not row['after']['technical_pass']],
        newly_passed_ids=[row['id'] for row in rows if not row['before']['technical_pass'] and row['after']['technical_pass']],
        before_steady_body_contact_samples=sum(row['before']['body_contact_peak_N']>1. for row in rows),
        after_steady_body_contact_samples=sum(row['after']['body_contact_peak_N']>1. for row in rows))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='samples'},indent=2))
    for row in rows:
        if row['id'] in ('diagnostic_0','diagnostic_5','diagnostic_14','diagnostic_19'):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
