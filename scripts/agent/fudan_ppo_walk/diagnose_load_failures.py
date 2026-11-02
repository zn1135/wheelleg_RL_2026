"""Quantify existing failure intervals and load spectra; no physics or policy imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def intervals(mask, time):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = (ends-starts)*.002
    return dict(fraction=float(mask.mean()),runs=len(starts),
        maximum_run_ms=float(lengths.max()*1000) if len(lengths) else 0.,
        first_s=float(time[starts[0]]) if len(starts) else None,
        intervals_s=[[float(time[start]),float(time[end-1]+.002)] for start,end in zip(starts,ends)])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    score = json.loads((args.recording/'diagnostic_summary.json').read_text())
    rows = []
    for sample in score['samples']:
        if sample['passed']:
            continue
        path = args.recording/'traces'/(sample['sample_id']+'.npz')
        with np.load(path) as saved:
            trace = {name:saved[name] for name in saved.files}
        duration = sample['case']['duration']
        last = (trace['time'] >= duration-1.-1e-9) & (trace['time'] < duration-1e-9)
        time = trace['time'][last]
        wheels = trace['wheel_support'][last]
        weight = trace['weight'][last]
        ratio = wheels.sum(-1)/weight
        low_each = (wheels < np.maximum(.1*weight,5.)[:,None]).any(-1)
        low_total = ratio < .8
        high_total = ratio > 1.2
        assert bool(not low_each.any()) == sample['balance']['each_wheel']
        assert bool(not low_total.any()) == sample['balance']['total_min']
        assert bool(not high_total.any()) == sample['balance']['total_max']
        spectrum = np.abs(np.fft.rfft(ratio-ratio.mean()))**2
        spectrum[0] = 0.
        frequencies = np.fft.rfftfreq(len(ratio),.002)
        dominant = float(frequencies[spectrum.argmax()]) if spectrum.sum() > 1e-12 else None
        force = trace['base_contact']
        peak = int(force.argmax())
        rows.append(dict(sample_id=sample['sample_id'],case=sample['case'],
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            last_second=dict(window_s=[duration-1.,duration],samples=len(time),
                weight_N=float(weight.mean()),total_weight_ratio_min=float(ratio.min()),
                total_weight_ratio_max=float(ratio.max()),total_weight_ratio_mean=float(ratio.mean()),
                total_weight_ratio_std=float(ratio.std()),each_wheel_below=intervals(low_each,time),
                total_below80pct=intervals(low_total,time),total_above120pct=intervals(high_total,time),
                any_wheel_load_violation=intervals(low_each|low_total|high_total,time),
                both_wheels_below1N=intervals((wheels<1.).all(-1),time),
                dominant_load_frequency_Hz=dominant,
                max_abs_pitch_deg=float(np.abs(trace['roll_pitch_deg'][last,1]).max()),
                vertical_velocity_rms_m_s=float(np.sqrt(np.mean(trace['linear_world'][last,2]**2)))),
            full_body_contact_over10N=intervals(force>10.,trace['time']),
            body_contact_peak_N=float(force[peak]),body_contact_peak_time_s=float(trace['time'][peak]),
            segment_tracking_passed=all(row['passed'] for row in sample['segments']),
            numerical_failure=sample['numerical_failure']))
    result = dict(scope='Saved7000 GPU failure traces; scoring reasons, not unique physical causality',
        physics_steps=0,model_forwards=0,optimizer_updates=0,samples=rows,
        timing_resolution_s=.002,frequency_resolution_Hz=1.,
        limitations=['Force zero is not a geometric proof of wheel lift',
            'One-second force FFT cannot identify control, contact geometry or integrator as the unique cause',
            'Original full per-substep actions, torques and joint velocities were not recorded'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        last=row['last_second']
        print(json.dumps(dict(sample=row['sample_id'],case=row['case']['id'],
            load_ratio=[last['total_weight_ratio_min'],last['total_weight_ratio_max']],
            bad_fraction=last['any_wheel_load_violation']['fraction'],
            longest_bad_ms=last['any_wheel_load_violation']['maximum_run_ms'],
            bad_runs=last['any_wheel_load_violation']['runs'],dominant_Hz=last['dominant_load_frequency_Hz'],
            body_peak=row['body_contact_peak_N'],body_time=row['body_contact_peak_time_s'],
            body_contact_bad_ms=row['full_body_contact_over10N']['maximum_run_ms'],
            max_pitch=last['max_abs_pitch_deg'],vz_rms=last['vertical_velocity_rms_m_s'])))


if __name__ == '__main__':
    main()
