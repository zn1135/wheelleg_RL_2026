"""Find measured rest-to-motion events and compare conditional breakaway loads; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    datasets = [inverse.load('wide', root/'inverse-preload-hypothesis'),
                inverse.load('original', Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original')),
                inverse.load('slow', root/'inverse-near-slow'), inverse.load('fast', root/'inverse-near-fast')]
    events = []; checks = []
    for data in datasets:
        force = 150.+750.*(data['compression']-.005)-50.*data['qvel'][:, inverse.SLIDE_INDEX]
        residual = data['target']-np.einsum('tij,tj->ti', data['spring_basis_nm_per_n'], force)
        for motor in range(4):
            q = data['qpos'][:, inverse.MOTOR_INDEX[motor]]; v = data['velocity'][:, motor]
            last = -100
            for i in range(35, len(q)-15):
                if i-last < 40 or abs(v[i]) < .03: continue
                quiet = q[i-30:i-10]
                if np.ptp(quiet) > .0015 or np.max(abs(v[i-30:i-10])) > .012: continue
                displacement = q[i+10]-np.median(quiet)
                if abs(displacement) < .003 or np.sign(displacement) != np.sign(v[i]): continue
                sign = np.sign(displacement); last = i
                # 用起动附近的稳健中位数，避免把一次求二阶导的尖峰当静摩擦阈值。
                load = float(np.median(residual[i-2:i+3, motor]))
                events.append(dict(dataset=data['name'], motor=['lf0', 'lf00', 'rf0', 'rf00'][motor],
                    motor_index=motor, time_s=float(data['time_s'][i]), motion_sign=float(sign),
                    previous_quiet_span_deg=float(np.rad2deg(np.ptp(quiet))),
                    next_100ms_displacement_deg=float(np.rad2deg(displacement)),
                    request_nm=float(np.median(data['drive_torque_nm'][i-2:i+3, motor])),
                    compression_m=float(data['compression'][i, motor//2]),
                    conditional_signed_breakaway_nm=float(sign*load), conditional_residual_nm=load,
                    velocity_rad_s=float(v[i]), other_motor_velocity_rad_s=float(data['velocity'][i, motor^1])))
        checks.append(dict(name=data['name'], source=data['source'], recording_sha256=data['stream_sha256'],
                           inverse_sha256=hashlib.sha256((Path(data['source'])/'inverse-loads.npz').read_bytes()).hexdigest()))
    summaries = []
    for motor in range(4):
        group = [e for e in events if e['motor_index'] == motor]
        row = dict(motor=['lf0', 'lf00', 'rf0', 'rf00'][motor], events=len(group))
        if group:
            y = np.array([e['conditional_signed_breakaway_nm'] for e in group])
            request = np.array([abs(e['request_nm']) for e in group])
            compression = np.array([e['compression_m'] for e in group])
            models = dict(constant=np.ones((len(group), 1)), load=np.column_stack((np.ones(len(group)), request)),
                          compression=np.column_stack((np.ones(len(group)), compression/.01)))
            row.update(negative_conditional_load_events=int(np.sum(y < 0)), request_range_nm=[float(request.min()), float(request.max())],
                       compression_range_mm=[float(compression.min()*1000), float(compression.max()*1000)], models={})
            for name, a in models.items():
                upper = [20.] if name == 'constant' else [20., 1. if name == 'load' else 20.]
                fit = lsq_linear(a, y, bounds=(0., upper))
                row['models'][name] = dict(coefficients=fit.x.tolist(), rmse_nm=float(np.sqrt(np.mean((a@fit.x-y)**2))),
                                           rank=int(np.linalg.matrix_rank(a)))
        summaries.append(row)
    report = dict(scope='Conditional event diagnostic only; no physical friction calibration or response acceptance',
                  environment_steps=0, spring_assumption=dict(reference_force_n=150., stiffness_n_m=750., damping_ns_m=50.),
                  event_definition='200ms nearly still, 100ms derivative guard, then >0.03rad/s and >0.003rad displacement in next100ms',
                  sources=checks, summaries=summaries, events=events,
                  limitations=['Smoothed inverse derivatives and nominal mass affect event load estimates',
                               'Spring and seal losses have not been independently measured',
                               'Request torque is not an independent shaft-torque measurement',
                               'Event regressions are descriptive; full continuous GPU response gates remain unchanged'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(summaries, indent=2))


if __name__ == '__main__': main()
