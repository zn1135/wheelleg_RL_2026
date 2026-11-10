"""Conditional force-balance check on existing measured paths; no integration or refitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse
from scripts.agent.real2sim import report_leg_response_joint as audit


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); profile = json.loads(args.parameters.read_text())
    paths = dict(wide=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/inverse-preload-hypothesis'),
        original=Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original'),
        slow=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/inverse-near-slow'),
        fast=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/inverse-near-fast'))
    rows = {}
    for name, path in paths.items():
        data = inverse.load(name, path); velocity = data['velocity']; slide_velocity = data['qvel'][:, inverse.SLIDE_INDEX]
        spring = audit.spring_force(data['qpos'][:, inverse.SLIDE_INDEX], slide_velocity, profile)
        basis = data['spring_basis_nm_per_n']
        # basis=-滑杆位移对电机角的导数；干摩擦对滑杆施加与速度相反的力。
        motor_required = data['target']-np.einsum('tij,tj->ti', basis, spring)
        motor_required += np.einsum('tij,tj->ti', basis,
            np.sign(slide_velocity)*np.asarray(profile['spring_friction_n']))
        motor_required -= velocity*np.asarray(profile['damping_nms_rad'])
        capacity = audit.native_motor_friction(data['qpos'][:, inverse.SLIDE_INDEX], velocity, profile)
        prediction = capacity*np.sign(velocity); error = motor_required-prediction
        axes = []
        for j in range(4):
            side = j//2; knee = data['qpos'][:, [1, 6][side]]; low, high = [(-.97, .14), (-.16, 1.01)][side]
            # 避开端点反力与静止滑杆摩擦的未知区间，仅诊断明确运动中的片段。
            moving = (abs(velocity[:, j])>.03)&(abs(slide_velocity[:, side])>.001)&(knee>low+.02)&(knee<high-.02)
            moving[:10] = moving[-10:] = False
            count = int(moving.sum())
            axes.append(dict(moving_interior_samples=count,
                conditional_motor_balance_rmse_nm=float(np.sqrt(np.mean(error[moving, j]**2))) if count else None,
                required_coulomb_negative_power_fraction=float(np.mean((motor_required[:, j]*velocity[:, j])[moving]<-.01)) if count else None,
                required_coulomb_range_p05_p95_nm=np.percentile(motor_required[moving, j], [5, 95]).tolist() if count else None))
        rows[name] = dict(axes=axes, source_hashes={str(path/f):audit.digest(path/f) for f in ('report.json', 'inverse-loads.npz')})
    result = dict(scope='Conditional diagnostic of one frozen candidate, not a new physical parameter fit',
        parameters_sha256=audit.digest(args.parameters), motor_order=['lf0', 'lf00', 'rf0', 'rf00'], sessions=rows,
        physical_steps=0, future_positions_injected_into_gpu_replay=False,
        limitations=['Existing measured-path inverse dynamics uses nominal masses and 110ms derivative smoothing.',
                     'Negative required Coulomb power is inconsistent with this candidate memoryless friction model; it can also indicate stored energy release or other load/model errors.',
                     'This check does not prove all spring/friction candidates infeasible or identify a unique physical cause.'])
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name:row['axes'] for name,row in rows.items()}, indent=2), flush=True)


if __name__ == '__main__': main()
