"""Isolate numerical solver paths with identical copied leg trajectories on GPU."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit
from scripts.agent.real2sim import report_leg_loop_failure as loops


def compare(directory, count):
    report = json.loads((directory/'report.json').read_text()); model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
    mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
    with np.load(directory/'trials.npz') as data: q = data['qpos'].copy(); v = data['qvel'].copy(); times = data['time_s'].copy()
    assert np.array_equal(q[0], np.broadcast_to(q[0, 0], q[0].shape))
    assert np.array_equal(v[0], np.broadcast_to(v[0, 0], v[0].shape))
    differences = []; max_gap = 0.; max_excess = 0.; branch_stable = True
    for branch in (0, 1):
        states = q[:, branch*count:(branch+1)*count, :][:, :, mq]
        differences.append(np.max(abs(np.degrees(states-states[:, :1])), axis=(0, 1)).tolist())
        # 全部复制量比较；代表轨迹另核销点和已标定限位，不能用软化闭链掩盖分叉。
        representative = q[:, branch*count]
        gaps = loops.all_gaps(model, representative)
        physical = loops.audit.metrics(model, times, representative, representative)[0]
        max_gap = max(max_gap, float(gaps.max()*1000))
        max_excess = max(max_excess, physical['max_limit_excess_deg'])
        branch_stable &= physical['first_triangle_area_sign_change_s'] == [None, None]
    return dict(max_identical_clone_difference_deg=differences, max_six_pin_gap_mm=max_gap,
                max_limit_excess_deg=max_excess, original_branch_preserved=branch_stable,
                copied_trajectory_consistent=bool(np.max(differences)<.01),
                constraints_passed=bool(max_gap<.1 and max_excess<.1 and branch_stable),
                gpu_environment_steps=report['new_gpu_environment_steps'])


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profiles', nargs='+', choices=['incremental', 'rebuild', 'cg', 'fixed', 'ordered', 'split'], default=['rebuild', 'cg'])
    parser.add_argument('--session', choices=list(fit.REFERENCES), default='original')
    parser.add_argument('--copies', type=int, default=24)
    parser.add_argument('--friction-model', choices=['speed_weakening', 'constant_static', 'constant_kinetic'], default='speed_weakening')
    parser.add_argument('--regularizations', nargs='+', choices=['original', 'friction', 'loop', 'both'], default=['original'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); fit.wp.init()
    profile = json.loads((args.experiment/'selected-parameters.json').read_text()); inputs = fit.input_data(args.session)
    profile = copy.deepcopy(profile)
    if args.friction_model != 'speed_weakening':
        capacity = profile['friction_nm'] if args.friction_model == 'constant_static' else (
            (np.asarray(profile['motor_dynamic_positive_nm'])+profile['motor_dynamic_negative_nm'])/2).tolist()
        profile['friction_nm'] = list(capacity)
        profile['motor_dynamic_positive_nm'] = list(capacity); profile['motor_dynamic_negative_nm'] = list(capacity)
    fit.search.write_json(args.output/'parameters.json', profile)
    results = {}; total = 0
    for solver in args.profiles:
        for regularization in args.regularizations:
            label = solver if regularization == 'original' else solver+'-'+regularization
            directory = args.output/label
            fit.run([profile]*args.copies, inputs, directory, solver_profile=solver, regularization=regularization)
            results[label] = compare(directory, args.copies); total += results[label]['gpu_environment_steps']
            fit.search.write_json(args.output/'progress.json', results)
            print(json.dumps(dict(solver=label, **results[label])), flush=True)
    fit.search.write_json(args.output/'report.json', dict(results=results, new_gpu_environment_steps=total,
        physical_parameters_changed=args.friction_model!='speed_weakening',
        geometry_mass_mapping_or_limits_changed=False, friction_parameters_changed=args.friction_model!='speed_weakening',
        friction_model=args.friction_model, future_measured_position_feedback=False,
        protected_sources=fit.audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), [])))
    shutil.copy2(__file__, args.output/Path(__file__).name)


if __name__ == '__main__': main()
