"""Compare identical frozen-candidate GPU inputs across recording and batch sizes."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); fit.wp.init()
    profile = json.loads((args.experiment/'selected-parameters.json').read_text())
    inputs = fit.input_data('original'); model = mujoco.MjModel.from_xml_path(str(args.experiment/'final-original/model.xml'))
    baseline_report=json.loads((args.experiment/'final-original/report.json').read_text())
    mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
    with np.load(args.experiment/'final-original/trials.npz') as saved: baseline = saved['qpos'][::10].copy()
    rows = {}; total = 0
    for count in (1, 24):
        directory = args.output/f'clones-{count:02d}'
        _, report = fit.run([profile]*count, inputs, directory, full=False,
            solver_profile=baseline_report.get('solver_profile','incremental'),
            regularization=baseline_report.get('numerical_regularization','original'))
        total += report['new_gpu_environment_steps']
        with np.load(directory/'trials.npz') as saved:
            q = saved['qpos']; error = np.degrees(q[:, [0, count], :][:, :, mq]-baseline[:, :, mq])
            assert np.array_equal(q[0], np.broadcast_to(baseline[0, 0], q[0].shape))
            within = []
            for branch in (0, 1):
                states = q[:, branch*count:(branch+1)*count, :][:, :, mq]
                within.append(np.max(abs(np.degrees(states-states[:, :1])), axis=(0, 1)).tolist())
        rows[str(count)] = dict(max_difference_from_archived_single_deg=abs(error).max(0).tolist(),
                               max_difference_between_identical_clones_deg=within)
    result = dict(identical_parameters=True, identical_initial_state=True, identical_recorded_torque=True,
        recording='original8s', physics_dt_s=.0002, new_gpu_environment_steps=total, comparisons=rows,
        repeatability_tolerance_deg=.01, reproducible=bool(max(max(np.max(v['max_difference_from_archived_single_deg']),
            np.max(v['max_difference_between_identical_clones_deg'])) for v in rows.values())<.01),
        conclusion_scope='Distinguishes execution/recording/batch sensitivity; does not uniquely identify its solver-level cause',
        hardware_access=False, shared_model_modified=False)
    fit.search.write_json(args.output/'report.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
