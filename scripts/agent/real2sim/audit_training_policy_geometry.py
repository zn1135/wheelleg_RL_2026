"""Check loop gaps and limits on saved GPU policy trajectories; zero integration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim.check_training_loop_integration import digest, trajectory_metrics, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    result = {}; sources = {}; ids = {}
    for branch in ('before', 'after'):
        directory = args.experiment/('policy-'+branch)
        model_path = directory/'fixtures/model.mjb'
        model = mujoco.MjModel.from_binary_path(str(model_path)); sources[str(model_path)] = digest(model_path)
        rows = {}
        for path in sorted((directory/'traces').glob('*.npz')):
            sources[str(path)] = digest(path)
            with np.load(path) as trace:
                assert np.isfinite(trace['qpos']).all() and np.isfinite(trace['motor_qvel_rad_s']).all()
                rows[path.stem] = trajectory_metrics(model, trace['qpos'][:, None], trace['motor_qvel_rad_s'][:, None])
                assert np.allclose(np.diff(trace['qpos_time']), .01)
        ids[branch] = sorted(rows); assert len(rows) == 20
        result[branch] = dict(samples=rows, max_pin_gap_mm=max(x['max_pin_gap_mm'] for x in rows.values()),
            max_joint_limit_excess_deg=max(x['max_joint_limit_excess_deg'] for x in rows.values()),
            original_assembly_branch=all(x['original_assembly_branch'] for x in rows.values()))
    assert ids['before'] == ids['after']
    after = result['after']
    passed = after['max_pin_gap_mm'] < .1 and after['max_joint_limit_excess_deg'] < .1 and after['original_assembly_branch']
    write(args.output/'verification.json', dict(results=result, thresholds=dict(pin_gap_mm=.1, limit_excess_deg=.1),
        sampled_policy_geometry_passed=passed, pose_sample_dt_s=.01, velocity_scope='motor velocities only',
        source_sha256=sources, new_physics_steps=0,
        limitations='Policy poses sampled every10ms; does not bound unrecorded2ms peaks or replace full987 acceptance'))
    print(json.dumps(dict(passed=passed, after_max_pin_gap_mm=after['max_pin_gap_mm'])), flush=True)


if __name__ == '__main__': main()
