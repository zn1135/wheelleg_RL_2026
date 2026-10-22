#!/usr/bin/env python3
"""Read a zero-hold training run and compare paired native CPU rollouts; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'sim2sim'))
import mj_sim2sim_huananhu_multimode as sim


def identity(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.out/Path(__file__).name)
    accumulator = EventAccumulator(str(args.run), size_guidance={'scalars': 0})
    accumulator.Reload()
    scalars = {}
    for tag in ('Train/mean_reward', 'Train/mean_episode_length', 'Loss/value',
                'Episode_Reward/zero_command_hold', 'Metrics/standup/velocity_error',
                'Metrics/standup/height_error', 'Episode_Termination/failure'):
        events = accumulator.Scalars(tag)
        scalars[tag] = [{'step': e.step, 'value': e.value, 'wall_time': e.wall_time} for e in events]
    iterations = accumulator.Scalars('Loss/value')
    (args.out/'tensorboard-scalars.json').write_text(json.dumps(scalars)+'\n')
    torch.set_num_threads(1)
    traces, summaries, starting_poses = {}, {}, []
    for name, checkpoint in (('parent', args.parent), ('candidate', args.candidate)):
        actor = sim.load_actor(checkpoint)
        env = sim.Playback('normal', (0., 0., .27))
        env.rng = np.random.default_rng(42)
        env.reset('ground')
        starting_poses.append((env.data.qpos.copy(), env.data.qvel.copy()))
        positions, velocities, rows = [env.data.qpos.copy()], [env.data.qvel.copy()], [env.metrics()]
        with torch.inference_mode():
            for _ in range(1500):
                action, torque = env.step(lambda obs: actor(torch.from_numpy(obs)[None]).numpy()[0])
                rows.append(env.metrics(action, torque))
                positions.append(env.data.qpos.copy()); velocities.append(env.data.qvel.copy())
        positions, velocities = np.array(positions), np.array(velocities)
        xy = positions[:, :2]
        distance = np.linalg.norm(xy-xy[100], axis=1)
        summaries[name] = {
            'checkpoint': identity(checkpoint), 'forward_2_to_22_m': float(xy[1100,0]-xy[100,0]),
            'max_plane_displacement_2_to_22_m': float(distance[100:1101].max()),
            'forward_2_to_30_m': float(xy[-1,0]-xy[100,0]),
            'both_wheels_contact_fraction_2_to_22': float(np.mean([r['both_wheels_contact'] for r in rows[100:1101]])),
            'base_contact_fraction_2_to_22': float(np.mean([r['base_contact'] for r in rows[100:1101]]))}
        traces[name] = positions
        (args.out/(name+'-cpu.json')).write_text(json.dumps(rows)+'\n')
        np.savez_compressed(args.out/(name+'-cpu.npz'), qpos=positions, qvel=velocities,
                            time_s=np.arange(1501)*.02)
        print(name, json.dumps(summaries[name]), flush=True)
    np.testing.assert_array_equal(starting_poses[0][0], starting_poses[1][0])
    np.testing.assert_array_equal(starting_poses[0][1], starting_poses[1][1])
    reference_path = ROOT/'artifacts/model6848-startup-review-20261022-122136/zero-command-frozen-v2/cpu-ground-seed42.json'
    reference = json.loads(reference_path.read_text())
    reproduction_error = float(abs(traces['parent'][1:,0]-np.array([r['x_m'] for r in reference])).max())
    if reproduction_error > 1e-9:
        raise ValueError(f'Original seed42 baseline changed: {reproduction_error}')
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    events = scalars['Train/mean_reward']
    axes[0].plot([x['step'] for x in events], [x['value'] for x in events])
    axes[0].set(xlabel='Logged iteration', ylabel='Mean training reward', title='Training reward is not hold acceptance')
    for name, states in traces.items():
        axes[1].plot(np.arange(100,1501)*.02, (states[100:,0]-states[100,0])*100, label=name)
    axes[1].axhline(5., color='gray', ls='--'); axes[1].axhline(-5., color='gray', ls='--')
    axes[1].axvline(22., color='gray', ls=':')
    axes[1].set(xlabel='Simulation time [s]', ylabel='Forward displacement from t=2 s [cm]',
                title='Paired native MuJoCo: ground seed42, zero command')
    axes[1].legend()
    for axis in axes: axis.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.out/'training-and-cpu.png', dpi=145); plt.close(fig)
    report = {'scope': 'Offline run inspection and paired native CPU check; not full behavioral/hardware acceptance',
        'run': str(args.run.resolve()), 'logged_first_iteration': iterations[0].step,
        'logged_last_iteration': iterations[-1].step, 'logged_iteration_count': len(iterations),
        'planned_additional_iterations': 1000, 'cpu': summaries,
        'paired_initial_qpos_qvel_equal': True, 'parent_reference_x_max_error_m': reproduction_error,
        'sources': [identity(p) for p in (Path(__file__), Path(sim.__file__), sim.MODEL_ASSET,
                                         sim.POSE_BANK, args.run/'params/env.yaml',
                                         args.run/'params/agent.yaml', reference_path)]}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, 'logged_iterations': len(iterations), 'cpu': summaries}), flush=True)


if __name__ == '__main__':
    main()
