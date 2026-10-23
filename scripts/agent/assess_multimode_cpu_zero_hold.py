#!/usr/bin/env python3
"""Compare two actors in native CPU MuJoCo from identical seeded ground poses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import mj_sim2sim_huananhu_multimode as sim


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=float, default=.27)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    dt = sim.DT * sim.DECIMATION
    initial, results = {}, {}
    for name, checkpoint in (('parent', args.parent), ('candidate', args.candidate)):
        actor = sim.load_actor(checkpoint)
        env = sim.Playback('normal', (0., 0., args.height))
        # 先设种子再生成趴地姿态，保证两模型的物理起点完全相同。
        env.rng = np.random.default_rng(args.seed)
        env.reset('ground')
        initial[name] = {'qpos': env.data.qpos.copy(), 'qvel': env.data.qvel.copy()}
        trajectory = [env.metrics()]
        with torch.inference_mode():
            for _ in range(round(25. / dt)):
                action, torque = env.step(lambda obs: actor(torch.from_numpy(obs)[None]).numpy()[0])
                trajectory.append(env.metrics(action, torque))
        window = [row for row in trajectory if 2. - 1e-6 <= row['time_s'] <= 22. + 1e-6]
        xy = np.array([[row['x_m'], row['y_m']] for row in window])
        results[name] = {
            'checkpoint': identity(checkpoint),
            'maximum_displacement_2_to_22_m': float(np.linalg.norm(xy - xy[0], axis=1).max()),
            'both_wheels_contact_fraction': float(np.mean([row['both_wheels_contact'] for row in window])),
            'base_contact_fraction': float(np.mean([row['base_contact'] for row in window])),
            'maximum_tilt_deg': max(max(abs(v) for v in row['roll_pitch_deg']) for row in window),
            'initial_qpos': initial[name]['qpos'].tolist(),
            'initial_qvel': initial[name]['qvel'].tolist(),
            'trajectory': trajectory}
    # 初态不同则拒绝输出“配对结果”，不将姿态差异算成模型效果。
    np.testing.assert_array_equal(initial['parent']['qpos'], initial['candidate']['qpos'])
    np.testing.assert_array_equal(initial['parent']['qvel'], initial['candidate']['qvel'])
    report = {'scope': 'Paired native CPU zero-command replay; not complete behavioral acceptance',
              'seed': args.seed, 'height_command_m': args.height, 'policy_dt_s': dt,
              'physics_dt_s': sim.DT, 'initial_qpos_qvel_equal': True, 'seconds': 25.,
              'measurement_window_s': [2., 22.], 'models': results,
              'sources': [identity(path) for path in (Path(__file__), Path(sim.__file__),
                                                     sim.MODEL_ASSET, sim.POSE_BANK)]}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value['maximum_displacement_2_to_22_m']
                      for name, value in results.items()}), flush=True)


if __name__ == '__main__':
    main()
