"""Measure full-field physics and native viewer synchronization without learning."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from sim2sim.rmuc2027_arena import ARENA, CHECKPOINT, ArenaPlayback, ground_height, model_cache
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--arena', type=Path, default=ARENA)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy, metadata = load_policy(CHECKPOINT)
    model = mujoco.MjModel.from_binary_path(str(model_cache(metadata, args.arena)))
    env = ArenaPlayback(model, metadata)
    env.reset((-5., -3., ground_height(model, -5., -3.)))
    poses = []
    times = []
    with torch.inference_mode():
        for _ in range(200):
            tick = time.perf_counter()
            obs = env.observation()
            action, _ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
            env.step(action[0].numpy())
            times.append(time.perf_counter() - tick)
            poses.append(env.data.qpos.copy())
    report = dict(scope='Finite performance diagnostic, not behavior acceptance',
        simulated_seconds=2., physics_wall_seconds=sum(times), realtime_factor=2. / sum(times),
        policy_step_median_ms=float(np.median(times) * 1000),
        policy_step_p95_ms=float(np.percentile(times, 95) * 1000),
        ngeom=model.ngeom, arena=str(args.arena), optimizer_updates=0)
    np.savez_compressed(args.output / 'poses.npz', qpos=np.asarray(poses))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.viewer:
        from mujoco import viewer
        handle = viewer.launch_passive(model, env.data)
        try:
            handle.sync()
            sync = {}
            for name, state_only in [('full', False), ('state_only', True)]:
                values = []
                for _ in range(20):
                    tick = time.perf_counter()
                    handle.sync(state_only=state_only)
                    values.append(time.perf_counter() - tick)
                sync[name] = dict(median_ms=float(np.median(values) * 1000),
                                  p95_ms=float(np.percentile(values, 95) * 1000))
            report['viewer_sync'] = sync
            report['sync_speedup'] = sync['full']['median_ms'] / sync['state_only']['median_ms']
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            time.sleep(.2)
        finally:
            handle.close()
            time.sleep(.2)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
