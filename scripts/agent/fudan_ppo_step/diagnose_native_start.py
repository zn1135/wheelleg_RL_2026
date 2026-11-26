"""Measure STEP startup in the native arena, with and without a WALK handoff."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk.policy import load_policy
from sim2sim.rmuc2027_arena import ArenaPlayback, REGIONAL_CHECKPOINT
from sim2sim.step_playground import ManualStepPolicies, build_model, layout


def sample(checkpoint, mode, surface, output):
    actor, saved = load_policy(checkpoint)
    switch = ManualStepPolicies(REGIONAL_CHECKPOINT, checkpoint, actor, saved, start_step=mode != 'walk_then_step')
    model = build_model(saved)
    env = ArenaPlayback(model, saved)
    spots = layout(saved)
    spot = next(item for item in spots if item['height'] == (.2 if surface == 'stair' else 0.))
    env.command[:] = (0., 0., .26 if mode == 'walk_then_step' else .35)
    env.reset((*spot['spawn'], 0.), yaw=0.)
    wheel_ids = [model.body(name).id for name in ('lfwheel', 'rfwheel')]
    rows = []

    def step():
        obs = env.observation()
        env.history[-19:-16] = obs[6:9]
        action, _ = switch.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
        env.step(action[0].numpy())

    with torch.inference_mode():
        for _ in range(0 if mode == 'cold_step' else 200):
            step()
        if mode == 'walk_then_step':
            switch.select('step', env)
        # Simulate holding Up without any lateral key after selecting STEP.
        env.command[0] = 2.
        start_h = float(env.data.qpos[2])
        start = time.perf_counter()
        for tick in range(180):
            step()
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, env.data.qpos[3:7])
            forward = float(rotation.reshape(3, 3)[:, 0] @ env.data.qvel[:3])
            rows.append([(tick+1)*env.policy_dt, forward, *env.data.qpos[:3],
                         *env.torque[[2, 5]], *env.data.xpos[wheel_ids, 0]])
        wall_s = time.perf_counter()-start
    data = np.asarray(rows)
    np.savez_compressed(output, data=data, columns=['time', 'vx', 'x', 'y', 'height',
                                                  'wheel_1_Nm', 'wheel_2_Nm', 'wheel_1_x', 'wheel_2_x'])

    def first_at(mask):
        return float(data[mask, 0][0]) if mask.any() else None

    before_edge = data[:, 7:].min(axis=1) < spot['edge'][0] if surface == 'stair' else np.ones(len(data), bool)
    approach = data[before_edge]
    return dict(model=saved['iterations'], surface=surface, start_mode=mode, requested_vx_m_s=2.,
        start_height_m=start_h, playback_step_wall_s=wall_s, simulated_s=1.8,
        action_delay_ms=int(env.delays.action_steps[0]), sensor_delay_ms=int(env.delays.sensor_steps[0]),
        first_0_5_m_s_s=first_at((data[:, 1]>=.5)&before_edge),
        first_1_m_s_s=first_at((data[:, 1]>=1.)&before_edge),
        first_1_9_m_s_s=first_at((data[:, 1]>=1.9)&before_edge),
        vx_at_s={str(t): float(data[np.abs(data[:, 0]-t).argmin(), 1]) for t in (.1, .25, .5, .75, 1., 1.25, 1.5)},
        approach_peak_m_s=float(approach[:, 1].max()),
        initial_wheel_torque_at_cap_fraction=float((abs(data[:50, 5:7])>=3.899).mean()),
        initial_wheel_torque_peak_Nm=float(abs(data[:50, 5:7]).max()),
        policy_switch=switch.record())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--step-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for surface in ('flat', 'stair'):
        for mode in ('cold_step', 'settled_step', 'walk_then_step'):
            rows.append(sample(args.step_checkpoint, mode, surface, args.output/f'{surface}-{mode}.npz'))
    (args.output/'summary.json').write_text(json.dumps(dict(cases=rows, optimizer_updates=0,
        scope='Native MuJoCo startup diagnosis', formal_acceptance=False), indent=2)+'\n')
