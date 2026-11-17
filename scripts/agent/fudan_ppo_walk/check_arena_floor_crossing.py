"""Run matched slow crossings of the rejected stepped floor and continuous floor."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from sim2sim.rmuc2027_arena import ARENA, COARSE_ARENA, CHECKPOINT, ROOT, ArenaPlayback, ground_height
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from scripts.agent.fudan_ppo_walk.check_arena_coarse import load


def finalize(output, results):
    with np.load(output / 'source.npz') as a, np.load(output / 'fixed.npz') as b:
        difference = float(np.max(abs(a['qpos'] - b['qpos'])))
        exact = bool(np.array_equal(a['qpos'], b['qpos']))
    passed = difference <= 1e-8 and results['fixed']['bump_physics_ticks'] <= results['source']['bump_physics_ticks']
    report = dict(passed=passed,
        scope='One nominal yaw90deg crossing per terrain; bump means wheel/field contact with horizontal normal ratio>0.2 and normal force>1N; not whole-field behavior acceptance',
        command=dict(stand_s=2., drive_s=5., vx_m_s=.5, yaw_rad_s=0., height_m=.26),
        fixed_source_qpos_exact=exact, fixed_source_qpos_max_error=difference,
        results=results, optimizer_updates=0, formal_acceptance=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    assert passed, report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy, saved = load_policy(CHECKPOINT)
    results = {}
    paths = [('source', ARENA), ('stepped', ROOT / 'artifacts/rmuc2027/voxel50-v2/arena.xml'), ('fixed', COARSE_ARENA)]
    for label, path in paths:
        model = load(saved, path)
        env = ArenaPlayback(model, saved)
        env.reset((-5., -4.2, ground_height(model, -5., -4.2)))
        # 朝+Y穿过已定位的两处人工台阶；同样清零速度并刷新延迟/历史。
        env.data.qpos[3:7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        env.data.qvel[:] = 0.
        env.velocity[:] = 0.
        mujoco.mj_forward(model, env.data)
        env.last_sample_q = env.data.qpos[env.q].copy()
        env.delays.reset(slice(None), torch.from_numpy(env.sensors()[None]))
        env.history = np.tile(env.observation(), 5)
        wheels = {model.body(n).id for n in ('lfwheel', 'rfwheel')}
        contact_ticks = [0]
        force_peak = [0.]
        impulse = [0.]
        original = env.apply_controls
        def controls(action):
            if env.steps >= 200:
                bump = False
                for i in range(env.data.ncon):
                    contact = env.data.contact[i]
                    a, b = int(contact.geom1), int(contact.geom2)
                    if not ((int(model.geom_bodyid[a]) in wheels and model.geom(b).name.startswith('arena/')) or
                            (int(model.geom_bodyid[b]) in wheels and model.geom(a).name.startswith('arena/'))):
                        continue
                    horizontal = float(np.linalg.norm(contact.frame[:2]))
                    if horizontal <= .2:
                        continue
                    force = np.empty(6)
                    mujoco.mj_contactForce(model, env.data, i, force)
                    if abs(force[0]) > 1.:
                        bump = True
                        force_peak[0] = max(force_peak[0], float(abs(force[0])))
                        impulse[0] += float(abs(force[0]) * horizontal * env.dt)
                contact_ticks[0] += int(bump)
            original(action)
        env.apply_controls = controls
        poses = []
        with torch.inference_mode():
            for step in range(700):
                env.command[:] = (.5 if step >= 200 else 0., 0., .26)
                obs = env.observation()
                env.history[-19:-16] = obs[6:9]
                action, _ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
                env.step(action[0].numpy())
                poses.append(env.data.qpos.copy())
        q = np.asarray(poses)
        w, x, y, z = q[:, 3:7].T
        roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1., 1.))
        results[label] = dict(completed=True, policy_steps=700, y_start_m=float(q[0, 1]), y_end_m=float(q[-1, 1]),
            bump_physics_ticks=contact_ticks[0], bump_peak_normal_force_N=force_peak[0],
            bump_horizontal_normal_impulse_Ns=impulse[0],
            peak_roll_pitch_deg=np.rad2deg(abs(np.array([roll, pitch])).max(axis=1)).tolist(),
            finite=bool(np.isfinite(q).all()), action_delay_steps=int(env.delays.action_steps[0]),
            sensor_delay_steps=int(env.delays.sensor_steps[0]))
        np.savez_compressed(args.output / (label + '.npz'), qpos=q, time=np.arange(1, 701) * .01)
        print(json.dumps({label: results[label]}), flush=True)
    finalize(args.output, results)


if __name__ == '__main__':
    main()
