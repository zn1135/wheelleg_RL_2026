"""Regional native playback geometry/material/finite-state smoke check; no acceptance or learning."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image
import torch

from sim2sim.rmuc2027_arena import (ArenaPlayback, REGIONAL_CHECKPOINT,
    build_regional_model, ground_height, regional_metadata)
from ct_mjlab.envs.fudan_ppo_walk.terrain import height_map
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, section_dict
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, default=REGIONAL_CHECKPOINT)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    policy, saved = load_policy(a.checkpoint)
    model = build_regional_model(saved)
    env = ArenaPlayback(model, regional_metadata(saved))
    terrain = SimpleNamespace(**saved['terrain'])
    axis, heights, spawns = height_map(terrain)
    ray = lambda x, y: ground_height(model, x, y, regional=True)
    spawn_error = max(abs(ray(x, y)-z) for x, y, z in spawns)
    angles = []
    for x, y, _ in spawns[4:8]:
        angles.append([float(np.degrees(np.arctan(ray(x+dx, y+dy)-ray(x+2*dx, y+2*dy))))
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))])
    np.testing.assert_allclose(angles, np.repeat(np.linspace(*terrain.slope_degrees, 4)[:, None], 4, 1), atol=1e-5)
    assert spawn_error < 1e-6
    # 同一高度场节点逐点核对，防止上下翻转或世界原点偏移。
    rough_error = max(abs(ray(x, y)-heights[i, j]) for i, y in enumerate(axis) if y <= 0.
                      for j, x in enumerate(axis) if x >= 0.)
    assert rough_error < 1e-6, rough_error
    assert np.max(np.abs(model.geom_friction[:, 0]-.8)) < 1e-12
    recipe = json.loads((a.checkpoint.parent/'effective-training-config.json').read_text())
    assert section_dict(FudanWalkCfg().normalization) == recipe['normalization']
    cases, traces = [], {}
    terrain_id = model.body('terrain').id
    with torch.inference_mode():
        for name, xy in (('flat', (-4., 4.)), ('slope5', (.25, 4.)), ('slope10', (8.25, 4.)),
                         ('slope15', (.25, 12.)), ('slope20', (8.25, 12.)), ('rough', (4., -4.))):
            env.command[:] = (0., 0., .26)
            env.reset((*xy, ray(*xy)), yaw=0.)
            contacts, friction_error, rows = 0, 0., []
            for tick in range(300):
                # 停车一秒、行驶一秒、再停车一秒，只检查入口和数值稳定性。
                env.command[0] = .5 if 100 <= tick < 200 else 0.
                obs = env.observation()
                env.history[-19:-16] = obs[6:9]
                action, _ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
                env.step(action[0].numpy())
                rows.append([env.data.time, *env.command, *env.data.qpos[:3], *env.data.qvel[:6]])
                for contact in env.data.contact:
                    if model.geom_bodyid[contact.geom1] == terrain_id or model.geom_bodyid[contact.geom2] == terrain_id:
                        contacts += 1
                        friction_error = max(friction_error, abs(float(contact.friction[0])-.8))
            assert contacts > 0 and friction_error < 1e-12
            traces[name] = np.asarray(rows)
            cases.append(dict(region=name, policy_steps=300, finite=True,
                contacts=contacts, friction_max_error=friction_error,
                final_position=env.data.qpos[:3].tolist()))
    np.savez_compressed(a.output/'traces.npz', **traces)
    env.command[:] = (0., 0., .26)
    env.reset((8.25, 4., ray(8.25, 4.)), yaw=0.)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 800
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    with mujoco.Renderer(model, width=1200, height=800) as renderer:
        for name, lookat, distance, elevation, azimuth in (
                ('terrain-overview', (0., 0., 0.), 37., -65., 90.),
                ('slope10-start', env.data.qpos[:3], 3.5, -25., 150.)):
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = lookat, distance, elevation, azimuth
            renderer.update_scene(env.data, camera=cam, scene_option=opt)
            Image.fromarray(renderer.render().copy()).save(a.output/f'{name}.png')
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == digest
    report = dict(passed=True, checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=digest,
        iterations=saved['iterations'], ngeom=model.ngeom, pyramid_face_angles_deg=angles,
        spawn_height_max_error_m=spawn_error, rough_grid_max_error_m=rough_error,
        normalization_matches_training=True, timing=saved['control_timing'],
        training_disturbances=saved['disturbances'], playback_disturbances=env.conditions.record(),
        cases=cases, physics_steps=6*301*env.decimation+env.decimation,
        optimizer_updates=0, formal_acceptance=False,
        scope='Native interactive entry smoke check only; not tracking/parking/fall acceptance')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
