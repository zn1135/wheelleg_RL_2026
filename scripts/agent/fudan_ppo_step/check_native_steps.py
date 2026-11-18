"""Check native platform geometry/spawns and finite controls; not GPU behavior acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from sim2sim.rmuc2027_arena import ArenaPlayback, build_model as build_plane, ground_height
from sim2sim.step_playground import CHECKPOINT, build_model, layout
from ct_mjlab.fudan_ppo_walk.policy import load_policy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    policy, metadata = load_policy(args.checkpoint)
    model, plane = build_model(metadata), build_plane(metadata, arena=None)
    spots = layout(metadata)
    step = metadata['training_recipe']['step']
    ray = lambda x, y: ground_height(model, x, y, regional=True)
    # 比较真实机器人参数，不把增加地形误作修改机器人控制或物理。
    for field in ('jnt_range', 'jnt_stiffness', 'dof_damping', 'dof_armature',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'eq_data'):
        np.testing.assert_array_equal(getattr(model, field), getattr(plane, field))
    for i in range(1, plane.nbody):
        source = plane.body(i)
        target = model.body(source.name)
        for field in ('mass', 'inertia', 'ipos'):
            np.testing.assert_array_equal(getattr(target, field), getattr(source, field))
    geometry = []
    for spot in spots:
        assert abs(ray(*spot['spawn'])) < 1e-9
        if spot['height']:
            x, y = spot['edge']
            sampled = [ray(x + dx, y) for dx in (-.005, .005, step['depth'] - .005, step['depth'] + .005)]
            np.testing.assert_allclose(sampled, [0., spot['height'], spot['height'], 0.], atol=1e-9)
            np.testing.assert_allclose([ray(x + .5, y + dy) for dy in
                (-step['width']/2 + .005, step['width']/2 - .005, step['width']/2 + .005)],
                [spot['height'], spot['height'], 0.], atol=1e-9)
            platform = model.geom(f'step_{spot["key"]}_1')
            np.testing.assert_allclose(platform.size, [step['depth']/2, step['width']/2, spot['height']/2])
            assert platform.type == mujoco.mjtGeom.mjGEOM_BOX
        geometry.append(dict(key=spot['key'], height_m=spot['height'], spawn=spot['spawn'], ray_checked=True))
    assert np.all(model.geom_friction[:, 0] == .8)
    env = ArenaPlayback(model, metadata)
    terrain_id = model.body('terrain').id
    cases, traces = [], {}
    with torch.inference_mode():
        for spot in spots:
            env.command[:] = (0., 0., .35)
            env.reset((*spot['spawn'], 0.), yaw=0.)
            rows, contacts, friction_error = [], 0, 0.
            for tick in range(200):
                env.command[0] = .5 if 50 <= tick < 150 else 0.
                obs = env.observation()
                env.history[-19:-16] = obs[6:9]
                action, _ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
                env.step(action[0].numpy())
                assert np.all(abs(env.torque) <= [40., 40., 3.9, 40., 40., 3.9])
                rows.append([env.data.time, *env.command, *env.data.qpos[:3], *env.data.qvel[:6]])
                for contact in env.data.contact:
                    if terrain_id in model.geom_bodyid[[contact.geom1, contact.geom2]]:
                        contacts += 1
                        friction_error = max(friction_error, abs(float(contact.friction[0]) - .8))
            assert contacts > 0 and friction_error < 1e-12
            traces['platform_' + spot['key']] = np.asarray(rows)
            cases.append(dict(key=spot['key'], policy_steps=200, finite=True, contacts=contacts,
                              contact_friction_error=friction_error))
    np.savez_compressed(args.output / 'traces.npz', **traces)
    selected = next(p for p in spots if p['height'] == .20)
    env.command[:] = (0., 0., .35)
    env.reset((*selected['spawn'], 0.), yaw=0.)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 800
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, width=1200, height=800) as renderer:
        for name, point, distance, elevation, azimuth in (
            ('overview', (-5., 0., 0.), 47., -65., 90.),
            ('platform200', env.data.qpos[:3], 3.5, -25., 150.)):
            camera = mujoco.MjvCamera()
            camera.lookat[:], camera.distance, camera.elevation, camera.azimuth = point, distance, elevation, azimuth
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            picture = Image.fromarray(renderer.render().copy())
            pen = ImageDraw.Draw(picture)
            pen.rectangle((0, 0, 1200, 30), fill='#18212b')
            pen.text((8, 8), '0 flat | 1:20 2:50 3:80 4:120 5:160 6:180 7:200 8:210 9:220 mm', fill='white')
            picture.save(args.output / (name + '.png'))
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == digest
    report = dict(passed=True, geometry=geometry, cases=cases, ngeom=model.ngeom,
        robot_parameters_unchanged=True, timing=metadata['control_timing'],
        checkpoint_sha256=digest, model_unchanged=True, optimizer_updates=0,
        physics_steps=len(spots) * 201 * env.decimation + env.decimation,
        formal_acceptance=False, scope='Native scene/finite-control smoke check, not stair success or GPU acceptance')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
