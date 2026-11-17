"""Compare coarse/full terrain rays and robot parameters; no behavioral acceptance."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from sim2sim.rmuc2027_arena import ARENA, CHECKPOINT, build_model, model_cache


def load(metadata, arena):
    cache = model_cache(metadata, arena)
    if cache.is_file():
        return mujoco.MjModel.from_binary_path(str(cache))
    model = build_model(metadata, arena)
    cache.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(cache))
    return model


def ray_setup(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    field = [i for i in range(model.ngeom) if model.geom(i).name.startswith('arena/')]
    model.geom_rgba[:, 3] = 0.
    model.geom_rgba[field, 3] = 1.
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    models = [load(metadata, p) for p in (ARENA, args.arena)]
    robot_checks = {}
    for name in ('nq', 'nv', 'nu', 'neq'):
        robot_checks[name] = getattr(models[0], name) == getattr(models[1], name)
    for name in ('jnt_range', 'dof_armature', 'dof_damping', 'dof_frictionloss', 'actuator_gainprm', 'actuator_ctrlrange'):
        robot_checks[name] = np.array_equal(getattr(models[0], name), getattr(models[1], name))
    bodies = [models[0].body(i).name for i in range(1, models[0].nbody)
              if not models[0].body(i).name.startswith('arena/')]
    for field in ('body_mass', 'body_inertia', 'body_ipos'):
        values = [np.array([getattr(m, field)[m.body(n).id] for n in bodies]) for m in models]
        robot_checks[field] = np.array_equal(*values)
    data = [ray_setup(m) for m in models]
    group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    rays = [(np.array([x, y, 5.]), np.array([0., 0., -1.]))
            for x in np.linspace(-13.5, 13.5, 55) for y in np.linspace(-7.5, 7.5, 31)]
    for y in np.linspace(-7.5, 7.5, 31):
        for z in (.15, .3, .5, 1.):
            rays.append((np.array([-14.5, y, z]), np.array([1., 0., 0.])))
            rays.append((np.array([14.5, y, z]), np.array([-1., 0., 0.])))
    values = []
    for point, direction in rays:
        distances = [mujoco.mj_ray(m, d, point, direction, group, True, -1, None)
                     for m, d in zip(models, data)]
        values.append(distances)
    distances = np.asarray(values)
    matched = (distances >= 0).all(1)
    missing = (distances[:, 0] >= 0) & (distances[:, 1] < 0)
    added = (distances[:, 0] < 0) & (distances[:, 1] >= 0)
    error = abs(distances[matched, 0] - distances[matched, 1])
    report = dict(robot_parameters_unchanged=all(robot_checks.values()), robot_checks=robot_checks,
        full_ngeom=models[0].ngeom, coarse_ngeom=models[1].ngeom, rays=len(rays),
        missing_hits=int(missing.sum()), added_hits=int(added.sum()),
        matched_hit_median_distance_difference_m=float(np.median(error)),
        matched_hit_p95_distance_difference_m=float(np.percentile(error, 95)),
        matched_hit_max_distance_difference_m=float(error.max()),
        matched_hit_differences_over_4cm=int((error > .04).sum()),
        scope='Sampled vertical/horizontal first-hit comparison; grazing-ray differences are not a bound on 3D surface error; native playground only',
        optimizer_updates=0, behavioral_acceptance=False)
    np.savez_compressed(args.output / 'rays.npz', distance=distances,
                        origin=np.array([x[0] for x in rays]), direction=np.array([x[1] for x in rays]))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    assert all(robot_checks.values())


if __name__ == '__main__':
    main()
