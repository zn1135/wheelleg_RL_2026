"""Check the default field perimeter and drive the native robot into each side."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk.policy import load_policy
from sim2sim.rmuc2027_arena import (ARENA, REGIONAL_CHECKPOINT, ArenaPlayback, configure_optimized_field,
                                  ground_height,
                                  model_cache, regional_metadata)
from sim2sim.rmuc2027_field import default_field
from sim2sim.rmuc2027_flat_collision import optimized_field


SIDES = (
    ('east', (13.15, 0.), (1., 0.), 13.98, 8.),
    ('west', (-13.15, 0.), (-1., 0.), 13.98, 8.),
    ('north', (11., 7.15), (0., 1.), 7.98, 14.),
    ('south', (-11., -7.15), (0., -1.), 7.98, 14.),
)


def check_walls(model, data, walls):
    # 只查询新围栏，量到内侧墙面并覆盖四个接合角。
    original_groups = model.geom_group.copy()
    ids = np.array(list(walls.values()))
    assert (model.geom_contype[ids] > 0).all() and (model.geom_conaffinity[ids] > 0).all()
    alpha = model.geom_rgba[ids, 3]
    assert (alpha > 0).all() and (alpha < 1).all()
    count = 0
    try:
        model.geom_group[:] = 0
        model.geom_group[ids] = 2
        group = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
        for _, _, direction, expected, half_width in SIDES:
            normal = np.array([*direction, 0.])
            lateral = np.array([-direction[1], direction[0], 0.])
            for width in np.linspace(-half_width+.03, half_width-.03, 41):
                for height in (-.55, .25, 1.65):
                    origin = width*lateral + [0., 0., height]
                    distance = mujoco.mj_ray(model, data, origin, normal, group, True, -1, None)
                    # 贴角的射线会与相邻墙重合，因此同时避开另一边墙的厚度。
                    assert abs(distance-expected) < 1e-9, (origin, normal, distance)
                    count += 1
        for x in (-13.98, 13.98):
            for y in (-7.98, 7.98):
                corner = np.array([x, y, 0.])
                length = np.linalg.norm(corner)
                for height in (-.55, .25, 1.65):
                    distance = mujoco.mj_ray(model, data, np.array([0., 0., height]), corner/length,
                                             group, True, -1, None)
                    assert abs(distance-length) < 1e-9, (corner, height, distance)
                    count += 1
    finally:
        model.geom_group[:] = original_groups
    return count


@torch.inference_mode()
def run(path):
    torch.set_num_threads(1)
    policy, saved = load_policy(REGIONAL_CHECKPOINT)
    metadata = regional_metadata(saved)
    arena = optimized_field(default_field(ARENA))
    model = mujoco.MjModel.from_binary_path(str(model_cache(metadata, arena)))
    configure_optimized_field(model)
    env = ArenaPlayback(model, metadata)
    mujoco.mj_forward(model, env.data)
    walls = {side: model.geom(f'arena/rmuc2027_fence_{side}').id for side, *_ in SIDES}
    rays = check_walls(model, env.data, walls)
    report = dict(ngeom=model.ngeom, perimeter_rays=rays, approaches={})
    for side, spawn, normal, bound, _ in SIDES:
        direction = np.array(normal)
        env.command[:] = (0., 0., .26)
        env.reset((*spawn, ground_height(model, *spawn)), yaw=np.arctan2(normal[1], normal[0]))
        force = np.empty(6)
        hits = 0
        peak_force = 0.
        root_peak = float(env.data.qpos[:2] @ direction)
        first_contact = None
        def sample():
            nonlocal hits, peak_force
            for i, contact in enumerate(env.data.contact):
                if walls[side] in contact.geom and contact.efc_address >= 0:
                    mujoco.mj_contactForce(model, env.data, i, force)
                    if force[0] > 1.:
                        hits += 1
                        peak_force = max(peak_force, float(force[0]))
        for tick in range(300):
            forward = .85 if tick >= 50 and first_contact is None else 0.
            env.command[0] = forward
            obs = env.observation()
            env.history[-19:-16] = obs[6:9]
            action, _ = policy.act_inference(torch.from_numpy(obs[None]), torch.from_numpy(env.history[None]))
            last_position = env.data.qpos[:3].copy()
            try:
                env.step(action[0].numpy(), sample=sample)
            except FloatingPointError as exc:
                raise AssertionError((side, tick, last_position.tolist(), hits, peak_force)) from exc
            root_peak = max(root_peak, float(env.data.qpos[:2] @ direction))
            if hits and first_contact is None:
                first_contact = tick
            if first_contact is not None and tick == first_contact + 49:
                break
        report['approaches'][side] = dict(spawn=spawn, max_outward_root_m=root_peak,
            wall_face_m=bound, loaded_contact_samples=hits, peak_normal_force_N=peak_force,
            blocked=hits>0 and root_peak<bound)
    (path/'perimeter.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report)
    assert all(case['blocked'] for case in report['approaches'].values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
