"""Check continuous support against the source terrain and the rejected voxel floor."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import torch

from sim2sim.rmuc2027_arena import ARENA, COARSE_ARENA, CHECKPOINT, ROOT
from scripts.agent.fudan_ppo_walk.check_arena_coarse import load, ray_setup


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    paths = [ARENA, ROOT / 'artifacts/rmuc2027/voxel50-v2/arena.xml', COARSE_ARENA]
    models = [load(saved, path) for path in paths]
    data = [ray_setup(model) for model in models]
    minimum_width = float(abs(data[0].body('lfwheel').xpos[1] - data[0].body('rfwheel').xpos[1]))
    group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    direction = np.array([0., 0., -1.])
    def ray(index, point, hit=None):
        return mujoco.mj_ray(models[index], data[index], point, direction, group, True, -1, hit)
    preserved_names = {models[2].geom(i).name for i in range(models[2].ngeom)}
    profiles = {}
    for name, points in (
        ('spawn_cross_slope', np.array([[-5., y, 5.] for y in np.arange(-4.2, -1.499, .005)])),
        ('spawn_forward', np.array([[x, -3., 5.] for x in np.arange(-6., -2.999, .005)]))):
        heights, road = [], []
        for point in points:
            hit = np.array([-1], dtype=np.int32)
            original = ray(0, point, hit)
            heights.append([5. - original, 5. - ray(1, point), 5. - ray(2, point)])
            road.append(hit[0] >= 0 and models[0].geom(int(hit[0])).name in preserved_names)
        heights, road = np.array(heights).T, np.array(road)
        adjacent = road[1:] & road[:-1]
        profiles[name] = dict(samples=len(points), road_samples=int(road.sum()),
            coarse_detail_samples=int((~road).sum()),
            source_max_step_m=float(np.abs(np.diff(heights[0]))[adjacent].max()),
            rejected_voxel_max_step_m=float(np.abs(np.diff(heights[1]))[adjacent].max()),
            fixed_max_step_m=float(np.abs(np.diff(heights[2]))[adjacent].max()),
            fixed_max_source_error_m=float(np.abs(heights[2, road] - heights[0, road]).max()),
            all_geometry_max_source_error_m=float(np.abs(heights[2] - heights[0]).max()))
        np.savez_compressed(args.output / (name + '.npz'), points=points, height=heights, road=road)
    exposed = []
    exposed_count = 0
    for old in range(models[0].ngeom):
        name = models[0].geom(old).name
        if not name.startswith(('arena/rmuc2027_surface_', 'arena/rmuc2027_repaired_surface_')) or models[0].geom_type[old] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = models[0].geom_dataid[old]
        start, count = models[0].mesh_vertadr[mid], models[0].mesh_vertnum[mid]
        vertices = models[0].mesh_vert[start:start + count] @ data[0].geom_xmat[old].reshape(3, 3).T + data[0].geom_xpos[old]
        hull = ConvexHull(vertices)
        triangles = vertices[hull.simplices[hull.equations[:, 2] > .85]]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * .5
        if area.sum() < .01:
            continue
        for triangle in triangles:
            point = triangle.mean(0) + [0., 0., .02]
            original = ray(0, point)
            if original < 0 or abs(original - .02) > .001:
                continue  # 埋在其他几何体中的面不作为可行驶表面样本。
            exposed_count += 1
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            footprint = []
            for dx, dy in ((0., 0.), (-minimum_width / 2, -minimum_width / 2),
                           (-minimum_width / 2, minimum_width / 2),
                           (minimum_width / 2, -minimum_width / 2), (minimum_width / 2, minimum_width / 2)):
                height = point[2] - .02 - (normal[:2] @ [dx, dy]) / normal[2]
                footprint.append(np.array([point[0] + dx, point[1] + dy, height + .45]))
            original_hits = [ray(0, p) for p in footprint]
            if any(d < 0 or abs(d - .45) > .01 for d in original_hits):
                continue  # 由原版判定轮距范围内是否有连续支撑和机身净空。
            for p, original_hit in zip(footprint, original_hits):
                fixed = ray(2, p)
                exposed.append(dict(name=name, point=p.tolist(), source_distance_m=original_hit,
                    fixed_distance_m=fixed, error_m=abs(fixed - original_hit) if fixed >= 0 else None))
    missing = [r for r in exposed if r['fixed_distance_m'] < 0]
    errors = [r['error_m'] for r in exposed if r['error_m'] is not None]
    failures = [r for r in exposed if r['error_m'] is None or r['error_m'] > .001]
    passed = (not failures and all(p['fixed_max_source_error_m'] <= .001 and
        p['fixed_max_step_m'] <= p['source_max_step_m'] + .001 for p in profiles.values()))
    report = dict(passed=passed, profiles=profiles, sampled_exposed_support_faces=len(exposed),
        support_missing=len(missing), support_error_over_1mm=len(failures),
        source_exposed_face_centroids=exposed_count,
        sampled_navigable_footprints=len(exposed) // 5,
        support_max_source_error_m=max(errors, default=0.), failure_examples=failures[:20],
        scope='5mm local profiles plus original-source exposed faces screened by wheel-span footprint and45cm clearance; source-only screening independent of preserved mesh selection; not all field points or behavioral acceptance',
        optimizer_updates=0)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'support-samples.json').write_text(json.dumps(exposed) + '\n')
    print(json.dumps(report), flush=True)
    assert passed, report


if __name__ == '__main__':
    main()
