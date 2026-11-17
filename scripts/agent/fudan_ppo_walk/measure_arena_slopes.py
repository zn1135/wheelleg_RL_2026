"""Measure exposed collision faces in the original cached RMUC2027 CAD conversion."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import torch

from sim2sim.rmuc2027_arena import ARENA, CHECKPOINT, REVISION, model_cache


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    meta = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cache = model_cache(meta, ARENA)
    if not cache.exists():
        raise FileNotFoundError('Original field cache required; this measurement never compiles or simulates the field')
    model = mujoco.MjModel.from_binary_path(str(cache))
    assert model.ngeom == 57187
    # 上游碰撞几何alpha=0，mj_ray会忽略；只在内存副本中设为可见以测真实碰撞面。
    model.geom_rgba[:, 3] = 1.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hulls = {}
    rows = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith('arena/') or model.geom_group[g] != 3:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            if mesh not in hulls:
                start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start+n].astype(float)
                if len(vertices) < 4 or np.linalg.matrix_rank(vertices-vertices[0]) < 3:
                    hulls[mesh] = None
                else:
                    hull = ConvexHull(vertices)
                    hulls[mesh] = (vertices[hull.simplices], hull.equations[:, :3])
            if hulls[mesh] is None:
                continue
            faces, normals = hulls[mesh]
            rotation = data.geom_xmat[g].reshape(3, 3)
            faces = faces@rotation.T+data.geom_xpos[g]
            normals = normals@rotation.T
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = model.geom_size[g]
            faces, normals = [], []
            rotation = data.geom_xmat[g].reshape(3, 3)
            for axis in range(3):
                for sign in (-1, 1):
                    normal = np.eye(3)[axis]*sign
                    other = [i for i in range(3) if i != axis]
                    corners = np.repeat((normal*s)[None], 4, axis=0)
                    corners[:, other] = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*s[other]
                    faces.extend((corners[[0, 1, 2]], corners[[0, 2, 3]]))
                    normals.extend((normal, normal))
            faces = np.array(faces)@rotation.T+data.geom_xpos[g]
            normals = np.array(normals)@rotation.T
        else:
            continue
        area = np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=1)/2
        angle = np.degrees(np.arccos(normals[:, 2].clip(-1, 1)))
        centre = faces.mean(1)
        keep = (area >= .015) & (angle > 2.) & (angle < 50.)
        for triangle, normal, ar, an, pos in zip(faces[keep], normals[keep], area[keep], angle[keep], centre[keep]):
            rows.append(dict(geom=name, geom_id=g, angle_deg=float(an), area_m2=float(ar),
                centre_m=pos.tolist(), normal=normal.tolist(), triangle=triangle.tolist()))
    # 从每个0.1°角度档的大面开始筛选；向下射线确认面没有被其他结构遮住。
    groups = defaultdict(list)
    for row in rows:
        groups[round(row['angle_deg'], 1)].append(row)
    exposed = []
    group_mask = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    for angle, candidates in sorted(groups.items()):
        for row in sorted(candidates, key=lambda r: -r['area_m2'])[:30]:
            point = np.array(row['centre_m'])
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, point+[0., 0., .02],
                np.array([0., 0., -1.]), group_mask, True, -1, hit)
            if abs(distance-.02) > .0001:
                continue
            # 三个分散位置也检查露出，排除只露出一点的内部斜板。
            triangle = np.array(row['triangle'])
            valid = True
            for vertex in triangle:
                sample = .6*point+.4*vertex
                d = mujoco.mj_ray(model, data, sample+[0., 0., .02],
                    np.array([0., 0., -1.]), group_mask, True, -1, None)
                valid &= abs(d-.02) < .0001
            if valid:
                exposed.append(row)
    bins = []
    assert exposed, 'No exposed collision faces measured'
    for angle in sorted({round(r['angle_deg'], 1) for r in exposed}):
        selected = [r for r in exposed if round(r['angle_deg'], 1) == angle]
        bins.append(dict(angle_deg=angle, exposed_samples=len(selected),
            sampled_face_area_m2=sum(r['area_m2'] for r in selected),
            largest_sample=max(selected, key=lambda r: r['area_m2'])))
    report = dict(source_revision=REVISION, source_xml=str(ARENA), cache=str(cache),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(), source_geoms=model.ngeom,
        minimum_triangle_area_m2=.015, exposed_sample_limit_per_angle=30,
        candidate_triangles=len(rows), exposed_triangles=len(exposed), angle_bins=bins,
        physics_steps=0, optimizer_updates=0,
        scope='Exposed original collision hull faces, angle to world horizontal; not official label attribution or robot traversability')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.output/'exposed-faces.json').write_text(json.dumps(exposed, indent=2)+'\n')
    for row in sorted(bins, key=lambda r: -r['sampled_face_area_m2'])[:30]:
        best = row['largest_sample']
        print(row['angle_deg'], round(row['sampled_face_area_m2'], 3), row['exposed_samples'], best['geom'], best['centre_m'], flush=True)


if __name__ == '__main__':
    main()
