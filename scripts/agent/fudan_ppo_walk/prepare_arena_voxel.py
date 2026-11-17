"""Preserve continuous support surfaces and coarsen the remaining field details."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import torch

from sim2sim.rmuc2027_arena import ARENA, CHECKPOINT, model_cache

PITCH = .05


def expand_xml(node, source):
    for child in list(node):
        if child.tag == 'include':
            path = source.parent / child.attrib['file']
            included = ET.parse(path).getroot()
            expand_xml(included, path)
            index = list(node).index(child)
            node.remove(child)
            for offset, item in enumerate(list(included)):
                node.insert(index + offset, item)
        else:
            if child.tag in ('mesh', 'texture', 'hfield') and 'file' in child.attrib:
                child.set('file', str((source.parent / child.attrib['file']).resolve()))
            expand_xml(child, source)


def consecutive(values):
    values = np.sort(values)
    boundaries = np.flatnonzero(np.diff(values) != 1)
    return zip(values[np.r_[0, boundaries + 1]], values[np.r_[boundaries, len(values) - 1]])


def restore_original_colliders(root, names):
    # 保留源网格的顶点顺序和局部坐标，避免重新凸包化改变接触算法结果。
    names = set(names)
    selected = []
    meshes = set()
    for filename in ('collision_geoms.xml', 'surface_repair_geoms.xml'):
        included = ET.parse(ARENA.parent / filename).getroot()
        for parent in included.iter():
            for geom in list(parent):
                if geom.tag == 'geom':
                    if geom.get('name') not in names:
                        parent.remove(geom)
                    elif 'mesh' in geom.attrib:
                        meshes.add(geom.attrib['mesh'])
        for child in included:
            if any(node.tag == 'geom' for node in child.iter()):
                selected.append(child)
    assets = root.find('asset')
    for filename in ('collision_assets.xml', 'surface_repair_assets.xml'):
        included = ET.parse(ARENA.parent / filename).getroot()
        expand_xml(included, ARENA.parent / filename)
        for mesh in included.iter('mesh'):
            if mesh.get('name') in meshes:
                assets.append(deepcopy(mesh))
    world = root.find('worldbody')
    for child in selected:
        world.append(deepcopy(child))


def support_faces(vertices, hull, minimum_width):
    faces = vertices[hull.simplices[hull.equations[:, 2] > .85]]
    area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) * .5
    if area.sum() < .01:
        return faces[:0]
    xy = np.unique(faces[:, :, :2].reshape(-1, 2), axis=0)
    polygon = xy[ConvexHull(xy).vertices]
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack((-edges[:, 1], edges[:, 0]), -1)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    widths = np.ptp(polygon @ normals.T, axis=0)
    # 小于双轮轮距的零件顶面不能当成可行驶道路。
    return faces if widths.min() >= minimum_width else faces[:0]


def connected_support_faces(vertices, hull, minimum_width, model, data):
    faces = vertices[hull.simplices[hull.equations[:, 2] > .85]]
    area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) * .5
    if area.sum() < .01:
        return faces[:0], []
    group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    direction = np.array([0., 0., -1.])
    patches = []
    radius = minimum_width / 2
    for triangle in faces:
        centre = triangle.mean(0)
        exposed = mujoco.mj_ray(model, data, centre + [0., 0., .02], direction, group, True, -1, None)
        if exposed < 0 or abs(exposed - .02) > .001:
            continue
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        corners = []
        valid = True
        for dx, dy in ((0., 0.), (-radius, -radius), (-radius, radius), (radius, radius), (radius, -radius)):
            z = centre[2] - (normal[:2] @ [dx, dy]) / normal[2]
            point = np.array([centre[0] + dx, centre[1] + dy, z])
            distance = mujoco.mj_ray(model, data, point + [0., 0., .45], direction, group, True, -1, None)
            if distance < 0 or abs(distance - .45) > .01:
                valid = False
                break
            if dx or dy:
                corners.append(point)
        if valid:
            square = np.asarray(corners)
            patches.extend((square[[0, 1, 2]], square[[0, 2, 3]]))
    # 由原版证明相邻三角片能共同支撑轮距，避免把窄三角片误当作小零件。
    return (faces, patches) if patches else (faces[:0], [])


def clear_road_voxels(cells, triangles):
    removed = 0
    clearance = .45
    for triangle in triangles:
        xy = triangle[:, :2]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if abs(normal[2]) < 1e-12:
            continue
        low = np.floor((xy.min(0) - PITCH) / PITCH).astype(int)
        high = np.floor((xy.max(0) + PITCH) / PITCH).astype(int)
        ix, iy = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
        centres = np.stack(((ix.ravel() + .5) * PITCH, (iy.ravel() + .5) * PITCH), -1)
        edges = np.roll(xy, -1, axis=0) - xy
        delta = centres[:, None] - xy[None]
        side = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
        tolerance = np.linalg.norm(edges, axis=1) * PITCH
        inside = ((side >= -tolerance).all(1) | (side <= tolerance).all(1))
        centres = centres[inside]
        height = triangle[0, 2] - ((centres - triangle[0, :2]) @ normal[:2]) / normal[2]
        raw_x, raw_y = ix.ravel()[inside], iy.ravel()[inside]
        x = raw_x.astype(np.int64) + 32768
        y = raw_y.astype(np.int64) + 32768
        z = np.floor(height / PITCH).astype(np.int64) + 32768
        # 这一带改用原始碰撞件，方块清除后再恢复相交的原始障碍物。
        before = len(cells)
        for offset in range(-1, int(np.ceil(clearance / PITCH)) + 1):
            cells.difference_update((x + (y << 16) + ((z + offset) << 32)).tolist())
        removed += before - len(cells)
    return removed


def prepare(output):
    metadata = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cache = model_cache(metadata, ARENA)
    if not cache.is_file():
        from sim2sim.rmuc2027_arena import build_model
        cache.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(build_model(metadata, ARENA), str(cache))
    model = mujoco.MjModel.from_binary_path(str(cache))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    minimum_width = float(abs(data.body('lfwheel').xpos[1] - data.body('rfwheel').xpos[1]))
    model.geom_rgba[:, 3] = 0.
    field = [i for i in range(model.ngeom) if model.geom(i).name.startswith('arena/')]
    model.geom_rgba[field, 3] = 1.
    output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(ARENA).getroot()
    omitted = {'collision_assets.xml', 'collision_geoms.xml', 'surface_repair_assets.xml', 'surface_repair_geoms.xml'}
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'include' and child.attrib['file'] in omitted:
                parent.remove(child)
    expand_xml(root, ARENA)
    world = root.find('worldbody')
    cells = set()
    original_boxes = []
    continuous_meshes, road_triangles, detail_meshes = [], [], []
    mesh_count = 0
    connected_supports = 0
    for gid in range(model.ngeom):
        if not model.geom(gid).name.startswith(('arena/rmuc2027_surface_', 'arena/rmuc2027_repaired_surface_')):
            continue
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            original_boxes.append(gid)
            corners = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)]) * model.geom_size[gid]
            corners = corners @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            hull = ConvexHull(corners)
            road_triangles.extend(support_faces(corners, hull, minimum_width))
            continue
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        hull = ConvexHull(vertices)
        triangles = vertices[hull.simplices]
        road_faces = support_faces(vertices, hull, minimum_width)
        if not len(road_faces):
            road_faces, patches = connected_support_faces(vertices, hull, minimum_width, model, data)
            if len(road_faces):
                connected_supports += 1
                road_triangles.extend(patches)
        if len(road_faces):
            continuous_meshes.append((gid, vertices))
            road_triangles.extend(road_faces)
            continue
        mesh_count += 1
        points = [vertices]
        for triangle in triangles:
            # 按三角形的长边和高采样，避免细长碎片生成平方级无用网格。
            edges = np.linalg.norm(triangle - np.roll(triangle, -1, axis=0), axis=1)
            edge = int(edges.argmax())
            a, b, c = triangle[edge], triangle[(edge + 1) % 3], triangle[(edge + 2) % 3]
            height = np.linalg.norm(np.cross(b - a, c - a)) / max(edges[edge], 1e-12)
            rows = max(1, int(np.ceil(height / (PITCH / 3))))
            for t in np.linspace(0., 1., rows + 1):
                left, right = a + (c - a) * t, b + (c - b) * t
                columns = max(1, int(np.ceil(np.linalg.norm(right - left) / (PITCH / 3))))
                points.append(np.linspace(left, right, columns + 1))
        index = np.floor(np.concatenate(points) / PITCH).astype(np.int64) + 32768
        assert ((index >= 0) & (index < 65536)).all()
        encoded = index[:, 0] + (index[:, 1] << 16) + (index[:, 2] << 32)
        encoded = np.unique(encoded)
        detail_meshes.append((gid, vertices, encoded))
        cells.update(encoded.tolist())
        if mesh_count % 10000 == 0:
            print(json.dumps(dict(meshes=mesh_count, occupied_voxels=len(cells))), flush=True)
    before_road_clear = cells.copy()
    removed_road_voxels = clear_road_voxels(cells, road_triangles)
    removed = before_road_clear - cells
    restored_details = []
    cells = set()
    for gid, vertices, encoded in detail_meshes:
        # 用实际贡献格判断恢复件，避免包围盒把大量无关细节也恢复回来。
        if any(int(code) in removed for code in encoded):
            restored_details.append((gid, vertices))
        else:
            cells.update(encoded.tolist())
    support_mesh_count = len(continuous_meshes)
    continuous_meshes.extend(restored_details)
    codes = np.sort(np.fromiter(cells, dtype=np.int64))
    # 先沿X连成条，再合并相同条的Y/Z相邻层，精确保持体素占据集合。
    boundaries = np.flatnonzero(np.diff(codes) != 1)
    starts, ends = codes[np.r_[0, boundaries + 1]], codes[np.r_[boundaries, len(codes) - 1]]
    stripes = defaultdict(list)
    for first, last in zip(starts, ends):
        stripes[(int(first >> 32), int(first & 65535), int(last & 65535))].append(int((first >> 16) & 65535))
    slabs = defaultdict(list)
    for (z, x0, x1), ys in stripes.items():
        for y0, y1 in consecutive(ys):
            slabs[(x0, x1, int(y0), int(y1))].append(z)
    boxes = []
    for (x0, x1, y0, y1), zs in slabs.items():
        for z0, z1 in consecutive(zs):
            low = (np.array([x0, y0, z0]) - 32768) * PITCH
            high = (np.array([x1, y1, z1]) - 32768 + 1) * PITCH
            # 相邻块重叠0.4mm，避免浮点边界产生射线漏检或极窄接触缝。
            boxes.append(((low + high) * .5, (high - low) * .5 + .0002))
    def numbers(values):
        return ' '.join(f'{x:.10g}' for x in np.asarray(values).ravel())
    for index, (position, size) in enumerate(boxes):
        ET.SubElement(world, 'geom', name=f'rmuc2027_voxel_{index:05d}', type='box',
            pos=numbers(position), size=numbers(size), contype='1', conaffinity='1', group='3',
            rgba='0 0.8 0.2 0', mass='0', priority='1', friction='.8 .01 .001',
            solref='.004 1', solimp='.95 .99 .001')
    original_names = [model.geom(gid).name.removeprefix('arena/') for gid, _ in continuous_meshes]
    original_names.extend(model.geom(gid).name.removeprefix('arena/') for gid in original_boxes)
    restore_original_colliders(root, original_names)
    ET.ElementTree(root).write(output / 'arena.xml', encoding='utf-8', xml_declaration=True)
    report = dict(method='Original continuous support meshes plus 50mm voxels away from support surfaces', pitch_m=PITCH,
        sampled_surface_triangle_spacing_m=PITCH / 3, original_mesh_geoms=mesh_count + support_mesh_count,
        detail_mesh_geoms=mesh_count,
        continuous_support_meshes=support_mesh_count, restored_near_road_detail_meshes=len(restored_details),
        connected_support_meshes=connected_supports,
        retained_collision_assets='Original XML assets, local coordinates, vertex order and geom attributes',
        removed_road_voxels=removed_road_voxels,
        support_upward_normal_z_min=.85, support_upward_area_min_m2=.01,
        support_minimum_width_m=minimum_width,
        original_geometry_road_clearance_m=.45,
        occupied_voxels=len(cells), merged_voxel_boxes=len(boxes), retained_original_boxes=len(original_boxes),
        box_half_extent_overlap_m=.0002,
        visuals='Original visuals preserved', closures_and_ripple_hfields='Preserved',
        limitations='Non-support details remain coarse; support continuity and dynamic crossings require separate checks; not formal acceptance')
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)


if __name__ == '__main__':
    main()
