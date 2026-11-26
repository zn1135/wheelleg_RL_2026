"""Merge adjacent horizontal prisms while retaining their occupied volume."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


LAYERS = ('collision', 'surface_repair')
UNITS = 1_000_000  # One integer unit is one micrometre.
BOX = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
SHAPE = {'name', 'type', 'mesh', 'size', 'pos', 'quat'}


def protected(low, high):
    # 台阶顶面、垂直台阶沿及接近和停车区域均保留原碰撞。
    return low[0] <= -4.4 and high[0] >= -7.3 and low[1] <= -3.9 and high[1] >= -7.8


def world_vertices(geom, meshes):
    if set(geom.attrib) & {'euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto'}:
        return None
    if geom.get('type') == 'box':
        vertices = BOX * np.fromstring(geom.get('size'), sep=' ')
    elif geom.get('type') == 'mesh':
        mesh = meshes[geom.get('mesh')]
        if set(mesh.attrib) - {'name', 'vertex'}:
            return None
        vertices = np.fromstring(mesh.get('vertex'), sep=' ').reshape(-1, 3)
    else:
        return None
    if geom.get('quat'):
        quat = np.fromstring(geom.get('quat'), sep=' ')
        vertices = Rotation.from_quat(np.roll(quat, -1)).apply(vertices)
    vertices += np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
    return vertices


def rings(geom, vertices):
    points = np.rint(vertices * UNITS).astype(np.int64)
    levels = np.unique(points[:, 2])
    if len(levels) != 2:
        return None
    bottom, top = levels
    lower = np.unique(points[points[:, 2] == bottom, :2], axis=0)
    upper = np.unique(points[points[:, 2] == top, :2], axis=0)
    if not np.array_equal(lower, upper) or len(upper) < 3 or np.linalg.matrix_rank(upper - upper[0]) < 2:
        return None
    contact = tuple(sorted((key, value) for key, value in geom.attrib.items() if key not in SHAPE))
    return (int(bottom), int(top), contact), hull(upper)


def hull(points):
    return tuple(map(tuple, points[ConvexHull(points).vertices].tolist()))


def edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def twice_area(polygon):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in edges(polygon))


def boundary_error(shape, second, merged):
    # 顶点到凸包各边的最小距离给出凹口的最大深度；共享边两侧原本无重叠。
    points = np.asarray(shape + second, dtype=float)
    vertices = np.asarray(merged, dtype=float)
    vectors = np.roll(vertices, -1, axis=0) - vertices
    offsets = points[:, None] - vertices
    distances = (vectors[:, 0] * offsets[:, :, 1] - vectors[:, 1] * offsets[:, :, 0])
    return float(np.max(np.min(distances / np.linalg.norm(vectors, axis=1), axis=1)))


def combine(polygons):
    active = dict(polygons)
    by_edge = defaultdict(set)
    for index, polygon in active.items():
        for edge in edges(polygon):
            by_edge[edge].add(index)
    groups = {index: [index] for index in active}
    errors = defaultdict(float)
    for index in polygons:
        while index in active:
            shape = active[index]
            adjacent = sorted({other for a, b in edges(shape) for other in by_edge[b, a]} - {index})
            for other in adjacent:
                second = active[other]
                merged = hull(np.asarray(shape + second))
                # 共享反向边。仅对原始两块允许 50 微米以内的边缘误差，避免误差累计。
                error = max(errors[index], errors[other])
                if twice_area(merged) != twice_area(shape) + twice_area(second):
                    if len(groups[index]) != 1 or len(groups[other]) != 1:
                        continue
                    error = boundary_error(shape, second, merged)
                    if error > 50:
                        continue
                errors[index] = error
                for previous in (index, other):
                    for edge in edges(active[previous]):
                        by_edge[edge].remove(previous)
                del active[other]
                groups[index].extend(groups.pop(other))
                active[index] = merged
                for edge in edges(merged):
                    by_edge[edge].add(index)
                break
            else:
                break
    return [(active[i], members, errors[i]) for i, members in groups.items() if len(members) > 1]


def merge_layer(source, target, layer):
    geoms_path, assets_path = f'{layer}_geoms.xml', f'{layer}_assets.xml'
    tree = ET.parse(source / geoms_path)
    body = tree.getroot().find('body')
    assets = ET.parse(source / assets_path)
    asset = assets.getroot().find('asset')
    meshes = {node.get('name'): node for node in asset}
    parts = list(body)
    eligible = defaultdict(dict)
    keep_in_body = set()
    for i, geom in enumerate(parts):
        vertices = world_vertices(geom, meshes)
        if vertices is None or protected(vertices.min(axis=0), vertices.max(axis=0)):
            keep_in_body.add(i)
            continue
        candidate = rings(geom, vertices)
        if candidate is not None:
            key, polygon = candidate
            eligible[key][i] = polygon

    replacements = {}
    consumed = set()
    record = []
    for (bottom, top, contact), items in eligible.items():
        for shape, members, error in combine(items):
            # 以局部质心为原点生成同厚的凸棱柱，保持原接触参数。
            xy = np.asarray(shape) / UNITS
            center = xy.mean(axis=0)
            mid = (top + bottom) / (2 * UNITS)
            vertices = ' '.join(f'{x:.10f} {y:.10f} {z-mid:.10f}'
                                for z in (bottom / UNITS, top / UNITS) for x, y in xy - center)
            first = min(members)
            name = f'rmuc2027_flat_{layer}_{first:05}'
            ET.SubElement(asset, 'mesh', name=name, vertex=vertices)
            replacements[first] = ET.Element('geom', dict(contact), name=name, type='mesh', mesh=name,
                                             pos=f'{center[0]:.10f} {center[1]:.10f} {mid:.10f}')
            consumed.update(members)
            record.append(dict(name=name, edge_error_um=error, replaces=[parts[i].get('name') for i in sorted(members)]))
    # 保留登台路线原有碰撞组。把场外静态面直接挂到世界坐标，
    # 避免每毫秒把这些不动的网格纳入动态物体的碰撞包围盒计算。
    body[:] = [parts[i] for i in range(len(parts)) if i in keep_in_body]
    world_geoms = [replacements.get(i, geom) for i, geom in enumerate(parts)
                   if i not in keep_in_body and (i in replacements or i not in consumed)]
    tree.getroot().extend(world_geoms)
    used = {geom.get('mesh') for geom in (*body, *world_geoms)}
    asset[:] = [mesh for mesh in asset if mesh.get('name') in used]
    tree.write(target / geoms_path, encoding='UTF-8', xml_declaration=True)
    assets.write(target / assets_path, encoding='UTF-8', xml_declaration=True)
    return record


def add_perimeter(world):
    # 赛场原平面范围为 x ±14 m、y ±8 m，四段内侧各向场内搭接 2 cm。
    for name, xy, half_xy in (
        ('east', (14.04, 0.), (.06, 8.10)),
        ('west', (-14.04, 0.), (.06, 8.10)),
        ('north', (0., 8.04), (14.10, .06)),
        ('south', (0., -8.04), (14.10, .06)),
    ):
        ET.SubElement(world, 'geom', name=f'rmuc2027_fence_{name}', type='box',
                      attrib={'class':'collision', 'group':'2', 'rgba':'.36 .54 .66 .26'},
                      pos=f'{xy[0]} {xy[1]} .55', size=f'{half_xy[0]} {half_xy[1]} 1.15', mass='0')


def optimized_field(baseline):
    source = baseline.parent
    version = hashlib.sha256(source.name.encode() + Path(__file__).read_bytes()).hexdigest()[:16]
    destination = source.parent / f'merged-flat-{version}'
    if (destination / 'model.xml').is_file():
        return destination / 'model.xml'
    with TemporaryDirectory(dir=source.parent) as staging:
        target = Path(staging) / 'field'
        target.mkdir()
        names = {f'{layer}_{kind}.xml' for layer in LAYERS for kind in ('geoms', 'assets')}
        model = ET.parse(baseline)
        model.find('visual/global').set('bvactive', 'false')
        add_perimeter(model.find('worldbody'))
        model.write(target / 'model.xml', encoding='UTF-8', xml_declaration=True)
        for entry in source.iterdir():
            if entry.name not in names and entry.name != 'model.xml':
                (target / entry.name).symlink_to(os.path.relpath(entry, destination))
        merged = {layer: merge_layer(source, target, layer) for layer in LAYERS}
        (target / 'merged.json').write_text(json.dumps(merged, indent=2) + '\n')
        try:
            target.rename(destination)
        except OSError:
            if not (destination / 'model.xml').is_file():
                raise
    return destination / 'model.xml'
