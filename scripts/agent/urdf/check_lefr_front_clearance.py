"""Static mesh check for the supplied CAD's thigh stop and calf plate.

Uses trimesh and python-fcl. This checks one identified contact pair, including
both joint endpoints and sampled interior poses. It is not a dynamics or
whole-robot collision acceptance check. Component face counts identify the
two inspected CAD parts, so this checker applies to this mesh revision only.
"""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        root = ET.fromstring(archive.read(next(
            name for name in archive.namelist() if name.endswith('.urdf'))))
        joint = root.find("joint[@name='lefr-front-01']")
        meshes = []
        for relation, face_count in (('parent', 236), ('child', 1636)):
            link_name = joint.find(relation).get('link')
            link = root.find(f"link[@name='{link_name}']")
            geometry = link.find('visual')
            assert geometry.find('origin').get('xyz') == '0 0 0'
            assert geometry.find('origin').get('rpy') == '0 0 0'
            mesh_node = geometry.find('geometry/mesh')
            assert mesh_node.get('scale', '1 1 1') == '1 1 1'
            filename = Path(mesh_node.get('filename')).name
            mesh_bytes = archive.read(next(
                name for name in archive.namelist()
                if name.endswith('/' + filename)))
            mesh = trimesh.load(BytesIO(mesh_bytes), file_type='stl')
            candidates = [part for part in mesh.split(only_watertight=False)
                          if len(part.faces) == face_count]
            assert len(candidates) == 1, 'CAD component identification changed'
            assert candidates[0].is_watertight
            meshes.append(candidates[0])

    origin = joint.find('origin')
    transform = trimesh.transformations.euler_matrix(
        *map(float, origin.get('rpy').split()))
    transform[:3, 3] = list(map(float, origin.get('xyz').split()))
    axis = list(map(float, joint.find('axis').get('xyz').split()))
    manager = trimesh.collision.CollisionManager()
    manager.add_object('thigh_stop', meshes[0])

    def pose(angle):
        return transform @ trimesh.transformations.rotation_matrix(angle, axis)

    def collides(angle):
        return manager.in_collision_single(meshes[1], pose(angle))

    def boundary(clear_angle, colliding_angle):
        assert not collides(clear_angle)
        assert collides(colliding_angle)
        for _ in range(35):
            middle = (clear_angle + colliding_angle) / 2
            if collides(middle):
                colliding_angle = middle
            else:
                clear_angle = middle
        return clear_angle

    lower_boundary = boundary(-0.2, -0.6)
    upper_boundary = boundary(-0.2, 0.2)
    limit = joint.find('limit')
    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
    angles = np.linspace(lower, upper, 1001)
    collisions = [float(angle) for angle in angles if collides(angle)]
    clearances = [manager.min_distance_single(meshes[1], pose(angle))
                  for angle in (lower, upper)]
    print(json.dumps(dict(
        scope='lefr-front-00 stop versus lefr-front-01 plate only',
        measured_boundaries_rad=[lower_boundary, upper_boundary],
        configured_limits_rad=[lower, upper],
        endpoint_clearance_m=clearances,
        sample_count=len(angles), collision_count=len(collisions),
        limits_within_boundaries=(lower >= lower_boundary and
                                  upper <= upper_boundary),
    ), indent=2))


if __name__ == '__main__':
    main()
