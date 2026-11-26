"""Audit 777 CAD parameters and the two knee stop/plate mesh pairs.

Run with the mjlab interpreter, numpy, trimesh and python-fcl. Face counts
identify the inspected CAD components in this particular export. The sweep
checks local stop/plate contact only, not bearings, other links or dynamics.
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
    report = {'mass_kg': {}, 'invalid_inertias': [], 'invalid_limits': []}
    with ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        root = ET.fromstring(archive.read(next(
            name for name in archive.namelist() if name.endswith('.urdf'))))
        for link in root.findall('link'):
            inertial = link.find('inertial')
            mass = float(inertial.find('mass').get('value'))
            report['mass_kg'][link.get('name')] = mass
            values = {key: float(value) for key, value in
                      inertial.find('inertia').attrib.items()}
            matrix = np.array([
                [values['ixx'], values['ixy'], values['ixz']],
                [values['ixy'], values['iyy'], values['iyz']],
                [values['ixz'], values['iyz'], values['izz']]])
            eigenvalues = np.linalg.eigvalsh(matrix)
            if mass <= 0 or eigenvalues[0] <= 0 or sum(eigenvalues[:2]) < eigenvalues[2]:
                report['invalid_inertias'].append(link.get('name'))
            if link.get('name') == 'imu-link':
                report['imu_inertia_kg_m2'] = values
        for joint in root.findall('joint'):
            limit = joint.find('limit')
            if limit is not None and float(limit.get('lower')) > float(limit.get('upper')):
                report['invalid_limits'].append(dict(
                    joint=joint.get('name'), **limit.attrib))
        report['total_mass_kg'] = sum(report['mass_kg'].values())
        report['extensions'] = [node.tag for node in root
                                if node.tag not in ('link', 'joint')]
        report['knees'] = {}
        for name, parent_faces, child_faces in (
                ('lefr-front-01', 236, 1636),
                ('right-front-01', 404, 1634)):
            joint = root.find(f"joint[@name='{name}']")
            parts = []
            for relation, face_count in (('parent', parent_faces), ('child', child_faces)):
                link_name = joint.find(relation).get('link')
                link = root.find(f"link[@name='{link_name}']")
                visual = link.find('visual')
                assert visual.find('origin').get('xyz') == '0 0 0'
                assert visual.find('origin').get('rpy') == '0 0 0'
                mesh_node = visual.find('geometry/mesh')
                assert mesh_node.get('scale', '1 1 1') == '1 1 1'
                filename = Path(mesh_node.get('filename')).name
                mesh_bytes = archive.read(next(
                    item for item in archive.namelist() if item.endswith('/' + filename)))
                mesh = trimesh.load(BytesIO(mesh_bytes), file_type='stl')
                candidates = [part for part in mesh.split(only_watertight=False)
                              if len(part.faces) == face_count]
                assert len(candidates) == 1, 'Inspected CAD component changed'
                assert candidates[0].is_watertight
                parts.append(candidates[0])
            origin = joint.find('origin')
            transform = trimesh.transformations.euler_matrix(
                *map(float, origin.get('rpy').split()))
            transform[:3, 3] = list(map(float, origin.get('xyz').split()))
            axis = list(map(float, joint.find('axis').get('xyz').split()))
            manager = trimesh.collision.CollisionManager()
            manager.add_object('thigh_stop', parts[0])
            lower = float(joint.find('limit').get('lower'))
            upper = float(joint.find('limit').get('upper'))
            angles = np.linspace(lower, upper, 1001)
            hits = []
            for angle in angles:
                pose = transform @ trimesh.transformations.rotation_matrix(angle, axis)
                hits.append(manager.in_collision_single(parts[1], pose))
            endpoint_distances = []
            for angle in (lower, upper):
                pose = transform @ trimesh.transformations.rotation_matrix(angle, axis)
                endpoint_distances.append(manager.min_distance_single(parts[1], pose))
            transitions = []
            for index in np.flatnonzero(np.diff(hits)):
                left, right = angles[index:index + 2]
                left_hit = hits[index]
                for _ in range(30):
                    middle = (left + right) / 2
                    pose = transform @ trimesh.transformations.rotation_matrix(middle, axis)
                    if manager.in_collision_single(parts[1], pose) == left_hit:
                        left = middle
                    else:
                        right = middle
                transitions.append(float((left + right) / 2))
            report['knees'][name] = dict(
                scope='thigh stop versus calf plate only',
                limits_rad=[lower, upper], sample_count=len(angles),
                collision_count=sum(hits),
                colliding_angles_rad=[float(q) for q, hit in zip(angles, hits) if hit],
                endpoint_clearance_m=endpoint_distances,
                contact_transition_angles_rad=transitions)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
