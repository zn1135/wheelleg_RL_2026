"""Compare a read-only reference USD extraction to the current MJCF; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    q = np.asarray(q, dtype=float); q = q/np.linalg.norm(q); w = q[0]; v = q[1:]
    skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return (w*w-v@v)*np.eye(3)+2*np.outer(v, v)+2*w*skew


def rod_coordinates(direction, point):
    direction = np.asarray(direction)[[0, 2]]; direction = direction/np.linalg.norm(direction)
    point = np.asarray(point)[[0, 2]]
    return [float(direction@point), float(direction[0]*point[1]-direction[1]*point[0])]


def reference_spring(joints, side):
    hinge = joints[side+'_spring1_joint']; slide = joints[side+'_spring2_joint']; loop = joints[side+'_s2r2_joint']
    assert loop['body0'].endswith(side+'_spring2_link'), 'Unexpected reference spring frames'
    a = slide['attributes']; axis = np.eye(3)[dict(X=0, Y=1, Z=2)[a['physics:axis']]]
    r0 = rotation(a['physics:localRot0']); r1 = rotation(a['physics:localRot1'])
    axis = r0@axis
    tip = np.asarray(a['physics:localPos0'])+r0@r1.T@(np.asarray(loop['attributes']['physics:localPos0'])-a['physics:localPos1'])
    tip -= np.asarray(hinge['attributes']['physics:localPos1'])
    zero_axial = float(tip@axis); lateral = float(np.linalg.norm(tip-zero_axial*axis))
    return dict(slide_axis_body0=axis.tolist(), zero_axial_mount_distance_m=zero_axial,
                fixed_lateral_eye_offset_m=lateral, zero_compression_axial_mount_distance_m=zero_axial+.06076,
                stroke_limits_joint_m=[a['physics:lowerLimit'], a['physics:upperLimit']],
                axial_mount_distance_limits_m=[zero_axial+a['physics:lowerLimit'], zero_axial+a['physics:upperLimit']],
                hinge_limits_deg=[hinge['attributes']['physics:lowerLimit'], hinge['attributes']['physics:upperLimit']])


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--usd-extract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); reference = json.loads(args.usd_extract.read_text())
    source = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'); tree = ET.parse(source).getroot()
    reference_joints = {j['name']: j for j in reference['joints']}
    reference_bodies = {b['name']: b for b in reference['bodies']}
    mapping = {'lf0':'left_rear1_link', 'lf1':'left_rear2_link', 'lf00':'left_front1_link',
               'lf01':'left_front2_link', 'lf02':'left_front3_link', 'lf03':'left_front4_link',
               'lfwheel':'left_wheel_link', 'rf0':'right_rear1_link', 'rf1':'right_rear2_link',
               'rf00':'right_front1_link', 'rf01':'right_front2_link', 'rf02':'right_front3_link',
               'rf03':'right_front4_link', 'rfwheel':'right_wheel_link'}
    masses = []
    for own, other in mapping.items():
        node = tree.find(f".//body[@name='{own}']/inertial"); item = reference_bodies[other]
        ours = float(node.get('mass')); theirs = float(item['attributes']['physics:mass'])
        masses.append(dict(current_body=own, reference_body=other, current_mass_kg=ours,
                           reference_mass_kg=theirs, current_to_reference_mass_ratio=ours/theirs))
    loops = []
    for prefix, side in [('lf', 'left'), ('rf', 'right')]:
        for own, other in [(prefix+'_loop1', side+'_f3r1_joint'), (prefix+'_loop2', side+'_f4r2_joint')]:
            node = tree.find(f"equality/connect[@name='{own}']"); joint = reference_joints[other]
            normalized = {mapping[node.get('body1')], mapping[node.get('body2')]}
            actual = {joint['body0'].split('/')[-1], joint['body1'].split('/')[-1]}
            assert normalized == actual
            loops.append(dict(current=own, reference=other, normalized_bodies=sorted(actual),
                              current_type='MuJoCo connect: three translational constraints', reference_type=joint['type'],
                              reference_excluded_from_articulation=joint['attributes'].get('physics:excludeFromArticulation')))
        joint = reference_joints[side+'_s2r2_joint']; assert joint['body1'].endswith(side+'_rear2_link')
        loops.append(dict(current=side+'_spring_loop', reference=side+'_s2r2_joint', normalized_bodies=[side+'_spring2_link', side+'_rear2_link'],
                          current_type='MuJoCo connect', reference_type=joint['type'], same_connectivity=True))
    springs = {}
    for side in ('left', 'right'):
        barrel = tree.find(f".//body[@name='{side}_spring1']"); rod = barrel.find(f"body[@name='{side}_spring2']")
        position = np.fromstring(rod.get('pos'), sep=' '); axial = float(position[2]); lateral = float(np.linalg.norm(position[:2]))
        ref = reference_spring(reference_joints, side)
        forces = []
        for length in (.20, .21, .22, .225):
            own_compression = max(0., axial+.010-length)
            ref_compression = max(0., ref['zero_compression_axial_mount_distance_m']-length)
            forces.append(dict(axial_mount_distance_m=length, current_elastic_force_n=float(np.clip(150./.09*own_compression, 0., 150.)),
                               reference_elastic_force_n=float(400.+(600.-400.)/.07*ref_compression)))
        springs[side] = dict(current=dict(zero_axial_mount_distance_m=axial, fixed_lateral_eye_offset_m=lateral,
                            zero_compression_axial_mount_distance_m=axial+.010, slide_axis=rod.find('joint').get('axis'),
                            slide_limited=rod.find('joint').get('limited'), hinge_limited=barrel.find('joint').get('limited')),
                            reference=ref, static_force_at_same_axial_distance=forces)
    mounts = []
    for own, side in [('lf', 'left'), ('rf', 'right')]:
        for kind, current_direction, current_site, other_joint, other_point in [
            ('upper', own+'1', side+'_gas_spring_upper', side+'_rear2_joint', side+'_spring1_joint'),
            ('lower', own+'wheel', side+'_gas_spring_lower', side+'_wheel_joint', side+'_s2r2_joint')]:
            current = rod_coordinates(np.fromstring(tree.find(f".//body[@name='{current_direction}']").get('pos'), sep=' '),
                                      np.fromstring(tree.find(f".//site[@name='{current_site}']").get('pos'), sep=' '))
            other = rod_coordinates(reference_joints[other_joint]['attributes']['physics:localPos0'],
                                    reference_joints[other_point]['attributes']['physics:localPos0' if kind == 'upper' else 'physics:localPos1'])
            mounts.append(dict(side=side, mount=kind, coordinates='along rod / perpendicular in XZ plane, meters',
                               current=current, reference=other,
                               invariant_point_difference_m=float(np.linalg.norm(np.asarray(current)-other))))
    guides = [b for b in reference['bodies'] if '_guide_' in b['name']]
    result = dict(scope='Read-only source structure and parameter comparison; no physical or dynamic equivalence claim',
                  environment_steps=0, current_source=str(source.resolve()), current_source_sha256=digest(source),
                  usd_extraction=str(args.usd_extract), usd_extraction_sha256=digest(args.usd_extract),
                  reference_source_files=reference['source_files'], body_mapping=mapping, mass_comparison=masses,
                  closed_loop_comparison=loops, spring_comparison=springs, mounting_geometry_in_rod_coordinates=mounts,
                  extra_reference_guide_bodies=len(guides), extra_reference_guide_mass_kg=sum(float(b['attributes']['physics:mass']) for b in guides),
                  shared_motor_armature_kg_m2=.015795, current_source_modified=False,
                  limitations=['Matched topology does not imply identical dimensions, masses, joint limits or dynamics',
                               'Reference front/rear naming is opposite to the corresponding long/short current-model branches',
                               'USD prismatic X and MJCF slide Z are local axes; compare physical extension direction and mount distance',
                               'Different zero poses prevent interpreting 60.76mm versus10mm as a direct physical preload ratio',
                               'The reference400N baseline is not the user hardware manufacturer150N rating'])
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(loop_connections_matched=len(loops), mass_comparison=masses[:7], spring_comparison=springs), indent=2))


if __name__ == '__main__': main()
