"""Propose knee ranges covering measured closed poses; never write the source XML."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim.report_h7_joint_endpoint import read_jsonl
from scripts.agent.real2sim import view_h7_joint_limits as viewer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--endpoint-reports', type=Path, nargs=8, required=True,
                        help='lf short/long/short/long, then rf short/long/short/long')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-raw', type=Path, nargs='*', default=[], help='Previously verified same-mapping eight-channel USB records')
    args = parser.parse_args()
    metadata = json.loads((args.session / 'metadata.json').read_text())
    source = Path(metadata['sources']['model.xml']['path'])
    source_bytes = (args.session / 'model.xml').read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == metadata['sources']['model.xml']['sha256']
    reports = [json.loads((p / 'report.json').read_text()) for p in args.endpoint_reports]
    assert [r['mark_index'] for r in reports] == list(range(1, 9)), 'Unexpected endpoint order'
    assert all(r['source_hashes']['model.xml'] == metadata['sources']['model.xml'] for r in reports)
    confirmed = json.loads((args.endpoint_reports[0] / 'side-confirmation.json').read_text())
    assert confirmed['model_side'] == 'lf' and confirmed['mark_index'] == 1
    for i, report in enumerate(reports):
        assert report['sample_count'] >= 90 and max(report['firmware_position_span_rad']) <= .003
        assert max(report['closure_m']) <= .001
        side = 'lf' if i < 4 else 'rf'
        if i:
            assert report['physical_side_assignment'] == side

    last_index = reports[-1]['mark_index']
    final_mark = read_jsonl(args.session / 'marks.jsonl')[last_index - 1]
    poses = [p for p in read_jsonl(args.session / 'poses.jsonl')
             if p['sample']['index'] <= final_mark['sample']['index']]
    assert poses
    original = source_bytes.decode()
    tree = ET.fromstring(original)
    current_tree = ET.fromstring(source.read_text())
    for name in ('lf1', 'rf1'):
        current_tree.find(f'.//joint[@name="{name}"]').set('range', tree.find(f'.//joint[@name="{name}"]').get('range'))
    assert ET.tostring(current_tree) == ET.tostring(tree), 'Current model geometry differs from recorded geometry'
    assert hashlib.sha256(viewer.CONFIG.read_bytes()).hexdigest() == metadata['sources']['machine_config.c']['sha256']
    references = []
    reference_angles = {'lf1': [], 'rf1': []}
    for path in args.reference_raw:
        raw = path.read_bytes()
        frames = viewer.JointFrames().feed(raw)
        assert frames, 'No eight-channel reference frames'
        pose = viewer.mirror.MirrorPose(machine_config=viewer.CONFIG)
        pose.data.qpos[:7] = (0., 0., .8, 1., 0., 0., 0.)
        previous = None
        unique = 0
        for frame in frames:
            if previous is not None and np.array_equal(previous, frame[::2]):
                continue
            joints = viewer.solve(pose, frame)
            assert joints is not None, 'Reference cannot close'
            for name in reference_angles:
                reference_angles[name].append(joints[name])
            previous = frame[::2].copy()
            unique += 1
        references.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), frames=len(frames), unique_poses=unique))
    changed = {}
    repeats = {}
    for side, offset in (('lf', 0), ('rf', 4)):
        joint_name = side + '1'
        recorded = np.array([p['joints_rad'][joint_name] for p in poses])
        assert np.max(np.abs(np.diff(recorded))) < math.pi, 'Possible assembly-branch discontinuity'
        values = np.concatenate((recorded, reference_angles[joint_name]))
        old = [float(x) for x in tree.find(f'.//joint[@name="{joint_name}"]').get('range').split()]
        # 只向外取整0.01rad以覆盖已到达姿态，不将它解释成额外机械安全余量。
        proposed = [math.floor(float(values.min()) * 100) / 100,
                    math.ceil(float(values.max()) * 100) / 100]
        changed[joint_name] = dict(old_rad=old, proposed_rad=proposed,
                                  observed_rad=[float(values.min()), float(values.max())],
                                  old_outside_count=int(np.count_nonzero((values < old[0]) | (values > old[1]))),
                                  proposed_outside_count=int(np.count_nonzero((values < proposed[0]) | (values > proposed[1]))))
        repeats[side] = {}
        for label, a, b in (('short', offset, offset + 2), ('long', offset + 1, offset + 3)):
            qa, qb = reports[a]['model_joints_rad'][joint_name], reports[b]['model_joints_rad'][joint_name]
            la, lb = reports[a]['model_hip_wheel_xz_length_m'][side], reports[b]['model_hip_wheel_xz_length_m'][side]
            repeats[side][label] = dict(knee_rad=[qa, qb], length_m=[la, lb],
                                       difference_deg=abs(math.degrees(qb - qa)), difference_mm=abs(lb - la) * 1000)
    candidate = original
    for name, item in changed.items():
        pattern = r'(<joint\b[^>]*\bname="' + name + r'"[^>]*\brange=")[^"]*(")'
        replacement = f"{item['proposed_rad'][0]:.2f} {item['proposed_rad'][1]:.2f}"
        candidate, count = re.subn(pattern, lambda m: m[1] + replacement + m[2], candidate)
        assert count == 1, name
    candidate = candidate.replace('采用源 Wheel-Legged-Gym/sim2sim/chuanliantui.xml 的 +Y 实体 rf1 限位（2026-09-24 修正）。',
                                  '2026-11-04：膝折叠限位按USB实测覆盖区间向外取整0.01rad；轴向/零位保持。')
    candidate_tree = ET.fromstring(candidate)
    for name, item in changed.items():
        candidate_tree.find(f'.//joint[@name="{name}"]').set('range', tree.find(f'.//joint[@name="{name}"]').get('range'))
    assert ET.tostring(candidate_tree) == ET.tostring(tree), 'Unexpected model change'
    # 编译时仅临时展开网格目录；交付候选保持源文件的相对路径。
    meshdir = str((source.parent / '../meshes').resolve())
    compiled_text = candidate.replace('meshdir="../meshes"', f'meshdir="{meshdir}"')
    model = mujoco.MjModel.from_xml_string(compiled_text)
    assert all(np.allclose(model.joint(name).range, item['proposed_rad']) for name,item in changed.items())
    assert max(max(p['closure_m']) for p in poses) <= .001
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'original.xml').write_bytes(source_bytes)
    (args.output / 'candidate.xml').write_text(candidate)
    (args.output / 'poses-through-endpoints.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in poses))
    report = dict(kind='Observed mechanical working-range proposal', changed_joints=changed,
                  endpoint_repeatability=repeats, closed_pose_count=len(poses),
                  maximum_closure_m=max(max(p['closure_m']) for p in poses),
                  endpoint_reports=[str(p) for p in args.endpoint_reports],
                  additional_usb_references=references,
                  source_xml=str(source), source_xml_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  candidate_xml_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                  rounding_rad=.01, source_xml_modified=False, physics_steps=0,
                  caveats=['Manual endpoints have measured repeatability error; not precision hard-stop metrology',
                           'All observed poses are included; two marked extension endpoints alone can miss the longest pose',
                           'Only knee-folding ranges are proposed; independent active-motor protection ranges are not identified',
                           'Eight-channel USB lacks source age, CAN sequence and CRC',
                           'No GPU dynamic replay or powered validation performed'])
    (args.output / 'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
