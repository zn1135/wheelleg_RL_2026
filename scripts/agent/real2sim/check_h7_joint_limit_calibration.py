"""Check applied knee ranges, all saved poses and endpoint reconstruction; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import view_h7_joint_limits as viewer
from scripts.agent.real2sim.report_h7_joint_endpoint import read_jsonl


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--firmware-source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = json.loads((args.calibration / 'report.json').read_text())
    source = Path(report['source_xml'])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == report['candidate_xml_sha256']
    pose = viewer.mirror.MirrorPose(machine_config=viewer.CONFIG)
    pose.data.qpos[:7] = (0., 0., .8, 1., 0., 0., 0.)

    def outside(joints):
        return [name for name, q in joints.items() if pose.model.joint(name).limited[0]
                and not pose.model.joint(name).range[0] <= q <= pose.model.joint(name).range[1]]

    endpoints = []
    for directory in report['endpoint_reports']:
        endpoint = json.loads((Path(directory) / 'report.json').read_text())
        for name, q in endpoint['model_joints_rad'].items():
            pose.data.qpos[pose.model.joint(name).qposadr[0]] = q
        values = np.empty(8)
        values[::2] = endpoint['firmware_position_median_rad']
        values[1::2] = endpoint['feedback_torque_median_nm']
        joints = viewer.solve(pose, values)
        assert joints is not None and not outside(joints)
        difference = max(abs(joints[n] - q) for n,q in endpoint['model_joints_rad'].items())
        assert difference < 1e-8
        endpoints.append(dict(mark=endpoint['mark_index'], max_pose_difference_rad=difference,
                              closure_m=list(pose.residual_m), outside=[]))
    poses = read_jsonl(args.calibration / 'poses-through-endpoints.jsonl')
    assert len(poses) == report['closed_pose_count'] and all(not outside(p['joints_rad']) for p in poses)
    references = []
    for ref in report['additional_usb_references']:
        raw = Path(ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        frames = viewer.JointFrames().feed(raw)
        previous = None
        for values in frames:
            if previous is not None and np.array_equal(previous, values[::2]):
                continue
            joints = viewer.solve(pose, values)
            assert joints is not None and not outside(joints)
            previous = values[::2].copy()
        references.append(dict(path=ref['path'], frames=len(frames), outside=[]))
    expected = json.loads(args.firmware_source_manifest.read_text())
    h7 = viewer.ROOT / 'deployment/H7_RL_ros'
    actual = {str(p.relative_to(h7)):hashlib.sha256(p.read_bytes()).hexdigest() for p in h7.rglob('*')
              if p.is_file() and p.suffix.lower() in {'.c', '.h', '.ioc', '.s', '.ld'}}
    assert expected == actual
    result = dict(ok=True, source_xml_sha256=source_hash, checked_endpoints=endpoints,
                  recorded_pose_count=len(poses), outside_model_limits=0, references=references,
                  unchanged_firmware_source_files=len(actual), physics_steps=0, gpu_dynamics_tested=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
