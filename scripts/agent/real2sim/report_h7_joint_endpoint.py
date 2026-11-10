"""Summarize a marked USB endpoint without assigning physical limits or changing mapping."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from scripts.agent.real2sim import view_h7_joint_limits as viewer


def read_jsonl(path):
    # 运行中的最后一行可能尚未写完，仅忽略不完整尾行。
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--mark-index', type=int, required=True, help='One-based endpoint index')
    parser.add_argument('--model-side', choices=('lf', 'rf'), help='Only after the user confirms physical side')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.session / 'metadata.json').read_text())
    for name in ('machine_config.c',):
        item = metadata['sources'][name]
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], name
    item = metadata['sources']['model.xml']
    recorded_model = (args.session / 'model.xml').read_bytes()
    assert hashlib.sha256(recorded_model).hexdigest() == item['sha256']
    current_model = Path(item['path']).read_bytes()
    recorded_tree, current_tree = ET.fromstring(recorded_model), ET.fromstring(current_model)
    # 事后改限位不改变运动学；其他几何变化必须拒绝复算旧会话。
    for name in ('lf1', 'rf1'):
        current_tree.find(f'.//joint[@name="{name}"]').set('range', recorded_tree.find(f'.//joint[@name="{name}"]').get('range'))
    assert ET.tostring(recorded_tree) == ET.tostring(current_tree), 'Recorded/current geometry differs'
    marks = read_jsonl(args.session / 'marks.jsonl')
    assert 1 <= args.mark_index <= len(marks), 'Endpoint index unavailable'
    mark = marks[args.mark_index - 1]
    end_index = mark['sample']['index']
    end_time = mark['sample']['host_elapsed_s']
    samples = [row for row in read_jsonl(args.session / 'samples.jsonl')
               if end_time - 1.1 <= row['host_elapsed_s'] <= end_time and row['index'] <= end_index]
    assert len(samples) >= 90, 'Insufficient holding samples'
    assert all(b['index'] == a['index'] + 1 for a, b in zip(samples, samples[1:])), 'Host decode gap'
    assert samples[-1]['index'] == end_index, 'Marked sample not yet flushed; retry after next write'
    angles = np.array([row['firmware_positions_rad'] for row in samples])
    torques = np.array([row['feedback_torques_nm'] for row in samples])
    span = np.ptp(angles, axis=0)
    assert np.max(span) <= .003, 'Holding window is moving'
    median = np.median(angles, axis=0)
    pose = viewer.mirror.MirrorPose(machine_config=viewer.CONFIG)
    pose.data.qpos[:7] = (0., 0., .8, 1., 0., 0., 0.)
    values = np.empty(8)
    values[::2], values[1::2] = median, np.median(torques, axis=0)
    # 沿已记录的装配分支求解，避免把另一闭链分支混入端点。
    for name, value in mark['joints_rad'].items():
        pose.data.qpos[pose.model.joint(name).qposadr[0]] = value
    joints = viewer.solve(pose, values)
    assert joints is not None, 'Median endpoint cannot close'
    lengths = {}
    for side in ('lf', 'rf'):
        delta = pose.data.body(side + 'wheel').xpos - pose.data.body(side + '0').xpos
        lengths[side] = float(np.linalg.norm(delta[[0, 2]]))
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(user_endpoint_label=mark['label'], mark_index=args.mark_index,
                  sample_count=len(samples), host_window_s=[samples[0]['host_elapsed_s'], samples[-1]['host_elapsed_s']],
                  firmware_position_median_rad=median.tolist(), firmware_position_span_rad=span.tolist(),
                  model_joints_rad=joints, model_joints_deg={n:float(np.rad2deg(q)) for n,q in joints.items()},
                  feedback_torque_median_nm=np.median(torques,axis=0).tolist(),
                  feedback_torque_max_abs_nm=np.max(np.abs(torques),axis=0).tolist(),
                  model_hip_wheel_xz_length_m=lengths, closure_m=pose.residual_m,
                  physical_side_assignment=args.model_side or 'Pending user clarification; do not infer from endpoint label',
                  source_model_modified=False, calibration_complete=False, physics_steps=0,
                  analysis_model_sha256=hashlib.sha256(current_model).hexdigest(),
                  analysis_model_differs_only_in_knee_limits=current_model != recorded_model,
                  limitations=['Host sample continuity is not CAN sequence validation',
                               'USB carries no motor source age or online bits',
                               'Feedback torque is a driver estimate',
                               'One endpoint is insufficient for joint-range calibration'],
                  source_hashes=metadata['sources'])
    (args.output / 'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (args.output / 'mark.json').write_text(json.dumps(mark,indent=2,ensure_ascii=False)+'\n')
    with (args.output / 'samples.jsonl').open('w') as file:
        for row in samples:
            file.write(json.dumps(row)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
