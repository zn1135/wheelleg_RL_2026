#!/usr/bin/env python3
"""Compare late-episode leg motion and parking from recorded teacher/student traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(time, joint_q, joint_dq, root):
    # 起立和调整位置不混入末段停车统计；两条轨迹均取15～20秒。
    mask = (time > 15.) & (time <= 20.)
    assert mask.sum() == 500 and np.allclose(np.diff(time), .01)
    q = joint_q[mask][:, [0, 1, 3, 4]]
    dq = joint_dq[mask][:, [0, 1, 3, 4]]
    xy = root[mask, :2]
    assert np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(root).all()
    # 加窗抑制截断边缘影响，只描述100Hz存档能分辨的振动频率。
    spectrum = np.abs(np.fft.rfft((q-q.mean(0))*np.hanning(len(q))[:, None], axis=0))
    frequencies = np.fft.rfftfreq(len(q), .01)
    dominant = frequencies[spectrum[1:].argmax(0)+1]
    speed = np.linalg.norm(root[mask, 7:9], axis=1)
    return dict(window_s=[15., 20.], samples=int(mask.sum()),
        leg_velocity_rms_rad_s=float(np.sqrt(np.square(dq).mean())),
        leg_velocity_rms_per_joint_rad_s=np.sqrt(np.square(dq).mean(0)).tolist(),
        dominant_leg_frequency_hz=dominant.tolist(),
        mean_horizontal_speed_m_s=float(speed.mean()),
        final_1s_mean_horizontal_speed_m_s=float(np.linalg.norm(root[time > 19., 7:9], axis=1).mean()),
        horizontal_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),
        max_horizontal_excursion_m=float(np.linalg.norm(xy-xy[0], axis=1).max()),
        horizontal_path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dataset', type=Path, required=True)
    parser.add_argument('--policy-source', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.teacher_dataset.read_text())
    episode = next(row for row in dataset['episodes'] if row['seed'] == args.seed and row['env'] == args.env)
    teacher_path = args.teacher_dataset.parent/episode['path']
    source = json.loads((args.policy_source/'source-check.json').read_text())
    policy_path = args.policy_source/f'policy-{args.seed}.npz'
    assert source['completed'] and digest(teacher_path) == episode['sha256']
    assert digest(policy_path) == source['seeds'][str(args.seed)]['source_trace_sha256']
    with np.load(teacher_path) as teacher, np.load(policy_path) as policy:
        metadata = json.loads(str(teacher['metadata']))
        assert metadata['xml_sha256'] == source['xml_sha256']
        teacher_result = measure(teacher['time'], teacher['motor_q'], teacher['motor_dq'], teacher['root'])
        policy_result = measure(policy['time'], policy['motor_q'][:, args.env],
                                policy['motor_dq'][:, args.env], policy['root'][:, args.env])
    result = dict(teacher=teacher_result, policy=policy_result,
        teacher_source=str(teacher_path), teacher_source_sha256=digest(teacher_path),
        policy_source=str(policy_path), policy_source_sha256=digest(policy_path),
        policy_checkpoint_sha256=source['checkpoint_sha256'], xml_sha256=source['xml_sha256'],
        seed=args.seed, env=args.env, source_sha256=digest(Path(__file__)), optimizer_updates=0,
        scope='Two saved zero-command episodes under the same XML; different initial poses and control interfaces. Descriptive comparison, not paired causal attribution, full-policy acceptance or a new frequency threshold. Recorded at 100Hz; no claims above 50Hz.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
