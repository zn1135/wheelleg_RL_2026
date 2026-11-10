"""Offline CoM geometry from saved GPU parking poses; no physics or policy calls."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; sources = {}; series = {}
    for height in ('h022', 'h026', 'h030'):
        reference_model = None
        for branch in ('8000_zero', '8000_plus1', '8000_plus2'):
            folder = args.recording / height / branch
            model_path = folder / 'fixtures/model.mjb'
            sources[str(model_path)] = hashlib.sha256(model_path.read_bytes()).hexdigest()
            model = mujoco.MjModel.from_binary_path(str(model_path))
            if reference_model is None:
                reference_model = model
            for field in ('body_mass', 'body_ipos', 'body_inertia'):
                assert np.array_equal(getattr(model, field), getattr(reference_model, field))
            data = mujoco.MjData(model)
            base = model.body('robot/base_link').id
            wheels = [model.body('robot/' + n).id for n in ('lfwheel', 'rfwheel')]
            masses = model.body_mass
            for sid in ('diagnostic_0', 'diagnostic_13', 'diagnostic_14'):
                path = folder / 'traces' / (sid + '.npz')
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as trace:
                    # 与旧报告一致：10～15秒，左闭右开；姿态100Hz、速度500Hz。
                    poses = trace['qpos'][(trace['qpos_time'] >= 10.) & (trace['qpos_time'] < 15.)]
                    mask = (trace['time'] >= 10.) & (trace['time'] < 15.)
                    pitch = trace['roll_pitch_deg'][mask, 1]
                    speed = trace['linear_body'][mask, 0]
                    encoder = trace['encoder_m_s'][(trace['policy_time'] >= 10.) & (trace['policy_time'] < 15.), 0]
                geometry = []
                for qpos in poses:
                    data.qpos[:] = qpos
                    # 只算已保存姿态的运动学和质量加权位置，不积分运动方程。
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                    com = np.sum(data.xipos * masses[:, None], axis=0) / masses.sum()
                    np.testing.assert_allclose(com, data.subtree_com[base], atol=1e-12)
                    axis = data.xpos[wheels].mean(axis=0)
                    rotation = data.xmat[base].reshape(3, 3)
                    forward = rotation[:, 0].copy(); forward[2] = 0.
                    forward /= np.linalg.norm(forward)
                    delta = com - axis
                    body_delta = data.xipos[base] - axis
                    origin_delta = data.xpos[base] - axis
                    geometry.append([delta @ forward, delta[2], body_delta @ forward,
                                     origin_delta @ forward, *((rotation.T @ delta)[[0, 2]])])
                geometry = np.asarray(geometry)
                series[(height, branch, sid)] = geometry
                mean = geometry.mean(axis=0)
                rows.append(dict(height=height, branch=branch, sample=sid, samples=len(poses),
                    pitch_deg=float(pitch.mean()), vx_m_s=float(speed.mean()),
                    encoder_vx_m_s=float(encoder.mean()), mass_kg=float(masses.sum()),
                    com_axle_forward_mm=float(mean[0]*1000),
                    com_axle_forward_std_mm=float(geometry[:, 0].std()*1000),
                    com_above_axle_mm=float(mean[1]*1000),
                    base_com_axle_forward_mm=float(mean[2]*1000),
                    base_origin_axle_forward_mm=float(mean[3]*1000),
                    com_axle_body_x_mm=float(mean[4]*1000),
                    com_axle_body_z_mm=float(mean[5]*1000)))
    for row in rows:
        old = next(r for r in rows if r['height'] == row['height'] and r['sample'] == row['sample'] and r['branch'] == '8000_zero')
        delta = np.deg2rad(row['pitch_deg'] - old['pitch_deg'])
        # 冻结原腿形后，绕轮轴增加相同pitch的几何估算；不是新的可实现动态轨迹。
        predicted = old['com_axle_forward_mm'] * np.cos(delta) + old['com_above_axle_mm'] * np.sin(delta)
        row['frozen_pose_rotation_shift_mm'] = float(predicted - old['com_axle_forward_mm'])
        row['observed_com_shift_mm'] = row['com_axle_forward_mm'] - old['com_axle_forward_mm']
        row['pose_adjustment_residual_mm'] = row['com_axle_forward_mm'] - predicted
    result = dict(rows=rows, input_sha256=sources, window_s=[10., 15.],
        definition='Mass-weighted whole robot CoM minus midpoint of wheel body origins, projected on horizontal body-forward axis; positive is forward',
        scope='Saved zero-extra-delay GPU trajectories, 8000 model, three heights x three parking cases x 0/1/2deg; offline position kinematics only',
        limitation='A correlation and frozen-pose geometry comparison, not independent mass/COM or policy/encoder causal intervention; axle proxy is not contact CoP',
        physics_steps=0, policy_forwards=0, optimizer_updates=0)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print('height branch pitch_deg vx_m_s com_x_mm com_z_mm frozen_rotation_mm residual_mm')
    for height in ('h022', 'h026', 'h030'):
        for branch in ('8000_zero', '8000_plus1', '8000_plus2'):
            group = [r for r in rows if r['height'] == height and r['branch'] == branch]
            keys = ('pitch_deg', 'vx_m_s', 'com_axle_forward_mm', 'com_above_axle_mm', 'frozen_pose_rotation_shift_mm', 'pose_adjustment_residual_mm')
            print(height, branch, *[f'{np.mean([r[k] for r in group]):.5f}' for k in keys])


if __name__ == '__main__':
    main()
