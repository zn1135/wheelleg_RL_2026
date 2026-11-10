"""Compare recorded holding torques with same-pose gravity and spring loads; no integration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror


def pose_load(pose, frame, firmware_angles):
    """Eliminate passive joints by virtual work, without using soft-constraint forces."""
    model, data = pose.model, pose.data
    pose.apply(frame, 0.)
    if pose.status[0] != 'OK':
        raise ValueError('Representative IMU orientation invalid')
    angles = pose.sign[:4] * (firmware_angles - pose.zero)
    for side in range(2):
        if not pose._solve_leg(side, angles[2 * side:2 * side + 2]):
            raise ValueError('Recorded pose cannot close')
    center = data.qpos.copy()
    closure = list(pose.residual_m)
    motor_dofs = [model.joint(name).dofadr[0] for name in mirror.MOTORS[:4]]
    spring_dofs = [model.joint(f'{side}_spring2_joint').dofadr[0] for side in ('left', 'right')]
    tangent = np.empty((model.nv, 4))
    numerical_gravity = np.empty(4)
    numerical_spring = np.empty((2, 4))
    epsilon = 1e-5
    for column in range(4):
        positions, potential, distances = [], [], []
        for direction in (-1., 1.):
            data.qpos[:] = center
            shifted = angles.copy()
            shifted[column] += direction * epsilon
            side = column // 2
            if not pose._solve_leg(side, shifted[2 * side:2 * side + 2]):
                raise ValueError('Perturbed pose cannot close')
            positions.append(data.qpos.copy())
            mujoco.mj_kinematics(model, data)
            potential.append(-float(np.sum(model.body_mass[:, None] * data.xipos * model.opt.gravity)))
            axial_lengths = []
            for name in ('left', 'right'):
                delta = data.site(f'{name}_gas_spring_lower').xpos - data.site(f'{name}_gas_spring_upper').xpos
                axis = data.xaxis[model.joint(f'{name}_spring1_joint').id]
                # 耳座有25mm横向偏置；150N沿伸缩轴，长度校验须排除该固定偏置。
                axial_lengths.append(np.sqrt(delta @ delta - (delta @ axis)**2))
            distances.append(axial_lengths)
        derivative = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, derivative, 2 * epsilon, positions[0], positions[1])
        tangent[:, column] = derivative
        numerical_gravity[column] = (potential[1] - potential[0]) / (2 * epsilon)
        numerical_spring[:, column] = (np.asarray(distances[1]) - distances[0]) / (2 * epsilon)
    data.qpos[:] = center
    data.qvel[:] = 0.
    data.ctrl[:] = 0.
    mujoco.mj_forward(model, data)  # 只计算此姿态的力和Jacobian，不推进时间。
    gravity = tangent.T @ data.qfrc_bias
    # 气弹簧执行器是gear=1的伸缩关节力，正值使其伸长。
    for side in ('left', 'right'):
        actuator = model.actuator(f'{side}_gas_spring_motor').id
        if model.actuator_gear[actuator, 0] != 1:
            raise ValueError('Unexpected spring actuator gear')
    spring_per_newton = -tangent[spring_dofs].T
    passive = -(tangent.T @ data.qfrc_passive)
    equality_rows = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[equality_rows]
    verification = dict(
        closure_max_m=max(closure),
        independent_motor_tangent_error=float(np.max(np.abs(tangent[motor_dofs] - np.eye(4)))),
        constraint_virtual_work_error=float(np.max(np.abs(jacobian @ tangent))),
        gravity_potential_derivative_error_nm=float(np.max(np.abs(gravity - numerical_gravity))),
        spring_distance_derivative_error_m_rad=float(np.max(np.abs(spring_per_newton.T + numerical_spring))))
    if (verification['closure_max_m'] > 1e-6 or
            verification['independent_motor_tangent_error'] > 1e-6 or
            verification['constraint_virtual_work_error'] > 1e-6 or
            verification['gravity_potential_derivative_error_nm'] > 1e-5 or
            verification['spring_distance_derivative_error_m_rad'] > 1e-6):
        raise ValueError(f'Static load reconstruction failed: {verification}')
    return gravity, spring_per_newton, passive, verification


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--decoded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta, asset, config = mirror.load_session(args.session)
    decoded_report = json.loads((args.decoded / 'report.json').read_text())
    if decoded_report['stream_sha256'] != meta['stream_sha256']:
        raise ValueError('Decoded input belongs to another stream')
    frame_parser = mirror.FrameParser()
    frames = []
    for event in mirror.iter_session_events(args.session):
        if event['type'] == 'data':
            frames.extend(f for f in frame_parser.feed(bytes.fromhex(event['data_hex'])) if int(f[152]) == 6)
    frames = np.asarray(frames)
    times = np.array([mirror.read_u32(f[8:10]) / 1000 for f in frames])
    moving = np.flatnonzero(np.abs(frames[:, 91]) > 1e-6)
    if not len(moving):
        raise ValueError('No dial movement separating stationary poses')
    windows = [('before_motion', times[moving[0]] - 10., times[moving[0]] - 2.),
               ('after_motion', times[moving[-1]] + 5., times[moving[-1]] + 15.)]
    states, geometries, controls = {}, {}, {}
    sequences = {name: [] for name, _, _ in windows}
    with (args.decoded / 'outputs.csv').open() as source:
        for row in csv.DictReader(source):
            time = int(row['begin_ns']) / 1e9
            selected = [name for name, lo, hi in windows if lo <= time <= hi]
            if not any(lo - .005 <= time <= hi + .005 for _, lo, hi in windows):
                continue
            for name in selected:
                sequences[name].append(int(row['sequence']))
            if row['kind'] == 'can':
                continue
            if row['strategy'] != '6' or row['fault'] != '0':
                raise ValueError('Selected window includes inactive or faulted records')
            key = int(row['id'])
            if row['kind'] == 'control':
                if int(row['flags_or_hal']) & 62 != 60:
                    raise ValueError('Selected window includes gated output')
                controls[key] = [float(row[f'request_{i}_firmware_nm']) for i in range(4)]
            elif row['kind'] == 'leg_state':
                states[key] = [time, *[float(row[f'leg_{i}_position_rad']) for i in range(4)],
                               *[float(row[f'leg_{i}_velocity_rad_s']) for i in range(4)],
                               *[float(row[f'leg_{i}_rx_age_us']) for i in range(4)]]
            elif row['kind'] == 'leg_geometry':
                if int(row['leg_valid_flags']) & 15 != 15:
                    raise ValueError('Invalid leg/test state')
                geometries[key] = [float(row[f'leg_{i}_feedback_torque_nm']) for i in range(4)]
    pose = mirror.MirrorPose(asset, config)
    pose.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, lo, hi in windows:
        seq = sequences[name]
        if not seq or np.any(np.diff(seq) != 1):
            raise ValueError(f'{name}: output sequence gaps')
        keys = [key for key, state in states.items() if lo <= state[0] <= hi]
        state = np.asarray([states[key] for key in keys])
        request = np.asarray([controls[key] for key in keys])
        feedback = np.asarray([geometries[key] for key in keys])
        window_frames = frames[(times >= lo) & (times <= hi)]
        if len(keys) < 100 or not len(window_frames) or np.any(window_frames[:, 91] != 0):
            raise ValueError(f'{name}: inadequate neutral-dial data')
        position_span = np.ptp(state[:, 1:5], axis=0)
        velocity_peak = np.max(np.abs(state[:, 5:9]), axis=0)
        if position_span.max() > .003 or velocity_peak.max() > .03 or state[:, 9:13].max() > 10000:
            raise ValueError(f'{name}: not stationary/fresh: span={position_span}, velocity={velocity_peak}')
        representative = window_frames[len(window_frames) // 2]
        angles = np.median(state[:, 1:5], axis=0)
        gravity, spring, passive, verification = pose_load(pose, representative, angles)
        sign = pose.sign[:4]
        nominal = gravity + passive + spring @ np.full(2, 150.)
        measured_model = feedback.mean(axis=0) * sign
        # 等效弹簧力仅是静态负载一致性诊断，残差包含未知静摩擦、支撑和扭矩误差。
        equivalent_force, _, rank, singular = np.linalg.lstsq(spring, measured_model - gravity - passive, rcond=None)
        equivalent_prediction = gravity + passive + spring @ equivalent_force
        result = dict(name=name, board_time_window_s=[lo, hi], state_frames=len(keys),
                      output_sequences_contiguous=True, max_data_age_us=float(state[:, 9:13].max()),
                      motor_position_span_rad=position_span.tolist(), peak_recorded_velocity_rad_s=velocity_peak.tolist(),
                      representative_motor_angles_firmware_rad=angles.tolist(),
                      representative_quaternion_wxyz=representative[22:26].tolist(),
                      recorded_request_mean_firmware_nm=request.mean(axis=0).tolist(),
                      recorded_feedback_mean_firmware_nm=feedback.mean(axis=0).tolist(),
                      feedback_minus_request_mean_abs_nm=np.mean(np.abs(feedback - request), axis=0).tolist(),
                      gravity_only_required_firmware_nm=(gravity * sign).tolist(),
                      passive_required_firmware_nm=(passive * sign).tolist(),
                      spring_required_firmware_nm_per_n=(spring * sign[:, None]).tolist(),
                      nominal_150n_required_firmware_nm=(nominal * sign).tolist(),
                      nominal_minus_feedback_firmware_nm=(nominal * sign - feedback.mean(axis=0)).tolist(),
                      equivalent_spring_force_n=equivalent_force.tolist(),
                      equivalent_force_remaining_residual_firmware_nm=((measured_model - equivalent_prediction) * sign).tolist(),
                      spring_matrix_rank=int(rank), spring_matrix_singular_values=singular.tolist(),
                      verification=verification)
        results.append(result)
        np.savez_compressed(args.output / (name + '.npz'), state=state, request=request, feedback=feedback,
                            tangent_gravity=gravity, spring_required_per_n=spring, passive_required=passive)
    report = dict(scope='Static same-pose virtual-work diagnostic; no integration, hardware access or physical parameter changes',
                  stream_sha256=meta['stream_sha256'], whole_stream_complete=decoded_report['complete_observed_span'],
                  source_model_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                  source_mapping_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  environment_steps=0, model_forwards=0, optimizer_updates=0, windows=results,
                  limitations=['Recorded torque is a drive estimate, not independent shaft torque',
                               'Stationary does not exclude static friction, mechanical support or hysteresis',
                               'Equivalent force is not an identified spring parameter',
                               'No velocity/acceleration fit or dynamic acceptance'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for source in (Path(__file__).resolve(), ROOT / 'sim2sim/real2sim_huananhu.py',
                   ROOT / 'sim2sim/wheel_response.py', asset, config):
        destination = args.output / 'source' / source.resolve().relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    x = np.arange(4)
    for ax, result in zip(axes, results):
        for shift, key, label in ((-.25, 'recorded_feedback_mean_firmware_nm', 'Drive torque estimate'),
                                  (0., 'gravity_only_required_firmware_nm', 'Model gravity only'),
                                  (.25, 'nominal_150n_required_firmware_nm', 'Model gravity + 150 N')):
            ax.bar(x + shift, result[key], width=.25, label=label)
        ax.set_xticks(x, mirror.MOTORS[:4]); ax.set_title(result['name']); ax.grid(axis='y', alpha=.3)
    axes[0].set_ylabel('Firmware motor torque (Nm)'); axes[1].legend()
    fig.tight_layout(); fig.savefig(args.output / 'static-load-comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
