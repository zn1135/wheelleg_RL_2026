#!/usr/bin/env python3
"""Audit saved initialization errors and kinematic assumptions without physics steps."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from scripts.agent.split.walk_sensor_velocity import SensorWheelVelocity


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(value):
    value = np.asarray(value, dtype=float)
    return dict(minimum=float(value.min()), median=float(np.median(value)),
        maximum=float(value.max()))


def rotate_inverse(quaternion, vector):
    xyz = quaternion[..., 1:]
    cross = np.cross(xyz, vector)
    return vector-2.*quaternion[..., :1]*cross+2.*np.cross(xyz, cross)


def first_time(mask, times):
    return None if not mask.any() else float(times[np.flatnonzero(mask)[0]])


def audit(root, role, gate):
    folder = root/role
    meta_file, trace_file = folder/'encoder.json', folder/'encoder.npz'
    candidate_file = root/'offline-r1'/f'{role}-candidate.npz'
    meta = json.loads(meta_file.read_text())
    assert meta['schedule'] == role and meta['feedback_mode'] == 'encoder'
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged']
    scene = Path(meta['scene_mjb'])
    assert digest(scene) == meta['scene_sha256']
    model = mujoco.MjModel.from_binary_path(str(scene))
    estimator = SensorWheelVelocity(model)
    with np.load(trace_file, allow_pickle=False) as saved:
        values, poses = saved['values'], saved['qpos']
        packet, times, valid = saved['policy_observation'], saved['policy_time_s'], saved['policy_valid']
        target, encoder = saved['target_com_m_s'], saved['encoder_m_s']
        nominal, initial_v = saved['fixed_nominal'], saved['fixed_qvel']
    with np.load(candidate_file, allow_pickle=False) as saved:
        candidate, available = saved['candidate_vx_m_s'], saved['available']
        assert np.array_equal(saved['policy_time_s'], times)
    assert packet.shape == (3000, 96, 25) and values.shape[:2] == (15000, 96)
    assert poses.shape == (3001, 96, model.nq)
    assert np.asarray(meta['episode_done']).sum() == 0
    assert valid.all() and available.all()
    initial = (times >= gate['start_s']) & (times < gate['end_s'])
    count = int(initial.sum())
    initial_ids = np.flatnonzero(initial)
    assert np.array_equal(initial_ids, np.arange(count))
    error = abs(candidate[:count]-target[:count, :, 0])
    peak_tick = error.argmax(0)
    envs = np.arange(96)
    peak_packet = packet[peak_tick, envs]
    predicted_again = estimator.predict(peak_packet, nominal)
    assert np.allclose(predicted_again, candidate[peak_tick, envs], rtol=0., atol=1e-10)

    def field(name):
        return values[..., slice(*meta['layout'][name])]

    # 当前策略拍起点与上拍最后一个物理子步配对，避免错用积分后的2ms样本。
    boundary_rows = np.maximum(np.arange(count)*5-1, 0)
    world_v = field('root_linear_velocity_w')[boundary_rows].astype(np.float64)
    free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    qadr, vadr = int(model.jnt_qposadr[free]), int(model.jnt_dofadr[free])
    world_v[0] = initial_v[:, vadr:vadr+3]
    quat = poses[:count, :, qadr+3:qadr+7].astype(np.float64)
    omega = packet[:count, :, :3].astype(np.float64)/.25
    raw_com = rotate_inverse(quat, world_v)+np.cross(omega, estimator.com_offset)
    label_difference = raw_com-target[:count]
    raw_error = abs(candidate[:count]-raw_com[..., 0])
    raw_peak_tick = raw_error.argmax(0)
    full_motor_order = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
    motor_q_order = ('lf0', 'lf00', 'rf0', 'rf00')
    motor_vadr = [int(model.joint('robot/'+name).dofadr[0]) for name in full_motor_order]
    motor_qadr = [int(model.joint('robot/'+name).qposadr[0]) for name in motor_q_order]
    joint_trace_ids = [meta['joint_names'].index(name) for name in full_motor_order]
    actual_dq = field('joint_dq')[boundary_rows][..., joint_trace_ids].copy()
    actual_dq[0] = initial_v[:, motor_vadr]
    actual_gyro_xy = field('angular_velocity_xy')[boundary_rows].copy()
    actual_gyro_xy[0] = initial_v[:, vadr+3:vadr+5]
    measured_q = packet[:count, :, 9:13]+nominal[None, :, [0, 1, 3, 4]]
    mapped_q_error = measured_q-poses[:count, :, motor_qadr]
    mapped_dq_error = packet[:count, :, 13:19]/.05-actual_dq
    mapped_gyro_xy_error = omega[..., :2]-actual_gyro_xy

    reconstructed = []
    for leg in estimator.legs:
        motor_q = peak_packet[:, leg['observation_ids']]+nominal[:, leg['nominal_ids']]-leg['refs']
        motor_dq = peak_packet[:, leg['velocity_ids']]/.05
        point, relative_velocity, knee, knee_rate = estimator._leg_state(leg, motor_q, motor_dq)
        joint = model.joint(leg['knee'])
        true_knee = poses[peak_tick, envs, int(joint.qposadr[0])]
        knee_trace_id = meta['joint_names'].index(leg['knee'].split('/')[-1])
        true_knee_rate = field('joint_dq')[np.maximum(5*peak_tick-1, 0), envs, knee_trace_id]
        true_knee_rate[peak_tick == 0] = initial_v[peak_tick == 0, int(joint.dofadr[0])]
        reconstructed.append(dict(leg=leg, point=point, relative_velocity=relative_velocity,
            knee=knee, knee_rate=knee_rate, true_knee=true_knee, true_knee_rate=true_knee_rate))

    def physical(row, env):
        if row is None:
            return None
        support = field('support')[row, env]
        return dict(time_s=float((row+1)*meta['dt']), support_N=support.tolist(),
            total_support_weight_fraction=float(support.sum()/meta['weight_N']),
            loaded=bool((support >= max(.1*meta['weight_N'], 5.)).all()
                        and support.sum() >= .8*meta['weight_N']),
            base_contact_N=float(field('base_contact')[row, env, 0]),
            height_m=float(field('height')[row, env, 0]),
            tilt_deg=float(field('tilt')[row, env, 0]),
            friction_ratio=field('friction_ratio')[row, env].tolist(),
            actual_leg_torque_Nm=field('actual_torque')[row, env, [0, 1, 3, 4]].tolist())

    base_id = model.body('robot/base_link').id
    physics_times = (np.arange(len(values))+1)*meta['dt']
    data = mujoco.MjData(model)
    records = []
    for env in envs:
        tick = int(peak_tick[env])
        first, previous, last = 5*tick, 5*tick-1 if tick else None, 5*tick+4
        data.qpos[:] = poses[tick, env]
        # 只求保存姿态的正向几何，不调用forward、step或动力学积分。
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[base_id].reshape(3, 3)
        leg_rows = []
        for result in reconstructed:
            leg = result['leg']
            wheel_id = model.body(leg['wheel']).id
            true_point = rotation.T@(data.xpos[wheel_id]-data.xpos[base_id])
            leg_rows.append(dict(passive_joint=leg['knee'],
                reconstructed_knee_rad=float(result['knee'][env]),
                actual_passive_knee_rad=float(result['true_knee'][env]),
                knee_error_deg=float(np.rad2deg(result['knee'][env]-result['true_knee'][env])),
                reconstructed_knee_rate_rad_s=float(result['knee_rate'][env]),
                actual_passive_knee_rate_rad_s=float(result['true_knee_rate'][env]),
                knee_rate_error_rad_s=float(result['knee_rate'][env]-result['true_knee_rate'][env]),
                reconstructed_wheel_point_body_m=result['point'][env].tolist(),
                actual_wheel_point_body_m=true_point.tolist(),
                wheel_point_error_m=(result['point'][env]-true_point).tolist(),
                wheel_center_world_z_m=float(data.xpos[wheel_id, 2]),
                reconstructed_wheel_relative_velocity_body_m_s=result['relative_velocity'][env].tolist()))
        support = field('support')[:, env]
        early = physics_times < gate['end_s']
        any_load = (support > 5.).any(-1) & early
        loaded = (support >= max(.1*meta['weight_N'], 5.)).all(-1) & (support.sum(-1) >= .8*meta['weight_N']) & early
        records.append(dict(env=int(env), peak_policy_tick=tick, peak_policy_time_s=float(times[tick]),
            compensated_vx_m_s=float(candidate[tick, env]),
            encoder_vx_m_s=float(encoder[tick, env, 0]), derived_com_vx_m_s=float(target[tick, env, 0]),
            raw_boundary_com_vx_m_s=float(raw_com[tick, env, 0]),
            peak_absolute_error_derived_m_s=float(error[tick, env]),
            absolute_error_raw_boundary_at_same_peak_m_s=float(raw_error[tick, env]),
            raw_boundary_peak_tick=int(raw_peak_tick[env]),
            raw_boundary_peak_absolute_error_m_s=float(raw_error[raw_peak_tick[env], env]),
            initial_encoder_peak_absolute_error_m_s=float(abs(encoder[:count, env, 0]-target[:count, env, 0]).max()),
            derived_minus_raw_com_m_s=label_difference[tick, env].tolist(),
            first_any_wheel_force_above5N_s=first_time(any_load, physics_times),
            first_sufficient_wheel_load_s=first_time(loaded, physics_times),
            physical_pre_action=physical(previous, env), physical_after_first2ms=physical(first, env),
            physical_after_policy10ms=physical(last, env),
            packet_gyro_body_rad_s=omega[tick, env].tolist(),
            packet_projected_gravity=packet[tick, env, 3:6].tolist(),
            packet_motor_q_rad=measured_q[tick, env].tolist(),
            packet_motor_dq_rad_s=(packet[tick, env, 13:19]/.05).tolist(),
            sensor_vs_boundary_q_max_abs_rad=float(abs(mapped_q_error[tick, env]).max()),
            sensor_vs_boundary_dq_max_abs_rad_s=float(abs(mapped_dq_error[tick, env]).max()),
            sensor_vs_boundary_gyro_xy_max_abs_rad_s=float(abs(mapped_gyro_xy_error[tick, env]).max()),
            passive_geometry_evaluation_only=leg_rows))

    all_leg_rows = [leg for record in records for leg in record['passive_geometry_evaluation_only']]
    pre_physical = [record['physical_pre_action'] for record in records]
    summary = dict(environments=96, first_episodes_survived=int(meta['survived']),
        peak_time_s_counts=dict(Counter(f'{times[tick]:.2f}' for tick in peak_tick)),
        peak_absolute_error_derived_m_s=describe([record['peak_absolute_error_derived_m_s'] for record in records]),
        initial_encoder_peak_absolute_error_m_s=describe([record['initial_encoder_peak_absolute_error_m_s'] for record in records]),
        raw_boundary_peak_absolute_error_m_s=describe(raw_error.max(0)),
        raw_vs_derived_com_vx_max_abs_m_s=float(abs(label_difference[..., 0]).max()),
        raw_vs_derived_com_xyz_max_abs_m_s=float(abs(label_difference).max()),
        raw_vs_derived_com_vx_at_original_peaks=describe(abs(label_difference[peak_tick, envs, 0])),
        raw_peak_tick_identical_count=int((raw_peak_tick == peak_tick).sum()),
        mapping_initial_max_abs=dict(motor_q_rad=float(abs(mapped_q_error).max()),
            motor_dq_rad_s=float(abs(mapped_dq_error).max()), gyro_xy_rad_s=float(abs(mapped_gyro_xy_error).max())),
        peak_pre_contact_loaded_count=sum(record['loaded'] for record in pre_physical if record is not None),
        peak_pre_total_support_weight_fraction=describe([record['total_support_weight_fraction'] for record in pre_physical if record is not None]),
        peak_pre_base_contact_max_N=max(record['base_contact_N'] for record in pre_physical if record is not None),
        peak_pre_friction_ratio_max=max(max(record['friction_ratio']) for record in pre_physical if record is not None),
        peak_passive_knee_absolute_error_deg=describe([abs(leg['knee_error_deg']) for leg in all_leg_rows]),
        peak_passive_knee_rate_absolute_error_rad_s=describe([abs(leg['knee_rate_error_rad_s']) for leg in all_leg_rows]),
        peak_wheel_point_error_norm_m=describe([np.linalg.norm(leg['wheel_point_error_m']) for leg in all_leg_rows]))
    inputs = [meta_file, trace_file, candidate_file, scene]
    return dict(role=role, seed=meta['seed'], summary=summary, per_environment=records,
        inputs={str(path.resolve()):digest(path) for path in inputs},
        prediction_recheck_max_abs_m_s=float(abs(predicted_again-candidate[peak_tick, envs]).max()),
        gyro_z_limit='Full gyro_z is the actual Actor sensor packet; no separate raw root-qvel-z series was saved. It is used only in this scoring conversion, never reconstructed from future poses.',
        timing='At policy tick k, qpos[k] and values[5*k-1] raw qvel are the pre-action boundary for k>0; k=0 uses fixed_qvel. Packet gyro and labels are saved at the same action start. Contact after_first2ms and after_policy10ms are explicitly later.',
        privileged_state_scope='True root translation/velocity, passive q/dq, contact and MuJoCo FK are evaluation evidence only; none enter the frozen sensor-only prediction.')


def main(args):
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lock_path = args.root/'offline-r1/gate-lock.json'
    lock = json.loads(lock_path.read_text())
    candidate_source = Path(inspect.getfile(SensorWheelVelocity)).resolve()
    assert digest(candidate_source) == lock['candidate_sha256']
    initial_gate = lock['gate_spec']['initial']
    runs = [audit(args.root, role, initial_gate) for role in ('discovery', 'validation')]
    assert digest(candidate_source) == lock['candidate_sha256']
    script = Path(__file__).resolve()
    result = dict(verified=True, simulation_steps=0, optimizer_steps=0, fitted_parameters=0,
        candidate_modified=False, gate_modified=False, collector_modified=False,
        script=str(script), script_sha256=digest(script), candidate_sha256=digest(candidate_source),
        gate_lock_sha256=digest(lock_path), initial_window_s=[initial_gate['start_s'], initial_gate['end_s']],
        raw_boundary_com_formula='R(qpos[k])^T * root_qvel_world[k] + packet_gyro_body[k] cross fixed_rootbody_com_offset',
        motor_q_order=['lf0', 'lf00', 'rf0', 'rf00'],
        motor_dq_order=['lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel'],
        mujoco_math_calls=['MjModel.from_binary_path', 'MjData', 'mj_kinematics'], runs=runs,
        interpretation='Initialization peaks persist after raw boundary COM scoring. Coinciding contact load/impact and passive-closure deviations test assumptions; correlations do not identify a unique cause. No startup gate, filter, bias, new estimator or training is introduced.',
        limits=['No independent raw gyro_z stream beyond the exact sensor packet.',
            'Contact force and friction ratio do not independently prove sliding or rigid-contact kinematics.',
            'Saved pose FK evaluates soft-closure differences but does not prove they uniquely cause the error.',
            'No task acceptance or compensated closed-loop behavior is measured.'])
    (args.output/'startup-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copy2(script, args.output/'source.py')
    print(json.dumps(dict(verified=True, output=str(args.output/'startup-audit.json'),
        runs=[dict(role=run['role'], summary=run['summary']) for run in runs]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
