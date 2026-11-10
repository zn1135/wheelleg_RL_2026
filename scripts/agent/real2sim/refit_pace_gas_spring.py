"""Separate measured-pose gravity/inertia from gas force and resistance; no integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from scripts.agent.real2sim import replay_pace_capture_gpu as replay
from scripts.agent.real2sim.check_leg_static_load import pose_load
from scripts.agent.real2sim.analyze_tilt_length import geometry


def normalized(quat):
    quat = np.asarray(quat, dtype=float).copy()
    for i in range(1, len(quat)):
        if quat[i] @ quat[i - 1] < 0:
            quat[i] *= -1
    return quat / np.linalg.norm(quat, axis=1, keepdims=True)


class Loads:
    def __init__(self, asset, config):
        self.pose = replay.mirror.MirrorPose(asset, config)
        self.model, self.data = self.pose.model, self.pose.data
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.model.geom_contype[:] = self.model.geom_conaffinity[:] = 0
        self.mq = np.array([self.model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]])
        self.mv = np.array([self.model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[:4]])
        wheels = [self.model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[4:]]
        self.passive = [i for i in range(6, self.model.nv) if i not in list(self.mv) + wheels]
        self.sq = [self.model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')]
        self.sv = [self.model.joint(s + '_spring2_joint').dofadr[0] for s in ('left', 'right')]
        self.max_virtual_work = self.max_closure = 0.

    def position(self, q_model, quat, wheel=None):
        self.data.qpos[:7] = [0., 0., .8, *quat]
        if wheel is not None:
            self.data.qpos[self.pose.wheel_q] = wheel
        for side in (0, 1):
            if not self.pose._solve_leg(side, q_model[2 * side:2 * side + 2]):
                raise ValueError('Measured pose cannot close')
        self.max_closure = max(self.max_closure, max(self.pose.residual_m))
        return self.data.qpos.copy()

    def evaluate(self, position, velocity, acceleration, first_quat):
        d, m = self.data, self.model
        d.qpos[:] = position
        d.qvel[:] = velocity
        d.ctrl[:] = 0.
        mujoco.mj_forward(m, d)
        mask = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac = d.efc_J.reshape(d.nefc, m.nv)[mask]
        tangent = np.zeros((m.nv, 4))
        tangent[self.mv] = np.eye(4)
        tangent[self.passive], _, rank, _ = np.linalg.lstsq(jac[:, self.passive], -jac[:, self.mv], rcond=None)
        if rank != len(self.passive):
            raise ValueError('Singular closed-chain projection')
        self.max_virtual_work = max(self.max_virtual_work, float(abs(jac @ tangent).max()))
        inertia = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, inertia, acceleration)
        nonspring = tangent.T @ (inertia + d.qfrc_bias - d.qfrc_passive)
        # 重力单独以同一姿态、零速度求出；腿倾角自然包含在所有连杆位姿内。
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        gravity = tangent.T @ d.qfrc_bias
        d.qpos[3:7] = first_quat
        mujoco.mj_forward(m, d)
        fixed_base_gravity = tangent.T @ d.qfrc_bias
        d.qpos[:] = position
        return nonspring, gravity, fixed_base_gravity, -tangent[self.sv].T


def decode(x):
    return dict(model='gas_spring_linear_with_joint_resistance', spring_reference_compression_m=.005,
                spring_reference_force_n=(150 * x[:2]).tolist(), spring_stiffness_n_m=(1000 * x[2:4]).tolist(),
                spring_damping_ns_m=(100 * x[4:6]).tolist(), spring_force_limit_n=[600., 600.],
                friction_nm=x[6:10].tolist(), damping_nms_rad=x[10:14].tolist())


def predict(parameters, compression, slide_v, basis, motor_v, friction=True):
    force = np.clip(np.asarray(parameters['spring_reference_force_n'])
                    + (compression - .005) * parameters['spring_stiffness_n_m']
                    - slide_v * parameters['spring_damping_ns_m'], 0., parameters['spring_force_limit_n'])
    torque = np.einsum('tij,tj->ti', basis, force)
    if friction:
        torque += np.sign(motor_v) * parameters['friction_nm'] + motor_v * parameters['damping_nms_rad']
    return torque, force


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--static-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    a, frames, ft, (asset, config), audit, pd_error = replay.inputs(args.session, args.paired)
    np.savez_compressed(args.output / 'verified-inputs.npz', frames=frames, ft=ft, **a)
    loads = Loads(asset, config)
    first_ns = a['board_ns'][0] / 1e9
    first_quat = normalized(frames[np.searchsorted(ft, first_ns, side='right') - 1:np.searchsorted(ft, first_ns, side='right'), 22:26])[0]
    full_quat = normalized(frames[:, 22:26])
    rows, gravity_checks = [], []
    for axis in range(4):
        ix = np.flatnonzero((a['stage'] == 0) & (a['axis'] == axis))[::5]
        bt = a['board_ns'][ix] / 1e9
        quat = normalized(np.column_stack([np.interp(bt, ft, full_quat[:, j]) for j in range(4)]))
        quat = normalized(savgol_filter(quat, 11, 3, axis=0))
        wheel = np.column_stack([np.interp(bt, ft, frames[:, 18 + j]) for j in range(2)]) * loads.pose.sign[4:]
        q = savgol_filter(loads.pose.sign[:4] * (a['q_firmware'][ix] - loads.pose.zero), 11, 3, axis=0)
        positions = np.array([loads.position(angles, orientation, w) for angles, orientation, w in zip(q, quat, wheel)])
        velocity = np.zeros((len(ix), loads.model.nv))
        for i in range(len(ix)):
            lo, hi = max(0, i - 1), min(len(ix) - 1, i + 1)
            mujoco.mj_differentiatePos(loads.model, velocity[i], .01 * (hi - lo), positions[lo], positions[hi])
        velocity = savgol_filter(velocity, 11, 3, axis=0)
        acceleration = savgol_filter(velocity, 11, 3, deriv=1, delta=.01, axis=0)
        drive = savgol_filter(a['request_firmware'][ix] * loads.pose.sign[:4], 11, 3, axis=0)
        outputs = [loads.evaluate(p, v, acc, first_quat) for p, v, acc in zip(positions, velocity, acceleration)]
        nominal, gravity, fixed, basis = [np.array([r[j] for r in outputs]) for j in range(4)]
        raw_fw = q * loads.pose.sign[:4] + loads.pose.zero
        tilt = np.column_stack([geometry(raw_fw[:, side * 2:side * 2 + 2], .21, .25)[1] for side in (0, 1)])
        for j in (250, 750, 1250, 1750, 2250, 2750):
            frame = frames[np.searchsorted(ft, bt[j], side='right') - 1].copy()
            frame[22:26] = quat[j]
            frame[18:20] = wheel[j] * loads.pose.sign[4:]
            g, _, _, check = pose_load(loads.pose, frame, raw_fw[j])
            difference = float(abs(g - gravity[j]).max())
            if difference > 1e-5:
                print(json.dumps({'axis': axis, 'sample': j, 'analytic_gravity': gravity[j].tolist(),
                                  'independent_gravity': g.tolist(), 'difference_nm': difference}), flush=True)
                raise ValueError('Independent gravity reconstruction mismatch')
            gravity_checks.append(dict(axis=axis, sample=j, torque_error_nm=difference, **check))
        rows.append(dict(q=q, v=velocity[:, loads.mv], compression=np.maximum(.01 - positions[:, loads.sq], 0.),
                         slide_v=velocity[:, loads.sv], basis=basis, gravity=gravity, fixed_gravity=fixed,
                         nonspring=nominal, target=drive - nominal, drive=drive, tilt=tilt, quat=quat,
                         time_s=np.arange(len(ix)) * .01, axis=np.full(len(ix), axis)))
        print(json.dumps({'dynamic_axis': axis, 'samples': len(ix)}), flush=True)
    dynamic = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    # 静置点按每帧实际姿态重算，再按原取点去重；静摩擦只作为容量区间。
    static_summary = json.loads(args.static_summary.read_text())
    static_rows, static_sources = [], []
    for ordinal, path in enumerate(static_summary['reports']):
        report = json.loads(Path(path).read_text())
        if report['model_sha256'] != replay.mirror.file_identity(asset)['sha256'] or report['config_sha256'] != replay.mirror.file_identity(config)['sha256']:
            raise ValueError('Static model or mapping identity differs')
        data_path = Path(report['source_frame_file'])
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if digest != report['source_frames_sha256']:
            raise ValueError('Static raw frame identity mismatch')
        with np.load(data_path) as source:
            f = source['frames']
        outputs = []
        for frame in f:
            q_model = loads.pose.sign[:4] * (frame[10:14] - loads.pose.zero)
            position = loads.position(q_model, normalized(frame[None, 22:26])[0], frame[18:20] * loads.pose.sign[4:])
            outputs.append(loads.evaluate(position, np.zeros(loads.model.nv), np.zeros(loads.model.nv), first_quat))
        g = np.mean([r[1] for r in outputs], axis=0)
        basis = np.mean([r[3] for r in outputs], axis=0)
        compression = []
        for frame in f[::max(1, len(f) // 10)]:
            pos = loads.position(loads.pose.sign[:4] * (frame[10:14] - loads.pose.zero), normalized(frame[None, 22:26])[0])
            compression.append(np.maximum(.01 - pos[loads.sq], 0.))
        sides = [s in report['fit_sides'] for s in ('left', 'right')]
        static_rows.append(dict(compression=np.mean(compression, axis=0), basis=basis, gravity=g,
                                target=np.mean(f[:, 146:150], axis=0) * loads.pose.sign[:4] - g,
                                sides=sides, old_gravity=np.array(report['gravity_required_nm']) * loads.pose.sign[:4],
                                old_angle=[leg['leg_angle_rad'] for leg in report['legs']], ordinal=ordinal))
        static_sources.append(dict(path=str(data_path), sha256=digest))
        print(json.dumps({'static_pose': ordinal + 1, 'total': len(static_summary['reports'])}), flush=True)
    static = {key: np.array([r[key] for r in static_rows]) for key in ('compression', 'basis', 'gravity', 'target', 'old_gravity', 'old_angle')}
    static_mask = np.repeat(np.array([r['sides'] for r in static_rows]), 2, axis=1)
    static_train = static_mask & (np.arange(len(static_rows))[:, None] % 2 == 0)
    static_test = static_mask & ~static_train
    train = ((dynamic['time_s'] >= 5.5) & (dynamic['time_s'] < 14.5))[:, None] & (abs(dynamic['v']) > .03)
    heldout = ((dynamic['time_s'] >= 15.5) & (dynamic['time_s'] < 24.5))[:, None] & (abs(dynamic['v']) > .03)
    weights = np.broadcast_to(1 / np.sqrt(np.maximum(train.sum(0), 1)), train.shape)
    static_weights = np.broadcast_to(1 / np.sqrt(np.maximum(static_train.sum(0), 1)), static_train.shape)
    # 无任意关节力矩偏置或位置补偿项，避免它们吸收弹簧推力。
    def residual(x):
        p = decode(x)
        prediction, _ = predict(p, dynamic['compression'], dynamic['slide_v'], dynamic['basis'], dynamic['v'])
        static_prediction, _ = predict(p, static['compression'], np.zeros_like(static['compression']), static['basis'], np.zeros((len(static_rows), 4)), friction=False)
        excess = np.maximum(abs(static_prediction - static['target']) - p['friction_nm'], 0.)
        return np.r_[(prediction - dynamic['target'])[train] * weights[train],
                     .5 * excess[static_train] * static_weights[static_train]]
    initial = np.r_[[1.2, 1.2], [2., 2.], [.5, .5], np.full(4, 1.), np.full(4, .2)]
    low = np.zeros(14)
    high = np.r_[[4., 4.], [20., 20.], [15., 15.], np.full(8, 10.)]
    fit = least_squares(residual, initial, bounds=(low, high), loss='soft_l1', f_scale=.025,
                        max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    parameters = decode(fit.x)
    prediction, force = predict(parameters, dynamic['compression'], dynamic['slide_v'], dynamic['basis'], dynamic['v'])
    static_prediction, static_force = predict(parameters, static['compression'], np.zeros_like(static['compression']), static['basis'], np.zeros((len(static_rows), 4)), friction=False)
    metrics = {}
    for name, mask in (('train', train), ('heldout', heldout)):
        metrics[name] = [float(np.sqrt(np.mean((prediction - dynamic['target'])[mask[:, j], j] ** 2))) for j in range(4)]
    raw_static_error = static_prediction - static['target']
    static_excess = np.maximum(abs(raw_static_error) - parameters['friction_nm'], 0.)
    for name, mask in (('static_train', static_train), ('static_heldout', static_test)):
        metrics[name] = dict(raw_torque_rmse_nm=float(np.sqrt(np.mean(raw_static_error[mask] ** 2))),
                             outside_friction_interval_rmse_nm=float(np.sqrt(np.mean(static_excess[mask] ** 2))))
    relative = normalized(dynamic['quat']) @ first_quat
    report = dict(scope='Gravity/inertia-separated gas-spring and resistance fit; not independent component calibration',
                  parameters=parameters, metrics=metrics, optimizer_nfev=fit.nfev,
                  dynamic_compression_range_mm=(np.stack((dynamic['compression'].min(0), dynamic['compression'].max(0))) * 1000).tolist(),
                  static_compression_range_mm=(np.stack((static['compression'].min(0), static['compression'].max(0))) * 1000).tolist(),
                  dynamic_force_range_n=np.stack((force.min(0), force.max(0))).tolist(),
                  static_force_range_n=np.stack((static_force.min(0), static_force.max(0))).tolist(),
                  leg_tilt_range_deg=np.degrees(np.stack((dynamic['tilt'].min(0), dynamic['tilt'].max(0)))).tolist(),
                  static_leg_tilt_range_deg=np.degrees(np.stack((static['old_angle'].min(0), static['old_angle'].max(0)))).tolist(),
                  recorded_base_orientation_span_deg=float(np.degrees(2 * np.arccos(np.clip(abs(relative), 0., 1.))).max()),
                  gravity_range_nm=np.stack((dynamic['gravity'].min(0), dynamic['gravity'].max(0))).tolist(),
                  variable_base_gravity_max_change_nm=np.max(abs(dynamic['gravity'] - dynamic['fixed_gravity']), axis=0).tolist(),
                  old_static_gravity_vs_framewise_max_change_nm=np.max(abs(static['gravity'] - static['old_gravity']), axis=0).tolist(),
                  previous_gravity_was_present=True, max_closure_m=loads.max_closure,
                  max_virtual_work_error=loads.max_virtual_work, independent_gravity_checks=gravity_checks,
                  raw_prefix_sha256=audit['stream_prefix_sha256'], raw_prefix_bytes=audit['stream_prefix_bytes'],
                  paired_npz_sha256=hashlib.sha256((args.paired / 'paired-inputs.npz').read_bytes()).hexdigest(),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (asset, config)},
                  static_sources=static_sources, new_gpu_environment_steps=0, policy_training_updates=0,
                  limitations=['Both spring and joint resistance remain conditional on nominal mass/COM and driver torque units',
                               'Static points have unknown stiction; only a friction interval is enforced',
                               'Recorded IMU rotation and leg posture included; fixture translation acceleration unavailable',
                               '150N is rated force,600N is only this diagnostic force ceiling',
                               'Heldout time/points share the same capture sessions; free GPU replay remains required'])
    if loads.max_closure > 1e-6 or loads.max_virtual_work > 1e-6:
        raise ValueError('Gravity/mechanism projection failed')
    (args.output / 'parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'inverse.npz', **dynamic, predicted=prediction, force=force, train=train, heldout=heldout)
    np.savez_compressed(args.output / 'static.npz', **static, predicted=static_prediction, force=static_force, train=static_train, heldout=static_test)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    print(json.dumps({k: v for k, v in report.items() if k not in ('independent_gravity_checks', 'static_sources')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
