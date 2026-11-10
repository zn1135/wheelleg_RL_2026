"""Fit local conservative load and dissipative resistance from PACE prechecks; no stepping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.reference / 'report.json').read_text())
    for path, digest in reference['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Nominal reference source changed')
    config = next(Path(p) for p in reference['source_sha256'] if p.endswith('machine_config.c'))
    if hashlib.sha256((args.paired / 'paired-inputs.npz').read_bytes()).hexdigest() != reference['paired_npz_sha256']:
        raise ValueError('Paired data identity mismatch')
    pose = mirror.MirrorPose(args.reference / 'model.xml', config)
    model, data = pose.model, pose.data
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    mq = np.array([model.joint(name).qposadr[0] for name in mirror.MOTORS[:4]])
    mv = np.array([model.joint(name).dofadr[0] for name in mirror.MOTORS[:4]])
    wheels = [model.joint(name).dofadr[0] for name in mirror.MOTORS[4:]]
    passive = [i for i in range(model.nv) if i not in list(mv) + wheels]
    sq = [model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')]
    sv = [model.joint(s + '_spring2_joint').dofadr[0] for s in ('left', 'right')]
    springs = [model.actuator(s + '_gas_spring_motor').id for s in ('left', 'right')]
    with np.load(args.paired / 'paired-inputs.npz') as source:
        paired = {key: source[key] for key in source.files}
    records = []
    max_closure = max_tangent_error = max_velocity_error = 0.
    for axis in range(4):
        ix = np.flatnonzero((paired['stage'] == 0) & (paired['axis'] == axis))[::5]
        if len(ix) != 3000:
            raise ValueError('Expected a complete30s segment sampled every10ms')
        q = pose.sign[:4] * (paired['q_firmware'][ix] - pose.zero)
        # 平滑只服务离线求导；GPU仍读取原始2ms目标、位置和初态。
        smooth = savgol_filter(q, 11, 3, axis=0)
        positions = []
        for angles in smooth:
            for side in (0, 1):
                if not pose._solve_leg(side, angles[2 * side:2 * side + 2]):
                    raise ValueError('Measured pose does not close')
            positions.append(data.qpos.copy())
            max_closure = max(max_closure, max(pose.residual_m))
        positions = np.asarray(positions)
        velocity = savgol_filter(positions, 11, 3, deriv=1, delta=.01, axis=0)
        acceleration = savgol_filter(positions, 11, 3, deriv=2, delta=.01, axis=0)
        drive = savgol_filter(paired['request_firmware'][ix] * pose.sign[:4], 11, 3, axis=0)
        dynamic, holding = [], []
        inertia = np.zeros(model.nv)
        for qfull, vfull, afull in zip(positions, velocity, acceleration):
            data.qpos[:] = qfull
            data.qvel[:] = vfull
            data.ctrl[:] = 0.
            data.ctrl[springs] = gas_spring_force(qfull[sq], vfull[sv])
            mujoco.mj_forward(model, data)
            mask = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            jac = data.efc_J.reshape(data.nefc, model.nv)[mask]
            tangent = np.zeros((model.nv, 4))
            tangent[mv] = np.eye(4)
            tangent[passive], _, rank, _ = np.linalg.lstsq(jac[:, passive], -jac[:, mv], rcond=None)
            if rank != len(passive):
                raise ValueError('Closed-chain tangent is singular')
            max_tangent_error = max(max_tangent_error, float(abs(jac @ tangent).max()))
            max_velocity_error = max(max_velocity_error, float(abs(jac @ vfull).max()))
            mujoco.mj_mulM(model, data, inertia, afull)
            nominal = data.qfrc_bias - data.qfrc_passive - data.qfrc_actuator
            dynamic.append(tangent.T @ (inertia + nominal))
            holding.append(tangent.T @ nominal)
        records.append(dict(axis=axis, time_s=np.arange(3000) * .01, q=positions[:, mq],
                            v=velocity[:, mv], drive=drive, nominal=np.asarray(dynamic),
                            residual=drive - np.asarray(dynamic), holding=np.asarray(holding)))
        print(json.dumps({'inverse_axis': axis, 'samples': len(ix)}), flush=True)
    q = np.concatenate([r['q'] for r in records])
    v = np.concatenate([r['v'] for r in records])
    residual = np.concatenate([r['residual'] for r in records])
    times = np.tile(np.arange(3000) * .01, 4)
    # 训练与留出时段之间留1秒，远大于0.11秒平滑窗，避免求导泄漏。
    train_time = (times >= 5.5) & (times < 14.5)
    test_time = (times >= 15.5) & (times < 24.5)
    qref = np.mean(q[train_time], axis=0)
    matrix = np.zeros((len(q), 4, 18))
    for joint in range(4):
        matrix[:, joint, joint] = 1.
        matrix[:, joint, 4 + joint] = q[:, joint] - qref[joint]
        matrix[:, joint, 10 + joint] = np.sign(v[:, joint])
        matrix[:, joint, 14 + joint] = v[:, joint]
    for side in range(2):
        x, y = 2 * side, 2 * side + 1
        matrix[:, x, 8 + side] = q[:, y] - qref[y]
        matrix[:, y, 8 + side] = q[:, x] - qref[x]
    moving = abs(v) > .03
    train = train_time[:, None] & moving
    test = test_time[:, None] & moving
    weights = np.broadcast_to(1 / np.sqrt(np.maximum(train.sum(0), 1)), train.shape)
    design, target = matrix[train] * weights[train, None], residual[train] * weights[train]
    regularizer = np.zeros((6, 18))
    regularizer[:, 4:10] = .02 * np.eye(6)
    low = np.r_[np.full(4, -15.), np.full(6, -30.), np.zeros(8)]
    high = np.r_[np.full(4, 15.), np.full(6, 30.), np.full(8, 10.)]
    robust = np.ones(len(target))
    for _ in range(6):
        fit = lsq_linear(np.vstack((design * robust[:, None], regularizer)),
                         np.r_[target * robust, np.zeros(6)], bounds=(low, high), tol=1e-11, max_iter=200)
        error = matrix[train] @ fit.x - residual[train]
        scale = max(.15, 1.4826 * np.median(abs(error - np.median(error))))
        robust = np.sqrt(np.minimum(1., 2 * scale / np.maximum(abs(error), 1e-12)))
    x = fit.x
    stiffness = np.diag(x[4:8])
    stiffness[0, 1] = stiffness[1, 0] = x[8]
    stiffness[2, 3] = stiffness[3, 2] = x[9]
    prediction = matrix @ x
    metrics = {}
    for name, mask in (('train_moving', train), ('heldout_moving', test)):
        metrics[name] = dict(samples=mask.sum(0).tolist(),
                             nominal_rmse_nm=[float(np.sqrt(np.mean(residual[mask[:, j], j] ** 2))) for j in range(4)],
                             fitted_rmse_nm=[float(np.sqrt(np.mean((prediction - residual)[mask[:, j], j] ** 2))) for j in range(4)])
    parameters = dict(model='local_conservative_load_plus_coulomb_viscous',
                      reference_q_model_rad=qref.tolist(), bias_nm=x[:4].tolist(),
                      stiffness_nm_rad=stiffness.tolist(), friction_nm=x[10:14].tolist(),
                      damping_nms_rad=x[14:18].tolist())
    report = dict(scope='Conditional inverse-load fit from low-frequency prechecks; requires independent free GPU replay',
                  parameters=parameters, metrics=metrics, training_segment_time_s=[5.5, 14.5],
                  heldout_segment_time_s=[15.5, 24.5], smoothing_window_s=.11,
                  regularization_stiffness=.02, max_pose_closure_m=max_closure,
                  max_virtual_work_error=max_tangent_error, max_velocity_constraint_error=max_velocity_error,
                  paired_npz_sha256=reference['paired_npz_sha256'], source_sha256=reference['source_sha256'],
                  new_gpu_environment_steps=0, policy_training_updates=0,
                  limitations=['Effective local correction, not an isolated spring or motor calibration',
                               'Static friction at zero velocity is an interval; quiet samples not fitted as zero friction',
                               'Heldout time comes from the same recording, not an independent physical experiment',
                               'Mass, geometry, signs, zeros, PD and nominal spring remain fixed'])
    if max_closure >= 1e-6 or max_tangent_error >= 1e-6:
        raise ValueError('Inverse geometry checks failed')
    (args.output / 'parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'inverse.npz', q=q, v=v, time_s=times, residual_nm=residual,
                        predicted_nm=prediction, matrix=matrix, train=train, heldout=test)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
