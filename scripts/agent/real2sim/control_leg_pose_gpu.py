"""Fixed-base GPU control of leg length/angle with a frozen fitted local-load plant."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from scripts.agent.real2sim.elastic_joint_resistance import initial_preload
from scripts.agent.real2sim.fit_leg_length_response import leg_pose


def inverse_pose(length, angle):
    """Current H7 branch, with its existing front-hip pi offset unchanged."""
    c = (np.asarray(length) ** 2 + .21 ** 2 - .25 ** 2) / (2 * .21 * length)
    if np.any(abs(c) >= 1):
        raise ValueError('Leg target is outside the five-bar workspace')
    half = np.arccos(c)
    return np.stack((-angle + half - np.pi / 2, -angle - half + np.pi / 2), axis=-1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--length-mm', type=float, default=190.)
    parser.add_argument('--angle-deg', type=float, default=30.)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([args.length_mm, args.angle_deg]).all() or not 140 <= args.length_mm <= 330:
        raise ValueError('Use finite leg targets within140..330mm')
    args.output.mkdir(parents=True, exist_ok=False)
    params = json.loads(args.parameters.read_text())
    reference = json.loads((args.reference / 'report.json').read_text())
    digest = hashlib.sha256(args.parameters.read_bytes()).hexdigest()
    if params['model'] != 'local_conservative_load_plus_coulomb_viscous' or digest != reference['effective_load_parameters_sha256']:
        raise ValueError('Use the frozen local-load candidate and its reference model')
    for source, expected in reference['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError('Protected source changed: ' + source)
    config = next(Path(source) for source in reference['source_sha256'] if source.endswith('machine_config.c'))
    model = mujoco.MjModel.from_xml_path(str(args.reference / 'model.xml'))
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    if abs(model.opt.timestep - .0002) > 1e-12:
        raise ValueError('Expected0.2ms physics model')
    # 必须在加载映射模型之前保存；MuJoCo保存接口使用最后一次加载的XML模板。
    mujoco.mj_saveLastXML(str(args.output / 'model.xml'), model)
    pose = mirror.MirrorPose(machine_config=config)
    pose.data.qpos[:7] = [0., 0., .8, *model.body('base_link').quat]
    mq = np.array([model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]])
    mv = np.array([model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]])
    motor = np.array([model.actuator(n + '_motor').id for n in mirror.MOTORS])
    sq = np.array([model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')])
    sv = np.array([model.joint(s + '_spring2_joint').dofadr[0] for s in ('left', 'right')])
    spring = np.array([model.actuator(s + '_gas_spring_motor').id for s in ('left', 'right')])
    knees = np.array([model.joint(n).id for n in ('lf1', 'rf1')])
    stiffness = np.asarray(params['stiffness_nm_rad'])
    qref = np.asarray(params['reference_q_model_rad'])
    data = mujoco.MjData(model)
    source_q = np.array([pose.model.joint(model.joint(j).name).qposadr[0] for j in range(model.njnt)])
    def closed(q):
        for side in (0, 1):
            if not pose._solve_leg(side, q[2 * side:2 * side + 2]):
                raise ValueError('Target path cannot close')
        data.qpos[:] = pose.data.qpos[source_q]
        data.qvel[:] = 0.
        if any(model.jnt_limited[j] and not model.jnt_range[j, 0] <= data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]
               for j in range(model.njnt)):
            raise ValueError('Target path exceeds calibrated joint limits')
        return data.qpos.copy()
    initial = closed(qref)
    initial_fw = pose.sign[:4] * qref + pose.zero
    start_length, start_angle, _ = leg_pose(initial_fw[None])
    steps, dt = 8000, .002
    times = np.arange(steps + 1) * dt
    blend_t = np.clip((times - 2.) / 4., 0., 1.)
    blend = blend_t ** 3 * (10 - 15 * blend_t + 6 * blend_t ** 2)
    target_length = start_length[0] + blend[:, None] * (args.length_mm / 1000 - start_length[0])
    target_angle = start_angle[0] + blend[:, None] * (np.deg2rad(args.angle_deg) - start_angle[0])
    target_fw = inverse_pose(target_length, target_angle).reshape(-1, 4)
    target_q = (target_fw - pose.zero) * pose.sign[:4]
    reconstructed = leg_pose(target_fw)
    if max(abs(reconstructed[0] - target_length).max(), abs(reconstructed[1] - target_angle).max()) > 1e-9:
        raise ValueError('Inverse/forward pose mapping mismatch')
    target_v = np.gradient(target_q, dt, axis=0)
    feedforward = np.zeros_like(target_q)
    last_q = None
    for step, q in enumerate(target_q):
        if last_q is not None and np.array_equal(last_q, q):
            feedforward[step] = feedforward[step - 1]
            continue
        closed(q)
        residual, _ = initial_preload(model, data, np.zeros(4), initial_spring_force=gas_spring_force(data.qpos[sq], np.zeros(2)))
        feedforward[step] = -residual + params['bias_nm'] + (q - qref) @ stiffness.T
        last_q = q.copy()
    closed(target_q[-1])
    target_knee = data.qpos[model.jnt_qposadr[knees]].copy()
    data.qpos[:] = initial
    data.qvel[:] = 0.
    import warp as wp
    import mujoco_warp as mw
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    wp.init()
    states, velocities, requests, applied, integral_saved = [], [], [], [], []
    integral = np.zeros(4)
    kp, kd, ki = 60., 2., 20.
    stopped_at = None
    max_gap = 0.
    measured_pose, measured_times = [], []
    began = time.monotonic()
    with wp.ScopedDevice('cuda:0'):
        gm = mw.put_model(model)
        gd = mw.put_data(model, data, nworld=1, nconmax=128, njmax=256)
        gd.qpos.assign(initial[None].astype(np.float32))
        gd.qvel.zero_()
        storage = ordered.buffers(gm, gd)
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mw.step1(gm, gd)
                ordered.apply(gd, storage)
                mw.step2(gm, gd)
        if np.any(gd.time.numpy()):
            raise ValueError('Capture advanced initial state')
        for step in range(steps + 1):
            q, v = gd.qpos.numpy()[0], gd.qvel.numpy()[0]
            length, angle, _ = leg_pose((q[mq] * pose.sign[:4] + pose.zero)[None])
            k = q[model.jnt_qposadr[knees]]
            invalid = (not np.isfinite(q).all() or not np.isfinite(v).all()
                       or length.min() < .14 or length.max() > .33
                       or np.any(k < model.jnt_range[knees, 0]) or np.any(k > model.jnt_range[knees, 1]))
            if invalid and stopped_at is None:
                stopped_at = step * dt
            error = target_q[step] - q[mq]
            raw = feedforward[step] + kp * error + kd * (target_v[step] - v[mv]) + integral
            torque = np.clip(raw, -40., 40.)
            if stopped_at is not None:
                torque[:] = 0.
            states.append(q.copy())
            velocities.append(v.copy())
            requests.append(torque.copy())
            integral_saved.append(integral.copy())
            load = np.zeros(model.nv, dtype=np.float32)
            load[mv] = -(np.asarray(params['bias_nm']) + (q[mq] - qref) @ stiffness.T)
            applied.append(load[mv].copy())
            if step % 5 == 0:
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                pairs = [pair for group in mirror.PIN_PAIRS for pair in group]
                pairs += [(s + '_spring_tip', s + '_gas_spring_lower') for s in ('left', 'right')]
                max_gap = max(max_gap, max(np.linalg.norm(data.site(x).xpos - data.site(y).xpos) for x, y in pairs))
                rotation = data.body('base_link').xmat.reshape(3, 3)
                lengths, angles = [], []
                for side in (0, 1):
                    delta = rotation.T @ (data.xanchor[model.joint(mirror.WHEELS[side]).id]
                                          - data.xanchor[model.joint(mirror.DRIVES[side][0]).id])
                    lengths.append(np.hypot(delta[0], delta[2]))
                    angles.append(np.arctan2(delta[0], -delta[2]))
                measured_pose.append([lengths, angles])
                measured_times.append(times[step])
            if step == steps:
                break
            can_integrate = (abs(raw) < 40.) | (raw * error < 0.)
            integral = np.clip(integral + ki * error * dt * can_integrate, -8., 8.)
            ctrl = np.zeros((1, model.nu), dtype=np.float32)
            ctrl[0, motor[:4]] = torque
            ctrl[0, spring] = gas_spring_force(q[sq], v[sv])
            gd.ctrl.assign(ctrl)
            gd.qfrc_applied.assign(load[None])
            wp.capture_launch(capture.graph)
            if step % 2000 == 0:
                print(json.dumps({'simulation_s': step * dt, 'elapsed_s': time.monotonic() - began}), flush=True)
    states, velocities, requests, measured_pose = map(np.asarray, (states, velocities, requests, measured_pose))
    actual_length, actual_angle, _ = leg_pose(states[:, mq] * pose.sign[:4] + pose.zero)
    final = times >= 13.
    report = dict(scope='Fixed-base model-control demo, not validation of fitted parameters or real hardware',
                  target_length_mm=args.length_mm, target_angle_deg=args.angle_deg,
                  controller=dict(kp=kp, kd=kd, ki=ki, torque_limit_nm=40., integral_limit_nm=8.,
                                  feedforward='Target-pose gravity, nominal spring and frozen local conservative load'),
                  final_length_mm=(actual_length[-1] * 1000).tolist(), final_angle_deg=np.degrees(actual_angle[-1]).tolist(),
                  last3s_max_length_error_mm=np.max(abs(actual_length[final] - args.length_mm / 1000), axis=0).__mul__(1000).tolist(),
                  last3s_max_angle_error_deg=np.degrees(np.max(abs(actual_angle[final] - np.deg2rad(args.angle_deg)), axis=0)).tolist(),
                  peak_request_nm=np.max(abs(requests), axis=0).tolist(), guard_stop_s=stopped_at,
                  max_pin_gap_mm=float(max_gap * 1000), target_knee_model_rad=target_knee.tolist(),
                  physics_dt_s=.0002, control_dt_s=dt, new_gpu_environment_steps=steps * 10,
                  frozen_parameters_sha256=digest, source_sha256=reference['source_sha256'],
                  reference_model_sha256=hashlib.sha256((args.reference / 'model.xml').read_bytes()).hexdigest(),
                  hardware_commands=False, parameter_fit_updates=0, policy_training_updates=0,
                  limitation='190mm/30deg lies outside identified posture neighborhood; model and controller share the same frozen load assumptions')
    np.savez_compressed(args.output / 'traces.npz', time_s=times, qpos=states, qvel=velocities,
                        target_q=target_q, target_v=target_v, feedforward_nm=feedforward,
                        request_nm=requests, integral_nm=np.asarray(integral_saved), applied_load_nm=np.asarray(applied),
                        actual_length_m=actual_length, actual_angle_rad=actual_angle,
                        target_length_m=target_length, target_angle_rad=target_angle,
                        geometry_time_s=np.asarray(measured_times), measured_geometry=measured_pose)
    (args.output / 'parameters.json').write_bytes(args.parameters.read_bytes())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(__file__, args.output / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for side, name in enumerate(('Left', 'Right')):
        axes[0].plot(times, actual_length[:, side] * 1000, label=name)
        axes[1].plot(times, np.degrees(actual_angle[:, side]), label=name)
    axes[0].plot(times, target_length[:, 0] * 1000, '--', color='gray', label='Target left')
    axes[1].plot(times, np.degrees(target_angle[:, 0]), '--', color='gray', label='Target left')
    for axis, name in enumerate(('LF', 'LR', 'RF', 'RR')):
        axes[2].plot(times, requests[:, axis], label=name)
    for ax in axes:
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Leg length (mm)')
    axes[1].set_ylabel('Leg angle (deg)')
    axes[2].set_ylabel('Motor request (Nm)')
    axes[2].set_xlabel('Simulation time (s)')
    fig.tight_layout()
    fig.savefig(args.output / 'control.png', dpi=145)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
