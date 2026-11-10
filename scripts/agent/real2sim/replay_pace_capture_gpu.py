"""Replay paired PACE precheck segments with recorded targets and simulated PD; no USB."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from h7_output_capture import CaptureParser
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from scripts.agent.real2sim.fit_leg_length_response import leg_pose


def inputs(session, paired):
    """Recheck the immutable raw prefix against every supplied paired sample."""
    audit = json.loads((paired / 'report.json').read_text())
    meta = json.loads((session / 'meta.json').read_text())
    paths = [Path(meta[key]['path']) for key in ('model', 'machine_config')]
    for key, path in zip(('model', 'machine_config'), paths):
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta[key]['sha256']:
            raise ValueError('Recorded ' + key + ' source has changed')
    raw, frames = CaptureParser(), mirror.FrameParser()
    digest = hashlib.sha256()
    remaining = audit['stream_prefix_bytes']
    saved_frames, targets, state, control, geometry = [], [], {}, {}, {}
    with (session / 'stream.jsonl').open('rb') as source:
        while remaining:
            line = source.readline(remaining)
            if not line or not line.endswith(b'\n'):
                raise ValueError('Raw prefix truncated')
            remaining -= len(line)
            digest.update(line)
            event = json.loads(line)
            if event['type'] != 'data':
                continue
            chunk = bytes.fromhex(event['data_hex'])
            saved_frames.extend(frames.feed(chunk))
            for packet in raw.feed(chunk):
                for row in packet['records']:
                    if row[21] != 7 or row[1] == 2:
                        continue
                    if row[1] == 5:
                        if row[20] & 1 and not (row[20] & 4 and row[22] == 3):
                            targets.append(row)
                    else:
                        {1: control, 3: state, 4: geometry}[row[1]][row[4]] = row
    if digest.hexdigest() != audit['stream_prefix_sha256']:
        raise ValueError('Raw prefix identity mismatch')
    with np.load(paired / 'paired-inputs.npz') as source:
        arrays = {key: source[key] for key in source.files}
    expected = dict(
        id=[r[4] for r in targets], elapsed_ms=[r[14] for r in targets],
        board_ns=[r[2] for r in targets], stage=[r[22] for r in targets],
        axis=[r[23] for r in targets], target_firmware=[r[6:10] for r in targets],
        q_firmware=[state[r[4]][6:10] for r in targets],
        v_firmware=[state[r[4]][10:14] for r in targets],
        request_firmware=[control[r[4]][6:10] for r in targets],
        feedback_firmware=[geometry[r[4]][6:10] for r in targets],
        kp=[r[10] for r in targets], kd=[r[11] for r in targets], limit=[r[12] for r in targets])
    if 'target_flags' in arrays:
        expected['target_flags'] = [r[20] for r in targets]
    if any(not np.array_equal(arrays[key], value) for key, value in expected.items()):
        raise ValueError('Paired arrays do not match raw records')
    if (audit['record_sequence_gaps'] or audit['queue_drop_increase_during_sweep']
            or any(r[5] or r[20] & 2 for r in targets)):
        raise ValueError('Faulted, stopped or incomplete sweep interval')
    reconstructed = np.clip(arrays['kp'][:, None] * (arrays['target_firmware'] - arrays['q_firmware'])
                            - arrays['kd'][:, None] * arrays['v_firmware'],
                            -arrays['limit'][:, None], arrays['limit'][:, None])
    error = float(abs(reconstructed - arrays['request_firmware']).max())
    if error > .001:
        raise ValueError('Recorded PD reconstruction mismatch')
    ft = np.array([mirror.read_u32(f[8:10]) / 1000 for f in saved_frames])
    return arrays, np.asarray(saved_frames), ft, paths, audit, error


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, help='Explicit local load/resistance candidate; diagnostic copy only')
    parser.add_argument('--segment', choices=('prechecks', 'formal-left', 'small-high'), default='prechecks',
                        help='Prechecks or the recorded interrupted formal LF segment; never synthesize missing samples')
    parser.add_argument('--recorded-gravity', action='store_true',
                        help='Replay measured fixture gravity direction; does not impose base angular inertia')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    a, frames, ft, (asset, config), audit, pd_error = inputs(args.session, args.paired)
    parameters = json.loads(args.parameters.read_text()) if args.parameters else None
    gas_candidate = parameters is not None and parameters.get('model') == 'gas_spring_linear_with_joint_resistance'
    if gas_candidate:
        expected = {'model', 'spring_reference_compression_m', 'spring_reference_force_n',
                    'spring_stiffness_n_m', 'spring_damping_ns_m', 'spring_force_limit_n',
                    'friction_nm', 'damping_nms_rad'}
        if set(parameters) != expected or parameters['spring_reference_compression_m'] != .005:
            raise ValueError('Unsupported gas-spring candidate')
        for field in expected - {'model', 'spring_reference_compression_m'}:
            value = np.asarray(parameters[field])
            count = 4 if field in ('friction_nm', 'damping_nms_rad') else 2
            if value.shape != (count,) or not np.isfinite(value).all() or value.min() < 0:
                raise ValueError('Invalid gas-spring parameter ' + field)
    elif parameters is not None:
        expected = {'model', 'reference_q_model_rad', 'bias_nm', 'stiffness_nm_rad',
                    'friction_nm', 'damping_nms_rad'}
        if set(parameters) != expected or parameters['model'] != 'local_conservative_load_plus_coulomb_viscous':
            raise ValueError('Unsupported local load candidate')
        for field in ('reference_q_model_rad', 'bias_nm', 'friction_nm', 'damping_nms_rad'):
            value = np.asarray(parameters[field])
            if value.shape != (4,) or not np.isfinite(value).all():
                raise ValueError('Invalid load parameter ' + field)
        stiffness = np.asarray(parameters['stiffness_nm_rad'])
        if stiffness.shape != (4, 4) or not np.isfinite(stiffness).all() or not np.allclose(stiffness, stiffness.T):
            raise ValueError('Load stiffness must be a finite symmetric matrix')
        if min(parameters['friction_nm'] + parameters['damping_nms_rad']) < 0:
            raise ValueError('Resistance must be nonnegative')
    pose = mirror.MirrorPose(asset, config)
    segments = ([(0, axis) for axis in range(4)] if args.segment == 'prechecks' else
                [(2, axis) for axis in range(4)] if args.segment == 'small-high' else [(1, 0)])
    if args.segment == 'small-high' and ('target_flags' not in a or not np.all(a['target_flags'] & 4)):
        raise ValueError('Independent high-frequency profile identity missing')
    indices = [np.flatnonzero((a['stage'] == stage) & (a['axis'] == axis)) for stage, axis in segments]
    for (stage, axis), ix in zip(segments, indices):
        start = axis * 30000 if args.segment == 'small-high' else (stage * 4 + axis) * 30000
        if not len(ix) or not np.array_equal(a['elapsed_ms'][ix], start + np.arange(len(ix)) * 2):
            raise ValueError('Missing beginning or discontinuous recorded segment')
        if args.segment in ('prechecks', 'small-high') and len(ix) != 15000:
            raise ValueError('Need four complete 30-second precheck segments')
        if len(ix) > 15000:
            raise ValueError('Recorded segment exceeds expected30s')
    nworld, nsteps = len(indices), len(indices[0])
    first_board = min(a['board_ns'][ix[0]] for ix in indices) / 1e9
    last_board = max(a['board_ns'][ix[-1]] for ix in indices) / 1e9
    observed = (ft >= first_board - .03) & (ft <= last_board + .03)
    np.savez_compressed(args.output / 'fixture.npz', board_s=ft[observed],
                        quaternion_wxyz=frames[observed, 22:26], gyro_rad_s=frames[observed, 26:29],
                        source_flags=frames[observed, 29], imu_age_ms=frames[observed, 30])
    frame0 = frames[np.searchsorted(ft, a['board_ns'][indices[0][0]] / 1e9, side='right') - 1]
    quat = frame0[22:26].astype(float)
    if not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1) > .01:
        raise ValueError('Invalid recorded base orientation')
    root = ET.parse(asset).getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((asset.parent / compiler.get('meshdir')).resolve()))
    base = root.find("worldbody/body[@name='base_link']")
    base.remove(base.find('freejoint'))
    base.set('pos', '0 0 .8')
    base.set('quat', ' '.join(map(str, quat)))
    if gas_candidate:
        for side, maximum in zip(('left', 'right'), parameters['spring_force_limit_n']):
            actuator = root.find(f"actuator/motor[@name='{side}_gas_spring_motor']")
            actuator.set('ctrlrange', f'0 {maximum}')
            actuator.set('forcerange', f'0 {maximum}')
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='.0002', integrator='implicitfast', jacobian='dense',
                         solver='Newton', iterations='100', tolerance='1e-10',
                         ls_iterations='100', ls_tolerance='.0001')
    xml = ET.tostring(root, encoding='unicode')
    (args.output / 'model.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    mq = np.array([model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]])
    mv = np.array([model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]])
    motor = np.array([model.actuator(n + '_motor').id for n in mirror.MOTORS])
    sq = np.array([model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')])
    sv = np.array([model.joint(s + '_spring2_joint').dofadr[0] for s in ('left', 'right')])
    spring = np.array([model.actuator(s + '_gas_spring_motor').id for s in ('left', 'right')])
    if parameters is not None:
        # 原生静摩擦约束和隐式阻尼；负载校正只作用于本次诊断副本。
        model.dof_frictionloss[mv] = parameters['friction_nm']
        model.dof_damping[mv] += parameters['damping_nms_rad']
        model.dof_solref[mv] = [.002, 1.]
        model.dof_solimp[mv] = [.999, .9999, .001, .5, 2.]
        mujoco.mj_saveLastXML(str(args.output / 'model.xml'), model)
    knees = np.array([model.joint(n).id for n in ('lf1', 'rf1')])
    initial_q, initial_v = [], []
    for ix in indices:
        index = ix[0]
        frame = frames[np.searchsorted(ft, a['board_ns'][index] / 1e9, side='right') - 1]
        if abs(mirror.read_u32(frame[8:10]) / 1000 - a['board_ns'][index] / 1e9) > .03:
            raise ValueError('No fresh mirror frame at segment start')
        pose.apply(frame, 0.)
        q = pose.sign[:4] * (a['q_firmware'][index] - pose.zero)
        v = pose.sign[:4] * a['v_firmware'][index]
        for side in (0, 1):
            if not pose._solve_leg(side, q[2 * side:2 * side + 2]):
                raise ValueError('Initial closed-chain solution failed')
        q0 = pose.data.qpos.copy()
        for side in (0, 1):
            if not pose._solve_leg(side, q[2 * side:2 * side + 2] + 1e-5 * v[2 * side:2 * side + 2]):
                raise ValueError('Initial passive velocity solution failed')
        full_v = np.zeros(pose.model.nv)
        mujoco.mj_differentiatePos(pose.model, full_v, 1e-5, q0, pose.data.qpos)
        for joint in range(model.njnt):
            source = pose.model.joint(model.joint(joint).name)
            data.qpos[model.jnt_qposadr[joint]] = q0[source.qposadr[0]]
            data.qvel[model.jnt_dofadr[joint]] = full_v[source.dofadr[0]]
        initial_q.append(data.qpos.copy())
        initial_v.append(data.qvel.copy())
    target = np.stack([pose.sign[:4] * (a['target_firmware'][ix] - pose.zero) for ix in indices], axis=1)
    real = np.stack([pose.sign[:4] * (a['q_firmware'][ix] - pose.zero) for ix in indices], axis=1)
    kp = np.stack([a['kp'][ix] for ix in indices], axis=1)
    kd = np.stack([a['kd'][ix] for ix in indices], axis=1)
    limit = np.stack([a['limit'][ix] for ix in indices], axis=1)
    recorded_gravity = None
    if gas_candidate or args.recorded_gravity:
        from scipy.spatial.transform import Rotation
        quaternions = frames[:, 22:26].astype(float).copy()
        for i in range(1, len(quaternions)):
            if quaternions[i] @ quaternions[i - 1] < 0:
                quaternions[i] *= -1
        rotations = []
        r0 = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        for ix in indices:
            bt = a['board_ns'][ix] / 1e9
            orientation = np.column_stack([np.interp(bt, ft, quaternions[:, j]) for j in range(4)])
            orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)
            matrices = Rotation.from_quat(orientation[:, [1, 2, 3, 0]]).as_matrix()
            body_gravity = np.einsum('tji,j->ti', matrices, model.opt.gravity)
            rotations.append(body_gravity @ r0.T)
        recorded_gravity = np.stack(rotations, axis=1).astype(np.float32)
    import warp as wp
    import mujoco_warp as mw
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    wp.init()
    simulated, speeds, torques, loads, spring_forces, slide_positions, slide_velocities = [], [], [], [], [], [], []
    blocked = np.zeros(nworld, dtype=bool)
    first_stop = [None] * nworld
    closure_max = np.zeros(nworld)
    started = time.monotonic()
    with wp.ScopedDevice('cuda:0'):
        gm = mw.put_model(model)
        if recorded_gravity is not None:
            # 固定机身副本中旋转重力方向，复现已知台架朝向；不覆写模拟关节状态。
            gm.opt.gravity = wp.array(recorded_gravity[0], dtype=wp.vec3)
        gd = mw.put_data(model, data, nworld=nworld, nconmax=128, njmax=256)
        gd.qpos.assign(np.asarray(initial_q, dtype=np.float32))
        gd.qvel.assign(np.asarray(initial_v, dtype=np.float32))
        storage = ordered.buffers(gm, gd)
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mw.step1(gm, gd)
                ordered.apply(gd, storage)
                mw.step2(gm, gd)
        if np.any(gd.time.numpy()):
            raise ValueError('Capture unexpectedly advanced the initial state')
        for step in range(nsteps):
            q, v = gd.qpos.numpy(), gd.qvel.numpy()
            lengths = leg_pose(q[:, mq] * pose.sign[:4] + pose.zero)[0]
            knee_q = q[:, model.jnt_qposadr[knees]]
            bad = (~np.isfinite(q).all(1) | ~np.isfinite(v).all(1)
                   | (lengths.min(1) < .14) | (lengths.max(1) > .33)
                   | (knee_q < model.jnt_range[knees, 0]).any(1)
                   | (knee_q > model.jnt_range[knees, 1]).any(1))
            for world in np.flatnonzero(bad & ~blocked):
                first_stop[world] = float(step * .002)
            blocked |= bad
            ctrl = np.zeros((nworld, model.nu), dtype=np.float32)
            ctrl[:, motor[:4]] = np.clip(kp[step, :, None] * (target[step] - q[:, mq])
                                         - kd[step, :, None] * v[:, mv],
                                         -limit[step, :, None], limit[step, :, None])
            ctrl[np.ix_(blocked, motor)] = 0.
            if recorded_gravity is not None:
                gm.opt.gravity.assign(recorded_gravity[step])
            if gas_candidate:
                force = (np.asarray(parameters['spring_reference_force_n'])
                         + (np.maximum(.01 - q[:, sq], 0.) - .005) * parameters['spring_stiffness_n_m']
                         - v[:, sv] * parameters['spring_damping_ns_m'])
                ctrl[:, spring] = np.clip(force, 0., parameters['spring_force_limit_n'])
            else:
                ctrl[:, spring] = gas_spring_force(q[:, sq], v[:, sv])
            applied = np.zeros((nworld, model.nv), dtype=np.float32)
            if parameters is not None and not gas_candidate:
                correction = (np.asarray(parameters['bias_nm'])
                              + (q[:, mq] - parameters['reference_q_model_rad']) @ stiffness.T)
                applied[:, mv] = -correction
                gd.qfrc_applied.assign(applied)
            simulated.append(q[:, mq].copy())
            speeds.append(v[:, mv].copy())
            torques.append(ctrl[:, motor[:4]].copy())
            loads.append(applied[:, mv].copy())
            spring_forces.append(ctrl[:, spring].copy())
            slide_positions.append(q[:, sq].copy())
            slide_velocities.append(v[:, sv].copy())
            if step % 5 == 0:
                for world in range(nworld):
                    data.qpos[:] = q[world]
                    mujoco.mj_kinematics(model, data)
                    pairs = [pair for group in mirror.PIN_PAIRS for pair in group]
                    pairs += [(s + '_spring_tip', s + '_gas_spring_lower') for s in ('left', 'right')]
                    closure_max[world] = max(closure_max[world], max(np.linalg.norm(data.site(x).xpos - data.site(y).xpos) for x, y in pairs))
            gd.ctrl.assign(ctrl)
            wp.capture_launch(capture.graph)
            if step % 2500 == 0:
                print(json.dumps({'segment_s': step * .002, 'blocked': blocked.tolist(),
                                  'elapsed_s': time.monotonic() - started}), flush=True)
    simulated = np.asarray(simulated)
    error = simulated - real
    report = dict(scope=('Four independent30s real segments, each initialized from its measured pose/velocity; not a continuous120s replay or parameter fit' if args.segment in ('prechecks', 'small-high')
                         else 'Recorded interrupted formal LF segment; fixed parameters, no fitting, missing motion not synthesized'),
                  recording_profile=args.segment,
                  segments=[dict(stage=stage, axis=axis, samples=len(ix), first_elapsed_ms=int(a['elapsed_ms'][ix[0]]),
                                 last_elapsed_ms=int(a['elapsed_ms'][ix[-1]])) for (stage, axis), ix in zip(segments, indices)],
                  raw_prefix_sha256=audit['stream_prefix_sha256'], raw_prefix_bytes=audit['stream_prefix_bytes'],
                  paired_npz_sha256=hashlib.sha256((args.paired / 'paired-inputs.npz').read_bytes()).hexdigest(),
                  recorded_pd_max_error_nm=pd_error, first_model_guard_stop_s=first_stop,
                  joint_rmse_deg=np.degrees(np.sqrt(np.mean(error ** 2, axis=0))).tolist(),
                  joint_max_error_deg=np.degrees(np.max(abs(error), axis=0)).tolist(),
                  max_pin_error_mm=(closure_max * 1000).tolist(),
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (asset, config)},
                  physics_dt_s=.0002, control_dt_s=.002, new_gpu_environment_steps=nsteps * 10 * nworld,
                  spring=('Explicit fitted gas curve, CAD10mm preload; evaluated every2ms' if gas_candidate else 'Shared linear0..150N, CAD10mm preload, damping50; evaluated every2ms'),
                  recorded_gravity_used=recorded_gravity is not None,
                  hardware_commands=False, policy_training_updates=0,
                  effective_load_parameters=parameters,
                  effective_load_parameters_sha256=(hashlib.sha256(args.parameters.read_bytes()).hexdigest() if args.parameters else None),
                  limitations=['Nominal load parameters were not identified', 'Stops remain in all-window error metrics',
                               ('Measured IMU gravity direction replayed; base angular inertial effects not imposed' if recorded_gravity is not None else 'IMU orientation fixed to initial recorded quaternion for each world'),
                               'USB/driver/CAN delays not separately modeled; feedback quantization retained in truth'])
    np.savez_compressed(args.output / 'replay.npz', time_s=np.arange(nsteps) * .002,
                        real=real, simulated=simulated, target=target,
                        velocity=np.asarray(speeds), request_nm=np.asarray(torques),
                        applied_load_nm=np.asarray(loads), spring_force_n=np.asarray(spring_forces),
                        slide_position_m=np.asarray(slide_positions), slide_velocity_m_s=np.asarray(slide_velocities),
                        gravity_m_s2=(recorded_gravity if recorded_gravity is not None else np.broadcast_to(model.opt.gravity, (nsteps, nworld, 3))))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(__file__, args.output / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nworld, 1, figsize=(11, max(4, nworld * 2.5)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    ts = np.arange(nsteps) * .002
    for world, (_, axis) in enumerate(segments):
        label = ('LF', 'LR', 'RF', 'RR')[axis]
        axes[world].plot(ts, np.degrees(real[:, world, axis]), label='Real')
        axes[world].plot(ts, np.degrees(simulated[:, world, axis]), label='GPU candidate' if parameters else 'GPU nominal model')
        axes[world].plot(ts, np.degrees(target[:, world, axis]), '--', alpha=.6, label='Recorded target')
        axes[world].set_ylabel(label + ' (deg)')
        axes[world].grid(alpha=.25)
        axes[world].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel('Segment time (s)')
    fig.tight_layout()
    fig.savefig(args.output / 'joint-response.png', dpi=140)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
