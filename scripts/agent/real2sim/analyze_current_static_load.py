"""Current USB static torque/leg-force diagnostic from an immutable journal tail; no stepping."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from scripts.agent.real2sim import check_leg_static_load as static
from scripts.agent.real2sim.analyze_tilt_length import geometry
from ct_mjlab.envs.chuanliantui import gas_spring


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hold-s', type=float, default=3., help='Required continuous stationary duration; use5 for calibration points')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.session / 'meta.json').read_text())
    model, config = [static.ROOT / meta[key]['project_path'] for key in ('model', 'machine_config')]
    for key, path in [('model', model), ('machine_config', config)]:
        assert digest(path) == meta[key]['sha256']
    source = args.session / 'stream.jsonl'
    end = source.stat().st_size
    with source.open('rb') as file:
        file.seek(max(0, end - 32 * 1024 * 1024))
        if file.tell():
            file.readline()
        begin = file.tell()
        content = file.read(end - begin)
    content = content[:content.rfind(b'\n') + 1]
    tail = args.output / 'source-tail.jsonl'
    tail.write_bytes(content)
    decoder = static.mirror.FrameParser()
    frames = []
    for line in content.splitlines():
        event = json.loads(line)
        if event['type'] == 'data':
            frames.extend(decoder.feed(bytes.fromhex(event['data_hex'])))
    raw = np.asarray(frames)
    assert len(raw), 'No valid mirror frames'
    # 同一镜像中的关节角/反馈力矩；过滤源龄，不以串口有数据代替反馈新鲜。
    valid = ((raw[:, 29].astype(int) & 31) == 31) & (raw[:, 2] == 0)
    valid &= (raw[:, 138:142].min(axis=1) >= 0) & (raw[:, 138:142].max(axis=1) <= 5)
    valid &= (raw[:, 144] >= 0) & (raw[:, 144] <= 20)
    valid &= np.isfinite(raw[:, np.r_[10:18, 22:26, 132:136, 146:150]]).all(axis=1)
    raw = raw[valid]
    time = np.array([static.mirror.read_u32(f[8:10]) / 1000 for f in raw])
    seq = np.array([static.mirror.read_u32(f[6:8]) for f in raw], dtype=np.int64)
    split = np.flatnonzero((np.diff(seq) != 1) | (np.diff(time) <= 0) | (np.diff(time) > .025)
                           | (np.diff(raw[:, 152]) != 0) | (np.diff(raw[:, 109]) != 0)) + 1
    selected = None
    for group in reversed(np.split(np.arange(len(raw)), split)):
        if len(group) < 100:
            continue
        indices = group[time[group] >= time[group[-1]] - args.hold_s]
        if len(indices) < 100 or np.ptp(time[indices]) < args.hold_s - .1:
            continue
        f = raw[indices]
        if (np.ptp(f[:, 10:14], axis=0).max() > .003 or abs(f[:, 14:18]).max() > .03
                or np.ptp(f[:, 146:150], axis=0).max() > .2 or abs(f[:, 91]).max() > 1e-7):
            continue
        quat = f[:, 22:26]
        angular_span = 2 * np.arccos(np.clip(abs(quat @ quat[len(quat) // 2]), 0., 1.)).max()
        if angular_span > np.deg2rad(.5):
            continue
        selected = indices
        break
    assert selected is not None, f'No {args.hold_s}s stationary/fresh/continuous neutral-dial window in recent tail'
    f = raw[selected]
    representative = f[len(f) // 2]
    q = np.median(f[:, 10:14], axis=0)
    feedback = np.mean(f[:, 146:150], axis=0)
    request = np.mean(f[:, 132:136], axis=0)
    pose = static.mirror.MirrorPose(model, config)
    pose.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    gravity, spring, passive, checks = static.pose_load(pose, representative, q)
    sign = pose.sign[:4]
    gravity, spring, passive = gravity * sign, spring * sign[:, None], passive * sign
    sq = [pose.model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')]
    slider = pose.data.qpos[sq].copy()
    compression = np.maximum(gas_spring.SPRING_INITIAL_COMPRESSION - slider, 0.)
    shared_force = gas_spring.gas_spring_force(slider, np.zeros(2))
    required = gravity + passive + spring @ shared_force
    constant150 = gravity + passive + spring @ np.full(2, 150.)
    import re
    block = config.read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1].split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0]
    lu, lg = [float(re.search(r'\.' + name + r'\s*=\s*([\d.]+)f', block)[1]) for name in ('leg_lu', 'leg_lg')]
    sides = []
    for side, label in enumerate(('left', 'right')):
        pair = q[2 * side:2 * side + 2]
        length, angle = geometry(pair[None], lu, lg)
        jac = np.empty((2, 2))
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = 1e-6
            lengths, angles = geometry(np.stack((pair + delta, pair - delta)), lu, lg)
            jac[:, j] = [(lengths[0] - lengths[1]) / 2e-6, (angles[0] - angles[1]) / 2e-6]
        section = slice(2 * side, 2 * side + 2)
        equivalent = np.linalg.solve(jac.T, feedback[section])
        gravity_eq = np.linalg.solve(jac.T, gravity[section])
        residual_eq = np.linalg.solve(jac.T, (required - feedback)[section])
        arm = spring[section, side]
        net = feedback[section] - gravity[section] - passive[section]
        candidate_force = float(arm @ net / (arm @ arm))
        candidate_residual = net - arm * candidate_force
        knee = pose.model.joint(('lf1', 'rf1')[side])
        knee_q = float(pose.data.qpos[knee.qposadr[0]])
        sides.append(dict(side=label, length_m=float(length[0]), leg_angle_rad=float(angle[0]),
                          spring_compression_m=float(compression[side]), model_spring_force_n=float(shared_force[side]),
                          spring_required_nm_per_n=arm.tolist(), leg_jacobian=jac.tolist(),
                          motor_equivalent_axial_force_n=float(equivalent[0]),
                          motor_equivalent_swing_torque_nm=float(equivalent[1]),
                          gravity_equivalent_axial_force_n=float(gravity_eq[0]),
                          model_residual_equivalent_axial_force_n=float(residual_eq[0]),
                          model_residual_equivalent_swing_torque_nm=float(residual_eq[1]),
                          unconstrained_equivalent_spring_force_n=candidate_force,
                          equivalent_spring_remaining_residual_nm=candidate_residual.tolist(),
                          knee_q_rad=knee_q, knee_range_rad=knee.range.tolist(),
                          knee_margin_rad=float(min(knee_q - knee.range[0], knee.range[1] - knee_q)),
                          axial_mapping_common_rotation_error=float(abs(jac[0].sum()))))
    report = dict(scope='Current-pose static virtual-work diagnostic; no physics integration or hardware commands',
                  source=dict(session=str(args.session), byte_begin=begin, byte_end=begin + len(content),
                              tail_sha256=digest(tail), metadata=meta),
                  model_sha256=digest(model), config_sha256=digest(config),
                  board_time_s=[float(time[selected[0]]), float(time[selected[-1]])], samples=len(f),
                  requested_stationary_duration_s=args.hold_s,
                  strategy=int(representative[152]), output_flags=int(representative[109]),
                  motor_order=['left_front', 'left_rear', 'right_front', 'right_rear'],
                  motor_positions_rad=q.tolist(), position_span_rad=np.ptp(f[:, 10:14], axis=0).tolist(),
                  peak_velocity_rad_s=abs(f[:, 14:18]).max(axis=0).tolist(),
                  feedback_mean_nm=feedback.tolist(), feedback_std_nm=np.std(f[:, 146:150], axis=0).tolist(),
                  request_mean_nm=request.tolist(), max_motor_age_ms=float(f[:, 138:142].max()),
                  max_imu_age_ms=float(f[:, 144].max()),
                  gravity_required_nm=gravity.tolist(), passive_required_nm=passive.tolist(),
                  model_spring_required_nm=(spring @ shared_force).tolist(),
                  model_total_required_nm=required.tolist(), model_minus_feedback_nm=(required - feedback).tolist(),
                  hypothetical_constant150_required_nm=constant150.tolist(),
                  verification=checks, legs=sides, environment_steps=0,
                  limitations=['Support/contact state requires operator evidence; no mechanical-stop contact inferred',
                               'Motor torque is driver-reported, not an independent shaft-torque measurement',
                               'Equivalent axial force is the motor contribution; not measured ground/support force',
                               'Shared spring curve and CAD mass are assumptions; 150N rating is not an actual force measurement',
                               'A single posture cannot distinguish spring force, static friction, external support and torque bias'])
    np.savez_compressed(args.output / 'static-frames.npz', frames=f, leg_spring_jacobian=spring)
    for source_path in (Path(__file__), Path(static.__file__), Path(static.mirror.__file__), Path(gas_spring.__file__), model, config):
        destination = args.output / 'source' / source_path.resolve().relative_to(static.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(4)
    for shift, value, label in [(-.25, feedback, 'Driver feedback'), (0., gravity, 'Model gravity only'),
                                (.25, required, 'Model gravity + current spring')]:
        axes[0].bar(x + shift, value, width=.25, label=label)
    axes[0].set_xticks(x, ['LF', 'LR', 'RF', 'RR'])
    axes[0].set_ylabel('Motor torque (Nm), firmware coordinates')
    axes[0].legend(fontsize=8)
    axes[1].bar([0, 1], [s['motor_equivalent_axial_force_n'] for s in sides])
    axes[1].set_xticks([0, 1], ['Left', 'Right'])
    axes[1].set_ylabel('Motor-only equivalent axial force (N)')
    for ax in axes:
        ax.axhline(0., color='black', lw=.7)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Static same-pose comparison; external support not yet confirmed')
    fig.tight_layout()
    fig.savefig(args.output / 'static-forces.png', dpi=150)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
