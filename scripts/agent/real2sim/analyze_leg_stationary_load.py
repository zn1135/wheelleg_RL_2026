"""Static load bounds from recorded, powered neutral-dial holds; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.optimize import linprog

from scripts.agent.real2sim import check_leg_static_load as static
from ct_mjlab.envs.chuanliantui import gas_spring


MODEL = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
SESSIONS = {
    'original':Path('artifacts/real2sim/h7-leg-capture-20261104/leg-neutral-motion-05'),
    'wide':Path('artifacts/real2sim/h7-response-validation-20261104-01/trial-01'),
    'powered':Path('artifacts/real2sim/h7-leg-soft-boundary-20261104-01/powered-01')}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def frames(session):
    identity = json.loads((session/'meta.json').read_text())['model']
    override = Path('artifacts/shared_validation/closed-chain-training-integration-20261104-01/before-source.xml')
    kwargs = {'model_path':override} if identity['sha256'] == digest(override) else {}
    meta, old_asset, config = static.mirror.load_session(session, **kwargs)
    parser = static.mirror.FrameParser(); result = []; rejected = 0
    for event in static.mirror.iter_session_events(session):
        if event['type'] != 'data': continue
        for frame in parser.feed(bytes.fromhex(event['data_hex'])):
            if int(frame[152]) != 6 or int(frame[109])&62 != 60 or abs(frame[91]) > 1e-7:
                rejected += 1; continue
            if (int(frame[29])&31 != 31 or frame[30] < 0 or frame[30] > 20
                    or frame[31] < 0 or frame[31] > 5 or np.min(frame[138:142]) < 0 or np.max(frame[138:142]) > 5):
                rejected += 1; continue
            row = static.mirror.telemetry_row(frame)
            if any(row.get(f'feedback_torque_{n}_firmware_nm') is None for n in static.mirror.MOTORS[:4]):
                rejected += 1; continue
            assert np.isfinite(frame[np.r_[10:18,22:26,132:136,146:150]]).all()
            result.append(frame.copy())
    return np.asarray(result), dict(session=str(session), stream_sha256=meta['stream_sha256'],
        recorded_model_sha256=meta['model']['sha256'], recorded_model_resolved=str(old_asset),
        config_sha256=digest(config), rejected_non_neutral_or_invalid_frames=rejected), config


def quiet_holds(raw):
    if not len(raw): return []
    time = np.array([static.mirror.read_u32(f[8:10])/1000 for f in raw])
    sequence = np.array([static.mirror.read_u32(f[6:8]) for f in raw], dtype=np.int64)
    split = np.flatnonzero((np.diff(time) > .025)|(np.diff(time) <= 0)|(np.diff(sequence) != 1))+1
    holds = []
    for group in np.split(np.arange(len(raw)), split):
        start = 0
        while start < len(group):
            stop = np.searchsorted(time[group], time[group[start]]+1., side='right')
            indices = group[start:stop]; start = stop
            if len(indices) < 90 or time[indices[-1]]-time[indices[0]] < .9: continue
            selected = raw[indices]
            if np.ptp(selected[:, 10:14], axis=0).max() > .003 or abs(selected[:, 14:18]).max() > .03: continue
            if np.ptp(selected[:, 132:136], axis=0).max() > .2: continue
            quat = selected[:, 22:26]; angular_span = 2*np.arccos(np.clip(abs(quat@quat[len(quat)//2]), 0., 1.)).max()
            if angular_span > np.deg2rad(.5): continue
            if holds:
                merged = np.r_[holds[-1], indices]
                if (indices[0] == holds[-1][-1]+1 and time[indices[0]]-time[holds[-1][-1]] < .025
                        and np.ptp(raw[merged, 10:14], axis=0).max() <= .003
                        and np.ptp(raw[merged, 132:136], axis=0).max() <= .2):
                    holds[-1] = merged; continue
            holds.append(indices)
    return holds


def force_interval(g, b, tau, capacity):
    lo, hi = 0., np.inf
    for gravity, arm, torque in zip(g, b, tau):
        if abs(arm) < 1e-10:
            if abs(gravity-torque) > capacity: return None
            continue
        a, z = sorted(((torque-gravity-capacity)/arm, (torque-gravity+capacity)/arm))
        lo = max(lo, a); hi = min(hi, z)
    return [float(lo), float(hi) if np.isfinite(hi) else None] if lo <= hi else None


def conditional_curve(rows, side, bounds=None):
    selected = [r for r in rows if r['knee_margin_rad'][side] > .02]
    if len(selected) < 3: return dict(interior_holds=len(selected), fitted=False)
    a = []; y = []
    for row in selected:
        for j in range(2*side, 2*side+2):
            b = row['spring_required_firmware_nm_per_n'][j][side]
            a.append([b, b*(row['compression_m'][side]-.005)])
            y.append(row['feedback_mean_nm'][j]-row['gravity_required_nm'][j])
    a = np.asarray(a); y = np.asarray(y)
    upper = np.vstack((np.c_[a, -np.ones(len(a))], np.c_[-a, -np.ones(len(a))]))
    bound = np.r_[y, -y]
    positivity = np.array([[-1., -(r['compression_m'][side]-.005), 0.] for r in selected])
    upper = np.vstack((upper, positivity)); bound = np.r_[bound, np.zeros(len(positivity))]
    bounds = [(0., None)]*3 if bounds is None else bounds
    fit = linprog([0., 0., 1.], A_ub=upper, b_ub=bound, bounds=bounds, method='highs')
    assert fit.success
    return dict(interior_holds=len(selected), fitted=True, reference_force_at_5mm_n=float(fit.x[0]),
        stiffness_n_m=float(fit.x[1]), minimum_uniform_unmodeled_static_torque_nm=float(fit.x[2]),
        design_rank=int(np.linalg.matrix_rank(a)),
        parameter_bounds=bounds,
        interpretation='Minimum residual static torque under nominal mass, linear nonnegative gas force and no external support; not independently identified spring/friction values')


def verify_report(directory, output):
    report = json.loads((directory/'report.json').read_text()); rows = report['holds']; count = 0
    for row in rows:
        for side in (0, 1):
            section = slice(2*side, 2*side+2)
            g = np.asarray(row['gravity_required_nm'])[section]
            b = np.asarray(row['spring_required_firmware_nm_per_n'])[section, side]
            tau = np.asarray(row['feedback_mean_nm'])[section]
            for capacity in (0., 1., 2., 4., 8.):
                interval = force_interval(g, b, tau, capacity)
                assert interval == row['conditional_force_intervals_n'][str(capacity)][side]
                oracle = linprog([0.], A_ub=np.r_[b, -b][:, None],
                    b_ub=np.r_[tau-g+capacity, g-tau+capacity], bounds=[(0., None)], method='highs')
                assert (interval is not None) == bool(oracle.success)
                if interval:
                    for endpoint in interval:
                        if endpoint is not None: assert abs(g+b*endpoint-tau).max() <= capacity+1e-7
                count += 1
    pairs = []
    for side in (0, 1):
        section = slice(2*side, 2*side+2)
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                dq = float(abs(np.asarray(a['angles_firmware_rad'])[section]-np.asarray(b['angles_firmware_rad'])[section]).max())
                dt = float(abs(np.asarray(a['feedback_mean_nm'])[section]-np.asarray(b['feedback_mean_nm'])[section]).max())
                if dq < .01 and dt > .5:
                    pairs.append(dict(side=side, first=a['id'], second=b['id'],
                        maximum_motor_angle_difference_deg=float(np.degrees(dq)), holding_torque_difference_nm=dt))
    active = [r for r in rows if max(abs(np.asarray(r['request_mean_nm']))) > .1]
    curves = {group:{str(side):dict(
        unbounded_nonnegative_curve=conditional_curve(values, side),
        old_search_box=conditional_curve(values, side, [(50., 250.), (0., 3500.), (0., None)]))
        for side in (0, 1)} for group, values in [('all_holds', rows), ('nonzero_request_holds', active)]}
    write(output/'verification.json', dict(source_report_sha256=digest(directory/'report.json'),
        interval_checks=count, interval_checks_passed=True, near_pose_pairs=pairs,
        curve_comparison=curves, nonzero_request_hold_count=len(active), new_physics_steps=0,
        interpretation='Conditional feasibility and static-balance checks, not unique physical parameter identification'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for side in (0, 1):
        x = np.array([r['compression_m'][side]*1000 for r in rows])
        tau = np.array([r['non_gravity_net_holding_torque_nm'][2*side:2*side+2] for r in rows])
        ax = axes[side, 0]
        ax.scatter(x, tau[:, 0], label='Long branch'); ax.scatter(x, -tau[:, 1], label='Negative short-branch torque', marker='x')
        ax.set_ylabel(('Left' if side == 0 else 'Right')+' net holding torque (Nm)')
        ax = axes[side, 1]
        for i, row in enumerate(rows):
            interval = row['conditional_force_intervals_n']['2.0'][side]
            if interval and interval[1] is not None: ax.plot([x[i], x[i]], interval, 'o-', color='tab:blue', alpha=.7, ms=3)
        grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(grid, np.clip(150*grid/90, 0, 150), label='Current shared linear model', color='tab:orange')
        ax.axhline(150, color='tab:green', ls='--', label='Constant150N hypothesis')
        ax.set_ylabel(('Left' if side == 0 else 'Right')+' conditional spring force (N)')
    for ax in axes.flat:
        ax.set_xlabel('Model spring compression (mm)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Static holds: blue force intervals assume unknown support <=2Nm per motor; not a spring calibration', fontsize=11)
    fig.tight_layout(); fig.savefig(output/'stationary-load-bounds.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-report', type=Path, help='Verify an existing report into a new directory; no raw replay')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if args.verify_report:
        verify_report(args.verify_report, args.output)
        shutil.copy2(__file__, args.output/Path(__file__).name)
        return
    model_hash = digest(MODEL); spring_hash = digest(Path(gas_spring.__file__))
    profile_path = Path('artifacts/real2sim/h7-leg-pose-fit-20261104-01/selected-parameters.json')
    profile = json.loads(profile_path.read_text()); all_rows = []; sources = {}; maxima = {}
    for name, session in SESSIONS.items():
        raw, source, config = frames(session); sources[name] = source; holds = quiet_holds(raw)
        pose = static.mirror.MirrorPose(MODEL, config); pose.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        rows = []
        for h, indices in enumerate(holds):
            f = raw[indices]; representative = f[len(f)//2]; angles = np.median(f[:, 10:14], axis=0)
            gravity, spring, passive, verify = static.pose_load(pose, representative, angles)
            for key, value in verify.items(): maxima[key] = max(maxima.get(key, 0.), value)
            sign = pose.sign[:4]; g = (gravity+passive)*sign; b = spring*sign[:, None]
            tau = np.mean(f[:, 146:150], axis=0); request = np.mean(f[:, 132:136], axis=0)
            sq = [pose.model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
            compression = np.maximum(.010-pose.data.qpos[sq], 0.)
            shared_force = gas_spring.gas_spring_force(pose.data.qpos[sq], np.zeros(2))
            fitted_force = np.clip(np.asarray(profile['spring_reference_force_n'])+
                np.asarray(profile['spring_stiffness_n_m'])*(compression-.005), 0., 600.)
            margin = []
            for knee in ('lf1', 'rf1'):
                joint = pose.model.joint(knee); q = pose.data.qpos[joint.qposadr[0]]
                margin.append(float(min(q-joint.range[0], joint.range[1]-q)))
            lengths = []
            for side in (0, 1):
                pair = angles[side*2:side*2+2]+[np.pi, 0.]
                ab = .21*np.column_stack((np.cos(pair), np.sin(pair))); delta = ab[1]-ab[0]
                normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
                lengths.append(float(np.linalg.norm(ab.mean(0)+np.sqrt(.25**2-(delta@delta)/4)*normal)))
            row = dict(id=f'{name}-{h:02d}', source=name, sample_count=len(f),
                board_start_s=static.mirror.read_u32(f[0,8:10])/1000, board_end_s=static.mirror.read_u32(f[-1,8:10])/1000,
                angles_firmware_rad=angles.tolist(), position_span_rad=np.ptp(f[:,10:14],axis=0).tolist(),
                velocity_peak_rad_s=abs(f[:,14:18]).max(0).tolist(), feedback_mean_nm=tau.tolist(), request_mean_nm=request.tolist(),
                feedback_request_difference_nm=(tau-request).tolist(), gravity_required_nm=g.tolist(),
                non_gravity_net_holding_torque_nm=(tau-g).tolist(), spring_required_firmware_nm_per_n=b.tolist(),
                compression_m=compression.tolist(), length_m=lengths, knee_margin_rad=margin,
                spring_pair_sum_error_m_rad=[float(abs(b[2*s:2*s+2,s].sum())) for s in (0,1)],
                common_non_spring_load_nm=[float((tau-g)[2*s:2*s+2].sum()) for s in (0,1)],
                candidate_required_static_support_nm={}, conditional_force_intervals_n={})
            assert max(row['spring_pair_sum_error_m_rad']) < 1e-6
            for label, force in [('shared_linear',shared_force),('constant150_hypothesis',np.full(2,150.)),('previous_pose_fit',fitted_force)]:
                row['candidate_required_static_support_nm'][label] = (g+b@force-tau).tolist()
            for capacity in (0., 1., 2., 4., 8.):
                row['conditional_force_intervals_n'][str(capacity)] = [force_interval(g[2*s:2*s+2], b[2*s:2*s+2,s], tau[2*s:2*s+2], capacity) for s in (0,1)]
            rows.append(row); all_rows.append(row)
        write(args.output/(name+'-holds.json'), rows)
        np.savez_compressed(args.output/(name+'-valid-neutral-frames.npz'), frames=raw)
        print(json.dumps(dict(session=name, valid_neutral_frames=len(raw), stationary_holds=len(rows))), flush=True)
    assert all_rows
    curves = {side:conditional_curve(all_rows, i) for i, side in enumerate(('left','right'))}
    result = dict(scope='Static combined-load and conditional spring/friction bounds; no hardware or dynamics',
        current_model_sha256=model_hash, gas_spring_source_sha256=spring_hash, candidate_profile_sha256=digest(profile_path),
        sources=sources, stationary_hold_count=len(all_rows), holds=all_rows, conditional_curve=curves,
        virtual_work_verification_maxima=maxima, new_physics_steps=0, policy_updates=0,
        thresholds=dict(minimum_hold_s=.9, position_span_rad=.003, recorded_speed_peak_rad_s=.03, request_span_nm=.2,
                        imu_angle_span_deg=.5, motor_age_ms=5., neutral_dial=True, powered_strategy=6, interior_knee_margin_rad=.02),
        limitations=['Static support includes motor/bearing friction, spring-seal stiction, possible stops or external support; these are not separated.',
                     'Nominal CAD masses/COMs and motor-reported torque are not independently calibrated.',
                     'Fixtures are assumed to support the body with wheels airborne and no hand contact, per original protocol; static forces alone cannot prove this.',
                     'Near-knee-limit holds are reported but excluded from curve inference; unknown gas-spring end stops may still matter.',
                     'Force intervals depend on an assumed static torque capacity, not a measured friction bound.',
                     'No damping or inertia can be identified from static holds; 100Hz snapshots do not rule out unrecorded transients.'])
    assert digest(MODEL) == model_hash and digest(Path(gas_spring.__file__)) == spring_hash
    write(args.output/'report.json', result)
    verify_report(args.output, args.output)
    shutil.copy2(__file__, args.output/Path(__file__).name)


if __name__ == '__main__': main()
