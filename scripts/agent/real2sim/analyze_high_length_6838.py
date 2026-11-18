"""Read-only audit of uploaded6838 high-height logs; offline inference, no physics/hardware/training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import periodogram
from scipy.spatial.transform import Rotation
import torch

from scripts.agent.real2sim.analyze_usb_pd500 import decode, NAMES

ROOT = Path(__file__).resolve().parents[3]
LEGS = ('lf', 'lr', 'rf', 'rr')
MOTORS = (*LEGS, 'lw', 'rw')
ORDER = ('lf', 'lr', 'lw', 'rf', 'rr', 'rw')
PHYSICAL = ('h7_us', 'sequence', 'quat_w', 'quat_x', 'quat_y', 'quat_z',
    'pitch_rad', 'roll_rad', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s',
    'left_length_m', 'right_length_m', 'left_angle_rad', 'right_angle_rad',
    'hip_to_wheel_height_m', 'cmd_leg_length_m', 'cmd_vx_m_s', 'cmd_yaw_rad_s')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path, names=NAMES):
    with path.open() as f:
        rows = csv.reader(f)
        header = next(rows)
        indices = [header.index(n) for n in names]
        return np.asarray([[float(row[i]) for i in indices] for row in rows])


def columns(a, names=NAMES):
    return {n: a[:, i] for i, n in enumerate(names)}


def valid(d):
    return ((d['rl_engaged'] == 1) & (d['pd_valid'] == 1) & (d['obs_valid'] == 1)
            & (d['warmup'] == 0))


def spectral(y, fs):
    f, p = periodogram(y, fs=fs, axis=0)
    band = (f >= 5) & (f <= 20)
    return dict(band_rms=np.sqrt(np.trapezoid(p[band], f[band], axis=0)).tolist(),
                peak_5to20_Hz=f[band][np.argmax(p[band], axis=0)].tolist())


def physical_metrics(a):
    d = columns(a, PHYSICAL)
    q = Rotation.from_quat(np.stack([d['quat_' + n] for n in ('x', 'y', 'z', 'w')], 1))
    dq = (q[:-1].inv() * q[1:]).as_rotvec() / (np.diff(d['h7_us'])[:, None] / 1e6)
    g = np.stack([d['gyro_' + n + '_rad_s'] for n in ('x', 'y', 'z')], 1)
    # 日志Euler列名不等于标准XYZ轴名，另用四元数推导角速度，避免按名字判断方向。
    euler = q.as_euler('xyz')
    return dict(rows=len(a),
        length_mean_m=[float(d[k].mean()) for k in ('left_length_m', 'right_length_m')],
        length_std_m=[float(d[k].std()) for k in ('left_length_m', 'right_length_m')],
        gyro_rms_rad_s=np.sqrt(np.mean(g*g, axis=0)).tolist(),
        gyro_vs_quaternion_derivative_corr=[float(np.corrcoef(dq[:, i], g[1:, i])[0, 1]) for i in range(3)],
        xyz_euler_std_deg=np.rad2deg(euler.std(0)).tolist(),
        csv_pitch_vs_xyz_euler_rmse_rad=np.sqrt(np.mean((euler-d['pitch_rad'][:, None])**2, 0)).tolist(),
        csv_roll_vs_xyz_euler_rmse_rad=np.sqrt(np.mean((euler-d['roll_rad'][:, None])**2, 0)).tolist(),
        gyro_spectrum=spectral(g, 200.))


def metrics(a, metadata):
    d = columns(a)
    t = d['h7_us'] / 1e6
    age = (d['h7_us'] - 1000*d['obs_publish_ms']) / 1000
    starts = np.flatnonzero(np.r_[True, np.diff(d['obs_sequence']) != 0])
    motors = {}
    for i, name in enumerate(MOTORS):
        wheel = i >= 4
        error = d['target_'+name]-d['current_'+name]
        vel, tau = d['velocity_rad_s_'+name], d['pd_model_nm_'+name]
        kp = metadata['control']['wheel_stiffness' if wheel else 'leg_stiffness']
        kd = metadata['control']['wheel_damping' if wheel else 'leg_damping']
        limit = 3.9 if wheel else 40.
        predicted = np.clip(kp*error-kd*vel, -limit, limit)
        free = abs(tau) < limit-1e-5
        design = error[:, None] if wheel else np.c_[error, -vel]
        fitted = np.linalg.lstsq(design[free], tau[free], rcond=None)[0]
        sign = (1, 1, -1, -1, 1, -1)[i]
        motors[name] = dict(fitted_PD=fitted.tolist(),
            PD_max_error_nm=float(abs(predicted-tau).max()),
            firmware_mapping_max_error_nm=float(abs(d['request_firmware_nm_'+name]-sign*tau).max()),
            saturation_fraction=float(np.mean(~free)), requested_torque_peak_nm=float(abs(tau).max()),
            requested_torque_rms_nm=float(np.sqrt(np.mean(tau*tau))),
            target_std=float(d['target_'+name].std()), measured_std=float(d['current_'+name].std()),
            velocity_rms_rad_s=float(np.sqrt(np.mean(vel*vel))),
            target_spectrum=spectral(d['target_'+name], 200.))
    return dict(rows=len(a), time_h7_s=[float(t[0]), float(t[-1])],
        dt_us_quantiles=np.quantile(np.diff(d['h7_us']), [0, .5, .95, 1]).tolist(),
        sequence_breaks=int(np.count_nonzero(np.diff(d['sequence']) != 1)),
        observation_sequence_breaks=int(np.count_nonzero(np.diff(d['obs_sequence'][starts]) != 1)),
        age_ms_quantiles=np.quantile(age, [0, .5, .95, 1]).tolist(),
        board_drop_increment=int(d['board_drops'][-1]-d['board_drops'][0]),
        faults=np.unique(d['fault']).astype(int).tolist(),
        command_min=np.stack([d['obs_6']/2, d['obs_7']/.25, d['obs_8']/5], 1).min(0).tolist(),
        command_max=np.stack([d['obs_6']/2, d['obs_7']/.25, d['obs_8']/5], 1).max(0).tolist(), motors=motors)


def policy_inputs(a):
    d = columns(a)
    start = np.flatnonzero(np.r_[True, np.diff(d['obs_sequence']) != 0])
    assert np.all(np.diff(d['obs_sequence'][start]) == 1)
    obs = np.stack([d[f'obs_{i}'][start] for i in range(25)], 1).astype(np.float32)
    # 略去开始四帧和最后一帧，确保历史完整且有下一帧接受动作可交叉核对。
    ids = np.arange(4, len(obs)-1)
    history = np.stack([obs[i-4:i+1].ravel() for i in ids])
    return d, start, ids, obs, obs[ids], history


def replay(policy, metadata, a):
    d, start, ids, obs, current, history = policy_inputs(a)
    target = np.stack([d['target_'+n][start] for n in ORDER], 1)
    actual = target.copy()
    actual[:, [0, 1, 3, 4]] = (actual[:, [0, 1, 3, 4]] - metadata['control']['nominal_leg_positions'])/.5
    actual[:, [2, 5]] /= 10.
    with torch.inference_mode():
        pred, latent = policy.act_inference(torch.from_numpy(current), torch.from_numpy(history))
    err = pred.numpy()-actual[ids]
    target_check = abs(actual[:-1]-obs[1:, 19:25]).max()
    assert target_check < 1e-5 and abs(err).max() < 1e-4
    return dict(frames=len(ids), max_action_error=float(abs(err).max()),
        rms_action_error=float(np.sqrt(np.mean(err*err))),
        next_accepted_action_mapping_error=float(target_check)), (current, history, d['h7_us'][start][ids]/1e6, pred.numpy(), latent.numpy())


def sensitivity(policy, inputs, stable_range, jitter_range):
    current, history, time, original, _ = inputs
    stable = (time >= stable_range[0]) & (time <= stable_range[1])
    jitter = (time >= jitter_range[0]) & (time <= jitter_range[1])
    base = np.asarray(spectral(original[jitter][:, [0, 1, 3, 4]]*.5, 100.)['band_rms'])
    result = {}
    groups = dict(gyro_all=range(3), gyro_x=[0], gyro_y=[1], gyro_z=[2],
        gravity=range(3, 6), leg_position=range(9, 13),
        leg_velocity=[13, 14, 16, 17], wheel_velocity=[15, 18], previous_action=range(19, 25))
    with torch.inference_mode():
        for name, indices in groups.items():
            for mode in ('both', 'current_only', 'history_only') if name.startswith('gyro_') else ('both',):
                slots = list(indices)
                a, b = current.copy(), history.reshape(-1, 5, 25).copy()
                median = np.median(current[stable][:, slots], axis=0)
                if mode != 'history_only': a[:, slots] = median
                if mode != 'current_only': b[:, :, slots] = median
                pred, _ = policy.act_inference(torch.from_numpy(a), torch.from_numpy(b.reshape(-1, 125)))
                amp = np.asarray(spectral(pred.numpy()[jitter][:, [0, 1, 3, 4]]*.5, 100.)['band_rms'])
                result[name+'/'+mode] = (amp/base).tolist()
        for height in (.22, .26, .28, .29):
            a, b = current.copy(), history.reshape(-1, 5, 25).copy()
            a[:, 8] = height*5
            b[:, :, 8] = height*5
            pred, _ = policy.act_inference(torch.from_numpy(a), torch.from_numpy(b.reshape(-1, 125)))
            amp = np.asarray(spectral(pred.numpy()[jitter][:, [0, 1, 3, 4]]*.5, 100.)['band_rms'])
            result[f'height{height}/both'] = (amp/base).tolist()
    return dict(baseline_5to20_target_rms_rad=base.tolist(), amplitude_ratios=result,
        scope='Replace only indicated recorded inputs with stable medians, recompute frozen network; not a closed-loop rollout or proof that filtering cures shaking')


def plot(out, focus, physical, stable_range, jitter_range, sensitivity_result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d, ph = columns(focus), columns(physical, PHYSICAL)
    zero = d['h7_us'][0]/1e6
    t, tp = d['h7_us']/1e6-zero, ph['h7_us']/1e6-zero
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, d['obs_8']/5, label='Height command')
    axes[0].plot(tp, ph['left_length_m'], label='Left geometric leg length')
    axes[0].plot(tp, ph['right_length_m'], label='Right geometric leg length')
    axes[0].set_ylabel('m (different references)')
    axes[1].plot(tp, ph['gyro_x_rad_s'], label='Body gyro X')
    axes[1].set_ylabel('rad/s')
    axes[2].plot(t, np.rad2deg(d['target_lr']), label='LR requested angle')
    axes[2].plot(t, np.rad2deg(d['current_lr']), label='LR measured angle')
    axes[2].set_ylabel('deg')
    for n in ('lr', 'rf'): axes[3].plot(t, d['pd_model_nm_'+n], label=n+' torque request', lw=.8)
    axes[3].set_ylabel('Nm (request)')
    axes[3].set_xlabel('Seconds since first valid RL sample')
    for ax in axes:
        ax.axvspan(stable_range[0]-zero, stable_range[1]-zero, alpha=.12, color='green')
        ax.axvspan(jitter_range[0]-zero, jitter_range[1]-zero, alpha=.12, color='red')
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax.grid(alpha=.2)
    axes[-1].set_xlim(0., t[-1])
    fig.suptitle('6838 free-standing: stable green / shaking red after height drop and return to0.30m')
    fig.tight_layout()
    fig.savefig(out/'onset.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, n in zip(axes, LEGS):
        ax.plot(t, np.rad2deg(d['target_'+n]), label='Requested angle', lw=.8)
        ax.plot(t, np.rad2deg(d['current_'+n]), label='Measured angle', lw=.8)
        ax.set_ylabel(n+' deg')
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel('Seconds since first valid RL sample')
    axes[-1].set_xlim(0., t[-1])
    fig.tight_layout()
    fig.savefig(out/'all-legs.png', dpi=150)
    plt.close(fig)
    labels = ['gyro_x/both', 'gyro_y/both', 'gyro_z/both', 'leg_velocity/both',
              'previous_action/both', 'height0.26/both']
    ratios = np.array([sensitivity_result['amplitude_ratios'][k] for k in labels])
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(LEGS): ax.bar(np.arange(len(labels))+(i-1.5)*.18, ratios[:, i], .18, label=name)
    ax.axhline(1., color='black', ls='--')
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15)
    ax.set_ylabel('5-20Hz target amplitude / original')
    ax.set_title('Frozen-network input substitution only; no closed-loop or hardware intervention')
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out/'input-sensitivity.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    a = p.parse_args()
    out = a.experiment.resolve()
    assert not (out/'analysis.json').exists()
    raw = out/'raw/高腿长抖腿'
    complete, contrast = raw/'6838高腿长完整录制', raw/'6838高腿长对照_162432'
    bins, physical = {}, {}
    with (complete/'full-session.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['source']
            if row['frame'] == 'pd': bins.setdefault(name, []).append([float(row[n]) for n in NAMES])
            elif row['frame'] == 'physical': physical.setdefault(name, []).append([float(row[n]) for n in PHYSICAL])
    torch.set_num_threads(1)
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    checkpoint = ROOT/'artifacts/fudan_ppo_walk/models/20261116_model6838_regions_scan_slope5to20_mu08/model_final.pt'
    identity = sha(checkpoint)
    assert identity == '8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
    policy, metadata = load_policy(checkpoint)
    decoding, episodes, focus, focus_physical, inputs = {}, [], None, None, None
    for name, csv_rows in bins.items():
        x, sessions, audit = decode(complete/name)
        assert np.array_equal(x, np.asarray(csv_rows)), name
        assert audit['crc_or_header_errors'] == 0
        audit.update(csv_exact=True, rows=len(x), sessions=np.unique(sessions).tolist())
        decoding[name] = audit
        d = columns(x)
        active = valid(d)
        if not active.any(): continue
        for reset in np.unique(d['reset_count'][active]):
            selected = active & (d['reset_count'] == reset)
            v = x[selected]
            measured = metrics(v, metadata)
            infer, data = replay(policy, metadata, v)
            episodes.append(dict(source=name, reset=int(reset), measured=measured, inference=infer))
            if int(reset) == 10:
                focus, focus_physical, inputs = v, np.asarray(physical[name]), data
    assert len(episodes) == 5 and focus is not None
    stable, jitter = read_csv(contrast/'stable.csv'), read_csv(contrast/'jitter.csv')
    all_pd = read_csv(contrast/'full-pd.csv')
    same_source = next(name for name in bins if name.startswith('records-20261117-162432'))
    assert np.array_equal(all_pd, np.asarray(bins[same_source]))
    lookup = {row[0]: row for row in all_pd}
    for data in (stable, jitter): assert all(np.array_equal(row, lookup[row[0]]) for row in data)
    ranges = [tuple(data[[0, -1], 0]/1e6) for data in (stable, jitter)]
    comparisons = {}
    for name, data, interval in zip(('stable', 'jitter'), (stable, jitter), ranges):
        pd_result = metrics(data, metadata)
        select = (focus_physical[:, 0]/1e6 >= interval[0]) & (focus_physical[:, 0]/1e6 <= interval[1])
        comparisons[name] = dict(PD=pd_result, physical=physical_metrics(focus_physical[select]))
    influence = sensitivity(policy, inputs, *ranges)
    plot(out, focus, focus_physical, *ranges, influence)
    fd = columns(focus)
    transition = ((fd['h7_us']/1e6 > ranges[0][1]) & (fd['h7_us']/1e6 < ranges[1][0])
                  & (fd['obs_8']/5 < .2999))
    assert transition.any()
    transition_report = dict(height_dip_h7_s=(fd['h7_us'][transition][[0, -1]]/1e6).tolist(),
        minimum_height_command_m=float((fd['obs_8'][transition]/5).min()),
        return_to_030_h7_s=ranges[1][0], observed_stable_duration_s=ranges[0][1]-ranges[0][0])
    assert sha(checkpoint) == identity
    report = dict(checkpoint_sha256=identity, user_condition='自由站立', raw_audit=decoding,
        CSV_slices_match_raw=True, episodes=episodes, focus=comparisons, sensitivity=influence,
        height_transition=transition_report,
        learned_model_unchanged=True, optimizer_updates=0, physics_steps=0,
        limits=['No independent hardware firmware/build identity in upload',
                'Quaternion and gyro share an IMU, so agreement is not independent ground truth',
                'CSV pitch/roll labels use a different axis naming than standard quaternion XYZ',
                'Offline input substitution is not a proposed on-hardware controller change'])
    (out/'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(episodes=len(episodes), raw_PD_rows=sum(v['rows'] for v in decoding.values()),
        replay_max_error=max(v['inference']['max_action_error'] for v in episodes),
        focus_target_std_rad={n:[comparisons[n]['PD']['motors'][m]['target_std'] for m in LEGS] for n in comparisons},
        gyro_x_amplitude_ratios=influence['amplitude_ratios']['gyro_x/both']), indent=2))


if __name__ == '__main__':
    main()
