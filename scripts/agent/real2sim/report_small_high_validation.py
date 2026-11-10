"""Report fresh small/high free replay without refitting the frozen local-load model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_npz(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--frozen-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / 'report.json').read_text())
    quality = json.loads((args.paired / 'report.json').read_text())
    frozen = json.loads((args.frozen_fit / 'gpu-replay/report.json').read_text())
    params = json.loads((args.frozen_fit / 'parameters.json').read_text())
    expected_hash = hashlib.sha256((args.frozen_fit / 'parameters.json').read_bytes()).hexdigest()
    if report['effective_load_parameters_sha256'] != expected_hash or frozen['effective_load_parameters_sha256'] != expected_hash:
        raise ValueError('Parameters changed or wrong candidate selected')
    if not quality['complete_independent_sequence'] or quality['profile'] != 'small-high-v1':
        raise ValueError('Need complete fresh independent small/high data')
    saved = load_npz(args.run / 'replay.npz')
    paired = load_npz(args.paired / 'paired-inputs.npz')
    if hashlib.sha256((args.paired / 'paired-inputs.npz').read_bytes()).hexdigest() != report['paired_npz_sha256']:
        raise ValueError('Paired data changed')
    old_range = load_npz(args.frozen_fit / 'inverse.npz')['q']
    fixture = load_npz(args.run / 'fixture.npz')
    times = saved['time_s']
    active = (times >= 5) & (times < 25)
    hold = (times >= 2) & (times < 4)
    frequency = 2 + (times - 5) * 3 / 20
    chirp_time = times - 5
    ramp = np.clip(np.minimum(chirp_time, 20 - chirp_time) / 2, 0, 1)
    envelope = ramp ** 3 * (10 - 15 * ramp + 6 * ramp ** 2)
    phase = 2 * np.pi * (2 * chirp_time + 3 * chirp_time ** 2 / 40)
    error = np.degrees(saved['simulated'] - saved['real'])
    q0 = saved['real'][0]
    qref = np.asarray(params['reference_q_model_rad'])
    fit_range = np.stack((old_range.min(0), old_range.max(0)))
    rows = []
    for world, label in enumerate(('LF', 'LR', 'RF', 'RR')):
        segment = report['segments'][world]
        ix = np.flatnonzero((paired['stage'] == 2) & (paired['axis'] == world))
        if len(ix) != 15000 or segment['axis'] != world or segment['stage'] != 2:
            raise ValueError('Wrong independently captured segment')
        begin, end = paired['board_ns'][ix[[0, -1]]] / 1e9
        mask = (fixture['board_s'] >= begin) & (fixture['board_s'] <= end)
        orientation = fixture['quaternion_wxyz'][mask].astype(float)
        orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)
        g = Rotation.from_quat(orientation[:, [1, 2, 3, 0]]).inv().apply([0., 0., -1.])
        span = np.degrees(np.arccos(np.clip(g @ g[0], -1., 1.)))
        bins = []
        for low, high in ((2, 3), (3, 4), (4, 5)):
            selected = active & (frequency >= low) & (frequency < high)
            # 常量和慢漂移只用于提取基频振幅；不改变原始角误差或仿真状态。
            design = np.column_stack((np.ones(selected.sum()), times[selected] - times[selected].mean(),
                                      envelope[selected] * np.sin(phase[selected]), envelope[selected] * np.cos(phase[selected])))
            measured, simulated = [np.linalg.lstsq(design, saved[key][selected, world, world], rcond=None)[0]
                                   for key in ('real', 'simulated')]
            measured_amplitude = float(np.hypot(*measured[2:]))
            simulated_amplitude = float(np.hypot(*simulated[2:]))
            phase_error = float(np.degrees(np.arctan2(simulated[3], simulated[2]) - np.arctan2(measured[3], measured[2])))
            phase_error = (phase_error + 180) % 360 - 180
            bins.append(dict(frequency_hz=[low, high], samples=int(selected.sum()),
                             rmse_deg=np.sqrt(np.mean(error[selected, world] ** 2, axis=0)).tolist(),
                             actual_fundamental_amplitude_deg=float(np.degrees(measured_amplitude)),
                             simulated_fundamental_amplitude_deg=float(np.degrees(simulated_amplitude)),
                             simulated_over_actual_amplitude=simulated_amplitude / measured_amplitude,
                             simulated_minus_actual_phase_deg=phase_error))
        rows.append(dict(joint=label, full30s_rmse_deg=float(np.sqrt(np.mean(error[:, world, world] ** 2))),
                         moving20s_rmse_deg=float(np.sqrt(np.mean(error[active, world, world] ** 2))),
                         max_abs_error_deg=float(abs(error[:, world, world]).max()),
                         hold_bias_deg=np.median(error[hold, world], axis=0).tolist(),
                         coupled_joint_full_rmse_deg=np.sqrt(np.mean(error[:, world] ** 2, axis=0)).tolist(),
                         actual_peak_to_peak_deg=float(np.degrees(np.ptp(saved['real'][active, world, world]))),
                         initial_q_minus_reference_deg=np.degrees(q0[world] - qref).tolist(),
                         initial_outside_observed_lowfreq_range_deg=np.degrees(np.maximum(np.maximum(fit_range[0] - q0[world], q0[world] - fit_range[1]), 0.)).tolist(),
                         gravity_direction_span_deg=float(span.max()),
                         gyro_peak_rad_s=np.max(abs(fixture['gyro_rad_s'][mask]), axis=0).tolist(),
                         max_mirror_gap_s=float(np.diff(fixture['board_s'][mask]).max()), frequency_bins=bins))
    summary = dict(scope='Fresh independent small/high record; original local-load parameters frozen, no optimization',
                   parameter_sha256=expected_hash, original_fitted_parameters_unchanged=True,
                   data_quality_passed=True, model_guard_stop_s=report['first_model_guard_stop_s'],
                   joints=rows, balanced_full_rmse_deg=float(np.sqrt(np.mean([r['full30s_rmse_deg'] ** 2 for r in rows]))),
                   balanced_moving_rmse_deg=float(np.sqrt(np.mean([r['moving20s_rmse_deg'] ** 2 for r in rows]))),
                   actual_peak_speed_rad_s=np.max(abs(paired['v_firmware']), axis=0).tolist(),
                   peak_request_nm=np.max(abs(paired['request_firmware']), axis=0).tolist(),
                   new_gpu_environment_steps=0, policy_training_updates=0,
                   limitations=['Right-leg start differs from fitted posture; pose and frequency effects must be separated',
                                'Base translation unmeasured; fixed-base replay is conditional on fixture stability',
                                'No automatic claim of global fit or all-posture acceptance'])
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for world, row in enumerate(rows):
        ax = axes[world]
        ax.plot(times, np.degrees(saved['real'][:, world, world]), color='black', label='Real')
        ax.plot(times, np.degrees(saved['simulated'][:, world, world]), color='tab:blue', label='Frozen local-load model')
        ax.axvspan(5, 25, color='gray', alpha=.04)
        ax.set_ylabel(row['joint'] + ' (deg)')
        ax.set_title('Full/moving RMSE %.3f / %.3f deg; base gravity span %.2f deg' %
                     (row['full30s_rmse_deg'], row['moving20s_rmse_deg'], row['gravity_direction_span_deg']), fontsize=10)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel('Segment time (s)')
    fig.suptitle('New +/-0.05rad, 2 to5Hz: independent validation, no refitting or angle shifting')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for row in rows:
        axes[0].plot([2.5, 3.5, 4.5], [b['simulated_over_actual_amplitude'] for b in row['frequency_bins']], 'o-', label=row['joint'])
        axes[1].plot([2.5, 3.5, 4.5], [b['simulated_minus_actual_phase_deg'] for b in row['frequency_bins']], 'o-', label=row['joint'])
    axes[0].axhline(1, color='black', ls='--')
    axes[1].axhline(0, color='black', ls='--')
    axes[0].set_ylabel('Simulated / real oscillation amplitude')
    axes[1].set_ylabel('Simulated - real phase (deg)')
    axes[1].set_xlabel('Command frequency band center (Hz)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle('Frequency-band fundamental estimates; offsets removed only for this diagnostic')
    fig.tight_layout()
    fig.savefig(args.output / 'amplitude-and-phase.png', dpi=145)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
