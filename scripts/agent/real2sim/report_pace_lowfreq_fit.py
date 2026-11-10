"""Compare baseline/fitted PACE free replays on predefined train and heldout windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fitted', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old_report = json.loads((args.baseline / 'report.json').read_text())
    new_report = json.loads((args.fitted / 'report.json').read_text())
    fit = json.loads((args.fit / 'report.json').read_text())
    if any(old_report[key] != new_report[key] for key in ('source_sha256', 'raw_prefix_sha256', 'paired_npz_sha256')):
        raise ValueError('Different inputs or protected model sources')
    if new_report['effective_load_parameters_sha256'] != hashlib.sha256((args.fit / 'parameters.json').read_bytes()).hexdigest():
        raise ValueError('Different load candidate from the frozen fit')
    parameters = json.loads((args.fit / 'parameters.json').read_text())
    with np.load(args.fit / 'inverse.npz') as source:
        inverse = {key: source[key] for key in source.files}
    direct = (np.asarray(parameters['bias_nm'])
              + (inverse['q'] - parameters['reference_q_model_rad']) @ np.asarray(parameters['stiffness_nm_rad']).T
              + np.sign(inverse['v']) * parameters['friction_nm']
              + inverse['v'] * parameters['damping_nms_rad'])
    inverse_error = float(abs(direct - inverse['predicted_nm']).max())
    if inverse_error > 1e-10:
        raise ValueError('Fit parameter reconstruction mismatch')
    for name, window in (('train', fit['training_segment_time_s']), ('heldout', fit['heldout_segment_time_s'])):
        expected = ((inverse['time_s'] >= window[0]) & (inverse['time_s'] < window[1]))[:, None] & (abs(inverse['v']) > .03)
        if not np.array_equal(expected, inverse[name]):
            raise ValueError('Fit/heldout sample mask changed')
    if np.any(inverse['train'] & inverse['heldout']):
        raise ValueError('Fit/heldout samples overlap')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.fitted / 'model.xml'))
    asset = next(path for path in new_report['source_sha256'] if path.endswith('.xml'))
    original = mujoco.MjModel.from_xml_path(asset)
    for axis, name in enumerate(('lf0', 'lf00', 'rf0', 'rf00')):
        idx, prior = model.joint(name).dofadr[0], original.joint(name).dofadr[0]
        if abs(model.dof_frictionloss[idx] - parameters['friction_nm'][axis]) > 1e-5:
            raise ValueError('Native friction does not match candidate')
        if abs(model.dof_damping[idx] - original.dof_damping[prior] - parameters['damping_nms_rad'][axis]) > 1e-5:
            raise ValueError('Native damping does not match candidate')
    with np.load(args.baseline / 'replay.npz') as source:
        old = {key: source[key] for key in source.files}
    with np.load(args.fitted / 'replay.npz') as source:
        new = {key: source[key] for key in source.files}
    for key in ('time_s', 'real', 'target'):
        if not np.array_equal(old[key], new[key]):
            raise ValueError('Target, truth or time changed')
    if any(stop is not None for stop in new_report['first_model_guard_stop_s']):
        raise ValueError('Candidate replay triggered its model guard')
    rows = []
    errors = [np.degrees(saved['simulated'] - saved['real']) for saved in (old, new)]
    for axis, label in enumerate(('LF', 'LR', 'RF', 'RR')):
        row = dict(joint=label)
        for name, window in (('train', fit['training_segment_time_s']), ('heldout', fit['heldout_segment_time_s']), ('full', [0, 30])):
            mask = (new['time_s'] >= window[0]) & (new['time_s'] < window[1])
            values = [error[mask, axis, axis] for error in errors]
            rmse = [float(np.sqrt(np.mean(value ** 2))) for value in values]
            row[name] = dict(baseline_rmse_deg=rmse[0], fitted_rmse_deg=rmse[1],
                             fitted_max_error_deg=float(abs(values[1]).max()),
                             improvement_percent=100 * (1 - rmse[1] / rmse[0]))
        rows.append(row)
    heldout = np.array([[row['heldout'][name] for row in rows] for name in ('baseline_rmse_deg', 'fitted_rmse_deg')])
    summary = dict(scope='Same-input four30s free GPU replays; heldout time not used in parameter fitting',
                   inverse_parameter_reconstruction_max_error_nm=inverse_error,
                   fit_heldout_masks_verified=True, native_resistance_parameters_verified=True,
                   joints=rows, heldout_joint_balanced_rmse_deg=np.sqrt(np.mean(heldout ** 2, axis=1)).tolist(),
                   candidate_all_coupled_joints_full_rmse_deg=np.sqrt(np.mean(errors[1] ** 2, axis=0)).tolist(),
                   sample_uses_unshifted_angles=True, physical_zero_changed=False,
                   new_gpu_environment_steps=0, policy_training_updates=0,
                   limitations=['Same recording and local posture range; not independent trajectory/general motion acceptance',
                                'Effective local load and resistance, not identified physical spring/motor components'])
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for axis, row in enumerate(rows):
        ax = axes[axis]
        for window, color, label in ((fit['training_segment_time_s'], 'gray', 'Fit window'),
                                      (fit['heldout_segment_time_s'], 'green', 'Heldout window')):
            ax.axvspan(*window, color=color, alpha=.07, label=label)
        ax.plot(new['time_s'], np.degrees(new['real'][:, axis, axis]), color='black', label='Real')
        ax.plot(old['time_s'], np.degrees(old['simulated'][:, axis, axis]), color='tab:orange', alpha=.7, label='Nominal')
        ax.plot(new['time_s'], np.degrees(new['simulated'][:, axis, axis]), color='tab:blue', label='Fitted')
        ax.set_ylabel(row['joint'] + ' (deg)')
        ax.grid(alpha=.25)
        ax.set_title('Heldout RMSE: %.3f -> %.3f deg' % (row['heldout']['baseline_rmse_deg'], row['heldout']['fitted_rmse_deg']), fontsize=10)
    axes[0].legend(fontsize=8, ncol=5)
    axes[-1].set_xlabel('Segment time (s)')
    fig.suptitle('Measured PACE prechecks: local load/resistance fit, no angle offset removal')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=140)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
