"""Independently verify gravity-separated spring fit and its saved GPU replay; no steps."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def read_npz(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--local-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fit = json.loads((args.fit / 'report.json').read_text())
    parameters = json.loads((args.fit / 'parameters.json').read_text())
    gpu = json.loads((args.fit / 'gpu-replay/report.json').read_text())
    if gpu['effective_load_parameters_sha256'] != hashlib.sha256((args.fit / 'parameters.json').read_bytes()).hexdigest():
        raise ValueError('GPU replay uses a different spring candidate')
    for path, digest in fit['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Protected source has changed')
    dynamic, static = [read_npz(args.fit / name) for name in ('inverse.npz', 'static.npz')]
    force_error = torque_error = 0.
    for values, moving in ((dynamic, True), (static, False)):
        slide_v = values['slide_v'] if moving else np.zeros_like(values['compression'])
        force = np.clip(np.asarray(parameters['spring_reference_force_n'])
                        + (values['compression'] - .005) * parameters['spring_stiffness_n_m']
                        - slide_v * parameters['spring_damping_ns_m'], 0., parameters['spring_force_limit_n'])
        torque = np.sum(values['basis'] * force[:, None, :], axis=2)
        if moving:
            torque += np.sign(values['v']) * parameters['friction_nm'] + values['v'] * parameters['damping_nms_rad']
        force_error = max(force_error, float(abs(force - values['force']).max()))
        torque_error = max(torque_error, float(abs(torque - values['predicted']).max()))
        if np.any(values['train'] & values['heldout']):
            raise ValueError('Fitted and heldout samples overlap')
    if max(force_error, torque_error) > 1e-9:
        raise ValueError('Gas fit reconstruction mismatch')
    cache = read_npz(args.fit / 'verified-inputs.npz')
    saved = read_npz(args.fit / 'gpu-replay/replay.npz')
    frames, ft = cache['frames'], cache['ft']
    first = np.flatnonzero((cache['stage'] == 0) & (cache['axis'] == 0))[0]
    f0 = frames[np.searchsorted(ft, cache['board_ns'][first] / 1e9, side='right') - 1]
    r0 = Rotation.from_quat(f0[[23, 24, 25, 22]]).as_matrix()
    quat = frames[:, 22:26].astype(float).copy()
    for i in range(1, len(quat)):
        if quat[i] @ quat[i - 1] < 0:
            quat[i] *= -1
    model = mujoco.MjModel.from_xml_path(str(args.fit / 'gpu-replay/model.xml'))
    gravity = []
    for axis in range(4):
        ix = np.flatnonzero((cache['stage'] == 0) & (cache['axis'] == axis))
        bt = cache['board_ns'][ix] / 1e9
        orientation = np.column_stack([np.interp(bt, ft, quat[:, j]) for j in range(4)])
        rotation = Rotation.from_quat(orientation[:, [1, 2, 3, 0]]).as_matrix()
        gravity.append(np.einsum('ij,tjk,k->ti', r0, rotation.transpose(0, 2, 1), model.opt.gravity))
    gravity_error = float(abs(np.stack(gravity, axis=1) - saved['gravity_m_s2']).max())
    if gravity_error > 1e-5:
        raise ValueError('Recorded gravity direction replay mismatch')
    for side, prefix in enumerate(('left', 'right')):
        actuator = model.actuator(prefix + '_gas_spring_motor').id
        if model.actuator_forcerange[actuator, 1] != parameters['spring_force_limit_n'][side]:
            raise ValueError('Candidate spring force clipped by a different native limit')
    old, local = [read_npz(p / 'replay.npz') for p in (args.baseline, args.local_fit)]
    for source in (old, local):
        if any(not np.array_equal(source[key], saved[key]) for key in ('time_s', 'real', 'target')):
            raise ValueError('Replay targets or truth changed')
    heldout = (saved['time_s'] >= 15.5) & (saved['time_s'] < 24.5)
    metrics = []
    for source in (old, local, saved):
        error = np.degrees(source['simulated'] - source['real'])
        per_joint = np.sqrt(np.mean(error[heldout] ** 2, axis=0)).diagonal()
        metrics.append(dict(joint_rmse_deg=per_joint.tolist(), balanced_rmse_deg=float(np.sqrt(np.mean(per_joint ** 2)))))
    missing_gravity_force = []
    for side in (0, 1):
        b = static['basis'][:, 2 * side:2 * side + 2, side]
        g = static['gravity'][:, 2 * side:2 * side + 2]
        missing_gravity_force.append(np.sum(b * g, axis=1) / np.sum(b * b, axis=1))
    result = dict(passed=True, spring_fit_force_reconstruction_error_n=force_error,
                  spring_fit_torque_reconstruction_error_nm=torque_error, recorded_gravity_error_m_s2=gravity_error,
                  heldout_metrics=dict(nominal=metrics[0], local_load=metrics[1], gas_spring=metrics[2]),
                  elastic_force_at30mm_n=(np.asarray(parameters['spring_reference_force_n']) + .025 * np.asarray(parameters['spring_stiffness_n_m'])).tolist(),
                  hypothetical_omitted_gravity_force_overestimate_n=[[float(np.min(x)), float(np.max(x))] for x in missing_gravity_force],
                  previous_gravity_already_included=True, new_gpu_environment_steps=0,
                  limitation='Hypothetical omitted-gravity bias is a projection diagnostic, not evidence that earlier fits omitted gravity. Spring/resistance separation remains conditional.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for side, name in enumerate(('Left', 'Right')):
        ax = axes[0, side]
        b = static['basis'][:, 2 * side:2 * side + 2, side]
        y = static['target'][:, 2 * side:2 * side + 2]
        projected = np.sum(b * y, axis=1) / np.sum(b * b, axis=1)
        x = static['compression'][:, side] * 1000
        grid = np.linspace(x.min(), x.max(), 150)
        curve = parameters['spring_reference_force_n'][side] + parameters['spring_stiffness_n_m'][side] * (grid / 1000 - .005)
        ax.scatter(x, projected, s=20, alpha=.7, label='Static projection; includes stiction')
        ax.plot(grid, curve, color='tab:red', label='New elastic curve')
        ax.plot(grid, np.clip(150 * grid / 90, 0, 150), '--', color='gray', label='Old nominal curve')
        ax.axhline(150, color='tab:green', ls=':', label='150N rated reference')
        ax.set_title(name + ' spring')
        ax.set_xlabel('Model compression (mm)')
        ax.set_ylabel('Axial force (N)')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        ax = axes[1, side]
        angle = np.degrees(static['old_angle'][:, side])
        ax.scatter(angle, static['gravity'][:, 2 * side], label='Front motor')
        ax.scatter(angle, static['gravity'][:, 2 * side + 1], label='Rear motor')
        ax.set_title(name + ': per-pose gravity retained')
        ax.set_xlabel('Leg tilt (deg)')
        ax.set_ylabel('Gravity torque, model coordinates (Nm)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Gravity-separated gas-spring refit; no physical calibration changes')
    fig.tight_layout()
    fig.savefig(args.output / 'spring-and-gravity.png', dpi=145)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for axis, label in enumerate(('LF', 'LR', 'RF', 'RR')):
        ax = axes[axis]
        ax.plot(saved['time_s'], np.degrees(saved['real'][:, axis, axis]), color='black', label='Real')
        ax.plot(local['time_s'], np.degrees(local['simulated'][:, axis, axis]), alpha=.7, label='Previous local load fit')
        ax.plot(saved['time_s'], np.degrees(saved['simulated'][:, axis, axis]), label='Gas spring + resistance')
        ax.axvspan(15.5, 24.5, color='green', alpha=.07)
        ax.set_ylabel(label + ' (deg)')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel('Segment time (s)')
    fig.tight_layout()
    fig.savefig(args.output / 'response-comparison.png', dpi=140)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
