"""Independently check saved PACE GPU replay targets, PD and measured inputs; no steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
from real2sim_huananhu import load_mapping


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / 'report.json').read_text())
    for source, digest in report['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError('Replay source has changed')
    if hashlib.sha256((args.paired / 'paired-inputs.npz').read_bytes()).hexdigest() != report['paired_npz_sha256']:
        raise ValueError('Paired input identity mismatch')
    config = next(Path(p) for p in report['source_sha256'] if p.endswith('machine_config.c'))
    sign, zero = load_mapping(config)
    with np.load(args.run / 'replay.npz') as source:
        saved = {key: source[key] for key in source.files}
    with np.load(args.paired / 'paired-inputs.npz') as source:
        paired = {key: source[key] for key in source.files}
    if any(not np.isfinite(value).all() for value in saved.values()):
        raise ValueError('Nonfinite replay data')
    initial_error = float(abs(saved['simulated'][0] - saved['real'][0]).max())
    pd_error = target_error = truth_error = 0.
    rows = []
    times = saved['time_s']
    if not np.array_equal(times, np.arange(len(times)) * .002):
        raise ValueError('Replay time grid changed')
    segments = report.get('segments', [dict(stage=0, axis=axis) for axis in range(4)])
    for world, segment in enumerate(segments):
        axis = segment['axis']
        label = ('LF', 'LR', 'RF', 'RR')[axis]
        ix = np.flatnonzero((paired['stage'] == segment['stage']) & (paired['axis'] == axis))
        if len(ix) != len(times):
            raise ValueError('Replay length does not match recorded segment')
        target = sign[:4] * (paired['target_firmware'][ix] - zero)
        truth = sign[:4] * (paired['q_firmware'][ix] - zero)
        target_error = max(target_error, float(abs(target - saved['target'][:, world]).max()))
        truth_error = max(truth_error, float(abs(truth - saved['real'][:, world]).max()))
        expected = np.clip(paired['kp'][ix, None] * (target - saved['simulated'][:, world])
                           - paired['kd'][ix, None] * saved['velocity'][:, world],
                           -paired['limit'][ix, None], paired['limit'][ix, None])
        stop = report['first_model_guard_stop_s'][world]
        if stop is not None:
            expected[times >= stop] = 0.
        pd_error = max(pd_error, float(abs(expected - saved['request_nm'][:, world]).max()))
        error = saved['simulated'][:, world, axis] - truth[:, axis]
        hold = (times >= 2) & (times < 4)
        moving = (times >= 5) & (times < 25)
        bias = float(np.median(error[hold]))
        rows.append(dict(joint=label, full_window_rmse_deg=float(np.degrees(np.sqrt(np.mean(error ** 2)))),
                         holding_bias_deg=float(np.degrees(bias)),
                         moving_rmse_after_holding_bias_removal_deg=float(np.degrees(np.sqrt(np.mean((error[moving] - bias) ** 2))))))
    if initial_error > 1e-6 or target_error > 1e-12 or truth_error > 1e-12 or pd_error > 1e-4:
        raise ValueError('Saved GPU replay failed target/truth/PD reconstruction')
    if np.max(report['max_pin_error_mm']) >= .1:
        raise ValueError('Replay closure exceeds0.1mm')
    load_error = 0.
    spring_error = 0.
    parameters = report.get('effective_load_parameters')
    gas_candidate = parameters is not None and parameters.get('model') == 'gas_spring_linear_with_joint_resistance'
    if gas_candidate:
        expected_force = np.clip(np.asarray(parameters['spring_reference_force_n'])
                                 + (np.maximum(.01 - saved['slide_position_m'], 0.) - .005) * parameters['spring_stiffness_n_m']
                                 - saved['slide_velocity_m_s'] * parameters['spring_damping_ns_m'],
                                 0., parameters['spring_force_limit_n'])
        spring_error = float(abs(expected_force - saved['spring_force_n']).max())
        load_error = float(abs(saved['applied_load_nm']).max())
        if spring_error > 1e-4 or load_error > 1e-8:
            raise ValueError('Gas force mismatch or unexpected independent joint load')
    elif parameters is not None:
        stiffness = np.asarray(parameters['stiffness_nm_rad'])
        if not np.allclose(stiffness, stiffness.T) or min(parameters['friction_nm'] + parameters['damping_nms_rad']) < 0:
            raise ValueError('Nonconservative stiffness or negative resistance')
        expected_load = -(np.asarray(parameters['bias_nm'])
                          + (saved['simulated'] - parameters['reference_q_model_rad']) @ stiffness.T)
        load_error = float(abs(expected_load - saved['applied_load_nm']).max())
        if load_error > 1e-5:
            raise ValueError('Applied conservative load reconstruction mismatch')
    result = dict(passed=True, initial_pose_max_error_rad=initial_error,
                  recorded_target_max_error_rad=target_error, recorded_truth_max_error_rad=truth_error,
                  simulated_pd_max_error_nm=pd_error, active_axis_metrics=rows,
                  applied_load_max_error_nm=load_error,
                  gas_spring_force_max_error_n=spring_error,
                  new_gpu_environment_steps=0,
                  limitation='Bias removal is an analysis metric only; physical zero/calibration unchanged. Numeric integrity checks do not establish independent physical acceptance.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
