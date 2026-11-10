"""Audit an isolated extension-stop change using complete GPU replays; zero new integration."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import report_leg_response_joint as report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    results = []; parameters = []; configs = []; initials = []
    for name in ('before', 'after'):
        directory = (args.experiment/('stop-only-'+name)).resolve()
        row, profile, config, plot = report.audit_run(directory, None)
        times, measured, predicted = plot; window = times <= .8
        error = predicted-measured[:, None, :]
        row['first_0_8s_rmse_deg'] = np.sqrt(np.mean(error[window]**2, axis=0)).tolist()
        row['first_0_8s_peak_deg'] = np.max(abs(error[window]), axis=0).tolist()
        row['first_0_8s_measured_span_deg'] = np.ptp(measured[window], axis=0).tolist()
        row['first_0_8s_model_span_deg'] = np.ptp(predicted[window], axis=0).tolist()
        with np.load(directory/'trials.npz') as data:
            initials.append((data['qpos'][0].copy(), data['qvel'][0].copy()))
        report.plot_run(args.output/(name+'.png'), plot)
        results.append(row); parameters.append(profile); configs.append(config)
    first, second = map(copy.deepcopy, parameters)
    changed = [first.pop('spring_extension_stop_m'), second.pop('spring_extension_stop_m')]
    report.require(first == second, 'Isolation experiment changed more than the extension-stop coordinate')
    for key in ('reference', 'recording_sha256', 'config_sha256', 'rc_inputs_sha256'):
        report.require(results[0][key] == results[1][key], 'Isolation input source differs: '+key)
    for a, b in zip(initials[0], initials[1]): report.require(np.array_equal(a, b), 'Initial states differ')
    protected = report.protected_sources(args.experiment, configs)
    completed = report.completed_gpu_reports(args.experiment)
    result = dict(scope='Isolated diagnostic geometry comparison; not an accepted spring-stroke calibration',
                  integrity_passed=True, only_parameter_changed='spring_extension_stop_m', stop_values_m=changed,
                  objective_gates_unchanged=True, full_response_thresholds_passed=all(r['response_thresholds_passed'] for r in results),
                  results=results, protected_source_checks=protected, completed_gpu_batches=completed,
                  total_completed_gpu_environment_steps=sum(x['environment_steps'] for x in completed),
                  new_environment_steps_in_this_report=0, training_model_adopted=False,
                  limitations=['Static equilibrium and initial-hold improvement do not prove full trajectory alignment',
                               'No independent measurement of the gas-spring stroke endpoint is available'])
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(integrity_passed=True, full_response_thresholds_passed=result['full_response_thresholds_passed'],
        first_0_8s_rmse_deg=[r['first_0_8s_rmse_deg'] for r in results]), indent=2))


if __name__ == '__main__': main()
