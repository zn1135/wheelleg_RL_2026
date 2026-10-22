#!/usr/bin/env python3
"""Refit conditional wheel response with whole-pulse cross-validation, offline only.

The command-torque scale stays fixed at one. These parameters describe response;
they do not independently identify physical inertia or mechanical shaft torque.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.optimize import least_squares

import real2sim_dynamics_huananhu as dyn

SPEED_BINS = (0., 2., 10., 20., float('inf'))
CASES = ('lower_contact', 'late_action', 'rolling_inferred')


def masks(pulse):
    active = pulse.fit_mask & (pulse.time < pulse.off)
    coast = pulse.fit_mask & ~active
    result = {'all': np.ones(len(pulse.time), dtype=bool), 'interior': pulse.fit_mask,
              'active': active, 'coast': coast}
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        result[f'speed_{lo:g}_{hi:g}'] = pulse.fit_mask & (abs(pulse.actual_v) >= lo) & (abs(pulse.actual_v) < hi)
    return result


def weights(pulse, balanced):
    if not balanced:
        return pulse.fit_mask.astype(float) / pulse.fit_mask.sum()
    # Each phase has equal total weight; nonempty speed bins share its weight.
    phases = [masks(pulse)[name] for name in ('active', 'coast')]
    phases = [m for m in phases if m.any()]
    result = np.zeros(len(pulse.time))
    for phase in phases:
        cells = [phase & (abs(pulse.actual_v) >= lo) & (abs(pulse.actual_v) < hi)
                 for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:])]
        cells = [cell for cell in cells if cell.any()]
        for cell in cells:
            result[cell] = 1 / (len(phases) * len(cells) * cell.sum())
    return result


def fit(pulses, initial, balanced):
    parameters, diagnostics = np.zeros((2, 3)), []
    for side in (0, 1):
        selected = [p for p in pulses if p.side == side]
        if {p.direction for p in selected} != {-1, 1}:
            raise ValueError('Both directions are required for each wheel')
        fitting_weights = [np.sqrt(weights(p, balanced)) for p in selected]

        def residual(x):
            values = np.zeros((2, 3)); values[side] = x
            return np.concatenate([(p.sampled(p.predict(values, 'request')[1]) - p.actual_v) * w
                                   for p, w in zip(selected, fitting_weights)])

        runs = [least_squares(residual, start, bounds=(np.zeros(3), dyn.UPPER),
                              x_scale=np.array((.004, .004, .15)), max_nfev=65,
                              ftol=2e-6, xtol=2e-6, gtol=2e-6)
                for start in (initial[side], np.array((.001, .015, .3)))]
        best = min(runs, key=lambda r: np.sum(r.fun**2))
        parameters[side] = best.x
        diagnostics.append({'side': side, 'pulses': [p.number for p in selected],
                            'runs': [{'cost': float(np.sum(r.fun**2)), 'x': r.x.tolist(),
                                      'success': bool(r.success), 'nfev': r.nfev} for r in runs]})
        print(json.dumps({'phase': 'fit', 'balanced': balanced, **diagnostics[-1]}), flush=True)
    return parameters, diagnostics


def metrics(pulse, values):
    q, v = pulse.predict(values, 'request')
    error = pulse.sampled(v) - pulse.actual_v
    return {name: {'samples': int(mask.sum()), 'error': dyn.stats(error[mask])}
            for name, mask in masks(pulse).items()}


def selection(rows):
    # Compare against the matching legacy fold, not the much worse original XML.
    scores = {name: float(np.mean([r['metrics']['all']['error']['rmse'] for r in rows if r['model'] == name]))
              for name in ('legacy', 'balanced')}
    checks = []
    for side in (0, 1):
        for direction in (-1, 1):
            for phase in ('active', 'coast'):
                values = {name: np.mean([r['metrics'][phase]['error']['rmse'] for r in rows
                                        if r['model'] == name and r['side'] == side and r['direction'] == direction])
                          for name in scores}
                checks.append({'side': side, 'direction': direction, 'phase': phase,
                               **{k: float(v) for k, v in values.items()},
                               'pass': bool(values['balanced'] <= 1.1 * values['legacy'])})
    accepted = scores['balanced'] < scores['legacy'] and all(r['pass'] for r in checks)
    return {'selected_objective': 'balanced' if accepted else 'legacy', 'mean_pulse_rmse': scores,
            'phase_direction_checks': checks,
            'rule': 'balanced must lower mean whole-pulse RMSE and avoid >10% regression in either phase/direction/wheel',
            'validation_is_reused': True}


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_rl(run, candidate, out, label, config):
    prior = json.loads((run / 'report.json').read_text())
    session = dyn.Session(dyn.ROOT / prior['command_arguments']['rl_session'], config=config)
    recorded = dyn.RecordedRL(session)
    with np.load(run / 'rl-states.npz') as archive:
        # Keep the previously solved initial states bit-for-bit for every model.
        selected = np.unique(archive['row'])[::5]
        original = (archive['model'] == 'original') & np.isin(archive['case'], CASES) & np.isin(archive['row'], selected)
        indices, cases = archive['row'][original], archive['case'][original]
        initial_q, initial_v = archive['qpos'][original, 0], archive['qvel'][original, 0]
    models = {'original': session.model_path, 'previous': run / 'candidate.xml', 'refit': candidate}
    predictors = {name: dyn.BodyPredictor(path, config, session.wheel_friction) for name, path in models.items()}
    rows, qs, vs, keys = [], [], [], []
    engage = next((i for i, f in enumerate(session.f) if int(f[153]) & 4), None) if label == 'new' else None
    for count, (i, case, q, v) in enumerate(zip(indices, cases, initial_q, initial_v)):
        i, case = int(i), str(case)
        assert session.continuous(i, i + 10)
        assert np.all((session.f[i:i + 11, 109].astype(int) & 63) == 61)
        actions, _ = recorded.actions(i, dyn.BODY_CASES[case][2])
        for name, predictor in predictors.items():
            pq, pv = predictor.predict(q, v, actions, case)
            qs.append(pq); vs.append(pv); keys.append((i, case, name))
            for h in dyn.HORIZONS:
                j, step = i + h // 10, h // 2
                row = {'row': i, 'case': case, 'model': name, 'horizon_ms': h,
                       'reported_no_contact': int(engage is not None and session.rx[j] <= session.rx[engage] + 2),
                       'extrapolated': int(np.max(abs(session.f[i:j + 1, 130:132])) > 37),
                       'pitch_error_deg': float(np.degrees(dyn.pitch(pq[step, 3:7]) - dyn.pitch(recorded.attitude(session.output_ms[j]))))}
                velocities = pv[step, predictor.env.v][dyn.N2I]
                positions = pq[step, predictor.env.q][dyn.N2I[:4]]
                row.update({motor + '_v_error': float(velocities[k] - session.f[j, 126 + k])
                            for k, motor in enumerate(dyn.mirror.MOTORS)})
                row.update({motor + '_q_error_deg': float(np.degrees(positions[k] - session.f[j, 122 + k]))
                            for k, motor in enumerate(dyn.mirror.MOTORS[:4])})
                rows.append(row)
        if count % 60 == 0:
            print(json.dumps({'phase': 'rl', 'session': label, 'initials_done': count, 'initials': len(indices)}), flush=True)
    write_csv(out / f'{label}-rl.csv', rows)
    np.savez_compressed(out / f'{label}-rl-states.npz', qpos=qs, qvel=vs,
                        row=[k[0] for k in keys], case=[k[1] for k in keys], model=[k[2] for k in keys])
    summaries = {}
    fields = ['pitch_error_deg'] + [n + '_v_error' for n in dyn.mirror.MOTORS] + [n + '_q_error_deg' for n in dyn.mirror.MOTORS[:4]]
    populations = {'all': lambda r: True, 'within_bench': lambda r: not r['extrapolated']}
    if label == 'new':
        populations.update(reported_no_contact=lambda r: bool(r['reported_no_contact']),
                           reported_no_contact_within_bench=lambda r: bool(r['reported_no_contact']) and not r['extrapolated'])
    for population, accept in populations.items():
        for case in CASES:
            for model in models:
                for h in dyn.HORIZONS:
                    subset = [r for r in rows if accept(r) and (r['case'], r['model'], r['horizon_ms']) == (case, model, h)]
                    summaries[f'{population}/{case}/{model}/{h}'] = {'windows': len(subset),
                        **{field: dyn.stats([r[field] for r in subset]) for field in fields}}
    return {'session': session.summary, 'source_run': str(run), 'models': {n: dyn.identity(p) for n, p in models.items()},
            'selected_rows': selected.tolist(), 'initial_state_policy': 'unchanged prior saved initial state for all three models',
            'summaries': summaries}


def run(args):
    out = args.out
    prior = json.loads((args.prior_run / 'report.json').read_text())
    config = dyn.ROOT / prior['command_arguments']['machine_config']
    session = dyn.Session(dyn.ROOT / prior['command_arguments']['bench_session'], config=config)
    pulses, rejected, _ = dyn.pulses_from_session(session)
    if rejected or len(pulses) != 8:
        raise ValueError('This study requires the original eight complete pulses')
    old = json.loads((args.prior_run / 'bench-report.json').read_text())
    baseline = np.array(old['fits']['request/forward']['parameters'])
    report = {'complete': False, 'automatic_adoption': False, 'physical_gap_eliminated': False,
              'scope': 'conditional command-response refit; fixed torque scale 1; no hardware operation',
              'arguments': {k: str(v) for k, v in vars(args).items()}, 'bench_session': session.summary,
              'model': dyn.identity(session.model_path), 'config': dyn.identity(config),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dyn.ROOT, text=True).strip(),
              'protocol': {'hypothesis': 'request midpoint; no edge retiming or torque gain fitting',
                           'objectives': ['legacy', 'balanced'], 'speed_bins_rad_s': [0, 2, 10, 20, 'infinity'],
                           'split': 'whole first vs second repetitions of each wheel/direction, then swap',
                           'rl_selection': 'every fifth previously accepted row, three fixed boundary cases; no RL fitting',
                           'limitation': 'all recorded datasets have been examined previously; no pristine holdout'},
              'folds': {}, 'cross_validation': []}
    source = out / 'source'; source.mkdir()
    for path in (Path(__file__), Path(dyn.__file__), Path(dyn.sim.__file__), Path(dyn.mirror.__file__), config):
        shutil.copy2(path, source / path.name)
    report['source_sha256'] = {p.name: dyn.identity(p)['sha256'] for p in source.iterdir()}
    for fold in ('forward', 'reverse'):
        fitting = [p for p in pulses if p.number in old['fits']['request/' + fold]['fit_pulses']]
        testing = [p for p in pulses if p not in fitting]
        legacy = np.array(old['fits']['request/' + fold]['parameters'])
        balanced, diagnostics = fit(fitting, legacy, True)
        report['folds'][fold] = {'fit_pulses': [p.number for p in fitting], 'validation_pulses': [p.number for p in testing],
                                 'legacy': legacy.tolist(), 'balanced': balanced.tolist(), 'diagnostics': diagnostics}
        for p in testing:
            for name, values in (('legacy', legacy), ('balanced', balanced)):
                report['cross_validation'].append({'fold': fold, 'pulse': p.number, 'side': p.side,
                    'direction': p.direction, 'model': name, 'metrics': metrics(p, values)})
        dyn.save_json(out / 'progress.json', report)
    report['selection'] = selection(report['cross_validation'])
    parameters, diagnostics = fit(pulses, baseline, report['selection']['selected_objective'] == 'balanced')
    report['final_parameters'] = parameters.tolist(); report['final_fit_diagnostics'] = diagnostics
    report['candidate_integrity'] = dyn.export_candidate(session.model_path, out / 'candidate.xml', parameters)
    report['bench_metrics'], traces = [], {}
    for p in pulses:
        for name, values in (('original', np.zeros((2, 3))), ('previous', baseline), ('refit', parameters)):
            q, v = p.predict(values, 'request')
            traces[f'{p.number}_{name}_q'] = q; traces[f'{p.number}_{name}_v'] = v
            report['bench_metrics'].append({'pulse': p.number, 'model': name, 'metrics': metrics(p, values)})
    np.savez_compressed(out / 'bench-states.npz', **traces)
    dyn.save_json(out / 'progress.json', report)
    report['rl'] = {}
    for label, directory in (('old', args.prior_run), ('new', args.rl_run)):
        report['rl'][label] = evaluate_rl(directory, out / 'candidate.xml', out, label, config)
        dyn.save_json(out / 'progress.json', report)
    assert dyn.identity(session.model_path) == report['model'] and dyn.identity(config) == report['config']
    report['complete'] = True
    dyn.save_json(out / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, required=True)
    parser.add_argument('--rl-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dyn.save_json(args.out / 'run.json', {'complete': False, 'phase': 'running'})
    try:
        run(args)
    except BaseException as exc:
        dyn.save_json(args.out / 'run.json', {'complete': False, 'phase': 'failed', 'error': str(exc)})
        raise
    dyn.save_json(args.out / 'run.json', {'complete': True, 'phase': 'finished'})


if __name__ == '__main__':
    main()
