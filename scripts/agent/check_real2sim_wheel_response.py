#!/usr/bin/env python3
"""Verify saved wheel-response sensitivity experiments; no hardware acceptance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys

import mujoco
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import mj_sim2sim_huananhu_multimode as sim
import real2sim_dynamics_huananhu as dyn


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_records(folder, starts, cases):
    report = read(folder / 'report.json')
    assert report['complete'] is True and report['automatic_adoption'] is False
    assert {(r['start_s'], r['case']) for r in report['results']} == {
        (s, c) for s in starts for c in cases}
    assert len(report['results']) == len(starts) * len(cases)
    checked = 0
    for result in report['results']:
        stem = f"from-{result['start_s']}s-{result['case']}"
        saved = read(folder / f'{stem}.json')
        assert saved['summary'] == result
        rows = saved['trajectory']
        states = np.load(folder / f'{stem}.npz')
        assert len(rows) == len(states['qpos']) == len(states['qvel']) == 1500
        inner = states['inner']
        assert inner.shape == (15000, 7) and np.isfinite(inner).all()
        for key in ('qpos', 'qvel'):
            assert np.isfinite(states[key]).all()
        np.testing.assert_array_equal([r['x_m'] for r in rows], states['qpos'][:, 0])
        np.testing.assert_array_equal([r['y_m'] for r in rows], states['qpos'][:, 1])
        np.testing.assert_array_equal([r['height_m'] for r in rows], states['qpos'][:, 2])
        settled = [r for r in rows if r['time_s'] >= 2]
        xy = np.array([[r['x_m'], r['y_m']] for r in settled])
        expected = {
            'body_vx_mean_abs_m_s': np.mean([abs(r['vx_m_s']) for r in settled]),
            'body_vx_bias_m_s': np.mean([r['vx_m_s'] for r in settled]),
            'yaw_rate_mean_abs_rad_s': np.mean([abs(r['yaw_rate_rad_s']) for r in settled]),
            'net_displacement_2_30_s_m': np.linalg.norm(xy[-1] - xy[0]),
            'path_length_2_30_s_m': np.linalg.norm(np.diff(xy, axis=0), axis=1).sum(),
            'wheel_error_mae_rad_s': np.mean(abs(inner[inner[:, 0] >= 2, 3:5]), axis=0),
            'base_contact_samples': sum(r['base_contact'] for r in settled),
            'both_wheel_contact_fraction': np.mean([r['both_wheels_contact'] for r in settled]),
            'max_tilt_deg': max(max(map(abs, r['roll_pitch_deg'])) for r in settled),
        }
        for key, value in expected.items():
            np.testing.assert_allclose(result[key], value, rtol=0., atol=1e-14)
            checked += np.asarray(value).size
    return int(checked)


def regenerate(base, folder, start, case, model, coeff, controller):
    env = sim.Playback('normal', (0., 0., .27), model_path=model,
                       wheel_friction_compensation=coeff)
    if start:
        state = np.load(base / 'settled-initials/initial-states.npz')
        mujoco.mj_resetData(env.model, env.data)
        env.data.qpos[:] = state[f'{start}_qpos']
        env.data.qvel[:] = state[f'{start}_qvel']
        env.last_action[:] = state[f'{start}_last_action']
        env.airborne_time = float(state[f'{start}_airborne_time'])
        env.stable_time = float(state[f'{start}_stable_time'])
        mujoco.mj_forward(env.model, env.data)
    stem = f'from-{start}s-{case}'
    with np.load(folder / f'{stem}.npz') as archive:
        saved = {key: archive[key] for key in ('qpos', 'qvel')}
    largest = 0.
    for index in range(1500):
        env.step(controller)
        for key in ('qpos', 'qvel'):
            difference = float(abs(getattr(env.data, key) - saved[key][index]).max())
            largest = max(largest, difference)
    assert largest == 0., (folder.name, start, case, largest)
    return largest


def verify_contact(base, folder_name='contact-real2sim'):
    folder = base / folder_name
    report = read(folder / 'report.json')
    assert report['complete'] and read(folder / 'run.json')['complete']
    assert report['automatic_adoption'] is False and report['physical_gap_eliminated'] is False
    for name, sha in report['source_sha256'].items():
        assert digest(ROOT / name) == sha
    for name, sha in report['saved_files'].items():
        assert digest(folder / name) == sha
    for info in report['models'].values():
        assert digest(Path(info['path'])) == info['sha256']
    prior = read(ROOT / 'artifacts/real2sim-dynamics-20261021/verified-run/report.json')
    arguments = prior['command_arguments']
    session = dyn.Session(ROOT / arguments['rl_session'], config=ROOT / arguments['machine_config'])
    assert session.summary == report['rl_session'] and session.wheel_response_mode == 0
    recorded = dyn.RecordedRL(session)
    predictors = {name: dyn.BodyPredictor(Path(info['path']), ROOT / arguments['machine_config'])
                  for name, info in report['models'].items()}
    # NPZ indexing decompresses a whole array; load once before checking every endpoint.
    with np.load(folder / 'states.npz') as archive:
        saved = {key: archive[key] for key in archive.files}
    index = {(int(r), str(c), str(m)): k for k, (r, c, m) in
             enumerate(zip(saved['row'], saved['case'], saved['model']))}
    assert len(index) == 272 * 3 * 4
    with (folder / 'predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(index) * len(dyn.HORIZONS)
    fields = ['pitch_error_deg'] + [n + '_v_error' for n in dyn.mirror.MOTORS] + [n + '_q_error_deg' for n in dyn.mirror.MOTORS[:4]]
    endpoint_values = 0
    for row in rows:
        start, horizon = int(row['row']), int(row['horizon_ms'])
        name, case = row['model'], row['case']
        k, step = index[start, case, name], horizon // 2
        q, v = saved['qpos'][k, step], saved['qvel'][k, step]
        env = predictors[name].env
        expected = {'pitch_error_deg': float(np.degrees(dyn.pitch(q[3:7]) -
                                  dyn.pitch(recorded.attitude(session.output_ms[start] + horizon))))}
        velocity = v[env.v][dyn.N2I]
        position = q[env.q][dyn.N2I[:4]]
        expected.update({n + '_v_error': float(velocity[i] - session.f[start + horizon // 10, 126 + i])
                         for i, n in enumerate(dyn.mirror.MOTORS)})
        expected.update({n + '_q_error_deg': float(np.degrees(position[i] - session.f[start + horizon // 10, 122 + i]))
                         for i, n in enumerate(dyn.mirror.MOTORS[:4])})
        for key, value in expected.items():
            assert float(row[key]) == value
            endpoint_values += 1
    for case in report['cases']:
        for name in report['models']:
            for h in dyn.HORIZONS:
                subset = [r for r in rows if (r['case'], r['model'], int(r['horizon_ms'])) == (case, name, h)]
                for key in fields:
                    assert dyn.stats([float(r[key]) for r in subset]) == report['summaries'][f'{case}/{name}/{h}'][key]
    with np.load(ROOT / 'artifacts/real2sim-coupling-20261021/fixed-initials/states.npz') as archive:
        old = {key: archive[key] for key in archive.files}
    old_keys = {(int(r), str(c), str(m)): k for k, (r, c, m) in enumerate(zip(old['row'], old['case'], old['model']))}
    matched = 0
    for (r, c, name), k in index.items():
        old_name = {'original_mesh': 'original', 'candidate_mesh': 'body_0.00',
                    'original': 'original', 'armature_fraction_1': 'body_0.00'}.get(name)
        for field in ('qpos', 'qvel'):
            np.testing.assert_array_equal(saved[field][k, 0], old[field][old_keys[r, c, 'original'], 0])
            if old_name:
                np.testing.assert_array_equal(saved[field][k], old[field][old_keys[r, c, old_name]])
                matched += 1
    regenerated = []
    for r in [report['selected_rows'][i] for i in (0, 135, 271)]:
        for case in report['cases']:
            actions, _ = recorded.actions(r, dyn.BODY_CASES[case][2])
            for name in report['models']:
                k = index[r, case, name]
                q, v = predictors[name].predict(saved['qpos'][k, 0], saved['qvel'][k, 0], actions, case)
                np.testing.assert_array_equal(q, saved['qpos'][k])
                np.testing.assert_array_equal(v, saved['qvel'][k])
                regenerated.append({'row': r, 'case': case, 'model': name, 'max_state_difference': 0.})
    return {'ok': True, 'scope': 'reused supported data and contact-model ablation; no physical gap acceptance',
            'endpoint_values_checked': endpoint_values, 'statistic_groups_checked': len(report['summaries']) * len(fields),
            'mesh_baseline_state_arrays_exactly_matched': matched, 'all_initial_states_bit_identical': True,
            'regenerated': regenerated, 'checker_sha256': digest(Path(__file__))}


def verify_torque_scale(base):
    result = verify_contact(base, 'torque-scale-family')
    folder = base / 'torque-scale-family'
    report = read(folder / 'report.json')
    bench = dyn.Session(report['bench_directory'], config=Path(report['config']['path']))
    assert bench.summary == report['bench_session']
    pulses, rejected, _ = dyn.pulses_from_session(bench)
    assert len(pulses) == 8 and not rejected
    extra = np.array(read(ROOT / 'artifacts/real2sim-dynamics-20261021/verified-run/candidate-parameters.json')['additional_parameters'])
    fixture = dyn.Fixture(bench.model_path, pulses[0].full_qpos)
    matrix = np.empty((2, 2)); mujoco.mj_fullM(fixture.m, fixture.d, matrix)
    inertia = np.diag(matrix)
    np.testing.assert_array_equal(inertia, report['inertia_reference_kg_m2'])
    original = sim.build_model(bench.model_path)
    for fraction in (0., .5, 1.):
        name = f'armature_fraction_{fraction:g}'
        model = sim.build_model(Path(report['models'][name]['path']))
        gain = (inertia + fraction * extra[:, 0]) / (inertia + extra[:, 0])
        np.testing.assert_array_equal(gain, report['requested_torque_scales'][name])
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
                      'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'eq_data', 'qpos0',
                      'geom_pos', 'geom_quat', 'geom_friction', 'geom_contype', 'geom_conaffinity', 'mesh_vert'):
            np.testing.assert_array_equal(getattr(original, field), getattr(model, field))
        for side, wheel in enumerate(dyn.mirror.WHEELS):
            dof = model.joint(wheel).dofadr[0]
            np.testing.assert_equal(model.dof_armature[dof], fraction * extra[side, 0])
            np.testing.assert_equal(model.dof_damping[dof], gain[side] * extra[side, 1])
            np.testing.assert_equal(model.dof_frictionloss[dof], gain[side] * extra[side, 2])
            np.testing.assert_equal(model.actuator_gear[model.actuator(wheel + '_motor').id, 0], gain[side])
    for record in report['pulse_results']:
        pulse = pulses[record['pulse'] - 1]
        name, gravity = record['model'], record['gravity']
        stem = f'pulse-{pulse.number}-gravity-{int(gravity)}-'
        with np.load(folder / (stem + name + '.npz')) as archive:
            q, v = archive['qpos'], archive['qvel']
        with np.load(folder / (stem + 'armature_fraction_1.npz')) as archive:
            ref_q, ref_v = archive['qpos'], archive['qvel']
        largest = float(max(abs(q - ref_q).max(), abs(v - ref_v).max()))
        assert record['max_state_difference_from_inertia_hypothesis'] == largest
        if not gravity:
            assert largest < 1e-4
        rmse = float(np.sqrt(np.mean((pulse.sampled(v) - pulse.actual_v) ** 2)))
        assert record['wheel_speed_rmse_rad_s'] == rmse
        model = dyn.Fixture(Path(report['models'][name]['path']), pulse.full_qpos)
        if not gravity:
            model.m.opt.gravity[:] = 0
        pq, pv = model.predict(pulse.inputs('request'), pulse.q0, pulse.v0, np.zeros((2, 3)))
        np.testing.assert_array_equal(pq, q)
        np.testing.assert_array_equal(pv, v)
    result.update(scope='torque-scale/inertia identifiability diagnostic; no physical torque calibration',
                  pulse_trajectories_regenerated=len(report['pulse_results']),
                  parameter_formula_and_mass_geometry_invariants_checked=True)
    return result


def verify_refit(folder):
    import real2sim_wheel_fit_huananhu as refit

    report = read(folder / 'report.json')
    assert report['complete'] and read(folder / 'run.json')['complete']
    assert report['automatic_adoption'] is False and report['physical_gap_eliminated'] is False
    for name, sha in report['source_sha256'].items():
        assert digest(folder / 'source' / name) == sha
    assert digest(Path(refit.__file__)) == report['source_sha256'][Path(refit.__file__).name]
    config = Path(report['config']['path'])
    assert digest(config) == report['config']['sha256']
    assert digest(Path(report['model']['path'])) == report['model']['sha256']
    session = dyn.Session(report['bench_session']['directory'], config=config)
    assert session.summary == report['bench_session']
    pulses, rejected, _ = dyn.pulses_from_session(session)
    assert len(pulses) == 8 and not rejected
    by_number = {p.number: p for p in pulses}
    for fold in report['folds'].values():
        training, validation = set(fold['fit_pulses']), set(fold['validation_pulses'])
        assert not training & validation and training | validation == set(by_number)
        assert len(training) == len(validation) == 4
        assert {(by_number[n].side, by_number[n].direction) for n in training} == {(s, d) for s in (0, 1) for d in (-1, 1)}
    assert set(report['folds']['forward']['fit_pulses']) == set(report['folds']['reverse']['validation_pulses'])
    for p in pulses:
        w = refit.weights(p, True)
        assert np.isfinite(w).all() and np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
        assert not w[~p.fit_mask].any()
        assert abs(w[refit.masks(p)['active']].sum() - .5) < 1e-12
        assert abs(w[refit.masks(p)['coast']].sum() - .5) < 1e-12
    cv = []
    for row in report['cross_validation']:
        p = by_number[row['pulse']]
        fold = report['folds'][row['fold']]
        assert row['pulse'] in fold['validation_pulses'] and row['pulse'] not in fold['fit_pulses']
        value = refit.metrics(p, np.array(fold[row['model']]))
        assert value == row['metrics']
        cv.append({**row, 'metrics': value})
    assert len(cv) == 16 and refit.selection(cv) == report['selection']
    driver = read(folder / 'driver-fit.json')
    assert digest(folder / 'driver_fit.py') == driver['source']['sha256']
    driver_functions = runpy.run_path(str(folder / 'driver_fit.py'))
    driver_regressions = 0
    for row in driver['validation']:
        p = by_number[row['pulse']]
        fold = driver['folds'][row['fold']]
        assert row['pulse'] in fold['validation_pulses'] and row['pulse'] not in fold['fit_pulses']
        predicted = driver_functions['predict'](p, fold['parameters_gain_tau_s_delay_s'][p.side])
        mask = driver_functions['mask'](p)
        requested = p.amplitude * ((p.time >= p.on) & (p.time < p.off))
        assert dyn.stats((predicted - p.current)[mask]) == row['driver_error_equivalent_nm']
        assert dyn.stats((requested - p.current)[mask]) == row['request_error_equivalent_nm']
        driver_regressions += row['driver_error_equivalent_nm']['rmse'] >= row['request_error_equivalent_nm']['rmse']
    assert driver['all_pulses_improve'] == (driver_regressions == 0)
    parameters = np.array(report['final_parameters'])
    assert parameters.shape == (2, 3) and np.all((parameters >= 0) & (parameters <= dyn.UPPER))
    original = mujoco.MjModel.from_xml_path(report['model']['path'])
    candidate = mujoco.MjModel.from_xml_path(str(folder / 'candidate.xml'))
    assert digest(folder / 'candidate.xml') == report['candidate_integrity']['sha256']
    wheel_v = [original.joint(n).dofadr[0] for n in dyn.mirror.WHEELS]
    for col, field in enumerate(dyn.FIELDS):
        actual = getattr(candidate, 'dof_' + field) - getattr(original, 'dof_' + field)
        np.testing.assert_array_equal(actual[wheel_v], parameters[:, col])
        assert not np.delete(actual, wheel_v).any()
    for field in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat',
                  'geom_type', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_contype', 'geom_conaffinity',
                  'jnt_axis', 'jnt_range', 'eq_data', 'eq_solref', 'eq_solimp', 'actuator_gear', 'mesh_vert'):
        np.testing.assert_allclose(getattr(original, field), getattr(candidate, field), rtol=1e-12, atol=1e-12)
    old = read(Path(report['arguments']['prior_run']) / 'bench-report.json')
    baseline = np.array(old['fits']['request/forward']['parameters'])
    bench_deltas = []
    with np.load(folder / 'bench-states.npz') as saved:
        for p in pulses:
            for name, values in (('original', np.zeros((2, 3))), ('previous', baseline), ('refit', parameters)):
                q, v = p.predict(values, 'request')
                delta = max(abs(q - saved[f'{p.number}_{name}_q']).max(), abs(v - saved[f'{p.number}_{name}_v']).max())
                assert delta == 0
                bench_deltas.append(float(delta))
                row = next(r for r in report['bench_metrics'] if (r['pulse'], r['model']) == (p.number, name))
                assert refit.metrics(p, values) == row['metrics']
    endpoints, statistics, regenerated, baseline_states = 0, 0, 0, 0
    max_regeneration_error = 0.
    fields = ['pitch_error_deg'] + [n + '_v_error' for n in dyn.mirror.MOTORS] + [n + '_q_error_deg' for n in dyn.mirror.MOTORS[:4]]
    for label, rl in report['rl'].items():
        session = dyn.Session(rl['session']['directory'], config=config)
        assert session.summary == rl['session']
        recorded = dyn.RecordedRL(session)
        predictors = {n: dyn.BodyPredictor(Path(info['path']), config) for n, info in rl['models'].items()}
        with np.load(folder / f'{label}-rl-states.npz') as archive:
            saved = {k: archive[k] for k in archive.files}
        lookup = {(int(i), str(c), str(m)): k for k, (i, c, m) in enumerate(zip(saved['row'], saved['case'], saved['model']))}
        assert len(lookup) == len(rl['selected_rows']) * len(refit.CASES) * 3
        with np.load(Path(rl['source_run']) / 'rl-states.npz') as archive:
            prior_rows, prior_cases, prior_models = archive['row'], archive['case'], archive['model']
            prior_lookup = {(int(i), str(c), str(m)): k for k, (i, c, m) in enumerate(zip(prior_rows, prior_cases, prior_models))}
            for field in ('qpos', 'qvel'):
                previous = archive[field]
                for (i, case, name), k in lookup.items():
                    prior_key = (i, case, 'original' if name == 'original' else 'candidate')
                    initial = prior_lookup[i, case, 'original']
                    np.testing.assert_array_equal(saved[field][k, 0], previous[initial, 0])
                    if name != 'refit':
                        np.testing.assert_array_equal(saved[field][k], previous[prior_lookup[prior_key]])
                        baseline_states += 1
        with (folder / f'{label}-rl.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(lookup) * len(dyn.HORIZONS)
        engage = next((i for i, f in enumerate(session.f) if int(f[153]) & 4), None) if label == 'new' else None
        grouped = {}
        for row in rows:
            i, h = int(row['row']), int(row['horizon_ms'])
            j, step, case, name = i + h // 10, h // 2, row['case'], row['model']
            assert session.continuous(i, j) and session.output_ms[j] - session.output_ms[i] == h
            assert np.all((session.f[i:j + 1, 109].astype(int) & 63) == 61)
            assert int(row['extrapolated']) == int(np.max(abs(session.f[i:j + 1, 130:132])) > 37)
            assert int(row['reported_no_contact']) == int(engage is not None and session.rx[j] <= session.rx[engage] + 2)
            k, env = lookup[i, case, name], predictors[name].env
            error = saved['qvel'][k, step, env.v][dyn.N2I] - session.f[j, 126:132]
            qerror = np.degrees(saved['qpos'][k, step, env.q][dyn.N2I[:4]] - session.f[j, 122:126])
            pitch = np.degrees(dyn.pitch(saved['qpos'][k, step, 3:7]) - dyn.pitch(recorded.attitude(session.output_ms[j])))
            np.testing.assert_allclose([float(row[f]) for f in fields], np.r_[pitch, error, qerror], rtol=0, atol=1e-10)
            endpoints += len(fields)
            populations = ['all'] + ([] if int(row['extrapolated']) else ['within_bench'])
            if label == 'new' and int(row['reported_no_contact']):
                populations += ['reported_no_contact'] + ([] if int(row['extrapolated']) else ['reported_no_contact_within_bench'])
            for population in populations:
                grouped.setdefault(f'{population}/{case}/{name}/{h}', []).append(row)
        for key, values in rl['summaries'].items():
            rows_for_key = grouped.get(key, [])
            assert len(rows_for_key) == values['windows']
            for field in fields:
                expected = dyn.stats([float(r[field]) for r in rows_for_key])
                if expected is None:
                    assert values[field] is None
                else:
                    np.testing.assert_allclose(list(expected.values()), list(values[field].values()), rtol=0, atol=1e-10)
                    statistics += len(expected)
        for case in refit.CASES:
            for name, predictor in predictors.items():
                matching = sorted(i for i, c, m in lookup if c == case and m == name)
                for i in (matching[0], matching[len(matching) // 2], matching[-1]):
                    k = lookup[i, case, name]
                    actions, _ = recorded.actions(i, dyn.BODY_CASES[case][2])
                    q, v = predictor.predict(saved['qpos'][k, 0], saved['qvel'][k, 0], actions, case)
                    delta = float(max(abs(q - saved['qpos'][k]).max(), abs(v - saved['qvel'][k]).max()))
                    assert delta == 0
                    max_regeneration_error = max(max_regeneration_error, delta)
                    regenerated += 1
    return {'ok': True, 'scope': 'offline refit split/weights, saved metrics, model invariants and physics regeneration; no real-world acceptance',
            'whole_pulse_cv_trajectories': len(cv), 'final_bench_trajectories': len(bench_deltas),
            'driver_validation_pulses': len(driver['validation']), 'driver_regressions': driver_regressions,
            'endpoint_values_verified': endpoints, 'summary_values_verified': statistics,
            'baseline_state_arrays_exactly_reproduced': baseline_states,
            'full_body_trajectories_regenerated': regenerated, 'max_regeneration_difference': max_regeneration_error,
            'checker_sha256': digest(Path(__file__))}


def verify_sine(folder):
    """Check a complete read-only bench acquisition, not a torque calibration."""
    session = dyn.Session(folder / 'wheel-sine-live-20261021')
    exit_session = dyn.Session(folder / 'wheel-sine-exit-20261021')
    assert session.meta['complete'] and exit_session.meta['complete']
    flash = read(ROOT / 'artifacts/wheel-bench-sine-1p70-5hz-flash-20261021/flash.json')
    assert flash['status'] == 'flashed_verified_reset'
    wave = flash['waveform']
    a = session.f
    flags = a[:, 109].astype(int)
    running = np.flatnonzero((a[:, 152] == 5) & ((flags & 2) == 0) & ((flags & 12) == 12))
    groups = np.split(running, np.flatnonzero(np.diff(running) > 1) + 1)
    assert len(groups) == 2 and all(len(group) for group in groups)
    assert np.all(a[:, 132:136] == 0), 'Unexpected leg request'
    assert not np.any((abs(a[:, 136]) > 1e-5) & (abs(a[:, 137]) > 1e-5)), 'Simultaneous wheel requests'
    assert np.all(a[:, 2] == 0) and np.all(a[:, 29] == 127)
    assert np.all((a[:, 1].astype(int) & 0x3f3) == 0), 'Legs, RL or LQR enabled, or fallen'
    assert all(session.summary[k] == 0 for k in ('missing', 'restarts', 'duplicates', 'out_of_order'))
    assert np.all(exit_session.f[:, 153] == 1) and np.all(exit_session.f[:, 132:138] == 0)
    runs = []
    for expected_side, group in enumerate(groups):
        first, last = int(group[0]), int(group[-1])
        assert session.continuous(first, last)
        side = int(np.argmax(np.max(abs(a[group, 136:138]), axis=0)))
        assert side == expected_side
        assert np.all(a[group, 137 - side] == 0)
        assert session.output_ms[last] - session.output_ms[first] == wave['duration_ms'] - 10
        assert np.max(abs(session.v[group, 4:])) < wave['speed_abort_rad_s']
        times = session.output_ms[group]
        command = a[group, 136 + side].astype(float)

        def waveform(start):
            # Match float32 arithmetic in the flashed C command generator.
            elapsed = (times - start).astype(np.float32)
            edge = np.minimum(elapsed, wave['duration_ms'] - elapsed)
            envelope = np.ones(len(edge), dtype=np.float32)
            ramp = edge < wave['ramp_each_end_ms']
            envelope[ramp] = np.float32(.5) * (np.float32(1) - np.cos(np.float32(np.pi) * edge[ramp] / np.float32(wave['ramp_each_end_ms'])))
            phase = np.float32(2 * np.pi) * np.float32(wave['frequency_hz']) * (elapsed * np.float32(.001))
            return np.float32(wave['peak_requested_nm']) * envelope * np.sin(phase)

        starts = times[0] - np.arange(10)
        errors = [np.max(abs(waveform(t) - command)) for t in starts]
        start = float(starts[np.argmin(errors)])
        error = float(min(errors))
        assert error < 1e-4, ('Sampled request does not match compiled sine', side, error)
        plateau = group[(times - start >= wave['ramp_each_end_ms']) &
                        (times - start < wave['duration_ms'] - wave['ramp_each_end_ms'])]
        omega = 2 * np.pi * wave['frequency_hz']

        def harmonic(time, values, trend=False):
            design = np.c_[np.sin(omega * time), np.cos(omega * time), np.ones(len(time))]
            if trend:
                design = np.c_[design, time - time.mean()]
            fit, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
            return {'amplitude': float(np.hypot(fit[0], fit[1])),
                    'phase_deg': float(np.degrees(np.arctan2(fit[1], fit[0]))),
                    'bias': float(fit[2]), 'trend_per_s': float(fit[3]) if trend else 0.,
                    'residual_rmse': float(np.sqrt(np.mean((values - design @ fit)**2)))}

        torque = harmonic((session.output_ms[plateau] - start) / 1000, session.torque[plateau, 4 + side])
        velocity = harmonic((session.source_ms[plateau, 4 + side] - start) / 1000, session.v[plateau, 4 + side])
        position = harmonic((session.source_ms[plateau, 4 + side] - start) / 1000,
                            session.q[plateau, 4 + side], trend=True)
        current = harmonic((session.source_ms[plateau, 4 + side] - start) / 1000,
                           a[plateau, 150 + side] * dyn.current_scale(session.config) * session.sign[4 + side])
        encoder_phase = (position['phase_deg'] + 90 + 180) % 360 - 180
        blocks = []
        for begin in (1., 3., 5., 7.):
            block = plateau[((session.output_ms[plateau] - start) / 1000 >= begin) &
                            ((session.output_ms[plateau] - start) / 1000 < begin + 2)]
            t = (session.source_ms[block, 4 + side] - start) / 1000
            u = harmonic((session.output_ms[block] - start) / 1000, session.torque[block, 4 + side])
            w = harmonic(t, session.v[block, 4 + side])
            q = harmonic(t, session.q[block, 4 + side], trend=True)
            c = harmonic(t, a[block, 150 + side] * dyn.current_scale(session.config) * session.sign[4 + side])
            blocks.append({'plateau_begin_s': begin, 'samples': len(block),
                           'current_phase_lag_deg': float((u['phase_deg'] - c['phase_deg'] + 180) % 360 - 180),
                           'velocity_lag_from_encoder_deg': float((q['phase_deg'] + 90 - w['phase_deg'] + 180) % 360 - 180),
                           'velocity_amplitude': w['amplitude'], 'velocity_bias': w['bias']})
        phase_lag = (torque['phase_deg'] - velocity['phase_deg'] + 180) % 360 - 180
        after = np.flatnonzero((session.output_ms > start + wave['duration_ms']) &
                              (session.output_ms < start + wave['duration_ms'] + 1000))
        assert len(after) and np.all(a[after, 132:138] == 0)
        runs.append({'side': side, 'wheel': dyn.mirror.WHEELS[side], 'rows': [first, last],
                     'host_start_s': float(session.rx[first]), 'host_end_s': float(session.rx[last]),
                     'inferred_generator_start_ms': start,
                     'run_rows': len(group), 'plateau_rows': len(plateau),
                     'request_waveform_max_error_nm': error,
                     'peak_requested_nm': float(max(abs(command))),
                     'peak_speed_rad_s': float(np.max(abs(session.v[group, 4 + side]))),
                     'steady_command': torque, 'steady_velocity': velocity,
                     'steady_position_with_linear_trend': position,
                     'encoder_derived_velocity_amplitude': omega * position['amplitude'],
                     'encoder_derived_velocity_phase_deg': float(encoder_phase),
                     'reported_velocity_lag_from_encoder_deg': float((encoder_phase - velocity['phase_deg'] + 180) % 360 - 180),
                     'steady_current_equivalent_using_firmware_scale': current,
                     'current_phase_lag_from_request_deg': float((torque['phase_deg'] - current['phase_deg'] + 180) % 360 - 180),
                     'plateau_two_second_blocks': blocks,
                     'velocity_per_request_amplitude': velocity['amplitude'] / torque['amplitude'],
                     'velocity_phase_lag_deg': float(phase_lag)})
    result = {'ok': True, 'scope': 'complete sine acquisition, request waveform and observed response only; not independently calibrated torque/inertia or whole-body acceptance',
            'session': session.summary, 'exit_session': exit_session.summary, 'runs': runs,
            'faults_zero': True, 'leg_requests_zero': True, 'one_wheel_at_a_time': True,
            'exit_disabled_zero_requests': True, 'waveform': wave,
            'flash_record_sha256': digest(ROOT / 'artifacts/wheel-bench-sine-1p70-5hz-flash-20261021/flash.json'),
            'limitations': ['100Hz samples are not the entire 1kHz CAN request history',
                            'one frequency does not independently identify torque gain, inertia and losses',
                            'velocity phase lag is the entire mechanical response, not control latency'],
            'checker_sha256': digest(Path(__file__))}
    if (folder / 'model-comparison/report.json').exists():
        result['model_comparison'] = verify_sine_models(folder, session)
    return result


def verify_sine_models(folder, session):
    base = folder / 'model-comparison'
    report = read(base / 'report.json')
    assert report['complete'] and read(base / 'run.json')['complete']
    assert report['new_fit_performed'] is False and report['automatic_adoption'] is False
    for name, sha in report['source_sha256'].items():
        assert digest(base / 'source' / name) == sha
    assert digest(session.model_path) == report['model']['sha256']
    prior = ROOT / 'artifacts/real2sim-dynamics-20261021/verified-run'
    assert digest(prior / 'candidate.xml') == report['prior_candidate']['sha256']
    np.testing.assert_array_equal(report['parameters']['prior_candidate'], read(prior / 'candidate-parameters.json')['additional_parameters'])
    np.testing.assert_array_equal(report['parameters']['original'], np.zeros((2, 3)))
    with np.load(base / 'states.npz') as archive:
        saved = {key: archive[key] for key in archive.files}
    lookup, regenerated, values_checked = {}, 0, 0
    for run in report['runs']:
        side, start = run['side'], run['generator_start_ms']
        grid, torque = saved[f'{side}_grid'], saved[f'{side}_input']
        selected = saved[f'{side}_rows']
        assert np.all(np.diff(grid) == 1)
        assert np.all(torque[:, 1 - side] == 0)
        elapsed = grid[:-1] - start
        active = (elapsed >= 0) & (elapsed < 10000)
        edge = np.minimum(elapsed[active], 10000 - elapsed[active])
        envelope = .5 * (1 - np.cos(np.pi * np.minimum(edge, 1000) / 1000))
        expected = np.zeros(len(torque))
        expected[active] = 1.7 * envelope * np.sin(2 * np.pi * 5 * elapsed[active] / 1000) * session.sign[4 + side]
        np.testing.assert_allclose(torque[:, side], expected, rtol=0, atol=1e-4)
        initial = run['initial_row']
        pose = dyn.pose_from_sample(session, initial)
        np.testing.assert_array_equal(saved[f'{side}_initial_pose'], pose.data.qpos)
        np.testing.assert_array_equal(saved[f'{side}_q0'], session.q[initial, 4:])
        np.testing.assert_array_equal(saved[f'{side}_v0'], session.v[initial, 4:])
        fixture = dyn.Fixture(session.model_path, pose.data.qpos)
        times = session.source_ms[selected, 4 + side]
        relative = (times - start) / 1000
        actual = session.v[selected, 4 + side]
        plateau = (relative >= 1) & (relative < 9)
        masks = {'full_sine': (relative >= 0) & (relative < 10), 'plateau': plateau, 'coast': relative >= 10}
        design = np.c_[np.sin(2 * np.pi * 5 * relative[plateau]), np.cos(2 * np.pi * 5 * relative[plateau]), np.ones(plateau.sum())]
        measured, _, _, _ = np.linalg.lstsq(design, actual[plateau], rcond=None)
        for name, parameters in report['parameters'].items():
            q, v = fixture.predict(torque, saved[f'{side}_q0'], saved[f'{side}_v0'], np.array(parameters))
            np.testing.assert_array_equal(q, saved[f'{side}_{name}_q'])
            np.testing.assert_array_equal(v, saved[f'{side}_{name}_v'])
            regenerated += 1
            predicted = np.interp(times, grid, v[:, side])
            result = run['models'][name]
            for key, mask in masks.items():
                assert dyn.stats((predicted - actual)[mask]) == result[key]
                values_checked += 5
            fitted, _, _, _ = np.linalg.lstsq(design, predicted[plateau], rcond=None)
            amp_error = 100 * (np.hypot(fitted[0], fitted[1]) / np.hypot(measured[0], measured[1]) - 1)
            phase_error = (np.degrees(np.arctan2(fitted[1], fitted[0]) - np.arctan2(measured[1], measured[0])) + 180) % 360 - 180
            np.testing.assert_allclose([amp_error, phase_error, fitted[2] - measured[2]],
                [result['amplitude_error_percent'], result['phase_error_deg'], result['bias_error_rad_s']], rtol=0, atol=1e-10)
            values_checked += 3
            for k, i in enumerate(selected):
                lookup[side, name, int(i)] = (relative[k], predicted[k], actual[k], predicted[k] - actual[k])
    with (base / 'predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(lookup)
    for row in rows:
        expected = lookup[int(row['side']), row['model'], int(row['row'])]
        np.testing.assert_allclose([float(row[f]) for f in ('time_s', 'predicted_rad_s', 'reported_rad_s', 'error_rad_s')],
                                   expected, rtol=0, atol=1e-10)
    return {'scope': 'unfitted nominal-request fixed-fixture comparison; no driver/sensor model or whole-body acceptance',
            'regenerated_trajectories': regenerated, 'max_state_difference': 0,
            'statistics_recomputed': values_checked, 'prediction_rows_checked': len(rows)}


def verify_backlash(folder):
    """Recompute all fixed-input backlash predictions and check the free-play mechanism."""
    import real2sim_backlash_huananhu as play
    acquisition = verify_sine(play.SOURCE)
    session = dyn.Session(play.SOURCE / 'wheel-sine-live-20261021')
    report = read(folder / 'report.json')
    assert report['complete'] and read(folder / 'run.json')['complete']
    assert not report['automatic_adoption'] and not report['physical_gap_eliminated']
    assert report['mujoco_version'] == mujoco.__version__
    assert report['session'] == session.summary
    assert report['observed_quantity'] == 'motor-equivalent velocity, not unmeasured output wheel velocity'
    for name, sha in report['source_sha256'].items():
        assert digest(folder / 'source' / name) == digest(ROOT / 'sim2sim' / name) == sha
    for item in report['input_files'] + [report['model'], report['config']]:
        assert digest(Path(item['path'])) == item['sha256']
    np.testing.assert_array_equal(report['parameters'], play.PARAMETERS)
    cases = {(0, 'motor', 0, .001)}
    cases |= {(g, loss, flank, .001) for g in (1, 2) for loss in ('motor', 'wheel') for flank in (-1, 0, 1)}
    cases |= {(g, loss, 0, dt) for dt in (.0005, .0001) for g in (0, 1, 2)
              for loss in (('motor',) if g == 0 else ('motor', 'wheel'))}
    expected = {(side, *case) for side in (0, 1) for case in cases}
    assert len(report['results']) == len(expected) == 46
    assert {(r['side'], r['gap_deg'], r['loss_location'], r['initial_flank'], r['timestep_s'])
            for r in report['results']} == expected
    saved = np.load(folder / 'states.npz')
    prior = np.load(play.SOURCE / 'model-comparison/states.npz')
    lookup, summary_values = {}, 0
    for item in report['results']:
        side, gap, loss = item['side'], item['gap_deg'], item['loss_location']
        dt, flank, key = item['timestep_s'], item['initial_flank'], item['key']
        tc = .0002 if dt == .0001 else .002
        assert item['limit_timeconstant_s'] == tc
        fixture = play.BacklashFixture(session.model_path, prior[f'{side}_initial_pose'], gap, loss, dt, tc)
        assert item['invariants'] == fixture.invariants
        assert fixture.m.nu == 2
        if gap:
            np.testing.assert_array_equal(fixture.m.jnt_range[fixture.m.joint(dyn.mirror.WHEELS[side] + '_play').id],
                                          [-np.radians(gap) / 2, np.radians(gap) / 2])
            assert fixture.m.dof_armature[fixture.rv[side]] == 0
            np.testing.assert_array_equal(fixture.m.actuator_trnid[:, 0],
                [fixture.m.joint(n).id for n in dyn.mirror.WHEELS])
            for n in dyn.mirror.WHEELS:
                np.testing.assert_array_equal(fixture.m.joint(n).axis, fixture.m.joint(n + '_play').axis)
        torque = np.repeat(prior[f'{side}_input'], round(.001 / dt), axis=0)
        np.testing.assert_array_equal(saved[key + '_torque'], torque)
        grid = prior[f'{side}_grid'][0] + np.arange(len(torque) + 1) * dt * 1000
        np.testing.assert_array_equal(saved[key + '_grid'], grid)
        q, v = fixture.predict(torque, prior[f'{side}_q0'], prior[f'{side}_v0'], flank)
        np.testing.assert_array_equal(saved[key + '_q'], q)
        np.testing.assert_array_equal(saved[key + '_v'], v)
        if gap == 0 and dt == .001:
            np.testing.assert_array_equal(q, prior[f'{side}_prior_candidate_q'])
            np.testing.assert_array_equal(v, prior[f'{side}_prior_candidate_v'])
            assert item['zero_gap_prior_state_difference'] == 0
        selected = prior[f'{side}_rows']
        times = session.source_ms[selected, 4 + side]
        start = acquisition['runs'][side]['inferred_generator_start_ms']
        t = (times - start) / 1000
        mask = (t >= 1) & (t < 9)
        actual = session.v[selected, 4 + side]
        motor = np.interp(times, grid, v[:, fixture.v[side]])
        wheel = np.interp(times, grid, v[:, fixture.v[side]] + (v[:, fixture.rv[side]] if gap else 0))
        metrics = dyn.stats((motor - actual)[mask])
        basis = np.c_[np.sin(2 * np.pi * 5 * t[mask]), np.cos(2 * np.pi * 5 * t[mask]), np.ones(mask.sum())]
        measured, _, _, _ = np.linalg.lstsq(basis, actual[mask], rcond=None)
        fitted, _, _, _ = np.linalg.lstsq(basis, motor[mask], rcond=None)
        metrics.update(phase_error_deg=float((np.degrees(np.arctan2(fitted[1], fitted[0]) -
                       np.arctan2(measured[1], measured[0])) + 180) % 360 - 180),
                       amplitude_error_percent=float(100 * (np.hypot(*fitted[:2]) / np.hypot(*measured[:2]) - 1)),
                       bias_error_rad_s=float(fitted[2] - measured[2]))
        for name, value in metrics.items():
            np.testing.assert_allclose(item['metrics'][name], value, rtol=0, atol=1e-10)
            summary_values += 1
        excess = np.maximum(abs(q[:, fixture.rq[side]]) - np.radians(gap) / 2, 0).max() if gap else 0
        np.testing.assert_allclose(item['limit_penetration_deg'], np.degrees(excess), rtol=0, atol=1e-12)
        np.testing.assert_allclose(item['unmeasured_wheel_motor_velocity_difference_rms'],
                                  np.sqrt(np.mean((wheel - motor)[mask]**2)), rtol=0, atol=1e-12)
        for i, row in enumerate(selected):
            lookup[key, int(row)] = (t[i], motor[i], wheel[i], actual[i], motor[i] - actual[i])
        print(json.dumps({'backlash_regenerated': key}), flush=True)
    with (folder / 'predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(lookup)
    seen = set()
    for row in rows:
        key = row['case'], int(row['row'])
        assert key not in seen
        seen.add(key)
        np.testing.assert_allclose([float(row[f]) for f in ('time_s', 'motor_rad_s', 'unmeasured_wheel_rad_s',
            'observed_motor_rad_s', 'error_rad_s')], lookup[key], rtol=0, atol=1e-10)
    # Inside the gap, motor torque moves the rotor while the unloaded wheel stays still.
    probes = []
    for sign in (-1, 1):
        fixture = play.BacklashFixture(session.model_path, prior['0_initial_pose'], 1, 'motor')
        fixture.m.opt.gravity[:] = 0
        q, v = fixture.predict(np.array([[float(sign), 0.]]), np.zeros(2), np.zeros(2), 0)
        assert sign * v[-1, fixture.v[0]] > .1
        assert abs(v[-1, fixture.v[0]] + v[-1, fixture.rv[0]]) < 1e-10
        assert abs(q[-1, fixture.rq[0]]) < np.radians(.5)
        probes.append(v[-1])
    np.testing.assert_allclose(probes[0], -probes[1], rtol=0, atol=1e-10)
    return {'ok': True, 'scope': 'conditional offline backlash sensitivity; no independent wheel-end measurement or physical gap acceptance',
            'regenerated_trajectories': len(report['results']), 'max_state_difference': 0,
            'statistics_recomputed': summary_values, 'prediction_rows_checked': len(rows),
            'free_play_motor_wheel_decoupling_and_sign_symmetry': True,
            'source_acquisition_and_original_comparison': acquisition,
            'report_sha256': digest(folder / 'report.json'), 'checker_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--contact-only', action='store_true')
    parser.add_argument('--torque-scale-only', action='store_true')
    parser.add_argument('--refit-only', action='store_true')
    parser.add_argument('--sine-only', action='store_true')
    parser.add_argument('--backlash-only', action='store_true')
    args = parser.parse_args()
    base = args.run.resolve()
    if args.contact_only or args.torque_scale_only or args.refit_only or args.sine_only or args.backlash_only:
        assert sum((args.contact_only, args.torque_scale_only, args.refit_only, args.sine_only, args.backlash_only)) == 1
        if args.backlash_only:
            result = verify_backlash(base)
        elif args.sine_only:
            result = verify_sine(base)
        elif args.refit_only:
            result = verify_refit(base)
        else:
            result = verify_torque_scale(base) if args.torque_scale_only else verify_contact(base)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    fit = ROOT / 'artifacts/real2sim-dynamics-20261021/verified-run'
    cases = ('original', 'candidate_no_compensation', 'candidate_full_precision', 'candidate_approved_h7')
    fit_coeff = np.array(read(fit / 'candidate-parameters.json')['additional_parameters'])[:, 2]
    h7_coeff = np.array(read(base / 'parameters.json')['additional_parameters'])[:, 2]
    settled = read(base / 'settled-initials/report.json')
    for path, sha in settled['source_sha256'].items():
        assert digest(ROOT / path) == sha, path
    checked = verify_records(base / 'settled-initials', (2, 5, 10), cases)
    checked += verify_records(base / 'cylinder-contact', (0, 2, 5, 10), cases)
    for old, new in [(sim.MODEL_ASSET, base / 'cylinder-contact/original-cylinder.xml'),
                     (fit / 'candidate.xml', base / 'cylinder-contact/candidate-cylinder.xml')]:
        a, b = sim.build_model(old), sim.build_model(new)
        assert (a.nq, a.nv, a.ngeom) == (b.nq, b.nv, b.ngeom)
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_axis', 'jnt_range',
                      'dof_armature', 'dof_damping', 'dof_frictionloss', 'actuator_gear', 'actuator_ctrlrange'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        state = np.load(base / 'settled-initials/initial-states.npz')
        matrices = []
        for model in (a, b):
            data = mujoco.MjData(model)
            data.qpos[:] = state['5_qpos']
            data.qvel[:] = state['5_qvel']
            mujoco.mj_forward(model, data)
            mass = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, data, mass)
            matrices.append(mass)
        np.testing.assert_array_equal(*matrices)
    fixed = read(base / 'fixed-actions/report.json')
    assert fixed['complete'] and len(fixed['cases']) == 4
    assert fixed['entry_sha256'] == digest(Path(sim.__file__))
    fixed_states = {}
    for case in fixed['cases']:
        name, coeff = case['actions'], case['coefficients']
        data = np.load(base / f'fixed-actions/actions-{name}-coeff-{coeff}.npz')
        assert len(data['qpos']) == len(data['qvel']) == 1500
        assert np.isfinite(data['qpos']).all() and np.isfinite(data['qvel']).all()
        fixed_states[name, coeff] = np.c_[data['qpos'], data['qvel']]
        if name == coeff:
            assert case['exact_saved_trajectory_reproduction_max_difference'] == 0.
    for name in ('full_precision', 'approved_h7'):
        delta = abs(fixed_states[name, 'full_precision'] - fixed_states[name, 'approved_h7'])
        assert fixed[name + '_fixed_actions_difference']['max_base_position_m'] == delta[:, :3].max()
    onnx_path = ROOT / 'exports/chuanliantui_multimode/model5849-h7-20261017/model_5849_opset13.onnx'
    evaluator = ReferenceEvaluator(onnx.load(str(onnx_path)))

    def controller(obs):
        return evaluator.run(None, {'obs': obs[None]})[0][0].astype(np.float32)

    regenerated = []
    for folder, start, case, model, coeff in [
        ('settled-initials', 5, 'candidate_full_precision', fit / 'candidate.xml', fit_coeff),
        ('settled-initials', 5, 'candidate_approved_h7', fit / 'candidate.xml', h7_coeff),
        ('settled-initials', 2, 'candidate_no_compensation', fit / 'candidate.xml', None),
        ('cylinder-contact', 0, 'candidate_approved_h7', base / 'cylinder-contact/candidate-cylinder.xml', h7_coeff),
        ('cylinder-contact', 5, 'candidate_full_precision', base / 'cylinder-contact/candidate-cylinder.xml', fit_coeff),
        ('cylinder-contact', 10, 'original', base / 'cylinder-contact/original-cylinder.xml', None),
    ]:
        diff = regenerate(base, base / folder, start, case, model, coeff, controller)
        regenerated.append({'folder': folder, 'start_s': start, 'case': case, 'max_state_difference': diff})
        print(json.dumps(regenerated[-1]), flush=True)
    result = {'ok': True, 'scope': 'saved simulation arithmetic and sampled regeneration only; no physical calibration or gap acceptance',
              'summary_values_checked': checked, 'settled_cases': 12, 'cylinder_cases': 16,
              'fixed_action_cases': 4, 'cylinder_mass_and_control_invariants': True,
              'fixed_configuration_mass_matrix_identical': True, 'regenerated': regenerated,
              'checker_sha256': digest(Path(__file__)), 'formal_xml_sha256': digest(sim.MODEL_ASSET)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
