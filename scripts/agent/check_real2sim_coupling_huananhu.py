#!/usr/bin/env python3
"""Verify the bounded inertia-allocation diagnostic; no hardware connections."""
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'sim2sim'))
import real2sim_coupling_huananhu as coupling

dyn = coupling.dyn


def run_checks(directory):
    directory = Path(directory)
    env = dyn.sim.Playback()
    parameters = np.array([[.004, .003, .12], [.0037, .0035, .06]])
    paths = []
    for fraction in (0., .15):
        path = directory/f'{fraction}.xml'
        coupling.export_allocation(dyn.sim.MODEL_ASSET, path, parameters, fraction)
        paths.append(path)
    try:
        coupling.export_allocation(dyn.sim.MODEL_ASSET, directory/'unphysical.xml', parameters, 1.)
    except ValueError as exc:
        assert 'inertia bound' in str(exc)
    else:
        raise AssertionError('Wheel body exceeds fixed-mass geometry bound')
    fixtures = [dyn.Fixture(path, env.data.qpos) for path in paths]
    inputs = np.zeros((500, 2)); inputs[:50] = [3, -3]
    states = [f.predict(inputs, np.zeros(2), np.zeros(2), np.zeros((2, 3))) for f in fixtures]
    error = float(max(np.max(abs(states[0][k]-states[1][k])) for k in (0, 1)))
    assert error < 1e-7, error
    matrices = []
    for path in paths:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model); data.qpos[:] = env.data.qpos
        mujoco.mj_forward(model, data)
        mass = np.empty((model.nv, model.nv)); mujoco.mj_fullM(model, data, mass)
        matrices.append(mass)
    wheels = [int(env.model.joint(n).dofadr[0]) for n in dyn.mirror.WHEELS]
    diagonal = float(np.max(abs(np.diag(matrices[0])[wheels]-np.diag(matrices[1])[wheels])))
    cross = float(np.max(abs((matrices[0]-matrices[1])[:6, wheels])))
    assert diagonal < 1e-12 and cross > 1e-5, (diagonal, cross)
    return {'fixed_fixture_trajectory_difference': error, 'wheel_axis_mass_difference': diagonal,
            'base_wheel_mass_cross_term_difference': cross, 'impossible_allocation_rejected': True}


def verify_run(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    assert report['complete'] and json.loads((directory/'run.json').read_text())['complete']
    assert not report['automatic_adoption'] and not report['physical_gap_eliminated']
    assert dyn.identity(directory/'source'/Path(coupling.__file__).name)['sha256'] == report['program']['sha256']
    for name, digest in report['source_snapshot'].items():
        assert dyn.identity(directory/'source'/name)['sha256'] == digest
    for name, digest in report['saved_files'].items():
        assert dyn.identity(directory/name)['sha256'] == digest
    old = Path(report['source_run'])
    prior = json.loads((old/'report.json').read_text())
    args = prior['command_arguments']
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT/value
    s = dyn.Session(resolve(args['rl_session']), config=resolve(args['machine_config']))
    r = dyn.RecordedRL(s)
    old_rl = json.loads((old/'rl-report.json').read_text())
    assert report['selected_rows'] == [x['row'] for x in old_rl['initialization'][::10]]
    envs = {'original': dyn.BodyPredictor(s.model_path, s.config, s.wheel_friction)}
    for name, data in report['exports'].items():
        path = directory/(name+'.xml')
        assert dyn.identity(path)['sha256'] == data['sha256']
        envs[name] = dyn.BodyPredictor(path, s.config, s.wheel_friction)
    with np.load(directory/'states.npz') as saved:
        qs, vs = saved['qpos'], saved['qvel']
        indices, cases, models = saved['row'], saved['case'], saved['model']
    lookup = {(int(i), str(c), str(m)): k for k, (i, c, m) in enumerate(zip(indices, cases, models))}
    fields = ['pitch_error_deg']+[n+'_v_error' for n in dyn.mirror.MOTORS]+[n+'_q_error_deg' for n in dyn.mirror.MOTORS[:4]]
    groups, endpoints, delta, endpoint_values = {}, 0, 0., {}
    with (directory/'predictions.csv').open() as stream:
        for row in csv.DictReader(stream):
            i, h = int(row['row']), int(row['horizon_ms'])
            case, model = row['case'], row['model']
            k, step, target = lookup[i, case, model], h//2, i+h//10
            env = envs[model].env
            vel, pos = vs[k, step, env.v][dyn.N2I], qs[k, step, env.q][dyn.N2I[:4]]
            wanted = {'pitch_error_deg': float(np.degrees(dyn.pitch(qs[k, step, 3:7])-
                       dyn.pitch(r.attitude(s.output_ms[i]+h))))}
            wanted.update({n+'_v_error': float(vel[j]-s.f[target, 126+j]) for j, n in enumerate(dyn.mirror.MOTORS)})
            wanted.update({n+'_q_error_deg': float(np.degrees(pos[j]-s.f[target, 122+j]))
                           for j, n in enumerate(dyn.mirror.MOTORS[:4])})
            delta = max(delta, max(abs(float(row[f])-wanted[f]) for f in fields))
            key = f'{case}/{model}/{h}'
            group = groups.setdefault(key, {f: [] for f in fields})
            for f in fields:
                group[f].append(float(row[f]))
            if model in ('original', 'body_0.00'):
                endpoint_values[i, case, 'candidate' if model == 'body_0.00' else model, h] = wanted
            endpoints += 1
    assert endpoints == report['windows']*len(coupling.CASES)*len(envs)*len(dyn.HORIZONS)
    assert delta < 1e-9, delta
    stat_delta, statistics = 0., 0
    for group, values in groups.items():
        for field, data in values.items():
            for key, value in dyn.stats(data).items():
                stat_delta = max(stat_delta, abs(value-report['summaries'][group][field][key]))
                statistics += 1
    assert stat_delta < 1e-9, stat_delta
    prior_delta, prior_endpoints = 0., 0
    with (old/'rl-predictions.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = int(row['row']), row['case'], row['model'], int(row['horizon_ms'])
            if key in endpoint_values:
                prior_delta = max(prior_delta, max(abs(float(row[f])-endpoint_values[key][f]) for f in fields))
                prior_endpoints += 1
    assert prior_endpoints == len(endpoint_values) and prior_delta < 1e-7, (prior_endpoints, prior_delta)
    regeneration = []
    for case in coupling.CASES:
        for name, predictor in envs.items():
            candidates = np.flatnonzero((cases == case) & (models == name))
            k = candidates[len(candidates)//2]
            actions, _ = r.actions(int(indices[k]), dyn.BODY_CASES[case][2])
            q, v = predictor.predict(qs[k, 0], vs[k, 0], actions, case)
            error = float(max(np.max(abs(q-qs[k])), np.max(abs(v-vs[k]))))
            assert error < 1e-10, (case, name, error)
            regeneration.append(error)
    sensitivity = json.loads((directory/'reconstruction-sensitivity.json').read_text())
    with np.load(directory/'reconstruction-states.npz') as saved:
        fresh_q, fresh_v = saved['qpos'], saved['qvel']
        fresh_rows, fresh_cases = saved['row'], saved['case']
    fresh_lookup = {(int(i), str(c)): k for k, (i, c) in enumerate(zip(fresh_rows, fresh_cases))}
    numeric_difference = 0.
    for row in sensitivity:
        i, case, step = row['row'], row['case'], row['horizon_ms']//2
        k, base = fresh_lookup[i, case], lookup[i, case, 'original']
        env = envs['original'].env
        expected = {'initial_q_max_difference': float(np.max(abs(fresh_q[k, 0]-qs[base, 0]))),
                    'initial_v_max_difference': float(np.max(abs(fresh_v[k, 0]-vs[base, 0]))),
                    'pitch_difference_deg': float(abs(np.degrees(dyn.pitch(fresh_q[k, step, 3:7])-
                                                               dyn.pitch(qs[base, step, 3:7])))),
                    'motor_velocity_max_difference_rad_s': float(np.max(abs(
                        fresh_v[k, step, env.v]-vs[base, step, env.v])))}
        numeric_difference = max(numeric_difference, max(abs(row[f]-v) for f, v in expected.items()))
    assert numeric_difference < 1e-9, numeric_difference
    for field in ('initial_q_max_difference', 'initial_v_max_difference', 'pitch_difference_deg',
                  'motor_velocity_max_difference_rad_s'):
        for name, value in dyn.stats([r[field] for r in sensitivity]).items():
            assert abs(report['reconstruction_sensitivity'][field][name]-value) < 1e-9
    for k in (0, len(fresh_rows)//2, len(fresh_rows)-1):
        case = str(fresh_cases[k])
        actions, _ = r.actions(int(fresh_rows[k]), dyn.BODY_CASES[case][2])
        q, v = envs['original'].predict(fresh_q[k, 0], fresh_v[k, 0], actions, case)
        np.testing.assert_allclose(q, fresh_q[k], rtol=0, atol=1e-10)
        np.testing.assert_allclose(v, fresh_v[k], rtol=0, atol=1e-10)
    return {'rl_endpoints_verified': endpoints, 'max_endpoint_difference': delta,
            'statistics_recalculated': statistics, 'max_statistics_difference': stat_delta,
            'prior_run_endpoints_reproduced': prior_endpoints, 'max_prior_run_difference': prior_delta,
            'full_trajectories_regenerated': len(regeneration)+3, 'max_regenerated_difference': max(regeneration),
            'reconstruction_sensitivity_endpoints_verified': len(sensitivity),
            'max_reconstruction_endpoint_difference': numeric_difference,
            'verifier': dyn.identity(Path(__file__))}
