#!/usr/bin/env python3
"""Offline inertia-allocation diagnostic, never a physical calibration.

Keep fixed-fixture wheel inertia and losses constant while moving a bounded
fraction of the conditional armature into each wheel's body inertia tensor.
The supported RL recording is reused for screening, not independent acceptance.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import real2sim_dynamics_huananhu as dyn

FRACTIONS = (0., .05, .10, .15)
CASES = ('lower_contact', 'late_action', 'rolling_inferred')


def body_tensor(model, body):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def wheel_bounds(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = []
    for name in dyn.mirror.WHEELS:
        joint = model.joint(name).id
        body = model.jnt_bodyid[joint]
        axis = model.jnt_axis[joint]
        radii = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[g]
            begin, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = model.mesh_vert[begin:begin+count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            local = (world-data.xpos[body]) @ data.xmat[body].reshape(3, 3) - model.jnt_pos[joint]
            radii.extend(np.linalg.norm(local-np.outer(local @ axis, axis), axis=1))
        radius, mass = float(max(radii)), float(model.body_mass[body])
        radial_com = model.body_ipos[body]-model.jnt_pos[joint]
        radial_com -= axis * (radial_com @ axis)
        inertia = float(axis @ body_tensor(model, body) @ axis + mass*(radial_com @ radial_com))
        bounds.append({'wheel': name, 'mass_kg': mass, 'mesh_radius_m': radius,
                       'axial_body_inertia_kg_m2': inertia,
                       'conditional_upper_kg_m2': mass*radius**2})
    return bounds


def export_allocation(source, output, parameters, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError('Allocation fraction outside [0,1]')
    original = mujoco.MjModel.from_xml_path(str(source))
    tree = ET.parse(source)
    bodies, wheels, expected = [], [], []
    bounds = wheel_bounds(original)
    for side, name in enumerate(dyn.mirror.WHEELS):
        j = original.joint(name).id
        body, dof = original.jnt_bodyid[j], original.jnt_dofadr[j]
        delta = fraction*parameters[side, 0]
        if bounds[side]['axial_body_inertia_kg_m2']+delta > bounds[side]['conditional_upper_kg_m2']+1e-12:
            raise ValueError(f'{name}: exceeds fixed-mass mesh-radius inertia bound')
        axis = original.jnt_axis[j]
        # A symmetric redistribution changes axial inertia by delta; mass/COM stay fixed.
        tensor = body_tensor(original, body) + delta*(.5*np.eye(3)+.5*np.outer(axis, axis))
        inertial = tree.find(f".//body[@name='{name}']/inertial")
        if delta:
            # Explicit principal axes avoid the compiler's absolute eigensolver tolerance
            # dominating differences between these small wheel inertia tensors.
            moments, rotation = np.linalg.eigh(tensor)
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
            for field in ('fullinertia', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
                inertial.attrib.pop(field, None)
            inertial.set('diaginertia', ' '.join(map(str, moments)))
            inertial.set('quat', ' '.join(map(str, quat)))
        joint = tree.find(f".//joint[@name='{name}']")
        for k, field in enumerate(dyn.FIELDS):
            extra = parameters[side, k]*(1-fraction if k == 0 else 1)
            joint.set(field, str(float(getattr(original, 'dof_'+field)[dof]+extra)))
        bodies.append(body); wheels.append(dof); expected.append(tensor)
    compiler = tree.find('compiler')
    compiler.set('meshdir', str((Path(source).parent/compiler.get('meshdir', '')).resolve()))
    tree.write(output, encoding='utf-8', xml_declaration=True)
    candidate = mujoco.MjModel.from_xml_path(str(output))
    if (candidate.nq, candidate.nv, candidate.neq) != (original.nq, original.nv, original.neq):
        raise ValueError('Topology changed')
    for field in ('body_mass', 'body_pos', 'body_quat', 'body_ipos', 'jnt_pos', 'jnt_axis', 'jnt_range',
                  'jnt_limited', 'eq_data', 'eq_solref', 'eq_solimp', 'geom_pos', 'geom_quat',
                  'geom_friction', 'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity',
                  'mesh_vert', 'actuator_gainprm', 'actuator_ctrlrange', 'actuator_gear', 'qpos0'):
        np.testing.assert_allclose(getattr(candidate, field), getattr(original, field), rtol=1e-12, atol=1e-12)
    for field in ('body_inertia', 'body_iquat'):
        np.testing.assert_allclose(np.delete(getattr(candidate, field), bodies, axis=0),
                                   np.delete(getattr(original, field), bodies, axis=0), rtol=1e-12, atol=1e-12)
    for body, tensor in zip(bodies, expected):
        np.testing.assert_allclose(body_tensor(candidate, body), tensor, rtol=1e-10, atol=1e-12)
    for k, field in enumerate(dyn.FIELDS):
        wanted = getattr(original, 'dof_'+field).copy()
        wanted[wheels] += parameters[:, k]*(1-fraction if k == 0 else 1)
        np.testing.assert_allclose(getattr(candidate, 'dof_'+field), wanted, rtol=1e-12, atol=1e-12)
    return {'fraction': fraction, 'sha256': dyn.identity(output)['sha256'], 'bounds': bounds,
            'status': 'passes necessary geometry bound only; not measured or adopted'}


def run(args):
    previous = json.loads((args.dynamics_run/'report.json').read_text())
    if not previous['complete'] or not json.loads((args.dynamics_run/'run.json').read_text())['complete']:
        raise ValueError('Input dynamics run is incomplete')
    arguments = previous['command_arguments']
    def resolve(value):
        return Path(value) if Path(value).is_absolute() else dyn.ROOT/value
    config = resolve(arguments['machine_config'])
    bench = dyn.Session(resolve(arguments['bench_session']), config=config)
    session = dyn.Session(resolve(arguments['rl_session']), model=bench.model_path, config=config)
    for field, path in (('model', session.model_path), ('config', config)):
        if dyn.identity(path)['sha256'] != previous['inputs'][field]['sha256']:
            raise ValueError(f'{field} changed since input run')
    parameters = np.array(json.loads((args.dynamics_run/'candidate-parameters.json').read_text())['additional_parameters'])
    prior_bench = json.loads((args.dynamics_run/'bench-report.json').read_text())
    np.testing.assert_array_equal(parameters, prior_bench['fits']['request/forward']['parameters'])
    prior_rl = json.loads((args.dynamics_run/'rl-report.json').read_text())
    previous_summary = dict(prior_rl['session'], directory=str(resolve(prior_rl['session']['directory'])))
    if session.summary != previous_summary:
        raise ValueError('RL session changed since input run')
    out = args.out
    source = out/'source'; source.mkdir()
    for path in (Path(__file__), Path(dyn.__file__), Path(dyn.sim.__file__), Path(dyn.mirror.__file__),
                 config, args.dynamics_run/'candidate-parameters.json',
                 dyn.ROOT/'scripts/agent/check_real2sim_coupling_huananhu.py'):
        shutil.copy2(path, source/path.name)
    selected = [x['row'] for x in prior_rl['initialization'][::10]]
    # Reusing bit-identical initial states separates parameter effects from closure
    # solver roundoff; some mesh contact trajectories amplify tiny pose changes.
    with np.load(args.dynamics_run/'rl-states.npz') as cached:
        indices, cases, models = cached['row'], cached['case'], cached['model']
        mask = np.isin(indices, selected) & np.isin(cases, CASES) & (models == 'original')
        keys_initial = list(zip(indices[mask].astype(int), cases[mask]))
        initial_q = cached['qpos'][mask, 0].copy()
        initial_v = cached['qvel'][mask, 0].copy()
    initial = {key: (q, v) for key, q, v in zip(keys_initial, initial_q, initial_v)}
    paths, exports = {'original': session.model_path}, {}
    for fraction in FRACTIONS:
        name = f'body_{fraction:.2f}'
        paths[name] = out/(name+'.xml')
        exports[name] = export_allocation(session.model_path, paths[name], parameters, fraction)
    pulses, rejected, _ = dyn.pulses_from_session(bench)
    if rejected:
        raise ValueError(f'Pulse rejection: {rejected}')
    invariance = []
    for pulse in pulses:
        pose = dyn.pose_from_sample(bench, pulse.start)
        baseline = None
        for name, path in list(paths.items())[1:]:
            fixture = dyn.Fixture(path, pose.data.qpos)
            q, v = fixture.predict(pulse.inputs('request'), bench.q[pulse.start, 4:],
                                   bench.v[pulse.start, 4:], np.zeros((2, 3)))
            if baseline is None:
                baseline = q, v
            error = float(max(np.max(abs(q-baseline[0])), np.max(abs(v-baseline[1]))))
            if error > 1e-7:
                raise ValueError(f'Fixed-fixture response changed: {name} P{pulse.number} {error}')
            invariance.append({'pulse': pulse.number, 'model': name, 'max_state_difference': error})
    predictors = {name: dyn.BodyPredictor(path, config, session.wheel_friction) for name, path in paths.items()}
    recorded = dyn.RecordedRL(session)
    rows, qs, vs, keys = [], [], [], []
    for count, index in enumerate(selected):
        window = slice(index, index+11)
        if not session.continuous(index, index+10) or not np.all(session.f[window, 154] == 3):
            raise ValueError('Previously selected window is no longer valid')
        if not np.all((session.f[window, 109].astype(int) & 63) == 61):
            raise ValueError('Applied PD snapshot unavailable')
        for case in CASES:
            lag = dyn.BODY_CASES[case][2]
            q, v = initial[index, case]
            actions, _ = recorded.actions(index, lag)
            for name, predictor in predictors.items():
                pq, pv = predictor.predict(q, v, actions, case)
                qs.append(pq); vs.append(pv); keys.append((index, case, name))
                for h in dyn.HORIZONS:
                    j, step = index+h//10, h//2
                    vel = pv[step, predictor.env.v][dyn.N2I]
                    pos = pq[step, predictor.env.q][dyn.N2I[:4]]
                    row = {'row': index, 'host_s': float(session.rx[index]), 'case': case,
                           'model': name, 'horizon_ms': h,
                           'pitch_error_deg': float(np.degrees(dyn.pitch(pq[step, 3:7])-
                                           dyn.pitch(recorded.attitude(session.output_ms[index]+h))))}
                    row.update({motor+'_v_error': float(vel[k]-session.f[j, 126+k])
                                for k, motor in enumerate(dyn.mirror.MOTORS)})
                    row.update({motor+'_q_error_deg': float(np.degrees(pos[k]-session.f[j, 122+k]))
                                for k, motor in enumerate(dyn.mirror.MOTORS[:4])})
                    rows.append(row)
        if count % 40 == 0:
            print(json.dumps({'windows': count+1, 'total': len(selected)}), flush=True)
    with (out/'predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/'states.npz', qpos=np.array(qs), qvel=np.array(vs),
                        row=np.array([k[0] for k in keys]), case=np.array([k[1] for k in keys]),
                        model=np.array([k[2] for k in keys]))
    sensitivity, reconstructed_q, reconstructed_v, reconstructed_keys = [], [], [], []
    lookup = {key: k for k, key in enumerate(keys)}
    original = predictors['original']
    for index in selected[::9]:
        q0, v0, meta = original.measured_state(recorded, index)
        for case in CASES:
            translation, gap, lag = dyn.BODY_CASES[case]
            q, v = q0.copy(), v0.copy()
            q[2] += gap
            v[:3] = meta['lower_velocity' if translation == 'lower' else 'rolling_velocity']
            actions, _ = recorded.actions(index, lag)
            pq, pv = original.predict(q, v, actions, case)
            reconstructed_q.append(pq); reconstructed_v.append(pv); reconstructed_keys.append((index, case))
            base = lookup[index, case, 'original']
            for h in dyn.HORIZONS:
                step = h//2
                sensitivity.append({'row': index, 'case': case, 'horizon_ms': h,
                    'initial_q_max_difference': float(np.max(abs(q-qs[base][0]))),
                    'initial_v_max_difference': float(np.max(abs(v-vs[base][0]))),
                    'pitch_difference_deg': float(abs(np.degrees(dyn.pitch(pq[step, 3:7])-
                                                               dyn.pitch(qs[base][step, 3:7])))),
                    'motor_velocity_max_difference_rad_s': float(np.max(abs(
                        pv[step, original.env.v]-vs[base][step, original.env.v])))})
    np.savez_compressed(out/'reconstruction-states.npz', qpos=np.array(reconstructed_q), qvel=np.array(reconstructed_v),
                        row=np.array([k[0] for k in reconstructed_keys]), case=np.array([k[1] for k in reconstructed_keys]))
    dyn.save_json(out/'reconstruction-sensitivity.json', sensitivity)
    fields = ['pitch_error_deg']+[n+'_v_error' for n in dyn.mirror.MOTORS]+[n+'_q_error_deg' for n in dyn.mirror.MOTORS[:4]]
    summaries = {}
    for case in CASES:
        for name in paths:
            for h in dyn.HORIZONS:
                subset = [r for r in rows if (r['case'], r['model'], r['horizon_ms']) == (case, name, h)]
                summaries[f'{case}/{name}/{h}'] = {field: dyn.stats([r[field] for r in subset]) for field in fields}
    report = {'complete': True, 'scope': 'reused supported RL data, bounded causal diagnostic; no independent acceptance',
              'source_run': str(args.dynamics_run), 'inputs': previous['inputs'],
              'bench_session': bench.summary, 'rl_session': session.summary,
              'selection': 'Every tenth previously accepted NORMAL start, entire recording; no fitting',
              'initial_states': 'bit-identical original-model initial states from verified dynamics run',
              'selected_rows': selected, 'windows': len(selected), 'exports': exports,
              'fixed_fixture_invariance': invariance, 'summaries': summaries,
              'reconstruction_sensitivity': {
                  'selection': 'Every ninth screened start; repeat closure solve using the same measurements',
                  'windows': len(reconstructed_keys),
                  **{field: dyn.stats([r[field] for r in sensitivity]) for field in
                     ('initial_q_max_difference', 'initial_v_max_difference', 'pitch_difference_deg',
                      'motor_velocity_max_difference_rad_s')}},
              'automatic_adoption': False, 'physical_gap_eliminated': False,
              'limitations': ['Wheel mass and enclosing radius are model assumptions, not measured truth',
                             'Inertia bound is necessary but insufficient for a realizable mass distribution',
                             'Real geared-rotor coupling is not identified by either allocation extreme',
                             'Unknown human support, root translation and torque scale remain confounded'],
              'program': dyn.identity(Path(__file__)), 'mujoco_version': mujoco.__version__}
    report['source_snapshot'] = {path.name: dyn.identity(path)['sha256'] for path in source.iterdir()}
    report['saved_files'] = {name: dyn.identity(out/name)['sha256'] for name in
                            ('predictions.csv', 'states.npz', 'reconstruction-states.npz', 'reconstruction-sensitivity.json')}
    dyn.save_json(out/'report.json', report)
    plot(report, out)


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    models = ('original',)+tuple(report['exports'])
    for row, case in enumerate(CASES):
        for col, (field, label) in enumerate((('lfwheel_v_error', 'Left wheel MAE (rad/s)'),
                                            ('rfwheel_v_error', 'Right wheel MAE (rad/s)'),
                                            ('pitch_error_deg', 'Pitch MAE (deg)'))):
            ax = axes[row, col]
            for name in models:
                ax.plot(dyn.HORIZONS, [report['summaries'][f'{case}/{name}/{h}'][field]['mae']
                                      for h in dyn.HORIZONS], 'o-', label=name)
            ax.set(xlabel='Prediction horizon (ms)', ylabel=label, title=case)
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Same fixed-fixture inertia, different base coupling\nReused supported data; no physical calibration')
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics-run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dyn.save_json(args.out/'run.json', {'complete': False, 'phase': 'running'})
    try:
        run(args)
    except BaseException as error:
        dyn.save_json(args.out/'run.json', {'complete': False, 'phase': 'failed', 'error': str(error)})
        raise
    dyn.save_json(args.out/'run.json', {'complete': True, 'phase': 'finished'})


if __name__ == '__main__':
    main()
