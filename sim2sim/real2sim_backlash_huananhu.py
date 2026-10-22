#!/usr/bin/env python3
"""Offline two-inertia backlash sensitivity against measured motor-side feedback.

Gap means total wheel-side free travel, not +/- that value. No hardware access,
parameter fitting, encoder remapping or modification of production XML occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import real2sim_dynamics_huananhu as dyn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/wheel-sine-test-20261021'
PRIOR = ROOT / 'artifacts/real2sim-dynamics-20261021/verified-run'
PARAMETERS = np.array(json.loads((PRIOR / 'candidate-parameters.json').read_text())['additional_parameters'])
FIELDS = ['mae', 'rmse', 'phase_error_deg', 'amplitude_error_percent', 'bias_error_rad_s']


def harmonic(time, values):
    basis = np.c_[np.sin(2 * np.pi * 5 * time), np.cos(2 * np.pi * 5 * time), np.ones(len(time))]
    x, _, _, _ = np.linalg.lstsq(basis, values, rcond=None)
    return np.array((np.hypot(x[0], x[1]), np.degrees(np.arctan2(x[1], x[0])), x[2]))


class BacklashFixture:
    """Motor armature + wheel body; bounded relative hinge transmits flank forces."""

    def __init__(self, model_path, initial_pose, gap_deg, loss_location, dt=.001, limit_timeconstant=.002):
        self.gap = np.radians(gap_deg)
        self.base = dyn.Fixture(model_path, initial_pose, dt=dt)
        self.base.parameters(PARAMETERS)
        if not gap_deg:
            self.m, self.d = self.base.m, self.base.d
        else:
            spec = mujoco.MjSpec.from_file(str(model_path))
            for i in range(1, self.base.m.nbody):
                body = spec.body(self.base.m.body(i).name)
                body.pos = self.base.m.body_pos[i]
                body.quat = self.base.m.body_quat[i]
            for joint in list(spec.joints):
                if joint.name not in dyn.mirror.WHEELS:
                    spec.delete(joint)
            for eq in list(spec.equalities):
                spec.delete(eq)
            for actuator in list(spec.actuators):
                if actuator.name not in tuple(n + '_motor' for n in dyn.mirror.WHEELS):
                    spec.delete(actuator)
            for side, name in enumerate(dyn.mirror.WHEELS):
                j = spec.joint(name)
                j.armature = self.base.m.dof_armature[self.base.v[side]]
                j.damping[:] = [self.base.m.dof_damping[self.base.v[side]] if loss_location == 'motor' else 0, 0, 0]
                j.frictionloss = self.base.m.dof_frictionloss[self.base.v[side]] if loss_location == 'motor' else 0
                relative = spec.body(name).add_joint(name=name + '_play', type=mujoco.mjtJoint.mjJNT_HINGE,
                    pos=j.pos.copy(), axis=j.axis.copy(), limited=1, range=[-self.gap / 2, self.gap / 2],
                    damping=[0, 0, 0], frictionloss=0, armature=0)
                relative.solref_limit = [limit_timeconstant, 1]
                relative.solimp_limit = [.99, .99, .001, .5, 2]
                if loss_location == 'wheel':
                    # Wheel bearing losses depend on absolute wheel speed, not relative play speed.
                    tendon = spec.add_tendon(name=name + '_absolute_loss',
                        damping=[self.base.m.dof_damping[self.base.v[side]], 0, 0],
                        frictionloss=self.base.m.dof_frictionloss[self.base.v[side]])
                    tendon.wrap_joint(name, 1.)
                    tendon.wrap_joint(name + '_play', 1.)
            self.m = spec.compile()
            self.m.geom_contype[:] = self.m.geom_conaffinity[:] = 0
            self.m.opt.timestep = dt
            self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            self.m.opt.tolerance = 1e-10
            self.m.opt.iterations = 100
            self.d = mujoco.MjData(self.m)
        self.q = np.array([self.m.joint(n).qposadr[0] for n in dyn.mirror.WHEELS])
        self.v = np.array([self.m.joint(n).dofadr[0] for n in dyn.mirror.WHEELS])
        self.a = np.array([self.m.actuator(n + '_motor').id for n in dyn.mirror.WHEELS])
        self.rq = np.array([self.m.joint(n + '_play').qposadr[0] for n in dyn.mirror.WHEELS]) if gap_deg else None
        self.rv = np.array([self.m.joint(n + '_play').dofadr[0] for n in dyn.mirror.WHEELS]) if gap_deg else None
        mujoco.mj_forward(self.m, self.d)
        mass = np.empty((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, self.d, mass)
        original = np.empty((self.base.m.nv, self.base.m.nv))
        mujoco.mj_forward(self.base.m, self.base.d)
        mujoco.mj_fullM(self.base.m, self.base.d, original)
        np.testing.assert_allclose(mass[np.ix_(self.v, self.v)], original, rtol=0, atol=1e-14)
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'geom_pos', 'geom_quat', 'geom_friction'):
            np.testing.assert_array_equal(getattr(self.m, field), getattr(self.base.m, field))
        self.invariants = {'minimum_mass_eigenvalue': float(np.linalg.eigvalsh(mass).min()),
                           'locked_rotor_mass_matrix_matches_candidate': True,
                           'mass_geometry_unchanged': True, 'nv': self.m.nv}
        assert self.invariants['minimum_mass_eigenvalue'] > 0

    def predict(self, torque, q0, v0, flank):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.q], self.d.qvel[self.v] = q0, v0
        if self.rq is not None:
            self.d.qpos[self.rq] = flank * self.gap / 2
        qs, vs = [self.d.qpos.copy()], [self.d.qvel.copy()]
        for value in torque:
            self.d.ctrl[self.a] = value
            mujoco.mj_step(self.m, self.d)
            qs.append(self.d.qpos.copy()); vs.append(self.d.qvel.copy())
        if max(self.d.warning.number) or not np.isfinite(qs).all() or not np.isfinite(vs).all():
            raise RuntimeError('Backlash simulation warning/nonfinite state')
        return np.array(qs), np.array(vs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    dyn.save_json(out / 'run.json', {'complete': False})
    source = out / 'source'; source.mkdir()
    for p in (Path(__file__), Path(dyn.__file__), Path(dyn.mirror.__file__), Path(dyn.sim.__file__)):
        shutil.copy2(p, source / p.name)
    session = dyn.Session(SOURCE / 'wheel-sine-live-20261021')
    protocol = json.loads((SOURCE / 'check-sine-final.json').read_text())
    with np.load(SOURCE / 'model-comparison/states.npz') as archive:
        prior = {key: archive[key] for key in archive.files}
    report = {'complete': False, 'automatic_adoption': False, 'physical_gap_eliminated': False,
              'operator_statement': 'wheel transmission, 1-2 degrees measured at wheel end',
              'gap_definition': 'TOTAL wheel-side free travel; relative hinge limits +/- gap/2',
              'method_reference': 'https://mujoco.readthedocs.io/en/stable/modeling.html#backlash',
              'model': dyn.identity(session.model_path), 'config': dyn.identity(session.config),
              'session': session.summary, 'parameters': PARAMETERS.tolist(),
              'hypotheses': 'All extra candidate armature assigned to motor equivalent. Losses entirely motor-side or wheel-side; initial flank -1/0/+1. No fit or velocity time shift.',
              'observed_quantity': 'motor-equivalent velocity, not unmeasured output wheel velocity',
              'mujoco_version': mujoco.__version__,
              'input_files': [dyn.identity(p) for p in (SOURCE / 'model-comparison/states.npz',
                  SOURCE / 'check-sine-final.json', PRIOR / 'candidate-parameters.json', PRIOR / 'candidate.xml')],
              'source_sha256': {p.name: dyn.identity(p)['sha256'] for p in source.iterdir()}, 'results': []}
    saved, rows = {}, []
    cases = [(0, 'motor', 0, .001, .002)]
    cases += [(g, loss, flank, .001, .002) for g in (1, 2) for loss in ('motor', 'wheel') for flank in (-1, 0, 1)]
    cases += [(g, loss, 0, .0005, .002) for g in (0, 1, 2) for loss in (('motor',) if g == 0 else ('motor', 'wheel'))]
    # A finite harder-flank sensitivity; not a fit to the measured trajectory.
    cases += [(g, loss, 0, .0001, .0002) for g in (0, 1, 2) for loss in (('motor',) if g == 0 else ('motor', 'wheel'))]
    for run in protocol['runs']:
        side = run['side']
        source_rows = prior[f'{side}_rows']
        source_time = session.source_ms[source_rows, 4 + side]
        relative_time = (source_time - run['inferred_generator_start_ms']) / 1000
        plateau = (relative_time >= 1) & (relative_time < 9)
        actual = session.v[source_rows, 4 + side]
        measured = harmonic(relative_time[plateau], actual[plateau])
        for gap, loss, flank, dt, limit_timeconstant in cases:
            key = f's{side}-gap{gap}-{loss}-flank{flank}-dt{dt}'
            fixture = BacklashFixture(session.model_path, prior[f'{side}_initial_pose'], gap, loss, dt, limit_timeconstant)
            torque = prior[f'{side}_input']
            if dt < .001:
                torque = np.repeat(torque, round(.001 / dt), axis=0)  # Same 1ms held command, finer integration only.
            grid = prior[f'{side}_grid'][0] + np.arange(len(torque) + 1) * dt * 1000
            q, v = fixture.predict(torque, prior[f'{side}_q0'], prior[f'{side}_v0'], flank)
            motor = np.interp(source_time, grid, v[:, fixture.v[side]])
            wheel = np.interp(source_time, grid, v[:, fixture.v[side]] + (v[:, fixture.rv[side]] if gap else 0))
            h = harmonic(relative_time[plateau], motor[plateau])
            error = motor - actual
            metrics = dyn.stats(error[plateau])
            metrics.update(phase_error_deg=float((h[1] - measured[1] + 180) % 360 - 180),
                           amplitude_error_percent=float(100 * (h[0] / measured[0] - 1)),
                           bias_error_rad_s=float(h[2] - measured[2]))
            excess = float(np.maximum(abs(q[:, fixture.rq[side]]) - fixture.gap / 2, 0).max()) if gap else 0.
            baseline_difference = None
            if gap == 0 and dt == .001:
                baseline_difference = float(max(abs(q - prior[f'{side}_prior_candidate_q']).max(),
                                                abs(v - prior[f'{side}_prior_candidate_v']).max()))
                assert baseline_difference == 0
            item = {'key': key, 'side': side, 'gap_deg': gap, 'loss_location': loss, 'initial_flank': flank,
                    'timestep_s': dt, 'metrics': metrics, 'limit_penetration_deg': float(np.degrees(excess)),
                    'limit_timeconstant_s': limit_timeconstant,
                    'zero_gap_prior_state_difference': baseline_difference, 'invariants': fixture.invariants,
                    'unmeasured_wheel_motor_velocity_difference_rms': float(np.sqrt(np.mean((wheel - motor)[plateau]**2)))}
            report['results'].append(item)
            saved[key + '_q'], saved[key + '_v'] = q, v
            saved[key + '_torque'] = torque
            saved[key + '_grid'] = grid
            for k, index in enumerate(source_rows):
                rows.append({'case': key, 'row': int(index), 'time_s': float(relative_time[k]),
                             'motor_rad_s': float(motor[k]), 'unmeasured_wheel_rad_s': float(wheel[k]),
                             'observed_motor_rad_s': float(actual[k]), 'error_rad_s': float(error[k])})
            print(json.dumps({'case': key, 'mae': metrics['mae'], 'phase_error_deg': metrics['phase_error_deg'],
                              'penetration_deg': item['limit_penetration_deg']}), flush=True)
    with (out / 'predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / 'states.npz', **saved)
    plot_results(out, rows, report)
    report['complete'] = True
    dyn.save_json(out / 'report.json', report)
    dyn.save_json(out / 'run.json', {'complete': True})


def plot_results(out, rows, report):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for side, label in enumerate(('Left', 'Right')):
        ax = axes[0, side]
        for gap, color in ((0, 'tab:blue'), (1, 'tab:orange'), (2, 'tab:green')):
            key = f's{side}-gap{gap}-motor-flank0-dt0.001'
            selected = [r for r in rows if r['case'] == key and 4 <= r['time_s'] <= 4.6]
            if not gap:
                ax.plot([r['time_s'] for r in selected], [r['observed_motor_rad_s'] for r in selected], 'k.', label='Measured motor')
            ax.plot([r['time_s'] for r in selected], [r['motor_rad_s'] for r in selected], color=color, label=f'{gap} deg play')
        ax.set(title=f'{label}: motor-side prediction', xlabel='Time from sine start (s)', ylabel='Motor-equivalent speed (rad/s)')
        ax.legend(fontsize=8)
        ax = axes[1, side]
        for dt, marker, text in ((.001, 'o', '1 ms'), (.0005, 's', '0.5 ms'), (.0001, '^', '0.1 ms, harder flank')):
            points = [r for r in report['results'] if r['side'] == side and r['timestep_s'] == dt]
            for gap in (0, 1, 2):
                vals = [r['metrics']['mae'] for r in points if r['gap_deg'] == gap]
                ax.plot([gap] * len(vals), vals, marker, color={.001: 'tab:blue', .0005: 'tab:orange', .0001: 'tab:green'}[dt], label=text if gap == 0 else None)
        ax.set(xlabel='Total wheel-side play (deg)', ylabel='Motor speed MAE (rad/s)', xticks=[0, 1, 2], title='1-9 s; all loss/flank hypotheses')
        ax.legend(fontsize=8)
        ax = axes[2, side]
        selected = [r for r in rows if r['case'] == f's{side}-gap2-motor-flank0-dt0.0001' and 4.05 <= r['time_s'] <= 4.15]
        ax.plot([r['time_s'] for r in selected], [r['motor_rad_s'] for r in selected], label='Predicted motor')
        ax.plot([r['time_s'] for r in selected], [r['unmeasured_wheel_rad_s'] for r in selected], label='Predicted wheel (UNMEASURED)')
        ax.plot([r['time_s'] for r in selected], [r['observed_motor_rad_s'] for r in selected], 'k.', label='Measured motor')
        ax.set(title='2 deg model near reversal, harder flank', xlabel='Time (s)', ylabel='Speed (rad/s)')
        ax.legend(fontsize=8)
    fig.suptitle('Backlash sensitivity: fixed nominal input, old candidate parameters; no refit')
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
