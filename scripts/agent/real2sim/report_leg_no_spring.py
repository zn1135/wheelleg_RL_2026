"""Independent archive, mass and input checks for equal-torque GPU diagnostics; zero integration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import diagnose_leg_no_spring_gpu as run
from scripts.agent.real2sim import report_leg_response_joint as joint_report


def require(condition, message):
    if not condition: raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = json.loads((args.experiment/'report.json').read_text()); args.output.mkdir(parents=True, exist_ok=False)
    source = Path(report['source_model']); require(run.sha(source) == report['source_model_sha256'], 'Source XML changed')
    source_model = mujoco.MjModel.from_xml_path(str(source)); results = []; plotted = {}
    for variant in report['variants']:
        directory = args.experiment/variant['name']
        require(run.sha(directory/'model.xml') == variant['model_xml_sha256'], 'Diagnostic template changed')
        model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
        model.geom_contype[:] = model.geom_conaffinity[:] = 0
        if not variant['nominal_passive_resistance']: model.dof_frictionloss[:] = 0.; model.dof_damping[:] = 0.
        if not variant['gravity']: model.opt.gravity[:] = 0.
        for side in ('left', 'right'):
            if not variant['spring_assembly_removed']: model.dof_damping[model.joint(side+'_spring2_joint').dofadr[0]] = 0.
        for body in range(1, model.nbody):
            original = source_model.body(model.body(body).name).id
            for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
                require(np.allclose(getattr(model, field)[body], getattr(source_model, field)[original], rtol=1e-12, atol=1e-12),
                        'Body inertial property changed: '+model.body(body).name+'/'+field)
        for name in run.NAMES:
            actual = model.joint(name); original = source_model.joint(name)
            require(np.array_equal(model.jnt_axis[actual.id], source_model.jnt_axis[original.id])
                    and np.array_equal(model.qpos0[actual.qposadr], source_model.qpos0[original.qposadr]), 'Motor axis or zero changed')
            require(np.array_equal(model.dof_armature[actual.dofadr], source_model.dof_armature[original.dofadr]), 'Motor armature changed')
        motors = [model.actuator(n+'_motor').id for n in run.NAMES]
        spring = [i for i in range(model.nu) if 'gas_spring' in model.actuator(i).name]
        if spring:
            require(np.all(model.actuator_gaintype[spring] == mujoco.mjtGain.mjGAIN_FIXED)
                    and np.all(model.actuator_biastype[spring] == mujoco.mjtBias.mjBIAS_NONE)
                    and np.all(model.actuator_dyntype[spring] == mujoco.mjtDyn.mjDYN_NONE), 'Gas actuator has an unaccounted force source')
        data = mujoco.MjData(model); maximum_kkt_error = 0.; minimum_mass_eigenvalue = np.inf
        with np.load(directory/'trajectories.npz') as saved:
            q = saved['qpos']; v = saved['qvel']; labels = saved['labels'].tolist(); controls = saved['controls']
            require(np.isfinite(q).all() and np.isfinite(v).all() and not np.any(v[0]), 'Invalid initial state')
            require(not spring or not np.any(controls[:, spring]), 'Gas force was not turned off')
            require(q.shape[:2] == (121, 18) and variant['new_environment_steps'] == 2160, 'Incomplete finite pulse run')
            for i, (pose, pattern) in enumerate(labels):
                require(np.array_equal(controls[i, motors], run.PATTERNS[pattern]), 'Equal-torque pulse changed')
                require(np.max(abs(saved['initial_qfrc_actuator'][i, saved['motor_v_indices']]-controls[i, motors])) < 1e-5,
                        'Applied initial motor torques differ')
                zero = labels.index([pose, 'zero']); baseline = q[40, zero, saved['motor_q_indices']]-q[0, zero, saved['motor_q_indices']]
                delta = np.degrees((q[40, i, saved['motor_q_indices']]-q[0, i, saved['motor_q_indices']]-baseline)*run.SIGN)
                row = next(r for r in variant['measurements'] if r['pose'] == pose and r['pattern'] == pattern)
                require(np.allclose(delta, row['net_40ms_motor_displacement_deg'], atol=1e-6), 'Reported pulse angles differ')
            for analytic in variant['analytics']:
                zero = labels.index([analytic['pose'], 'zero']); data.qpos[:] = q[0, zero]; data.qvel[:] = 0.
                mujoco.mj_forward(model, data)
                mass = np.empty((model.nv, model.nv)); mujoco.mj_fullM(model, data, mass)
                minimum_mass_eigenvalue = min(minimum_mass_eigenvalue, float(np.linalg.eigvalsh(mass).min()))
                equality = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                jac = data.efc_J.reshape(data.nefc, model.nv)[equality]
                rank = int(np.linalg.matrix_rank(jac, tol=1e-8)) if len(jac) else 0
                require(rank == analytic['equality_rank'] and model.nv-rank == analytic['independent_dofs'], 'Closed-loop mobility differs')
                torque = np.zeros(model.nv); torque[saved['motor_v_indices']] = run.PATTERNS['retract']
                free = np.linalg.solve(mass, torque)
                # 用完整约束方程解加速度，独立于采集程序的降维惯量计算。
                if len(jac):
                    inv_jt = np.linalg.solve(mass, jac.T)
                    multiplier = np.linalg.lstsq(jac@inv_jt, jac@free, rcond=1e-10)[0]
                    constrained = free-inv_jt@multiplier
                else: constrained = free
                difference = float(np.max(abs(constrained[saved['motor_v_indices']]-analytic['analytic_retract_acceleration_rad_s2'])))
                maximum_kkt_error = max(maximum_kkt_error, difference)
                require(difference < .001, 'Independent constrained-inertia calculation differs')
            require(minimum_mass_eigenvalue > 0., 'Nonpositive mechanism inertia')
            middle = labels.index(['middle', 'retract']); zero = labels.index(['middle', 'zero'])
            require(not saved['joint_limit_touched'][:41, middle].any() and not saved['joint_limit_touched'][:41, zero].any(),
                    'Middle-pose comparison touched a joint limit')
            delta = np.degrees(((q[:41, middle]-q[0, middle])-(q[:41, zero]-q[0, zero]))[:, saved['motor_q_indices']]*run.SIGN)
            plotted[variant['name']] = dict(time_ms=saved['time_s'][:41]*1000, delta_deg=delta,
                pin_error_mm=float(saved['closure_error_m'][:41, middle].max()*1000))
            results.append(dict(variant=variant['name'], gas_force_disabled=True, source_body_inertia_preserved=True,
                source_motor_axes_zeros_armatures_preserved=True, independent_kkt_max_error_rad_s2=maximum_kkt_error,
                minimum_mass_eigenvalue=minimum_mass_eigenvalue, middle_retract_40ms_delta_deg=delta[-1].tolist(),
                front_to_rear_absolute_angle_ratio=float(abs(delta[-1, 0]/delta[-1, 1])),
                middle_retract_first40ms_max_pin_error_mm=plotted[variant['name']]['pin_error_mm']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    names = [('closed_stiff_ideal', 'Closed loop, stiff constraints'), ('closed_soft_ideal', 'Closed loop, source constraints'),
             ('spring_assembly_removed', 'Closed loop, entire spring assembly removed'), ('open_ideal', 'Artificial open-loop mechanism')]
    for axis, (name, title) in zip(axes.flat, names):
        d = plotted[name]; axis.plot(d['time_ms'], abs(d['delta_deg'][:, 0]), label='Front motor')
        axis.plot(d['time_ms'], abs(d['delta_deg'][:, 1]), label='Rear motor')
        axis.set_title(title); axis.grid(alpha=.25); axis.set_ylabel('Absolute angle change (deg)'); axis.set_xlabel('Time (ms)')
        axis.text(.04, .76, f"40 ms: {abs(d['delta_deg'][-1,0]):.2f} / {abs(d['delta_deg'][-1,1]):.2f} deg", transform=axis.transAxes)
    axes[0, 0].legend(); fig.suptitle('Gas force off; gravity and friction off; equal and opposite 2 Nm pulses')
    fig.tight_layout(); fig.savefig(args.output/'equal-torque-comparison.png', dpi=150); plt.close(fig)
    protected = joint_report.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), [])
    result = dict(integrity_passed=True, scope=report['scope'], source_model_sha256=report['source_model_sha256'],
                  new_environment_steps_in_this_report=0, verified_gpu_environment_steps=report['new_environment_steps'],
                  results=results, protected_source_checks=protected, response_fit_accepted=False,
                  conclusions=['Gas force off does not make the source model respond with equal motor angles',
                               'Closed-loop mobility is four leg motor plus two wheel coordinates; no motor is geometrically locked',
                               'Opening constraints increases the angle imbalance; spring assembly removal makes only a small difference',
                               'Source soft constraints deform more than the stiff diagnostic; inertia imbalance remains in the stiff mechanism'],
                  limitations=['A mathematically consistent loop does not independently prove that the CAD model matches hardware',
                               'Inertia preservation checks are not a measurement of the real robot masses or centers of mass'])
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(integrity_passed=True, results=results, physical_steps_in_report=0), indent=2))


if __name__ == '__main__': main()
