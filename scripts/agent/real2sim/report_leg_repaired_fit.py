"""Audit all 0.2ms saved states/forces of a repaired-loop fit, without any dynamics steps."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit
from scripts.agent.real2sim import report_leg_response_joint as audit
from scripts.agent.real2sim import report_leg_loop_failure as loops


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', choices=tuple(fit.REFERENCES),
                        help='Audit selected recordings; coverage is explicitly included in the report')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.experiment/'report.json').read_text()); rows = {}; all_pass = True; solver_settings={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in fit.REFERENCES:
        if args.sessions and name not in args.sessions: continue
        directory = args.experiment/('final-'+name); result = json.loads((directory/'report.json').read_text())
        profile = result['results'][0]['parameters']; assert profile == parent['selected_parameters']
        assert len(result['results']) == 1 and result['physics_backend'] == 'mujoco_warp_gpu'
        for path, value in result['input_sha256'].items(): assert audit.digest(Path(path)) == value
        assert audit.digest(Path(result['rc_path'])) == result['rc_sha256']
        assert audit.digest(directory/'model.xml') == result['model_sha256']
        inputs = fit.input_data(name); model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
        solver_settings[name]=dict(profile=result.get('solver_profile','incremental'),iterations=model.opt.iterations,
            tolerance=model.opt.tolerance,solver_type=int(model.opt.solver),cone_type=int(model.opt.cone))
        source = mujoco.MjModel.from_xml_string(ET.tostring(fit.canonical_xml(inputs), encoding='unicode'))
        fields = ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
                  'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'eq_data', 'qpos0', 'site_pos', 'dof_armature']
        assert all(np.array_equal(getattr(model, f), getattr(source, f)) for f in fields), 'Physical arrays differ from canonical shared model; legacy inertia is prohibited'
        assert model.opt.timestep == .0002
        assert np.all(model.eq_solref[:, 0] == .0004)
        assert np.all(model.jnt_solref[model.jnt_limited.astype(bool), 0] == .0004)
        mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
        mv = [model.joint(n).dofadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
        sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
        sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left', 'right')]
        aid = [model.actuator(n+'_motor').id for n in fit.prior.replay.mirror.MOTORS]
        sid = [model.actuator(s+'_gas_spring_motor').id for s in ('left', 'right')]
        saved = np.load(directory/'physics.npz'); q = saved['qpos']; v = saved['qvel']; ctrl = saved['ctrl']
        capacity = saved['friction_capacity']; times = saved['time_s']
        assert q.shape[1:] == (2, model.nq) and np.isfinite(q).all() and np.isfinite(v).all()
        assert np.isfinite(ctrl).all() and np.isfinite(capacity).all() and capacity.min()>=0.
        assert np.max(abs(ctrl[:, :, aid[:4]]))<=40.000001
        commands = inputs['saved']['submitted_model_torque'][:, 0]
        assert len(q) == len(commands)*5 and result['new_gpu_environment_steps'] == len(q)*2
        audit.close(q[0], np.broadcast_to(inputs['saved']['qpos'][0, 0], q[0].shape), 'Initial pose', 1e-7)
        audit.close(v[0], np.broadcast_to(inputs['saved']['qvel'][0, 0], v[0].shape), 'Initial velocity', 1e-7)
        audit.close(ctrl[:, 0, aid], np.repeat(commands, 5, axis=0), 'Recorded motor torque', 1e-6)
        spring = audit.spring_force(q[:, :, sq], v[:, :, sv], profile)
        spring_error = audit.close(ctrl[:, :, sid], spring, 'Spring control every physics step', .002)
        resistance = audit.native_motor_friction(q[:, :, sq], v[:, :, mv], profile)
        friction_error = audit.close(capacity[:, :, mv], resistance, 'Motor friction capacity every physics step', 2e-5)
        audit.close(capacity[:, :, sv], np.broadcast_to(profile['spring_friction_n'], capacity[:, :, sv].shape),
                    'Slider friction capacity', 2e-5)
        recovering = np.full(1, inputs['recovering']); blocked = np.zeros(1, dtype=bool); expected = []
        for k in range(len(commands)):
            expected.append(fit.controller(q[k*5, 1:2], v[k*5, 1:2], mq, mv, inputs, k, recovering, blocked)[0])
        feedback_error = audit.close(ctrl[:, 1, aid[:4]], np.repeat(expected, 5, axis=0), 'Held feedback torque', 5e-5)
        assert not np.any(ctrl[:, 1, aid[4:]])
        qfull = np.concatenate((q, saved['final_qpos'][None])); tfull = np.r_[times, len(commands)*.001]
        raw = inputs['saved']['real_motor_state']; absolute = tfull+inputs['report']['window_start_board_s']
        real = np.column_stack([np.interp(absolute, raw[:, 0], raw[:, j+1]) for j in range(4)])
        error = np.degrees(qfull[:, :, mq]*inputs['sign'][:4]+inputs['zero']-real[:, None, :])
        branches = []
        for branch, key in enumerate(('recorded_torque', 'recorded_target')):
            physical, gaps, area, excess = loops.audit.metrics(model, tfull, qfull[:, branch], qfull[:, branch])
            all_gaps = loops.all_gaps(model, qfull[:, branch])
            assert np.allclose(all_gaps[:, :4], gaps, atol=1e-12)
            stable = bool(all_gaps.max()<.0001 and excess.max()<np.radians(.1)
                          and physical['first_triangle_area_sign_change_s'] == [None, None])
            rmse, peak, terminal = audit.metrics(error[:, branch])
            passed = bool(stable and rmse.max()<3. and peak.max()<5. and abs(terminal).max()<1. and not blocked[0])
            all_pass &= passed
            branches.append(dict(input_branch=key, rmse_deg=rmse.tolist(), peak_deg=peak.tolist(),
                terminal_deg=abs(terminal).tolist(), max_all_six_pin_gaps_mm=float(all_gaps.max()*1000),
                max_limit_excess_deg=physical['max_limit_excess_deg'],
                first_triangle_area_sign_change_s=physical['first_triangle_area_sign_change_s'],
                numerical_loop_stability_passed=stable, response_thresholds_passed=passed))
        with np.load(directory/'trials.npz') as sampled:
            audit.close(sampled['qpos'][:-1], q[::5], '1ms state archive', 0.)
            audit.close(sampled['qvel'][:-1], v[::5], '1ms velocity archive', 0.)
        for branch, key in enumerate(('recorded_torque', 'recorded_target')):
            sampled_error = error[::5, branch]
            rmse, peak, terminal = audit.metrics(sampled_error)
            for field, value in [('rmse_deg', rmse), ('peak_deg', peak), ('terminal_deg', abs(terminal))]:
                audit.close(value, result['results'][0][key][field], 'Reported '+key+' '+field, 1e-5)
        rows[name] = dict(branches=branches, submitted_control_checks=dict(spring_max_error_n=spring_error,
            motor_friction_capacity_max_error_nm=friction_error, feedback_motor_max_error_nm=feedback_error),
            profile_equal_to_frozen_selected=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        stride = 50
        for j, ax in enumerate(axes.flat):
            ax.plot(tfull[::stride], np.degrees(real[::stride, j]-real[0, j]), label='Measured')
            for b, label in enumerate(('Recorded torque', 'Recorded target feedback')):
                prediction = (qfull[:, b, mq[j]]-qfull[0, b, mq[j]])*inputs['sign'][j]
                ax.plot(tfull[::stride], np.degrees(prediction[::stride]), label=label, alpha=.8)
            ax.set_title(fit.prior.replay.mirror.MOTORS[j]); ax.set_ylabel('Angle change (deg)'); ax.set_xlabel('Time (s)'); ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8); fig.suptitle(name+' - repaired loops; one frozen parameter set; response fit provisional')
        fig.tight_layout(); fig.savefig(args.output/(name+'-joint-response.png'), dpi=150); plt.close(fig)
        print(json.dumps(dict(verified_session=name, results=branches)), flush=True)
    protected = audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), [])
    parity={}
    for value,expected in parent.get('pool_report_hashes',{}).items():
        path=Path(value);assert audit.digest(path)==expected
        pool=json.loads(path.read_text());name=path.parent.name;index=parent['selected_index'];count=len(pool['results'])
        assert pool['results'][index]['parameters']==parent['selected_parameters']
        with np.load(path.with_name('trials.npz')) as batched, np.load(args.experiment/('final-'+name)/'trials.npz') as single:
            model=mujoco.MjModel.from_xml_path(str(args.experiment/('final-'+name)/'model.xml'))
            mq=[model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
            stride=int(round(pool['saved_dt_s']/.001))
            assert np.allclose(single['time_s'][::stride],batched['time_s'],atol=1e-12)
            difference=np.degrees(single['qpos'][::stride,:,:][:,:,mq]-batched['qpos'][:,[index,count+index],:][:,:,mq])
            parity[name]=dict(max_same_grid_motor_difference_deg=abs(difference).max(0).tolist(),
                             saved_positions_bitwise_identical=bool(np.all(difference==0)))
    result = dict(integrity_passed=True, response_thresholds_passed=all_pass, sessions=rows,
                  audited_sessions=list(rows), all_four_recordings_audited=len(rows)==len(fit.REFERENCES),
                  thresholds=dict(joint_rmse_deg=3., joint_peak_deg=5., terminal_deg=1., pin_gap_mm=.1),
                  verified_gpu_environment_steps=parent.get('verified_gpu_environment_steps',parent['new_gpu_environment_steps']),
                  new_gpu_environment_steps=parent['new_gpu_environment_steps'], new_physics_steps_in_report=0,
                  protected_source_checks=protected, physical_parameter_arrays_preserved=fields,
                  single_vs_batch_same_grid=parity,solver_settings=solver_settings,
                  runtime_solver_options=dict(integrator='implicitfast', jacobian='dense', iterations=100,
                                              tolerance=1e-10, impratio=10., timestep_s=.0002),
                  shared_model_modified=False, hardware_access=False, policy_training_updates=0,
                  limitations=['All four recordings are existing development/regression data, not a new blind hardware trial.',
                               'Estimated spring/friction are effective parameters, not independently measured component values.',
                               'The XML is a template; the fitting script explicitly sets runtime solver options and per-world resistance.'])
    search_path = args.output/'verification.json'; fit.search.write_json(search_path, result)
    print(json.dumps(dict(integrity_passed=True, response_thresholds_passed=all_pass)), flush=True)


if __name__ == '__main__': main()
