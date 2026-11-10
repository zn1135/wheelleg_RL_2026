"""Independently check saved GPU loop replays and physical-parameter preservation; zero steps."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import diagnose_leg_loop_failure as audit
from scripts.agent.real2sim import report_leg_response_joint as joint_report


def all_gaps(model, qpos):
    pairs = [p for leg in audit.replay.mirror.PIN_PAIRS for p in leg]
    pairs += [(s+'_spring_tip', s+'_gas_spring_lower') for s in ('left', 'right')]
    ids = np.array([[model.site(n).id for n in pair] for pair in pairs])
    data = mujoco.MjData(model); result = []
    for q in qpos:
        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
        positions = data.site_xpos[ids]
        result.append(np.linalg.norm(positions[:, 0]-positions[:, 1], axis=1))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('artifacts/real2sim/h7-real-vs-mujoco-view-20261104-01'))
    parser.add_argument('--diagnostics', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sessions = {}; total = 0; all_pass = True; hashes = {}
    for directory in args.diagnostics:
        summary = json.loads((directory/'report.json').read_text()); total += summary['gpu_environment_steps']
        for name in summary['sessions']:
            baseline = args.baseline/name
            meta = json.loads((baseline/'report.json').read_text())
            for path, digest in meta['source_reference_hashes'].items():
                assert audit.sha(Path(path)) == digest; hashes[path] = digest
            assert audit.sha(baseline/'comparison.npz') == meta['comparison_sha256']
            assert audit.sha(Path(meta['session'])/'stream.jsonl') == meta['recording_sha256']
            old = np.load(baseline/'comparison.npz'); source = mujoco.MjModel.from_xml_path(str(baseline/'source-model.xml'))
            model_path = directory/name/'limits_fine-model.xml'
            model = mujoco.MjModel.from_xml_path(str(model_path))
            # 逐项核对只改变约束软硬度和步长，几何、质量、关节范围及执行器保持。
            fields = ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
                      'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'qpos0', 'eq_data', 'eq_obj1id', 'eq_obj2id',
                      'site_pos', 'dof_armature', 'dof_damping', 'dof_frictionloss', 'actuator_gear',
                      'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange']
            assert all(np.array_equal(getattr(source, f), getattr(model, f)) for f in fields)
            assert model.opt.timestep == .0002
            saved = np.load(directory/name/'limits_fine-replay.npz')
            assert np.array_equal(saved['submitted_model_torque_nm'], old['submitted_model_torque_nm'])
            times = saved['time_s']; q = saved['qpos']; rows = {}
            assert len(q) == len(old['submitted_model_torque_nm'])+1
            real = np.column_stack([np.interp(times, old['time_s'], old['qpos'][:, 0, j]) for j in range(model.nq)])
            for i, gas in enumerate(('on', 'off')):
                key = 'limits_fine_gas_'+gas
                result, gaps, area, excess = audit.metrics(model, times, q[:, i], real)
                full_gaps = all_gaps(model, q[:, i])
                assert np.allclose(full_gaps[:, :4], gaps, atol=1e-12)
                prior = summary['sessions'][name][key]
                for field in ['max_pin_gap_mm', 'max_limit_excess_deg', 'joint_rmse_deg']:
                    assert np.allclose(result[field], prior[field], atol=1e-10)
                passed = bool(full_gaps.max()<.0001 and excess.max()<np.radians(.1)
                              and result['first_triangle_area_sign_change_s'] == [None, None])
                all_pass &= passed
                rows[key] = dict(**result, max_all_six_pin_gaps_mm=float(full_gaps.max()*1000),
                                 numerical_loop_stability_passed=passed,
                                 response_fit_passed=bool(max(result['joint_rmse_deg'])<3
                                     and max(result['joint_peak_error_deg'])<5
                                     and max(result['joint_terminal_error_deg'])<1 and passed))
            sessions[name] = rows
            if name == 'original':
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                for branch, label, color in [(0, 'Measured pose reconstruction', '#0088aa'),
                                             (1, 'Original loop constraints', '#dd6611'),
                                             (3, 'Stiff loops only', '#559944'),
                                             (None, 'Stiff loops + limits; 0.2 ms', '#aa55aa')]:
                    t, state = (times, q[:, 0]) if branch is None else (old['time_s'], old['qpos'][:, branch])
                    gaps, area, ex, _ = audit.geometry(source, state)
                    axes[0].semilogy(t, np.maximum(gaps.max(1)*1000, 1e-6), label=label, color=color)
                    axes[1].plot(t, area[:, 0]*1e4, color=color, label=label)
                    j = source.joint('lf00').qposadr[0]
                    axes[2].plot(t, np.degrees(state[:, j]-state[0, j]), color=color, label=label)
                axes[0].axhline(.1, color='red', linestyle='--', linewidth=1)
                axes[0].set_ylabel('Main loop pin gap (mm)'); axes[0].legend(fontsize=8, ncol=2)
                axes[1].axhline(0., color='red', linestyle='--', linewidth=1)
                axes[1].set_ylabel('Left signed triangle area (cm²)')
                axes[2].set_ylabel('Left rear motor change (deg)'); axes[2].set_xlabel('Recorded time (s)')
                for axis in axes: axis.grid(alpha=.2)
                fig.suptitle('Same recorded torque, gas spring ON; physical parameters unchanged')
                fig.tight_layout(); fig.savefig(args.output/'loop-failure-before-after.png', dpi=150); plt.close(fig)
    result = dict(integrity_passed=True, numerical_loop_stability_passed=all_pass,
                  verified_gpu_environment_steps=total, sessions=sessions,
                  new_physics_steps_in_report=0, source_inputs_sha256=hashes,
                  protected_source_checks=joint_report.protected_sources(
                      Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), []),
                  physical_parameter_arrays_checked=fields, shared_model_modified=False,
                  response_fit_accepted=False, hardware_access=False,
                  limitations=['Real visualization is reconstructed from motor encoders, not a camera.',
                               'Numerical closure and limits are repaired only in diagnostic copies.',
                               'Remaining large response errors are not explained or fitted by this repair.'])
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(integrity_passed=True, numerical_loop_stability_passed=all_pass,
                          verified_gpu_environment_steps=total, response_fit_accepted=False)), flush=True)


if __name__ == '__main__': main()
