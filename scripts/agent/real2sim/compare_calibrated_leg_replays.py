"""Compare old/calibrated knee ranges with identical GPU inputs and training springs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads((path / 'report.json').read_text()) for path in (args.before, args.after)]
    states = [np.load(path / 'replay.npz') for path in (args.before, args.after)]
    for key in ('stream_sha256', 'config_sha256', 'dt_s', 'duration_s', 'physics_backend',
                'worlds', 'steps_per_world', 'constraint_check', 'solver_options', 'spring_model',
                'spring_damping_ns_m', 'training_spring_source_sha256'):
        assert reports[0][key] == reports[1][key], key
    assert reports[0]['spring_model'] == 'linear_0_150N_90mm_CAD10mm_damping50'
    assert reports[1]['calibrated_joint_range_changes'] and not reports[0]['calibrated_joint_range_changes']
    for key in ('time_s', 'real_geometry', 'real_motor_state', 'initial_qpos', 'initial_qvel'):
        assert np.array_equal(states[0][key], states[1][key]), key
    for key in ('qpos', 'qvel'):
        assert np.array_equal(states[0][key][0], states[1][key][0]), 'Initial ' + key
    assert np.array_equal(states[0]['submitted_model_torque'][:, 0], states[1]['submitted_model_torque'][:, 0])

    trees = [ET.parse(p / 'fixed-base-model.xml').getroot() for p in (args.before, args.after)]
    mesh_digests = []
    for tree in trees:
        meshdir = Path(tree.find('compiler').get('meshdir'))
        mesh_digests.append({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in meshdir.iterdir() if p.is_file()})
    assert mesh_digests[0] == mesh_digests[1], 'Mesh geometry differs'
    trees[1].find('compiler').set('meshdir', trees[0].find('compiler').get('meshdir'))
    changes = {}
    for name in ('lf1', 'rf1'):
        old, new = [tree.find(f".//joint[@name='{name}']") for tree in trees]
        changes[name] = dict(before=old.get('range'), after=new.get('range'))
        new.set('range', old.get('range'))
    assert ET.tostring(trees[0]) == ET.tostring(trees[1]), 'Unexpected model change'

    model = mujoco.MjModel.from_xml_path(str(args.after / 'fixed-base-model.xml'))
    qids = [model.joint(name).qposadr[0] for name in replay.mirror.MOTORS[:4]]
    sq = [model.joint(side + '_spring2_joint').qposadr[0] for side in ('left', 'right')]
    sv = [model.joint(side + '_spring2_joint').dofadr[0] for side in ('left', 'right')]
    spring_errors = []
    for state in states:
        expected = replay.training_spring.gas_spring_force(state['qpos'][:-1,:,sq].astype(np.float32),
                                                          state['qvel'][:-1,:,sv].astype(np.float32))
        error = float(np.max(np.abs(expected - state['spring_force_n'])))
        assert error < 2e-5, error
        spring_errors.append(error)
    assert hashlib.sha256(Path(replay.training_spring.__file__).read_bytes()).hexdigest() == reports[1]['training_spring_source_sha256']
    config = Path(reports[1]['session']) / 'meta.json'
    config_path = Path(json.loads(config.read_text())['machine_config']['path'])
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == reports[1]['config_sha256']
    sign, zero = replay.mirror.load_mapping(config_path)
    time = states[0]['time_s']
    absolute_time = time + reports[1]['window_start_board_s']
    measured = states[0]['real_motor_state']
    real_q = np.column_stack([np.interp(absolute_time, measured[:,0], measured[:,1 + i]) for i in range(4)])
    simulated = [np.take(s['qpos'], qids, axis=2) * sign[:4] + zero for s in states]
    metrics = {}
    for branch, name in enumerate(('recorded_torque', 'recorded_target_feedback')):
        metrics[name] = {}
        for label, report, sim in zip(('before', 'after'), reports, simulated):
            rmse = np.sqrt(np.mean((sim[:,branch] - real_q) ** 2, axis=0))
            assert np.allclose(rmse, report['metrics'][name]['motor_position_rmse_rad'], atol=1e-10)
            metrics[name][label] = dict(motor_rmse_deg=np.degrees(rmse).tolist(),
                combined_motor_rmse_deg=float(np.degrees(np.sqrt(np.mean(rmse ** 2)))),
                leg_length_rmse_mm=(1000 * np.array(report['metrics'][name]['encoder_solved_leg_length_rmse_m'])).tolist(),
                knee_limits=report['metrics'][name]['knee_limits'])
        metrics[name]['improved_motor_count'] = int(np.count_nonzero(np.array(metrics[name]['after']['motor_rmse_deg']) < metrics[name]['before']['motor_rmse_deg']))
    result = dict(verified_only_knee_range_change=True, verified_same_inputs_and_initial_states=True,
                  spring_model=reports[1]['spring_model'], spring_force_recompute_error_n=spring_errors,
                  knee_range_changes=changes, before_initial_limit_excess_rad=reports[0]['initial_joint_limit_excess_rad'],
                  after_initial_limit_excess_rad=reports[1]['initial_joint_limit_excess_rad'], metrics=metrics,
                  before_feedback_blocked=reports[0]['feedback_branch_blocked'], after_feedback_blocked=reports[1]['feedback_branch_blocked'],
                  new_environment_steps=sum(r['new_environment_steps'] for r in reports), postprocess_steps=0,
                  policy_forwards=0, optimizer_updates=0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_time = absolute_time - reports[1]['first_dial_board_s']
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    for i, motor in enumerate(replay.mirror.MOTORS[:4]):
        for branch in range(2):
            ax = axes[i,branch]
            ax.plot(plot_time,np.degrees(real_q[:,i]),color='black',label='Recorded robot')
            for label, sim in zip(('Old limits', 'Measured limits'), simulated):
                ax.plot(plot_time,np.degrees(sim[:,branch,i]),label=label,alpha=.85)
            ax.set_ylabel(motor+' (deg)');ax.grid(alpha=.25)
    axes[0,0].set_title('Same recorded torque');axes[0,1].set_title('Same recorded length targets + feedback')
    axes[0,0].legend();axes[0,1].legend()
    for ax in axes[-1]:ax.set_xlabel('Time from first dial input (s)')
    fig.tight_layout();fig.savefig(args.output/'motor-comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for branch,ax in enumerate(axes):
        for label,state in zip(('Old','Measured'),states):
            for side,style in enumerate(('-', '--')):
                ax.plot(plot_time[:-1],state['knee_limit_force_nm'][:,branch,side],style,label=label+' '+('lf1','rf1')[side])
        ax.set_ylabel(('Torque replay' if branch==0 else 'Target feedback')+'\nlimit reaction (Nm)');ax.grid(alpha=.25);ax.legend()
    axes[-1].set_xlabel('Time from first dial input (s)');fig.tight_layout()
    fig.savefig(args.output/'knee-limit-reactions.png',dpi=150);plt.close(fig)
    for source in (Path(__file__),Path(replay.__file__),Path(replay.training_spring.__file__)):
        dest=args.output/'source'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
