"""Compare saved leg-loop GPU diagnostics; no integration or hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.baseline, *args.candidate]
    reports = [json.loads((p / 'report.json').read_text()) for p in paths]
    states = [np.load(p / 'replay.npz') for p in paths]
    residuals = [np.load(p / 'constraint-residuals.npz') for p in paths]
    baseline_xml = ET.parse(paths[0] / 'fixed-base-model.xml').getroot()
    baseline = reports[0]
    changes = []
    for path, report, state in zip(paths, reports, states):
        if report.get('spring_model','constant_150N')!=baseline.get('spring_model','constant_150N'):
            raise ValueError(f'{path}: different spring model; this comparison isolates loop constraints')
        for key in ('stream_sha256', 'source_model_sha256', 'config_sha256',
                    'dt_s', 'duration_s', 'physics_backend', 'worlds', 'steps_per_world',
                    'extended_joint_limits_rad', 'disabled_joint_limits'):
            if report[key] != baseline[key]:
                raise ValueError(f'{path}: different {key}')
        for key in ('time_s', 'real_geometry', 'real_motor_state'):
            if not np.array_equal(state[key], states[0][key]):
                raise ValueError(f'{path}: different input {key}')
        for key in ('qpos', 'qvel'):
            if not np.array_equal(state[key][0], states[0][key][0]):
                raise ValueError(f'{path}: different initial {key}')
        # 反馈分支力矩随模拟状态变化；只有直接力矩分支必须逐拍相同。
        if not np.array_equal(state['submitted_model_torque'][:, 0],
                              states[0]['submitted_model_torque'][:, 0]):
            raise ValueError(f'{path}: different replayed torque')
        xml = ET.parse(path / 'fixed-base-model.xml').getroot()
        delta = {}
        option = xml.find('option')
        old_option = baseline_xml.find('option')
        if option is not None and old_option is None:
            if set(option.attrib) != {'iterations', 'tolerance', 'ls_iterations', 'ls_tolerance'}:
                raise ValueError('Unexpected solver option changes')
            delta['solver_options'] = dict(option.attrib)
            xml.remove(option)
        for connection in xml.find('equality').findall('connect'):
            old = baseline_xml.find(f"equality/connect[@name='{connection.get('name')}']")
            for key in ('solref', 'solimp'):
                if connection.get(key) != old.get(key):
                    delta[f"{connection.get('name')}.{key}"] = connection.get(key)
                    connection.set(key, old.get(key))
        if ET.tostring(xml) != ET.tostring(baseline_xml):
            raise ValueError(f'{path}: changes beyond solver/loop compliance')
        changes.append(delta)
    summary = {}
    for path, report in zip(paths, reports):
        summary[path.name] = {}
        for name, metrics in report['metrics'].items():
            summary[path.name][name] = dict(
                leg_length_rmse_mm=(np.asarray(metrics['encoder_solved_leg_length_rmse_m']) * 1000).tolist(),
                motor_angle_rmse_deg=np.degrees(metrics['motor_position_rmse_rad']).tolist(),
                max_pin_residual_mm=(np.asarray(metrics['max_loop_pin_residual_m']) * 1000).tolist(),
                max_spring_residual_mm=(np.asarray(metrics['max_spring_loop_residual_m']) * 1000).tolist(),
                solver_iterations=metrics.get('solver_iterations'))
    result = dict(metrics=summary, verified_same_inputs_and_initial_states=True,
                  model_attribute_changes=changes,
                  new_gpu_environment_steps=sum(r['new_environment_steps'] for r in reports),
                  postprocess_environment_steps=0, network_forwards=0, optimizer_updates=0)
    result['solver_only_state_arrays_identical'] = {
        path.name: all(np.array_equal(state[key], states[0][key]) for key in
                       ('qpos', 'qvel', 'submitted_model_torque'))
        for path, report, state in zip(paths, reports, states)
        if report.get('constraint_check') == 'solver'}
    # 仅计算锚点位置，核对约束连接点；不推进物理步。
    model = mujoco.MjModel.from_xml_path(str(paths[0] / 'fixed-base-model.xml'))
    pose = mujoco.MjData(model)
    pose.qpos[:] = states[0]['qpos'][0, 0]
    mujoco.mj_kinematics(model, pose)
    result['initial_equality_anchor_gap_m'] = {}
    for index in range(model.neq):
        first, second = model.eq_obj1id[index], model.eq_obj2id[index]
        if model.eq_objtype[index] == mujoco.mjtObj.mjOBJ_BODY:
            one = pose.xpos[first] + pose.xmat[first].reshape(3, 3) @ model.eq_data[index, :3]
            two = pose.xpos[second] + pose.xmat[second].reshape(3, 3) @ model.eq_data[index, 3:6]
        elif model.eq_objtype[index] == mujoco.mjtObj.mjOBJ_SITE:
            one, two = pose.site_xpos[first], pose.site_xpos[second]
        else:
            raise ValueError('Unexpected loop anchor object type')
        result['initial_equality_anchor_gap_m'][model.equality(index).name] = float(np.linalg.norm(one - two))
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__).resolve(), root / 'scripts/agent/real2sim/replay_leg_test_gpu.py',
               root / 'sim2sim/real2sim_huananhu.py', root / 'sim2sim/wheel_response.py',
               root / 'deployment/H7_RL_ros/imcalib/config/machine_config.c',
               Path(baseline['source_model'])]
    result['source_sha256'] = {}
    for source in sources:
        relative = source.relative_to(root)
        result['source_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = args.output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if result['source_sha256'][str(sources[-1].relative_to(root))] != baseline['source_model_sha256']:
        raise ValueError('Source model changed since replay')
    if result['source_sha256']['deployment/H7_RL_ros/imcalib/config/machine_config.c'] != baseline['config_sha256']:
        raise ValueError('Source mapping changed since replay')
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for branch, ax in enumerate(axes):
        for report, residual in zip(reports, residuals):
            label = report.get('constraint_check') or 'baseline'
            ax.plot(residual['time_s'], np.max(residual['pin_residual_m'][:, branch], axis=1) * 1000,
                    label=label)
        ax.set_ylabel(('Torque replay' if branch == 0 else 'Target feedback') + '\npin gap (mm)')
        ax.grid(alpha=.3)
        ax.legend()
    axes[-1].set_xlabel('Time from replay start (s)')
    fig.tight_layout()
    fig.savefig(args.output / 'loop-residual-comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
