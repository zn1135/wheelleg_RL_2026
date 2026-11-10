"""Export the frozen right-leg model hold-torque table for the slow hardware trial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from scripts.agent.real2sim.control_leg_pose_gpu import inverse_pose, mirror, initial_preload, gas_spring_force


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    reference = root / 'artifacts/pace-joint-capture/20261108-lowfreq-load-fit-01'
    parameters = reference / 'parameters.json'
    params = json.loads(parameters.read_text())
    model = mujoco.MjModel.from_xml_path(str(reference / 'gpu-replay/model.xml'))
    data = mujoco.MjData(model)
    pose = mirror.MirrorPose(machine_config=root / 'deployment/H7_RL_ros/imcalib/config/machine_config.c')
    source_q = [pose.model.joint(model.joint(j).name).qposadr[0] for j in range(model.njnt)]
    sq = [model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')]
    qref = np.array(params['reference_q_model_rad'])
    stiffness = np.array(params['stiffness_nm_rad'])
    lengths = np.arange(180, 341, 10) / 1000
    angles = np.deg2rad(np.arange(-10, 41, 5))

    def torque(length, angle):
        q = qref.copy()
        q[2:] = (inverse_pose(length, angle) - pose.zero[2:]) * pose.sign[2:4]
        for side in (0, 1):
            assert pose._solve_leg(side, q[2 * side:2 * side + 2])
        data.qpos[:] = pose.data.qpos[source_q]
        data.qvel[:] = 0
        residual, _ = initial_preload(model, data, np.zeros(4), initial_spring_force=gas_spring_force(data.qpos[sq], np.zeros(2)))
        return (-residual + params['bias_nm'] + (q - qref) @ stiffness.T)[2:]

    table = np.array([[torque(length, angle) for angle in angles] for length in lengths])
    errors = []
    for length in (lengths[:-1] + lengths[1:]) / 2:
        for angle in (angles[:-1] + angles[1:]) / 2:
            i = np.searchsorted(lengths, length) - 1
            j = np.searchsorted(angles, angle) - 1
            errors.append(table[i:i + 2, j:j + 2].mean(axis=(0, 1)) - torque(length, angle))
    digest = hashlib.sha256(parameters.read_bytes()).hexdigest()
    rows = ',\n'.join('    {' + ', '.join('{' + ', '.join(f'{v:.9g}f' if not float(v).is_integer() else f'{v:.1f}f' for v in pair) + '}' for pair in row) + '}' for row in table)
    header = ('/* Right-leg MODEL-coordinate hold torque, Nm; frozen local-load candidate ' + digest + '.\n'
              ' * Length180..340mm/10mm, body-vertical angle -10..40deg/5deg.\n'
              ' * Includes nominal gravity/spring + fitted local conservative load; excludes dry friction.\n'
              ' * Diagnostic feedforward only; no encoder/polarity/torque-scale calibration. */\n'
              '#ifndef JOINT_POSE_LOAD_H\n#define JOINT_POSE_LOAD_H\n'
              'static const float joint_pose_load[17][11][2] = {\n' + rows + '\n};\n#endif\n')
    (args.output / 'joint_pose_load.h').write_text(header)
    (args.output / 'parameters.json').write_bytes(parameters.read_bytes())
    report = dict(parameters_sha256=digest, side='robot_right', length_mm=[180, 340, 10], angle_deg=[-10, 40, 5],
                  max_midpoint_interpolation_error_nm=np.max(abs(np.array(errors)), axis=0).tolist(),
                  new_physics_steps=0, hardware_commands=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
