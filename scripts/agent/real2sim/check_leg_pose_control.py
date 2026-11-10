"""Independently check a saved GPU leg-pose control run without further integration."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    params = json.loads((args.run / 'parameters.json').read_text())
    assert hashlib.sha256((args.run / 'parameters.json').read_bytes()).hexdigest() == report['frozen_parameters_sha256']
    for source, expected in report['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
    model = mujoco.MjModel.from_xml_path(str(args.run / 'model.xml'))
    trace = np.load(args.run / 'traces.npz')
    assert all(np.isfinite(trace[key]).all() for key in trace.files)
    mq = [model.joint(n).qposadr[0] for n in ('lf0', 'lf00', 'rf0', 'rf00')]
    mv = [model.joint(n).dofadr[0] for n in ('lf0', 'lf00', 'rf0', 'rf00')]
    q, v = trace['qpos'][:, mq], trace['qvel'][:, mv]
    control = report['controller']
    error = trace['target_q'] - q
    raw = trace['feedforward_nm'] + control['kp'] * error + control['kd'] * (trace['target_v'] - v) + trace['integral_nm']
    request_error = float(abs(np.clip(raw, -40, 40) - trace['request_nm']).max())
    expected_integral = np.clip(trace['integral_nm'][:-1] + control['ki'] * error[:-1] * .002 * ((abs(raw[:-1]) < 40) | (raw[:-1] * error[:-1] < 0)), -8, 8)
    integral_error = float(abs(expected_integral - trace['integral_nm'][1:]).max())
    load = -(np.asarray(params['bias_nm']) + (q - params['reference_q_model_rad']) @ np.asarray(params['stiffness_nm_rad']).T)
    load_error = float(abs(load - trace['applied_load_nm']).max())
    # 轮轴与髋轴的真实模型坐标，独立核对关节角公式得到的长度/角度。
    geometry_length_error = float(abs(trace['measured_geometry'][:, 0] - trace['actual_length_m'][::5]).max() * 1000)
    geometry_angle_error = float(np.degrees(abs(trace['measured_geometry'][:, 1] - trace['actual_angle_rad'][::5])).max())
    settled = trace['time_s'] >= 13
    final_length_error = float(abs(trace['actual_length_m'][settled] * 1000 - report['target_length_mm']).max())
    final_angle_error = float(abs(np.degrees(trace['actual_angle_rad'][settled]) - report['target_angle_deg']).max())
    knee_margins = []
    for name in ('lf1', 'rf1'):
        joint = model.joint(name)
        values = trace['qpos'][:, joint.qposadr[0]]
        knee_margins.append(float(np.minimum(values - joint.range[0], joint.range[1] - values).min()))
    checks = dict(request_reconstruction_max_nm=request_error, integral_reconstruction_max_nm=integral_error,
                  load_reconstruction_max_nm=load_error, geometry_length_max_error_mm=geometry_length_error,
                  geometry_angle_max_error_deg=geometry_angle_error, last3s_max_length_error_mm=final_length_error,
                  last3s_max_angle_error_deg=final_angle_error, min_knee_margin_rad=knee_margins,
                  max_pin_gap_mm=report['max_pin_gap_mm'], new_physics_steps=0, hardware_access=False)
    checks['passed'] = bool(request_error < 1e-6 and integral_error < 1e-6 and load_error < 2e-6
                            and geometry_length_error < .01 and geometry_angle_error < .01
                            and final_length_error < 1 and final_angle_error < .2
                            and min(knee_margins) > 0 and report['max_pin_gap_mm'] < .1
                            and report['guard_stop_s'] is None)
    (args.run / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))
    assert checks['passed']


if __name__ == '__main__':
    main()
