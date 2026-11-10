"""Verify saved GPU right-leg traces against actual C requests and independent geometry."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

from scripts.agent.real2sim.precheck_right_leg_pose import State, mirror, leg_pose


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    h7 = root / 'deployment/H7_RL_ros'
    algorithm = h7 / 'imcalib/Algorithm'
    with np.load(args.run / 'traces.npz') as saved:
        trace = {key: saved[key] for key in saved.files}
    assert all(np.isfinite(value).all() for value in trace.values())
    assert len(trace['time_s']) == 18501 and np.max(abs(np.diff(trace['time_s']) - .002)) < 1e-12
    model = mujoco.MjModel.from_xml_path(str(args.run / 'model.xml'))
    pose = mirror.MirrorPose(machine_config=h7 / 'imcalib/config/machine_config.c')
    mq = [model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]]
    mv = [model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]]
    q, v = trace['qpos'], trace['qvel']
    count = q.shape[1]
    sources = [algorithm / name for name in ('joint_capture.c', 'joint_chirp.c')]
    ffour = ctypes.c_float * 4
    states = [State() for _ in range(count)]
    previous = trace.get('initial_previous_firmware_nm', np.zeros((count, 4))).copy()
    stops = [None] * count
    request_error, target_error, integral_peak = 0., 0., 0.
    signs = (ctypes.c_int8 * 6)(*pose.sign.astype(int))
    zero = ffour(*pose.zero)
    kp, kd = ffour(60, 60, 60, 60), ffour(2, 2, 2, 2)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'verify.so'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-shared', '-fPIC', *map(str, sources), '-lm', '-o', str(path)], check=True)
        lib = ctypes.CDLL(str(path))
        update = lib.Joint_Capture_Update
        update.argtypes = [ctypes.POINTER(State), ctypes.c_uint32, ctypes.c_uint8, *([ctypes.POINTER(ctypes.c_float)] * 3), ctypes.POINTER(ctypes.c_int8), *([ctypes.POINTER(ctypes.c_float)] * 3), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        guard = lib.Joint_Capture_Geometry
        guard.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float]
        for step in range(len(q)):
            for trial in range(count):
                fw_q = q[step, trial, mq] * pose.sign[:4] + pose.zero
                output = ffour()
                valid = guard(ffour(*fw_q), .21, .25)
                ok = update(ctypes.byref(states[trial]), step * 2, valid, ffour(*fw_q), ffour(*(v[step, trial, mv] * pose.sign[:4])), ffour(*previous[trial]), signs, zero, kp, kd, .21, .25, 40., output)
                if not ok and stops[trial] is None: stops[trial] = step * .002
                previous[trial] = list(output)
                request_error = max(request_error, float(np.max(abs(previous[trial] * pose.sign[:4] - trace['torque_nm'][step, trial]))))
                target_error = max(target_error, float(np.max(abs(np.array(states[trial].target) - trace['target_q'][step, trial]))))
                integral_peak = max(integral_peak, max(abs(value) for value in states[trial].integral))
    length, angle, _ = leg_pose(q[:, :, mq] * pose.sign[:4] + pose.zero)
    data = mujoco.MjData(model)
    max_gap, length_parity, angle_parity = 0., 0., 0.
    for step in range(0, len(q), 5):
        for trial in range(count):
            data.qpos[:] = q[step, trial]
            mujoco.mj_kinematics(model, data)
            max_gap = max(max_gap, max(np.linalg.norm(data.site(a).xpos - data.site(b).xpos) for group in mirror.PIN_PAIRS for a, b in group))
            delta = data.body('base_link').xmat.reshape(3, 3).T @ (data.xanchor[model.joint('rfwheel').id] - data.xanchor[model.joint('rf0').id])
            length_parity = max(length_parity, abs(np.hypot(delta[0], delta[2]) - length[step, trial, 1]))
            angle_parity = max(angle_parity, abs(np.arctan2(delta[0], -delta[2]) - angle[step, trial, 1]))
    lerror = np.max(abs(length[-1500:, :, 1] - .19), axis=0) * 1000
    metadata=json.loads((args.run/'report.json').read_text()) if (args.run/'report.json').exists() else {}
    target_angle=np.deg2rad(metadata.get('target_right_angle_deg',30.))
    aerror = np.degrees(np.max(abs(angle[-1500:, :, 1] - target_angle), axis=0))
    peak = np.max(abs(trace['torque_nm']), axis=(0, 1))
    joint = model.joint('rf1')
    margin = float(np.minimum(q[:, :, joint.qposadr[0]] - joint.range[0], joint.range[1] - q[:, :, joint.qposadr[0]]).min())
    report = dict(passed=bool(all(stop is None for stop in stops) and request_error < 1e-5 and target_error < 1e-6 and max(lerror) < 1 and max(aerror) < .2 and max_gap < .0001 and margin > 0 and not np.any(trace['torque_nm'][:, :, :2]) and peak.max() < 40),
                  guard_stop_s=stops, last3s_right_length_max_error_mm=lerror.tolist(), last3s_right_angle_max_error_deg=aerror.tolist(),
                  final_right_length_mm=(length[-1, :, 1] * 1000).tolist(), final_right_angle_deg=np.degrees(angle[-1, :, 1]).tolist(),
                  peak_request_nm=peak.tolist(), peak_right_speed_rad_s=float(np.max(abs(v[:, :, mv[2:]]))),
                  peak_moving_phase_right_speed_rad_s=float(np.max(abs(v[2500:, :, mv[2:]]))),
                  min_right_knee_margin_rad=margin, max_pin_gap_mm=max_gap * 1000,
                  geometry_length_max_error_mm=length_parity * 1000, geometry_angle_max_error_deg=float(np.degrees(angle_parity)),
                  actual_C_request_reconstruction_max_nm=request_error, target_reconstruction_max_rad=target_error,
                  integral_peak_nm=integral_peak, local_load_factors=[.7, 1., 1.3], synthetic_prepared_length_mm=255,
                  profile=metadata.get('profile','pose'),target_right_angle_deg=float(np.degrees(target_angle)),
                  verified_saved_gpu_environment_steps=18500 * 10 * count, new_gpu_environment_steps=0,
                  hardware_commands=False, scope='Actual C necessary-condition test on synthetic255mm initial pose; not real robot proof',
                  report_recovery='Reconstruct C/guards and geometry from saved states without rerunning physics; this also recovers the -04 numpy.bool_ reporting failure.')
    report['slow_holding_torque_entry_passed'] = bool(np.max(abs(v[:, :, mv[2:]])) < .15) if np.any(trace.get('initial_previous_firmware_nm', 0)) else None
    if report['slow_holding_torque_entry_passed'] is False: report['passed'] = False
    if metadata.get('profile')=='length-only':
        command_angle=leg_pose(trace['target_q']*pose.sign[:4]+pose.zero)[1][:,:,1]
        report['max_target_angle_drift_deg']=float(np.degrees(np.max(abs(command_angle-target_angle))))
        report['max_actual_angle_deviation_deg']=np.degrees(np.max(abs(angle[:,:,1]-target_angle),axis=0)).tolist()
        report['moving_max_actual_angle_deviation_deg']=np.degrees(np.max(abs(angle[2500:12501,:,1]-target_angle),axis=0)).tolist()
        report['passed']=bool(report['passed'] and report['max_target_angle_drift_deg']<.0001)
    for source in (*sources, algorithm / 'joint_capture.h', algorithm / 'joint_chirp.h', algorithm / 'joint_pose_load.h'):
        shutil.copy2(source, args.run / source.name)
    protected = [mirror.MODEL, h7 / 'imcalib/config/machine_config.c']
    report['source_sha256'] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in (*sources, *protected, algorithm / 'joint_pose_load.h')}
    (args.run / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
