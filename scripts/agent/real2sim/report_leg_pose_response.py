"""Audit length/tilt and wheel-position errors from saved 0.2ms GPU states; no integration."""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_length_response as pose_fit
from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def summarize(real, simulated):
    rl, rt, rp = pose_fit.leg_pose(real)
    sl, st, sp = pose_fit.leg_pose(simulated)
    length = (sl-rl[:, None])*1000
    tilt = np.degrees(np.arctan2(np.sin(st-rt[:, None]), np.cos(st-rt[:, None])))
    endpoint = np.linalg.norm(sp-rp[:, None], axis=-1)*1000
    rows = {}
    for index, branch in enumerate(('recorded_torque', 'recorded_target')):
        rows[branch] = {}
        for prefix, values, unit in [('length', length, 'mm'), ('tilt', tilt, 'deg'), ('endpoint', endpoint, 'mm')]:
            e = values[:, index]
            rows[branch].update({prefix+'_rmse_'+unit:np.sqrt(np.mean(e*e, axis=0)).tolist(),
                prefix+'_peak_'+unit:abs(e).max(0).tolist(), prefix+'_terminal_'+unit:abs(e[-1]).tolist()})
    return rows


def board_geometry(inputs):
    session = Path(inputs['report']['session']); decoded = session.with_name(session.name+'-output')
    report = json.loads((decoded/'report.json').read_text())
    assert report['stream_sha256'] == inputs['report']['stream_sha256']
    start = inputs['report']['window_start_board_s']; end = start+inputs['report']['duration_s']
    states = {}; feedback = {}; requests = {}; angles = []; geometry = []; counter = 0; missing_feedback = 0
    with (decoded/'outputs.csv').open() as handle:
        for row in csv.DictReader(handle):
            time = int(row['begin_ns'])/1e9
            if not start-.1 <= time <= end+.1: continue
            if row['kind'] == 'leg_state':
                states[int(row['id'])] = [float(row[f'leg_{j}_position_rad']) for j in range(4)]
                fields = [row[f'leg_{j}_feedback_torque_nm'] for j in range(4)]
                if all(fields): feedback[int(row['id'])] = [float(value) for value in fields]
                else: missing_feedback += 1
            elif row['kind'] == 'control':
                requests[int(row['id'])] = [float(row[f'request_{j}_firmware_nm']) for j in range(4)]
            elif row['kind'] == 'leg_geometry' and int(row['id']) in states:
                counter += 1
                if counter % 10: continue
                angles.append(states[int(row['id'])])
                geometry.append([float(row[key]) for key in ['left_leg_length_m', 'right_leg_length_m',
                                                            'left_leg_angle_rad', 'right_leg_angle_rad']])
    assert len(angles) > 10
    length, tilt, _ = pose_fit.leg_pose(np.asarray(angles)); geometry = np.asarray(geometry)
    tilt_error = np.degrees(np.arctan2(np.sin(tilt-geometry[:, 2:]), np.cos(tilt-geometry[:, 2:])))
    length_error = abs(length-geometry[:, :2])*1000
    assert length_error.max() < .01 and abs(tilt_error).max() < .01
    paired = sorted(set(feedback)&set(requests))
    torque = dict(paired_count=0, missing_feedback_records=missing_feedback,
                  scope='The complete-output CSV fields are unfilled; this check cannot compare request and feedback torque')
    if paired:
        requested = np.asarray([requests[k] for k in paired]); returned = np.asarray([feedback[k] for k in paired])
        torque = dict(paired_count=len(paired), missing_feedback_records=missing_feedback,
            request_rms_nm=np.sqrt(np.mean(requested**2, axis=0)).tolist(),
            feedback_rms_nm=np.sqrt(np.mean(returned**2, axis=0)).tolist(),
            difference_rms_nm=np.sqrt(np.mean((returned-requested)**2, axis=0)).tolist(),
            difference_mean_nm=np.mean(returned-requested, axis=0).tolist(),
            scope='Same frame-id firmware fields; feedback may be delayed/estimated and is not an independently calibrated load sensor')
    return dict(sample_count=len(angles), length_max_error_mm=length_error.max(0).tolist(),
        tilt_max_error_deg=abs(tilt_error).max(0).tolist(), decoded_report_sha256=fit.audit.digest(decoded/'report.json'),
        request_vs_motor_feedback=torque,
        decoded_csv_sha256=fit.audit.digest(decoded/'outputs.csv'))


def mirror_torque(inputs):
    """Use the existing 100Hz mirror when the 1kHz output CSV lacks torque feedback."""
    mirror = fit.prior.replay.mirror; parser = mirror.FrameParser(); requests = []; feedback = []
    start = inputs['report']['window_start_board_s']; end = start+inputs['report']['duration_s']
    for event in mirror.iter_session_events(Path(inputs['report']['session'])):
        if event['type'] != 'data': continue
        for frame in parser.feed(bytes.fromhex(event['data_hex'])):
            time = mirror.read_u32(frame[8:10])/1000
            if not start <= time <= end: continue
            row = mirror.telemetry_row(frame)
            a = [row.get(f'gated_torque_{name}_firmware_nm') for name in mirror.MOTORS[:4]]
            b = [row.get(f'feedback_torque_{name}_firmware_nm') for name in mirror.MOTORS[:4]]
            if any(value is None for value in a+b): continue
            requests.append(a); feedback.append(b)
    if not requests: return dict(valid_samples=0, scope='No jointly valid request/feedback mirror samples')
    a = np.asarray(requests); b = np.asarray(feedback)
    return dict(valid_samples=len(a), request_rms_nm=np.sqrt(np.mean(a*a, axis=0)).tolist(),
        feedback_rms_nm=np.sqrt(np.mean(b*b, axis=0)).tolist(),
        difference_rms_nm=np.sqrt(np.mean((b-a)**2, axis=0)).tolist(),
        difference_mean_nm=np.mean(b-a, axis=0).tolist(),
        scope='100Hz latest output and motor-reported torque; source validity/age gated by the production decoder, timing/torque calibration not independently identified')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', choices=tuple(fit.REFERENCES))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.experiment/'report.json').read_text()); results = {}; baselines = {}; board = {}
    for name in fit.REFERENCES:
        if args.sessions and name not in args.sessions: continue
        inputs = fit.input_data(name); directory = args.experiment/('final-'+name)
        rows, trace = pose_fit.pose_metrics(directory, inputs)
        assert rows[0] == parent['pose_results'][name]
        with np.load(directory/'leg-length.npz') as saved:
            for key, value in trace.items(): fit.audit.close(saved[key], value, 'Saved pose '+key, 0.)
        board[name] = board_geometry(inputs)
        model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
        mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
        raw = inputs['saved']['real_motor_state']
        with np.load(directory/'physics.npz') as saved:
            time = np.r_[saved['time_s'], trace['time_s'][-1]]
            q = np.concatenate((saved['qpos'][:, :, mq], saved['final_qpos'][:, mq][None]))
        absolute = time+inputs['report']['window_start_board_s']
        real = np.column_stack([np.interp(absolute, raw[:, 0], raw[:, j+1]) for j in range(4)])
        results[name] = summarize(real, q*inputs['sign'][:4]+inputs['zero'])
        baselines[name] = {}
        for key, root in [('previous_length', 'artifacts/real2sim/h7-leg-length-fit-20261104-02-sidewise'),
                          ('previous_joint', 'artifacts/real2sim/h7-ordered-refit-20261104-02-best')]:
            rows, _ = pose_fit.pose_metrics(Path(root)/('final-'+name), inputs)
            baselines[name][key] = rows[0]
    def combined(names, key):
        values = [x*x for n in names if n in results for x in results[n]['recorded_torque'][key]]
        return float(np.sqrt(np.mean(values))) if values else None
    summary = dict(integrity_passed=True, source_report_sha256=fit.audit.digest(args.experiment/'report.json'),
        new_physics_steps=0, full_physics_sample_dt_s=.0002, results=results, baseline_1ms_results=baselines,
        board_geometry_checks=board,
        audited_sessions=list(results), all_four_recordings_audited=len(results)==len(fit.REFERENCES),
        combined_fit_length_rms_mm=combined(['original', 'wide'], 'length_rmse_mm'),
        combined_fit_tilt_rms_deg=combined(['original', 'wide'], 'tilt_rmse_deg'),
        combined_fit_endpoint_rms_mm=combined(['original', 'wide'], 'endpoint_rmse_mm'),
        shared_model_modified=False, hardware_access=False, physical_parameter_identification_unique=False)
    fit.search.write_json(args.output/'verification.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
