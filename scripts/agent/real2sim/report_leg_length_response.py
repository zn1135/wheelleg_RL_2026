"""Recompute length errors and firmware geometry agreement, with zero dynamics steps."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_length_response as length_fit
from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.experiment/'report.json').read_text())
    results = {}; before = {}; geometry_error = {}; parity = {}
    for name in fit.REFERENCES:
        inputs = fit.input_data(name); directory = args.experiment/('final-'+name)
        rows, trace = length_fit.metrics(directory, inputs)
        assert rows[0] == parent['length_results'][name]
        with np.load(directory/'leg-length.npz') as saved:
            for key, value in trace.items(): fit.audit.close(saved[key], value, 'Saved length '+key, 0.)
        g = inputs['saved']['real_geometry']; raw = inputs['saved']['real_motor_state']
        measured_angles = np.column_stack([np.interp(g[:, 0], raw[:, 0], raw[:, j+1]) for j in range(4)])
        geometry_error[name] = (abs(length_fit.lengths(measured_angles)-g[:, 2:4]).max(0)*1000).tolist()
        assert max(geometry_error[name]) < .01
        model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
        mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
        with np.load(directory/'physics.npz') as saved:
            time = np.r_[saved['time_s'], trace['time_s'][-1]]
            q = np.concatenate((saved['qpos'][:, :, mq], saved['final_qpos'][:, mq][None]))
        angles = q*inputs['sign'][:4]+inputs['zero']
        absolute = time+inputs['report']['window_start_board_s']
        measured_angles = np.column_stack([np.interp(absolute, raw[:, 0], raw[:, j+1]) for j in range(4)])
        error = (length_fit.lengths(angles)-length_fit.lengths(measured_angles)[:, None])*1000
        results[name] = {}
        for index, branch in enumerate(('recorded_torque', 'recorded_target')):
            e = error[:, index]
            results[name][branch] = dict(rmse_mm=np.sqrt(np.mean(e*e, axis=0)).tolist(),
                peak_mm=abs(e).max(0).tolist(), terminal_mm=abs(e[-1]).tolist())
        before[name] = parent['previous_angle_fit_length_results'][name]
        if name in ('original', 'wide'):
            batches = sorted(args.experiment.glob('generation-*/'+name))
            batches += [Path(p).parent for p in parent['source_pool_report_hashes'] if Path(p).parent.name == name]
            parity[name] = []
            for side in (0, 1):
                def matches_side(profile):
                    for key, wanted in parent['selected_parameters'].items():
                        section = slice(side, side+1) if len(wanted) == 2 else slice(2*side, 2*side+2)
                        if not np.allclose(profile[key][section], wanted[section], rtol=0., atol=1e-10): return False
                    return True
                for batch in batches:
                    batch_report = json.loads((batch/'report.json').read_text())
                    matches = [i for i, r in enumerate(batch_report['results']) if matches_side(r['parameters'])]
                    if not matches: continue
                    index = matches[0]; count = len(batch_report['results']); joints = mq[2*side:2*side+2]
                    with np.load(batch/'trials.npz') as a, np.load(directory/'trials.npz') as b:
                        stride = int(round(batch_report['saved_dt_s']/.001))
                        assert np.allclose(a['time_s'], b['time_s'][::stride], atol=1e-12)
                        difference = np.degrees(a['qpos'][:, [index, count+index]][:, :, joints]-b['qpos'][::stride, :, :][:, :, joints])
                    parity[name].append(dict(side=side, source=str(batch), max_joint_difference_deg=float(abs(difference).max()),
                                             positions_bitwise_identical=bool(np.all(difference == 0))))
                    break
            assert len(parity[name]) == 2
    def combined(data, names):
        return float(np.sqrt(np.mean([v*v for n in names for v in data[n]['recorded_torque']['rmse_mm']])))
    result = dict(integrity_passed=True, new_physics_steps=0, source_report_sha256=fit.audit.digest(args.experiment/'report.json'),
        length_definition='H7 five-bar motor-angle geometry; same formula for measured and simulated angles',
        actual_board_geometry_max_difference_mm=geometry_error, full_physics_sample_dt_s=.0002,
        full_physics_sample_results=results, previous_1ms_results=before,
        combined_fit_rms_mm=combined(results, ['original', 'wide']),
        previous_fit_rms_mm=combined(before, ['original', 'wide']),
        combined_regression_rms_mm=combined(results, ['slow', 'fast']),
        previous_regression_rms_mm=combined(before, ['slow', 'fast']),
        selection_batch_vs_single_motor_parity=parity,
        shared_model_modified=False, hardware_access=False, accepted_as_full_joint_identification=False)
    fit.search.write_json(args.output/'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
