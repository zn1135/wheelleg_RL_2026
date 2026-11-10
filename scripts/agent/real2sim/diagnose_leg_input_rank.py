"""Audit independent input directions in existing joint recordings; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    data = [inverse.load('wide', root/'inverse-preload-hypothesis'), inverse.load('slow', root/'inverse-near-slow'),
            inverse.load('fast', root/'inverse-near-fast'),
            inverse.load('original', Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original'))]
    rows = []
    for d in data:
        for side in (0, 1):
            section = slice(2*side, 2*side+2); u = d['drive_torque_nm'][:, section]
            singular = np.linalg.svd(u, compute_uv=False)
            common = u.sum(1)/np.sqrt(2.); differential = (u[:, 0]-u[:, 1])/np.sqrt(2.)
            rows.append(dict(dataset=d['name'], side=['lf', 'rf'][side], samples=len(u),
                singular_values=singular.tolist(), second_to_first_singular_value=float(singular[1]/singular[0]),
                rank_at_relative_tolerance_1e_5=int(np.sum(singular > singular[0]*1e-5)),
                common_request_rms_nm=float(np.sqrt(np.mean(common**2))),
                differential_request_rms_nm=float(np.sqrt(np.mean(differential**2))),
                maximum_absolute_common_request_nm=float(max(abs(common)))))
    report = dict(scope='Independent input-direction audit, not proof that a trajectory fit is impossible',
                  environment_steps=0, results=rows,
                  sources=[dict(name=d['name'], source=d['source'], recording_sha256=d['stream_sha256'],
                      inverse_sha256=hashlib.sha256((Path(d['source'])/'inverse-loads.npz').read_bytes()).hexdigest()) for d in data],
                  missing_identification_directions=['Common direction: both active model joint angles change together, approximately preserving leg length',
                      'Current recordings excite differential direction: both active model joint torques oppose each other'],
                  proposed_future_measurement_basis_model_order_lf0_lf00_rf0_rf00=[[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]],
                  limitations=['This is a data coverage diagnosis, not an executable hardware procedure',
                      'A single independent input direction does not by itself prevent fitting its observed trajectories',
                      'It does prevent treating all motor/seal/spring load combinations as independently measured',
                      'Actual shaft torque and fixture translation are not independently recorded'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
