"""Compare opposite-direction loads at nearby measured poses; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    datasets = [inverse.load('wide', root/'inverse-preload-hypothesis'),
                inverse.load('original', Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original')),
                inverse.load('slow', root/'inverse-near-slow'), inverse.load('fast', root/'inverse-near-fast')]
    rows = []; summaries = []
    for dataset_index, data in enumerate(datasets):
        for side in (0, 1):
            section = slice(2*side, 2*side+2)
            q = data['qpos'][:, inverse.MOTOR_INDEX[section]]
            v = data['velocity'][:, section]
            slide_v = data['qvel'][:, inverse.SLIDE_INDEX[side]]
            valid = np.ones(len(q), dtype=bool); valid[:10] = valid[-10:] = False
            positive = np.flatnonzero(valid & (slide_v > .001) & (np.linalg.norm(v, axis=1) > .03))
            negative = np.flatnonzero(valid & (slide_v < -.001) & (np.linalg.norm(v, axis=1) > .03))
            pairs = []
            if len(positive) and len(negative):
                tree = cKDTree(q[negative])
                for i in positive:
                    candidates = tree.query_ball_point(q[i], np.deg2rad(.5)*np.sqrt(2))
                    eligible = []
                    for local in candidates:
                        j = negative[local]
                        if np.max(abs(q[i]-q[j])) > np.deg2rad(.5) or abs(data['time_s'][i]-data['time_s'][j]) < .5:
                            continue
                        speed_ratio = np.linalg.norm(v[i])/np.linalg.norm(v[j])
                        opposition = -np.dot(v[i], v[j])/(np.linalg.norm(v[i])*np.linalg.norm(v[j]))
                        if .5 <= speed_ratio <= 2. and opposition >= .9:
                            eligible.append((np.linalg.norm(q[i]-q[j]), j))
                    if eligible:
                        _, j = min(eligible); pairs.append((i, j))
            # 一次反向样本只计一次，避免长时间缓慢采样重复加权。
            used = set(); unique = []
            for i, j in sorted(pairs, key=lambda pair: np.linalg.norm(q[pair[0]]-q[pair[1]])):
                if j not in used: used.add(j); unique.append((i, j))
            force = []; orthogonal = []; separation = []; halfgap = []
            for i, j in unique:
                basis = .5*(data['spring_basis_nm_per_n'][i, section, side]+data['spring_basis_nm_per_n'][j, section, side])
                middle = .5*(data['target'][i, section]+data['target'][j, section])
                # 只有正反向阻力对称时，中点投影才等于弹簧力；不把它写成实测推力。
                projected = np.dot(basis, middle)/np.dot(basis, basis)
                transverse = middle-basis*projected
                gap = .5*(data['target'][i, section]-data['target'][j, section])
                compression = .5*(data['compression'][i, side]+data['compression'][j, side])
                force.append(projected); orthogonal.append(transverse); halfgap.append(gap)
                separation.append(np.rad2deg(np.max(abs(q[i]-q[j]))))
                rows.append([dataset_index, side, i, j, compression, projected, *transverse, *gap])
            entry = dict(dataset=data['name'], side=['lf', 'rf'][side], pairs=len(unique),
                         positive_samples=len(positive), negative_samples=len(negative))
            if unique:
                entry.update(position_separation_max_deg=float(max(separation)),
                             conditional_midpoint_force_percentiles_n=np.percentile(force, [5, 50, 95]).tolist(),
                             non_spring_midpoint_rms_nm=np.sqrt(np.mean(np.asarray(orthogonal)**2, axis=0)).tolist(),
                             direction_halfgap_rms_nm=np.sqrt(np.mean(np.asarray(halfgap)**2, axis=0)).tolist())
            summaries.append(entry)
    np.savez_compressed(args.output/'pairs.npz', rows=np.asarray(rows),
                        columns=np.array(['dataset_index', 'side', 'positive_index', 'negative_index', 'compression_m',
                                          'conditional_force_n', 'transverse_front_nm', 'transverse_rear_nm', 'gap_front_nm', 'gap_rear_nm']))
    report = dict(scope='Measured-pose inverse-load diagnostic only; no GPU replay acceptance', environment_steps=0,
                  pose_tolerance_each_axis_deg=.5, minimum_velocity_opposition=.9, speed_ratio_range=[.5, 2.],
                  summaries=summaries, sources=[dict(name=d['name'], inverse_source=d['source'], recording_sha256=d['stream_sha256'],
                                                   inverse_sha256=hashlib.sha256((Path(d['source'])/'inverse-loads.npz').read_bytes()).hexdigest()) for d in datasets],
                  limitations=['Midpoint spring estimate assumes symmetric direction-dependent resistance, approximately matched speeds, nominal mass and the existing rigid-body model.',
                               'Nearby configurations are not identical; sparse pairs do not identify a spring curve.',
                               'Drive request torque is not independent shaft-torque measurement.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__': main()
