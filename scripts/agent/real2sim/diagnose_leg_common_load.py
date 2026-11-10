"""Separate spring-independent common loads from directional resistance; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from scripts.agent.real2sim.fit_leg_inverse_curve import load


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    data = [load('wide', root/'inverse-preload-hypothesis'), load('slow', root/'inverse-near-slow'),
            load('fast', root/'inverse-near-fast'),
            load('original', Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original'))]
    results = []
    for side in (0, 1):
        arrays = []; targets = []; groups = []; cancellation = 0.
        section = slice(2*side, 2*side+2)
        for d in data:
            cancellation = max(cancellation, float(np.max(abs(d['spring_basis_nm_per_n'][:, section, side].sum(1)))))
            v = d['velocity'][:, section]
            mask = (abs(v) > .03).all(1) & (v[:, 0]*v[:, 1] < 0)
            mask[:10] = mask[-10:] = False
            # 两个主动关节都运动且反向时，相加消去气弹簧负载；只剩共同方向的负载。
            design = np.column_stack((v[:, 0] > 0, -(v[:, 0] < 0).astype(float),
                                      v[:, 1] > 0, -(v[:, 1] < 0).astype(float), v))
            target = d['target'][:, section].sum(1)
            arrays.append(design[mask]/np.sqrt(max(mask.sum(), 1)))
            targets.append(target[mask]/np.sqrt(max(mask.sum(), 1)))
            groups.append((d['name'], design[mask], target[mask]))
        require_cancellation = cancellation < 1e-8
        assert require_cancellation, 'Spring does not cancel in the common motor direction'
        a = np.concatenate(arrays); b = np.concatenate(targets)
        symmetric = np.column_stack((a[:, 0]+a[:, 1], a[:, 2]+a[:, 3], a[:, 4:6]))
        sym = lsq_linear(symmetric, b, bounds=(0., 8.)).x
        # 设计矩阵不满秩；小正则只挑选一种最小范数种子，不声称单项阻力被辨识。
        directional = lsq_linear(np.vstack((a, np.eye(6)*.001)), np.r_[b, np.zeros(6)], bounds=(0., 8.)).x
        row = dict(side=['lf', 'rf'][side], spring_common_basis_max_m=cancellation,
                   independent_combinations=int(np.linalg.matrix_rank(a)), coefficient_count=6,
                   minimum_norm_directional_coefficients=directional.tolist(), symmetric_coefficients=sym.tolist(), groups=[])
        for name, design, target in groups:
            xs = np.column_stack((design[:, 0]+design[:, 1], design[:, 2]+design[:, 3], design[:, 4:6]))
            row['groups'].append(dict(dataset=name, samples=len(target),
                symmetric_rmse_nm=float(np.sqrt(np.mean((xs@sym-target)**2))) if len(target) else None,
                directional_rmse_nm=float(np.sqrt(np.mean((design@directional-target)**2))) if len(target) else None))
        results.append(row)
    report = dict(scope='Conditional inverse-load diagnosis only; not a physical calibration or response acceptance',
                  environment_steps=0, results=results,
                  sources=[dict(name=d['name'], path=d['source'], stream_sha256=d['stream_sha256'],
                                inverse_sha256=hashlib.sha256((Path(d['source'])/'inverse-loads.npz').read_bytes()).hexdigest()) for d in data],
                  limitations=['Nominal rigid-body masses and observed poses are assumed',
                               'Directional resistance and an omitted conservative load are not uniquely separated',
                               'Six individual coefficients cannot be identified from four independent combinations'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
