"""Compare direction-dependent static load fits; never edit model or command hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import lsq_linear


def metrics(residual):
    return dict(torque_rmse_nm=float(np.sqrt(np.mean(residual ** 2))),
                max_abs_torque_error_nm=float(abs(residual).max()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fit = json.loads(args.fit.read_text())
    samples, excluded = [], []
    for item in fit['samples']:
        pose = json.loads(Path(item['report']).read_text())
        point, = [p for p in pose['source_points'] if p['side'] == item['side']]
        row = dict(item, direction=point['direction'], board_time_s=point['board_time_s'],
                   angle_deg=point['angle_deg'])
        (samples if row['direction'] in ('extend', 'contract') else excluded).append(row)
    x = np.array([s['compression_mm'] for s in samples])
    b = np.array([s['spring_arm_nm_per_n'] for s in samples])
    y = np.array([s['net_required_nm'] for s in samples])
    x0 = float(x.min())
    groups = [(side, direction) for side in ('left', 'right') for direction in ('extend', 'contract')]
    train = np.zeros(len(samples), dtype=bool)
    # 每侧、每方向按板时交错留出，评分点不参与该次参数求解；仍属同一轮采集。
    for side, direction in groups:
        indices = [i for i, s in enumerate(samples) if (s['side'], s['direction']) == (side, direction)]
        indices.sort(key=lambda i: samples[i]['board_time_s'][0])
        train[indices[::2]] = True
    assert train.any() and (~train).any()
    results = {}
    for name, count in (('shared_linear', 1), ('direction_linear', 2), ('side_direction_linear', 4)):
        features = np.zeros((len(samples), 2 * count))
        for i, s in enumerate(samples):
            group = 0 if count == 1 else (('extend', 'contract').index(s['direction']) if count == 2
                                          else groups.index((s['side'], s['direction'])))
            features[i, 2 * group:2 * group + 2] = [1., x[i] - x0]
        design = (b[:, :, None] * features[:, None, :]).reshape(-1, 2 * count)

        def solve(mask):
            a = design[np.repeat(mask, 2)]
            target = y[mask].ravel()
            result = lsq_linear(a, target, bounds=(0., np.inf), tol=1e-12)
            assert result.success and np.linalg.matrix_rank(a) == 2 * count
            # 与独立的无约束解对照；若无约束解可行，两种解应一致。
            unconstrained = np.linalg.lstsq(a, target, rcond=None)[0]
            check = None
            if unconstrained.min() >= 0:
                check = float(abs(unconstrained - result.x).max())
                assert check < 1e-7
            force = features @ result.x
            residual = b * force[:, None] - y
            return dict(parameters=result.x.tolist(), force_n=force.tolist(), residual_nm=residual.tolist(),
                        train=metrics(residual[mask]), heldout=metrics(residual[~mask]) if (~mask).any() else None,
                        independent_lstsq_max_parameter_difference=check)

        results[name] = dict(full_fit=solve(np.ones(len(samples), dtype=bool)),
                             interleaved_validation=solve(train), parameters_per_group=2)
    gaps = []
    for side in ('left', 'right'):
        branches = []
        for direction in ('extend', 'contract'):
            selected = sorted((s for s in samples if s['side'] == side and s['direction'] == direction),
                              key=lambda s: s['compression_mm'])
            branches.append((np.array([s['compression_mm'] for s in selected]),
                             np.array([s['best_uncapped_force_n'] for s in selected])))
        lo, hi = max(a[0].min() for a in branches), min(a[0].max() for a in branches)
        grid = np.linspace(lo, hi, 100)
        # 同压缩量插值对照；两方向腿倾角可能不同，不能把差值全部命名为弹簧摩擦。
        delta = np.interp(grid, *branches[1]) - np.interp(grid, *branches[0])
        gaps.append(dict(side=side, compression_range_mm=[float(lo), float(hi)],
                         median_contract_minus_extend_n=float(np.median(delta)),
                         range_contract_minus_extend_n=[float(delta.min()), float(delta.max())]))
    report = dict(source=str(args.fit.resolve()), source_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
                  samples=samples, excluded_initial_samples=excluded, train_mask=train.tolist(),
                  reference_compression_mm=x0, group_order=dict(direction=['extend', 'contract'], side_direction=groups),
                  force_formula='F_ref + k_N_per_mm * (compression_mm - reference_compression_mm), both coefficients >= 0',
                  results=results, interpolated_direction_gaps=gaps, physics_steps=0, model_modified=False,
                  limitations=['Interleaved points from one sweep are not independent repeat-run or dynamic validation',
                               'Direction gaps also include angle-dependent load, friction and torque bias; not isolated gas-spring friction',
                               'Fits use nominal gravity/mass/geometry; conditional equivalent force is not a measured spring force'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(__file__, args.output / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for side, ax in zip(('left', 'right'), axes[:2]):
        for direction, color in (('extend', 'tab:blue'), ('contract', 'tab:red')):
            indices = [i for i, s in enumerate(samples) if s['side'] == side and s['direction'] == direction]
            indices.sort(key=lambda i: x[i])
            ax.plot(x[indices], [samples[i]['best_uncapped_force_n'] for i in indices], 'o', color=color,
                    label=direction + ' per-pose projection', markersize=4)
            forces = np.array(results['side_direction_linear']['full_fit']['force_n'])
            ax.plot(x[indices], forces[indices], color=color, label=direction + ' fitted line')
        ax.axhline(150, ls='--', color='gray', label='150N nominal reference')
        ax.set_title(side)
        ax.set_xlabel('Spring compression in model (mm)')
        ax.set_ylabel('Conditional equivalent force (N)')
        ax.legend(fontsize=7)
    labels = ['Single\ncurve', 'Direction\ncurves', 'Side + direction\ncurves']
    axes[2].bar(labels, [v['interleaved_validation']['heldout']['torque_rmse_nm'] for v in results.values()])
    axes[2].set_ylabel('Held-out static torque RMSE (Nm)')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('One static sweep: direction-dependent effective load; no dynamic validation')
    fig.tight_layout()
    fig.savefig(args.output / 'static-direction-fits.png', dpi=150)
    print(json.dumps(dict(samples=len(samples), gaps=gaps,
                          results={k: dict(full=v['full_fit']['train'], validation=v['interleaved_validation']['heldout'])
                                   for k, v in results.items()}), indent=2))


if __name__ == '__main__':
    main()
