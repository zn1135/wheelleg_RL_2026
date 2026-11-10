"""Fit uncapped gas-spring hypotheses to frozen static torque reports; no physics or model edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import lsq_linear


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples, sources = [], []
    model_hash = config_hash = None
    for path in args.reports:
        report = json.loads(path.read_text())
        if model_hash is None:
            model_hash, config_hash = report['model_sha256'], report['config_sha256']
        assert (model_hash, config_hash) == (report['model_sha256'], report['config_sha256'])
        sources.append(dict(path=str(path.resolve()), sha256=digest(path)))
        g = np.array(report['gravity_required_nm']) + report['passive_required_nm']
        tau = np.array(report['feedback_mean_nm'])
        for side, leg in enumerate(report['legs']):
            if 'fit_sides' in report and leg['side'] not in report['fit_sides']:
                continue
            section = slice(2 * side, 2 * side + 2)
            b = np.array(leg['spring_required_nm_per_n'])
            y = (tau - g)[section]
            # 两电机共同约束一个弹簧推力，F>=0，不设置推力上限。
            force = max(0., float(b @ y / (b @ b)))
            oracle = np.linalg.lstsq(b[:, None], y, rcond=None)[0][0]
            assert abs(force - max(0., oracle)) < 1e-9
            residual = b * force - y
            samples.append(dict(report=str(path.resolve()), side=leg['side'], length_mm=leg['length_m'] * 1000,
                                compression_mm=leg['spring_compression_m'] * 1000,
                                gravity_nm=g[section].tolist(), feedback_nm=tau[section].tolist(),
                                spring_arm_nm_per_n=b.tolist(), net_required_nm=y.tolist(),
                                shared_model_force_n=leg['model_spring_force_n'], best_uncapped_force_n=force,
                                independent_remaining_error_nm=residual.tolist(),
                                spring_plus_required_rank=int(np.linalg.matrix_rank(np.column_stack((b, y)))),
                                residual_orthogonality_error=float(abs(b @ residual))))
    b = np.array([s['spring_arm_nm_per_n'] for s in samples])
    y = np.array([s['net_required_nm'] for s in samples])
    x = np.array([s['compression_mm'] for s in samples])
    x0 = float(x.min())
    # 以已测最小压缩量为参考，推力及压缩增力斜率非负；只解释已测区间。
    design = np.stack((b, b * (x - x0)[:, None]), axis=-1).reshape(-1, 2)
    line = lsq_linear(design, y.ravel(), bounds=(0., np.inf), tol=1e-12)
    assert line.success
    linear_force = line.x[0] + line.x[1] * (x - x0)
    constant = max(0., float(np.sum(b * y) / np.sum(b * b)))
    independent = np.array([s['best_uncapped_force_n'] for s in samples])
    cases = dict(current_shared_model=np.array([s['shared_model_force_n'] for s in samples]),
                 best_per_pose_with_150n_cap=np.minimum(independent, 150.),
                 independent_uncapped_per_pose=independent,
                 shared_uncapped_constant=np.full(len(x), constant),
                 shared_uncapped_linear=linear_force)
    results = {}
    for name, force in cases.items():
        residual = b * force[:, None] - y
        results[name] = dict(forces_n=force.tolist(), residual_nm=residual.tolist(),
                             max_abs_torque_error_nm=float(abs(residual).max()),
                             torque_rmse_nm=float(np.sqrt(np.mean(residual ** 2))))
    report = dict(scope='Uncapped static spring hypotheses from frozen same-pose gravity/torque projections',
                  sources=sources, model_sha256=model_hash, config_sha256=config_hash,
                  samples=samples, results=results, shared_constant_n=constant,
                  shared_linear=dict(reference_compression_mm=x0, force_at_reference_n=float(line.x[0]),
                                     stiffness_n_per_mm=float(line.x[1]),
                                     observed_compression_range_mm=[float(x.min()), float(x.max())],
                                     design_rank=int(np.linalg.matrix_rank(design))),
                  verification=dict(independent_force_matches_lstsq=True,
                                    residual_orthogonality_error_max=max(s['residual_orthogonality_error'] for s in samples)),
                  physics_steps=0, model_modified=False, force_upper_bound_n=None,
                  limitations=['Forces absorb unmodeled friction, nominal mass/COM and drive-torque bias; not measured spring forces',
                               'Independent per-pose forces are a diagnostic best case, not one identified spring curve',
                               'Fit and score use the same samples; no held-out validation or extrapolation beyond observed compression'])
    np.savez_compressed(args.output / 'static-fit-inputs.npz', compression_mm=x, spring_arm_nm_per_n=b, net_required_nm=y)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(__file__, args.output / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for side, color in [('left', 'tab:blue'), ('right', 'tab:orange')]:
        indices = np.array([s['side'] == side for s in samples])
        axes[0].scatter(x[indices], independent[indices], color=color, label=side + ' per-pose best')
    grid = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(grid, line.x[0] + line.x[1] * (grid - x0), color='black', label='Shared linear fit')
    axes[0].axhline(150, ls='--', color='gray', label='150N reference')
    axes[0].set_xlabel('Model spring compression (mm)')
    axes[0].set_ylabel('Conditional fitted spring force (N)')
    axes[0].legend(fontsize=8)
    labels = ['Current\nmodel', '150N cap\nper pose', 'Uncapped\nper pose', 'Shared\nconstant', 'Shared\nlinear']
    axes[1].bar(labels, [r['max_abs_torque_error_nm'] for r in results.values()])
    axes[1].set_ylabel('Maximum torque mismatch (Nm)')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Uncapped static hypotheses: fitted and scored on the same recorded poses', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / 'uncapped-static-spring.png', dpi=150)
    print(json.dumps(dict(samples=[{k: s[k] for k in ('side', 'length_mm', 'best_uncapped_force_n')} for s in samples],
                          results={k: v['max_abs_torque_error_nm'] for k, v in results.items()},
                          shared_linear=report['shared_linear']), indent=2))


if __name__ == '__main__':
    main()
