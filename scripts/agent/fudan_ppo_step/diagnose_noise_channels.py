"""STEP4200按传感通道分组的GPU噪声诊断，复用已核对采集器，不训练。"""
import argparse
import ast
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from . import diagnose_high_stance_noise as core

BLOCKS = dict(gyro_x=[0], gyro_yz=[1, 2], gravity=[3, 4, 5],
              joint_angle=list(range(9, 13)), joint_velocity=list(range(13, 19)))


def design(step_scales):
    full = [.075] * 6 + [0.] * 3 + [.03] * 4 + [.1125] * 6 + [0.] * 6
    profiles = dict(clean=[0.] * 25, step=step_scales)
    for name, slots in BLOCKS.items():
        profiles[name] = [full[i] if i in slots else 0. for i in range(25)]
    profiles['walk6838'] = full
    cases = [dict(key=f'hold_h{h:.2f}', height_command=h, speed=0., height=0.) for h in (.22, .30, .35)]
    cases += [dict(key=f'up200_v{v:.2f}', height_command=.35, speed=v, height=.2) for v in (.3, 1., 2.5)]
    groups = [dict(profile='clean', seed=71001)]
    groups += [dict(profile=p, seed=s) for s in (71001, 71002, 71003) for p in profiles if p != 'clean']
    samples = [dict(c, **g, id=f'{g["profile"]}_s{g["seed"]}__{c["key"]}') for g in groups for c in cases]
    return profiles, cases, groups, samples


def run(out, deadline):
    original = core.write

    def write(path, value):
        if path.name == 'plan.json':
            # 上次全噪声高位16秒后已越界；本轮先声明统一5～10秒窗口，不按结果挑时段。
            value['measurement']['hold_window_s'] = [5., 10.]
            value['scope'] = '132 nominal GPU trajectories: isolated observation channels, three noise seeds; not hardware/full task acceptance'
            value['noise_blocks'] = BLOCKS
            value['driver_source_sha256'] = core.sha(__file__)
            value['question'] = 'Which observation-noise block alone drives leg-target jitter and zero-command drift? Effects need not add linearly.'
            shutil.copy2(__file__, out / 'channel-source.py')
        original(path, value)

    with patch.object(core, 'design', design), patch.object(core, 'write', write):
        core.run(out, deadline)


def report(out):
    core.report(out)
    plan = json.loads((out / 'plan.json').read_text())
    metrics = json.loads((out / 'metrics.json').read_text())
    rows, groups = [], []
    for row in metrics['rows']:
        with np.load(out / 'traces' / (row['id'] + '.npz')) as tr:
            n = int(tr['valid'].sum())
            speed = np.linalg.norm(tr['linear_world'][:n, :2], axis=1)
            delta = tr['position'][:n] - tr['position'][0]
            rows.append(dict(id=row['id'], profile=row['profile'], key=row['key'],
                valid_duration_s=n*.01, whole20_complete=n == 2000,
                crossed=bool(tr['crossed'][:n].any()),
                xy_displacement_m=float(np.linalg.norm(delta[-1, :2])),
                zero_command_speed_mean=float(speed[abs(tr['command'][:n, :2]).max(-1) < 1e-6].mean())
                    if (abs(tr['command'][:n, :2]).max(-1) < 1e-6).any() else None,
                final_speed_m_s=float(speed[-1]),
                task_passed=bool(row['task_score'] and row['task_score']['passed']),
                first_failure=None if n == 2000 else dict(time_s=n*.01, reason=row['stop_reason'],
                    relative_position_m=(tr['position'][n-1]-tr['origin']).tolist(), gravity=tr['gravity'][n-1].tolist())))
    for profile in plan['noise_scales']:
        chosen = [r for r in rows if r['profile'] == profile]
        stairs = [r for r in chosen if r['key'].startswith('up200')]
        hold = [r for r in chosen if r['key'] == 'hold_h0.35']
        groups.append(dict(profile=profile, stairs_passed=sum(r['task_passed'] for r in stairs),
            stairs_total=len(stairs), stairs_crossed=sum(r['crossed'] for r in stairs),
            hold35_complete=sum(r['whole20_complete'] for r in hold), hold35_total=len(hold),
            hold35_displacement_mean_m=float(np.mean([r['xy_displacement_m'] for r in hold])),
            hold35_final_speed_mean_m_s=float(np.mean([r['final_speed_m_s'] for r in hold]))))
    core.write(out / 'channel-summary.json', dict(groups=groups, rows=rows,
        optimizer_updates=0, warning='Single-block effects are not additive; complete20s displacement and early-ending displacement are distinguished.'))
    plot(out, metrics)
    print(json.dumps(groups, ensure_ascii=False), flush=True)


def plot(out, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['clean', 'step', 'gyro_x', 'gyro_yz', 'gravity', 'joint_angle', 'joint_velocity', 'walk6838']
    labels = ['None', 'Step small', 'Gyro X', 'Gyro Y/Z', 'Gravity', 'Joint angle', 'Joint speed', 'Full6838']
    rows = {r['profile']: r for r in metrics['aggregates'] if r['case'] == 'hold_h0.35'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key, title, degrees in zip(axes.flat,
            ('target_band8to14_rad', 'actual_joint_band8to14_rad', 'gyroX_band8to14_rad_s', 'horizontal_speed_peak'),
            ('Leg targets: 8-14Hz RMS (deg)', 'Actual leg joints: 8-14Hz RMS (deg)',
             'True gyroX: 8-14Hz RMS (rad/s)', 'Horizontal speed peak (m/s)'), (True, True, False, False)):
        values = [float(np.mean(rows[n][key])) if key in rows[n] else np.nan for n in names]
        if degrees:
            values = np.degrees(values)
        ax.bar(np.arange(len(names)), values, color=['#78909c', '#4caf50', '#42a5f5', '#7e57c2', '#ffa726', '#26a69a', '#ec407a', '#ef5350'])
        ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha='right')
        ax.set_title(title); ax.grid(axis='y', alpha=.2)
        if key == 'horizontal_speed_peak':
            ax.axhline(.05, color='black', ls='--', lw=1, label='Parking limit .05')
            ax.legend(fontsize=8)
    fig.suptitle('Frozen STEP4200 | h=0.35m | fixed [5,10)s | same initial state / paired noise seeds\n'
                 'Noise groups tested alone; interactions are not additive. Missing windows are not zero.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out / 'high35-channel-comparison.png', dpi=140)
    plt.close(fig)


def verify(out):
    for name in ('design', 'run'):
        def body(path):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name))
        assert body(Path(__file__)) == body(out / 'channel-source.py'), name
    plan = json.loads((out / 'plan.json').read_text())
    assert core.sha(out / 'channel-source.py') == plan['driver_source_sha256']
    for name, slots in BLOCKS.items():
        assert all(value == (plan['noise_scales']['walk6838'][i] if i in slots else 0.)
                   for i, value in enumerate(plan['noise_scales'][name]))
    previous = core.ROOT / 'artifacts/fudan_ppo_step/20261118-high-stance-noise4200-01'
    old_plan = json.loads((previous / 'plan.json').read_text())
    assert old_plan['checkpoint_sha256'] == plan['checkpoint_sha256']
    differences = {}
    with np.load(previous / 'same-initial-state.npz') as old, np.load(out / 'same-initial-state.npz') as new:
        for key in old.files:
            a, b = old[key][0], new[key][0]
            assert a.shape == b.shape
            if not np.array_equal(a, b):
                differences[key] = float(abs(a.astype(float)-b.astype(float)).max())
    with np.load(previous / 'policy-noise.npz') as old, np.load(out / 'policy-noise.npz') as new:
        same_draws = np.array_equal(old['uniform'][:, :6], new['uniform'][:, :6])
        assert same_draws
    old_rows = {r['id']: r for r in json.loads((previous / 'metrics.json').read_text())['rows']}
    new_rows = {r['id']: r for r in json.loads((out / 'metrics.json').read_text())['rows']}
    common = [sid for sid in old_rows if sid in new_rows and old_rows[sid]['task_score'] is not None]
    lost = [sid for sid in common if old_rows[sid]['task_score']['passed'] and not new_rows[sid]['task_score']['passed']]
    gained = [sid for sid in common if not old_rows[sid]['task_score']['passed'] and new_rows[sid]['task_score']['passed']]
    core.write(out / 'cross-batch-audit.json', dict(previous=str(previous), same_checkpoint=True,
        previous_metrics_sha256=core.sha(previous / 'metrics.json'), same_uniform_draws=same_draws,
        changed_initial_arrays=differences, common_stair_samples=len(common), lost=lost, gained=gained,
        interpretation='Batches are not bitwise identical; compare noise channels within this matched batch. No learning or changed weights between batches.'))
    shutil.copy2(__file__, out / 'channel-report-source.py')
    shutil.copy2(core.__file__, out / 'collector-report-source.py')
    core.verify(out)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('run', 'report', 'render', 'verify'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float)
    p.add_argument('--case', default='hold_h0.35')
    p.add_argument('--channel', choices=BLOCKS, default='joint_velocity')
    args = p.parse_args()
    if args.mode == 'run':
        if args.deadline is None:
            p.error('GPU run requires --deadline')
        run(args.output, args.deadline)
    elif args.mode == 'report':
        report(args.output)
    elif args.mode == 'render':
        core.render(args.output, args.case, ('clean', args.channel, 'walk6838'))
    else:
        verify(args.output)


if __name__ == '__main__':
    main()
