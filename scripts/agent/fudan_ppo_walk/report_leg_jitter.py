"""Read-only leg smoothness analysis of paired GPU qpos/action recordings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; sources = {}; plot = {}
    for height in ('h022', 'h026', 'h030'):
        for sid, lo, hi in (('diagnostic_0', 2., 7.), ('diagnostic_13', 10., 15.), ('diagnostic_14', 10., 15.)):
            for branch in ('baseline', 'candidate'):
                folder = args.recording / height / branch
                mapping_path = folder / 'fixtures/mapping.json'
                mapping = json.loads(mapping_path.read_text())
                indices = np.asarray(mapping['motor_q'])[[0, 1, 3, 4]]
                path = folder / 'traces' / (sid + '.npz')
                for source in (path, mapping_path):
                    sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                with np.load(path) as trace:
                    time = trace['qpos_time']; assert np.allclose(np.diff(time), .01)
                    i, j = round(lo / .01), round(hi / .01)
                    q = trace['qpos'][i:j+1, indices].astype(np.float64)
                    action = trace['action'][i:j, :][:, [0, 1, 3, 4]].astype(np.float64)
                # 由100Hz保存位置做相邻差分；不是500Hz瞬时关节速度。
                velocity = np.diff(q, axis=0) / .01
                centered = q[:-1] - q[:-1].mean(axis=0)
                power = np.abs(np.fft.rfft(centered, axis=0))**2; power[0] = 0.
                frequency = np.fft.rfftfreq(len(centered), .01)
                rows.append(dict(height=height, sample_id=sid, branch=branch, window=[lo, hi],
                    joints=['lf0', 'lf00', 'rf0', 'rf00'],
                    combined_velocity_rms_rad_s=float(np.sqrt(np.mean(velocity**2))),
                    per_joint_velocity_rms_rad_s=np.sqrt(np.mean(velocity**2, axis=0)).tolist(),
                    per_joint_angle_range_deg=np.rad2deg(np.ptp(q, axis=0)).tolist(),
                    dominant_frequency_hz=frequency[power.argmax(axis=0)].tolist(),
                    target_step_rms_deg=np.rad2deg(np.sqrt(np.mean(np.diff(action, axis=0)**2, axis=0)) * .5).tolist()))
                if sid == 'diagnostic_0':
                    plot[(height, branch)] = (np.arange(i, j)*.01, centered[:, 1], (action[:, 1]-action[:, 1].mean())*.5)
    result = dict(rows=rows, input_sha256=sources, physics_steps=0, model_forwards=0, optimizer_updates=0,
        scope='100Hz saved motor positions and actions; excludes startup and replay loop reset; no new physics',
        limitation='Finite-difference leg speed is not instantaneous 500Hz qvel; frequencies above50Hz are not resolved',
        verdict='rejected_by_user_severe_leg_oscillation', newly_invented_acceptance_threshold=False)
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), constrained_layout=True)
    for row, height in enumerate(('h022', 'h026', 'h030')):
        for branch, label in (('baseline', '8000'), ('candidate', '9000')):
            t, q, target = plot[(height, branch)]
            axes[row, 0].plot(t, np.rad2deg(q), label=label, linewidth=1.)
            axes[row, 1].plot(t, np.rad2deg(target), label=label, linewidth=1.)
        axes[row, 0].set_title(height + ': lf00 measured angle, mean removed')
        axes[row, 1].set_title(height + ': lf00 commanded target, mean removed')
        for ax in axes[row]:
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (deg)'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(args.output / 'oscillation.png', dpi=150); plt.close(fig)
    text = ['# 9000轮严重抖腿：既有轨迹核对', '', '用户明确因严重抖腿拒绝本模型；统一运动评分45/45未覆盖腿部平滑性，不能视为模型被接受。', '',
        '| 高度/工况 | 8000四腿速度RMS rad/s | 9000 | 倍率 |', '|---|---:|---:|---:|']
    for height in ('h022', 'h026', 'h030'):
        for sid in ('diagnostic_0', 'diagnostic_13', 'diagnostic_14'):
            old, new = [next(r for r in rows if r['height']==height and r['sample_id']==sid and r['branch']==b) for b in ('baseline','candidate')]
            x, y = old['combined_velocity_rms_rad_s'], new['combined_velocity_rms_rad_s']
            text.append(f'| {height}/{sid} | {x:.5f} | {y:.5f} | {y/x:.1f}× |')
    text += ['', '100Hz位置差分，原地2～7秒、停车10～15秒，排除启动和播放器回到开头的跳变。主振动频率约2.8～3.0Hz；原始动作目标也往复变化，不能仅归因于显示跳帧。不能据此分离策略、估速和物理接触的独立因果。', '',
        '本次没有新增物理、网络前向、训练或奖励修改，没有把任何新平滑阈值追溯应用于历史验收。下一步先拆现有轨迹中停车收益和动作一阶/二阶差分惩罚，再决定修改；关节瞬时速度及完整力矩惩罚缺测须如实保留。', '',
        '[原始量与输入指纹](metrics.json) · [关节及动作曲线](oscillation.png)']
    (args.output / 'README.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[4:14]))


if __name__ == '__main__':
    main()
