"""只读检查传统台阶CSV；完成标记与几何高度不作为登台成功证据。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def inspect(source, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(source.resolve()),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='Offline CSV inspection only; no simulation or learning', tracks=[])
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    with ZipFile(source) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.csv'):
                continue
            raw = archive.read(name)
            reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
            rows = list(reader)
            data = {key: np.array([float(row[key]) for row in rows]) for key in reader.fieldnames}
            phase = data['stair_phase'].astype(int)
            active = np.isin(phase, [1, 2, 5])
            indices = np.flatnonzero(active)
            if not len(indices):
                raise ValueError(f'{name}: no active stair phase')
            start, last = indices[0], indices[-1]
            # 各次独立对齐动作触发，不拼接不同上电的板端时间。
            t = (data['h7_us']-data['h7_us'][start])*1e-6
            dt = np.diff(data['h7_us'])*1e-6
            phase_runs = []
            for first, stop in zip(np.r_[0, np.flatnonzero(np.diff(phase))+1],
                                   np.r_[np.flatnonzero(np.diff(phase))+1, len(phase)]):
                phase_runs.append(dict(phase=int(phase[first]), start_s=float(t[first]),
                    duration_s=float(t[stop]-t[first]) if stop<len(t) else float(t[-1]-t[first]+np.median(dt))))
            torque = np.stack([data['request_nm_'+joint] for joint in ('lf','lr','rf','rr')], axis=-1)
            peaks = {}
            for side in ('left', 'right'):
                values = np.abs(data[side+'_angle_rad'])
                index = indices[np.argmax(values[active])]
                peaks[side] = dict(angle_abs_peak_rad=float(values[index]), peak_time_s=float(t[index]),
                    length_min_m=float(data[side+'_length_m'][active].min()),
                    length_max_m=float(data[side+'_length_m'][active].max()))
            post = np.arange(len(t))>last
            item = dict(file=Path(name).name, sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows),
                finite=bool(all(np.isfinite(x).all() for x in data.values())),
                dt_min_s=float(dt.min()), dt_max_s=float(dt.max()), dt_median_s=float(np.median(dt)),
                sequence_contiguous=bool((np.diff(data['sequence'])==1).all()),
                board_drops_change=float(np.ptp(data['board_drops'])),
                fault_codes=np.unique(data['fault']).astype(int).tolist(),
                active_trial_ids=np.unique(data['trial'][active]).astype(int).tolist(),
                phase_runs=phase_runs, duration_s=float(t[last+1]) if last+1<len(t) else None,
                active_pitch_abs_peak_deg=float(np.rad2deg(np.abs(data['pitch_rad'][active]).max())),
                post_pitch_abs_peak_deg=float(np.rad2deg(np.abs(data['pitch_rad'][post]).max())) if post.any() else None,
                active_hip_request_abs_peak_Nm=float(np.abs(torque[active]).max()),
                left_right_angle_abs_difference_peak_rad=float(np.abs(data['left_angle_rad'][active]-data['right_angle_rad'][active]).max()),
                leg_peaks=peaks,
                terminal_control_condition=int(data['control_condition'][-1]),
                terminal_result=int(data['stair_result'][-1]),
                terminal_exit_reason=int(data['stair_exit_reason'][-1]))
            report['tracks'].append(item)
            label=Path(name).stem
            axes[0].plot(t, data['left_angle_rad'], alpha=.5, label=label)
            axes[0].plot(t, data['right_angle_rad'], alpha=.5, linestyle='--')
            axes[1].plot(t, data['left_length_m'], alpha=.5)
            axes[1].plot(t, data['right_length_m'], alpha=.5, linestyle='--')
            axes[2].plot(t, np.rad2deg(data['pitch_rad']), alpha=.5)
            axes[3].plot(t, np.abs(torque).max(axis=-1), alpha=.5)
        notes = [dict(file=name, text=archive.read(name).decode('utf-8-sig'))
                 for name in archive.namelist() if name.endswith('.md')]
    report['attached_notes_unverified'] = notes
    report['limitations'] = ['Stair height and external success labels absent',
        'No wheel contacts or root translation recorded',
        'Requested torques are not executed PPO actions',
        'Hip-to-wheel height is geometry, not measured ground clearance',
        'Left/right angle curves alone cannot establish first wheel contact or unloading']
    for axis, label in zip(axes, ['Virtual leg angle [rad]', 'Virtual leg length [m]',
                                 'Pitch [deg]', 'Max hip torque request [Nm]']):
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title('10 traditional stair records; solid=left, dashed=right; no learning')
    axes[-1].set_xlabel('Time from phase 1 start [s]')
    fig.tight_layout()
    fig.savefig(output/'tracks.png', dpi=150)
    plt.close(fig)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('attached_notes_unverified',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.source, args.output)
