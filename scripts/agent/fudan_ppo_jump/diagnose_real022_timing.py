"""对已核验Jump022日志补查反馈停更先后、阶段事件与成功仿真初态。"""
import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .analyze_real022 import read_csv, matrix, LEGS
from .probe_takeoff_work import ROOT, write_json


def run(source_report, output):
    source = json.loads(source_report.read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with zipfile.ZipFile(source['source']) as archive:
        for session in source['sessions']:
            name = session['name']
            stair = read_csv(archive, name+'/h7-stair.csv')
            net = read_csv(archive, name+'/model-input-actions.csv')
            pd = read_csv(archive, name+'/h7-pd.csv')
            times = stair['h7_us']
            for episode in [e for e in source['episodes'] if e['session'] == name]:
                start = episode['start_board_us']
                result = dict(label=episode['label'], phase_observations=[])
                for phase in episode['observed_phases']:
                    at = start+phase['at_ms']*1000
                    j = min(int(np.searchsorted(times, at)), len(times)-1)
                    n = int(np.argmin(abs(net['board_time_us']-at)))
                    before = (times >= at-30000) & (times <= at)
                    acceleration = np.linalg.norm(matrix(stair, ['acc_x_g', 'acc_y_g', 'acc_z_g'])[before], axis=1)
                    result['phase_observations'].append(dict(**phase,
                        geometric_sample_offset_ms=float((times[j]-at)/1000),
                        leg_length_m=[float(stair[k+'_length_m'][j]) for k in ('left', 'right')],
                        leg_length_rate_m_s=[float(stair[k+'_length_rate_m_s'][j]) for k in ('left', 'right')],
                        support_force_estimate_N=float(net['support_force_n'][n]),
                        previous30ms_min_accel_g=float(acceleration.min())))
                delay = episode['first_fault_after_ms']
                if delay is not None:
                    fault = start+delay*1000
                    j = int(np.searchsorted(times, fault, side='right'))-1
                    freezes = []
                    for motor in LEGS:
                        feedback = matrix(stair, ['angle_rad_'+motor, 'velocity_rad_s_'+motor, 'feedback_nm_'+motor])
                        k = j
                        while k > 0 and np.array_equal(feedback[k-1], feedback[j]):
                            k -= 1
                        freezes.append(dict(motor=motor, constant_feedback_from_ms=float((times[k]-start)/1000),
                            constant_duration_before_fault_ms=float((fault-times[k])/1000),
                            last_triplet=feedback[j].tolist()))
                    result.update(fault_after_ms=delay, feedback_freeze=freezes)
                records.append(result)
    sim_root = ROOT/'artifacts/fudan_ppo_jump/20261123-height022-01/v0_h0.22'
    cases = json.loads((sim_root/'probe/cases.json').read_text())
    metadata = json.loads((sim_root/'metadata.json').read_text())
    qadr = np.asarray(metadata['motor_qadr'])[[0, 1, 3, 4]]
    rows = []
    for case in cases:
        with np.load(sim_root/'probe'/f'start{case["id"]:02d}.npz') as raw:
            first = case['jumps'][0]
            lift = round(first['takeoff_s']*1000)
            rows.append(dict(case=case['id'], initial_leg_length_m=raw['length'][2000].tolist(),
                minimum_leg_length_m=raw['length'][2000:lift].min(0).tolist(),
                takeoff_leg_length_m=raw['length'][lift].tolist(),
                takeoff_after_ms=lift-2000, flight_ms=first['airtime_s']*1000,
                handoff_after_ms=int(np.flatnonzero(raw['mode'])[-1]+1)*10-2000,
                max_target_error_before_takeoff_rad=abs(raw['target'][2000:lift]-raw['qpos_pre'][2000:lift, qadr]).max(0).tolist()))
    comparison = {key: np.median([r[key] for r in rows], axis=0).tolist() for key in rows[0] if key != 'case'}
    write_json(output/'result.json', dict(source_report=str(source_report.resolve()),
        archive_sha256=source['archive_sha256'], real_episodes=records, simulation_cases=rows,
        simulation_medians=comparison, optimizer_updates=0, physics_steps=0,
        scope='Reported triplet constancy precedes fault. No individual receive timestamp or driver error code is present, so physical failure mechanism remains unconfirmed.'))
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fault_rows = [r for r in records if 'fault_after_ms' in r]
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    for i, row in enumerate(fault_rows):
        for side, color, offset in [('lf', 'C0', -.12), ('lr', 'C1', .12)]:
            f = next(x for x in row['feedback_freeze'] if x['motor'] == side)
            begin = f['constant_feedback_from_ms']
            ax.plot([begin, row['fault_after_ms']], [i+offset]*2, color=color, lw=4,
                    label=('左前反馈不变' if side == 'lf' else '左后反馈不变') if i == 0 else None)
            ax.scatter([begin], [i+offset], color=color)
            ax.text((begin+row['fault_after_ms'])/2, i+offset-.10,
                    f'{f["constant_duration_before_fault_ms"]:.1f} ms', ha='center', fontsize=9)
        ax.plot([row['fault_after_ms']]*2, [i-.28, i+.28], color='red', lw=2,
                label='fault=4 / 撤销出力许可' if i == 0 else None)
    ax.set_yticks(range(len(fault_rows)), [r['label'] for r in fault_rows])
    ax.set_xlabel('相对本次首个Jump推理的时间 / ms')
    ax.set_title('左侧反馈先停止变化，随后报故障')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(axis='x', alpha=.2)
    ax.set_xlim(140, 550)
    fig.savefig(output/'fault-timing.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(simulation_medians=comparison, faults=fault_rows), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_report, args.output)
