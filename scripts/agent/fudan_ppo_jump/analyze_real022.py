"""只读分析实机Jump ZIP中的推理、PD和反馈CSV，核对原始BIN摘要。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.train import load_jump_policy
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from .probe_takeoff_work import CHECKPOINT, CHECKPOINT_SHA

LEGS = ('lf', 'lr', 'rf', 'rr')


def read_csv(archive, name):
    with archive.open(name) as file:
        rows = list(csv.DictReader(io.TextIOWrapper(file, encoding='utf-8-sig')))
    return {key: np.array([row[key] for row in rows], dtype=str if key in ('phase', 'model') else float)
            for key in rows[0]}


def matrix(data, names):
    return np.stack([data[name] for name in names], axis=1)


def binary_check(data):
    offset, kinds = 0, {}
    while offset+16 <= len(data):
        kind, size, _ = struct.unpack_from('<IIQ', data, offset)
        if offset+16+size > len(data):
            break
        kinds[str(kind)] = kinds.get(str(kind), 0)+1
        offset += 16+size
    return dict(complete_records=sum(kinds.values()), kinds=kinds, trailing_bytes=len(data)-offset)


def draw(output, label, net, pd, stair, start, stop, phases, exit_ms=None):
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    ps = (pd['h7_us'] >= start-100000) & (pd['h7_us'] <= stop+100000)
    ss = (stair['h7_us'] >= start-100000) & (stair['h7_us'] <= stop+100000)
    pt, st = (pd['h7_us'][ps]-start)/1000, (stair['h7_us'][ss]-start)/1000
    for j, name in enumerate(LEGS):
        color = f'C{j}'
        axes[0].plot(pt, pd['target_'+name][ps]-pd['current_'+name][ps], color=color, label=name)
        axes[1].plot(st, stair['request_nm_'+name][ss], color=color, label=name+' request')
        axes[1].plot(st, stair['feedback_nm_'+name][ss], color=color, ls='--', alpha=.65)
    axes[0].set_ylabel('目标减实际角 / rad')
    axes[0].legend(ncol=4)
    axes[1].set_ylabel('固件坐标力矩 / N·m')
    axes[1].legend(ncol=4)
    for name, label_side in [('left', '左腿'), ('right', '右腿')]:
        axes[2].plot(st, stair[name+'_length_m'][ss]*100, label=label_side)
    axes[2].set_ylabel('日志实际腿长 / cm')
    axes[2].legend()
    accel = np.linalg.norm(matrix(stair, ['acc_x_g', 'acc_y_g', 'acc_z_g'])[ss], axis=1)
    axes[3].plot(st, accel, label='加速度模长 / g')
    axes[3].plot(pt, pd['drive_permit'][ps], label='出力许可')
    axes[3].axhline(.3, color='grey', ls=':', label='0.3g参考')
    axes[3].set_ylabel('加速度 / 许可')
    axes[3].set_xlabel('相对首个Jump推理的时间 / ms')
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=.2)
        for phase in phases:
            axis.axvline(phase['at_ms'], color='grey', alpha=.4, ls=':')
        if exit_ms is not None:
            axis.axvline(exit_ms, color='black', ls='--', lw=1)
    axes[0].set_title(label+'，力矩虚线为反馈；黑竖线为下一次WALK记录')
    figure.savefig(output/(label+'.png'), dpi=130)
    plt.close(figure)


def run(archive_path, output):
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    policy, saved = load_jump_policy(CHECKPOINT, 'cpu')
    result = dict(source=str(archive_path), archive_sha256=sha(archive_path), checkpoint_sha256=CHECKPOINT_SHA,
                  sessions=[], episodes=[], optimizer_updates=0, physical_rollout_steps=0)
    with zipfile.ZipFile(archive_path) as archive:
        expected = {line.split(None, 1)[1].replace('\\', '/'): line.split()[0]
                    for line in archive.read('SHA256.txt').decode('utf-8-sig').splitlines() if line.strip()}
        folders = sorted({n.split('/')[0] for n in archive.namelist() if '/' in n})
        for session_index, folder in enumerate(folders):
            binary = next(n for n in archive.namelist() if n.startswith(folder+'/') and n.endswith('.bin'))
            data = archive.read(binary)
            digest = hashlib.sha256(data).hexdigest()
            assert digest == expected[binary]
            net, pd, stair = [read_csv(archive, folder+'/'+n+'.csv') for n in ('model-input-actions', 'h7-pd', 'h7-stair')]
            jump = net['model'] == '5200'
            # 导出标量pitch/roll标签曾互换，统一按记录四元数计算模型轴姿态。
            quaternion = matrix(stair, ['quat_'+axis for axis in 'wxyz'])
            attitude = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_euler('xyz')
            obs = matrix(net, ['obs_'+str(i) for i in range(25)])
            history = matrix(net, ['history_'+str(i) for i in range(125)])
            actions = matrix(net, ['action_'+str(i) for i in range(6)])
            with torch.inference_mode():
                calculated, _ = policy.act_inference(torch.tensor(obs[jump], dtype=torch.float32), torch.tensor(history[jump], dtype=torch.float32))
            session = dict(name=folder, binary_sha256=digest, binary_framing=binary_check(data), jump_frames=int(jump.sum()),
                network_action_max_error=float(np.max(abs(calculated.numpy()-actions[jump]))),
                height_slots_min=float(history[jump].reshape(-1, 5, 25)[:, :, 8].min()),
                height_slots_max=float(history[jump].reshape(-1, 5, 25)[:, :, 8].max()),
                command_min=obs[jump, 6:9].min(0).tolist(), command_max=obs[jump, 6:9].max(0).tolist(),
                history_latest_frame_max_error=float(np.max(abs(history[jump, -25:]-obs[jump]))))
            result['sessions'].append(session)
            starts = np.flatnonzero(jump & ~np.r_[False, jump[:-1]])
            for idx in starts:
                end = idx+1
                while end < len(jump) and jump[end]:
                    end += 1
                t0 = net['board_time_us'][idx]
                complete_exit = bool(end < len(jump))
                t1 = net['board_time_us'][end] if complete_exit else net['board_time_us'][end-1]+10000
                pm = (pd['h7_us'] >= t0) & (pd['h7_us'] < t1)
                sm = (stair['h7_us'] >= t0) & (stair['h7_us'] < t1)
                phases = [dict(phase=net['phase'][i], at_ms=float((net['board_time_us'][i]-t0)/1000))
                          for i in range(idx, end) if i == idx or net['phase'][i] != net['phase'][i-1]]
                faults = np.flatnonzero(pm & (pd['fault'] != 0))
                fault_us = float(pd['h7_us'][faults[0]]) if len(faults) else None
                # 动作分析截止到首次故障，故障后的零腿长和撤力另列。
                valid_stop = fault_us if fault_us is not None else t1
                pv = pm & (pd['h7_us'] < valid_stop) & (pd['pd_valid'] > 0) & (pd['drive_permit'] > 0)
                sv = sm & (stair['h7_us'] < valid_stop) & (stair['fault'] == 0)
                target = matrix(pd, ['target_'+n for n in LEGS])[pv]
                current = matrix(pd, ['current_'+n for n in LEGS])[pv]
                velocity = matrix(pd, ['velocity_rad_s_'+n for n in LEGS])[pv]
                torque = matrix(pd, ['pd_model_nm_'+n for n in LEGS])[pv]
                lengths = matrix(stair, ['left_length_m', 'right_length_m'])[sv]
                length_difference = lengths[:, 0]-lengths[:, 1]
                largest_difference = int(np.argmax(abs(length_difference)))
                acc = np.linalg.norm(matrix(stair, ['acc_x_g', 'acc_y_g', 'acc_z_g'])[sv], axis=1)
                local_dt = np.diff(pd['h7_us'][pm])
                label = f'session{session_index+1}_seq{int(net["sequence"][idx])}'
                item = dict(label=label, session=folder, request_sequence=int(net['sequence'][idx]),
                    start_board_us=int(t0), last_jump_board_us=int(net['board_time_us'][end-1]),
                    next_policy_observed=complete_exit, next_policy=net['phase'][end] if complete_exit else None,
                    next_policy_after_ms=float((t1-t0)/1000) if complete_exit else None,
                    command_min=obs[idx:end, 6:9].min(0).tolist(), command_max=obs[idx:end, 6:9].max(0).tolist(),
                    forward_command_over_training_max_frames=int((abs(obs[idx:end, 6]/3) > saved['jump_config']['commands']['velocity_max']+1e-6).sum()),
                    gap_last_jump_to_next_record_ms=float((t1-net['board_time_us'][end-1])/1000) if complete_exit else None,
                    observed_phases=phases, inference_frames=int(end-idx),
                    max_inference_gap_ms=float(np.diff(net['board_time_us'][idx:end]).max()/1000) if end-idx>1 else None,
                    previous_walk_height_command_m=float(net['obs_8'][idx-1]/5) if idx else None,
                    first_fault_after_ms=(fault_us-t0)/1000 if fault_us is not None else None,
                    fault_values=np.unique(pd['fault'][pm]).astype(int).tolist(),
                    control_gap_max_ms=float(local_dt.max()/1000),
                    pd_formula_max_error_Nm=float(np.max(abs(np.clip(20*(target-current)-velocity, -40, 40)-torque))),
                    target_error_max_rad=abs(target-current).max(0).tolist(),
                    requested_torque_abs_max_Nm=abs(torque).max(0).tolist(),
                    feedback_torque_abs_max_Nm=abs(matrix(stair, ['feedback_nm_'+n for n in LEGS])[sv]).max(0).tolist(),
                    leg_length_start_m=lengths[0].tolist(), leg_length_min_before_fault_m=lengths.min(0).tolist(),
                    leg_length_max_before_fault_m=lengths.max(0).tolist(),
                    simultaneous_leg_length_difference_max_m=float(length_difference[largest_difference]),
                    largest_leg_difference_after_ms=float((stair['h7_us'][sv][largest_difference]-t0)/1000),
                    legs_at_largest_difference_m=lengths[largest_difference].tolist(),
                    pitch_roll_abs_peak_deg=np.rad2deg(abs(attitude[sv][:, [1, 0]]).max(0)).tolist(),
                    accel_norm_min_before_fault_g=float(acc.min()), low_0p3g_samples_before_fault=int((acc<.3).sum()),
                    low_0p3g_sample_times_ms=((stair['h7_us'][sv][acc<.3]-t0)/1000).tolist())
                if fault_us is not None:
                    after = (stair['h7_us'] >= fault_us) & (stair['h7_us'] <= fault_us+300000)
                    item['fault_300ms_feedback_distinct_triplets'] = [len(np.unique(matrix(stair, ['angle_rad_'+n, 'velocity_rad_s_'+n, 'feedback_nm_'+n])[after], axis=0)) for n in LEGS]
                    item['drive_permit_min_after_fault'] = float(pd['drive_permit'][pm & (pd['h7_us'] >= fault_us)].min())
                result['episodes'].append(item)
                draw(output, label, net, pd, stair, t0, min(t1, t0+1200000), phases,
                     (t1-t0)/1000 if complete_exit else None)
            print(folder, 'analyzed', len(starts), 'episodes', flush=True)
    result['scope'] = 'CSV input/output and recorded controller/feedback analysis. BIN hashes and framing checked; CSV fields not independently redecoded from binary. Acceleration and estimated phase alone do not establish flight height. Torque feedback is the logged motor estimate.'
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['episodes'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.output)
