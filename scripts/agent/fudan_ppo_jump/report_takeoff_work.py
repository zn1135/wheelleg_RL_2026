"""从冻结策略的 GPU 录制离线重建质心、做功和起跳能量。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .measure_flat import events


LEG_IDS = np.array([0, 1, 3, 4])
WHEEL_IDS = np.array([2, 5])
LEG_NAMES = ('lf0', 'lf00', 'rf0', 'rf00')
REFERENCE_WORK_J = 85.14016040788734


def derive_states(model, qpos, qvel):
    """仅运行运动学和惯量计算，不调用积分或接触求解。"""
    data = mujoco.MjData(model)
    root = model.body('robot/base_link').id
    wheels = [model.body('robot/' + name).id for name in ('lfwheel', 'rfwheel')]
    hips = [model.joint('robot/' + name).id for name in ('lf0', 'rf0')]
    mass = float(model.body_subtreemass[root])
    matrix = np.empty((model.nv, model.nv))
    pos, vel, length = np.empty((len(qpos), 3)), np.empty((len(qpos), 3)), np.empty((len(qpos), 2))
    kinetic = np.empty(len(qpos))
    for i, (q, v) in enumerate(zip(qpos, qvel)):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_subtreeVel(model, data)
        mujoco.mj_crb(model, data)
        mujoco.mj_fullM(model, data, matrix)
        pos[i], vel[i] = data.subtree_com[root], data.subtree_linvel[root]
        # mj_crb 生成的惯量矩阵包含模型 dof_armature，避免遗漏转子动能。
        kinetic[i] = .5 * v @ matrix @ v
        local = (data.xpos[wheels] - data.xanchor[hips]) @ data.xmat[root].reshape(3, 3)
        length[i] = np.linalg.norm(local[:, (0, 2)], axis=1)
    vertical = .5 * mass * vel[:, 2] ** 2
    horizontal = .5 * mass * np.sum(vel[:, :2] ** 2, axis=1)
    return dict(com_pos=pos, com_vel=vel, length=length, kinetic_J=kinetic,
                potential_J=-mass * (pos @ model.opt.gravity),
                vertical_kinetic_J=vertical, horizontal_kinetic_J=horizontal,
                remaining_kinetic_J=kinetic-vertical-horizontal)


def phase_summary(start, end, work, delta, force, states, dt):
    part = work[start:end]
    positive = np.maximum(part, 0.).sum(axis=0)
    negative = np.minimum(part, 0.).sum(axis=0)
    net = part.sum(axis=0)
    travel = np.abs(delta[start:end]).sum(axis=0)
    leg_net = float(net[LEG_IDS].sum())
    mechanical_delta = float(states['kinetic_J'][end] + states['potential_J'][end]
                             - states['kinetic_J'][start] - states['potential_J'][start])
    return dict(start_index=int(start), end_index=int(end), duration_ms=(end-start)*dt*1000,
        positive_work_J=positive.tolist(), negative_work_J=negative.tolist(), net_work_J=net.tolist(),
        axis_absolute_travel=travel.tolist(), axis_signed_travel=delta[start:end].sum(axis=0).tolist(),
        leg_positive_work_J=float(positive[LEG_IDS].sum()), leg_negative_work_J=float(negative[LEG_IDS].sum()),
        leg_net_work_J=leg_net, wheel_net_work_J=float(net[WHEEL_IDS].sum()),
        spring_net_work_J=float(net[6:].sum()), total_actuator_work_J=float(net.sum()),
        leg_38Nm_ms=(np.sum(np.abs(force[start:end, LEG_IDS]) >= 38., axis=0)*dt*1000).tolist(),
        leg_axis_path_40Nm_bound_J=(40*travel[LEG_IDS]).tolist(),
        leg_axis_positive_gap_to_bound_J=(40*travel[LEG_IDS]-positive[LEG_IDS]).tolist(),
        leg_axis_net_gap_to_bound_J=(40*travel[LEG_IDS]-net[LEG_IDS]).tolist(),
        leg_path_40Nm_bound_J=float(40*travel[LEG_IDS].sum()),
        leg_net_over_85J_reference=leg_net/REFERENCE_WORK_J,
        leg_positive_over_85J_reference=float(positive[LEG_IDS].sum())/REFERENCE_WORK_J,
        mechanical_energy_change_J=mechanical_delta,
        unclosed_energy_J=mechanical_delta-float(net.sum()),
        energy_start_J={key: float(states[key][start]) for key in states if key.endswith('_J')},
        energy_end_J={key: float(states[key][end]) for key in states if key.endswith('_J')})


def draw_case(output, number, raw, states, work, indices, dt, qadr):
    switch, low, lift, land = indices
    begin, end = max(0, switch-50), min(len(work), land+1)
    time = (np.arange(begin, end)-switch)*dt*1000
    phase = slice(begin, end)
    figure, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True, constrained_layout=True)
    for j, name in enumerate(LEG_NAMES):
        axes[0].plot(time, raw['actual_force'][phase, LEG_IDS[j]], label=name)
        axes[1].plot(time, raw['qpos_pre'][phase, qadr[LEG_IDS[j]]], label=name)
    axes[0].axhline(40, color='grey', ls=':', lw=.8)
    axes[0].axhline(-40, color='grey', ls=':', lw=.8)
    axes[0].set_ylabel('实际力矩 / N·m')
    axes[0].legend(ncol=4, loc='upper right')
    axes[1].set_ylabel('关节角 / rad')
    axes[2].plot(time, states['length'][phase, 0]*100, label='左腿')
    axes[2].plot(time, states['length'][phase, 1]*100, label='右腿')
    axes[2].set_ylabel('实际腿长 / cm')
    axes[2].legend(loc='upper right')
    axes[3].plot(time, states['com_vel'][phase, 2], label='整机质心')
    axes[3].plot(time, raw['trace'][phase, 19], label='机身原点记录值', alpha=.6)
    axes[3].set_ylabel('竖直速度 / m/s')
    axes[3].legend(loc='upper right')
    cumulative = np.vstack((np.zeros((1, 8)), np.cumsum(work, axis=0)))
    cumulative -= cumulative[switch]
    for ids, name in ((LEG_IDS, '四腿电机'), (WHEEL_IDS, '轮电机'), (np.array([6, 7]), '气弹簧')):
        axes[4].plot(time, cumulative[phase][:, ids].sum(axis=1), label=name)
    axes[4].set_ylabel('切模后累计净功 / J')
    axes[4].set_xlabel('相对切入 Jump 的时间 / ms')
    axes[4].legend(loc='upper right')
    for axis in axes:
        axis.grid(alpha=.2)
        for point, color in ((switch, 'grey'), (low, 'tab:green'), (lift, 'tab:red')):
            axis.axvline((point-switch)*dt*1000, color=color, ls='--', lw=.8)
    axes[0].set_title(f'第 {number:02d} 例　绿线为最低质心，红线为首次离地')
    figure.savefig(output/f'start{number:02d}.png', dpi=150)
    plt.close(figure)


def report(recording: Path, *, output_name='report') -> dict:
    recording = Path(recording)
    output = recording/output_name
    output.mkdir(exist_ok=False)
    capture = json.loads((recording/'capture.json').read_text())
    audit = json.loads((recording/'audit.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(recording/'model.mjb'))
    dt = float(capture['physics_dt'])
    qadr = np.array(capture['motor_qadr'] + capture['spring_qadr'], dtype=int)
    vadr = np.array(capture['motor_vadr'] + capture['spring_vadr'], dtype=int)
    assert len(qadr) == model.nu == 8 and np.isclose(dt, .001)
    assert np.allclose(model.actuator_gear[:, 0], 1.)
    assert np.array_equal(model.jnt_qposadr[model.actuator_trnid[:, 0]], qadr)
    names = [model.actuator(i).name for i in range(8)]
    assert [n.split('/')[-1] for n in names] == [n.split('/')[-1] for n in capture['actuator_names']]
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rows, derived, plot_data = [], {}, {}
    for number in range(20):
        with np.load(recording/'probe'/f'start{number:02d}.npz') as saved:
            raw = {key: saved[key] for key in saved.files}
        trace, mode = raw['trace'], raw['mode']
        count = len(trace)
        assert len(raw['qpos_pre']) == count and count == len(mode)*10
        assert all(np.isfinite(raw[key]).all() for key in ('qpos_pre', 'qpos_post', 'qvel_pre', 'qvel_post', 'actual_force'))
        # 无复位的完整轨迹中，上一积分后态就是下一积分前态。
        position_gap = float(np.max(np.abs(raw['qpos_post'][:-1]-raw['qpos_pre'][1:])))
        velocity_gap = float(np.max(np.abs(raw['qvel_post'][:-1]-raw['qvel_pre'][1:])))
        assert position_gap < 1e-6 and velocity_gap < 1e-6
        qpos = np.vstack((raw['qpos_pre'][0], raw['qpos_post']))
        qvel = np.vstack((raw['qvel_pre'][0], raw['qvel_post']))
        states = derive_states(model, qpos, qvel)
        delta = (raw['qpos_post'][:, qadr].astype(float)-raw['qpos_pre'][:, qadr])
        work = raw['actual_force'].astype(float)*delta
        velocity_delta = raw['qvel_post'][:, vadr]*dt
        mismatch = np.abs(delta-velocity_delta)
        jumps = events(trace, mode, physics_dt=dt)
        assert jumps, f'第 {number} 例缺少符合原定义的首跳'
        lift = round(jumps[0]['takeoff_s']/dt)
        land = lift+round(jumps[0]['airtime_s']/dt)
        switch = int(np.flatnonzero(mode)[0])*10
        assert switch == 2000 and land < count
        low = switch+int(np.argmin(states['com_pos'][switch:lift+1, 2]))
        apex = lift+int(np.argmax(states['com_pos'][lift:land+1, 2]))
        position_rate = (states['com_pos'][2:]-states['com_pos'][:-2])/(2*dt)
        velocity_error = position_rate-states['com_vel'][1:-1]
        flight_time = np.arange(land-lift+1)*dt
        ballistic_position = (states['com_pos'][lift] + flight_time[:, None]*states['com_vel'][lift]
                              + .5*flight_time[:, None]**2*model.opt.gravity)
        ballistic_velocity = states['com_vel'][lift]+flight_time[:, None]*model.opt.gravity
        flight_position_error = states['com_pos'][lift:land+1]-ballistic_position
        flight_velocity_error = states['com_vel'][lift:land+1]-ballistic_velocity
        phases = {name: phase_summary(a, b, work, delta, raw['actual_force'], states, dt)
                  for name, a, b in (('switch_to_lowest', switch, low),
                                     ('lowest_to_takeoff', low, lift), ('switch_to_takeoff', switch, lift),
                                     ('last_10ms', lift-10, lift))}
        near_limit = np.abs(raw['actual_force'][switch:lift, LEG_IDS]) >= 38.
        last_near_limit = []
        for axis in range(4):
            hits = np.flatnonzero(near_limit[:, axis])
            last_near_limit.append(float((lift-switch-int(hits[-1])-1)*dt*1000) if len(hits) else None)
        last_contact = (trace[lift-10:lift, 8:10] > .5).any(axis=1)
        last_under_limit = np.abs(raw['actual_force'][lift-10:lift, LEG_IDS]) < 38.
        item = dict(case=number, switch_index=switch, lowest_index=low, lift_index=lift, land_index=land,
            com_apex_index=apex, wheel_peak_m=jumps[0]['peak_lower_wheel_m'],
            takeoff_com_vz_m_s=float(states['com_vel'][lift, 2]),
            takeoff_base_vz_m_s=float(trace[lift, 19]),
            takeoff_com_height_m=float(states['com_pos'][lift, 2]),
            measured_com_rise_m=float(states['com_pos'][apex, 2]-states['com_pos'][lift, 2]),
            ballistic_com_rise_m=float(states['com_vel'][lift, 2]**2/(2*abs(model.opt.gravity[2]))),
            lowest_leg_lengths_m=states['length'][low].tolist(), takeoff_leg_lengths_m=states['length'][lift].tolist(),
            phases=phases, boundary_position_gap=position_gap, boundary_velocity_gap=velocity_gap,
            ms_from_last_38Nm_interval_end_to_takeoff=last_near_limit,
            last_10ms_any_wheel_contact_ms=float(last_contact.sum()*dt*1000),
            last_10ms_under38Nm_with_contact_ms=(np.sum(last_under_limit & last_contact[:, None], axis=0)*dt*1000).tolist(),
            dq_minus_post_v_dt_max=mismatch.max(axis=0).tolist(),
            dq_minus_post_v_dt_rms=np.sqrt(np.mean(mismatch**2, axis=0)).tolist(),
            com_position_derivative_velocity_rms_m_s=np.sqrt(np.mean(velocity_error**2, axis=0)).tolist(),
            com_position_derivative_velocity_max_m_s=np.abs(velocity_error).max(axis=0).tolist(),
            flight_ballistic_position_rms_m=np.sqrt(np.mean(flight_position_error**2, axis=0)).tolist(),
            flight_ballistic_position_max_m=np.abs(flight_position_error).max(axis=0).tolist(),
            flight_ballistic_velocity_rms_m_s=np.sqrt(np.mean(flight_velocity_error**2, axis=0)).tolist(),
            flight_ballistic_velocity_max_m_s=np.abs(flight_velocity_error).max(axis=0).tolist(),
            lift_step_leg_work_J=float(work[lift, LEG_IDS].sum()),
            pd_refresh_count=int(np.count_nonzero(raw['refresh'])),
            recorded_length_vs_pre_max_m=float(np.max(np.abs(raw['length']-states['length'][:-1]))),
            recorded_length_vs_post_max_m=float(np.max(np.abs(raw['length']-states['length'][1:]))))
        rows.append(item)
        for key, values in states.items():
            derived[f'start{number:02d}_{key}'] = values
        derived[f'start{number:02d}_actuator_work_J'] = work
        plot_data[number] = (raw, states, work, (switch, low, lift, land))
    np.savez_compressed(output/'derived.npz', **derived)
    ordered = sorted(rows, key=lambda row: row['wheel_peak_m'])
    best, middle = ordered[-1]['case'], ordered[(len(ordered)-1)//2]['case']
    for number in {best, middle, 5}:
        draw_case(output, number, *plot_data[number], dt, qadr)
    repeat_difference = [item['baseline_repeat_peak_difference_m'] for item in audit['cases']]
    probe_difference = [item['probe_vs_baseline_peak_difference_m'] for item in audit['cases']]
    reproduction = dict(initial_exact=sum(item['initial_exact'] for item in audit['cases']),
        capture_checks_passed=audit['capture_checks_passed'],
        historical_tolerance_passed=audit['historical_tolerance_passed'],
        bitwise_probe_equivalence_passed=audit['bitwise_probe_equivalence_passed'],
        ordinary_repeat_peak_difference_m=[min(repeat_difference), max(repeat_difference)],
        probe_vs_baseline_peak_difference_m=[min(probe_difference), max(probe_difference)],
        historical_cases_outside_tolerance=sum(not item['within_historical_tolerance'] for item in audit['cases']))
    summary = dict(cases=rows, best_case=best, median_height_case=middle, original_best_case=5,
        reproduction_audit=reproduction,
        physics_dt=dt, mass_kg=float(model.body_subtreemass[model.body('robot/base_link').id]),
        actuator_names=names, reference_motor_work_J=REFERENCE_WORK_J,
        timing='第 i 行力矩作用于 qpos_pre[i] 到 qpos_post[i]。接触属于积分前求解状态。'
               '状态数组第 i 项为该步前态，最后一项为末步后态。首个无轮载行 lift 的前态作为离地边界。',
        energy_residual_definition='Δ(总动能+重力势能)−八个执行器净功；包含未单独计量的被动阻尼、摩擦、接触及离散误差。',
        reference_definition='85.140 J 为四轴40 N·m、16到31 cm对称单调伸腿的理想电机功；'
                             '实际轨迹40Σ|Δq|允许往复路径，只用于相同轨迹力矩利用率比较。',
        optimizer_updates=0, cpu_use='只读运动学、质心速度与惯量重建；原始轨迹来自GPU物理积分。')
    (output/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    table = []
    for item in rows:
        phase = item['phases']['lowest_to_takeoff']
        table.append(dict(case=item['case'], wheel_peak_cm=100*item['wheel_peak_m'],
            com_vz_m_s=item['takeoff_com_vz_m_s'], low_to_lift_ms=phase['duration_ms'],
            leg_positive_J=phase['leg_positive_work_J'], leg_negative_J=phase['leg_negative_work_J'],
            leg_net_J=phase['leg_net_work_J'], path_40Nm_J=phase['leg_path_40Nm_bound_J'],
            wheel_J=phase['wheel_net_work_J'], spring_J=phase['spring_net_work_J'],
            energy_residual_J=phase['unclosed_energy_J']))
    with (output/'cases.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    def median(key):
        return float(np.median([row[key] for row in table]))
    text = [
        '# Jump 5200 起跳做功诊断', '',
        f'冻结模型，20 个原初态开展新 GPU 复测，2 m/s 助跑 2 秒后切入 Jump。名义质量 {summary["mass_kg"]:.4f} kg。', '',
        f'双轮轮底峰值中位 {median("wheel_peak_cm"):.2f} cm，最高 {rows[best]["wheel_peak_m"]*100:.2f} cm。'
        f'离地整机质心竖直速度中位 {median("com_vz_m_s"):.3f} m/s。', '',
        f'最低质心至离地阶段，四腿电机正功中位 {median("leg_positive_J"):.2f} J，'
        f'负功 {median("leg_negative_J"):.2f} J，净功 {median("leg_net_J"):.2f} J。'
        f'相同实际转角路径的满力矩上界为 {median("path_40Nm_J"):.2f} J。', '',
        f'该阶段气弹簧净功中位 {median("spring_J"):.2f} J，轮电机净功 {median("wheel_J"):.2f} J，'
        f'能量未闭合量 {median("energy_residual_J"):.2f} J。', '',
        '四轴满力矩、腿长16到31 cm的旧解析参考电机功为85.140 J。实际路径长度、方向及初始动能不同，'
        '与此参考的比值不能直接作为高度提升比例。', '',
        f'20 个初态逐值一致。普通 GPU 重复采集的逐例峰值差为 {100*min(repeat_difference):+.2f}～{100*max(repeat_difference):+.2f} cm，'
        f'探针采集对普通基线为 {100*min(probe_difference):+.2f}～{100*max(probe_difference):+.2f} cm。'
        f'{reproduction["historical_cases_outside_tolerance"]} 例超出旧轨迹复现容差。'
        '上述做功与能量描述本次采集轨迹，探针对轨迹零影响尚缺逐位等价证据。[复现审计](../audit.json)', '',
        summary['timing'], '', summary['energy_residual_definition'], '',
        '图中全部横轴相对切入 Jump。绿线为最低质心，红线为首次离地。', '',
        f'![最高样本](start{best:02d}.png)', '', f'![中位高度代表样本](start{middle:02d}.png)', '',
        '原最高第05例同时保存为 [start05.png](start05.png)。', '',
        '[逐例汇总](cases.csv) · [完整计算](report.json) · [逐毫秒派生量](derived.npz)', '']
    axis_bound = np.median([row['phases']['lowest_to_takeoff']['leg_axis_path_40Nm_bound_J'] for row in rows], axis=0)
    axis_positive = np.median([np.asarray(row['phases']['lowest_to_takeoff']['positive_work_J'])[LEG_IDS] for row in rows], axis=0)
    axis_gap = np.median([row['phases']['lowest_to_takeoff']['leg_axis_positive_gap_to_bound_J'] for row in rows], axis=0)
    text += ['逐轴最低质心至离地的中位数如下，差额按每例计算后取中位。', '',
             '| 电机 | 同路径满力矩上界 / J | 实际正功 / J | 正功差额 / J |',
             '| --- | ---: | ---: | ---: |']
    for axis, name in enumerate(LEG_NAMES):
        text.append(f'| {name} | {axis_bound[axis]:.2f} | {axis_positive[axis]:.2f} | {axis_gap[axis]:.2f} |')
    last_net = np.median([row['phases']['last_10ms']['leg_net_work_J'] for row in rows])
    last_bound = np.median([row['phases']['last_10ms']['leg_path_40Nm_bound_J'] for row in rows])
    last_gap = np.median([row['phases']['last_10ms']['leg_path_40Nm_bound_J']-row['phases']['last_10ms']['leg_net_work_J'] for row in rows])
    reduced = np.median([row['last_10ms_under38Nm_with_contact_ms'] for row in rows], axis=0)
    text += ['', f'离地前最后10 ms，四腿电机净功中位 {last_net:.2f} J，同路径满力矩上界 {last_bound:.2f} J，'
             f'逐例差额中位 {last_gap:.2f} J。仍有轮承重且实际力矩小于38 N·m的时间，'
             f'按表中电机顺序为 {reduced.tolist()} ms。', '',
             '这些差额描述当前轨迹上剩余的功容量。改变出力时序会同时改变离地时刻、姿态和路径，'
             '实际增高需新的对照验证。', '']
    (output/'README.md').write_text('\n'.join(text))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output-name', '--output', default='report')
    args = parser.parse_args()
    report(args.recording, output_name=args.output_name)
