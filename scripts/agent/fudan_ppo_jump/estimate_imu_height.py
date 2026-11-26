"""按现有H7竖直投影对实机IMU双积分，保留漂移与初值敏感性。"""
import argparse
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation

from .analyze_real022 import read_csv, matrix
from .probe_takeoff_work import ROOT, write_json
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha

GRAVITY = 9.81


def vertical_acceleration(quaternion, acceleration_g):
    # 复用大机 Attitude_Accel_Vertical 的轴映射，零比力时应得到向下的-g。
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    specific_world = rotation.apply(acceleration_g*np.array([1., -1., -1.]))
    return (specific_world[:, 2]-1.)*GRAVITY


def integrate(time, acceleration, bias, anchor_s=-.2):
    anchor = int(np.searchsorted(time, anchor_s))
    velocity = cumulative_trapezoid(acceleration-bias, time, initial=0.)
    velocity -= velocity[anchor]  # 假设起跳前这一时刻的竖直速度为0。
    height = cumulative_trapezoid(velocity, time, initial=0.)
    height -= np.interp(0., time, height)
    return velocity, height, anchor


def check_math():
    time = np.linspace(0., 1., 201)
    rotation = Rotation.from_euler('xyz', np.c_[.4*time, -.3*time, time])
    quaternion = rotation.as_quat()[:, [3, 0, 1, 2]]
    gravity_body = rotation.inv().apply(np.tile([0., 0., 1.], (len(time), 1)))
    static = vertical_acceleration(quaternion, gravity_body*[1., -1., -1.])
    freefall = vertical_acceleration(quaternion, np.zeros((len(time), 3)))
    velocity, height, _ = integrate(time, np.full(len(time), 2.), 0., 0.)
    assert abs(static).max() < 1e-12
    assert abs(freefall+GRAVITY).max() < 1e-12
    assert abs(velocity[-1]-2.) < 1e-12 and abs(height[-1]-1.) < 1e-12
    return dict(rotated_stationary_max_accel_error=float(abs(static).max()),
                freefall_accel=-GRAVITY, constant_accel_final_velocity=float(velocity[-1]),
                constant_accel_final_displacement=float(height[-1]))


def run(archive_path, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = check_math()
    result = dict(archive_sha256=sha(archive_path), math_checks=checks, trials=[],
        gravity_m_s2=GRAVITY, method='H7 big-machine quaternion projection; pre-trigger mean bias; two trapezoidal integrals.',
        bias_window_s=[-.9, -.2], zero_vertical_velocity_anchor_s=-.2,
        height_reference='IMU position at first Jump inference, relative height zero.',
        constraints='No landing height or velocity constraint; no ground clamp; no filtering to force return.',
        limitations=['Velocity initialization is assumed, not measured.',
            'IMU lever arm to body origin/COM is unknown; relative IMU displacement approximates body translation.',
            'Bias, attitude error and impact sampling cause accumulated drift. Sensitivity bands are not confidence intervals.',
            'Reference projection is current local H7 source; deployed firmware identity is not independently established.'],
        projection_source=str(ROOT/'deployment/H7_RL/imcalib/Algorithm/estimation/Attitude_Algorithm.c'),
        projection_source_sha256=sha(ROOT/'deployment/H7_RL/imcalib/Algorithm/estimation/Attitude_Algorithm.c'))
    arrays = {'archive_sha256': np.array(result['archive_sha256'])}
    plots = []
    with zipfile.ZipFile(archive_path) as archive:
        numbers = sorted(int(re.fullmatch(r'试次(\d+)-H7反馈\.csv', n)[1]) for n in archive.namelist()
                         if re.fullmatch(r'试次(\d+)-H7反馈\.csv', n))
        for number in numbers:
            data = read_csv(archive, f'试次{number}-H7反馈.csv')
            t = data['rel_s']
            assert np.all(np.diff(t) > 0) and np.diff(t).max() < .01
            assert np.ptp(data['board_us']-t*1e6) < 1.
            q = matrix(data, ['quat_'+str(j) for j in range(4)])
            a = matrix(data, ['acc_g_'+str(j) for j in range(3)])
            assert np.isfinite(q).all() and np.isfinite(a).all()
            az = vertical_acceleration(q, a)
            quiet = (t >= -.9) & (t < -.2)
            bias = float(az[quiet].mean())
            velocity, height, anchor = integrate(t, az, bias)
            family = []
            for lo, hi in ((-.9, -.5), (-.5, -.2), (-.9, -.2)):
                mask = (t >= lo) & (t < hi)
                for start in (-.4, -.2, 0.):
                    _, alternative, _ = integrate(t, az, float(az[mask].mean()), start)
                    family.append(alternative)
            family = np.asarray(family)
            jump_window = (t >= 0) & (t <= 1.)
            candidates = np.flatnonzero(jump_window)
            peak = int(candidates[np.argmax(height[jump_window])])
            peaks = family[:, jump_window].max(axis=1)
            summary = dict(trial=number, samples=len(t), time_span_s=[float(t[0]), float(t[-1])],
                initial_bias_m_s2=bias, initial_accel_std_m_s2=float(az[quiet].std()),
                zero_velocity_anchor_actual_s=float(t[anchor]),
                relative_height_peak_first_second_m=float(height[peak]), peak_time_s=float(t[peak]),
                peak_range_under_baseline_choices_m=[float(peaks.min()), float(peaks.max())],
                final_relative_height_m=float(height[-1]), final_velocity_m_s=float(velocity[-1]),
                maximum_sample_gap_ms=float(np.diff(t).max()*1000),
                maximum_input_accel_norm_g=float(np.linalg.norm(a, axis=1).max()))
            result['trials'].append(summary)
            prefix = f'trial{number}_'
            for name, value in dict(board_us=data['board_us'], relative_s=t, height_m=height,
                velocity_m_s=velocity, acceleration_m_s2=az, corrected_acceleration_m_s2=az-bias,
                baseline_sensitivity_min_m=family.min(0), baseline_sensitivity_max_m=family.max(0)).items():
                arrays[prefix+name] = value
            plots.append((number, t, az-bias, velocity, height, family))
    np.savez_compressed(output/'height.npz', **arrays)
    write_json(output/'report.json', result)
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    for number, t, az, velocity, height, family in plots:
        label = f'试次{number}'
        short = (t >= -.2) & (t <= 1.)
        axes[0].plot(t[short], az[short], label=label, lw=1)
        axes[1].plot(t[short], velocity[short], label=label)
        line, = axes[2].plot(t[short], height[short]*100, label=label)
        axes[2].fill_between(t[short], family[:, short].min(0)*100, family[:, short].max(0)*100,
                             color=line.get_color(), alpha=.13)
        axes[3].plot(t, height*100, label=label)
    for axis, ylabel in zip(axes, ('竖直加速度 / m/s²', '积分竖直速度 / m/s', '短窗相对高度 / cm', '全程相对高度 / cm')):
        axis.set_ylabel(ylabel);axis.grid(alpha=.2);axis.legend();axis.axhline(0, color='grey', lw=.7)
        axis.set_xlabel('相对首次Jump推理 / s')
    axes[0].set_title('按现有H7坐标映射去重力后积分；初速度假设为0')
    axes[2].set_title('阴影为基准窗口/初值时刻敏感性，非统计置信区间', fontsize=11)
    axes[3].set_title('保留落地后漂移，未强制归零', fontsize=11)
    fig.savefig(output/'height-estimate.png', dpi=150);plt.close(fig)
    print(json.dumps(result['trials'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.output)
