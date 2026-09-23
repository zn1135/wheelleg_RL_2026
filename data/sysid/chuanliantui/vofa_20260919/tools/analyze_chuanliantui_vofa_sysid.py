#!/usr/bin/env python3
"""生成 chuanliantui VOFA+ 原始 CSV 的准静态辨识准入报告。

不改写原始 VOFA 导出文件。髋数据仅抽取在线、左右目标一致且目标稳定的
保持段末尾窗口；轮数据只检查是否为预期的 10 通道电流平台帧。输出用于
决定能否进入后续 MuJoCo 回放，不直接回填任何动力学参数。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from math import pi
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
HIP_CHANNELS = 32
WHEEL_SYSID_CHANNELS = 10
HIP_SAMPLE_HZ = 200.0
HIP_TARGET_TOL_RAD = 1e-4
EXPECTED_WHEEL_COMMAND_RAW = (-3276.0, -2457.0, -1638.0, -819.0, -409.0, 409.0, 819.0, 1638.0, 2457.0, 3276.0)


@dataclass(frozen=True)
class HipWindow:
    """一个用于准静态对齐的保持段末尾窗口。"""

    pose_id: str
    source_start_frame: int
    source_end_frame: int
    fit_start_frame: int
    fit_end_frame: int
    target_thigh_rad: float
    target_shin_rad: float
    rows: Sequence[Sequence[float]]


@dataclass(frozen=True)
class WheelPlateau:
    """一个以轮测试帧时间戳界定的恒电流平台。"""

    source_start_frame: int
    source_end_frame: int
    command_raw: float
    rows: Sequence[Sequence[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hip-csv", type=Path, default=REPO_ROOT / "髋关节.csv", help="32 通道髋 VOFA 原始 CSV"
    )
    parser.add_argument(
        "--wheel-csv", type=Path, default=REPO_ROOT / "轮关节.csv", help="轮 VOFA 原始 CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sysid" / "chuanliantui" / "vofa_20260919",
        help="仅写入派生 QA/统计结果的目录",
    )
    parser.add_argument("--hold-window-s", type=float, default=1.5, help="每段末尾用于统计的窗口 [s]")
    parser.add_argument("--min-stable-s", type=float, default=1.5, help="可接受目标稳定段的最短时长 [s]")
    parser.add_argument("--torque-limit-nm", type=float, default=10.0, help="髋测试声明的力矩限幅 [N m]")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_float_csv(path: Path) -> Tuple[List[str], List[List[float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV 为空: {}".format(path)) from exc
        rows: List[List[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError("{}:{} 列数为 {}，期望 {}".format(path, line_number, len(row), len(header)))
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError("{}:{} 含非浮点数".format(path, line_number)) from exc
    return header, rows


def is_hip_target_matched(row: Sequence[float]) -> bool:
    return (
        int(row[0]) == 255
        and abs(row[1] - row[7]) <= HIP_TARGET_TOL_RAD
        and abs(row[4] - row[10]) <= HIP_TARGET_TOL_RAD
    )


def target_key(row: Sequence[float]) -> Tuple[float, float]:
    return (round(row[1], 4), round(row[4], 4))


def contiguous_target_runs(rows: Sequence[Sequence[float]]) -> Iterable[Tuple[int, int, Tuple[float, float]]]:
    """返回在线且左右一致的相同目标连续区间，帧索引为零基且 end 为开区间。"""
    start = None
    previous_key = None
    for index, row in enumerate(rows):
        key = target_key(row) if is_hip_target_matched(row) else None
        if key != previous_key:
            if previous_key is not None:
                yield start, index, previous_key
            start = index if key is not None else None
            previous_key = key
    if previous_key is not None:
        yield start, len(rows), previous_key


def select_hip_windows(rows: Sequence[Sequence[float]], args: argparse.Namespace) -> List[HipWindow]:
    minimum_frames = round(args.min_stable_s * HIP_SAMPLE_HZ)
    window_frames = round(args.hold_window_s * HIP_SAMPLE_HZ)
    selected: List[HipWindow] = []
    for start, end, (thigh, shin) in contiguous_target_runs(rows):
        count = end - start
        # 0/0 是上电后的待机目标，不是扫描姿态。
        if count < minimum_frames or (abs(thigh) <= HIP_TARGET_TOL_RAD and abs(shin) <= HIP_TARGET_TOL_RAD):
            continue
        fit_start = max(start, end - window_frames)
        pose_id = "pose_{:02d}".format(len(selected) + 1)
        selected.append(
            HipWindow(
                pose_id=pose_id,
                source_start_frame=start,
                source_end_frame=end - 1,
                fit_start_frame=fit_start,
                fit_end_frame=end - 1,
                target_thigh_rad=thigh,
                target_shin_rad=shin,
                rows=rows[fit_start:end],
            )
        )
    return selected


def maximum_abs(rows: Sequence[Sequence[float]], column: int) -> float:
    return max(abs(row[column]) for row in rows)


def write_hip_summary(path: Path, windows: Sequence[HipWindow], torque_limit_nm: float) -> List[Dict[str, object]]:
    fields = (
        "pose_id",
        "source_start_frame",
        "source_end_frame",
        "fit_start_frame",
        "fit_end_frame",
        "fit_samples",
        "target_thigh_rad",
        "target_shin_rad",
        "q_lf0_mean_rad",
        "q_lf00_mean_rad",
        "tau_lf0_mean_Nm",
        "tau_lf00_mean_Nm",
        "q_rf0_mean_rad",
        "q_rf00_mean_rad",
        "tau_rf0_mean_Nm",
        "tau_rf00_mean_Nm",
        "leg_pitch_left_mean_rad",
        "leg_length_left_mean_m",
        "leg_pitch_right_mean_rad",
        "leg_length_right_mean_m",
        "max_abs_tau_Nm",
        "accepted_for_quasistatic_fit",
        "rejection_reason",
    )
    records: List[Dict[str, object]] = []
    for window in windows:
        max_abs_tau = max(maximum_abs(window.rows, column) for column in (3, 6, 9, 12))
        accepted = max_abs_tau < torque_limit_nm
        records.append(
            {
                "pose_id": window.pose_id,
                "source_start_frame": window.source_start_frame,
                "source_end_frame": window.source_end_frame,
                "fit_start_frame": window.fit_start_frame,
                "fit_end_frame": window.fit_end_frame,
                "fit_samples": len(window.rows),
                "target_thigh_rad": window.target_thigh_rad,
                "target_shin_rad": window.target_shin_rad,
                "q_lf0_mean_rad": mean(row[2] for row in window.rows),
                "q_lf00_mean_rad": mean(row[5] for row in window.rows),
                "tau_lf0_mean_Nm": mean(row[3] for row in window.rows),
                "tau_lf00_mean_Nm": mean(row[6] for row in window.rows),
                "q_rf0_mean_rad": mean(row[8] for row in window.rows),
                "q_rf00_mean_rad": mean(row[11] for row in window.rows),
                "tau_rf0_mean_Nm": mean(row[9] for row in window.rows),
                "tau_rf00_mean_Nm": mean(row[12] for row in window.rows),
                "leg_pitch_left_mean_rad": mean(row[28] for row in window.rows),
                "leg_length_left_mean_m": mean(row[29] for row in window.rows),
                "leg_pitch_right_mean_rad": mean(row[30] for row in window.rows),
                "leg_length_right_mean_m": mean(row[31] for row in window.rows),
                "max_abs_tau_Nm": max_abs_tau,
                "accepted_for_quasistatic_fit": str(accepted).lower(),
                "rejection_reason": "" if accepted else "力矩达到或超过声明的 ±{:.1f} N m 限幅".format(torque_limit_nm),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return records


def is_wheel_test_row(row: Sequence[float]) -> bool:
    """判断前 10 列是否满足 wheel-test-vofa.md 的物理字段范围。

    VOFA 导出的 CSV 可以有多余列；轮测试的有效载荷固定是 I0..I9，故不以
    总列数拒绝整份文件。I8/I9 的微秒时间戳与 I4/I5 的编码器范围用于排除
    同一文件开头混入的非轮测试帧。
    """
    return (
        int(row[0]) == 255
        and 0.0 <= row[4] <= 8191.0
        and 0.0 <= row[5] <= 8191.0
        and row[8] >= 1_000_000.0
        and row[9] >= 1_000_000.0
    )


def wheel_plateaus(rows: Sequence[Sequence[float]]) -> Iterable[WheelPlateau]:
    """按源帧连续性和命令值切分测试平台，不跨越丢帧区间拼接。"""
    start = None
    command_raw = None
    previous_index = None
    collected: List[Sequence[float]] = []
    for index, row in enumerate(rows):
        command = row[1] if is_wheel_test_row(row) else None
        same_plateau = (
            command is not None
            and command == command_raw
            and previous_index is not None
            and index == previous_index + 1
        )
        if not same_plateau:
            if collected:
                yield WheelPlateau(start, previous_index, command_raw, tuple(collected))
            collected = []
            start = index if command is not None else None
            command_raw = command
        if command is not None:
            collected.append(row)
            previous_index = index
        else:
            previous_index = None
    if collected:
        yield WheelPlateau(start, previous_index, command_raw, tuple(collected))


def write_wheel_summary(path: Path, rows: Sequence[Sequence[float]]) -> List[Dict[str, object]]:
    fields = (
        "plateau_id",
        "source_start_frame",
        "source_end_frame",
        "cmd_raw",
        "cmd_A",
        "duration_s",
        "frame_count",
        "observed_hz",
        "tail_window_samples",
        "left_current_mean_A",
        "right_current_mean_A",
        "left_speed_mean_rad_s",
        "right_speed_mean_rad_s",
        "mean_can_delay_ms",
        "accepted_for_steady_state_fit",
        "reason",
    )
    records: List[Dict[str, object]] = []
    for plateau in wheel_plateaus(rows):
        if plateau.command_raw == 0.0:
            continue
        duration_us = plateau.rows[-1][8] - plateau.rows[0][8]
        duration_s = duration_us / 1e6
        tail_rows = [row for row in plateau.rows if row[8] >= plateau.rows[-1][8] - 300_000.0]
        accepted = duration_s >= 0.5 and len(tail_rows) >= 20
        records.append(
            {
                "plateau_id": "plateau_{:02d}".format(len(records) + 1),
                "source_start_frame": plateau.source_start_frame,
                "source_end_frame": plateau.source_end_frame,
                "cmd_raw": plateau.command_raw,
                "cmd_A": plateau.command_raw / 819.2,
                "duration_s": duration_s,
                "frame_count": len(plateau.rows),
                "observed_hz": len(plateau.rows) / duration_s if duration_s > 0 else 0.0,
                "tail_window_samples": len(tail_rows),
                "left_current_mean_A": mean(row[2] for row in tail_rows) / 819.2,
                "right_current_mean_A": mean(row[3] for row in tail_rows) / 819.2,
                "left_speed_mean_rad_s": mean(row[6] for row in tail_rows),
                "right_speed_mean_rad_s": mean(row[7] for row in tail_rows),
                "mean_can_delay_ms": mean(row[9] - row[8] for row in tail_rows) / 1000.0,
                "accepted_for_steady_state_fit": str(accepted).lower(),
                "reason": "" if accepted else "平台时长不足 0.5 s 或末尾 0.3 s 少于 20 帧",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return records


def wrap_to_pi(angle: float) -> float:
    """将角度误差规约到 [-pi, pi)，避免编码器跨圈被误判为大偏差。"""
    return (angle + pi) % (2.0 * pi) - pi


def affine_fit(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """拟合 y = slope*x + intercept，并返回残差指标。"""
    if len(points) < 2:
        raise ValueError("仿射拟合至少需要两个点")
    x_mean = mean(point[0] for point in points)
    y_mean = mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0.0:
        raise ValueError("自变量没有变化，无法拟合")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [y - (slope * x + intercept) for x, y in points]
    rmse = (sum(residual * residual for residual in residuals) / len(residuals)) ** 0.5
    total = sum((y - y_mean) ** 2 for _, y in points)
    r_squared = 1.0 - sum(residual * residual for residual in residuals) / total if total else 1.0
    return {"slope": slope, "intercept": intercept, "rmse": rmse, "r_squared": r_squared}


def write_wheel_equivalent_fit(path: Path, records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """拟合稳态 I = I_c + k_v*|omega|，作为摩擦/阻尼的可复现实测代理。

    电流尚未换算为轮端力矩，故这些结果不能直接填入 MuJoCo 的
    frictionloss/damping；其目的是保留每侧、每个转向的原始可比标定量。
    """
    fields = (
        "wheel", "direction", "input_plateau_count", "fit_point_count", "excluded_stall_count",
        "coulomb_current_A", "viscous_current_A_per_rad_s", "fit_rmse_A", "fit_r_squared",
        "status", "note",
    )
    output: List[Dict[str, object]] = []
    for wheel, current_key, speed_key in (
        ("left", "left_current_mean_A", "left_speed_mean_rad_s"),
        ("right", "right_current_mean_A", "right_speed_mean_rad_s"),
    ):
        for direction, sign in (("positive", 1.0), ("negative", -1.0)):
            candidates = [
                record for record in records
                if record["accepted_for_steady_state_fit"] == "true" and float(record["cmd_A"]) * sign > 0
            ]
            # 低于 5 rad/s 的平台尚可能受静摩擦支配，不能用于线性粘滞段。
            points = [
                (sign * float(record[speed_key]), sign * float(record[current_key]))
                for record in candidates
                if sign * float(record[speed_key]) >= 5.0 and sign * float(record[current_key]) > 0.0
            ]
            rejected = len(candidates) - len(points)
            result: Dict[str, object] = {
                "wheel": wheel,
                "direction": direction,
                "input_plateau_count": len(candidates),
                "fit_point_count": len(points),
                "excluded_stall_count": rejected,
                "coulomb_current_A": "",
                "viscous_current_A_per_rad_s": "",
                "fit_rmse_A": "",
                "fit_r_squared": "",
                "status": "insufficient_data",
                "note": "至少需要两个非低速稳态点",
            }
            if len(points) >= 2:
                fit = affine_fit(points)
                result.update(
                    {
                        "coulomb_current_A": fit["intercept"],
                        "viscous_current_A_per_rad_s": fit["slope"],
                        "fit_rmse_A": fit["rmse"],
                        "fit_r_squared": fit["r_squared"],
                        "status": "observational_fit_only",
                        "note": "I = I_c + k_v*|omega|；未做电流—力矩和传动效率换算",
                    }
                )
            output.append(result)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return output


def write_hip_tracking_fit(path: Path, records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """汇总位置扫描的静态跟踪偏差；不将下发力矩误当成真实动力学力矩。"""
    fields = (
        "side", "joint", "accepted_pose_count", "mean_tracking_error_rad", "rmse_tracking_error_rad",
        "min_tracking_error_rad", "max_tracking_error_rad", "mean_commanded_torque_Nm", "status", "note",
    )
    output: List[Dict[str, object]] = []
    layouts = (
        ("left", "thigh", "target_thigh_rad", "q_lf0_mean_rad", "tau_lf0_mean_Nm"),
        ("left", "shin", "target_shin_rad", "q_lf00_mean_rad", "tau_lf00_mean_Nm"),
        ("right", "thigh", "target_thigh_rad", "q_rf0_mean_rad", "tau_rf0_mean_Nm"),
        ("right", "shin", "target_shin_rad", "q_rf00_mean_rad", "tau_rf00_mean_Nm"),
    )
    accepted = [record for record in records if record["accepted_for_quasistatic_fit"] == "true"]
    for side, joint, target_key, position_key, torque_key in layouts:
        errors = [wrap_to_pi(float(record[position_key]) - float(record[target_key])) for record in accepted]
        result: Dict[str, object] = {
            "side": side,
            "joint": joint,
            "accepted_pose_count": len(errors),
            "mean_tracking_error_rad": mean(errors) if errors else "",
            "rmse_tracking_error_rad": (mean(error * error for error in errors) ** 0.5) if errors else "",
            "min_tracking_error_rad": min(errors) if errors else "",
            "max_tracking_error_rad": max(errors) if errors else "",
            "mean_commanded_torque_Nm": mean(float(record[torque_key]) for record in accepted) if accepted else "",
            "status": "quasistatic_observation_only" if errors else "insufficient_data",
            "note": "下发力矩非实际输出力矩；无时间戳，不能拟合惯量、时延或动态阻尼",
        }
        output.append(result)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return output


def longest_active_online_span(rows: Sequence[Sequence[float]]) -> Tuple[int, int]:
    """返回最长的全在线、非零参考轨迹区间，end 为开区间。"""
    spans: List[Tuple[int, int]] = []
    start = None
    for index, row in enumerate(rows):
        active = int(row[0]) == 255 and any(abs(row[column]) > HIP_TARGET_TOL_RAD for column in (1, 4, 7, 10))
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(rows)))
    if not spans:
        raise ValueError("未找到全在线的非零髋关节参考轨迹")
    return max(spans, key=lambda span: span[1] - span[0])


def write_hip_trajectory_replay(path: Path, rows: Sequence[Sequence[float]]) -> Dict[str, object]:
    """导出 real2sim 逐样本回放载荷。

    左右腿在斜坡起点可从各自上一帧实测角出发，因而短暂出现左右参考
    不同是可接受的；回放必须使用各自的四路参考，不能为追求对称而丢帧。
    """
    start, end = longest_active_online_span(rows)
    fields = (
        "source_frame", "t_s",
        "q_ref_lf0_rad", "q_lf0_rad", "tau_cmd_lf0_Nm",
        "q_ref_lf00_rad", "q_lf00_rad", "tau_cmd_lf00_Nm",
        "q_ref_rf0_rad", "q_rf0_rad", "tau_cmd_rf0_Nm",
        "q_ref_rf00_rad", "q_rf00_rad", "tau_cmd_rf00_Nm",
        "leg_pitch_left_rad", "leg_length_left_m", "leg_pitch_right_rad", "leg_length_right_m",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for step, row in enumerate(rows[start:end]):
            writer.writerow(
                {
                    "source_frame": start + step,
                    "t_s": step / HIP_SAMPLE_HZ,
                    "q_ref_lf0_rad": row[1], "q_lf0_rad": row[2], "tau_cmd_lf0_Nm": row[3],
                    "q_ref_lf00_rad": row[4], "q_lf00_rad": row[5], "tau_cmd_lf00_Nm": row[6],
                    "q_ref_rf0_rad": row[7], "q_rf0_rad": row[8], "tau_cmd_rf0_Nm": row[9],
                    "q_ref_rf00_rad": row[10], "q_rf00_rad": row[11], "tau_cmd_rf00_Nm": row[12],
                    "leg_pitch_left_rad": row[28], "leg_length_left_m": row[29],
                    "leg_pitch_right_rad": row[30], "leg_length_right_m": row[31],
                }
            )
    return {
        "source_start_frame": start,
        "source_end_frame": end - 1,
        "sample_count": end - start,
        "duration_s": (end - start) / HIP_SAMPLE_HZ,
        "sample_hz_assumed": HIP_SAMPLE_HZ,
    }


def wheel_summary(rows: Sequence[Sequence[float]], column_count: int, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    test_rows = [row for row in rows if is_wheel_test_row(row)]
    accepted_commands = {
        record["cmd_raw"]
        for record in records
        if record["accepted_for_steady_state_fit"] == "true"
    }
    missing_commands = [command for command in EXPECTED_WHEEL_COMMAND_RAW if command not in accepted_commands]
    return {
        "frame_count": len(rows),
        "column_count": column_count,
        "payload_columns_used": "I0..I9",
        "expected_payload_column_count": WHEEL_SYSID_CHANNELS,
        "all_online_frame_count": sum(int(row[0]) == 255 for row in rows),
        "wheel_test_frame_count": len(test_rows),
        "wheel_test_timestamp_start_us": test_rows[0][8] if test_rows else None,
        "wheel_test_timestamp_end_us": test_rows[-1][8] if test_rows else None,
        "status": "accepted_for_steady_state_wheel_fit_with_frame_gaps" if not missing_commands else "incomplete_wheel_test",
        "accepted_plateau_count": sum(record["accepted_for_steady_state_fit"] == "true" for record in records),
        "plateau_fragment_count": len(records),
        "accepted_command_raw": sorted(accepted_commands),
        "missing_expected_command_raw": missing_commands,
        "detail": "只使用前 10 列轮测试载荷；额外列不参与轮侧辨识。+3 A 平台被 22 帧非轮测试数据切断，使用切断后的完整尾窗；电流平台末尾 0.3 s 的均值已写入 wheel_plateau_windows.csv。",
    }


def write_readme(path: Path, hip: Dict[str, object], trajectory: Dict[str, object], wheel: Dict[str, object], records: Sequence[Dict[str, object]], wheel_records: Sequence[Dict[str, object]], wheel_fits: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    accepted = sum(record["accepted_for_quasistatic_fit"] == "true" for record in records)
    wheel_fit_count = sum(record["status"] == "observational_fit_only" for record in wheel_fits)
    content = (
        "# VOFA 原始数据准入结果\n\n"
        "本目录由 `scripts/agent/analyze_chuanliantui_vofa_sysid.py` 生成；不替代或改写仓库根目录的原始 CSV。\n\n"
        "## 髋/腿\n\n"
        "- 原始帧：{}；按 200 Hz 重建时长 {:.3f} s。\n"
        "- 全在线帧：{}；左右参考不同帧：{}（斜坡从各侧上一帧实测角起算时允许出现）。\n"
        "- `hip_trajectory_replay.csv`：连续全在线参考轨迹 {} 帧 / {:.3f} s（源帧 {}–{}）；"
        "用于 MuJoCo 以同一控制器逐样本回放。\n"
        "- 检出的稳定目标段：{}；通过声明的 ±{:.1f} N·m 窗口限幅检查：{}。\n"
        "- `hip_quasistatic_windows.csv` 每行是一个稳定段末尾 {:.1f} s 的均值，只用于准静态对齐。\n"
        "- 本帧无硬件时间戳；初次回放按固定 200 Hz 对齐，时延只可作为待优化的离散样本偏移。\n\n"
        "## 轮\n\n"
        "- 原始帧：{}；CSV 有 {} 列，但轮测试只使用前 10 列 `I0…I9`。\n"
        "- 识别到轮测试有效帧：{}；时间戳范围 {:.3f}–{:.3f} s。\n"
        "- 非零电流平台片段：{}；通过稳态窗口检查：{}；覆盖命令档位：{}。\n"
        "- 判定：{}。\n"
        "- {}\n\n"
        "## 本批可交付的辨识量\n\n"
        "- `wheel_equivalent_fits.csv`：{} 侧向/转向的稳态等效电流模型，形式为 `I = I_c + k_v·|ω|`。"
        "它是观测拟合，尚未按电流常数、总传动比及效率换算为轮端力矩，不能直接填 MuJoCo。\n"
        "- `hip_trajectory_replay.csv`：完整动态参考、实测角、下发力矩与腿任务坐标；"
        "这是 real2sim 对齐输入，不是仅供静态统计的数据。\n"
        "- `hip_quasistatic_tracking.csv`：稳定段摘要，用于辅助检查零位/几何/重力或气弹簧偏置。\n\n"
        "## 后续边界\n\n"
        "应在 MuJoCo 复刻固件的控制律后，以 `hip_trajectory_replay.csv` 的参考轨迹驱动模型，"
        "将模拟实测角与真机实测角逐点对齐，再拟合关节 `armature`、`damping`、`frictionloss`。"
        "下发力矩用于核对控制器输出；不可单独当作真实力矩标定。\n"
    ).format(
        hip["frame_count"], hip["duration_s"], hip["all_online_frame_count"], hip["target_mismatch_frame_count"],
        trajectory["sample_count"], trajectory["duration_s"], trajectory["source_start_frame"], trajectory["source_end_frame"],
        hip["stable_target_segment_count"], args.torque_limit_nm, accepted, args.hold_window_s,
        wheel["frame_count"], wheel["column_count"], wheel["wheel_test_frame_count"], wheel["wheel_test_timestamp_start_us"] / 1e6,
        wheel["wheel_test_timestamp_end_us"] / 1e6, wheel["plateau_fragment_count"], wheel["accepted_plateau_count"], len(wheel["accepted_command_raw"]), wheel["status"], wheel["detail"], wheel_fit_count,
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.hold_window_s <= 0 or args.min_stable_s <= 0 or args.torque_limit_nm <= 0:
        raise ValueError("保持窗口、最短稳定时间和力矩限幅必须为正")
    hip_header, hip_rows = read_float_csv(args.hip_csv)
    wheel_header, wheel_rows = read_float_csv(args.wheel_csv)
    if len(hip_header) != HIP_CHANNELS:
        raise ValueError("髋 CSV 必须为 {} 通道，实际为 {}".format(HIP_CHANNELS, len(hip_header)))
    if len(wheel_header) not in (WHEEL_SYSID_CHANNELS, HIP_CHANNELS):
        raise ValueError("轮 CSV 必须为 10 或 32 通道，实际为 {}".format(len(wheel_header)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = select_hip_windows(hip_rows, args)
    records = write_hip_summary(args.output_dir / "hip_quasistatic_windows.csv", windows, args.torque_limit_nm)
    hip_trajectory = write_hip_trajectory_replay(args.output_dir / "hip_trajectory_replay.csv", hip_rows)
    wheel_records = write_wheel_summary(args.output_dir / "wheel_plateau_windows.csv", wheel_rows)
    hip_tracking = write_hip_tracking_fit(args.output_dir / "hip_quasistatic_tracking.csv", records)
    wheel_fits = write_wheel_equivalent_fit(args.output_dir / "wheel_equivalent_fits.csv", wheel_records)
    hip_summary: Dict[str, object] = {
        "source": str(args.hip_csv.resolve().relative_to(REPO_ROOT)),
        "source_sha256": sha256(args.hip_csv),
        "frame_count": len(hip_rows),
        "sample_hz_assumed": HIP_SAMPLE_HZ,
        "duration_s": len(hip_rows) / HIP_SAMPLE_HZ,
        "all_online_frame_count": sum(int(row[0]) == 255 for row in hip_rows),
        "target_mismatch_frame_count": sum(not is_hip_target_matched(row) for row in hip_rows),
        "stable_target_segment_count": len(windows),
        "hold_window_s": args.hold_window_s,
        "declared_torque_limit_nm": args.torque_limit_nm,
        "accepted_window_count": sum(record["accepted_for_quasistatic_fit"] == "true" for record in records),
        "trajectory_replay": hip_trajectory,
        "windows": records,
        "limitations": [
            "源帧没有硬件时间戳；只可按固定 200 Hz 重建近似时间轴。",
            "力矩列是下发力矩，不是经电流/标定确认的实际输出力矩。",
            "轨迹可用于 real2sim 曲线对齐；需复刻固件控制律后才可拟合动力学参数。",
        ],
    }
    wheel = wheel_summary(wheel_rows, len(wheel_header), wheel_records)
    wheel.update({"source": str(args.wheel_csv.resolve().relative_to(REPO_ROOT)), "source_sha256": sha256(args.wheel_csv)})
    (args.output_dir / "hip_qa.json").write_text(json.dumps(hip_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "wheel_qa.json").write_text(json.dumps(wheel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(args.output_dir / "README.md", hip_summary, hip_trajectory, wheel, records, wheel_records, wheel_fits, args)
    print("wrote hip replay: {} frames / {:.3f} s; {} stable hip windows ({} accepted, {} channel summaries); {} wheel plateaus ({} accepted, {} directional fits)".format(
        hip_trajectory["sample_count"], hip_trajectory["duration_s"], len(windows), hip_summary["accepted_window_count"], len(hip_tracking), wheel["plateau_fragment_count"], wheel["accepted_plateau_count"],
        sum(record["status"] == "observational_fit_only" for record in wheel_fits)
    ))


if __name__ == "__main__":
    main()
