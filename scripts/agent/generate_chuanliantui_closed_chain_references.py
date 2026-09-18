#!/usr/bin/env python3
"""生成 chuanliantui 的模型闭链位置映射检查轨迹。

这些 model_reference.csv 只验证训练串联虚拟关节到 MuJoCo 闭链模型主动轴的
几何求解与 connect 残差；真机与模型电机安装位置不同，禁止用本脚本输出
reference.csv 或将其下发。力矩 real2sim 应以真实/模型 FK 与 Jacobian 映射实现。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2sim.chuanliantui_closed_adapter import ClosedChainAdapter


VIRTUAL_JOINT_ORDER = ("lf0", "lf1", "lfwheel", "rf0", "rf1", "rfwheel")
PHYSICAL_ACTIVE_ORDER = ("lf0", "lf00", "rf0", "rf00")
DEFAULT_XML = REPO_ROOT / "sim2sim" / "chuanliantui.xml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "sysid" / "chuanliantui" / "reference"

# 与 ChuanliantuiCfg.init_state.default_joint_angles 一致。pose_low/mid/high
# 是围绕该微蹲姿态的低幅闭链辨识姿态，不是地面起立或行走轨迹。
POSES: Dict[str, Tuple[float, float, float, float]] = {
    "pose_low": (-0.12, 0.20, 0.12, -0.20),
    "pose_mid": (-0.06, 0.10, 0.06, -0.10),
    "pose_high": (-0.02, 0.03, 0.02, -0.03),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="真实闭链 MJCF")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-hz", type=int, default=200, help="CSV 采样率 [Hz]")
    parser.add_argument("--settle-s", type=float, default=2.0, help="轨迹前、后保持时间 [s]")
    parser.add_argument("--motion-s", type=float, default=6.0, help="中间激励时长 [s]")
    parser.add_argument("--cycles", type=int, default=2, help="中间激励周期数")
    parser.add_argument("--amplitude-rad", type=float, default=0.02,
                        help="虚拟 lf0/lf1 的激励幅值 [rad]")
    parser.add_argument("--reference-version", default="v1")
    return parser.parse_args()


def virtual_pose(seed: Tuple[float, float, float, float], excitation: float) -> np.ndarray:
    """把镜像的虚拟腿目标组装为训练 DOF 顺序。"""
    lf0, lf1, rf0, rf1 = seed
    # 两个虚拟腿同时做镜像低幅屈伸；轮始终不参与关节辨识轨迹。
    return np.array(
        (lf0 + excitation, lf1 - excitation, 0.0, rf0 - excitation, rf1 + excitation, 0.0),
        dtype=np.float64,
    )


def smooth_excitation(t: float, settle_s: float, motion_s: float, cycles: int, amplitude_rad: float) -> float:
    """端点速度为零的双向激励；保持段的激励严格为零。"""
    if not settle_s <= t <= settle_s + motion_s:
        return 0.0
    phase = 2.0 * math.pi * cycles * (t - settle_s) / motion_s
    return amplitude_rad * math.sin(phase) ** 3


def connect_residual(adapter: ClosedChainAdapter) -> float:
    return max(float(np.max(np.abs(adapter._pin_residual(side)))) for side in adapter._SIDES)


def physical_active_pose(adapter: ClosedChainAdapter, target: np.ndarray) -> Tuple[float, float, float, float]:
    adapter.set_virtual_pose(target)
    values = tuple(float(adapter.data.qpos[adapter.qpos[name]]) for name in PHYSICAL_ACTIVE_ORDER)
    residual = connect_residual(adapter)
    if residual > 1e-8:
        raise RuntimeError("闭链 connect 残差过大: {:.3e} m".format(residual))
    return values


def rows_for_pose(adapter: ClosedChainAdapter, seed: Tuple[float, float, float, float], args: argparse.Namespace) -> Iterable[Tuple[float, float, float, float, float]]:
    total_s = 2.0 * args.settle_s + args.motion_s
    sample_count = int(round(total_s * args.sample_hz)) + 1
    for index in range(sample_count):
        t = index / args.sample_hz
        excitation = smooth_excitation(t, args.settle_s, args.motion_s, args.cycles, args.amplitude_rad)
        yield (t, *physical_active_pose(adapter, virtual_pose(seed, excitation)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reference(output_dir: Path, pose_name: str, seed: Tuple[float, float, float, float],
                    adapter: ClosedChainAdapter, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_csv_path = output_dir / "model_reference.csv"
    min_values = np.full(4, np.inf)
    max_values = np.full(4, -np.inf)
    max_residual = 0.0

    with model_csv_path.open("w", newline="", encoding="utf-8") as model_stream:
        model_writer = csv.writer(model_stream, lineterminator="\n")
        model_writer.writerow(("time_s", "q_model_lf0_rad", "q_model_lf00_rad", "q_model_rf0_rad", "q_model_rf00_rad"))
        for row in rows_for_pose(adapter, seed, args):
            values = np.asarray(row[1:], dtype=np.float64)
            min_values = np.minimum(min_values, values)
            max_values = np.maximum(max_values, values)
            max_residual = max(max_residual, connect_residual(adapter))
            model_writer.writerow(("{:.6f}".format(row[0]), *("{:.9f}".format(value) for value in values)))

    metadata = {
        "reference_id": output_dir.name,
        "pose": pose_name,
        "source_model": str(args.xml.relative_to(REPO_ROOT)),
        "generator": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "physical_active_joint_order": list(PHYSICAL_ACTIVE_ORDER),
        "virtual_joint_order": list(VIRTUAL_JOINT_ORDER),
        "virtual_pose_seed_rad": dict(zip(("lf0", "lf1", "rf0", "rf1"), seed)),
        "sample_hz": args.sample_hz,
        "settle_s": args.settle_s,
        "motion_s": args.motion_s,
        "cycles": args.cycles,
        "virtual_excitation_amplitude_rad": args.amplitude_rad,
        "q_model_min_rad": dict(zip(PHYSICAL_ACTIVE_ORDER, min_values.tolist())),
        "q_model_max_rad": dict(zip(PHYSICAL_ACTIVE_ORDER, max_values.tolist())),
        "max_connect_residual_m": max_residual,
        "model_reference_sha256": sha256(model_csv_path),
        "hardware_reference_sha256": None,
        "hardware_export": "disabled: requires FK_real/J_real torque mapping, not affine position calibration",
        "hardware_note": "model_reference.csv 仅为模型闭链位置映射检查，不能下发真机。真实/模型电机安装位置不同，硬件力矩回放必须先建立并验证 FK_real/J_real 与 FK_model/J_model。",
    }
    (output_dir / "reference_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("generated {} (sha256={}, max_residual={:.3e} m)".format(
        model_csv_path.relative_to(REPO_ROOT), metadata["model_reference_sha256"], max_residual
    ))


def main() -> None:
    args = parse_args()
    if args.sample_hz < 200:
        raise ValueError("--sample-hz 必须不低于 200 Hz")
    if args.settle_s <= 0 or args.motion_s <= 0 or args.cycles < 1 or args.amplitude_rad <= 0:
        raise ValueError("保持时间、运动时间、周期数和幅值必须为正")
    if not args.xml.is_file():
        raise FileNotFoundError("闭链 MJCF 不存在: {}".format(args.xml))

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if model.neq != 4:
        raise RuntimeError("期望 4 个 connect 约束，实际 neq={}".format(model.neq))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = ClosedChainAdapter(mujoco, model, data)

    for pose_name, seed in POSES.items():
        reference_id = "{}_{}".format(pose_name, args.reference_version)
        write_reference(args.output_root / reference_id, pose_name, seed, adapter, args)


if __name__ == "__main__":
    main()
