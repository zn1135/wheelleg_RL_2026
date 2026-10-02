#!/usr/bin/env python3
"""Read-only board/CAD check of four zero-pose virtual leg angles.

Uses a disabled 2701 observation capture and a separate disabled STATUS
capture. It never opens a serial device or sends a motor command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2sim.chuanliantui_closed_adapter import ClosedChainAdapter, _PlanarLegGeometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bilateral-fit", type=Path, required=True)
    parser.add_argument("--diag-records", type=Path, required=True)
    parser.add_argument("--status-records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit = json.loads(args.bilateral_fit.read_text(encoding="utf-8"))
    if not fit.get("left_mirror_pose_user_confirmed"):
        raise RuntimeError("left mirrored pose was not user-confirmed")
    board = [json.loads(line) for line in args.diag_records.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    samples = [item for item in board if item.get("event") == "sample"]
    status = [json.loads(line) for line in args.status_records.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 100 or len(status) < 25:
        raise RuntimeError("insufficient disabled samples")
    for item in samples:
        header = item["header"]
        if header["layout"] != 2701 or header["online_mask"] != 63 or header["dm_enabled_mask"] != 0:
            raise RuntimeError("diagnostic was not fully online and DM-disabled")
        if len(item["obs"]) != 25:
            raise RuntimeError("observation dimension mismatch")
    for item in status:
        remote, motor = item["remote"], item["motor"]
        if (remote["online"] != 1 or remote["left_switch"] != 2
            or remote["right_switch"] != 2 or remote["armed"] != 0
            or remote["permit"] != 0 or remote["enabled_mask"] != 0
            or motor["online"] != 63 or motor["enabled"] != 0
            or any(value != 0.0 for value in motor["target"])):
            raise RuntimeError("STATUS capture was not safe and disabled")
    dm = np.asarray([item["motor"]["dm_pos"] for item in status], dtype=float)
    if dm.shape != (len(status), 4) or not np.all(np.isfinite(dm)):
        raise RuntimeError("nonfinite or wrong-sized DM feedback")
    if np.max(np.ptp(dm, axis=0)) > 0.005:
        raise RuntimeError("supported zero pose moved during STATUS capture")
    pos = np.mean(dm, axis=0)
    zeros = fit["conditional_cad_angle_zero_by_model_side_rad"]
    model = mujoco.MjModel.from_xml_path(str(ROOT / "sim2sim/chuanliantui.xml"))
    left_model = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[0])
    right_model = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[1])
    lf_front = -wrap(pos[3] - zeros["lf_physical_right"]["odd_front_dm3"])
    lf_rear = -wrap(pos[2] + np.pi - zeros["lf_physical_right"]["even_rear_dm2"])
    rf_front = wrap(pos[1] - zeros["rf_physical_left"]["odd_front_dm1"])
    rf_rear = wrap(pos[0] + np.pi - zeros["rf_physical_left"]["even_rear_dm0"])
    lf_knee, _ = left_model.knee_and_jacobian(lf_front, lf_rear)
    rf_knee, _ = right_model.knee_and_jacobian(rf_front, rf_rear)
    expected = np.array((lf_front, lf_knee, rf_front, rf_knee), dtype=float)
    defaults = np.array((-0.06, 0.10, 0.06, -0.10), dtype=float)
    actual = np.asarray([item["obs"][9:13] for item in samples], dtype=float) + defaults
    if not np.all(np.isfinite(actual)):
        raise RuntimeError("nonfinite board joint observation")
    difference = actual - expected
    max_error = np.max(np.abs(difference), axis=0)
    if np.max(max_error) > 0.002:
        raise RuntimeError("board zero observation differs from independent CAD reference: "
                           + str(max_error.tolist()))
    result = {
        "result": "pass", "scope": "disabled same-pose numerical board/CAD comparison only; no physical sign or torque proof",
        "diagnostic_samples": len(samples), "status_samples": len(status),
        "dm_pos_zero_rad_mean": pos.tolist(),
        "expected_joint_pos_lf0_lf1_rf0_rf1_rad": expected.tolist(),
        "board_joint_pos_mean_rad": np.mean(actual, axis=0).tolist(),
        "max_abs_joint_pos_error_rad": max_error.tolist(),
        "source_sha256": {"fit": sha(args.bilateral_fit), "diag": sha(args.diag_records),
                          "status": sha(args.status_records)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
