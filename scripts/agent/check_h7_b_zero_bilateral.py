#!/usr/bin/env python3
"""Crosscheck a supported four-DM zero pose against the one-side CAD fit.

Read-only analysis: never opens a serial device or changes motor calibration.
The left-side result is conditional on both legs having the same mirrored CAD
pose; a second independent left-side dimension is needed to verify it.
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
from scripts.agent.fit_h7_b_zero_from_cad import wheel_xz
from sim2sim.chuanliantui_closed_adapter import ClosedChainAdapter, _PlanarLegGeometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--original-capture", type=Path, required=True)
    parser.add_argument("--supported-status", type=Path, required=True)
    parser.add_argument("--left-mirror-confirmed", action="store_true",
                        help="user confirmed both legs have the same mirrored CAD pose")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    original = json.loads((args.original_capture / "report.json").read_text(encoding="utf-8"))
    initial = np.array([original["channels"]["dm{}_pos_rad".format(i)]["mean"]
                        for i in range(4)], dtype=float)
    samples = [json.loads(line) for line in args.supported_status.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 25:
        raise RuntimeError("supported capture needs at least 25 samples")
    for entry in samples:
        remote, motor = entry["remote"], entry["motor"]
        if (remote["online"] != 1 or remote["left_switch"] != 2
            or remote["right_switch"] != 2 or remote["permit"] != 0
            or remote["armed"] != 0 or remote["enabled_mask"] != 0
            or motor["online"] != 63 or motor["enabled"] != 0
            or any(value != 0.0 for value in motor["target"])):
            raise RuntimeError("capture is not fully disabled with zero targets")
    current = np.asarray([item["motor"]["dm_pos"] for item in samples], dtype=float)
    if current.shape != (len(samples), 4) or not np.all(np.isfinite(current)):
        raise RuntimeError("invalid DM position capture")
    current_mean = current.mean(axis=0)
    q_right = np.asarray(fit["nearest_solution_front_rear_rad"], dtype=float)
    if q_right.shape != (2,) or not np.all(np.isfinite(q_right)):
        raise RuntimeError("invalid CAD fit")
    # H7 model lf = physical right: front/rear positions are negative input
    # angles. Model rf = physical left: same mirrored CAD pose is positive.
    right_odd = initial[3] + q_right[0]
    right_even = initial[2] + np.pi + q_right[1]
    left_odd = initial[1] + q_right[0]
    left_even = initial[0] + np.pi + q_right[1]
    model_path = ROOT / "sim2sim/chuanliantui.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    right = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[0])
    left = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[1])

    def endpoint(geometry, wheel_body, front, rear):
        offset = np.asarray(model.body_pos[model.body(wheel_body).id])[[0, 2]]
        point, knee, _ = wheel_xz(geometry, offset, front, rear)
        return {"front_rear_rad": [float(front), float(rear)],
                "knee_rad": float(knee), "wheel_xz_m": point.tolist()}

    right_now = endpoint(right, "lfwheel", right_odd - current_mean[3],
                         right_even - current_mean[2] - np.pi)
    left_now = endpoint(left, "rfwheel", current_mean[1] - left_odd,
                        current_mean[0] + np.pi - left_even)
    result = {
        "status": ("bilateral CAD zero candidate under user-confirmed mirror pose"
                   if args.left_mirror_confirmed else
                   "right CAD zero candidate; left zero conditional on mirrored same pose"),
        "left_mirror_pose_user_confirmed": args.left_mirror_confirmed,
        "old_capture_dm_pos_zero_rad": initial.tolist(),
        "supported_repeat_samples": len(samples),
        "supported_dm_pos_zero_mean_rad": current_mean.tolist(),
        "supported_dm_pos_zero_std_rad": current.std(axis=0).tolist(),
        "supported_minus_old_rad": (current_mean - initial).tolist(),
        "cad_right_model_input_front_rear_rad": q_right.tolist(),
        "conditional_cad_angle_zero_by_model_side_rad": {
            "lf_physical_right": {"odd_front_dm3": float(right_odd),
                                  "even_rear_dm2": float(right_even)},
            "rf_physical_left": {"odd_front_dm1": float(left_odd),
                                 "even_rear_dm0": float(left_even)},
        },
        "supported_pose_with_conditional_offsets": {
            "physical_right": right_now, "physical_left": left_now,
        },
        "machine_config_dm_zero_unchanged": [0.476998, 1.974491,
                                               0.476998, 1.974491],
        "source_sha256": {"fit": sha(args.fit),
                          "original_report": sha(args.original_capture / "report.json"),
                          "supported_status": sha(args.supported_status),
                          "xml": sha(model_path)},
        "limitations": [
            "The SolidWorks wheel endpoint and arm angles show one side only.",
            "The left offsets use the mirrored pose stated by the user; no independent left-side shaft angle or wheel endpoint was supplied.",
            "This snapshot cannot prove the sign of a moving motor shaft or torque output.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
