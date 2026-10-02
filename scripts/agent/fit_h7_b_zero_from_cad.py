#!/usr/bin/env python3
"""Fit a *candidate* B CAD-angle offset to a matching SolidWorks zero pose.

This is a one-pose geometric fit, not a physical direction/Jacobian validation.
It never opens serial/CAN, flashes firmware or changes configured motor zeros.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2sim.chuanliantui_closed_adapter import (ClosedChainAdapter,
                                                    _PlanarLegGeometry, _rotate)

OLD_ODD_ZERO = 0.6675945815
OLD_EVEN_ZERO = 2.4768720991


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wheel_xz(geometry, wheel_offset, front, rear):
    knee, jac = geometry.knee_and_jacobian(front, rear)
    point = (_rotate(geometry.sign * front, geometry.front1_offset)
             + _rotate(geometry.sign * (front + knee), wheel_offset))
    return point, knee, jac


def image_angle_check(geometry, fitted, hip, front_tip, rear_tip, pixel_error):
    """Read CAD input-arm angles from visible shaft/pin centers in the image.

    The screenshot looks along the model's y axis: image right is model -x,
    image down is model -z. Each selected center may vary independently by
    `pixel_error` in both image coordinates.
    """
    if pixel_error < 0:
        raise ValueError("pixel_error must be nonnegative")
    centers = (np.asarray(hip, dtype=float), np.asarray(front_tip, dtype=float),
               np.asarray(rear_tip, dtype=float))
    if any(center.shape != (2,) or not np.all(np.isfinite(center)) for center in centers):
        raise ValueError("image centers must be finite x/y pixel pairs")

    def angle_at(shaft, tip, base):
        delta = tip - shaft
        if np.linalg.norm(delta) < 1:
            raise ValueError("image arm must span at least one pixel")
        cad_vector = np.array((-delta[0], -delta[1]))
        return float(wrap(geometry.sign * (np.arctan2(cad_vector[1], cad_vector[0])
                                            - np.arctan2(base[1], base[0]))))

    arms = {}
    for name, tip, base, fitted_q in (("front_long", centers[1], geometry.front1_offset, fitted[0]),
                                      ("rear_short", centers[2], geometry.rear1_offset, fitted[1])):
        measured = angle_at(centers[0], tip, base)
        deviations = []
        for hx in (-pixel_error, pixel_error):
            for hy in (-pixel_error, pixel_error):
                for tx in (-pixel_error, pixel_error):
                    for ty in (-pixel_error, pixel_error):
                        shifted = angle_at(centers[0] + (hx, hy), tip + (tx, ty), base)
                        deviations.append(float(wrap(shifted - measured)))
        difference = float(wrap(fitted_q - measured))
        arms[name] = {
            "tip_pixel": tip.tolist(),
            "shaft_to_tip_length_pixel": float(np.linalg.norm(tip - centers[0])),
            "image_derived_model_input_rad": measured,
            "image_derived_model_input_deg": float(np.rad2deg(measured)),
            "fitted_minus_image_deg": float(np.rad2deg(difference)),
            "corner_perturbation_deg": np.rad2deg([min(deviations), max(deviations)]).tolist(),
            "fit_inside_pixel_error_bounds": min(deviations) <= difference <= max(deviations),
        }
    return {"hip_shaft_pixel": centers[0].tolist(), "center_selection_error_pixel": pixel_error,
            "coordinate_convention": "image right = model -x, image down = model -z",
            "status": "independent approximate screenshot angle check, not a physical shaft measurement",
            "arms": arms}


def fit(model, pos, horizontal_mm, vertical_mm):
    geometry = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[0])  # physical right = model lf
    wheel_offset = np.asarray(model.body_pos[model.body("lfwheel").id])[[0, 2]]
    target = np.array((-horizontal_mm, -vertical_mm), dtype=float) / 1000.0
    initial = np.array((-wrap(pos[3] - OLD_ODD_ZERO),
                        -wrap(pos[2] + np.pi - OLD_EVEN_ZERO)))

    def residual(q):
        try:
            point, _, _ = wheel_xz(geometry, wheel_offset, q[0], q[1])
            return point - target
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.array((10.0, 10.0))

    solutions = []
    for front in np.linspace(-2.2, -0.3, 12):
        for rear in np.linspace(-2.2, -0.3, 12):
            result = least_squares(residual, (front, rear), max_nfev=80)
            if np.linalg.norm(result.fun) < 1.0e-6 and not any(
                    np.linalg.norm(result.x - entry) < 1.0e-2 for entry in solutions):
                solutions.append(result.x)
    if not solutions:
        raise RuntimeError("SolidWorks endpoint has no reachable B solution")
    solutions.sort(key=lambda q: np.linalg.norm(q - initial))
    solved = solutions[0]
    predicted, knee, jac = wheel_xz(geometry, wheel_offset, *solved)
    before = residual(initial) + target
    odd_zero = float(pos[3] + solved[0])
    even_zero = float(pos[2] + np.pi + solved[1])
    result = {
        "cad_target_xz_m": target.tolist(),
        "current_b_front_rear_rad": initial.tolist(),
        "current_b_wheel_xz_m": before.tolist(),
        "current_b_minus_cad_mm": ((before - target) * 1000.0).tolist(),
        "solutions_found_in_grid": len(solutions),
        "nearest_solution_front_rear_rad": solved.tolist(),
        "nearest_solution_knee_rad": float(knee),
        "nearest_solution_jacobian": np.asarray(jac).tolist(),
        "nearest_solution_wheel_xz_m": predicted.tolist(),
        "front_rear_correction_deg": np.rad2deg(solved - initial).tolist(),
        "provisional_odd_slot_cad_zero_rad": odd_zero,
        "provisional_even_slot_cad_zero_rad": even_zero,
        "original_odd_even_slot_cad_zero_rad": [OLD_ODD_ZERO, OLD_EVEN_ZERO],
    }
    # Shared CAD offsets imply a prediction for the other physical side, but
    # the right-side SolidWorks dimension does not independently verify it.
    other = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[1])
    other_offset = np.asarray(model.body_pos[model.body("rfwheel").id])[[0, 2]]
    other_front = wrap(pos[1] - odd_zero)
    other_rear = wrap(pos[0] + np.pi - even_zero)
    other_point, other_knee, _ = wheel_xz(other, other_offset, other_front, other_rear)
    result["left_side_unverified_prediction"] = {
        "front_rear_rad": [float(other_front), float(other_rear)],
        "knee_rad": float(other_knee), "wheel_xz_m": other_point.tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizontal-mm", type=float, default=319.93)
    parser.add_argument("--vertical-mm", type=float, default=107.49)
    parser.add_argument("--hip-pixel", nargs=2, type=float)
    parser.add_argument("--front-tip-pixel", nargs=2, type=float)
    parser.add_argument("--rear-tip-pixel", nargs=2, type=float)
    parser.add_argument("--center-error-pixel", type=float, default=3.0)
    args = parser.parse_args()
    provided = (args.hip_pixel, args.front_tip_pixel, args.rear_tip_pixel)
    if any(value is not None for value in provided) and not all(
            value is not None for value in provided):
        parser.error("provide all three image center pairs together")
    report = json.loads((args.capture / "report.json").read_text(encoding="utf-8"))
    pos = np.array([report["channels"]["dm{}_pos_rad".format(i)]["mean"]
                    for i in range(4)], dtype=float)
    model_path = ROOT / "sim2sim/chuanliantui.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    outcome = fit(model, pos, args.horizontal_mm, args.vertical_mm)
    if all(value is not None for value in provided):
        geometry = _PlanarLegGeometry(model, ClosedChainAdapter._SIDES[0])
        outcome["solidworks_image_angle_check"] = image_angle_check(
            geometry, outcome["nearest_solution_front_rear_rad"],
            args.hip_pixel, args.front_tip_pixel, args.rear_tip_pixel,
            args.center_error_pixel)
    outcome.update({"scope": "single-pose SolidWorks endpoint fit; no physical sign or torque approval",
                    "source_pose_user_confirmed_same_as_disabled_capture": True,
                    "measurement_origin_user_confirmed_hip_motor_axis": True,
                    "solidworks_horizontal_mm": args.horizontal_mm,
                    "solidworks_vertical_mm": args.vertical_mm,
                    "solidworks_image_sha256": sha(args.image),
                    "raw_capture_sha256": report["raw_sha256"],
                    "cad_xml_sha256": sha(model_path),
                    "h7_observation_source_sha256": sha(args.capture.parents[1] / "App/lower_observation.c"),
                    "dm_pos_zero_rad": pos.tolist(),
                    "interpretation": "Candidate offsets only. B shaft identity and per-slot dm_sign/dm_zero are unchanged; require a second pose or shaft-angle measurement and all nonzero bench gates."})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(outcome, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
