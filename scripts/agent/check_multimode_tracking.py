#!/usr/bin/env python3
"""Offline check of the flat tracking criteria used by assess_multimode_tracking.py.

Loads `episode_verdict` and the tolerance constants from the assessment script itself
(one source of truth) and exercises the arithmetic and verdict logic on synthetic
accumulators: perfect tracking, exactly at each tolerance, just outside each tolerance,
the tilt boundary, base contact, a fall, an early termination, a numerical failure, and
the zero-observation guard.

This does not build an environment, step physics, load a checkpoint or run the
assessment; it only proves the criterion code does what the protocol document says.
It cannot show that a policy tracks anything.

Usage (repo root):
  mjlab_training/.venv/bin/python scripts/agent/check_multimode_tracking.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSESS = REPO_ROOT / "scripts" / "agent" / "assess_multimode_tracking.py"


def load_assessment():
    spec = importlib.util.spec_from_file_location("assess_multimode_tracking", ASSESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_assessment()
    verdict = module.episode_verdict
    vx_tol = module.VX_TOL
    yaw_tol = module.YAW_TOL
    height_tol = module.HEIGHT_TOL
    tilt_tol = module.TILT_TOL_DEG

    base = dict(observed_s=10.0, vx_err_abs_sum=0.0, vx_err_signed_sum=0.0, vy_abs_sum=0.0,
                yaw_err_abs_sum=0.0, yaw_err_signed_sum=0.0, height_err_abs_sum=0.0,
                tilt_max_deg=0.0, base_contact_s=0.0, numerical_failure=False, survived=True)
    cases = []

    def case(name, expected_passed, expected_failed=(), **overrides):
        kwargs = dict(base)
        kwargs.update(overrides)
        result = verdict(**kwargs)
        cases.append({
            "case": name,
            "passed": result["passed"],
            "expected_passed": expected_passed,
            "failed_checks": result["failed_checks"],
            "expected_failed_checks": sorted(expected_failed),
            "ok": (result["passed"] == expected_passed
                   and sorted(result["failed_checks"]) == sorted(expected_failed)),
        })

    case("perfect_tracking", True)
    # Exactly at the tolerance must pass; the comparison is inclusive.
    case("vx_at_tolerance", True, vx_err_abs_sum=vx_tol * 10.0)
    case("yaw_at_tolerance", True, yaw_err_abs_sum=yaw_tol * 10.0)
    case("height_at_tolerance", True, height_err_abs_sum=height_tol * 10.0)
    case("tilt_at_tolerance", True, tilt_max_deg=tilt_tol)
    case("vx_just_over_tolerance", False, ("tracked_vx",),
         vx_err_abs_sum=vx_tol * 10.0 + 1e-6)
    case("yaw_just_over_tolerance", False, ("tracked_yaw",),
         yaw_err_abs_sum=yaw_tol * 10.0 + 1e-6)
    case("height_just_over_tolerance", False, ("tracked_height",),
         height_err_abs_sum=height_tol * 10.0 + 1e-6)
    case("tilt_over_tolerance", False, ("upright",), tilt_max_deg=tilt_tol + 1e-6)
    case("base_contact", False, ("no_base_contact",), base_contact_s=0.02)
    case("fell_over", False, ("survived",), survived=False)
    case("numerical_failure", False, ("numerical_ok",), numerical_failure=True)
    case("all_conditions_failed", False,
         ("tracked_vx", "tracked_yaw", "tracked_height", "upright", "no_base_contact",
          "survived", "numerical_ok"),
         vx_err_abs_sum=99.0, yaw_err_abs_sum=99.0, height_err_abs_sum=99.0,
         tilt_max_deg=99.0, base_contact_s=1.0, survived=False, numerical_failure=True)
    # Bias and lateral speed are reported but must not gate the verdict.
    case("bias_and_lateral_do_not_gate", True, vx_err_signed_sum=-5.0, vy_abs_sum=3.0)

    # Means are time-normalised by the observed window, not by the step count.
    scaled = verdict(**{**base, "vx_err_abs_sum": 2.0, "observed_s": 10.0})
    cases.append({
        "case": "mean_is_per_observed_second",
        "ok": abs(scaled["means"]["vx_error_m_s"] - 0.2) < 1e-12,
        "value": scaled["means"]["vx_error_m_s"],
    })

    guard = None
    try:
        verdict(**{**base, "observed_s": 0.0})
        guard = "no error raised"
    except ValueError as error:
        guard = str(error)
    cases.append({"case": "zero_observation_guard", "ok": isinstance(guard, str)
                  and guard != "no error raised", "message": guard})

    # The documented protocol numbers must match the constants this check enforces.
    documented = {"vx_err_abs_m_s": 0.30, "yaw_err_abs_rad_s": 0.30, "height_err_abs_m": 0.03,
                  "tilt_max_deg": 25.0}
    observed = {"vx_err_abs_m_s": vx_tol, "yaw_err_abs_rad_s": yaw_tol,
                "height_err_abs_m": height_tol, "tilt_max_deg": tilt_tol}
    tolerance_match = observed == documented

    failed = [case for case in cases if not case["ok"]]
    report = {
        "kind": "offline criterion check; no environment, no physics, no checkpoint",
        "assessment": str(ASSESS.relative_to(REPO_ROOT)),
        "cases": cases,
        "tolerances": observed,
        "documented_tolerances": documented,
        "tolerances_match_documentation": tolerance_match,
        "failures": [case["case"] for case in failed] + ([] if tolerance_match else ["tolerance_drift"]),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["failures"]:
        raise SystemExit(f"跟踪判据检查失败：{report['failures']}")
    print("跟踪判据检查通过", flush=True)


if __name__ == "__main__":
    main()
