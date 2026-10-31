#!/usr/bin/env python3
"""Read standing-task TensorBoard/assessment artifacts and write an offline report.

No simulator, policy inference, training, hardware or source modification is used.
Different checkpoint hashes stay in separate groups. A previously passing case
that fails or is absent from a candidate is a regression, even when its average
success rate improves.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

MOTOR_NAMES = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
LEG_INDICES = (0, 1, 3, 4)
ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SOURCE_SUFFIXES = (
    "scripts/agent/assess_chuanliantui_stand.py",
    "mjlab_training/src/ct_mjlab/envs/chuanliantui_stand/mdp.py",
    "mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_mdp.py",
    "mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_mdp.py",
)


def identity(path):
    path = Path(path).resolve()
    result = {"path": str(path), "exists": path.is_file()}
    if result["exists"]:
        data = path.read_bytes()
        result.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return result


def finite_number(value):
    return isinstance(value, (float, int, np.number)) and np.isfinite(value)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def statistics(values):
    values = np.asarray([value for value in values if finite_number(value)], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "maximum": float(values.max())}


def load_training(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util
    import yaml

    paths = sorted(run.glob("events.out.tfevents.*"))
    scalars, duplicate_count = defaultdict(dict), 0
    for path in paths:
        events = EventAccumulator(str(path), size_guidance={"scalars": 0, "tensors": 0},
                                  purge_orphaned_data=False)
        events.Reload()
        for kind in ("scalars", "tensors"):
            for tag in events.Tags().get(kind, []):
                points = events.Scalars(tag) if kind == "scalars" else events.Tensors(tag)
                for point in points:
                    array = np.asarray(point.value if kind == "scalars" else
                                       tensor_util.make_ndarray(point.tensor_proto))
                    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
                        continue
                    record = {"step": int(point.step), "wall_time": float(point.wall_time),
                              "value": float(array.reshape(-1)[0])}
                    previous = scalars[tag].get(record["step"])
                    if previous is not None:
                        duplicate_count += 1
                    # A resumed writer may repeat a step; retain the most recent observation.
                    if previous is None or record["wall_time"] >= previous["wall_time"]:
                        scalars[tag][record["step"]] = record
    series = {tag: sorted(points.values(), key=lambda row: row["step"])
              for tag, points in sorted(scalars.items())}
    recipe = [identity(path) for path in sorted((run / "params").glob("*.yaml"))]
    policy_dt = None
    env_path = run / "params/env.yaml"
    if env_path.is_file():
        # BaseLoader reads saved Python-tagged YAML as data, never imports its callables.
        env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
        policy_dt = float(env["decimation"]) * float(env["sim"]["mujoco"]["timestep"])
    metadata = {"run": str(run), "event_files": [identity(path) for path in paths],
                "recipe_files": recipe, "policy_dt_s": policy_dt,
                "scalar_tags": list(series), "duplicate_tag_steps": duplicate_count,
                "duplicate_resolution": "latest wall_time for each tag/step",
                "metrics": {}}
    for tag, rows in series.items():
        good = [row for row in rows if np.isfinite(row["value"])]
        metadata["metrics"][tag] = {
            "points": len(rows), "nonfinite_points": len(rows) - len(good),
            "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
            "last_value": rows[-1]["value"],
            "tail_100_mean": float(np.mean([row["value"] for row in good[-100:]])) if good else None,
            **statistics([row["value"] for row in rows])}
    return series, metadata


def case_key(row):
    """Semantic case identity survives hold-only vs all-suite case_id numbering."""
    return (int(row["seed"]), row["profile"], row["kind"], row.get("direction"),
            round(float(row.get("magnitude", 0.)), 12), int(row["replicate"]))


def load_assessments(paths):
    groups, sources = {}, []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("task") != "chuanliantui_stand" or not isinstance(report.get("episodes"), list):
            raise ValueError(f"Not a chuanliantui_stand assessment: {path}")
        checkpoint = report["checkpoint"]
        digest = checkpoint["sha256"]
        actual = identity(checkpoint["path"])
        if actual["exists"] and actual["sha256"] != digest:
            raise ValueError(f"Assessment checkpoint has changed: {checkpoint['path']}")
        group = groups.setdefault(digest, {"checkpoint": checkpoint, "checkpoint_now": actual,
                                          "reports": [], "rows": {}})
        source = {**identity(path), "expected_cases": report["expected_cases"],
                  "completed_cases": report["completed_cases"], "reported_passed": report["passed"],
                  "suite": report["suite"], "profiles": report["profiles"], "seeds": report["seeds"],
                  "physics_dt_s": report["physics_dt_s"], "policy_dt_s": report["policy_dt_s"],
                  "criteria": report["criteria"], "xml": report["xml"],
                  "sources": report.get("sources", [])}
        if report["completed_cases"] != len(report["episodes"]):
            raise ValueError(f"Reported completed_cases differs from episode rows: {path}")
        group["reports"].append(source)
        sources.append(source)
        for row in report["episodes"]:
            key = case_key(row)
            if key in group["rows"]:
                raise ValueError(f"Duplicate checkpoint/case identity {digest[:12]} {key} in {path}")
            group["rows"][key] = {**row, "assessment_file": str(path.resolve()),
                                   "checkpoint_sha256": digest,
                                   "physics_dt_s": report["physics_dt_s"]}
    return groups, sources


class TrajectoryReader:
    """Keep only one compressed batch's arrays in memory at a time."""
    def __init__(self):
        self.path, self.data, self.error = None, None, None

    def metrics(self, row):
        path = Path(row["trajectory"])
        if not path.is_absolute() and not path.is_file():
            path = ROOT / path
        if path != self.path:
            self.path, self.data, self.error = path, None, None
            try:
                with np.load(path, allow_pickle=False) as archive:
                    self.data = {name: archive[name] for name in (
                        "physics", "physics_time_s", "physics_channels", "qpos",
                        "start_index", "end_index", "anchor_xy")}
            except (OSError, ValueError, KeyError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
        if self.data is None:
            return {"trajectory_error": self.error}
        data, env = self.data, row["env"]
        channels = {str(name): i for i, name in enumerate(data["physics_channels"])}
        raw = data["physics"][:, env]
        end_time = row.get("ended_s")
        active = data["physics_time_s"] <= end_time + 1e-8 if end_time is not None else np.ones(len(raw), bool)
        rollout = raw[active]
        start, end = int(data["start_index"][env]), int(data["end_index"][env])
        window = raw[start:end] if start >= 0 else raw[:0]
        dt = float(row["physics_dt_s"])
        result = {"trajectory_error": None,
                  "initial_qpos_sha256": hashlib.sha256(data["qpos"][0, env].tobytes()).hexdigest(),
                  "derived_xy_speed_scope": "world XY displacement / physics_dt; raw 500 Hz positions"}
        for label, samples in (("window", window), ("rollout", rollout)):
            good = np.isfinite(samples).all(-1)
            if not good.any():
                continue
            xy = samples[:, [channels["x"], channels["y"]]]
            # Full-rollout diagnostics start at the first physical sample, not a new acceptance anchor.
            anchor = data["anchor_xy"][env] if label == "window" else xy[0]
            velocity = np.diff(np.vstack((anchor, xy)), axis=0) / dt
            # Never bridge a missing/nonfinite tick with an artificial one-tick derivative.
            adjacent_good = good & np.r_[np.isfinite(anchor).all(), good[:-1]]
            speed = np.linalg.norm(velocity[adjacent_good], axis=-1)
            q = samples[good][:, [channels[f"q_{MOTOR_NAMES[i]}"] for i in LEG_INDICES]]
            result.update({f"{label}_xy_speed_rms_m_s": float(np.sqrt(np.mean(speed ** 2))) if len(speed) else None,
                           f"{label}_xy_speed_peak_m_s": float(speed.max()) if len(speed) else None,
                           f"{label}_xy_speed_mean_m_s": float(speed.mean()) if len(speed) else None,
                           f"{label}_diagnostic_displacement_m": float(np.linalg.norm(xy[good]-anchor, axis=-1).max()),
                           f"{label}_diagnostic_leg_peak_to_peak_deg": np.rad2deg(np.ptp(q, axis=0)).tolist()})
        return result


def failure_reasons(row):
    if row["passed"]:
        return []
    reasons = [row["failed_reason"]] if row.get("failed_reason") else []
    w = row["window"]
    if w["duration_s"] < 20. - 1e-6:
        reasons.append("incomplete_20s_window")
    speed_v2 = "maximum_body_speed_m_s" in w
    first = ("maximum_body_speed_m_s", .05, "body_speed_over_0_05_m_s") if speed_v2 else ("maximum_displacement_m", .05, "drift_over_5cm")
    for key, limit, label in (first,
                              ("maximum_tilt_deg", 25., "tilt_over_25deg"),
                              ("maximum_height_error_m", .03, "height_error_over_3cm")):
        if finite_number(w.get(key)) and w[key] > limit + 1e-7:
            reasons.append(label)
    legs = [value for value in w.get("leg_peak_to_peak_rad", []) if finite_number(value)]
    if speed_v2:
        speeds = [value for value in w["leg_max_abs_speed_rad_s"] if finite_number(value)]
        if speeds and max(speeds) > .5 + 1e-7:
            reasons.append("physical_leg_speed_over_0_5_rad_s")
    elif legs and max(legs) > math.radians(2.) + 1e-7:
        reasons.append("physical_leg_motion_over_2deg")
    if w["duration_s"] > 0 and w["wheel_support_fraction"] < .95 - 1e-8:
        reasons.append("wheel_support_below_95pct")
    if w["duration_s"] > 0 and w["base_contact_fraction"] > .01 + 1e-8:
        reasons.append("base_contact_over_1pct")
    if w["duration_s"] > 0 and not w["finite"]:
        reasons.append("nonfinite_measured_window")
    return list(dict.fromkeys(reasons or ["assessment_failed"]))


def flatten_row(row, diagnostics):
    w = row["window"]
    result = {name: row.get(name) for name in (
        "checkpoint_sha256", "assessment_file", "profile", "seed", "case_id", "kind",
        "magnitude", "direction", "replicate", "batch", "env", "phase", "passed",
        "delivered", "delivered_s", "fell", "recovered", "anchor_s", "ended_s",
        "failed_reason", "resets_after_result_frozen", "runtime_error", "trajectory")}
    result["failure_reasons"] = ";".join(failure_reasons(row))
    # The vector is the actual impulse; direction/magnitude labels alone are insufficient.
    vector = row.get("vector")
    result["applied_vector_x"] = vector[0] if vector is not None and len(vector) == 2 else None
    result["applied_vector_y"] = vector[1] if vector is not None and len(vector) == 2 else None
    for key in ("duration_s", "maximum_displacement_m", "maximum_tilt_deg", "maximum_height_error_m",
                "wheel_support_fraction", "base_contact_fraction", "finite"):
        result[f"window_{key}"] = w[key]
    if "maximum_body_speed_m_s" in w:
        result["window_maximum_body_speed_m_s"] = w["maximum_body_speed_m_s"]
        for motor, value in zip((MOTOR_NAMES[i] for i in LEG_INDICES), w["leg_max_abs_speed_rad_s"], strict=True):
            result[f"{motor}_max_abs_speed_rad_s"] = value
    for name, values, motors, scale in (
        ("peak_to_peak_deg", w["leg_peak_to_peak_rad"], [MOTOR_NAMES[i] for i in LEG_INDICES], 180/math.pi),
        ("qd_rms_rad_s", w["qd_rms_rad_s"], MOTOR_NAMES, 1.),
        ("dominant_frequency_hz", w["leg_dominant_frequency_hz"], [MOTOR_NAMES[i] for i in LEG_INDICES], 1.)):
        for motor, value in zip(motors, values, strict=True):
            result[f"{motor}_{name}"] = value * scale if finite_number(value) else None
    angles = [value for value in w["leg_peak_to_peak_rad"] if finite_number(value)]
    result["maximum_leg_peak_to_peak_deg"] = math.degrees(max(angles)) if angles else None
    if row.get("anchor_s") is not None and row.get("delivered_s") is not None:
        result["settled_after_push_s"] = row["anchor_s"] - row["delivered_s"]
    else:
        result["settled_after_push_s"] = None
    result["actual_recovery_time_s"] = result["settled_after_push_s"] if row["fell"] and row["recovered"] else None
    for key, value in diagnostics.items():
        if isinstance(value, list):
            for side, item in zip(("lf0", "lf00", "rf0", "rf00"), value, strict=True):
                result[f"{key}_{side}"] = item
        else:
            result[key] = value
    rollout_legs = diagnostics.get("rollout_diagnostic_leg_peak_to_peak_deg", [])
    result["rollout_maximum_leg_peak_to_peak_deg"] = max(rollout_legs) if rollout_legs else None
    for side, index in (("left", 0), ("right", 1)):
        for key in ("hip_wheel_angle_peak_to_peak_deg", "hip_wheel_length_min_m", "hip_wheel_length_max_m"):
            values = row.get("geometry", {}).get(key)
            result[f"{side}_{key}"] = values[index] if values is not None else None
    return result


def aggregate(rows):
    reasons = Counter(reason for row in rows for reason in row["failure_reasons"].split(";") if reason)
    delivered = sum(bool(row["delivered"]) for row in rows)
    fallen = sum(bool(row["fell"]) for row in rows)
    recovered = sum(bool(row["fell"] and row["recovered"]) for row in rows)
    metrics = ("window_maximum_body_speed_m_s", "lf0_max_abs_speed_rad_s", "lf00_max_abs_speed_rad_s",
               "rf0_max_abs_speed_rad_s", "rf00_max_abs_speed_rad_s", "maximum_leg_peak_to_peak_deg", "window_maximum_displacement_m",
               "window_xy_speed_rms_m_s", "window_xy_speed_peak_m_s", "actual_recovery_time_s",
               "settled_after_push_s", "window_duration_s", "rollout_xy_speed_rms_m_s",
               "rollout_xy_speed_peak_m_s", "rollout_diagnostic_displacement_m",
               "rollout_maximum_leg_peak_to_peak_deg")
    return {"cases": len(rows), "passed": sum(bool(row["passed"]) for row in rows),
            "pass_fraction": sum(bool(row["passed"]) for row in rows) / len(rows) if rows else None,
            "failure_reasons": dict(reasons.most_common()), "delivered": delivered, "fell": fallen,
            "recovered_after_actual_fall": recovered,
            "fall_recoveries_then_passed_20s": sum(bool(row["fell"] and row["recovered"] and row["passed"]) for row in rows),
            "recovery_fraction_among_actual_falls": recovered / fallen if fallen else None,
            "metrics": {metric: statistics([row.get(metric) for row in rows]) for metric in metrics},
            "per_leg_peak_to_peak_deg": {motor: statistics([row.get(f"{motor}_peak_to_peak_deg") for row in rows])
                                         for motor in (MOTOR_NAMES[i] for i in LEG_INDICES)},
            "rollout_per_leg_peak_to_peak_deg": {
                motor: statistics([row.get(f"rollout_diagnostic_leg_peak_to_peak_deg_{motor}") for row in rows])
                for motor in (MOTOR_NAMES[i] for i in LEG_INDICES)},
            "per_motor_qd_rms_rad_s": {motor: statistics([row.get(f"{motor}_qd_rms_rad_s") for row in rows])
                                     for motor in MOTOR_NAMES}}


def compare_baseline(candidate, baseline):
    """Missing prior successes cannot disappear into a denominator change."""
    matched, lost, missing_successes, improved = [], [], [], []
    for key, before in baseline.items():
        after = candidate.get(key)
        if after is None:
            if before["passed"]:
                missing_successes.append(list(key))
            continue
        result = {"case_key": list(key), "baseline_passed": bool(before["passed"]),
                  "candidate_passed": bool(after["passed"])}
        for metric in ("maximum_leg_peak_to_peak_deg", "window_maximum_displacement_m",
                       "window_xy_speed_rms_m_s", "actual_recovery_time_s"):
            a, b = before.get(metric), after.get(metric)
            result[f"delta_{metric}"] = b - a if finite_number(a) and finite_number(b) else None
        a, b = before.get("initial_qpos_sha256"), after.get("initial_qpos_sha256")
        result["initial_qpos_equal"] = a == b if a is not None and b is not None else None
        before_vector = [before.get(f"applied_vector_{axis}") for axis in ("x", "y")]
        after_vector = [after.get(f"applied_vector_{axis}") for axis in ("x", "y")]
        result["applied_vector_equal"] = (before_vector == after_vector
            if all(finite_number(value) for value in before_vector + after_vector) else None)
        result["baseline_applied_vector"] = before_vector
        result["candidate_applied_vector"] = after_vector
        matched.append(result)
        if before["passed"] and not after["passed"]:
            lost.append(list(key))
        elif not before["passed"] and after["passed"]:
            improved.append(list(key))
    return {"regressed": bool(lost or missing_successes), "matched_cases": len(matched),
            "lost_previous_successes": lost, "missing_previous_successes": missing_successes,
            "improved_cases": improved, "additional_candidate_cases": len(set(candidate)-set(baseline)),
            "case_key_fields": ["seed", "profile", "kind", "direction", "magnitude", "replicate"],
            "paired_initial_state_mismatches": sum(row["initial_qpos_equal"] is False for row in matched),
            "paired_initial_state_unavailable": sum(row["initial_qpos_equal"] is None for row in matched),
            "applied_vector_mismatches": sum(row["applied_vector_equal"] is False for row in matched),
            "applied_vector_unavailable": sum(row["applied_vector_equal"] is None for row in matched),
            "matched": matched}


def protocol_sources(report):
    sources = report.get("sources", [])
    result = {}
    for suffix in PROTOCOL_SOURCE_SUFFIXES:
        aliases = (suffix, suffix.replace('scripts/agent/', 'scripts/agent/stand/'))
        matches = {source['sha256'] for source in sources
                   if any(source['path'].replace('\\', '/').endswith('/' + name) for name in aliases)}
        # 路径迁移不改变来源角色；哈希不同或多个冲突来源仍不能判为兼容。
        result[suffix] = next(iter(matches)) if len(matches) == 1 else None
    return result


def protocol_signature(report):
    fields = {"criteria": report["criteria"], "xml_sha256": report["xml"]["sha256"],
              "physics_dt_s": report["physics_dt_s"], "policy_dt_s": report["policy_dt_s"],
              "critical_sources": protocol_sources(report)}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["no_samples"])
        writer.writeheader()
        writer.writerows(clean_json(row) for row in rows)


def plot_training(series, metadata, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, tags, ylabel, scale=1.):
        for tag in tags:
            data = series[tag]
            x, y = np.asarray([r["step"] for r in data]), np.asarray([r["value"] for r in data]) * scale
            ax.plot(x, y, linewidth=1.1, alpha=.85, label=tag.split("/", 1)[-1])
        ax.set(xlabel="Training iteration", ylabel=ylabel)
        ax.grid(alpha=.2)
        if tags:
            ax.legend(fontsize=8, ncol=2, loc="best")
        else:
            ax.text(.5, .5, "No matching scalar events yet", ha="center", va="center", transform=ax.transAxes)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    draw(axes[0], [tag for tag in ("Train/mean_reward",) if tag in series], "Mean episode reward")
    preferred = ("stand_low_velocity", "stand_quiet_legs", "track_height_exp_tight", "stand_still_lin_vel",
                 "leg_joint_vel", "termination")
    reward_tags = [f"Episode_Reward/{name}" for name in preferred if f"Episode_Reward/{name}" in series]
    if not reward_tags:
        reward_tags = [tag for tag in series if tag.startswith("Episode_Reward/")][:6]
    draw(axes[1], reward_tags, "Logged reward components")
    fig.suptitle(Path(metadata["run"]).name)
    fig.savefig(output / "reward_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    dt = metadata["policy_dt_s"]
    draw(axes[0], [tag for tag in ("Train/mean_episode_length",) if tag in series],
         "Mean episode duration (s)" if dt is not None else "Mean episode length (policy steps)",
         scale=dt if dt is not None else 1.)
    tags = [tag for tag in series if tag in (
        "Episode_Metrics/strict_standing_fraction", "Episode_Metrics/recovery_timeout",
        "Episode_Metrics/disturbance_applied", "Metrics/standup/strict_standing_fraction",
        "Metrics/standup/recovery_timeout", "Metrics/standup/disturbance_applied",
        "Episode_Termination/failure", "Episode_Termination/time_out")]
    draw(axes[1], tags, "Logged episode metric")
    fig.suptitle(Path(metadata["run"]).name)
    fig.savefig(output / "episode_curves.png", dpi=160)
    plt.close(fig)


def plot_representative_hold(group, digest, output):
    """Render every recorded physics sample; a failed rollout is not a hold window."""
    import matplotlib
    import matplotlib.pyplot as plt

    holds = [row for row in group["rows"].values() if row["kind"] == "hold"]
    if not holds:
        return {"file": None, "reason": "No hold cases in the supplied assessment files"}
    row = next((sample for sample in holds if not sample["passed"]), holds[0])
    reader = TrajectoryReader()
    diagnostics = reader.metrics(row)
    result = {"case_key": list(case_key(row)), "assessment_file": row["assessment_file"],
              "trajectory": row["trajectory"], "passed": row["passed"],
              "selection": "first failed hold, otherwise first hold", "file": None}
    if reader.data is None:
        return {**result, "reason": diagnostics["trajectory_error"]}
    archive, env = reader.data, row["env"]
    channels = {str(name): i for i, name in enumerate(archive["physics_channels"])}
    times = archive["physics_time_s"]
    end_s = float(row["ended_s"]) if row.get("ended_s") is not None else float(times[-1])
    keep = (times >= 0.) & (times <= end_s + 1e-8)
    times, raw = times[keep], archive["physics"][keep, env]
    if not len(times):
        return {**result, "reason": "No physical samples before the sample's end time"}
    anchor_s = row.get("anchor_s")
    anchor_exists = anchor_s is not None and int(archive["start_index"][env]) >= 0
    xy = raw[:, [channels["x"], channels["y"]]]
    reference = archive["anchor_xy"][env] if anchor_exists else xy[0]
    displacement = np.linalg.norm(xy - reference, axis=-1)
    intervals = np.diff(times)
    if not np.all(intervals > 0.):
        raise ValueError(f"Physical trajectory timestamps are not strictly increasing: {row['trajectory']}")
    speed_v2 = "vx_world_m_s" in channels and "vy_world_m_s" in channels
    speed = (np.linalg.norm(raw[:, [channels["vx_world_m_s"], channels["vy_world_m_s"]]], axis=-1) if speed_v2
             else np.linalg.norm(np.diff(xy, axis=0) / intervals[:, None], axis=-1))
    names = [MOTOR_NAMES[index] for index in LEG_INDICES]
    q = raw[:, [channels[f"q_{motor}"] for motor in names]]
    leg_delta = np.rad2deg(q - q[0])
    leg_ranges = np.rad2deg(np.nanmax(q, axis=0) - np.nanmin(q, axis=0))
    height = raw[:, channels["height_m"]]
    tilt = raw[:, [channels["roll_deg"], channels["pitch_deg"]]]
    total_tilt = np.rad2deg(np.arccos(np.clip(
        np.cos(np.deg2rad(tilt[:, 0].astype(np.float64)))
        * np.cos(np.deg2rad(tilt[:, 1].astype(np.float64))), -1., 1.)))
    filename = output / f"trajectory_{digest[:12]}.png"
    scope = ("Full rollout diagnosis; shaded segment starts at the recorded acceptance anchor"
             if anchor_exists else
             f"DIAGNOSTIC ONLY: no acceptance anchor; XY reference is the first recorded sample at {times[0]:.3f} s")
    # Disable Matplotlib's vertex simplification too: short 2 ms peaks remain in the plotted path.
    with matplotlib.rc_context({"path.simplify": False, "agg.path.chunksize": 0}):
        fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True, constrained_layout=True)
        axes[0].axhspan(.24, .30, color="#46a36b", alpha=.15, label="0.27 +/- 0.03 m")
        axes[0].axhline(.27, color="#277544", linestyle=":", linewidth=1.)
        axes[0].plot(times, height, color="#245fba", linewidth=1.1, label="Body height")
        axes[0].set_ylabel("Height (m)")
        axes[1].axhspan(0., 25., color="#46a36b", alpha=.12, label="Body-Z tilt <=25 deg")
        axes[1].plot(times, total_tilt, linewidth=1.2, label="Body-Z angle to vertical")
        axes[1].plot(times, tilt[:, 0], linewidth=.7, linestyle="--", alpha=.6, label="Raw roll")
        axes[1].plot(times, tilt[:, 1], linewidth=.7, linestyle="--", alpha=.6, label="Raw pitch")
        axes[1].set_ylabel("Tilt (deg)")
        axes[2].plot(times, displacement, color="#245fba", linewidth=1., label=(
            "Distance from acceptance anchor" if anchor_exists else "Distance from initial recorded XY"))
        if anchor_exists and not speed_v2:
            axes[2].hlines(.05, anchor_s, end_s, color="#b84435", linestyle="--", label="Window limit: 0.05 m")
        axes[2].set_ylabel("XY displacement (m)")
        axes[3].plot(times if speed_v2 else times[1:], speed, color="#8250a4", linewidth=1.,
                     label="Raw500Hz world XY speed" if speed_v2 else "Derived diagnostic XY speed")
        if speed_v2 and anchor_exists:
            axes[3].hlines(.05, anchor_s, end_s, color="#b84435", linestyle="--", label="Window speed limit0.05m/s")
        axes[3].set_ylabel("XY speed (m/s)")
        for index, motor in enumerate(names):
            axes[4].plot(times, leg_delta[:, index], linewidth=.9,
                         label=f"{motor}: full-rollout p-p {leg_ranges[index]:.2f} deg")
        axes[4].set_ylabel("Leg angle - first sample (deg)")
        for motor in names:
            axes[5].plot(times, raw[:, channels[f"qd_{motor}"]], linewidth=.8, label=motor)
        axes[5].set_ylabel("Leg speed (rad/s)")
        if speed_v2 and anchor_exists:
            for limit in (-.5, .5):
                axes[5].hlines(limit, anchor_s, end_s, color="#b84435", linestyle="--")
        axes[6].step(times, raw[:, channels["wheel_supported"]], where="post", color="#277544",
                     linewidth=1.2, label="Both wheels support (>5 N each, >=70% weight combined)")
        axes[6].step(times, raw[:, channels["base_contact"]], where="post", color="#b84435",
                     linewidth=1., label="Body contact (>1 N)")
        axes[6].set(ylim=(-.1, 1.15), yticks=(0, 1), yticklabels=("No", "Yes"), ylabel="Physical contact", xlabel="Time from reset (s)")
        for index, ax in enumerate(axes):
            ax.grid(alpha=.2)
            ax.set_xlim(0., max(end_s, float(times[-1])))
            if row.get("delivered_s") is not None:
                ax.axvline(row["delivered_s"], color="#d48123", linestyle="--", linewidth=1.1,
                           label="Disturbance delivered" if index == 0 else None)
            if anchor_exists:
                ax.axvline(anchor_s, color="#222222", linestyle="--", linewidth=1.1,
                           label="Acceptance anchor" if index == 0 else None)
                ax.axvspan(anchor_s, min(anchor_s + 20., end_s), color="#46a36b", alpha=.06)
            ax.legend(fontsize=8, loc="best", ncol=2)
        model_name = Path(group["checkpoint"]["path"]).name
        verdict = "PASSED" if row["passed"] else f"FAILED: {row.get('failed_reason') or 'window criterion'}"
        fig.suptitle(f"{model_name} | {row['profile']} | seed {row['seed']} | hold {row['replicate']} | {verdict}\n"
                     f"{scope}\nAll {len(times):,} raw physics samples retained; leg p-p here covers the complete rollout", fontsize=11)
        fig.savefig(filename, dpi=160)
        plt.close(fig)
    return {**result, "file": str(filename), "scope": scope, "samples": len(times),
            "first_physics_sample_s": float(times[0]), "end_s": end_s,
            "median_sample_interval_s": float(np.median(intervals)) if len(intervals) else None,
            "downsampling": False, "matplotlib_path_simplification": False,
            "anchor_s": anchor_s, "delivered_s": row.get("delivered_s"),
            "xy_reference_m": reference.tolist(), "full_rollout_leg_peak_to_peak_deg": leg_ranges.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="One training run, containing TensorBoard events and params")
    parser.add_argument("--assessment", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--baseline", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, help="Record this checkpoint; otherwise record the latest model_N.pt in the run")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not run.is_dir():
        parser.error("--run must be an existing training run directory")
    if output.exists():
        parser.error("--output must be a new directory")
    assessment_paths = [path.resolve() for group in args.assessment for path in group]
    baseline_paths = [path.resolve() for group in args.baseline for path in group]
    if baseline_paths and not assessment_paths:
        parser.error("--baseline requires at least one --assessment")
    groups, assessments = load_assessments(assessment_paths)
    baselines, baseline_sources = load_assessments(baseline_paths)
    if len(baselines) > 1:
        parser.error("Baseline reports must describe one checkpoint; separate its seed/profile files in one --baseline")
    series, training = load_training(run)
    checkpoint = args.checkpoint
    selection = "explicit" if checkpoint is not None else "latest_numeric_filename"
    if checkpoint is None:
        candidates = [path for path in run.glob("model_*.pt") if path.stem[6:].isdigit()]
        checkpoint = max(candidates, key=lambda path: int(path.stem[6:]), default=None)
    if checkpoint is not None and not checkpoint.is_file():
        parser.error("--checkpoint must name an existing model")
    selected = identity(checkpoint) if checkpoint is not None else None
    if selected is not None:
        selected.update(selection=selection, within_run=Path(selected["path"]).parent == run,
                        has_assessment=selected["sha256"] in groups)
    output.mkdir(parents=True)
    reader, rows, summaries, baseline_rows = TrajectoryReader(), [], {}, {}
    for digest, group in baselines.items():
        for key, row in group["rows"].items():
            baseline_rows[key] = flatten_row(row, reader.metrics(row))
    for digest, group in groups.items():
        flat = {key: flatten_row(row, reader.metrics(row)) for key, row in group["rows"].items()}
        values = list(flat.values())
        rows.extend(values)
        by_profile, by_case = defaultdict(list), defaultdict(list)
        for row in values:
            by_profile[row["profile"]].append(row)
            by_case[row["kind"]].append(row)
        summaries[digest] = {"checkpoint": group["checkpoint"], "checkpoint_now": group["checkpoint_now"],
            "reports": group["reports"], "overall": aggregate(values),
            "by_profile": {key: aggregate(part) for key, part in by_profile.items()},
            "by_case_kind": {key: aggregate(part) for key, part in by_case.items()},
            "regression": compare_baseline(flat, baseline_rows) if baseline_sources else None}
        if baseline_sources:
            comparison = summaries[digest]["regression"]
            compared_reports = [*group["reports"], *baseline_sources]
            signatures = {protocol_signature(report) for report in compared_reports}
            comparison["critical_source_hashes_available"] = all(
                all(protocol_sources(report).values()) for report in compared_reports)
            comparison["protocol_compatible"] = len(signatures) == 1 and comparison["critical_source_hashes_available"]
            comparison["critical_source_suffixes"] = PROTOCOL_SOURCE_SUFFIXES
            comparison["verified_no_regression"] = (
                not comparison["regressed"] and comparison["protocol_compatible"]
                and not comparison["paired_initial_state_mismatches"]
                and not comparison["paired_initial_state_unavailable"]
                and not comparison["applied_vector_mismatches"]
                and not comparison["applied_vector_unavailable"])
    write_csv(output / "assessment_samples.csv", rows)
    write_csv(output / "baseline_samples.csv", list(baseline_rows.values()))
    write_csv(output / "training_scalars.csv", [dict(tag=tag, **point) for tag, points in series.items() for point in points])
    plot_training(series, training, output)
    for digest, group in groups.items():
        summaries[digest]["representative_hold_trace"] = plot_representative_hold(group, digest, output)
    summary = {"kind": "offline standing-training and behavioral-assessment analysis",
        "created_at": datetime.now().astimezone().isoformat(), "source": identity(__file__),
        "training": training, "selected_checkpoint": selected, "assessments": assessments,
        "baseline_assessments": baseline_sources, "models": summaries,
        "scope": "Training reward is diagnostic; only assessment cases describe standing behavior. Different model hashes are never pooled.",
        "speed_definition": "Speed-v2 acceptance uses raw world XY qvel at 500 Hz; positional finite differences remain diagnostic for historical traces.",
        "regression_rule": "Any previous successful case that fails or is missing is a regression. Verified preservation also requires matching initial states, actual disturbance vectors and complete matching assessment/standing/fall-gate source hashes; reward and runner source differences alone do not invalidate the protocol.",
        "artifacts": {name: str(output / name) for name in (
            "reward_curves.png", "episode_curves.png", "training_scalars.csv", "assessment_samples.csv", "baseline_samples.csv")}}
    (output / "summary.json").write_text(json.dumps(clean_json(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "models": len(summaries), "assessment_samples": len(rows),
                      "regressed_models": [digest for digest, report in summaries.items()
                                           if report["regression"] and report["regression"]["regressed"]]}), flush=True)


if __name__ == "__main__":
    main()
