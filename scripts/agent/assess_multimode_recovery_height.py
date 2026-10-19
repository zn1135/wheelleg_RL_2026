#!/usr/bin/env python3
"""Paired CPU sim2sim diagnostics before/after the recovery-height change.

Run both source versions with the same checkpoint, initial qpos/qvel and command.
Report exact trajectory differences, recovery-mode switch times, and final 1 s
balance using the thresholds in the wheel-balance v3.1 document. This samples the
CPU entry's resets, not the full training acceptance distribution or H7 firmware.

Usage (repo root):
  mjlab_training/.venv/bin/python scripts/agent/assess_multimode_recovery_height.py \
    --before <pre-change-entry.py> --checkpoint <model_*.pt> \
    --output artifacts/<new-name>/result.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim2sim"))
import mj_sim2sim_huananhu_multimode as entry


def load_before(path):
    spec = importlib.util.spec_from_file_location("before_recovery_height", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A saved source snapshot lives outside sim2sim; use the same physical assets.
    module.MODEL_ASSET, module.POSE_BANK = entry.MODEL_ASSET, entry.POSE_BANK
    return module


def support(env):
    """Ground reaction on each wheel and net ground force on the base, in world axes."""
    wheels, base = np.zeros(2), np.zeros(3)
    force = np.zeros(6)
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if env.floor_geom not in (contact.geom1, contact.geom2):
            continue
        mujoco.mj_contactForce(env.model, env.data, i, force)
        world = contact.frame.reshape(3, 3).T @ force[:3]
        # MuJoCo's positive contact force acts on geom2.
        body = env.model.geom_bodyid[contact.geom2 if contact.geom1 == env.floor_geom
                                     else contact.geom1]
        if contact.geom2 == env.floor_geom:
            world = -world
        for index, wheel in enumerate(env.wheel_bodies):
            if body == wheel:
                wheels[index] += world[2]
        if body == env.base_body:
            base += world
    return wheels, float(np.linalg.norm(base))


def rollout(env, controller, seconds):
    dt = entry.DT * entry.DECIMATION
    weight = float(env.model.body_mass.sum() * np.linalg.norm(env.model.opt.gravity))
    trace = {key: [] for key in ("qpos", "qvel", "action", "recovered", "height_m",
                                "tilt_deg", "wheel_up_n", "base_force_n", "balance")}
    recovered_at, hold_steps, error = None, 0, None
    for _ in range(round(seconds / dt)):
        try:
            action, _ = env.step(controller)
        except FloatingPointError as exc:
            error = str(exc)
            break
        row = env.metrics()
        wheel, base = support(env)
        _, angular = env.body_velocity()
        tilt = max(abs(v) for v in row["roll_pitch_deg"])
        # Current final-balance v3.1: rolling/yaw are allowed; wheel support matters.
        balanced = bool(tilt <= 35. and base <= 1.
                        and (wheel >= max(.1 * weight, 5.)).all()
                        and .8 * weight <= wheel.sum() <= 1.2 * weight
                        and np.linalg.norm(angular[:2]) <= .5)
        hold_steps = hold_steps + 1 if balanced else 0
        if env.recovered and recovered_at is None:
            recovered_at = float(env.data.time)
        values = (env.data.qpos.copy(), env.data.qvel.copy(), action.copy(),
                  bool(env.recovered), row["height_m"], tilt, wheel, base, balanced)
        for key, value in zip(trace, values, strict=True):
            trace[key].append(value)
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    completed = error is None and len(trace["height_m"]) == round(seconds / dt)
    summary = {
        "completed": completed, "numerical_error": error, "recovered_at_s": recovered_at,
        "final_balance_hold_s": hold_steps * dt,
        "final_balanced": bool(completed and hold_steps * dt >= 1. - 1e-9),
        "weight_n": weight,
    }
    if len(trace["height_m"]):
        summary.update(final_height_m=float(arrays["height_m"][-1]),
                       final_tilt_deg=float(arrays["tilt_deg"][-1]),
                       final_wheel_up_n=arrays["wheel_up_n"][-1].tolist(),
                       final_base_force_n=float(arrays["base_force_n"][-1]))
    return arrays, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ground-starts", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = load_before(args.before)
    actor = entry.load_actor(args.checkpoint)

    def controller(obs):
        return actor(torch.from_numpy(obs)[None]).numpy()[0]

    normal_commands = ((0., 0., .30), (.5, 0., .30), (-.5, 0., .30),
                       (1.5, 0., .30), (-1.5, 0., .30),
                       (0., .5, .27), (0., -.5, .34))
    cases = [(f"normal-{i}", "normal", command, 0)
             for i, command in enumerate(normal_commands)]
    cases += [(f"ground-{seed}", "ground", (0., 0., .30), seed)
              for seed in range(args.ground_starts)]
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    traces, results = {}, []
    old, new = before.Playback(), entry.Playback()
    with torch.inference_mode():
        for name, pose, command, seed in cases:
            for env in (old, new):
                env.rng = np.random.default_rng(seed)
                env.command[:] = command
                env.reset(pose)
            assert np.array_equal(old.data.qpos, new.data.qpos), "unpaired initial positions"
            assert np.array_equal(old.data.qvel, new.data.qvel), "unpaired initial velocities"
            initial_qpos, initial_qvel = new.data.qpos.tolist(), new.data.qvel.tolist()
            a, old_result = rollout(old, controller, args.seconds)
            b, new_result = rollout(new, controller, args.seconds)
            gaps = {}
            for key in ("qpos", "qvel", "action"):
                count = min(len(a[key]), len(b[key]))
                gaps[key] = (float(np.max(np.abs(a[key][:count] - b[key][:count])))
                             if count else None)
            count = min(len(a["recovered"]), len(b["recovered"]))
            hits = np.flatnonzero(a["recovered"][:count] != b["recovered"][:count])
            result = {"case": name, "pose": pose, "seed": seed, "command": list(command),
                      "initial_qpos": initial_qpos, "initial_qvel": initial_qvel,
                      "before": old_result, "after": new_result, "max_abs_gap": gaps,
                      "first_mode_difference_s": None if not len(hits) else
                      float((hits[0] + 1) * entry.DT * entry.DECIMATION),
                      "regressed": old_result["final_balanced"] and not new_result["final_balanced"]}
            results.append(result)
            for version, data in (("before", a), ("after", b)):
                traces.update({f"{name}/{version}/{key}": value for key, value in data.items()})
            print(json.dumps({key: result[key] for key in
                              ("case", "max_abs_gap", "first_mode_difference_s", "regressed")}),
                  flush=True)
    np.savez_compressed(output_dir / "trajectories.npz", **traces)
    summary = {}
    for pose in ("normal", "ground"):
        rows = [r for r in results if r["pose"] == pose]
        summary[pose] = {
            "pairs": len(rows),
            "before_balanced": sum(r["before"]["final_balanced"] for r in rows),
            "after_balanced": sum(r["after"]["final_balanced"] for r in rows),
            "regressed_cases": [r["case"] for r in rows if r["regressed"]],
            "changed_mode_cases": [r["case"] for r in rows if r["first_mode_difference_s"] is not None],
            "identical_physics_cases": [r["case"] for r in rows
                                        if all(v == 0. for v in r["max_abs_gap"].values())
                                        and r["before"]["completed"] and r["after"]["completed"]],
            "numerical_errors": [r["case"] for r in rows
                                 if r["before"]["numerical_error"] or r["after"]["numerical_error"]],
        }
    report = {"scope": "paired native CPU recovery-height diagnostics; not full acceptance",
              "seconds": args.seconds, "controller": "pytorch", "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": entry.sha256(args.checkpoint),
              "before_source": str(args.before), "before_sha256": entry.sha256(args.before),
              "after_sha256": entry.sha256(Path(entry.__file__)),
              "model_asset_sha256": entry.sha256(entry.MODEL_ASSET),
              "pose_bank_sha256": entry.sha256(entry.POSE_BANK),
              "balance_criterion": "wheel-balance-final-1s-v3.1 on native CPU contacts",
              "summary": summary, "cases": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
