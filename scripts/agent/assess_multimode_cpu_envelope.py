#!/usr/bin/env python3
"""CPU-entry envelope diagnostics: start distribution, friction sweep, engine trace.

Follow-up to the 2026-10-17 contact-parameter audit. All modes run the independent
CPU MuJoCo entry (sim2sim/mj_sim2sim_huananhu_multimode.py, Playback) at 50 Hz with
the checkpoint actor; verdicts use assess_multimode_tracking's episode_verdict
arithmetic and tolerances.

Modes:
  starts  Replay N training-distribution normal starts at one command. The start
          distribution replicates sample_reset_state's normal branch in numpy from
          the same pose-bank npz (left/right rows drawn independently, root z from
          the tilt-clearance rule, root roll/pitch U(+-.15), yaw U(+-pi), wheel
          spin U(+-2pi), wheel speed U(+-50), leg motor speed U(+-pi/2), zero root
          velocity). A deterministic stance start is always measured as reference.
          --friction training additionally draws per-start contact materials exactly
          like training's huananhu_randomization._material (wheel slide
          min(U(.5,1.2), U(.4,1.)), base slide U(.01,.1), solref (2*dt, 1)).
  sweep   Wheel sliding-friction sweep at one command from the stance start
          (--slides). The override writes the same model field the audit checked.
  trace   Run the same stance-start command on the CPU entry and on the training
          engine (warp, play config, stance reset, via
          measure_multimode_leg_motion._run_mjlab) and report how the 50 Hz
          trajectories diverge (per-step max |dq| over the four leg motors,
          per-step max |dqd| over the two wheels, |dheight|, first-crossing
          times) plus each engine's fall time. Both engines take one zero-action
          flush step after the reset (the warp assessment convention), so the
          recordings align in time. --twin-perturb EPS
          re-runs the CPU entry with EPS added to the initial root height and
          reports the twin's divergence from the unperturbed run: the intrinsic
          sensitivity that engine-level numerical differences also ride on.

Fall emulation on the CPU side mirrors the training failure gate: down = base
contact or |tilt| > 40 deg, failure = down sustained for the task's recovery
window (2.0 s continuous), measurement window frozen at failure like the warp
assessment does.

Diagnostic tool, not an acceptance protocol.

Usage (repo root, mjlab venv):
  mjlab_training/.venv/bin/python scripts/agent/assess_multimode_cpu_envelope.py \
    --mode starts --checkpoint <model_*.pt> --starts 64 --timeout 20 \
    --output artifacts/<name>/cpu-starts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "agent"))
sys.path.insert(0, str(REPO_ROOT / "sim2sim"))
sys.path.insert(0, str(REPO_ROOT / "mjlab_training" / "src"))

import mujoco  # noqa: E402
import torch  # noqa: E402

import assess_multimode_tracking as assess  # noqa: E402
import measure_multimode_leg_motion as measure  # noqa: E402
import mj_sim2sim_huananhu_multimode as entry  # noqa: E402

RECOVERY_WINDOW_S = 2.0
DOWN_TILT_DEG = 40.0


def _cpu_wheel_geom_ids(env):
    model = env.model
    ids = []
    for name in ("lfwheel", "rfwheel"):
        body = model.body(name).id
        ids.extend(i for i in range(model.ngeom)
                   if model.geom_bodyid[i] == body
                   and (model.geom_contype[i] or model.geom_conaffinity[i]))
    return np.asarray(ids, dtype=np.intp)


def _cpu_base_geom_ids(env):
    model = env.model
    body = model.body("base_link").id
    return np.asarray([i for i in range(model.ngeom)
                       if model.geom_bodyid[i] == body
                       and (model.geom_contype[i] or model.geom_conaffinity[i])],
                      dtype=np.intp)


def _sample_bank_starts(count, seed):
    """sample_reset_state's normal branch, replicated in numpy from the same bank."""
    with np.load(entry.POSE_BANK, allow_pickle=False) as bank:
        left_pos, right_pos = bank["normal_left_pos"], bank["normal_right_pos"]
        left_tilt, right_tilt = bank["normal_left_tilt"], bank["normal_right_tilt"]
        base_tilt = float(bank["base_tilt"])
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(count):
        li = int(rng.integers(len(left_pos)))
        ri = int(rng.integers(len(right_pos)))
        minimum = min(float(left_tilt[li]), float(right_tilt[ri]))
        minimum = min(minimum, base_tilt)          # torch clamp(max=base_tilt_min_z)
        root_z = max(0.38, 0.002 - minimum)        # CLEARANCE=0.002, NORMAL_ROOT_HEIGHT=0.38
        roll, pitch, yaw = (rng.uniform(-.15, .15), rng.uniform(-.15, .15),
                            rng.uniform(-np.pi, np.pi))
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        quat = (cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp)
        starts.append({
            "joints": np.concatenate([left_pos[li], right_pos[ri]]).astype(np.float64),
            "root_z": float(root_z), "quat": tuple(float(v) for v in quat),
            "wheel_q": rng.uniform(-2 * np.pi, 2 * np.pi, 2),
            "leg_v": rng.uniform(-np.pi / 2, np.pi / 2, 4),
            "wheel_v": rng.uniform(-50., 50., 2),
        })
    return starts


def _write_start(env, start):
    model = env.model
    env.reset("normal")
    env.data.qpos[:7] = (0., 0., start["root_z"], *start["quat"])
    for name, value in zip(entry.POSE_JOINTS, start["joints"], strict=True):
        env.data.qpos[model.joint(name).qposadr[0]] = value
    for name, value in zip(("lfwheel", "rfwheel"), start["wheel_q"], strict=True):
        env.data.qpos[model.joint(name).qposadr[0]] = value
    env.data.qvel[:] = 0.
    for name, value in zip(("lf0", "lf00", "rf0", "rf00"), start["leg_v"], strict=True):
        env.data.qvel[model.joint(name).dofadr[0]] = value
    for name, value in zip(("lfwheel", "rfwheel"), start["wheel_v"], strict=True):
        env.data.qvel[model.joint(name).dofadr[0]] = value
    mujoco.mj_forward(model, env.data)
    residual = env._closed_chain_residual(model, env.data)
    if not np.isfinite(residual) or residual > 1e-6:
        raise RuntimeError(f"sampled start violates the closed chain: {residual:.3e} m")


def _set_materials(model, wheel_ids, base_ids, *, wheel_slide=None, base_slide=None,
                   training_solref=False):
    if wheel_slide is not None:
        model.geom_friction[wheel_ids, 0] = wheel_slide
    if base_slide is not None:
        model.geom_friction[base_ids, 0] = base_slide
    if training_solref:
        for ids in (wheel_ids, base_ids):
            model.geom_solref[ids, 0] = 2 * entry.DT
            model.geom_solref[ids, 1] = 1.


def _rollout(env, actor, command, seconds, *, settle=None):
    """One 50 Hz rollout; returns per-step arrays including body vx and yaw rate."""
    torch.set_num_threads(1)
    env.command[:] = command

    def controller(obs):
        with torch.inference_mode():
            return actor(torch.from_numpy(obs)[None]).numpy()[0]

    # One zero-action flush step after the reset, mirroring the warp side
    # (measure_multimode_leg_motion._run_mjlab / assess_multimode_tracking);
    # without it the two recorded series are offset by one control step.
    env.step(lambda obs: np.zeros(len(entry.MOTORS)))

    control_dt = entry.DT * entry.DECIMATION
    total = int(round(seconds / control_dt))
    t, q, qd, action, vx, yaw_rate, height, tilt, contact = ([] for _ in range(9))
    for step in range(total):
        applied, _ = env.step(controller)
        t.append((step + 1) * control_dt)
        q.append(env.data.qpos[env.q].copy())
        qd.append(env.data.qvel[env.v].copy())
        action.append(np.asarray(applied, dtype=np.float64))
        row = env.metrics()
        vx.append(row["vx_m_s"])
        yaw_rate.append(row["yaw_rate_rad_s"])
        height.append(row["height_m"])
        tilt.append(max(abs(v) for v in row["roll_pitch_deg"]))
        contact.append(bool(row["base_contact"]))
    return {"t": np.asarray(t), "q": np.asarray(q), "qd": np.asarray(qd),
            "action": np.asarray(action), "vx": np.asarray(vx),
            "yaw_rate": np.asarray(yaw_rate), "height": np.asarray(height),
            "tilt": np.asarray(tilt), "contact": np.asarray(contact)}


def _first_failure(rollout):
    """Failure step index (down >= recovery window) or None; matches the task gate."""
    down = rollout["contact"] | (rollout["tilt"] > DOWN_TILT_DEG)
    steps = round(RECOVERY_WINDOW_S / (rollout["t"][1] - rollout["t"][0]))
    run = 0
    for index, value in enumerate(down):
        run = run + 1 if value else 0
        if run >= steps:
            return index - run + 1
    return None


def _verdict(rollout, command, settle, numerical=False):
    t, dt = rollout["t"], rollout["t"][1] - rollout["t"][0]
    fail_step = _first_failure(rollout)
    measured = (t >= settle) & (np.arange(len(t)) <= fail_step if fail_step is not None
                                else np.ones(len(t), dtype=bool))
    mask = measured.astype(np.float64) * dt
    observed = float(mask.sum())
    if observed <= 0.0:
        return {"means": {"observed_s": 0.0}, "checks": {"survived": False},
                "failed_checks": ["no_measured_steps"], "passed": False,
                "tolerances": {"vx_m_s": assess.VX_TOL, "yaw_rad_s": assess.YAW_TOL,
                               "height_m": assess.HEIGHT_TOL,
                               "tilt_deg": assess.TILT_TOL_DEG,
                               "base_contact_n": assess.BASE_CONTACT_N}}
    verdict = assess.episode_verdict(
        observed_s=observed,
        vx_err_abs_sum=float((np.abs(command[0] - rollout["vx"]) * mask).sum()),
        vx_err_signed_sum=float(((rollout["vx"] - command[0]) * mask).sum()),
        vy_abs_sum=0.0,
        yaw_err_abs_sum=float((np.abs(command[1] - rollout["yaw_rate"]) * mask).sum()),
        yaw_err_signed_sum=float(((rollout["yaw_rate"] - command[1]) * mask).sum()),
        height_err_abs_sum=float((np.abs(command[2] - rollout["height"]) * mask).sum()),
        tilt_max_deg=float(rollout["tilt"][measured].max()) if measured.any() else 0.0,
        base_contact_s=float((rollout["contact"].astype(np.float64) * mask).sum()),
        numerical_failure=numerical,
        survived=fail_step is None)
    verdict["ended_step"] = None if fail_step is None else int(fail_step + 1)
    verdict["fail_time_s"] = None if fail_step is None else float(t[fail_step])
    return verdict


def _first_crossing(values, threshold):
    hits = np.nonzero(values > threshold)[0]
    return None if len(hits) == 0 else int(hits[0])


def mode_starts(args, actor, env):
    starts = _sample_bank_starts(args.starts, args.seed)
    wheel_ids, base_ids = _cpu_wheel_geom_ids(env), _cpu_base_geom_ids(env)
    rows = []

    def record(label, run_seed, start):
        if args.friction == "training":
            rng = np.random.default_rng(run_seed + 100000)
            wheel_slide = float(min(rng.uniform(.5, 1.2), rng.uniform(.4, 1.)))
            base_slide = float(rng.uniform(.01, .1))
            _set_materials(env.model, wheel_ids, base_ids,
                           wheel_slide=wheel_slide, base_slide=base_slide,
                           training_solref=True)
        else:
            wheel_slide, base_slide = float(env.model.geom_friction[wheel_ids[0], 0]), None
        rollout = _rollout(env, actor, args.command, args.seconds)
        verdict = _verdict(rollout, args.command, args.settle)
        rows.append({"start": label, "seed": run_seed,
                     "wheel_slide": wheel_slide, "base_slide": base_slide, **verdict})

    env.reset("normal")
    record("stance", args.seed, None)
    for index, start in enumerate(starts):
        _write_start(env, start)
        record(f"bank{index}", args.seed * 1000 + index, start)
    summary = {
        "count": len(rows),
        "passed": int(sum(row["passed"] for row in rows)),
        "success_rate": sum(row["passed"] for row in rows) / len(rows),
        "rate_survived": sum(row["checks"]["survived"] for row in rows) / len(rows),
        "failure_checks": sorted({name for row in rows for name in row["failed_checks"]}),
    }
    bad = [row for row in rows if not row["passed"]]
    for row in bad[:12]:
        print(f"  fail {row['start']:>7s} slide {row['wheel_slide']:.3f} "
              f"checks {row['failed_checks']} fail_t {row['fail_time_s']}")
    return {"rows": rows, "summary": summary}


def mode_sweep(args, actor, env):
    wheel_ids, base_ids = _cpu_wheel_geom_ids(env), _cpu_base_geom_ids(env)
    rows = []
    for slide in args.slides:
        env.reset("normal")
        _set_materials(env.model, wheel_ids, base_ids, wheel_slide=slide)
        rollout = _rollout(env, actor, args.command, args.seconds)
        verdict = _verdict(rollout, args.command, args.settle)
        rows.append({"wheel_slide": slide, **verdict})
        print(f"  slide {slide:.3f} passed {verdict['passed']} "
              f"fail_t {verdict['fail_time_s']} checks {verdict['failed_checks']}")
    return {"rows": rows}


def mode_trace(args, actor, env):
    env.reset("normal")
    cpu = _rollout(env, actor, args.command, args.seconds)
    twin = None
    if args.twin_perturb:
        env.reset("normal")
        env.data.qpos[2] += args.twin_perturb
        mujoco.mj_forward(env.model, env.data)
        twin = _rollout(env, actor, args.command, args.seconds)
        env.reset("normal")
    warp_namespace = argparse.Namespace(
        checkpoint=args.checkpoint, expect_sha256=args.expect_sha256, reset="stance",
        envs=1, seed=args.seed, device=args.device, vx=float(args.command[0]),
        yaw=float(args.command[1]), height=float(args.command[2]), seconds=args.seconds)
    arrays, _, _ = measure._run_mjlab(warp_namespace)
    length = min(len(cpu["t"]), len(arrays["t"]))
    warp_q = np.asarray(arrays["q"])[:length, 0]
    warp_qd = np.asarray(arrays["qd"])[:length, 0]
    warp_height = np.asarray(arrays["height"])[:length, 0]
    warp_contact = np.asarray(arrays["contact"])[:length, 0]
    warp_tilt = np.asarray(arrays["tilt"])[:length, 0]
    t = np.asarray(arrays["t"])[:length]
    leg_cols, wheel_cols = (0, 1, 3, 4), (2, 5)
    dq = np.abs(cpu["q"][:length] - warp_q).max(axis=1)
    dq_legs = np.abs(cpu["q"][:length, leg_cols] - warp_q[:, leg_cols]).max(axis=1)
    dqd_wheels = np.abs(cpu["qd"][:length, wheel_cols] - warp_qd[:, wheel_cols]).max(axis=1)
    dh = np.abs(cpu["height"][:length] - warp_height)
    thresholds_dq = (0.01, 0.05, 0.1, 0.5)
    thresholds_dqd = (0.1, 1., 5.)
    thresholds_dh = (0.005, 0.02, 0.05)

    def crossings(series, thresholds, times):
        return {str(th): (None if (idx := _first_crossing(series, th)) is None
                          else float(times[idx])) for th in thresholds}

    sample_times = [s for s in (1., 2., 5., 10., 15.) if s < args.seconds - 0.1]
    sample_times.append(round(args.seconds - 0.2, 3))
    samples = []
    for sample in sample_times:
        index = int(np.argmin(np.abs(t - sample)))
        samples.append({"t": float(t[index]), "dq_max": float(dq[index]),
                        "dq_legs": float(dq_legs[index]),
                        "dqd_wheels": float(dqd_wheels[index]),
                        "dheight": float(dh[index])})
    def fall_time(times, contact):
        hits = np.nonzero(contact)[0]
        return None if len(hits) == 0 else float(times[hits[0]])
    payload = {
        "command": {"vx_m_s": args.command[0], "yaw_rad_s": args.command[1],
                    "height_m": args.command[2]},
        "duration_s": args.seconds,
        "cpu_fall_time_s": fall_time(cpu["t"], cpu["contact"]),
        "cpu_tilt_max_deg": float(cpu["tilt"].max()),
        "warp_fall_time_s": fall_time(t, warp_contact),
        "warp_tilt_max_deg": float(warp_tilt.max()),
        "divergence": {"dq_legs_first_crossing_s": crossings(dq_legs, thresholds_dq, t),
                       "dqd_wheels_first_crossing_s": crossings(dqd_wheels, thresholds_dqd, t),
                       "dheight_first_crossing_s": crossings(dh, thresholds_dh, t),
                       "dq_all_first_crossing_s": crossings(dq, thresholds_dq, t),
                       "samples": samples},
        "twin_perturb": args.twin_perturb,
    }
    if twin is not None:
        cpu_q, cpu_t = cpu["q"][:length], cpu["t"][:length]
        twin_dq = np.abs(twin["q"][:length] - cpu_q).max(axis=1)
        twin_dq_legs = np.abs(twin["q"][:length, leg_cols] - cpu_q[:, leg_cols]).max(axis=1)
        sample_index = [min(int(np.argmin(np.abs(cpu_t - s))), length - 1) for s in sample_times]
        payload["twin"] = {
            "dq_legs_first_crossing_s": crossings(twin_dq_legs, thresholds_dq, cpu_t),
            "dq_all_first_crossing_s": crossings(twin_dq, thresholds_dq, cpu_t),
            "samples": [{"t": float(cpu_t[index]), "dq_legs": float(twin_dq_legs[index])}
                        for index in sample_index],
        }
    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(npz_path, t=cpu["t"], cpu_q=cpu["q"], cpu_qd=cpu["qd"],
                        cpu_action=cpu["action"],
                        warp_action=np.asarray(arrays["action"])[:, 0],
                        cpu_height=cpu["height"], cpu_contact=cpu["contact"],
                        cpu_tilt=cpu["tilt"],
                        warp_t=arrays["t"], warp_q=np.asarray(arrays["q"])[:, 0],
                        warp_qd=np.asarray(arrays["qd"])[:, 0],
                        warp_height=np.asarray(arrays["height"])[:, 0],
                        warp_contact=np.asarray(arrays["contact"])[:, 0],
                        warp_tilt=np.asarray(arrays["tilt"])[:, 0],
                        twin_q=None if twin is None else twin["q"])
    print("cpu fall:", payload["cpu_fall_time_s"], "warp fall:", payload["warp_fall_time_s"])
    print("dq_legs crossings:", payload["divergence"]["dq_legs_first_crossing_s"])
    print("dqd_wheels crossings:", payload["divergence"]["dqd_wheels_first_crossing_s"])
    print("samples:", json.dumps(samples))
    if "twin" in payload:
        print("twin dq_legs crossings:", payload["twin"]["dq_legs_first_crossing_s"])
    print("WROTE", npz_path)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("starts", "sweep", "trace"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expect-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--height", type=float, default=.30)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--settle", type=float, default=assess.SETTLE_S)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--friction", choices=("fixed", "training"), default="fixed",
                        help="starts mode: XML friction, or a per-start training draw")
    parser.add_argument("--slides", default="0.4,0.45,0.475,0.5,0.525,0.55,0.6,0.7,0.85,1.0")
    parser.add_argument("--twin-perturb", type=float, default=None,
                        help="trace mode: root-height perturbation EPS for the CPU twin")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seconds <= args.settle:
        raise SystemExit("--seconds must exceed --settle")
    if args.mode == "sweep":
        args.slides = [float(item) for item in args.slides.split(",") if item.strip()]
    args.command = (args.vx, args.yaw, args.height)
    actor = entry.load_actor(args.checkpoint, args.expect_sha256)
    env = entry.Playback("normal", args.command)
    result = {"starts": mode_starts, "sweep": mode_sweep, "trace": mode_trace}[args.mode](
        args, actor, env)
    payload = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": measure.sha256(Path(__file__).resolve()),
        "git_head": measure.git_head(),
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": measure.sha256(args.checkpoint),
        "command": {"vx_m_s": args.vx, "yaw_rad_s": args.yaw, "height_m": args.height},
        "duration_s": args.seconds, "settle_s": args.settle,
        "friction_mode": args.friction if args.mode == "starts" else None,
        "entry_sha256": measure.sha256(REPO_ROOT / "sim2sim" /
                                       "mj_sim2sim_huananhu_multimode.py"),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
