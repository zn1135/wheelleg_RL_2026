#!/usr/bin/env python3
"""Envelope/start robustness on the training engine under training-time contact materials.

Follow-up to the 2026-10-17 contact-parameter audit. Runs the documented flat-tracking
grid (or a single command point) on the training backend (warp, play configuration,
no observation noise/delay) while restoring parts of the training-time CONTACT
randomization that `play=True` otherwise switches off:

  xml        nothing; the play/eval baseline (XML slide 0.5, XML solref).
  solref     training-time contact solref only: (2*physics_dt, 1) on the wheel and
             base collision geoms, XML sliding friction. Isolates contact softness.
  randomized mirrors huananhu_randomization._material for those geoms: per-env
             sliding friction min(U(.5,1.2), U(.4,1.)) on the wheels, U(.01,.1) on
             the base, plus the same training solref. This is the material part of
             the training domain randomization, which the play config drops.
  pinned     fixed per-env wheel sliding friction from --wheel-slides (one value
             per environment; requires a single --vx/--yaw/--height point). For
             friction-threshold sweeps; no solref or base changes.

Starts: `--reset bank` (training pose-bank normal branch, the same reset the tracking
assessment uses) or `--reset stance` (the deterministic CPU-entry start, 0.38 m
level). Commands are pinned per env exactly like assess_multimode_tracking, and the
per-env verdicts use that script's `episode_verdict` arithmetic. `--envs-per-combo N`
repeats each command point N times, so a single point with `--envs-per-combo 64` is a
distributional start (and, with `--material randomized`, start x friction) sweep.

Reported per env: command, wheel/base sliding friction drawn or pinned, verdict,
ended_step. Aggregates: overall, per command point, and wheel-slide bins.

This is a diagnostic robustness measurement, not an acceptance protocol; duration is
20 s by default (the tracking protocol's 12 s is available via --seconds 12), and the
scope of any claim stays in docs/interfaces/chuanliantui-huananhu-multimode-tracking-v1.md.

Usage (repo root, mjlab venv):
  mjlab_training/.venv/bin/python scripts/agent/assess_multimode_envelope_friction.py \
    --checkpoint <model_*.pt> --output artifacts/<name>/grid-randomized.json \
    --material randomized --envs-per-combo 8
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "agent"))
sys.path.insert(0, str(REPO_ROOT / "mjlab_training" / "src"))

import assess_multimode_tracking as assess  # noqa: E402
import measure_multimode_leg_motion as measure  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import (  # noqa: E402
    EventTermCfg, RecomputeLevel, requires_model_fields)
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from ct_mjlab.envs import task_registry  # noqa: E402
from ct_mjlab.envs.chuanliantui import huananhu_randomization as hr  # noqa: E402

TASK = assess.TASK
FRICTION_BINS = ((.4, .5), (.5, .55), (.55, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01))


def _geom_groups(env):
    """Wheel/base collision geom ids; same selection as hr.HuananhuRandomization."""
    import torch
    robot = env.scene["robot"]
    model = env.sim.mj_model
    base_id = int(robot.indexing.body_ids[robot.body_names.index("base_link")])
    wheel_bodies = {int(robot.indexing.body_ids[robot.body_names.index(n)])
                    for n in ("lfwheel", "rfwheel")}
    collision = lambda i: bool(model.geom_contype[i] or model.geom_conaffinity[i])
    base_ids = [i for i in range(model.ngeom)
                if model.geom_bodyid[i] == base_id and collision(i)]
    wheel_ids = [i for i in range(model.ngeom)
                 if model.geom_bodyid[i] in wheel_bodies and collision(i)]
    return (torch.tensor(base_ids, dtype=torch.long, device=env.device),
            torch.tensor(wheel_ids, dtype=torch.long, device=env.device))


@requires_model_fields("geom_friction", "geom_solref", recompute=RecomputeLevel.none)
def randomize_contact_materials(env, env_ids):
    """Per-env wheel/base friction and contact solref, exactly the training draws."""
    state = hr.HuananhuRandomization(env)
    hr._material(env, state, "base", state.base_geom_ids, (.01, .1), (.01, .1))
    hr._material(env, state, "wheel", state.wheel_geom_ids, (.5, 1.2), (.4, 1.))
    env._envelope_material_state = state


@requires_model_fields("geom_solref", recompute=RecomputeLevel.none)
def force_training_solref(env, env_ids):
    """Training-time contact solref on the wheel/base geoms, friction untouched."""
    base_ids, wheel_ids = _geom_groups(env)
    for ids in (base_ids, wheel_ids):
        env.sim.model.geom_solref[:, ids, 0] = 2 * env.physics_dt
        env.sim.model.geom_solref[:, ids, 1] = 1.


@requires_model_fields("geom_friction", recompute=RecomputeLevel.none)
def pin_wheel_friction(env, env_ids, slides):
    """Fixed per-env wheel sliding friction (one --wheel-slides entry per env)."""
    import torch
    _, wheel_ids = _geom_groups(env)
    values = torch.as_tensor(slides, dtype=env.sim.model.geom_friction.dtype,
                             device=env.device)
    if values.numel() != env.num_envs:
        raise ValueError(f"pinned slides {values.numel()} != num_envs {env.num_envs}")
    env.sim.model.geom_friction[:, wheel_ids, 0] = values[:, None]


def run_grid(args, seed, combos):
    """One seed: build env, pin commands, step, verdicts. Mirrors assess.evaluate_seed."""
    import torch
    configure_torch_backends()
    cfg, agent = task_registry.get_cfgs(TASK, play=True)
    num_envs = len(combos)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.episode_length_s = max(float(cfg.episode_length_s), args.seconds + 5.)
    cfg.commands["standup"].resampling_time_range = (1e9, 1e9)
    cfg.events["reset_pose"] = EventTermCfg(
        func=measure._bank_reset if args.reset == "bank" else measure._stance_reset,
        mode="reset")
    material_fn = {"xml": None, "solref": force_training_solref,
                   "randomized": randomize_contact_materials}.get(args.material)
    if material_fn is not None:
        cfg.events["envelope_materials"] = EventTermCfg(func=material_fn, mode="startup")
    if args.material == "pinned":
        cfg.events["envelope_materials"] = EventTermCfg(
            func=pin_wheel_friction, mode="startup",
            params={"slides": tuple(args.wheel_slides)})
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = task_registry._task(TASK).runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent),
            device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        command = torch.tensor(combos, device=env.device, dtype=torch.float32)
        obs, _ = env.reset(seed=seed)
        state = env.command_manager.get_term("standup")
        base_ids, wheel_ids = _geom_groups(env)
        wheel_slide = env.sim.model.geom_friction[:, wheel_ids, 0].detach().cpu().numpy()
        base_slide = env.sim.model.geom_friction[:, base_ids, 0].detach().cpu().numpy()
        assess.pin_commands(env, state, command)
        obs, _, _, _, _ = env.step(torch.zeros((num_envs, 6), device=env.device))
        dt = env.step_dt
        settle_steps = round(args.settle / dt)
        total_steps = round(args.seconds / dt)
        names = ("vx_abs", "vx_signed", "vy_abs", "yaw_abs", "yaw_signed", "height_abs",
                 "observed", "base_contact")
        acc = {name: torch.zeros(num_envs, device=env.device) for name in names}
        tilt_max = torch.zeros(num_envs, device=env.device)
        jump_steps = torch.zeros(num_envs, device=env.device)
        live = torch.ones(num_envs, device=env.device, dtype=torch.bool)
        ended = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        ended_step = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        numerical = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        with torch.inference_mode():
            for step in range(total_steps):
                assess.pin_commands(env, state, command)
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                assess.assert_actor_command(obs, command, step)
                measured = live & (step + 1 > settle_steps)
                linear = env.scene["robot"].data.root_link_lin_vel_b
                angular = env.scene["robot"].data.root_link_ang_vel_b
                gravity = env.scene["robot"].data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.asin(gravity[:, 0].clamp(-1, 1)).abs()
                tilt_deg = torch.rad2deg(torch.maximum(roll, pitch))
                base_history = env.scene["base_contact"].data.force_history
                base_contact = base_history.norm(dim=-1).amax(dim=(1, 2)) > assess.BASE_CONTACT_N
                vx_error = command[:, 0] - linear[:, 0]
                yaw_error = command[:, 1] - angular[:, 2]
                height_error = state.height_above_ground - command[:, 2]
                mask = measured.float()
                acc["vx_abs"] += vx_error.abs() * mask * dt
                acc["vx_signed"] += (linear[:, 0] - command[:, 0]) * mask * dt
                acc["vy_abs"] += linear[:, 1].abs() * mask * dt
                acc["yaw_abs"] += yaw_error.abs() * mask * dt
                acc["yaw_signed"] += (angular[:, 2] - command[:, 1]) * mask * dt
                acc["height_abs"] += height_error.abs() * mask * dt
                acc["observed"] += mask * dt
                acc["base_contact"] += (base_contact & measured).float() * dt
                jump_steps += (state.mode()[:, assess.JUMP_INDEX] > 0).float() * mask * dt
                tilt_max = torch.where(measured, torch.maximum(tilt_max, tilt_deg), tilt_max)
                numerical |= live & state.numerical_failure
                newly_done = terminated | timeout
                first_done = newly_done & live
                ended |= first_done
                ended_step = torch.where(first_done, torch.full_like(ended_step, step + 1),
                                         ended_step)
                live &= ~first_done
                if not bool(live.any()):
                    break
                if bool(newly_done.any()):
                    obs, _ = env.reset(env_ids=newly_done.nonzero().flatten())
                    assess.pin_commands(env, state, command)
        rows = []
        for index, (vx_cmd, yaw_cmd, height_cmd) in enumerate(combos):
            if float(acc["observed"][index]) <= 0.0:
                empty = {key: None for key in ("vx_error_m_s", "vx_bias_m_s", "vy_abs_m_s",
                                               "yaw_error_rad_s", "yaw_bias_rad_s",
                                               "height_error_m", "tilt_max_deg",
                                               "base_contact_s", "observed_s")}
                verdict = {
                    "means": empty,
                    "checks": {"tracked_vx": False, "tracked_yaw": False,
                               "tracked_height": False, "upright": False,
                               "no_base_contact": False, "survived": False,
                               "numerical_ok": not bool(numerical[index])},
                    "failed_checks": ["no_measured_steps"], "passed": False,
                    "tolerances": {"vx_m_s": assess.VX_TOL, "yaw_rad_s": assess.YAW_TOL,
                                   "height_m": assess.HEIGHT_TOL,
                                   "tilt_deg": assess.TILT_TOL_DEG,
                                   "base_contact_n": assess.BASE_CONTACT_N},
                }
            else:
                verdict = assess.episode_verdict(
                    observed_s=float(acc["observed"][index]),
                    vx_err_abs_sum=float(acc["vx_abs"][index]),
                    vx_err_signed_sum=float(acc["vx_signed"][index]),
                    vy_abs_sum=float(acc["vy_abs"][index]),
                    yaw_err_abs_sum=float(acc["yaw_abs"][index]),
                    yaw_err_signed_sum=float(acc["yaw_signed"][index]),
                    height_err_abs_sum=float(acc["height_abs"][index]),
                    tilt_max_deg=float(tilt_max[index]),
                    base_contact_s=float(acc["base_contact"][index]),
                    numerical_failure=bool(numerical[index]),
                    survived=not bool(ended[index]))
            rows.append({
                "env": index, "seed": seed,
                "command": {"vx_m_s": vx_cmd, "yaw_rad_s": yaw_cmd, "height_m": height_cmd},
                "wheel_slide": [float(v) for v in wheel_slide[index]],
                "wheel_slide_mean": float(np.mean(wheel_slide[index])),
                "base_slide_mean": float(np.mean(base_slide[index])),
                "jump_seconds": float(jump_steps[index]),
                "ended_step": int(ended_step[index]),
                **verdict,
            })
        extras = {"seed": seed, "num_envs": num_envs, "settle_s": args.settle,
                  "duration_s": args.seconds,
                  "survived": int(sum(row["checks"]["survived"] for row in rows)),
                  "numerical_failures": int(sum(not row["checks"]["numerical_ok"]
                                                for row in rows))}
        return rows, extras
    finally:
        env.close()


def summarize(rows):
    summary = assess.aggregate(rows)
    by_command = {}
    for row in rows:
        key = (row["command"]["vx_m_s"], row["command"]["yaw_rad_s"],
               row["command"]["height_m"])
        group = by_command.setdefault(key, [0, 0])
        group[0] += 1
        group[1] += int(row["passed"])
    summary["by_command"] = [
        {"vx_m_s": key[0], "yaw_rad_s": key[1], "height_m": key[2],
         "count": value[0], "passed": value[1]}
        for key, value in sorted(by_command.items())]
    bins = []
    for low, high in FRICTION_BINS:
        selected = [row for row in rows
                    if low <= row["wheel_slide_mean"] < high]
        if selected:
            bins.append({"wheel_slide": [low, high], "count": len(selected),
                         "passed": int(sum(row["passed"] for row in selected))})
    summary["wheel_slide_bins"] = bins
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material", choices=("xml", "solref", "randomized", "pinned"),
                        default="randomized")
    parser.add_argument("--wheel-slides", default=None,
                        help="comma-separated per-env wheel sliding friction (pinned mode)")
    parser.add_argument("--reset", choices=("bank", "stance"), default="bank")
    parser.add_argument("--vx", default=assess.DEFAULT_VX)
    parser.add_argument("--yaw", default=assess.DEFAULT_YAW)
    parser.add_argument("--height", default=assess.DEFAULT_HEIGHT)
    parser.add_argument("--envs-per-combo", type=int, default=8)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--settle", type=float, default=assess.SETTLE_S)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-success-rate", type=float, default=None,
                        help="exit non-zero when the pooled success rate is below this value")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seconds <= args.settle:
        raise SystemExit("--seconds must exceed --settle")
    vx = assess.parse_list(args.vx, float)
    yaw = assess.parse_list(args.yaw, float)
    height = assess.parse_list(args.height, float)
    combos = assess.build_combos(vx, yaw, height)
    if args.material == "pinned":
        if args.wheel_slides is None:
            raise SystemExit("pinned material requires --wheel-slides")
        args.wheel_slides = [float(item) for item in args.wheel_slides.split(",")
                             if item.strip()]
        if not (len(vx) == 1 and len(yaw) == 1 and len(height) == 1):
            raise SystemExit("pinned material requires a single --vx/--yaw/--height point")
        combos = combos * len(args.wheel_slides)
    elif args.envs_per_combo > 1:
        combos = [combo for combo in combos for _ in range(args.envs_per_combo)]
    seeds = assess.parse_list(args.seeds, int)
    all_rows, seed_extras = [], []
    for seed in seeds:
        rows, extras = run_grid(args, seed, combos)
        all_rows.extend(rows)
        seed_extras.append(extras)
    summary = summarize(all_rows)
    payload = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": measure.sha256(Path(__file__).resolve()),
        "git_head": measure.git_head(),
        "task": TASK,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": measure.sha256(args.checkpoint),
        "material": args.material,
        "wheel_slides_pinned": args.wheel_slides if args.material == "pinned" else None,
        "reset": args.reset,
        "seeds": seeds,
        "settle_s": args.settle,
        "duration_s": args.seconds,
        "aggregate": summary,
        "seeds_detail": seed_extras,
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"material={args.material} reset={args.reset} envs={len(all_rows)} "
          f"pass={summary['passed']}/{summary['count']} "
          f"rate={summary['success_rate']:.4f}")
    print("failure_checks:", summary["failure_checks"])
    print("wheel_slide_bins:", json.dumps(summary["wheel_slide_bins"]))
    bad = [row for row in all_rows if not row["passed"]
           and row["failed_checks"] != ["no_measured_steps"]]
    for row in bad[:12]:
        print(f"  fail env {row['env']:4d} cmd {row['command']} "
              f"slide {row['wheel_slide_mean']:.3f} checks {row['failed_checks']}")
    print("WROTE", args.output)
    if args.min_success_rate is not None and summary["success_rate"] < args.min_success_rate:
        raise SystemExit(f"success rate {summary['success_rate']:.4f} below "
                         f"--min-success-rate {args.min_success_rate}")


if __name__ == "__main__":
    main()
