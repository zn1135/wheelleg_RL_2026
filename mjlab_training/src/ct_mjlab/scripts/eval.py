"""Deployment-style verification of the multimode policy's recover (stand-up) phase.

Ported from the frozen model_3023 evaluator; only imports/registry access changed.

Runs a `Chuanliantui-Huananhu-Multimode-v0` checkpoint in the deployment-like
``play`` configuration (no domain randomization, observation noise/delay, action
delay, spring bias or pushes) and reports the recover phase separately from the
rest of the episode. Every environment is counted once, on its first episode.

Why a strict criterion is needed
--------------------------------
The task's own ``recovered`` flag is loose: ``stable`` only asks for no base
contact, ``|roll|,|pitch| < 40 deg`` and ``height_above_ground > 0.20 m`` held for
0.5 s. A ground start already sits at 0.25 m and level, so the flag can fire
without the robot ever pushing itself up. This script therefore also reports a
strict criterion that additionally requires both wheels to carry the robot and
the height to track the command, plus a ``stable_at_t0`` diagnostic that shows
whether the loose criterion is satisfied by the start pose itself.

Mid-episode recover path
------------------------
``--knockdown-at S`` tips every live env over at S seconds with the same angular
impulse the training knockdown uses, then reports ``knockdown_hits`` (impulses
delivered), ``knocked_down`` (robots that actually went down) and the strict
recovery that follows each one. Delivery and knockdown are counted separately
because a wheeled robot rolls weak impulses away instead of falling over.

Usage:
  scripts/mjlab.sh eval --checkpoint <model.pt> --num-envs 64 --seeds 42 \
    --start-mode force --terrain all --output logs/mjlab/evaluation/<name>/result.json
  scripts/mjlab.sh eval --checkpoint <model.pt> --num-envs 64 --seeds 42 \
    --start-mode force --terrain all --knockdown-at 5 \
    --output logs/mjlab/evaluation/<name>/result.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.huananhu_poses import sample_reset_state
from ct_mjlab.envs.chuanliantui.multimode_mdp import (MODE_NAMES, OUT_OF_PATCH_RADIUS_M,
                                                       TERRAIN_NAMES)
from ct_mjlab.envs.chuanliantui.multimode_env_cfg import KNOCKDOWN_ANGULAR_SPEED_RANGE

MULTIMODE_TASK_ID = "chuanliantui"

ROOT = Path(__file__).resolve().parents[4]
RECOVER_INDEX = MODE_NAMES.index("recover")
FLAT_INDEX = TERRAIN_NAMES.index("flat")
TILT_LIMIT_RAD = 0.2618          # 15 degrees, the strict orientation gate.
HEIGHT_TOLERANCE_M = 0.03
STABLE_DURATION_S = 0.5
HOLD_WINDOW_S = 15.0
# Termination codes, matching the order used by the other Huananhu evaluators.
TERMINATIONS = ("unfinished", "time_out", "physical", "out_of_patch", "numerical")


def force_ground_reset(env, env_ids):
    """Reset selected envs from the ground pose, ignoring the task's 50/50 draw.

    Mirrors ``multimode_mdp.reset_pose`` with ``ground`` forced true, so the
    sampled pose still comes from the same pose-bank ground branch the training
    distribution uses. Only the ordinary half of the population is dropped.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    state = env.command_manager.get_term("standup")
    ground = torch.ones(len(env_ids), device=env.device, dtype=torch.bool)
    state.ground_start[env_ids] = ground
    sample_reset_state(env, env_ids, ground)
    if hasattr(env, "_huananhu_rewards"):
        env._huananhu_rewards.step = -1


def force_knockdown(env, env_ids, speed_range=KNOCKDOWN_ANGULAR_SPEED_RANGE):
    """Tip the selected envs over with a world-frame angular-velocity impulse.

    Mirrors ``huananhu_randomization.knockdown_root_velocity`` without its
    "already standing" gate, so an evaluation can put every live env on the
    ground at a fixed time and measure the mid-episode recover path. The default
    range is the training recipe's own (``KNOCKDOWN_ANGULAR_SPEED_RANGE``), shared
    with the training event and ``play`` so the three cannot drift apart; a
    wheeled robot rolls weaker impulses away instead of falling over, so sweep
    upward rather than downward. It used to be a stale (12., 16.) that the recipe
    no longer uses -- a default that under-tests is worse than no default.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return
    robot = env.scene["robot"]
    velocity = robot.data.root_link_vel_w[env_ids].clone()
    speed = torch.empty(len(env_ids), device=env.device).uniform_(*speed_range)
    angle = torch.rand(len(env_ids), device=env.device) * (2 * torch.pi)
    velocity[:, 3] = speed * angle.cos()
    velocity[:, 4] = speed * angle.sin()
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def git_state() -> dict:
    import subprocess
    def run(*cmd):
        return subprocess.check_output(cmd, cwd=ROOT, text=True)
    try:
        return {"head": run("git", "rev-parse", "HEAD").strip(),
                "working_tree_status": run("git", "status", "--short"),
                "git_diff_sha256": hashlib.sha256(
                    subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)).hexdigest()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"head": None, "working_tree_status": None, "git_diff_sha256": None}


def evaluate_seed(args, seed: int) -> tuple[dict, list[dict], dict]:
    """Run one seed and return (aggregates, per-episode rows, provenance extras)."""
    configure_torch_backends()
    cfg, agent = task_registry.get_cfgs(MULTIMODE_TASK_ID, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    if args.start_mode == "force":
        cfg.events["reset_pose"] = EventTermCfg(func=force_ground_reset, mode="reset")
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = task_registry._task(MULTIMODE_TASK_ID).runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        # env.reset(seed=...) reaches mjlab's seed_rng, which seeds python, numpy,
        # torch and warp, so the sampled ground poses are reproducible per seed.
        obs, _ = env.reset(seed=seed)

        state = env.command_manager.get_term("standup")
        robot = env.scene["robot"]
        knee_ids, _ = robot.find_joints(("lf1", "rf1"), preserve_order=True)
        n, dt = env.num_envs, env.step_dt
        terrain = state.terrain_kind.clone()
        ground = state.ground_start.clone()
        command = state.command.clone()
        start_qpos = env.sim.data.qpos.clone()

        live = torch.ones(n, device=env.device, dtype=torch.bool)
        duration = torch.zeros(n, device=env.device)
        # Loose (task-native) criterion.
        weak = torch.zeros(n, device=env.device, dtype=torch.bool)
        weak_time = torch.full((n,), float("nan"), device=env.device)
        stable_at_t0 = torch.zeros(n, device=env.device, dtype=torch.bool)
        recover_seconds = torch.zeros(n, device=env.device)
        # Strict criterion: command height, 15 deg tilt, no base contact, both wheels.
        strict_run = torch.zeros(n, device=env.device)
        strict = torch.zeros(n, device=env.device, dtype=torch.bool)
        strict_time = torch.full((n,), float("nan"), device=env.device)
        hold_ok = torch.zeros(n, device=env.device)
        hold_elapsed = torch.zeros(n, device=env.device)
        # Rise evidence.
        start_height = torch.zeros(n, device=env.device)
        peak_height = torch.zeros(n, device=env.device)
        final_height = torch.zeros(n, device=env.device)
        start_knee = torch.zeros(n, 2, device=env.device)
        final_knee = torch.zeros(n, 2, device=env.device)
        wheels_at_end = torch.zeros(n, device=env.device, dtype=torch.bool)
        survived = torch.zeros(n, device=env.device, dtype=torch.bool)
        numerical = torch.zeros(n, device=env.device, dtype=torch.bool)
        terminal = torch.full((n,), -1, device=env.device, dtype=torch.long)
        # Mid-episode recover path: set when an env is tipped over, then measured
        # independently of the reset-time strict criterion above.
        hit = torch.zeros(n, device=env.device, dtype=torch.bool)
        knocked = torch.zeros(n, device=env.device, dtype=torch.bool)
        knock_strict_run = torch.zeros(n, device=env.device)
        knock_strict = torch.zeros(n, device=env.device, dtype=torch.bool)
        knock_time = torch.full((n,), float("nan"), device=env.device)

        captured, capture_trace = [], []
        first = True
        knockdown_step = None if args.knockdown_at is None else round(args.knockdown_at / dt)
        with torch.inference_mode():
            for step in range(args.steps):
                active = live.clone()
                if knockdown_step is not None and step == knockdown_step:
                    force_knockdown(env, live.nonzero().flatten(), tuple(args.knockdown_speed))
                    hit |= live
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                # `knocked` means the robot actually went down, not merely that an
                # impulse was delivered; a wheeled robot rolls weak impulses away.
                knocked |= hit & state.fallen

                height = state.height_above_ground
                cmd_height = state.command[:, 2]
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.asin(gravity[:, 0].clamp(-1, 1)).abs()
                base_contact = env.scene["base_contact"].data.force.norm(dim=-1).amax(-1) > 1.
                wheel_force = env.scene["wheel_contact"].data.force_history.norm(dim=-1).amax(dim=-1)
                both_wheels = (wheel_force > 5.).all(-1)

                duration += active.float() * dt
                recover_seconds += state.mode()[:, RECOVER_INDEX] * active.float() * dt
                if first:
                    stable_at_t0 |= active & (state.stable_time > 0.)
                    start_height = torch.where(active, height, start_height)
                    start_knee = torch.where(active[:, None], robot.data.joint_pos[:, knee_ids], start_knee)
                    first = False

                loose = state.stable_time >= STABLE_DURATION_S
                newly_weak = active & ~weak & loose
                weak_time = torch.where(newly_weak, duration, weak_time)
                weak |= active & loose

                strict_ok = ((height - cmd_height).abs() <= HEIGHT_TOLERANCE_M) & (roll <= TILT_LIMIT_RAD) & (
                    pitch <= TILT_LIMIT_RAD) & ~base_contact & both_wheels
                strict_run = torch.where(active & strict_ok, strict_run + dt, torch.where(active, 0., strict_run))
                newly_strict = active & ~strict & (strict_run >= STABLE_DURATION_S)
                strict_time = torch.where(newly_strict, duration, strict_time)
                strict |= active & (strict_run >= STABLE_DURATION_S)
                in_knock = knocked & active
                knock_strict_run = torch.where(in_knock & strict_ok, knock_strict_run + dt,
                                               torch.where(in_knock, 0., knock_strict_run))
                newly_knock = in_knock & ~knock_strict & (knock_strict_run >= STABLE_DURATION_S)
                # Only the knockdown path has a time origin to subtract. Without
                # --knockdown-at this line used to raise
                # TypeError: unsupported operand type(s) for -: 'Tensor' and 'NoneType',
                # which took down the whole plain (non-knockdown) evaluation.
                if args.knockdown_at is not None:
                    knock_time = torch.where(newly_knock,
                                             duration - args.knockdown_at, knock_time)
                knock_strict |= in_knock & (knock_strict_run >= STABLE_DURATION_S)
                # Fraction of the 15 s window after the first strict success.
                in_window = active & strict & (hold_elapsed < HOLD_WINDOW_S)
                hold_ok += (in_window & strict_ok).float() * dt
                hold_elapsed += in_window.float() * dt

                peak_height = torch.where(active, torch.maximum(peak_height, height), peak_height)
                final_height = torch.where(active, height, final_height)
                final_knee = torch.where(active[:, None], robot.data.joint_pos[:, knee_ids], final_knee)
                wheels_at_end = torch.where(active, both_wheels, wheels_at_end)
                numerical |= active & state.numerical_failure

                # Read the task's own bound: the classification must not call a real
                # physical failure "out_of_patch" because the radius was hardcoded here
                # (the comparison tools exclude that label from the regression veto).
                out_of_patch = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).abs().amax(-1) > OUT_OF_PATCH_RADIUS_M
                ended = active & (terminated | timeout)
                terminal = torch.where(ended & state.numerical_failure, 3, terminal)
                terminal = torch.where(ended & ~state.numerical_failure & out_of_patch, 2, terminal)
                terminal = torch.where(ended & ~state.numerical_failure & ~out_of_patch & terminated, 1, terminal)
                terminal = torch.where(ended & timeout & ~terminated, 0, terminal)
                survived |= ended & timeout & ~terminated

                if args.capture and step % 2 == 0:
                    idx = 0
                    if bool(active[idx]):
                        captured.append(env.sim.data.qpos[idx].cpu().numpy().copy())
                        capture_trace.append([float(duration[idx]), float(height[idx]),
                                              float(state.stable_time[idx]), bool(strict[idx].item())])

                live &= ~(terminated | timeout)
                if not live.any():
                    break
                done = terminated | timeout
                if done.any():
                    obs, _ = env.reset(env_ids=done.nonzero().flatten())
                if step % 200 == 199:
                    print(f"PROGRESS seed={seed} step={step + 1} finished={int((~live).sum())}/{n}", flush=True)

        terrain_cpu, ground_cpu, command_cpu = terrain.cpu().tolist(), ground.cpu().tolist(), command.cpu().tolist()
        weak_cpu, strict_cpu = weak.cpu().tolist(), strict.cpu().tolist()
        weak_t, strict_t = weak_time.cpu().tolist(), strict_time.cpu().tolist()
        hit_cpu = hit.cpu().tolist()
        knocked_cpu, knock_cpu = knocked.cpu().tolist(), knock_strict.cpu().tolist()
        knock_t = knock_time.cpu().tolist()
        rows = []
        for i in range(n):
            rows.append({
                "env": i, "terrain": TERRAIN_NAMES[terrain_cpu[i]], "ground_start": bool(ground_cpu[i]),
                "initial_command": command_cpu[i], "duration_s": float(duration[i]),
                "termination": TERMINATIONS[int(terminal[i]) + 1],
                "stable_at_t0": bool(stable_at_t0[i]),
                "weak_recovered": bool(weak_cpu[i]), "weak_recover_time_s": weak_t[i],
                "recover_mode_seconds": float(recover_seconds[i]),
                "strict_recovered": bool(strict_cpu[i]), "strict_recover_time_s": strict_t[i],
                "strict_hold_fraction": float(hold_ok[i] / max(hold_elapsed[i], dt)),
                "start_height_m": float(start_height[i]), "peak_height_m": float(peak_height[i]),
                "final_height_m": float(final_height[i]),
                "height_gain_m": float(peak_height[i] - start_height[i]),
                "start_knee_rad": start_knee[i].cpu().tolist(), "final_knee_rad": final_knee[i].cpu().tolist(),
                "both_wheels_at_end": bool(wheels_at_end[i]),
                "survived_20s": bool(survived[i]), "numerical_failure": bool(numerical[i]),
                "knockdown_hit": bool(hit_cpu[i]),
                "knocked_down": bool(knocked_cpu[i]),
                "knockdown_recovered": bool(knock_cpu[i]),
                "knockdown_recover_time_s": knock_t[i],
            })

        def aggregate(subset: list[dict]) -> dict:
            if not subset:
                return {"count": 0}
            def rate(key):
                return sum(row[key] for row in subset) / len(subset)
            times = sorted(row["strict_recover_time_s"] for row in subset if row["strict_recovered"])
            knocked_rows = [row for row in subset if row["knocked_down"]]
            knock_times = sorted(row["knockdown_recover_time_s"] for row in knocked_rows
                                 if row["knockdown_recovered"])
            knock_ok = sum(row["knockdown_recovered"] for row in knocked_rows)
            return {
                "count": len(subset),
                "knockdown_hits": sum(row["knockdown_hit"] for row in subset),
                "knocked_down": len(knocked_rows),
                "knockdown_recovered": knock_ok,
                "knockdown_recovery_rate": knock_ok / len(knocked_rows) if knocked_rows else None,
                "median_knockdown_recover_time_s": (
                    float(np.median(knock_times)) if knock_times else None),
                "weak_recovered": sum(row["weak_recovered"] for row in subset),
                "weak_recovery_rate": rate("weak_recovered"),
                "strict_recovered": sum(row["strict_recovered"] for row in subset),
                "strict_recovery_rate": rate("strict_recovered"),
                "stable_at_t0": sum(row["stable_at_t0"] for row in subset),
                "stable_at_t0_rate": rate("stable_at_t0"),
                "mean_recover_mode_seconds": sum(row["recover_mode_seconds"] for row in subset) / len(subset),
                "median_strict_recover_time_s": float(np.median(times)) if times else None,
                "p90_strict_recover_time_s": float(np.percentile(times, 90)) if times else None,
                "mean_strict_hold_fraction": sum(row["strict_hold_fraction"] for row in subset) / len(subset),
                "mean_height_gain_m": sum(row["height_gain_m"] for row in subset) / len(subset),
                "mean_start_height_m": sum(row["start_height_m"] for row in subset) / len(subset),
                "mean_final_height_m": sum(row["final_height_m"] for row in subset) / len(subset),
                "both_wheels_at_end": sum(row["both_wheels_at_end"] for row in subset),
                "survived_20s": sum(row["survived_20s"] for row in subset),
                "numerical_failures": sum(row["numerical_failure"] for row in subset),
                "termination_counts": {name: sum(row["termination"] == name for row in subset)
                                       for name in TERMINATIONS},
            }

        signature = hashlib.sha256(
            np.asarray(terrain_cpu, dtype=np.int32).tobytes()
            + np.asarray(ground_cpu, dtype=np.bool_).tobytes()
            + np.asarray(command_cpu, dtype=np.float32).tobytes()
            + start_qpos.cpu().numpy().astype(np.float32).tobytes()).hexdigest()

        ground_rows = [row for row in rows if row["ground_start"]]
        flat_ground = [row for row in ground_rows if row["terrain"] == TERRAIN_NAMES[FLAT_INDEX]]
        aggregates = {
            "all": aggregate(rows),
            "ground_start": aggregate(ground_rows),
            "ground_start_flat": aggregate(flat_ground),
            "by_terrain": {name: aggregate([row for row in ground_rows if row["terrain"] == name])
                           for name in TERRAIN_NAMES},
        }
        extras = {"initial_state_signature": signature, "num_envs": n,
                  "physics_dt_s": env.physics_dt, "policy_dt_s": dt,
                  "ground_start_envs": len(ground_rows), "ground_start_flat_envs": len(flat_ground),
                  "knockdown_at_s": args.knockdown_at,
                  "knockdown_speed_range": list(args.knockdown_speed)}
        if args.capture:
            extras["capture_env"] = 0
        return {"aggregates": aggregates, "extras": extras}, rows, {"captured": captured,
                                                                    "trace": capture_trace}
    finally:
        env.close()


def main() -> None:
    cfg, _ = task_registry.get_cfgs(MULTIMODE_TASK_ID, play=True)
    policy_dt = cfg.sim.mujoco.timestep * cfg.decimation
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="summary JSON path")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--start-mode", choices=("force", "sample"), default="force",
                        help="force: every env starts on the ground; sample: the task's configured draw")
    parser.add_argument("--terrain", choices=("flat", "all"), default="flat",
                        help="which terrain the headline numbers are taken from")
    parser.add_argument("--knockdown-at", type=float, default=None,
                        help="seconds into the episode to tip every live env over, then "
                             "measure the mid-episode recover path (default: off)")
    parser.add_argument("--knockdown-speed", type=float, nargs=2,
                        default=KNOCKDOWN_ANGULAR_SPEED_RANGE,
                        metavar=("MIN", "MAX"),
                        help="root angular-velocity impulse range in rad/s for "
                             "--knockdown-at (default: the training range)")
    parser.add_argument("--steps", type=int, default=round(20. / policy_dt),
                        help=f"max control steps (default: 20 s at {1. / policy_dt:g} Hz)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture", action="store_true", help="save env 0's first-episode qpos")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    per_seed, headline_key = {}, "ground_start_flat" if args.terrain == "flat" else "ground_start"
    for seed in seeds:
        (result, rows, capture) = evaluate_seed(args, seed)
        aggregates, extras = result["aggregates"], result["extras"]
        per_seed[str(seed)] = {**extras, "aggregates": aggregates}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stem = args.output.with_suffix("")
        (stem.parent / f"{stem.name}_episodes_seed{seed}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        if args.capture and capture["captured"]:
            np.savez_compressed(stem.parent / f"{stem.name}_trajectory_seed{seed}.npz",
                                qpos=np.asarray(capture["captured"]),
                                trace=np.asarray(capture["trace"]))
        print(f"SEED {seed} " + json.dumps({k: v for k, v in aggregates[headline_key].items()
                                            if k != "termination_counts"}, ensure_ascii=False), flush=True)

    def cross(key: str) -> dict:
        values = [per_seed[str(seed)]["aggregates"][headline_key].get(key) for seed in seeds]
        values = [v for v in values if v is not None]
        if not values:
            return {"values": []}
        return {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values))}

    summary = {
        "time": datetime.now().astimezone().isoformat(),
        "task": MULTIMODE_TASK_ID,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "script": str(Path(__file__).resolve()), "script_sha256": script_sha,
        **git_state(),
        "config": {"num_envs": args.num_envs, "start_mode": args.start_mode, "terrain": args.terrain,
                   "steps": args.steps, "policy_dt_s": policy_dt,
                   "max_episode_s": args.steps * policy_dt, "play": True,
                   "domain_randomization": False, "obs_noise_or_delay": False},
        "criteria": {
            "weak": "state.stable_time >= 0.5 (task-native recovered flag)",
            "strict": f"|h - cmd_h| <= {HEIGHT_TOLERANCE_M} m and |roll|,|pitch| <= 15 deg "
                      f"and no base contact and both wheels, held >= {STABLE_DURATION_S} s",
        },
        "headline_group": headline_key,
        "seeds": seeds,
        "per_seed": per_seed,
        "cross_seed": {key: cross(key) for key in
                       ("weak_recovery_rate", "strict_recovery_rate", "stable_at_t0_rate",
                        "mean_recover_mode_seconds", "median_strict_recover_time_s",
                        "mean_strict_hold_fraction", "survived_20s")},
        "episodes_file_pattern": f"{args.output.with_suffix('').name}_episodes_seed<seed>.json",
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("SUMMARY", json.dumps(summary["cross_seed"], ensure_ascii=False), flush=True)
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
