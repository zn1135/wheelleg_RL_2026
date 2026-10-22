#!/usr/bin/env python3
"""Measure flat-ground linear-velocity and yaw-rate tracking of a multimode checkpoint.

Scope: the 2026-10-17 flat tracking phase. Runs the `chuanliantui` task in the play
configuration (no domain randomization, no observation noise or delay) and replaces
the sampled commands with a fixed grid: one constant command per environment for the
whole episode, so every row of the report is a commanded operating point instead of a
random draw.

Not a training acceptance by itself. It measures one checkpoint in one physics backend
with constant commands, after a settle window. It does not replace the paired
stand-up/recovery acceptance (`assess_multimode_knockdown_recovery.py`), the separate
CPU MuJoCo replay, or a hardware result.

Reading the report
------------------
Per-env values are averaged only over steps after SETTLE_S and only while the env is
still live. A fall or any other termination freezes that env and is recorded; the env
is not restarted inside the measurement, so each environment contributes exactly one
episode and the success rate is not inflated by reruns. The measurement window is shorter
than the task's own episode length, so ``survived`` means the env never terminated inside
the window (there is no timeout to reach); ``ended_step`` records when a terminating env stopped.

Criteria (per env, all must hold):
  tracked_vx       mean |vx_cmd - vx|      <= VX_TOL after the settle window
  tracked_yaw      mean |yaw_cmd - omega_z| <= YAW_TOL
  tracked_height   mean |h_cmd - h|        <= HEIGHT_TOL
  upright          max tilt                <= TILT_TOL_DEG
  no_base_contact  no base_link contact force above 1 N
  survived         ran to DURATION_S without a physical or numerical termination
  numerical_ok     no non-finite state or runaway velocity

The tolerance defaults here are the single source of truth for the criteria; the
documented protocol is docs/interfaces/chuanliantui-huananhu-multimode-tracking-v1.md
and the arithmetic is checked offline by scripts/agent/check_multimode_tracking.py.

Jump requests are suppressed for the measurement (jump_cooldown is pinned high): a
takeoff would add a disturbance that is not part of the tracking criterion. mode_jump
is reported so a suppressed request cannot pass unnoticed.

Usage (repo root):
  mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
    --checkpoint <model_*.pt> --output artifacts/<new-name>/result.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mjlab_training" / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402
from ct_mjlab.envs import task_registry  # noqa: E402
from ct_mjlab.envs.chuanliantui.huananhu_poses import sample_reset_state  # noqa: E402
from ct_mjlab.envs.chuanliantui.multimode_mdp import MODE_NAMES, TERRAIN_NAMES  # noqa: E402

TASK = "chuanliantui"
JUMP_INDEX = MODE_NAMES.index("jump")
FLAT_INDEX = TERRAIN_NAMES.index("flat")

SETTLE_S = 2.0
DURATION_S = 12.0
VX_TOL = 0.30
YAW_TOL = 0.30
HEIGHT_TOL = 0.03
TILT_TOL_DEG = 25.0
BASE_CONTACT_N = 1.0
JUMP_COOLDOWN_PIN = 1.0e9
DEFAULT_VX = "0,-0.5,0.5,-1.0,1.0,-1.5,1.5"
DEFAULT_YAW = "0,-0.5,0.5,-1.0,1.0,-1.5,1.5"
DEFAULT_HEIGHT = "0.27,0.34"


def episode_verdict(*, observed_s, vx_err_abs_sum, vx_err_signed_sum, vy_abs_sum,
                    yaw_err_abs_sum, yaw_err_signed_sum, height_err_abs_sum,
                    tilt_max_deg, base_contact_s, numerical_failure, survived,
                    vx_tol=VX_TOL, yaw_tol=YAW_TOL, height_tol=HEIGHT_TOL,
                    tilt_tol_deg=TILT_TOL_DEG):
    """Turn one env's measured sums into means, per-condition checks and a verdict.

    Plain floats in, plain dict out: the offline check imports this function so the
    arithmetic cannot drift away from what the assessment enforces.
    """
    if observed_s <= 0.0:
        raise ValueError(
            "observed_s must be positive: an env with no live measured step has no verdict")
    means = {
        "vx_error_m_s": vx_err_abs_sum / observed_s,
        "vx_bias_m_s": vx_err_signed_sum / observed_s,
        "vy_abs_m_s": vy_abs_sum / observed_s,
        "yaw_error_rad_s": yaw_err_abs_sum / observed_s,
        "yaw_bias_rad_s": yaw_err_signed_sum / observed_s,
        "height_error_m": height_err_abs_sum / observed_s,
        "tilt_max_deg": float(tilt_max_deg),
        "base_contact_s": float(base_contact_s),
        "observed_s": float(observed_s),
    }
    checks = {
        "tracked_vx": means["vx_error_m_s"] <= vx_tol,
        "tracked_yaw": means["yaw_error_rad_s"] <= yaw_tol,
        "tracked_height": means["height_error_m"] <= height_tol,
        "upright": means["tilt_max_deg"] <= tilt_tol_deg,
        "no_base_contact": means["base_contact_s"] <= 0.0,
        "survived": bool(survived),
        "numerical_ok": not bool(numerical_failure),
    }
    return {
        "means": means,
        "checks": checks,
        "failed_checks": sorted(name for name, ok in checks.items() if not ok),
        "passed": all(checks.values()),
        "tolerances": {"vx_m_s": vx_tol, "yaw_rad_s": yaw_tol, "height_m": height_tol,
                       "tilt_deg": tilt_tol_deg, "base_contact_n": BASE_CONTACT_N},
    }


def force_normal_reset(env, env_ids):
    """Reset from the normal standing pose only, never from the ground pose bank.

    The tracking measurement needs an upright start; the ground branch would spend the
    settle window standing up instead of tracking.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    state = env.command_manager.get_term("standup")
    ground = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    state.ground_start[env_ids] = ground
    sample_reset_state(env, env_ids, ground)
    if hasattr(env, "_huananhu_rewards"):
        env._huananhu_rewards.step = -1


def pin_commands(env, state, command, env_ids=None):
    """Pin the fixed command grid, and make every later resample produce it too.

    The term draws commands inside compute() both on its own timer and on every reset, which
    would replace the grid for one control step and hand the actor a command that is not part
    of the measurement. Instead of chasing each draw, the term's own `_resample_command` is
    replaced for this run by a function that writes the grid for exactly the envs being
    resampled, so reset-time and timer-time draws both land on the grid. The command buffer is
    also written immediately, because an observation rebuilt between the reset and the next
    compute() would otherwise still carry the drawn command.
    """
    ids = slice(None) if env_ids is None else env_ids

    def pinned_resample(resample_ids):
        state.sampled[resample_ids] = command[resample_ids]
        state.standing[resample_ids] = False
        state.heading[resample_ids] = False
        state.special_mode[resample_ids] = -1
        state.ground_start[resample_ids] = False
        state.ground_command[resample_ids] = 0.
        state.zero_hold[resample_ids] = False  # 固定行走命令不受训练的整回合零速采样覆盖。
        # Jump requests are suppressed: a takeoff is a disturbance the tracking criterion
        # does not ask about, and `can_jump` requires jump_cooldown <= 0.
        state.jump_cooldown[resample_ids] = JUMP_COOLDOWN_PIN

    # Instance attribute shadowing the class method: the term calls `self._resample_command(ids)`.
    state._resample_command = pinned_resample
    pinned_resample(ids)
    state._command[ids] = command[ids]


def assert_actor_command(obs, command, step):
    """Fail loudly unless the policy's own command slots are the pinned grid.

    Slots 0:3 are [vx, 0, yaw_rate] and slot 3 is height*5 (v14_observations.basic), so this
    checks the numbers the actor actually consumes instead of only the command term's state.
    """
    actor = obs['actor']
    expected = torch.cat((command[:, :1], torch.zeros_like(command[:, :1]), command[:, 1:2],
                          command[:, 2:3].clamp(0., 1.) * 5.), dim=-1)
    if not torch.allclose(actor[:, :4], expected, atol=1e-5):
        raise AssertionError(f'actor command slots differ from the grid at step {step}: '
                             f'{actor[0, :4].tolist()} vs {expected[0].tolist()}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    def run(*cmd):
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    try:
        return {"head": run("git", "rev-parse", "HEAD").strip(),
                "working_tree_status": run("git", "status", "--short"),
                "git_diff_sha256": hashlib.sha256(
                    subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=REPO_ROOT)).hexdigest()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"head": None, "working_tree_status": None, "git_diff_sha256": None}


def parse_list(text, cast):
    return [cast(item) for item in str(text).split(",") if item.strip() != ""]


def build_combos(vx_values, yaw_values, height_values):
    return [(vx, yaw, height) for height in height_values
            for yaw in yaw_values for vx in vx_values]


def evaluate_seed(args, seed, combos, *, record_dir=None):
    """Run one seed. Returns (per-env rows, extras)."""
    configure_torch_backends()
    cfg, agent = task_registry.get_cfgs(TASK, play=True)
    num_envs = len(combos)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    # Never resample: the command manager resamples when its timer expires, which would
    # replace the pinned grid for one control step and feed the actor a random command.
    # The per-step assertion below still checks every command the manager produces.
    cfg.commands["standup"].resampling_time_range = (1e9, 1e9)
    cfg.events["reset_pose"] = EventTermCfg(func=force_normal_reset, mode="reset")
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = task_registry._task(TASK).runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        command = torch.tensor(combos, device=env.device, dtype=torch.float32)
        obs, _ = env.reset(seed=seed)

        state = env.command_manager.get_term("standup")
        robot = env.scene["robot"]
        hip_ids, _ = robot.find_bodies(('lf0', 'rf0'), preserve_order=True)
        wheel_ids, _ = robot.find_bodies(('lfwheel', 'rfwheel'), preserve_order=True)
        # reset() samples commands, so the observation it returned carries a random one. Pin the
        # grid, then flush with one zero-action step: the observation manager returns its cached
        # group buffers outside a step, so simply recomputing it would still show the reset draw.
        # That single step is harmless (the robot is standing) and its 20 ms fall inside the
        # settle window, so nothing measured is affected.
        pin_commands(env, state, command)
        obs, _, _, _, _ = env.step(torch.zeros((num_envs, 6), device=env.device))
        dt = env.step_dt
        settle_steps = round(SETTLE_S / dt)
        total_steps = round(DURATION_S / dt)

        names = ("vx_abs", "vx_signed", "vy_abs", "yaw_abs", "yaw_signed", "height_abs",
                 "observed", "base_contact")
        acc = {name: torch.zeros(num_envs, device=env.device) for name in names}
        tilt_max = torch.zeros(num_envs, device=env.device)
        jump_steps = torch.zeros(num_envs, device=env.device)
        live = torch.ones(num_envs, device=env.device, dtype=torch.bool)
        ended = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        ended_step = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        numerical = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
        terrain = state.terrain_kind.clone()
        leg_sum = torch.zeros((num_envs, 2), device=env.device)
        leg_min = torch.full_like(leg_sum, float('inf'))
        leg_max = torch.full_like(leg_sum, -float('inf'))
        recorded_qpos, recorded_lengths = [], []
        if record_dir is not None:
            import mujoco
            record_dir.mkdir(parents=True, exist_ok=False)
            mujoco.mj_saveModel(env.sim.mj_model, str(record_dir / 'scene.mjb'), None)

        with torch.inference_mode():
            for step in range(total_steps):
                pin_commands(env, state, command)
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                # Assert on the observation this step produced, which is the input the next
                # action is taken on: it is computed inside the step, after the pin, so it is the
                # actor's real command input. Recomputing outside a step returns a cached buffer.
                assert_actor_command(obs, command, step)
                measured = live & (step + 1 > settle_steps)

                linear = robot.data.root_link_lin_vel_b
                angular = robot.data.root_link_ang_vel_b
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.asin(gravity[:, 0].clamp(-1, 1)).abs()
                tilt_deg = torch.rad2deg(torch.maximum(roll, pitch))
                # Use the contact sensor's 8-step force history, as the reward does: a
                # base that touches down inside a physics substep and is already off the
                # ground at the control step would keep the single-sample test at zero.
                base_history = env.scene["base_contact"].data.force_history
                base_contact = base_history.norm(dim=-1).amax(dim=(1, 2)) > BASE_CONTACT_N

                vx_error = command[:, 0] - linear[:, 0]
                yaw_error = command[:, 1] - angular[:, 2]
                height_error = state.height_above_ground - command[:, 2]
                mask = measured.float()
                # Measure actual hip-to-wheel length in the body XZ plane;
                # neither the height command nor world base height is leg length.
                delta = robot.data.body_link_pos_w[:, wheel_ids] - robot.data.body_link_pos_w[:, hip_ids]
                quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
                leg = quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, 2, 3)
                lengths = torch.linalg.vector_norm(leg[:, :, (0, 2)], dim=-1)
                if record_dir is not None:
                    recorded_qpos.append(env.sim.data.qpos.detach().cpu().numpy().copy())
                    recorded_lengths.append(lengths.detach().cpu().numpy().copy())
                leg_sum += lengths * mask[:, None] * dt
                leg_min = torch.where(measured[:, None], torch.minimum(leg_min, lengths), leg_min)
                leg_max = torch.where(measured[:, None], torch.maximum(leg_max, lengths), leg_max)
                acc["vx_abs"] += vx_error.abs() * mask * dt
                acc["vx_signed"] += (linear[:, 0] - command[:, 0]) * mask * dt
                acc["vy_abs"] += linear[:, 1].abs() * mask * dt
                acc["yaw_abs"] += yaw_error.abs() * mask * dt
                acc["yaw_signed"] += (angular[:, 2] - command[:, 1]) * mask * dt
                acc["height_abs"] += height_error.abs() * mask * dt
                acc["observed"] += mask * dt
                acc["base_contact"] += (base_contact & measured).float() * dt
                jump_steps += (state.mode()[:, JUMP_INDEX] > 0).float() * mask * dt
                tilt_max = torch.where(measured, torch.maximum(tilt_max, tilt_deg), tilt_max)

                # The actor and the rewards must have seen the grid, never a resample.
                if not torch.allclose(state.command[:, :2], command[:, :2]):
                    raise AssertionError(
                        f'command drifted from the grid at step {step}: '
                        f'{state.command[0].tolist()} vs {command[0].tolist()}')
                numerical |= live & state.numerical_failure
                # mjlab forbids stepping a terminated env again with auto_reset=False, so every
                # env that just ended must be reset -- including one that already left the
                # statistics and ended again inside its ignored episode. Only the first
                # termination per env enters the statistics.
                newly_done = terminated | timeout
                first_done = newly_done & live
                ended |= first_done
                ended_step = torch.where(first_done, torch.full_like(ended_step, step + 1), ended_step)
                live &= ~first_done
                if not bool(live.any()):
                    break
                if bool(newly_done.any()):
                    obs, _ = env.reset(env_ids=newly_done.nonzero().flatten())
                    # These envs have already left the statistics, so the one step the loop takes
                    # on this reset observation cannot bias the measurement; the next iteration
                    # pins again and the assertion checks the observation that step produces.
                    pin_commands(env, state, command)

        rows = []
        for index, (vx_cmd, yaw_cmd, height_cmd) in enumerate(combos):
            if float(acc["observed"][index]) <= 0.0:
                # The env ended inside the settle window, so there is nothing to average.
                # Report it as a failure with empty means instead of letting the verdict
                # guard abort the whole report.
                empty = {key: None for key in ("vx_error_m_s", "vx_bias_m_s", "vy_abs_m_s",
                                               "yaw_error_rad_s", "yaw_bias_rad_s", "height_error_m",
                                               "tilt_max_deg", "base_contact_s", "observed_s")}
                verdict = {
                    "means": empty,
                    "checks": {"tracked_vx": False, "tracked_yaw": False, "tracked_height": False,
                               "upright": False, "no_base_contact": False, "survived": False,
                               "numerical_ok": not bool(numerical[index])},
                    "failed_checks": ["no_measured_steps"], "passed": False,
                    "tolerances": {"vx_m_s": VX_TOL, "yaw_rad_s": YAW_TOL, "height_m": HEIGHT_TOL,
                                   "tilt_deg": TILT_TOL_DEG, "base_contact_n": BASE_CONTACT_N},
                }
            else:
                verdict = episode_verdict(
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
                    survived=not bool(ended[index]),
                )
            rows.append({
                "env": index, "seed": seed, "terrain": TERRAIN_NAMES[int(terrain[index])],
                "command": {"vx_m_s": vx_cmd, "yaw_rad_s": yaw_cmd, "height_m": height_cmd},
                "jump_seconds": float(jump_steps[index]),
                "ended_step": int(ended_step[index]),
                "leg_length_xz_m": {
                    side: {"mean": float(leg_sum[index, j] / acc['observed'][index]),
                           "min": float(leg_min[index, j]), "max": float(leg_max[index, j])}
                    if float(acc['observed'][index]) > 0 else None
                    for j, side in enumerate(('left', 'right'))},
                **verdict,
            })
        extras = {
            "seed": seed, "num_envs": num_envs, "settle_s": SETTLE_S, "duration_s": DURATION_S,
            "steps": total_steps, "measured_steps": max(total_steps - settle_steps, 0),
            "terrain_families": sorted({row["terrain"] for row in rows}),
            "flat_only": all(row["terrain"] == TERRAIN_NAMES[FLAT_INDEX] for row in rows),
            "jump_seconds_total": float(sum(row["jump_seconds"] for row in rows)),
            "survived": int(sum(row["checks"]["survived"] for row in rows)),
            "numerical_failures": int(sum(not row["checks"]["numerical_ok"] for row in rows)),
        }
        if record_dir is not None:
            np.savez_compressed(record_dir / 'trajectory.npz',
                                qpos=np.asarray(recorded_qpos),
                                leg_lengths=np.asarray(recorded_lengths), command=np.asarray(combos),
                                time_s=(np.arange(len(recorded_qpos)) + 1) * dt)
        return rows, extras
    finally:
        env.close()


def aggregate(rows):
    if not rows:
        return {"count": 0}
    def rate(check):
        return sum(row["checks"][check] for row in rows) / len(rows)
    def mean_of(key):
        values = [row["means"][key] for row in rows if row["means"].get(key) is not None]
        return float(np.mean(values)) if values else None
    return {
        "count": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "success_rate": sum(row["passed"] for row in rows) / len(rows),
        "rate_tracked_vx": rate("tracked_vx"),
        "rate_tracked_yaw": rate("tracked_yaw"),
        "rate_tracked_height": rate("tracked_height"),
        "rate_upright": rate("upright"),
        "rate_no_base_contact": rate("no_base_contact"),
        "rate_survived": rate("survived"),
        "mean_vx_error_m_s": mean_of("vx_error_m_s"),
        "mean_vx_bias_m_s": mean_of("vx_bias_m_s"),
        "mean_yaw_error_rad_s": mean_of("yaw_error_rad_s"),
        "mean_yaw_bias_rad_s": mean_of("yaw_bias_rad_s"),
        "mean_height_error_m": mean_of("height_error_m"),
        "mean_lateral_speed_m_s": mean_of("vy_abs_m_s"),
        "mean_tilt_max_deg": mean_of("tilt_max_deg"),
        "failure_checks": sorted({name for row in rows for name in row["failed_checks"]}),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="summary JSON path")
    parser.add_argument("--vx", default=DEFAULT_VX, help=f"comma-separated m/s grid (default {DEFAULT_VX})")
    parser.add_argument("--yaw", default=DEFAULT_YAW, help=f"comma-separated rad/s grid (default {DEFAULT_YAW})")
    parser.add_argument("--height", default=DEFAULT_HEIGHT,
                        help=f"comma-separated m grid (default {DEFAULT_HEIGHT})")
    parser.add_argument("--envs-per-combo", type=int, default=1,
                        help="repeated envs per command point (identical command, different reset draw)")
    parser.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,43")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-dir", type=Path, default=None,
                        help="optional new directory for recorded states and render model")
    parser.add_argument("--min-success-rate", type=float, default=None,
                        help="exit non-zero when the pooled success rate is below this value")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.envs_per_combo < 1:
        raise SystemExit("--envs-per-combo must be at least 1")
    combos = build_combos(parse_list(args.vx, float), parse_list(args.yaw, float),
                          parse_list(args.height, float))
    if args.envs_per_combo > 1:
        combos = [combo for combo in combos for _ in range(args.envs_per_combo)]
    seeds = parse_list(args.seeds, int)

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    per_seed, all_rows = {}, []
    for seed in seeds:
        rows, extras = evaluate_seed(args, seed, combos, record_dir=(
            None if args.record_dir is None else args.record_dir / f'seed{seed}'))
        per_seed[str(seed)] = {"extras": extras, "aggregate": aggregate(rows)}
        all_rows.extend(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stem = args.output.with_suffix("")
        (stem.parent / f"{stem.name}_envs_seed{seed}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        print(f"SEED {seed} " + json.dumps(per_seed[str(seed)]["aggregate"], ensure_ascii=False), flush=True)

    pools = {
        "all": all_rows,
        "standing_vx0_yaw0": [row for row in all_rows
                              if row["command"]["vx_m_s"] == 0.0 and row["command"]["yaw_rad_s"] == 0.0],
        "moving": [row for row in all_rows
                   if row["command"]["vx_m_s"] != 0.0 or row["command"]["yaw_rad_s"] != 0.0],
    }
    for magnitude in (0.5, 1.0, 1.5):
        pools[f"vx_abs_{magnitude}"] = [row for row in all_rows
                                        if abs(row["command"]["vx_m_s"]) == magnitude]
        pools[f"yaw_abs_{magnitude}"] = [row for row in all_rows
                                         if abs(row["command"]["yaw_rad_s"]) == magnitude]
    groups = {name: aggregate(rows) for name, rows in pools.items()}
    # Carry the per-env verdict fields too: without them a reader of this summary alone cannot
    # tell which sample failed or when it stopped (the raw rows are in the _envs_seed*.json).
    by_command = [
        {"env": row["env"], "seed": row["seed"], "terrain": row["terrain"],
         "ended_step": row["ended_step"], "jump_seconds": row["jump_seconds"],
         "command": row["command"], "passed": row["passed"],
         "failed_checks": row["failed_checks"], **row["means"], "checks": row["checks"]}
        for row in all_rows]

    summary = {
        "kind": "flat tracking measurement in the play configuration; no training",
        "task": TASK,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "script": str(Path(__file__).resolve()), "script_sha256": script_sha,
        **git_state(),
        "config": {
            "play": True, "domain_randomization": False, "obs_noise_or_delay": False,
            "start_pose": "normal only", "commands": "fixed per env for the whole episode",
            "jump_requests": "suppressed (jump_cooldown pinned)",
            "grid": {"vx_m_s": parse_list(args.vx, float), "yaw_rad_s": parse_list(args.yaw, float),
                     "height_m": parse_list(args.height, float), "envs_per_combo": args.envs_per_combo},
            "seeds": seeds, "settle_s": SETTLE_S, "duration_s": DURATION_S,
        },
        "criteria": {
            "note": "tolerances are the module constants of this script and are re-checked by "
                    "scripts/agent/check_multimode_tracking.py",
            "vx_err_abs_m_s": VX_TOL, "yaw_err_abs_rad_s": YAW_TOL, "height_err_abs_m": HEIGHT_TOL,
            "tilt_max_deg": TILT_TOL_DEG, "base_contact_n": BASE_CONTACT_N,
        },
        "per_seed": per_seed,
        "groups": groups,
        "by_command": by_command,
        "scope": "one checkpoint, one physics backend, constant commands; not a training acceptance",
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("GROUPS " + json.dumps({name: {"count": group.get("count"),
                                         "success_rate": group.get("success_rate")}
                                  for name, group in groups.items()}, ensure_ascii=False), flush=True)
    print(f"WROTE {args.output}", flush=True)

    if args.min_success_rate is not None:
        observed = groups["all"]["success_rate"]
        if observed < args.min_success_rate:
            raise SystemExit(f"跟踪成功率 {observed:.4f} 低于门槛 {args.min_success_rate:.4f}")


if __name__ == "__main__":
    main()
