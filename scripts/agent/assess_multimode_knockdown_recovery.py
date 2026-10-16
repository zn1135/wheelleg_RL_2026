#!/usr/bin/env python3
"""中途倒地恢复验收：结束前连续 1 秒保持双轮平衡（v3）。

当前成功看 `final_balanced`：完整观察到预定结束，末段连续 1 秒双轮承重、机身不撑地、
倾角 ≤35°、横滚/俯仰角速度 ≤0.5 rad/s；允许滚动、转向与少量腿部接触，
双轮仍须承担至少 80% 重量，不要求高度跟踪或限时起立。
`last5_*` 与 `recovered` 保留原 v2 定义，只作历史对照，不再决定当前成功与否。
以下 v2 说明只解释保留字段。

与 `scripts/mjlab.sh eval` 的分工：eval 用锁存的"曾严格起立"做地面起立统计，本入口用
验收 v2 的**同时成立**组合条件，并区分倒地类型。两者口径不同，不可互相替代。

一切从冲量送达时刻起算（2026-10-16 修正）
----------------------------------------
脚本从回合开始就记第一次 `fallen`，而 50% 回合是地面初态——复位起立本身就能满足 v2，
且 `recovered`／`recover_time` 是锁存的、只在首次恢复时赋值。修正前测到的 113 次
"恢复"里有 107 次发生在冲量之前，`kd-acceptance-4022` 的 505 次里有 424 次如此，即
这些字段根本没在回答"被击倒后能否恢复"。现在冲量送达时清零全部倒地／恢复／累计器，
逐环境记 `hit_time_s`，`recovered`／`recover_time_s`／`refell` 均自该时刻起算；
`observed_after_hit_s` 给出冲量之后实际观察了多久。未送达冲量的回合（`hit=false`）
其冲量后字段为空，不应计入分母。

组合条件（逐条同时成立）
------------------------
轮端接触；每轮向上支撑 ≥整机重量 10% 且 ≥5 N；双轮合计 80%～120%；非轮部件 ≤5%；
机身接触力 ≤1 N；机身倾角 ≤15°；根线速度 ≤0.15 m/s；根角速度 ≤0.5 rad/s；
全程数值有限。

与验收 v2 的关系（2026-10-16 复核）
----------------------------------
验收 v2 的成功判据是"末 5 秒至少 95% 的控制步满足组合条件"，并明确写着"曾连续
0.5 秒成功仅表示曾经恢复，不能替代最终稳定平衡"。据此：

* `last5_pure_v2_fraction ≥ 0.95` 才是文档口径的**稳定双轮平衡**（文档类别 2）；
  `last5_v2_fraction` 另加高度 ±3 cm，对应**高度跟踪**（类别 3）。文档要求三类分开
  报告，这两个字段不要混用。
* `recovered` 是文档说的**"曾恢复"**——连续 0.5 秒满足组合条件，是能力标记，**不是
  验收通过**；报"恢复率"时必须说明它低于文档判据。
* `RECOVER_DEADLINE_S`（默认 2.0 s）是**本脚本的度量构造，验收 v2 里没有定义**，
  `--recover-deadline` 可覆盖（放宽用于观察"慢起立"）；跌倒窗口默认跟随训练配置的
  `recovery_window_s`，`--recovery-window` 可覆盖（放到不小于回合剩余长度即等于关掉
  窗口）。
* 配对否决协议（原成功样本任一失效即判退化）属于 `eval` 的 64 初态路径；本入口是
  能力探测，不是验收协议本身。

设计要点
--------
* 环境按**轮转**分配强度档（`combos[i % 组合数]`），不是按 `per_combo` 分块。地形按
  环境索引分块（128 环境 63/26/20/19、192 环境 95/39/29/29，平地连成一片），分块
  分配会让每档强度落在单一地形族上、量到的是地形而不是强度；轮转后每档横跨四种
  地形，并逐环境记录 `terrain_kind`。每个 seed 各建一次环境（复位 RNG 是全局的，
  无法在一个环境里换 seed）。
* 命令全程强制 vx=0、yaw=0、height=0.30 m：每拍改写 `sampled`／`ground_command`／
  `standing`，让命令管理器自己算出零速命令，而不是事后覆盖 `_command`
  （后者会被下一拍的 `advance()` 覆盖掉）。
* 击倒用显式方向与强度，方向在 8 个等间隔水平方向间均衡分配；但与训练事件一样
  **只打已经站稳 0.5 秒的机器人**（`(~fallen) & (stable_time >= 0.5)`），到点未满足
  就挂起等待，所以 `knock_at` 只是到期时刻、真正的送达时刻逐环境记在 `hit_time_s`；
  `eligible` 记到期、`hit` 记确实送达，未送达者不从分母剔除而是单独计数。
* 倒地类型按轨迹分类，不用单一 `fallen` 标志；现行为 `contact_only`（只机身触地）／
  `tilted`（倾角 >40°）／`tilted_deep`（>60°）／`inverted`（>90°，且是**全回合历史
  OR**——只说明曾越过 90°，不要求末态仍倒扣）。末态请读 `final_tilt_deg`（最后一个
  存活拍的倾角），它不受这个标签影响。
* 非轮部件支撑用新增的全身接触传感器减去两轮支撑（`undesired_contact` 只覆盖机身与
  腿连杆，会漏掉气弹簧件，故不用它当判据）。
* 末段平衡：`last5_pure_v2_fraction` 是最后 5 秒里纯 v2 组合条件成立的时间占比，
  `last5_observed_s` 是其实际观测时长——回合提前终止时这个分母远小于 5 秒，读比例
  时必须一起看；`last5_v2_fraction` 另加了高度 ±3 cm，两者不要混用。
* 数值与配对：输出按 seed／强度／时刻／初态／地形的分组统计，以及"按旧 0.4 秒规则
  是否本会终止"，用来区分窗口救回与策略真会起。

已知未修项：`termination="physical"` 笼统表示非超时终止，未区分窗口到期／越界／数值，
需配合 `observed_after_hit_s` 判断是否为窗口到期。

这是有限测量，不训练，不改冻结配方。用法（仓库根目录）：

  mjlab_training/.venv/bin/python scripts/agent/assess_multimode_knockdown_recovery.py \
    --checkpoint <model_*.pt> --envs-per-combo 32 --magnitudes 12,16,20,25,30,40 \
    --knockdown-at 3,9 --seeds 43 --recovery-window 12 --recover-deadline 12 \
    --output logs/mjlab/evaluation/<name>/result.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
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
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from ct_mjlab.envs import task_registry  # noqa: E402
from ct_mjlab.envs.chuanliantui import multimode_mdp  # noqa: E402

TASK = "chuanliantui"
GRAVITY = 9.81
# Acceptance v2 thresholds.
WHEEL_SHARE_MIN = .10          # each wheel's upward support / total weight
WHEEL_FORCE_MIN_N = 5.
COMBINED_SHARE = (.80, 1.20)   # both wheels / total weight
NON_WHEEL_SHARE_MAX = .05      # every other part / total weight
BASE_FORCE_MAX_N = 1.
TILT_MAX_RAD = 0.2618          # 15 degrees
LIN_VEL_MAX = .15
ANG_VEL_MAX = .5
HEIGHT_TOLERANCE_M = .03
HEIGHT_COMMAND_M = .30
STABLE_HOLD_S = .5             # continuous compliance required
RECOVER_DEADLINE_S = 2.0       # ... completed within this of the fall
LAST5_FRACTION_MIN = .95
FALL_TILT_RAD = 0.6981         # 40 degrees, the task's own fallen gate
LEG_LOADED_SHARE = .30
OLD_RULE_STEPS = 20            # the 0.4 s contact/tilt gate this recipe replaced
# A single 20 ms step of base contact after recovery is noise, not "it fell over
# again"; the headline re-fall count requires this much continuous fallen time,
# and the any-step count is reported alongside so nothing is hidden.
REFELL_HOLD_S = .2
DIRECTIONS = 8
FINAL_BALANCE_HOLD_S = 1.
FINAL_BALANCE_TILT_RAD = np.radians(35.)
FINAL_BALANCE_CRITERIA = {
    "name": "wheel-balance-final-1s-v3.1", "hold_s": FINAL_BALANCE_HOLD_S,
    "tilt_max_deg": 35., "roll_pitch_ang_vel_max": ANG_VEL_MAX, "lin_vel_max": None,
    "yaw_vel_max": None,
    "height_tolerance_m": None, "requires_full_episode": True,
    "wheel_share_min": WHEEL_SHARE_MIN, "wheel_force_min_n": WHEEL_FORCE_MIN_N,
    "combined_share": list(COMBINED_SHARE), "non_wheel_share_max": None,
    "base_force_max_n": BASE_FORCE_MAX_N,
}


def final_balance_mask(tilt, contact, wheel_up, roll_pitch_speed, weight):
    """允许滚动和转向；腿部少量接触由双轮至少 80% 承重约束，不单独否决。"""
    return ((tilt <= FINAL_BALANCE_TILT_RAD)
            & (contact <= BASE_FORCE_MAX_N)
            & (wheel_up >= max(WHEEL_SHARE_MIN * weight, WHEEL_FORCE_MIN_N)).all(-1)
            & (wheel_up.sum(-1) >= COMBINED_SHARE[0] * weight)
            & (wheel_up.sum(-1) <= COMBINED_SHARE[1] * weight)
            & (roll_pitch_speed <= ANG_VEL_MAX))


def final_balance_passed(hold_s, duration_s, episode_seconds, dt, hit, numerical, termination):
    # 提前物理终止、未受扰或数值异常不能凭终止前的站立片段算成功。
    return bool(hit and not numerical and termination != "physical"
                and duration_s >= episode_seconds - dt / 2
                and hold_s >= FINAL_BALANCE_HOLD_S - 1e-6)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs-per-combo", type=int, default=32)
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--magnitudes", default="8,12,16", help="击倒角速度档 rad/s")
    parser.add_argument("--knockdown-at", default="3,9", help="击倒时刻秒")
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--recovery-window", type=float, default=None,
                        help="覆盖 recovery_window_s（秒）；用于检验「未恢复」是否为窗口到期所致")
    parser.add_argument("--recover-deadline", type=float, default=2.0,
                        help="倒地后多久内满足 v2 才算恢复（秒）；放宽以观察「慢起立」")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def git_state() -> dict:
    def run(*cmd):
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    try:
        return {"head": run("git", "rev-parse", "HEAD").strip(),
                "working_tree_status": run("git", "status", "--short"),
                "git_diff_sha256": hashlib.sha256(
                    subprocess.check_output(["git", "diff", "HEAD", "--binary"],
                                            cwd=REPO_ROOT)).hexdigest()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"head": None, "working_tree_status": None, "git_diff_sha256": None}


def build_env(args, seed: int, magnitudes, knockdown_times):
    """One env per seed; every env carries its (magnitude, time, direction) combo."""
    cfg, agent = task_registry.get_cfgs(TASK, play=True)
    per_combo = args.envs_per_combo
    combos = [(m, t) for m in magnitudes for t in knockdown_times]
    n = per_combo * len(combos)
    cfg.scene.num_envs = n
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    if args.recovery_window is not None:
        # Training ends a stand-up attempt recovery_window_s after the fall, so a
        # probe run under the training window cannot tell a policy that gave up
        # from one the gate cut off. Widening it here separates the two.
        cfg.commands["standup"].recovery_window_s = args.recovery_window
    # Balanced 50/50 draw instead of the task's 0.8, so both initial-state classes
    # are covered; the reset event keeps its pose-bank branches.
    cfg.events["reset_pose"] = EventTermCfg(
        func=multimode_mdp.reset_pose, mode="reset", params={"ground_probability": .5})
    # Whole-robot contact sensor: non-wheel support = all - wheels. `undesired_contact`
    # misses the gas-spring bodies, so it cannot serve as the acceptance criterion.
    cfg.scene.sensors += (ContactSensorCfg(
        name="all_robot_contact", primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        fields=("found", "force"), reduce="netforce", history_length=1),)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    # Round-robin, not contiguous blocks. Terrain columns are assigned to
    # environments by index (flat/slope/step/stair as 63/26/20/19 blocks at 128
    # envs, 95/39/29/29 at 192), so a block assignment puts every magnitude on a
    # single terrain family and the sweep measures terrain instead of magnitude.
    stride = len(combos)
    magnitude = torch.tensor([combos[i % stride][0] for i in range(n)], device=args.device)
    knock_at = torch.tensor([combos[i % stride][1] for i in range(n)], device=args.device)
    direction = torch.tensor(
        [2 * torch.pi * ((i // stride) % DIRECTIONS) / DIRECTIONS for i in range(n)],
        device=args.device)
    return env, agent, magnitude, knock_at, direction


def force_zero_command(state) -> None:
    """Force vx=0, yaw=0, height=0.30 through the command manager's own inputs.

    Overwriting `_command` after a step would be undone by the next step's
    `advance()` before the observation is read, so the sampled command, the
    ground-start override and the standing flag are pinned instead.
    """
    state.sampled[:, 0] = 0.
    state.sampled[:, 1] = 0.
    state.sampled[:, 2] = HEIGHT_COMMAND_M
    state.ground_command[:, :] = 0.
    state.standing[:] = True
    state.special_mode[:] = -1
    state.heading[:] = False


def main():
    args = parse_args()
    # The criterion is read as a module global from inside the per-seed loop, so
    # rebinding it here is what makes --recover-deadline take effect.
    global RECOVER_DEADLINE_S
    RECOVER_DEADLINE_S = args.recover_deadline
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    magnitudes = [float(v) for v in args.magnitudes.split(",")]
    knockdown_times = [float(v) for v in args.knockdown_at.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    configure_torch_backends()

    all_rows, extras = [], {}
    for seed in seeds:
        rows, seed_extras = run_seed(args, seed, magnitudes, knockdown_times)
        all_rows += rows
        extras[f"seed_{seed}"] = seed_extras
    report = {"kind": "finite mid-episode knockdown acceptance; no training",
              "time": datetime.now().astimezone().isoformat(),
              "task": TASK, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "script": str(Path(__file__).resolve()),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              **git_state(),
              "config": {"envs_per_combo": args.envs_per_combo, "seeds": seeds,
                         "magnitudes": magnitudes, "knockdown_at_s": knockdown_times,
                         "episode_seconds": args.episode_seconds,
                         "directions": DIRECTIONS, "start_mix": "50/50 normal/ground"},
              "criteria": {"wheel_share_min": WHEEL_SHARE_MIN,
                           "wheel_force_min_n": WHEEL_FORCE_MIN_N,
                           "combined_share": list(COMBINED_SHARE),
                           "non_wheel_share_max": NON_WHEEL_SHARE_MAX,
                           "base_force_max_n": BASE_FORCE_MAX_N,
                           "tilt_max_deg": round(np.degrees(TILT_MAX_RAD), 3),
                           "lin_vel_max": LIN_VEL_MAX, "ang_vel_max": ANG_VEL_MAX,
                           "height_tolerance_m": HEIGHT_TOLERANCE_M,
                           "stable_hold_s": STABLE_HOLD_S,
                           "recover_deadline_s": RECOVER_DEADLINE_S,
                           "refell_hold_s": REFELL_HOLD_S,
                           "last5_fraction_min": LAST5_FRACTION_MIN,
                           "final_balance": FINAL_BALANCE_CRITERIA},
              "episodes": all_rows, "extras": extras,
              "aggregates": aggregate(all_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["aggregates"], indent=2, ensure_ascii=False))
    print(f"WROTE {args.output}")


def run_seed(args, seed, magnitudes, knockdown_times):
    env, agent, magnitude, knock_at, direction = build_env(
        args, seed, magnitudes, knockdown_times)
    try:
        runner = task_registry._task(TASK).runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions),
            asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset(seed=seed)

        state = env.command_manager.get_term("standup")
        robot = env.scene["robot"]
        dt = env.step_dt
        n = env.num_envs
        steps = round(args.episode_seconds / dt)
        weight = float(env.sim.get_default_field("body_mass")[
            robot.indexing.body_ids].sum()) * GRAVITY

        ground_start = state.ground_start.clone()
        live = torch.ones(n, device=env.device, dtype=torch.bool)
        hit = torch.zeros(n, device=env.device, dtype=torch.bool)
        eligible = torch.zeros(n, device=env.device, dtype=torch.bool)
        fell = torch.zeros(n, device=env.device, dtype=torch.bool)
        fall_time = torch.full((n,), float("nan"), device=env.device)
        recovered = torch.zeros(n, device=env.device, dtype=torch.bool)
        recover_time = torch.full((n,), float("nan"), device=env.device)
        stable_run = torch.zeros(n, device=env.device)
        refell = torch.zeros(n, device=env.device, dtype=torch.bool)
        refell_any_step = torch.zeros(n, device=env.device, dtype=torch.bool)
        refell_run = torch.zeros(n, device=env.device)
        old_rule_steps = torch.zeros(n, dtype=torch.long, device=env.device)
        old_rule_would_fire = torch.zeros(n, device=env.device, dtype=torch.bool)
        numerical = torch.zeros(n, device=env.device, dtype=torch.bool)
        last5_ok = torch.zeros(n, device=env.device)
        last5_n = torch.zeros(n, device=env.device)
        height_ok = torch.zeros(n, device=env.device, dtype=torch.bool)
        max_tilt = torch.zeros(n, device=env.device)
        inverted = torch.zeros(n, device=env.device, dtype=torch.bool)
        base_touched = torch.zeros(n, device=env.device, dtype=torch.bool)
        max_leg_share = torch.zeros(n, device=env.device)
        last5_nonwheel = torch.zeros(n, device=env.device)
        t0_support = None
        max_wheel_support = 0.
        max_abs_wheel_force = 0.
        duration = torch.zeros(n, device=env.device)
        terminal = torch.full((n,), -1, device=env.device, dtype=torch.long)
        # Per-env impulse delivery time. `knock_at` is only the time the impulse
        # becomes due; it is held back until the robot has stood 0.5 s, so the
        # moment that matters is the one recorded here.
        hit_time = torch.full((n,), float("nan"), device=env.device)
        last5_v2_only = torch.zeros(n, device=env.device)
        final_tilt = torch.zeros(n, device=env.device)
        final_balance_steps = torch.zeros(n, dtype=torch.long, device=env.device)

        with torch.inference_mode():
            for step in range(steps):
                t = step * dt
                active = live.clone()
                force_zero_command(state)
                # Knockdown: fire at each env's own time, once, and only at a robot
                # that has actually stood for 0.5 s (recorded either way).
                due = active & ~hit & (t >= knock_at)
                if bool(due.any()):
                    steady = (~state.fallen) & (state.stable_time >= STABLE_HOLD_S)
                    fire = due & steady
                    eligible |= due
                    hit |= fire
                    if bool(fire.any()):
                        ids = fire.nonzero().flatten()
                        velocity = robot.data.root_link_vel_w[ids].clone()
                        velocity[:, 3] = magnitude[ids] * torch.cos(direction[ids])
                        velocity[:, 4] = magnitude[ids] * torch.sin(direction[ids])
                        robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
                        # The impulse is this measurement's time zero. Everything the
                        # robot did before it is discarded -- above all the ground-start
                        # stand-up, which satisfies the v2 criterion on its own. Without
                        # this reset `recovered` latches before the knockdown ever lands
                        # (measured: 107 of 113 "recoveries" completed first) and the
                        # post-knockdown stand-up is never observed at all, because
                        # `recovered`/`recover_time` only ever record the first one.
                        hit_time[ids] = duration[ids]
                        fell[ids] = False
                        fall_time[ids] = 0.
                        recovered[ids] = False
                        recover_time[ids] = float("nan")
                        stable_run[ids] = 0.
                        final_balance_steps[ids] = 0
                        refell[ids] = False
                        refell_any_step[ids] = False
                        refell_run[ids] = 0.
                        old_rule_steps[ids] = 0
                        old_rule_would_fire[ids] = False
                        max_tilt[ids] = 0.
                        final_tilt[ids] = 0.
                        inverted[ids] = False
                        base_touched[ids] = False
                        max_leg_share[ids] = 0.
                        height_ok[ids] = False
                        last5_ok[ids] = 0.
                        last5_v2_only[ids] = 0.
                        last5_n[ids] = 0.
                        last5_nonwheel[ids] = 0.

                obs, _, terminated, timeout, _ = env.step(policy(obs))
                force_zero_command(state)
                duration += active.float() * dt

                contact = env.scene["base_contact"].data.force.norm(dim=-1).amax(-1)
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
                tilt = torch.maximum(roll.abs(), pitch.abs())
                # ContactSensor's net wrench points from the primary body to the
                # terrain, so the terrain's upward support is its negative z.
                raw_wheel = env.scene["wheel_contact"].data.force
                raw_all = env.scene["all_robot_contact"].data.force
                wheel_up = -raw_wheel[:, :, 2]
                non_wheel_up = ((-raw_all[:, :, 2].sum(-1)) - wheel_up.sum(-1)).clamp_min(0.)
                max_abs_wheel_force = max(max_abs_wheel_force, float(raw_wheel.abs().max()))
                tilt_ok = tilt <= TILT_MAX_RAD
                v2 = (tilt_ok
                      & (contact <= BASE_FORCE_MAX_N)
                      & (wheel_up.min(-1).values >= max(WHEEL_SHARE_MIN * weight,
                                                         WHEEL_FORCE_MIN_N))
                      & (wheel_up.sum(-1) >= COMBINED_SHARE[0] * weight)
                      & (wheel_up.sum(-1) <= COMBINED_SHARE[1] * weight)
                      & (non_wheel_up <= NON_WHEEL_SHARE_MAX * weight)
                      & (robot.data.root_link_lin_vel_b.norm(dim=-1) <= LIN_VEL_MAX)
                      & (robot.data.root_link_ang_vel_b.norm(dim=-1) <= ANG_VEL_MAX))
                height_ok_step = (state.height_above_ground - HEIGHT_COMMAND_M).abs() <= HEIGHT_TOLERANCE_M
                combined = v2 & height_ok_step
                final_ok = final_balance_mask(
                    tilt, contact, wheel_up,
                    robot.data.root_link_ang_vel_b[:, :2].norm(dim=-1), weight)
                # 只累计紧贴结束时刻的连续合格步；中间任意一拍失败即清零。
                final_balance_steps = torch.where(
                    active, torch.where(hit & final_ok, final_balance_steps + 1, 0),
                    final_balance_steps)

                down = state.fallen & active
                newly_fell = down & ~fell
                fall_time = torch.where(newly_fell, duration, fall_time)
                fell |= down
                stable_run = torch.where(active & combined, stable_run + dt,
                                         torch.where(active, 0., stable_run))
                # Acceptance v2's own wording: 0.5 s of continuous compliance within
                # 2 s of the fall. A latched "was once strict" does not qualify.
                in_deadline = active & fell & ~recovered & (duration - fall_time <= RECOVER_DEADLINE_S)
                newly_recovered = in_deadline & (stable_run >= STABLE_HOLD_S)
                recover_time = torch.where(newly_recovered, duration - fall_time, recover_time)
                recovered |= newly_recovered
                refell_any_step |= recovered & down
                # Continuous, not cumulative. The old form kept the accumulated
                # value while the robot stood, so two separate short falls could
                # trip the 0.2 s threshold with neither of them lasting that long.
                refell_run = torch.where(recovered & down, refell_run + dt,
                                         torch.zeros_like(refell_run))
                refell |= refell_run >= REFELL_HOLD_S

                # What the 0.4 s gate this recipe replaced would have done.
                old_fail = (contact > BASE_FORCE_MAX_N) | (tilt > FALL_TILT_RAD)
                old_rule_steps = torch.where(
                    active & old_fail & ~state.numerical_failure,
                    (old_rule_steps + 1).clamp_max(OLD_RULE_STEPS), torch.zeros_like(old_rule_steps))
                old_rule_would_fire |= (old_rule_steps >= OLD_RULE_STEPS) & ~recovered

                in_last5 = active & (duration > args.episode_seconds - 5.)
                last5_n += in_last5.float() * dt
                last5_ok += (in_last5 & combined).float() * dt
                last5_v2_only += (in_last5 & v2).float() * dt
                last5_nonwheel += (in_last5 * (non_wheel_up / weight)).float() * dt
                height_ok |= in_last5 & height_ok_step
                max_tilt = torch.maximum(max_tilt, torch.where(active, tilt, torch.zeros_like(tilt)))
                final_tilt = torch.where(active, tilt, final_tilt)
                inverted |= active & (gravity[:, 2] > 0.)
                base_touched |= active & (contact > BASE_FORCE_MAX_N)
                max_leg_share = torch.maximum(max_leg_share,
                                              torch.where(active, non_wheel_up / weight,
                                                          torch.zeros_like(non_wheel_up)))
                numerical |= active & state.numerical_failure
                if t0_support is None:
                    t0_support = float(wheel_up.sum(-1).mean())
                max_wheel_support = max(max_wheel_support, float(wheel_up.sum(-1).max()))

                ended = active & (terminated | timeout)
                terminal = torch.where(ended & terminated, 1, terminal)
                terminal = torch.where(ended & timeout & ~terminated, 0, terminal)
                done = terminated | timeout
                if bool(done.any()):
                    env.reset(env_ids=done.nonzero().flatten())
                live &= ~done

        # The sensor's sign convention and coverage are load-bearing. A robot that
        # stands at any point in the batch must have carried roughly its whole
        # weight on the two wheels, so a lower bound on the run maximum is the
        # robust check; requiring it at a single early step would instead trip on
        # a sensor buffer that is still filling.
        if max_wheel_support < 0.5 * weight:
            raise SystemExit(
                f"轮端支撑读取异常：全程双轮合计最大 {max_wheel_support:.1f} N，"
                f"整机重量 {weight:.1f} N（t0 均值 {t0_support}，原始张量最大绝对值 "
                f"{max_abs_wheel_force:.3f}）。传感器符号、分量或覆盖有误，验收判据不可用。")

        rows = []
        for i in range(n):
            # Classify by how far it actually went over, plus whether it *settled*
            # propped on non-wheel parts. The run maximum of non-wheel support is
            # impact-dominated (self-collision forces spike when the linkage
            # slams shut), so the settled last-5-s mean is the meaningful one.
            settled_share = (float(last5_nonwheel[i] / last5_n[i])
                             if float(last5_n[i]) else 0.)
            fall_type = "no_fall"
            if bool(fell[i]):
                if bool(inverted[i]):
                    fall_type = "inverted"
                elif settled_share >= LEG_LOADED_SHARE:
                    fall_type = "settled_on_non_wheel"
                elif float(max_tilt[i]) > np.radians(60.):
                    fall_type = "tilted_deep"
                elif float(max_tilt[i]) > np.radians(40.):
                    fall_type = "tilted"
                else:
                    fall_type = "contact_only"
            rows.append({
                "seed": seed, "env": i,
                "magnitude": float(magnitude[i]), "knockdown_at_s": float(knock_at[i]),
                "direction_rad": float(direction[i]),
                "ground_start": bool(ground_start[i]),
                "eligible": bool(eligible[i]), "hit": bool(hit[i]),
                # Impulse-anchored fields. `last5_observed_s` exposes how much of the
                # last 5 s was actually observed (an episode terminated at 15.02 s
                # contributes a sliver, not 5 s), and the pure-v2 fraction leaves the
                # +-3 cm height gate out, so a zero can be read as "not balanced"
                # rather than "not balanced or not at height".
                "terrain_kind": int(state.terrain_kind[i]),
                "hit_time_s": None if not bool(hit[i]) else float(hit_time[i]),
                "observed_after_hit_s": (None if not bool(hit[i])
                                         else float(duration[i] - hit_time[i])),
                "last5_observed_s": float(last5_n[i]),
                "last5_pure_v2_fraction": (float(last5_v2_only[i] / last5_n[i])
                                           if float(last5_n[i]) else None),
                "final_tilt_deg": float(np.degrees(final_tilt[i].item())),
                "final_balance_hold_s": int(final_balance_steps[i]) * dt,
                "final_balanced": final_balance_passed(
                    int(final_balance_steps[i]) * dt, float(duration[i]),
                    args.episode_seconds, dt, bool(hit[i]), bool(numerical[i]),
                    ("unfinished", "time_out", "physical")[int(terminal[i]) + 1]),
                "fell": bool(fell[i]), "fall_type": fall_type,
                "fall_time_s": None if not bool(fell[i]) else float(fall_time[i]),
                "recovered": bool(recovered[i]),
                "recover_time_s": None if not bool(recovered[i]) else float(recover_time[i]),
                "refell": bool(refell[i]),
                "refell_any_step": bool(refell_any_step[i]),
                "old_rule_would_terminate": bool(old_rule_would_fire[i]),
                "last5_v2_fraction": (float(last5_ok[i] / last5_n[i]) if float(last5_n[i]) else None),
                "height_ok_last5": bool(height_ok[i]),
                "max_tilt_deg": float(np.degrees(max_tilt[i].item())),
                "settled_non_wheel_share": settled_share,
                "max_non_wheel_share_incl_impacts": float(max_leg_share[i]),
                "numerical_failure": bool(numerical[i]),
                "termination": ("unfinished", "time_out", "physical")[int(terminal[i]) + 1],
                "duration_s": float(duration[i]),
            })
        seed_extras = {"num_envs": n, "weight_n": weight, "t0_wheel_support_n": t0_support,
                       "recovery_window_s": env.command_manager.get_term("standup").cfg.recovery_window_s,
                       "max_wheel_support_n": max_wheel_support,
                       "max_abs_wheel_force": max_abs_wheel_force,
                       "force_shapes": {"wheel_contact": list(raw_wheel.shape),
                                        "all_robot_contact": list(raw_all.shape)}}
        return rows, seed_extras
    finally:
        env.close()


def _rate(rows, key):
    return (sum(bool(r[key]) for r in rows) / len(rows)) if rows else None


def aggregate(rows):
    def group(subset):
        if not subset:
            return {"episodes": 0}
        fell = [r for r in subset if r["fell"]]
        recovered = [r for r in fell if r["recovered"]]
        times = sorted(r["recover_time_s"] for r in recovered)
        fracs = [r["last5_v2_fraction"] for r in subset if r["last5_v2_fraction"] is not None]
        return {
            "episodes": len(subset),
            "hit": sum(r["hit"] for r in subset),
            "not_steady_at_knockdown": sum(r["eligible"] and not r["hit"] for r in subset),
            "fell": len(fell),
            "fell_types": {t: sum(r["fall_type"] == t for r in fell)
                           for t in ("contact_only", "tilted", "tilted_deep", "inverted",
                                     "settled_on_non_wheel")},
            "max_tilt_deg": {"p50": float(np.median([r["max_tilt_deg"] for r in fell]))
                             if fell else None,
                             "max": max((r["max_tilt_deg"] for r in fell), default=None)},
            "mean_settled_non_wheel_share": float(np.mean(
                [r["settled_non_wheel_share"] for r in subset])) if subset else None,
            "recovered": len(recovered),
            "recovery_rate": (len(recovered) / len(fell)) if fell else None,
            "final_balanced": sum(r["final_balanced"] for r in fell),
            "final_balance_rate": _rate(fell, "final_balanced"),
            "median_recover_time_s": float(np.median(times)) if times else None,
            "max_recover_time_s": max(times) if times else None,
            "refell": sum(r["refell"] for r in fell),
            "refell_any_step": sum(r["refell_any_step"] for r in fell),
            "old_rule_would_terminate": sum(r["old_rule_would_terminate"] for r in fell),
            "last5_v2_ok": sum(1 for r in subset
                               if (r["last5_v2_fraction"] or 0) >= LAST5_FRACTION_MIN),
            "mean_last5_v2_fraction": float(np.mean(fracs)) if fracs else None,
            "height_ok_last5": sum(r["height_ok_last5"] for r in subset),
            "numerical_failures": sum(r["numerical_failure"] for r in subset),
            "terminations": {t: sum(r["termination"] == t for r in subset)
                             for t in ("unfinished", "time_out", "physical")},
        }

    normal = [r for r in rows if not r["ground_start"]]
    ground = [r for r in rows if r["ground_start"]]
    by_knockdown = [r for r in rows if r["fell"]]
    return {
        "all": group(rows),
        "normal_start": group(normal),
        "ground_start": group(ground),
        "by_seed": {str(s): group([r for r in rows if r["seed"] == s])
                    for s in sorted({r["seed"] for r in rows})},
        "by_magnitude": {str(m): group([r for r in rows if r["magnitude"] == m])
                         for m in sorted({r["magnitude"] for r in rows})},
        "by_knockdown_at": {str(t): group([r for r in rows if r["knockdown_at_s"] == t])
                            for t in sorted({r["knockdown_at_s"] for r in rows})},
        "headline": {
            "normal_start_falls": len([r for r in normal if r["fell"]]),
            "normal_start_recovered": len([r for r in normal if r["fell"] and r["recovered"]]),
            "ground_start_falls": len([r for r in ground if r["fell"]]),
            "ground_start_recovered": len([r for r in ground if r["fell"] and r["recovered"]]),
            "meets_60_falls_zero_failure": (
                len([r for r in normal if r["fell"]]) >= 60
                and all(r["final_balanced"] for r in normal if r["fell"])
                and len([r for r in ground if r["fell"]]) >= 60
                and all(r["final_balanced"] for r in ground if r["fell"])),
            "note": "current success = final_balanced (final continuous 1 s, v3); recovered and last5_* retain legacy v2 definitions and do not veto current success",
        },
    }


if __name__ == "__main__":
    main()
