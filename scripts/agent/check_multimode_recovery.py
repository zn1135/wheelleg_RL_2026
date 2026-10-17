#!/usr/bin/env python3
"""检查 multimode 中途倒地后的 recover 重新触发、恢复窗口与恢复期奖励门控。

用 play 配置（无域随机化、无观测噪声/延迟）加载 model_3023，先让机器人站稳，
再用 knockdown_root_velocity 施加一次角速度冲量把它掀翻，然后逐步断言：

1. recovery_window_s 已接进命令配置（默认 2.0）；
2. 手动测试冲量确实把站立机器人掀翻，以验证倒地后的恢复机制；训练不注册该冲量事件；
3. 倒地后 recover 通道重新点亮，且**普通初态（ground_start=False）的回合也会亮**，
   这是本次改动的核心：旧代码里 recover 只由复位时的 ground_start 决定；
4. 倒地期间 undesired_contact 奖励恒为 0（恢复期门控）；
5. 没有回合在"连续倒地时间 <= 窗口"时被终止——即 0.4 秒的旧门限确实被放宽
   （数值异常与越出地块仍允许立即终止）。

只做有限机制检查，不训练，不替代行为验收。

用法（仓库根目录）：
  mjlab_training/.venv/bin/python scripts/agent/check_multimode_recovery.py
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mjlab_training" / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from ct_mjlab.envs import task_registry  # noqa: E402
from ct_mjlab.envs.chuanliantui.huananhu_randomization import (  # noqa: E402
    knockdown_root_velocity)
from ct_mjlab.envs.chuanliantui.multimode_mdp import (  # noqa: E402
    MODE_NAMES, OUT_OF_PATCH_RADIUS_M)

TASK = "chuanliantui"
CHECKPOINT = REPO_ROOT / "artifacts/multimode-v1-replay-20261014/models/model_3023.pt"
RECOVER_INDEX = MODE_NAMES.index("recover")
# Same bound the task terminates on; a hardcoded 2.5 m would flag a legitimate
# excursion as a premature termination.
OUT_OF_PATCH_M = OUT_OF_PATCH_RADIUS_M


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stand-seconds", type=float, default=2.0, help="冲量前先站稳的时长")
    parser.add_argument("--watch-seconds", type=float, default=4.0, help="冲量后观察的时长")
    parser.add_argument("--impulse-min", type=float, default=18.0, help="击倒冲量角速度下限 rad/s")
    parser.add_argument("--impulse-max", type=float, default=28.0, help="击倒冲量角速度上限 rad/s")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-config", action="store_true",
                        help="用训练配置（play=False）检查未注册主动击倒事件且恢复窗口保留")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_backends()
    cfg, agent = task_registry.get_cfgs(TASK, play=not args.training_config)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    if args.training_config:
        forced_knockdowns = [name for name, event in cfg.events.items()
                             if event.func is knockdown_root_velocity]
        if forced_knockdowns:
            raise SystemExit(f"训练仍注册了主动击倒事件：{forced_knockdowns}")
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = task_registry._task(TASK).runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions),
            asdict(agent), device=args.device)
        runner.load(str(CHECKPOINT), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset(seed=args.seed)

        state = env.command_manager.get_term("standup")
        dt = env.step_dt
        n = env.num_envs
        window = state.cfg.recovery_window_s
        if window is None:
            raise SystemExit("recovery_window_s 未接进命令配置，本次改动未生效")

        def advance(steps):
            nonlocal obs
            for _ in range(steps):
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                done = terminated | timeout
                if bool(done.any()):
                    env.reset(env_ids=done.nonzero().flatten())

        with torch.inference_mode():
            if args.training_config:
                # Natural falls remain valid; do not require them or confuse them
                # with an explicitly registered angular-velocity knockdown event.
                onsets = 0
                for _ in range(round(12.0 / dt)):
                    obs, _, terminated, timeout, _ = env.step(policy(obs))
                    onsets += int(state.knockdown.sum())
                    done = terminated | timeout
                    if bool(done.any()):
                        env.reset(env_ids=done.nonzero().flatten())
                print(json.dumps({"kind": "training-config recovery without forced knockdown",
                                  "num_envs": n, "seconds": 12.0,
                                  "forced_knockdown_events": forced_knockdowns,
                                  "recovery_window_s": window,
                                  "natural_fall_onsets": onsets}, indent=2))
                print("训练配置无主动击倒、保留恢复窗口检查通过", flush=True)
                return
            advance(round(args.stand_seconds / dt))

            upright = (~state.fallen) & (state.stable_time >= .5)
            if not bool(upright.any()):
                raise SystemExit("冲量前没有站立的机器人，检查无法进行")
            ground_start = state.ground_start.clone()
            recover_before = state.mode()[:, RECOVER_INDEX] > 0
            knockdown_counter_before = float(state.metrics["mode_knockdown"].sum())

            knockdown_root_velocity(
                env, None, angular_speed_range=(args.impulse_min, args.impulse_max))
            # Round-trip diagnostic. If this reads back ~0 the impulse never
            # reached the sim's qvel and the mechanism is a no-op; if it reads
            # back the requested speed but max_tilt_deg stays small, the physics
            # absorbed it instead (a wheeled robot can roll a pitch impulse away).
            free_v = env.scene["robot"].indexing.free_joint_v_adr
            applied_speed = float(env.sim.data.qvel[:, free_v][:, 3:].norm(dim=-1).max())

            reset_during = torch.zeros(n, device=env.device, dtype=torch.bool)
            down_seen = torch.zeros(n, device=env.device, dtype=torch.bool)
            rearmed = torch.zeros(n, device=env.device, dtype=torch.bool)
            gating_violations = 0
            premature = []
            max_fallen_time = 0.
            max_tilt_deg = 0.
            for _ in range(round(args.watch_seconds / dt)):
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                fallen = state.fallen
                down_seen |= fallen
                rearmed |= fallen & (state.mode()[:, RECOVER_INDEX] > 0)
                if bool(fallen.any()):
                    contact = env._huananhu_rewards.compute(env)["undesired_contact"]
                    if bool((contact[fallen] != 0).any()):
                        gating_violations += 1
                max_fallen_time = max(max_fallen_time, float(state.fallen_time.max()))
                gravity = env.scene["robot"].data.projected_gravity_b
                max_tilt_deg = max(max_tilt_deg, float(
                    torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1., 1.))).max()))
                # A termination inside the window is only legitimate for numerical
                # failure or leaving the patch; anything else means the old 0.4 s
                # gate is still in force.
                for env_id in terminated.nonzero().flatten().tolist():
                    if bool(state.numerical_failure[env_id]):
                        continue
                    if float(state.fallen_time[env_id]) > window:
                        continue
                    offset = (env.scene["robot"].data.root_link_pos_w[env_id, :2]
                              - env.scene.env_origins[env_id, :2])
                    if float(offset.abs().amax()) <= OUT_OF_PATCH_M:
                        premature.append(env_id)
                done = terminated | timeout
                reset_during |= done
                if bool(done.any()):
                    env.reset(env_ids=done.nonzero().flatten())

            # Do not drop envs that reset during the watch: an env that fell and
            # then timed out on the recovery window is exactly the failure case,
            # so excluding it would report a survivorship-biased knockdown count.
            # Episode identity is only needed for the normal/ground split, so that
            # one still requires an env that stayed inside its episode.
            stable_envs = ~reset_during
            knocked = down_seen & upright
            knocked_in_episode = knocked & stable_envs
            normal_start_knocked = knocked_in_episode & ~ground_start
            report = {
                "kind": "finite mid-episode recovery check; no training",
                "task": TASK, "num_envs": n, "seed": args.seed,
                "recovery_window_s": window,
                "upright_before_impulse": int(upright.sum()),
                "ground_start_envs": int(ground_start.sum()),
                "recover_channel_on_before_impulse": int(recover_before.sum()),
                "knocked_down_after_impulse": int(knocked.sum()),
                "knocked_down_then_reset": int((knocked & reset_during).sum()),
                "knocked_down_normal_start": int(normal_start_knocked.sum()),
                "recover_rearmed": int((rearmed & knocked).sum()),
                "recover_rearmed_normal_start": int((rearmed & normal_start_knocked).sum()),
                "max_fallen_time_s": max_fallen_time,
                "applied_angular_speed_max": applied_speed,
                "max_tilt_deg": max_tilt_deg,
                "undesired_contact_gating_violations": gating_violations,
                "premature_terminations": premature,
                "mode_knockdown_delta": float(state.metrics["mode_knockdown"].sum())
                                        - knockdown_counter_before,
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))

            failures = []
            if report["recover_channel_on_before_impulse"] != 0:
                failures.append("冲量前就有回合点亮 recover，前提不成立")
            if report["knocked_down_after_impulse"] == 0:
                failures.append("手动测试冲量没能掀翻任何站立机器人，无法验证倒地恢复机制")
            if report["recover_rearmed_normal_start"] == 0:
                failures.append("普通初态回合倒地后 recover 未重新点亮：改动未生效")
            if gating_violations:
                failures.append(f"倒地期间 undesired_contact 未归零 {gating_violations} 拍")
            if premature:
                failures.append(f"仍有回合在窗口内被终止（旧 0.4 秒门限未放宽）：{premature}")
            if failures:
                raise SystemExit("；".join(failures))
            print("机制检查通过", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
