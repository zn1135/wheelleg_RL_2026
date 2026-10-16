#!/usr/bin/env python3
"""检查 multimode 复位的地面（recover）初态比例是否等于配置值。

用训练配置无头构建 `chuanliantui`，反复整体复位并统计 `standup.ground_start`。
除比例本身，还核对三件事：配置字段与复位事件参数一致（防止两处脱钩）；地面组
根高落在姿态库地面根高且水平（标志与姿态一致）；各地形族比例一致（复位抽签不
受地形影响）。

只做有限分布检查，不调用 learn()，不是行为验收。

用法（仓库根目录）：
  mjlab_training/.venv/bin/python scripts/agent/check_multimode_ground_ratio.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mjlab_training" / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from ct_mjlab.envs import task_registry  # noqa: E402
from ct_mjlab.envs.chuanliantui.huananhu_poses import GROUND_ROOT_HEIGHT  # noqa: E402
from ct_mjlab.envs.chuanliantui.multimode_mdp import TERRAIN_NAMES  # noqa: E402

TASK = "chuanliantui"
HEIGHT_TOLERANCE_M = .02
TILT_TOLERANCE_DEG = .5


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--resets", type=int, default=40, help="整体复位次数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=.05, help="允许的比例绝对偏差")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_backends()
    cfg, _ = task_registry.get_cfgs(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        state = env.command_manager.get_term("standup")
        expected = env.cfg.commands["standup"].ground_start_probability
        event_param = env.cfg.events["reset_pose"].params["ground_probability"]
        if abs(event_param - expected) > 1e-9:
            raise SystemExit(
                f"复位事件参数 {event_param} 与配置字段 {expected} 不一致")

        samples = ground_hits = 0
        per_terrain: dict[str, list[int]] = {}
        ground_z, ground_tilt = [], []
        with torch.inference_mode():
            for _ in range(args.resets):
                env.reset()
                flag = state.ground_start.clone()
                robot = env.scene["robot"]
                z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
                tilt = torch.rad2deg(torch.maximum(roll.abs(), pitch.abs()))

                samples += flag.numel()
                ground_hits += int(flag.sum())
                if flag.any():
                    ground_z.append(z[flag])
                    ground_tilt.append(tilt[flag])
                for index, name in enumerate(TERRAIN_NAMES):
                    rows = state.terrain_kind == index
                    if rows.any():
                        hit, seen = per_terrain.get(name, [0, 0])
                        per_terrain[name] = [hit + int(flag[rows].sum()), seen + int(rows.sum())]

        ratio = ground_hits / samples
        gz = torch.cat(ground_z)
        gt = torch.cat(ground_tilt)
        report = {
            "kind": "finite reset-distribution check; no training",
            "task": TASK,
            "num_envs": args.num_envs,
            "resets": args.resets,
            "samples": samples,
            "expected_ground_ratio": expected,
            "observed_ground_ratio": ratio,
            "tolerance": args.tolerance,
            "per_terrain_ground_ratio": {
                name: hit / seen for name, (hit, seen) in sorted(per_terrain.items())},
            "ground_root_height_m": {
                "expected": GROUND_ROOT_HEIGHT,
                "mean": gz.mean().item(),
                "max_abs_deviation": (gz - GROUND_ROOT_HEIGHT).abs().max().item()},
            "ground_tilt_deg": {"max": gt.max().item()},
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))

        if abs(ratio - expected) > args.tolerance:
            raise SystemExit(f"地面比例 {ratio:.4f} 偏离期望 {expected}")
        if (gz - GROUND_ROOT_HEIGHT).abs().max() > HEIGHT_TOLERANCE_M:
            raise SystemExit("地面初态根高偏离姿态库地面根高")
        if gt.max() > TILT_TOLERANCE_DEG:
            raise SystemExit("地面初态不是水平姿态，标志与姿态可能不一致")
    finally:
        env.close()


if __name__ == "__main__":
    main()
