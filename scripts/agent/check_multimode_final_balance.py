#!/usr/bin/env python3
"""检查最终 1 秒平衡判据；可用已保存的传感器轨迹重算，不重新仿真。

  mjlab_training/.venv/bin/python scripts/agent/check_multimode_final_balance.py \
    --recording-dir logs/mjlab/evaluation/kd-failure-videos-4850 \
    --output artifacts/multimode-final-balance-v3/result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

import assess_multimode_knockdown_recovery as assess


def check_criterion():
    # 前两例允许 20°倾角，以及腿部承担少量重量；后续逐项违反必要条件。
    tilt = np.radians([20., 0., 36., 0., 0., 0., 0., 0.])
    contact = np.array([0., 0., 0., 2., 0., 0., 0., 0.])
    wheels = np.full((8, 2), 50.)
    wheels[1] = [45., 45.]
    wheels[4] = [0., 100.]
    wheels[5] = [30., 30.]
    wheels[6] = [70., 70.]
    angular = np.zeros((8, 3))
    angular[:, 2] = 1.  # 转向速度不用于判断是否翻倒。
    angular[7, 0] = .6
    values = (tilt, contact, wheels, np.linalg.norm(angular[:, :2], axis=-1))
    expected = np.array([True, True, False, False, False, False, False, False])
    np.testing.assert_array_equal(assess.final_balance_mask(*values, 100.), expected)
    result = assess.final_balance_mask(*(torch.from_numpy(v) for v in values), 100.)
    np.testing.assert_array_equal(result.numpy(), expected)
    passed = assess.final_balance_passed
    assert passed(1., 20., 20., .02, True, False, "time_out")
    assert not passed(.98, 20., 20., .02, True, False, "time_out")
    assert not passed(1., 19., 20., .02, True, False, "time_out")
    assert not passed(1., 20., 20., .02, True, False, "physical")
    assert not passed(1., 20., 20., .02, False, False, "time_out")
    assert not passed(1., 20., 20., .02, True, True, "time_out")
    print("PASS: NumPy/Torch 判据一致；倾角、承重、接触、角速度、时长及终止检查通过")


def check_comparison():
    # “曾恢复”丢失但最终站稳，不应否决；最终失效必须否决，旧指标不能冒充新指标。
    row = {"seed": 43, "env": 0, "magnitude": 16., "knockdown_at_s": 3.,
           "hit": True, "fell": True, "recovered": True, "terrain_kind": 0,
           "last5_pure_v2_fraction": 1., "final_tilt_deg": 1.,
           "final_balanced": True, "final_balance_hold_s": 1.2}
    baseline = {"task": "chuanliantui", "config": {},
                "criteria": {"final_balance": assess.FINAL_BALANCE_CRITERIA},
                "episodes": [row]}
    candidate = json.loads(json.dumps(baseline))
    candidate["episodes"][0]["recovered"] = False
    script = Path(__file__).with_name("compare_multimode_knockdown.py")
    with tempfile.TemporaryDirectory() as folder:
        old, new = Path(folder) / "old.json", Path(folder) / "new.json"

        def compare(expected, *options):
            old.write_text(json.dumps(baseline))
            new.write_text(json.dumps(candidate))
            result = subprocess.run([sys.executable, str(script), "--baseline", str(old),
                                     "--candidate", str(new), *options], text=True,
                                    capture_output=True)
            assert result.returncode == expected, result.stdout + result.stderr

        compare(0)
        candidate["episodes"][0]["final_balanced"] = False
        candidate["episodes"][0]["final_balance_hold_s"] = .98
        compare(1)
        for report in (baseline, candidate):
            report["criteria"] = {}
            report["episodes"][0].pop("final_balanced")
            report["episodes"][0].pop("final_balance_hold_s")
            report["episodes"][0]["recovered"] = True
        compare(2)
        compare(0, "--criterion", "legacy-v2")
    print("PASS: 配对退出码 0/1/2、历史口径显式选择与最终平衡否决检查通过")


def reassess(recording_dir, output):
    saved = json.loads((recording_dir / "reproduced-episodes.json").read_text())
    rows = {r["env"]: r for r in saved["episodes"]}
    weight = saved["extras"]["weight_n"]
    dt, episode_seconds = .02, 20.
    results = []
    archive_path = recording_dir / "recorded-states.npz"
    with np.load(archive_path) as states:
        for col, env_id in enumerate(states["env_ids"]):
            row = dict(rows[int(env_id)])
            count = round(row["duration_s"] / dt)
            g = states["gravity"][:count, col]
            tilt = np.maximum(abs(np.arctan2(-g[:, 1], -g[:, 2])),
                              abs(np.arcsin(np.clip(g[:, 0], -1., 1.))))
            contact = np.linalg.norm(states["base_force"][:count, col], axis=-1).max(-1)
            wheels = -states["wheel_force"][:count, col, :, 2]
            angular = np.linalg.norm(states["angular"][:count, col, :2], axis=-1)
            ok = assess.final_balance_mask(tilt, contact, wheels, angular, weight)
            # 与在线计数相同：只记冲量后、紧贴回合结束的连续合格步，不锁存早先成功。
            run = 0
            hit_step = round(row["hit_time_s"] / dt) if row["hit"] else count
            for step in range(hit_step, count):
                run = run + 1 if ok[step] else 0
            hold_s = run * dt
            row["final_balance_hold_s"] = hold_s
            row["final_balanced"] = assess.final_balance_passed(
                hold_s, row["duration_s"], episode_seconds, dt, row["hit"],
                row["numerical_failure"], row["termination"])
            results.append(row)
    report = {
        "kind": "offline final-balance-v3.1 reassessment; recorded samples only; no simulation",
        "criteria": {"final_balance": assess.FINAL_BALANCE_CRITERIA},
        "recording_dir": str(recording_dir.resolve()),
        "recorded_states_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "episode_seconds": episode_seconds, "step_dt": dt, "weight_n": weight,
        "episodes": results,
        "final_balanced": sum(r["final_balanced"] for r in results),
        "total_recorded": len(results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    lines = ["# 最终双轮平衡 v3.1：已录制轨迹重算", "",
             "结束前连续1秒双轮承担80%～120%重量、每轮至少10%且≥5N、机身接触力≤1N，",
             "倾角≤35°、横滚/俯仰角速度合量≤0.5 rad/s；允许滚动、转向与少量腿部接触。",
             "高度与起身用时不作成功门槛。必须完整观察到20秒且无数值异常。",
             "仅重算已有7条轨迹，不重新仿真；旧JSON、视频、README及复现失败证据均保留。",
             "视频中的末5秒占比是历史v2指标，不代表当前是否成功。", "",
             "| env | 最终平衡 | 结束前连续合格时间 | 历史末5秒占比 | 视频 |",
             "|---|---|---:|---:|---|"]
    videos = {v["env"]: v["file"] for v in json.loads((recording_dir / "videos.json").read_text())}
    for row in results:
        video = (recording_dir / videos[row["env"]]).resolve()
        lines.append(f"| {row['env']} | {'通过' if row['final_balanced'] else '未通过'} | "
                     f"{row['final_balance_hold_s']:.2f} s | {row['last5_pure_v2_fraction']:.3f} | "
                     f"[回放]({video}) |")
    lines += ["", "该结果只覆盖已录制样本，不能外推为完整384环境或与父模型配对的结论。", ""]
    with (output.parent / "README.md").open("x") as f:
        f.write("\n".join(lines))
    print(f"WROTE {output}; 最终平衡 {report['final_balanced']}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check_criterion()
    check_comparison()
    if args.recording_dir:
        if args.output is None:
            parser.error("重算轨迹须指定 --output，保留原始产物")
        reassess(args.recording_dir, args.output)


if __name__ == "__main__":
    main()
