#!/usr/bin/env python3
"""Compare the wheel-load NORMAL rule on identical recorded stand_v2 first episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(before_path, after_path, *, allow_different_checkpoint=False):
    before, after = [json.loads(p.read_text()) for p in (before_path, after_path)]
    keys = ("xml_sha256", "seed", "num_envs", "duration_s", "policy_dt_s")
    if not allow_different_checkpoint:
        keys = ("checkpoint_sha256", *keys)
    for name in keys:
        if before[name] != after[name]:
            raise ValueError(f"Paired diagnostic differs: {name}")
    with np.load(before_path.parent/"measurements.npz") as b, np.load(after_path.parent/"measurements.npz") as a:
        for name in ("initial_qpos", "initial_qvel", "weight_N", "fields"):
            if not np.array_equal(b[name], a[name]):
                raise ValueError(f"Paired initial state differs: {name}")
        start = max(0, round((before["duration_s"]-5.)/before["policy_dt_s"]))
        old, new = b["values"][start:], a["values"][start:]
        old_active, new_active = b["active"][start:], a["active"][start:]
        weight = b["weight_N"]
    success = lambda r: {x["env"] for x in r["episodes"] if x["final_balance_hold_s"] >= 1.-1e-5}
    old_success, new_success = success(before), success(after)
    old_bad = (old[:, :, 9] > 1.) & (old[:, :, 12] < .5) & old_active
    new_bad = (new[:, :, 9] > 1.) & (new[:, :, 12] < .5) & new_active
    persistent = old_bad.sum(0)/old_active.sum(0).clip(min=1) >= .5
    groups = {}
    for name, ids in (("previously_balanced", sorted(old_success)),
                      ("previously_base_supported_NORMAL", np.where(persistent)[0].tolist())):
        if not ids:
            groups[name] = dict(env_count=0, env_ids=[])
            continue
        forces = old[:, ids, 7:9]
        groups[name] = dict(env_count=len(ids), env_ids=ids,
            weight_N=weight[ids].tolist(),
            left_N_quantiles=np.quantile(forces[:, :, 0], [0., .01, .5, .99, 1.]).tolist(),
            right_N_quantiles=np.quantile(forces[:, :, 1], [0., .01, .5, .99, 1.]).tolist(),
            total_N_quantiles=np.quantile(forces.sum(-1), [0., .01, .5, .99, 1.]).tolist(),
            total_weight_ratio_quantiles=np.quantile(forces.sum(-1)/weight[None, ids], [0., .01, .5, .99, 1.]).tolist())
    def balanced_details(values, ids):
        if not ids:
            return {"count": 0}
        v = values[:, sorted(ids)]
        return dict(count=len(ids), median_height_m=float(np.median(v[:, :, 2])),
            median_planar_speed_m_s=float(np.median(v[:, :, 5])),
            median_max_leg_speed_rad_s=float(np.median(v[:, :, 6])),
            median_left_wheel_support_N=float(np.median(v[:, :, 7])),
            median_right_wheel_support_N=float(np.median(v[:, :, 8])),
            height_022_plus_minus_003_all_last5=int((np.abs(v[:, :, 2]-.22) <= .03).all(0).sum()),
            planar_speed_le_005_all_last5=int((v[:, :, 5] <= .05).all(0).sum()),
            leg_speed_le_05_all_last5=int((v[:, :, 6] <= .5).all(0).sum()))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(scope="40s single-seed diagnostic pairing, not formal 500Hz/20s acceptance",
        before=str(before_path), after=str(after_path),
        before_sha256=sha(before_path), after_sha256=sha(after_path),
        before_checkpoint_sha256=before["checkpoint_sha256"], after_checkpoint_sha256=after["checkpoint_sha256"],
        different_checkpoint_allowed=allow_different_checkpoint, exact_initial_state_match=True,
        wheel_force_statistics=groups, quantile_order=[0., .01, .5, .99, 1.],
        before_balance_count=len(old_success), after_balance_count=len(new_success),
        before_balanced_last5=balanced_details(old, old_success),
        after_balanced_last5=balanced_details(new, new_success),
        lost_balance_envs=sorted(old_success-new_success), added_balance_envs=sorted(new_success-old_success),
        before_persistent_base_supported_NORMAL=int(persistent.sum()),
        after_persistent_base_supported_NORMAL=int((new_bad.sum(0)/new_active.sum(0).clip(min=1) >= .5).sum()),
        after_any_base_supported_NORMAL=int(new_bad.any(0).sum()),
        previous_bad_envs_remaining_persistent=np.where(persistent &
            (new_bad.sum(0)/new_active.sum(0).clip(min=1) >= .5))[0].tolist(),
        parameter_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-different-checkpoint", action="store_true",
        help="Compare different models; changes cannot be attributed to one rule")
    args = parser.parse_args()
    result = compare(args.before, args.after, allow_different_checkpoint=args.allow_different_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "wheel_force_statistics"}, indent=2))
    if result["lost_balance_envs"]:
        raise SystemExit(2)
