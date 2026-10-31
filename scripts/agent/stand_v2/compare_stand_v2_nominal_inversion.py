#!/usr/bin/env python3
"""Recompute paired500Hz zero-speed inversion branches and v3.1 balance from saved data."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.stand_v2.summarize_stand_v2_split_inversion import branches, label


GROUPS = ("same", "left_front_right_rear", "left_rear_right_front")


def measure(directory):
    report = json.loads((directory / "result.json").read_text())
    assert report["parameter_updates"] == 0 and report["actor_and_encoder_unchanged"]
    assert (report["physics_dt_s"], report["samples"], report["duration_s"]) == (.002, 20000, 40)
    with np.load(directory / "initial.npz") as source:
        initial = {k: source[k] for k in source.files}
    with np.load(directory / "episodes.npz") as source:
        episodes = {k: source[k] for k in source.files}
    with np.load(directory / "trajectories.npz") as source:
        trace, weight = source["physical"], source["weight"]
    fields = report["physical_fields"]
    field = lambda key: trace[..., fields.index(key)]
    active, finite = field("active") > .5, field("finite") > .5
    inverted = (field("gravity_z") > 0.) & active & finite
    ever = inverted.any(0)
    assert np.array_equal(ever, episodes["inverted"])
    assert int(ever.sum()) == report["ever_inverted"]
    assert np.array_equal((finite | ~active).all(0), episodes["finite"])
    assert np.array_equal(weight, initial["weight"])
    wheels = np.stack((field("left_support_N"), field("right_support_N")), -1)
    balanced = ((wheels >= np.maximum(.1 * weight, 5.)[None, :, None]).all(-1)
        & (wheels.sum(-1) >= .8 * weight) & (wheels.sum(-1) <= 1.2 * weight)
        & (field("base_force_N") <= 1.)
        & (np.maximum(field("roll_rad"), field("pitch_rad")) <= np.deg2rad(35.))
        & (field("roll_pitch_speed_rad_s") <= .5) & finite & active)
    hold = np.zeros(report["num_envs"], dtype=np.int64)
    for row in balanced:
        hold = np.where(row, hold + 1, 0)
    final = episodes["survived"] & episodes["finite"] & (hold >= 500)
    assert np.array_equal(final, episodes["final_balanced"])
    passed = episodes["ever_stood"] & final
    assert np.array_equal(passed, episodes["passed"])
    for key in ("ever_stood", "passed", "finite", "survived"):
        assert report[key] == int(episodes[key].sum())
    # 第0行已经积分2ms，首次倒置时间须加一个子步，不能直接用行号乘dt。
    first = np.where(ever, (inverted.argmax(0) + 1) * .002, np.nan)
    x = branches(directory / "scene.mjb", initial["qpos"])
    group_of_env = np.array([label(row) for row in x])
    grouped = {}
    for group in GROUPS:
        ids = np.flatnonzero(group_of_env == group)
        hit = ids[ever[ids]]
        times = [round(float(first[i]), 3) for i in hit]
        grouped[group] = dict(n=len(ids), ever_inverted=len(hit), inverted_ids=hit.tolist(),
            first_inversion_s=times, first_inversion_range_s=[min(times), max(times)] if times else None,
            ever_stood=int(episodes["ever_stood"][ids].sum()), passed=int(passed[ids].sum()))
    per_env = [dict(env=i, branch=str(group_of_env[i]), ever_inverted=bool(ever[i]),
        first_inversion_s=round(float(first[i]), 3) if ever[i] else None,
        first_stand_s=float(episodes["first_stand_s"][i]) if episodes["ever_stood"][i] else None,
        inverted_before_first_stand=bool(ever[i] and (not episodes["ever_stood"][i]
                                                    or first[i] < episodes["first_stand_s"][i])),
        ever_stood=bool(episodes["ever_stood"][i]), passed=bool(passed[i])) for i in range(len(ever))]
    summary = dict(checkpoint=report["checkpoint"], checkpoint_sha256=report["checkpoint_sha256"],
        n=len(ever), ever_inverted=int(ever.sum()), ever_stood=int(episodes["ever_stood"].sum()),
        passed=int(passed.sum()), survived=int(episodes["survived"].sum()), finite=int(episodes["finite"].sum()),
        inverted_then_passed=int((ever & passed).sum()), groups=grouped, per_environment=per_env)
    return summary, report, initial, episodes, group_of_env


def main(args):
    groups = []
    for conditions in ("nominal", "training"):
        a, ra, ia, ea, ga = measure(args.root / f"baseline-{conditions}-{args.seed}")
        b, rb, ib, eb, gb = measure(args.root / f"candidate-{conditions}-{args.seed}")
        for key in ("conditions", "seed", "num_envs", "duration_s", "physics_dt_s", "policy_dt_s", "xml_sha256", "sources"):
            assert ra[key] == rb[key], key
        assert ra["conditions"] == conditions and ra["seed"] == args.seed
        assert ia.keys() == ib.keys()
        assert all(np.array_equal(ia[key], ib[key]) for key in ia)
        assert np.array_equal(ga, gb)
        if conditions == "training" and args.seed == 43:
            assert [int((ga == g).sum()) for g in GROUPS] == [78, 20, 30]
        groups.append(dict(conditions=conditions, seed=args.seed, initial_arrays_equal=list(ia),
            baseline=a, candidate=b,
            avoided_inversion_ids=np.flatnonzero(ea["inverted"] & ~eb["inverted"]).tolist(),
            newly_inverted_ids=np.flatnonzero(~ea["inverted"] & eb["inverted"]).tolist(),
            lost_success_ids=np.flatnonzero(ea["passed"] & ~eb["passed"]).tolist(),
            gained_success_ids=np.flatnonzero(~ea["passed"] & eb["passed"]).tolist()))
    report = dict(statistical_recomputation_passed=True, groups=groups,
        old_success_retained=not any(row["lost_success_ids"] for row in groups),
        scope="Same initial states, zero-speed40s/500Hz; no new acceptance criterion",
        limitations=["One training continuation and one evaluation seed; no causal isolation of reward vs extra training",
                     "Training randomization with deterministic actions; no independentCPU/H7/real-robot approval",
                     "First inversion times use post-integration sampling: (index+1)*0.002s"])
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for row in groups:
        print(row["conditions"], {side: {key: row[side][key] for key in ("ever_inverted", "ever_stood", "passed")}
                                  for side in ("baseline", "candidate")},
              "lost_success_ids", row["lost_success_ids"])
    print("Statistical recomputation passed; old success retained:", report["old_success_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    main(parser.parse_args())
