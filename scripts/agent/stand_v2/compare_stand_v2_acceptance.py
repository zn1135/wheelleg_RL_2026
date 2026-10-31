#!/usr/bin/env python3
"""Verify recorded 500 Hz stand_v2 verdicts and compare identical initial states."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.stand_v2.assess_chuanliantui_stand_v2 import CRITERION, FIELDS, digest


def verify(path):
    r = json.loads(path.read_text())
    assert r["criterion"] == CRITERION and r["physics_dt_s"] == .002
    assert r["policy_dt_s"] == .01 and r["duration_s"] == 20.
    assert not r["drift_is_failure_gate"] and r["parameter_updates"] == 0
    assert r["actor_and_encoder_unchanged"]
    assert digest(r["checkpoint"]) == r["checkpoint_sha256"]
    assert digest(path.parent/"physical.npz") == r["physical_sha256"]
    assert digest(path.parent/"qpos.npz") == r["qpos_sha256"]
    assert digest(Path(__file__).with_name("assess_chuanliantui_stand_v2.py")) == r["source_sha256"]
    with np.load(path.parent/"physical.npz") as saved:
        v, weight = saved["values"], saved["weight_N"]
        assert tuple(saved["fields"]) == FIELDS
        assert v.shape == (10000, r["num_envs"], len(FIELDS))
        assert np.allclose(saved["time_s"], (np.arange(10000)+1)*.002)
        initial = {k: saved[k].copy() for k in ("initial_qpos", "initial_qvel", "weight_N")}
    with np.load(path.parent/"qpos.npz") as saved:
        assert saved["qpos"].shape[:2] == (2001, r["num_envs"])
        assert np.array_equal(saved["qpos"][0], initial["initial_qpos"])
    # Independent last-500-ticks conjunction: speed and displacement are absent.
    tail = v[-500:]
    valid = ((tail[:, :, :2] >= np.maximum(weight*.1, 5.)[None, :, None]).all(-1)
        & (tail[:, :, :2].sum(-1) >= .8*weight)
        & (tail[:, :, :2].sum(-1) <= 1.2*weight)
        & (tail[:, :, 2] <= 1.) & (tail[:, :, 3:5].max(-1) <= np.deg2rad(35.))
        & (tail[:, :, 5] <= .5) & (tail[:, :, 12] > .5) & (tail[:, :, 13] > .5))
    assert not (np.diff(v[:, :, 13], axis=0) > 0).any(), "Failed sample became active again"
    rollout = json.loads((path.parent/"rollout/result.json").read_text())
    for i, row in enumerate(r["rows"]):
        full = bool((v[:, i, 13] > .5).all() and rollout["episodes"][i]["survived_to_end"]
                    and (v[v[:, i, 13] > .5, i, 12] > .5).all())
        balanced = bool(full and valid[:, i].all())
        assert row["env"] == i and row["full_20s"] == full
        assert row["final_balanced"] == balanced
        assert row["ever_stood"] == rollout["episodes"][i]["ever_isaac_stood"]
        assert row["passed"] == bool(balanced and row["ever_stood"])
        assert np.isclose(row["last5_median_speed_m_s"],
                          np.median(np.linalg.norm(v[-2500:, i, 7:9], axis=-1)))
    assert r["passed"] == sum(x["passed"] for x in r["rows"])
    for source, sha in r["sources"].items():
        assert digest(source) == sha, source
    return r, initial


def main(args):
    before, bi = verify(args.before)
    after, ai = verify(args.after)
    for key in ("criterion", "num_envs", "seed", "duration_s", "physics_dt_s", "policy_dt_s", "xml_sha256"):
        assert before[key] == after[key], key
    for key in bi:
        assert np.array_equal(bi[key], ai[key]), key
    old = {x["env"] for x in before["rows"] if x["passed"]}
    new = {x["env"] for x in after["rows"] if x["passed"]}
    result = dict(criterion=CRITERION, before=str(args.before), after=str(args.after),
        before_checkpoint_sha256=before["checkpoint_sha256"],
        after_checkpoint_sha256=after["checkpoint_sha256"],
        exact_initial_states_and_weight_match=True, all_recorded_verdicts_verified=True,
        source_and_checkpoint_hashes_verified=True, before_passed=len(old), after_passed=len(new),
        lost_envs=sorted(old-new), added_envs=sorted(new-old),
        drift_is_failure_gate=False, status="regressed_rejected" if old-new else "old_successes_preserved")
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if old-new:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    main(parser.parse_args())
