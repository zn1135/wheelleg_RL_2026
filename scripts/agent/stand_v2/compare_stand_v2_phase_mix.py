#!/usr/bin/env python3
"""Verify paired phase-mix evaluations and aggregate per-seed outcomes, without simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest


def training_summary(path, baseline_run):
    report = json.loads(path.read_text())
    assert report["completed"] and report["actual_optimizer_iterations"] == 500
    assert report["final_iteration"] == 2499 and report["hold_violations"] == 0
    candidate_run = Path(report["log_dir"])
    for name, sha in report["sources"].items():
        if name.startswith("mjlab_training/src/ct_mjlab/"):
            baseline_source = baseline_run/"snapshot"/"ct_mjlab"/Path(name).relative_to("mjlab_training/src/ct_mjlab")
            assert digest(baseline_source) == sha, name
    parent = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
    child = torch.load(candidate_run/"model_2499.pt", map_location="cpu", weights_only=False)
    optimizer_steps = {}
    for key in ("optimizer_state_dict", "encoder_optimizer_state_dict"):
        before = [int(v["step"]) for v in parent[key]["state"].values()]
        after = [int(v["step"]) for v in child[key]["state"].values()]
        assert set(after) == {min(before)+10000} and len(set(before)) == 1
        optimizer_steps[key] = dict(before=min(before), after=min(after), increment=10000)
    finite = all(torch.isfinite(v).all() for key in ("actor_state_dict", "critic_state_dict")
                 for v in child[key].values() if torch.is_tensor(v))
    assert finite
    records = np.asarray([r[1:] for r in report["monitor_records"] if r[0] == "train"])
    assert len(records) == 500 and (records[:, 1:4].sum(-1) > 0).all()
    completed, stood = report["completed_and_stood"]["train"]
    old = EventAccumulator(str(baseline_run), size_guidance={"scalars":0}); old.Reload()
    new = EventAccumulator(str(candidate_run), size_guidance={"scalars":0}); new.Reload()
    metrics = {}
    for label, events in (("baseline", old), ("candidate", new)):
        metrics[label] = {}
        for key in ("Metrics/standup/isaac_recovered", "Loss/value", "Loss/learning_rate", "Policy/mean_std"):
            values = [v for v in events.Scalars(key) if 2000 <= v.step <= 2499]
            metrics[label][key] = dict(count=len(values), mean=float(np.mean([v.value for v in values])),
                last=values[-1].value, minimum=min(v.value for v in values), maximum=max(v.value for v in values))
    # 每环境重置数不等，候选必须按实际结束回合数加权，不能直接平均每批日志。
    baseline_success = metrics["baseline"]["Metrics/standup/isaac_recovered"]["mean"]
    return dict(production_sources_equal_to_historical_run=True, actual_updates=500,
        final_iteration=2499, candidate_sha256=digest(candidate_run/"model_2499.pt"),
        checkpoint_parameters_finite=bool(finite), optimizer_steps=optimizer_steps,
        all_500_batches_have_first5s_samples=True, first5s_fraction=float(records[:, 1:4].sum()/records[:, 1:5].sum()),
        resets_per_batch=[int(records[:, 5].min()), int(records[:, 5].max())],
        completed_episodes=completed, stood_episodes=stood, weighted_success_rate=stood/completed,
        historical_2000_2499_success_rate=baseline_success,
        interpretation="Candidate first cycle includes prelude experience; compare fixed-policy evaluations separately",
        tensorboard_metrics=metrics)


def main(args):
    rows = []
    checkpoint_iterations = None
    for conditions in ("nominal", "training"):
        for seed in args.seeds:
            left = args.root/f"baseline-{conditions}-{seed}"
            right = args.root/f"candidate-{conditions}-{seed}"
            a, b = [json.loads((p/"result.json").read_text()) for p in (left, right)]
            if checkpoint_iterations is None:
                checkpoint_iterations = {label: torch.load(report["checkpoint"], map_location="cpu",
                    weights_only=False)["iter"] for label, report in (("baseline", a), ("candidate", b))}
            for key in ("conditions", "seed", "num_envs", "duration_s", "policy_dt_s",
                        "physics_dt_s", "samples", "xml_sha256", "sources"):
                assert a[key] == b[key], key
            for report in (a, b):
                assert report["parameter_updates"] == 0 and report["actor_and_encoder_unchanged"]
                assert digest(report["checkpoint"]) == report["checkpoint_sha256"]
                for source, sha in report["sources"].items():
                    assert digest(source) == sha, source
            with np.load(left/"initial.npz") as ia, np.load(right/"initial.npz") as ib:
                assert set(ia.files) == set(ib.files)
                for key in ia.files:
                    assert np.array_equal(ia[key], ib[key]), (conditions, seed, key)
            with np.load(left/"episodes.npz") as ea, np.load(right/"episodes.npz") as eb:
                for report, episodes in ((a, ea), (b, eb)):
                    passed = (episodes["ever_stood"] & episodes["survived"] & episodes["finite"]
                              & (episodes["final_hold_steps"] >= 500))
                    assert np.array_equal(passed, episodes["passed"])
                    for key in ("ever_stood", "passed"):
                        assert report[key] == int(episodes[key].sum())
                row = dict(conditions=conditions, seed=seed, n=a["num_envs"],
                    initial_state_and_physics_equal=True,
                    baseline_ever_stood=a["ever_stood"], candidate_ever_stood=b["ever_stood"],
                    baseline_passed=a["passed"], candidate_passed=b["passed"],
                    baseline_inverted=a["ever_inverted"], candidate_inverted=b["ever_inverted"],
                    baseline_stand_time_s=a["mean_first_stand_s"], candidate_stand_time_s=b["mean_first_stand_s"],
                    baseline_passed_speed_m_s=a["passed_last5_mean_speed"],
                    candidate_passed_speed_m_s=b["passed_last5_mean_speed"],
                    baseline_survived=a["survived"], candidate_survived=b["survived"],
                    baseline_finite=a["finite"], candidate_finite=b["finite"],
                    lost_ids=np.flatnonzero(ea["passed"] & ~eb["passed"]).tolist(),
                    gained_ids=np.flatnonzero(~ea["passed"] & eb["passed"]).tolist())
                rows.append(row)
    totals = []
    for conditions in ("nominal", "training"):
        group = [r for r in rows if r["conditions"] == conditions]
        total = {key:sum(r[key] for r in group) for key in ("n", "baseline_ever_stood",
            "candidate_ever_stood", "baseline_passed", "candidate_passed", "baseline_inverted", "candidate_inverted")}
        total.update(conditions=conditions, lost=sum(len(r["lost_ids"]) for r in group),
            gained=sum(len(r["gained_ids"]) for r in group))
        totals.append(total)
    result = dict(verified=True, rows=rows, totals=totals, checkpoint_iterations=checkpoint_iterations,
        comparison_status="regressed_rejected" if any(r["lost_ids"] for r in rows) else "no_old_success_lost",
        limitations=["One training run, seed42: evaluation seeds are not independent training repetitions",
            f"Compared baseline{checkpoint_iterations['baseline']} with candidate{checkpoint_iterations['candidate']}; not a matched second training run",
            "Candidate has 40s physical prelude without optimizer updates; RNG sequence differs",
            "Training-condition evaluation uses deterministic actions, not stochastic training actions",
            "40s/500Hz protocol differs from earlier 40s/100Hz and 20s/500Hz reports"],
        source_sha256=digest(__file__))
    if args.training_report:
        result["training"] = training_summary(args.training_report, args.baseline_run)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if result["comparison_status"] == "regressed_rejected":
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    main(parser.parse_args())
