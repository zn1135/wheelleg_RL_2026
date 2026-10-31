#!/usr/bin/env python3
"""Verify the bounded std-floor continuation; no simulation or optimizer updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.modules.distribution import GaussianDistribution

from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest


def main(args):
    result = json.loads((args.training/"result.json").read_text())
    assert result["completed"] and result["actual_optimizer_iterations"] == 500
    assert result["start_iteration"] == 2500 and result["final_iteration"] == 2999
    assert result["exploration"] == dict(min_std=.05, entropy_coef=.005)
    assert result["std_floor_violations"] == result["hold_violations"] == 0
    assert min(result["lowest_observed_std"]) >= .05-1e-8
    assert result["restored_model_parameters_exact"]
    assert result["prelude"]["actor_unchanged"] and result["prelude"]["optimizers_unchanged"]
    parent = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    target = Path(result["log_dir"])/"model_2999.pt"
    child = torch.load(target, map_location="cpu", weights_only=False)
    assert digest(result["checkpoint"]) == result["checkpoint_sha256"]
    assert parent["iter"] == 2499 and child["iter"] == 2999
    assert child["infos"]["phase_mix_experiment"]["exploration"] == result["exploration"]
    steps = {}
    for key in ("optimizer_state_dict", "encoder_optimizer_state_dict"):
        before = {int(v["step"]) for v in parent[key]["state"].values()}
        after = {int(v["step"]) for v in child[key]["state"].values()}
        assert before == {50000} and after == {60000}
        steps[key] = dict(before=50000, after=60000)
    for key in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.isfinite(v).all() for v in child[key].values() if torch.is_tensor(v))
        assert any(not torch.equal(v, parent[key][name]) for name, v in child[key].items())
    # 下限仅作用于随机动作采样；确定性回放仍直接使用策略均值。
    distribution = GaussianDistribution(6, std_range=(.05, 1e6))
    distribution.load_state_dict({"std_param": child["actor_state_dict"]["distribution.std_param"]})
    mean = torch.linspace(-1., 1., 48).reshape(8, 6)
    distribution.update(mean)
    assert torch.equal(distribution.deterministic_output(mean), mean)
    assert (distribution.std >= .05-1e-8).all()
    records = np.asarray([v[1:] for v in result["monitor_records"] if v[0] == "train"])
    assert len(records) == 500 and (records[:, 1:4].sum(-1) > 0).all()
    manifest_equal = True
    for name, sha in result["sources"].items():
        if name.startswith("mjlab_training/src/ct_mjlab/"):
            assert digest(name) == sha
            old = args.parent_source/Path(name).relative_to("mjlab_training/src/ct_mjlab")
            assert digest(old) == sha
    completed, stood = result["completed_and_stood"]["train"]
    report = dict(verified=True, actual_updates=500, iterations=[2500, 2999],
        parent_sha256=digest(result["checkpoint"]), checkpoint=str(target.resolve()),
        checkpoint_sha256=digest(target), exploration=result["exploration"],
        native_sampling_floor_verified=True, deterministic_output_unaffected=True,
        std_floor_violations=0, lowest_observed_std=result["lowest_observed_std"],
        final_raw_std=distribution.std_param.detach().tolist(),
        final_effective_std=distribution.std[0].detach().tolist(),
        optimizer_steps=steps, production_sources_equal_to_parent=manifest_equal,
        restored_model_parameters_exact=True, parameters_finite_and_updated=True,
        all_500_batches_have_getup_samples=True,
        first5s_fraction=float(records[:, 1:4].sum()/records[:, 1:5].sum()),
        reset_range=[int(records[:, 5].min()), int(records[:, 5].max())],
        completed_episodes=completed, stood_episodes=stood,
        weighted_training_success_rate=stood/completed,
        scope="Budget, checkpoint, sampling floor and physical phase distribution; behavior measured separately",
        source_sha256=digest(__file__))
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
