#!/usr/bin/env python3
"""Historical 39-to-41 migration; target is the frozen r25/r26 velocity profile."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mjlab_training/src"))


def migrate_state(source):
    info = source.get("infos") or {}
    if (info.get("experiment_profile_contract") != "chuanliantui-stand-39-82-v1"
            or info.get("observation_contract") != "chuanliantui-huananhu-multimode-39-82-v1"
            or info.get("physics_dt_s") != .002 or info.get("policy_dt_s") != .01):
        raise ValueError("Migration accepts only the previous own-task 39/82 checkpoint")
    result = copy.deepcopy(source)
    weight = result["actor_state_dict"]["mlp.0.weight"]
    if weight.shape != (256, 39):
        raise ValueError(f"Unexpected actor input shape {weight.shape}")
    result["actor_state_dict"]["mlp.0.weight"] = torch.nn.functional.pad(weight, (0, 2))
    moments = []
    # Parameter order/count do not change; only the input weight gains two columns.
    for parameter_id, state in result["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.shape == (256, 39):
                state[key] = torch.nn.functional.pad(value, (0, 2))
                moments.append({"parameter_id": parameter_id, "moment": key})
    if sorted(x["moment"] for x in moments) != ["exp_avg", "exp_avg_sq"]:
        raise ValueError("Expected exactly the actor input weight's two Adam moments")
    result["infos"] = info | {"observation_contract": "chuanliantui-stand-41-82-planar-velocity-v1",
        "experiment_profile_contract": "chuanliantui-stand-41-82-v1",
        "velocity_input_migration": {"from_actor_dim": 39, "to_actor_dim": 41,
            "new_weight_columns_zero": True, "adam_moments_preserved_and_padded": moments,
            "new_velocity_inputs_trained": False, "source_recipe_version": info.get("recipe_version")}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Migration must preserve all existing checkpoints and reports")
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    result = migrate_state(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.report.write_text(json.dumps({"source": str(args.input.resolve()), "source_sha256": digest(args.input),
        "output": str(args.output.resolve()), "output_sha256": digest(args.output),
        "migration": result["infos"]["velocity_input_migration"], "optimizer_reset": False,
        "training_updates": 0, "scope": "Tensor migration only; runtime equivalence requires finite checks."}, indent=2)+"\n")


if __name__ == "__main__":
    main()
