#!/usr/bin/env python3
"""Export the historical Multimode model_3023 deterministic actor to ONNX.

This utility reads the versioned checkpoint directly and exports only its actor.
Observation construction, recovery state, PD control and physical simulation stay outside the ONNX graph.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = ROOT / "artifacts" / "multimode-v1-replay-20261014"
TASK = "Chuanliantui-Huananhu-Multimode-v0"
CHECKPOINT = IMPORT_ROOT / "models" / "model_3023.pt"
CHECKPOINT_SHA256 = "624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a"
ACTION_ORDER = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
OBSERVATION_CONTRACT = "chuanliantui-huananhu-multimode-39-82-v1"
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> None:
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("This tool only exports the expected historical model_3023 checkpoint")


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default) + "\n")



class Actor(torch.nn.Module):
    """The exact deterministic 39-D actor body stored in model_3023."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(39, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 6),
        )

    def forward(self, obs):
        return self.mlp(obs)


def load_actor(checkpoint: Path):
    import torch
    validate_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = saved.get("infos", {})
    if info.get("observation_contract") != OBSERVATION_CONTRACT:
        raise ValueError("Checkpoint does not declare the model_3023 39-D observation contract")
    state = {name: value for name, value in saved["actor_state_dict"].items() if name.startswith("mlp.")}
    actor = Actor().eval()
    actor.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Checkpoint actor contains a non-finite parameter")
    return actor


def export_onnx(args) -> None:
    import onnx
    import torch
    from onnx.reference import ReferenceEvaluator

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = load_actor(checkpoint)
    probe = torch.zeros((1, 39), dtype=torch.float32)
    torch.onnx.export(reference, probe, str(output), input_names=["obs"], output_names=["actions"],
                      opset_version=args.opset, do_constant_folding=True,
                      dynamic_axes=None, export_params=True, dynamo=False)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    main_opset = next((int(item.version) for item in model.opset_import if item.domain in ("", "ai.onnx")), None)
    if main_opset != args.opset:
        raise RuntimeError(f"Exported ONNX opset {main_opset}, expected {args.opset}")
    onnx.save_model(model, str(output), save_as_external_data=False)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    input_shape = [int(d.dim_value) for d in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [int(d.dim_value) for d in model.graph.output[0].type.tensor_type.shape.dim]
    if (model.graph.input[0].name, input_shape, model.graph.output[0].name, output_shape) != (
            "obs", [1, 39], "actions", [1, 6]):
        raise RuntimeError("Exported ONNX graph does not have fixed [1,39] -> [1,6] interface")
    rng = torch.Generator().manual_seed(0)
    samples = (torch.rand((args.samples, 39), generator=rng) * 2.0 - 1.0).float()
    evaluator = ReferenceEvaluator(model)
    expected, actual = [], []
    with torch.inference_mode():
        for sample in samples:
            single = sample.unsqueeze(0)
            expected.append(reference(single).numpy()[0])
            actual.append(evaluator.run(None, {"obs": single.numpy()})[0][0])
    expected_np, actual_np = np.stack(expected), np.stack(actual)
    max_abs = float(np.abs(expected_np - actual_np).max())
    if max_abs > args.atol:
        raise RuntimeError(f"ONNX mismatch {max_abs:.3e} exceeds {args.atol:.3e}")
    np.savez_compressed(output.with_name(f"{output.stem}_reference_vectors.npz"),
                        obs=samples.numpy(), actions=expected_np)
    record = {
        "created_at": datetime.now().astimezone().isoformat(),
        "task": TASK,
        "observation_contract": OBSERVATION_CONTRACT,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "onnx": str(output), "onnx_sha256": sha256(output), "onnx_bytes": output.stat().st_size,
        "opset": args.opset, "input": {"name": "obs", "shape": [1, 39], "dtype": "float32"},
        "output": {"name": "actions", "shape": [1, 6], "dtype": "float32", "order": list(ACTION_ORDER)},
        "fixed_batch": True, "weights_embedded": True,
        "numeric_check": {"samples": args.samples, "max_abs_diff": max_abs, "atol": args.atol},
        "reference_vectors": str(output.with_name(f"{output.stem}_reference_vectors.npz")),
        "graph_scope": "deterministic actor only; no observation construction, PD, torque clipping or motor mapping",
    }
    write_json(output.with_name(f"{output.stem}_export.json"), record)
    print("EXPORT " + json.dumps(record, ensure_ascii=False, default=json_default), flush=True)

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--opset", type=int, choices=(13, 18), required=True)
    result.add_argument("--samples", type=int, default=256)
    result.add_argument("--atol", type=float, default=1e-4)
    return result

def main() -> None:
    args = parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {args.checkpoint}")
    export_onnx(args)


if __name__ == "__main__":
    main()
