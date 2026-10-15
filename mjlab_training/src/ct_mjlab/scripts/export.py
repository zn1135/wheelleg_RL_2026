"""Export a multimode checkpoint's deterministic actor to a fixed batch=1 ONNX graph.

Observation construction, mode flags, PD control and the 50 Hz schedule stay
outside the graph. A reference-vector check compares ONNX with PyTorch.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.multimode_observations import OBSERVATION_CONTRACT
from ct_mjlab.envs.chuanliantui.pd_action import CONTROL_CONTRACT

ACTION_ORDER = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
OBS_DIM, ACTION_DIM = 39, 6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_actor(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    infos = saved.get('infos') or {}
    for key, wanted in (('observation_contract', OBSERVATION_CONTRACT), ('control_contract', CONTROL_CONTRACT)):
        if infos.get(key) != wanted:
            raise ValueError(f'Checkpoint {key} is {infos.get(key)!r}, expected {wanted!r}')
    state = {k[len('mlp.'):]: v for k, v in saved['actor_state_dict'].items() if k.startswith('mlp.')}
    layers, index = [], 0
    while f'{index}.weight' in state:
        weight = state[f'{index}.weight']
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        layers.append(torch.nn.ELU())
        index += 2
    actor = torch.nn.Sequential(*layers[:-1]).eval()
    actor.load_state_dict(state, strict=True)
    if actor[0].in_features != OBS_DIM or actor[-1].out_features != ACTION_DIM:
        raise ValueError(f'Actor is {actor[0].in_features}->{actor[-1].out_features}, expected 39->6')
    return actor


def export(checkpoint, output, *, opset=13, samples=256, atol=1e-4):
    import onnx
    from onnx.reference import ReferenceEvaluator

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    actor = load_actor(checkpoint)
    torch.onnx.export(actor, torch.zeros(1, OBS_DIM), str(output), input_names=['obs'],
                      output_names=['actions'], opset_version=opset, do_constant_folding=True,
                      dynamic_axes=None, dynamo=False)
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    obs = (torch.rand(samples, OBS_DIM, generator=torch.Generator().manual_seed(0)) * 2 - 1).float()
    evaluator = ReferenceEvaluator(model)
    with torch.inference_mode():
        expected = np.stack([actor(row[None]).numpy()[0] for row in obs])
    actual = np.stack([evaluator.run(None, {'obs': row[None].numpy()})[0][0] for row in obs])
    max_abs = float(np.abs(expected - actual).max())
    if max_abs > atol:
        raise RuntimeError(f'ONNX mismatch {max_abs:.3e} exceeds {atol:.3e}')
    vectors = output.with_name(f'{output.stem}_reference_vectors.npz')
    np.savez_compressed(vectors, obs=obs.numpy(), actions=expected)
    record = dict(
        created_at=datetime.now().astimezone().isoformat(), observation_contract=OBSERVATION_CONTRACT,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        onnx=str(output), onnx_sha256=sha256(output), opset=opset,
        input=dict(name='obs', shape=[1, OBS_DIM], dtype='float32'),
        output=dict(name='actions', shape=[1, ACTION_DIM], dtype='float32', order=list(ACTION_ORDER)),
        numeric_check=dict(samples=samples, max_abs_diff=max_abs, atol=atol), reference_vectors=str(vectors),
        graph_scope='deterministic actor only; no observation construction, PD, clipping or motor mapping')
    output.with_name(f'{output.stem}_export.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opset', type=int, choices=(13, 18), default=13)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output, opset=args.opset,
                            samples=args.samples, atol=args.atol), indent=2))


if __name__ == '__main__':
    main()
