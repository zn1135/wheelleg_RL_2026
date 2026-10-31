"""One ONNX per role, with shared 125-history input and no switch in the graph."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .env_cfg import command_cfg, make_ppo_cfg
from .network import HistoryActor
from .runner import validate


def load_policy(checkpoint, role, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    infos = saved.get('infos') or {}
    validate(infos, command_cfg(role))
    dummy = TensorDict(dict(actor=torch.zeros(1,25), history=torch.zeros(1,125)), batch_size=[1])
    cfg = make_ppo_cfg(role).actor
    actor = HistoryActor(dummy, {'actor':['actor']}, 'actor', 6, hidden_dims=cfg.hidden_dims,
        activation=cfg.activation, distribution_cfg=cfg.distribution_cfg)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    # Match native inference: disable observation-normalizer updates and exploration.
    actor.to(device).eval()
    return actor, infos


def export(checkpoint, output, *, opset=13, samples=256, atol=1e-4):
    import onnx
    from onnx.reference import ReferenceEvaluator
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    infos = torch.load(checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
    role = infos.get('policy_role')
    if role not in ('getup', 'walk'):
        raise ValueError(f'Not a split-policy checkpoint: {role!r}')
    actor, infos = load_policy(checkpoint, role)
    model = actor.as_onnx().eval()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, model.get_dummy_inputs(), str(output), input_names=model.input_names,
        output_names=model.output_names, opset_version=opset, dynamo=False)
    graph = onnx.load(str(output)); onnx.checker.check_model(graph)
    inputs = {v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.graph.input}
    assert inputs == {'observation_history':[1,125]}
    history = torch.rand(samples,125,generator=torch.Generator().manual_seed(42))*2-1
    with torch.inference_mode():
        actions, velocity = model(history)
        direct = actor(TensorDict(dict(actor=history[:,-25:], history=history), batch_size=[samples]))
    assert float((actions-direct).abs().max()) <= atol
    evaluator = ReferenceEvaluator(graph)
    actual = [evaluator.run(None, {'observation_history':v[None].numpy()}) for v in history]
    errors = dict(actions=float(np.abs(np.concatenate([v[0] for v in actual])-actions.numpy()).max()),
        velocity=float(np.abs(np.concatenate([v[1] for v in actual])-velocity.numpy()).max()))
    if max(errors.values()) > atol:
        raise RuntimeError(f'Split ONNX mismatch: {errors}')
    digest = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    vectors = output.with_name(output.stem+'_reference_vectors.npz')
    np.savez_compressed(vectors, observation_history=history.numpy(), actions=actions.numpy(), velocity=velocity.numpy())
    record = dict(policy_role=role, observation_contract=infos['observation_contract'],
        switch_contract=infos['switch_contract'], checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        onnx=str(output), onnx_sha256=digest(output), inputs=inputs,
        outputs=dict(actions=[1,6], velocity=[1,3]), numeric_check=errors, samples=samples, atol=atol,
        scope='One independent encoder+actor; caller owns role selection, physical history, commands and PD')
    if role == 'walk':
        record.update(command_contract=infos['command_contract'],
                      height_range=infos['command_config']['height_range'])
    else:
        # 平滑有跨拍状态，由图外动作执行器维护，ONNX仍输出原始动作。
        record.update(action_filter_contract=infos['action_filter_contract'],
                      action_filter=infos['action_filter'])
    output.with_name(output.stem+'_export.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
