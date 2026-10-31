"""Export both the history estimator and actor, with paired numeric reference vectors."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .env_cfg import make_stand_v2_ppo_cfg
from .network import HistoryActor
from .observations import ACTOR_DIM, HISTORY_LENGTH, OBSERVATION_CONTRACT
from .runner import task_contracts
from .rewards import restore_tracking_reward_cfg
from .mdp import StandV2CommandCfg
from .mdp import CURRICULUM_VERSION, LEGACY_RECOVERY_CONTRACT, ZERO_COMMAND_CONTRACT, TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT
from ..chuanliantui.pd_action import CONTROL_CONTRACT


EXPORT_CONTRACT = "stand-v2-history160-single-input-v1"


def export(checkpoint, output, *, opset=13, samples=256, atol=1e-4):
    import onnx
    from onnx.reference import ReferenceEvaluator
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    infos = saved.get("infos") or {}
    command = infos.get("command_contract", ZERO_COMMAND_CONTRACT)
    if command not in (ZERO_COMMAND_CONTRACT, TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT):
        raise ValueError(f"Incompatible stand_v2 command_contract: {command!r}")
    reward_cfg = StandV2CommandCfg()
    restore_tracking_reward_cfg(reward_cfg, infos)
    position = command == POSITION_COMMAND_CONTRACT
    yaw = command in (YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
    contracts = task_contracts(command != ZERO_COMMAND_CONTRACT, command in (SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT),
        reward_cfg.tracking_weight_while_active, reward_cfg.tracking_tight_term,
        kernel_denominator=reward_cfg.tracking_tight_kernel_denominator, tight_weight=reward_cfg.tracking_tight_weight,
        yaw=yaw, **(infos['yaw_control_config'] if yaw else {}),
        position=position, **(infos['position_hold_config'] if position else {}))
    if yaw and infos['standup_curriculum'].get('yaw_control') != contracts['yaw_control_config']:
        raise ValueError('Yaw control curriculum configuration differs from export metadata')
    if position and infos['standup_curriculum'].get('position_hold') != contracts['position_hold_config']:
        raise ValueError('Position hold curriculum configuration differs from export metadata')
    if not position and ('position_hold_config' in infos or 'position_hold' in infos.get('standup_curriculum', {})):
        raise ValueError('Position hold metadata requires position hold contract')
    for key, expected in {"observation_contract": OBSERVATION_CONTRACT,
                          **{k:v for k,v in contracts.items() if k != "command_contract"},
                          "curriculum_version": CURRICULUM_VERSION,
                          "control_contract": CONTROL_CONTRACT}.items():
        if infos.get(key) != expected:
            raise ValueError(f"Incompatible stand_v2 export {key}: {infos.get(key)!r}")
    cfg = make_stand_v2_ppo_cfg().actor
    dummy = TensorDict({"actor": torch.zeros(1, ACTOR_DIM),
        "history": torch.zeros(1, ACTOR_DIM * HISTORY_LENGTH)}, batch_size=[1])
    actor = HistoryActor(dummy, {"actor": ["actor"]}, "actor", 6,
        hidden_dims=cfg.hidden_dims, activation=cfg.activation, distribution_cfg=cfg.distribution_cfg)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    model = actor.as_onnx().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, model.get_dummy_inputs(), str(output),
        input_names=model.input_names, output_names=model.output_names,
        opset_version=opset, dynamo=False)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    inputs = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
              for value in graph.graph.input}
    if inputs != {"observation_history": [1, ACTOR_DIM * HISTORY_LENGTH]}:
        raise RuntimeError(f"Unexpected history ONNX inputs: {inputs}")
    evaluator = ReferenceEvaluator(graph)
    generator = torch.Generator().manual_seed(42)
    history = torch.rand(samples, ACTOR_DIM * HISTORY_LENGTH, generator=generator) * 2 - 1
    observations = history[:, -ACTOR_DIM:].clone()
    with torch.inference_mode():
        actions, velocity = model(history)
        # Compare with the existing two-observation-group inference path as well.
        actor.eval()
        expected_actions = actor(TensorDict({"actor": observations, "history": history}, batch_size=[samples]))
        expected_velocity = actor.encoder(history) / 2.
    policy_errors = {"actions": float((actions - expected_actions).abs().max()),
                     "velocity": float((velocity - expected_velocity).abs().max())}
    if max(policy_errors.values()) > atol:
        raise RuntimeError(f"History export differs from checkpoint policy: {policy_errors}")
    outputs = [evaluator.run(None, {"observation_history": h[None].numpy()}) for h in history]
    errors = {"actions": float(np.abs(np.concatenate([v[0] for v in outputs])-actions.numpy()).max()),
              "velocity": float(np.abs(np.concatenate([v[1] for v in outputs])-velocity.numpy()).max())}
    if max(errors.values()) > atol:
        raise RuntimeError(f"History ONNX mismatch: {errors}")
    vectors = output.with_name(output.stem + "_reference_vectors.npz")
    np.savez_compressed(vectors, observation_history=history.numpy(),
        actions=actions.numpy(), velocity=velocity.numpy())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(created_at=datetime.now().astimezone().isoformat(),
        export_contract=EXPORT_CONTRACT,
        recovery_contract=infos.get("recovery_contract", LEGACY_RECOVERY_CONTRACT),
        observation_contract=OBSERVATION_CONTRACT, **contracts, curriculum_version=CURRICULUM_VERSION,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), onnx=str(output), onnx_sha256=sha(output),
        inputs=inputs, current_observation_slice=[-ACTOR_DIM, None],
        outputs={"actions": [1, 6], "velocity": [1, 3]}, numeric_check=errors,
        checkpoint_policy_check=policy_errors, opset=opset,
        velocity_units="body-frame center-of-mass m/s; encoder latent is twice this value",
        samples=samples, atol=atol, reference_vectors=str(vectors),
        graph_scope="encoder + actor; caller owns history, preprocessing, reset and PD")
    if command in (SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT):
        record["speed_command_config"] = infos["standup_curriculum"]["speed_curriculum"]["config"]
    if yaw:
        record["yaw_control_scope"] = ("Caller locks current heading when turn request becomes zero; "
            "wrapped heading error times heading_gain, limited yaw rate and acceleration, "
            "immediate zero and reference invalidation during recovery. This logic is outside ONNX.")
    if position:
        record['position_hold_scope'] = ("Caller locks measured xy only after zero raw vx request, yaw readiness and "
            "continuous low measured forward speed; projects current-minus-anchor onto current heading, applies "
            "hard dead zone, negative proportional gain, velocity limit and existing vx slew. Clear anchor on "
            "nonzero raw vx, loss of readiness or physical reset. Lateral error is diagnostic only. "
            "Position measurement and all outer-loop state are outside ONNX; training uses simulation truth.")
    output.with_name(output.stem + "_export.json").write_text(json.dumps(record, indent=2)+"\n")
    return record
