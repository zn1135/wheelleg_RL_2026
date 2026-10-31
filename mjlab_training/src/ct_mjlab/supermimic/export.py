"""Export only the estimator+actor; reject unrelated PPO or asset contracts."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .networks import make_actor, act
from .learner import ACTOR_TRANSFER_CONTRACTS, asset_digest
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.runner import contracts


def export(checkpoint, output):
    import onnx
    from onnx.reference import ReferenceEvaluator
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    vectors = output.with_name(output.stem+'_reference_vectors.npz')
    metadata = output.with_name(output.stem+'_export.json')
    for path in (output,vectors,metadata):
        if path.exists():
            raise FileExistsError(path)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (saved.get('algorithm_contract') not in ACTOR_TRANSFER_CONTRACTS
            or saved.get('infos') != contracts(command_cfg('walk'))
            or saved.get('asset_sha256') != asset_digest()):
        raise ValueError('Not a compatible SuperMimic walk checkpoint')
    actor = make_actor('cpu').eval()
    actor.load_state_dict(saved['actor'])
    model = actor.as_onnx().eval()
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,model.get_dummy_inputs(),str(output),opset_version=13,dynamo=False,
        input_names=model.input_names,output_names=model.output_names)
    graph = onnx.load(str(output)); onnx.checker.check_model(graph)
    evaluator = ReferenceEvaluator(graph)
    history = torch.rand(32,125,generator=torch.Generator().manual_seed(43))*2-1
    with torch.no_grad():
        action, velocity = model(history)
        torch.testing.assert_close(action,act(actor,history),atol=1e-6,rtol=1e-6)
    result = [evaluator.run(None,{'observation_history':row[None].numpy()}) for row in history]
    errors = dict(actions=float(np.abs(np.concatenate([x[0] for x in result])-action.numpy()).max()),
                  velocity=float(np.abs(np.concatenate([x[1] for x in result])-velocity.numpy()).max()))
    if max(errors.values())>1e-4:
        raise RuntimeError(f'ONNX numerical mismatch: {errors}')
    np.savez_compressed(vectors,observation_history=history.numpy(),actions=action.numpy(),velocity=velocity.numpy())
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(algorithm_contract=saved['algorithm_contract'],infos=saved['infos'],
        asset_sha256=saved['asset_sha256'],checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        onnx=str(output),onnx_sha256=digest(output),inputs={'observation_history':[1,125]},
        outputs={'actions':[1,6],'velocity':[1,3]},numeric_max_error=errors,
        policy_dt_s=.01,physics_dt_s=.002,h7_owner_review='待确认',
        behavior_acceptance='未验收；导出检查只确认数值和接口',
        scope='Experimental walk-only raw actions; existing CPU/PPO loaders are not adapted')
    metadata.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record
