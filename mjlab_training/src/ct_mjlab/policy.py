"""Versioned checkpoints and deterministic actor exports (no legacy encoder)."""
import copy
import json
from pathlib import Path
import hashlib
import torch
from tensordict import TensorDict
from .config import manifest, runner_cfg

def actor_model():
    from rsl_rl.models import MLPModel
    obs=TensorDict({'actor':torch.zeros(1,35),'critic':torch.zeros(1,78)},batch_size=[1])
    cfg=copy.deepcopy(runner_cfg()['actor']);cfg.pop('class_name')
    return MLPModel(obs,{'actor':['actor'],'critic':['critic']},'actor',6,**cfg)

def check_manifest(record):
    expected=manifest()
    for key in ('interface','source_commit','source_task','asset_sha256','recipe_sha256','poses_sha256',
                'control','actor_dim','critic_dim','action_dim','joints','physics_dt','decimation','height','gas_spring_force_n'):
        if json.loads(json.dumps(record.get(key)))!=json.loads(json.dumps(expected[key])):
            raise ValueError(f'Checkpoint interface mismatch: {key}; old Isaac/H7 models are incompatible')

def load_actor(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    check_manifest(saved.get('infos',{}).get('interface',{}))
    actor=actor_model();actor.load_state_dict(saved['actor_state_dict'],strict=True)
    return actor.to(device).eval()

def runner_class():
    from rsl_rl.runners import OnPolicyRunner
    class VersionedRunner(OnPolicyRunner):
        def save(self,path,infos=None):
            info=dict(infos or {});info['interface']=manifest()
            super().save(path,infos=info)
        def load(self,path,*args,**kwargs):
            saved=torch.load(path,map_location='cpu',weights_only=False)
            check_manifest(saved.get('infos',{}).get('interface',{}))
            return super().load(path,*args,**kwargs)
    return VersionedRunner

def export(path,output):
    import numpy as np
    import onnx
    import onnxruntime as ort
    actor=load_actor(path)
    # Fixed float32 [1,35] -> [1,6]. Source has no empirical normalization.
    model=torch.nn.Sequential(actor.obs_normalizer,actor.mlp).eval()
    output=Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,35),str(output),input_names=['observation'],
        output_names=['action'],opset_version=13,dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    sess=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(42);error=0.
    for _ in range(32):
        obs=rng.normal(size=(1,35)).astype(np.float32)
        expected=model(torch.from_numpy(obs)).detach().numpy()
        got=sess.run(None,{'observation':obs})[0]
        error=max(error,float(np.max(np.abs(expected-got))))
    if error>1e-5:raise ValueError(f'ONNX error {error}')
    record=manifest();record.update(checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),opset=13,max_abs_error=error,
        preprocessing='outside graph: source noise/delay/scales and real-joint order',
        h7_compatible=False)
    output.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record
