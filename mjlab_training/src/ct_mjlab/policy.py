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

def checkpoint_is_legacy(path):
    if path is None:return False
    saved=torch.load(path,map_location='cpu',weights_only=False)
    return saved.get('infos',{}).get('interface',{}).get('interface') == 'chuanliantui-mjlab-huananhu-v14-flat-standup-r5'

def checkpoint_uses_constant_modes(path):
    if path is None:return False
    saved=torch.load(path,map_location='cpu',weights_only=False)
    return saved.get('infos',{}).get('interface',{}).get('interface') == 'chuanliantui-mjlab-wheelleg-recover-r6'

def checkpoint_policy_joint_order(path):
    from .reference_training.model import ACTUATED_JOINTS, POLICY_JOINTS
    if path is None:return POLICY_JOINTS
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=saved.get('infos',{}).get('interface',{})
    check_manifest(record)
    if record.get('interface') in ('chuanliantui-mjlab-wheelleg-recover-r6',
                                  'chuanliantui-mjlab-wheelleg-recover-r7'):
        return ACTUATED_JOINTS
    return POLICY_JOINTS

def checkpoint_knee_limit_weight(path):
    if path is None:return None
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=saved.get('infos',{}).get('interface',{})
    check_manifest(record)
    return record.get('environment',{}).get('rewards',{}).get('knee_limit',{}).get('weight')

def checkpoint_reset_config(path):
    """Restore the initial-state distribution recorded with historical policies."""
    if path is None:return None
    from .reference_training.huananhu_env_cfg import RESET_CONFIG_FIELDS
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=saved.get('infos',{}).get('interface',{})
    check_manifest(record)
    cfg=record['environment']['commands']['standup']
    return {name:cfg[name] for name in RESET_CONFIG_FIELDS}

def checkpoint_height_gate(path):
    """Replay each policy with its recorded velocity-reward height gate."""
    if path is None:return None
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=saved.get('infos',{}).get('interface',{})
    check_manifest(record)
    if record.get('interface')=='chuanliantui-mjlab-huananhu-v14-flat-standup-r5':return None
    # r6/r7/r8 predate reward metadata; their archived kernels used 3–6 cm.
    return tuple(record.get('training_rewards', {'height_gate':(.03,.06)})['height_gate'])


def checkpoint_command_config(path):
    """Replay recorded commands, including the zero-velocity r6–r13 profiles."""
    if path is None:return None
    from .reference_training.huananhu_env_cfg import COMMAND_CONFIG_FIELDS
    from .reference_training.huananhu_mdp import HuananhuCommandCfg
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=saved.get('infos',{}).get('interface',{})
    check_manifest(record)
    if checkpoint_is_legacy(path):return None
    cfg=record['environment']['commands']['standup']
    defaults=HuananhuCommandCfg()
    return {name:cfg.get(name,getattr(defaults,name)) for name in COMMAND_CONFIG_FIELDS}


def check_manifest(record, *, training=False):
    from .config import legacy_manifest
    legacy=record.get('interface')=='chuanliantui-mjlab-huananhu-v14-flat-standup-r5'
    if legacy:
        if training:raise ValueError('Legacy r5 checkpoint cannot resume r14 training')
        expected=legacy_manifest()
        for key in expected.keys()-{'training_rewards'}:
            if json.loads(json.dumps(record.get(key)))!=json.loads(json.dumps(expected[key])):
                raise ValueError(f'Legacy checkpoint interface mismatch: {key}')
    elif record.get('interface') in ('chuanliantui-mjlab-wheelleg-recover-r6',
                                    'chuanliantui-mjlab-wheelleg-recover-r7',
                                    'chuanliantui-mjlab-wheelleg-recover-r8',
                                    'chuanliantui-mjlab-wheelleg-recover-r9',
                                    'chuanliantui-mjlab-wheelleg-recover-r10',
                                    'chuanliantui-mjlab-wheelleg-recover-r11',
                                    'chuanliantui-mjlab-wheelleg-recover-r12',
                                    'chuanliantui-mjlab-wheelleg-recover-r13'):
        version=record['interface'].rsplit('-',1)[-1]
        if training:raise ValueError(f'Historical {version} checkpoint cannot resume r14 training without explicit migration')
        expected=json.loads(Path(__file__).with_name('reference_training').joinpath(f'{version}_interface.json').read_text())
        if json.loads(json.dumps(record))!=expected:
            raise ValueError(f'Historical {version} checkpoint manifest mismatch')
        current=manifest()
        for key in ('asset_sha256','poses_sha256','reset_bank_sha256'):
            if record[key]!=current[key]:raise ValueError(f'Historical {version} robot mismatch: {key}')
    elif json.loads(json.dumps(record))!=json.loads(json.dumps(manifest())):
        raise ValueError('r14 checkpoint manifest mismatch; explicit migration required')

def load_actor(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    check_manifest(saved.get('infos',{}).get('interface',{}))
    actor=actor_model();actor.load_state_dict(saved['actor_state_dict'],strict=True)
    return actor.to(device).eval()

def runner_class():
    from .reference_training.entry import runner_class as current
    return current()

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
    saved=torch.load(path,map_location='cpu',weights_only=False)
    record=copy.deepcopy(saved.get('infos',{}).get('interface',{}))
    record.setdefault('training_rewards',
        {'profile':'legacy-scut-v14-flat-v0-ungated', 'height_gate':None} if checkpoint_is_legacy(path) else {'profile':'wheelleg-recover-v12','height_gate':[.03,.06]})
    record.update(checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),opset=13,max_abs_error=error,
        preprocessing='outside graph: source noise/delay/scales and real-joint order',
        h7_compatible=False)
    output.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record
