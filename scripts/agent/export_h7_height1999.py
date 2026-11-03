"""Export the hash-pinned historical height1999 deterministic actor without mjlab.

Only archived configuration classes are replaced while unpickling metadata;
tensor reconstruction uses PyTorch. No environment or training callback executes.
"""
import argparse
from datetime import datetime
import hashlib
import json
import pickle
import subprocess
import types
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

EXPECTED = '1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027'


class Metadata:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('mjlab', 'ct_mjlab', 'mujoco', 'warp')):
            return type(name, (Metadata,), {})
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('This exporter accepts only the audited historical height1999')
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False,
                       pickle_module=types.SimpleNamespace(Unpickler=MetadataUnpickler, __name__='pickle'))
    info = saved['infos']
    env = info['environment']
    actor_cfg = info['interface']['ppo']['actor']
    assert list(actor_cfg['hidden_dims']) == [256, 128, 64]
    assert actor_cfg['activation'] == 'elu' and not actor_cfg['obs_normalization']
    model = torch.nn.Sequential(torch.nn.Linear(35, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 128), torch.nn.ELU(),
                                torch.nn.Linear(128, 64), torch.nn.ELU(), torch.nn.Linear(64, 6)).eval()
    weights = saved['actor_state_dict']
    assert set(weights) == {'distribution.std_param'} | {'mlp.' + k for k in model.state_dict()}
    model.load_state_dict({k: weights['mlp.' + k] for k in model.state_dict()}, strict=True)
    args.output.mkdir(parents=True)
    path = args.output / 'model_1999_opset13.onnx'
    torch.onnx.export(model, torch.zeros(1, 35), str(path), input_names=['observation'],
                      output_names=['action'], opset_version=13, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 35)).astype(np.float32)
    x[0] = 0
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    actual = np.concatenate([session.run(None, {'observation': row[None]})[0] for row in x])
    error = float(np.abs(actual - expected).max())
    if error > 1e-5:
        raise ValueError(f'ONNX error {error} exceeds historical export tolerance')
    np.savez(args.output / 'reference_vectors.npz', observation=x, action=expected)
    np.savetxt(args.output / 'reference_inputs.csv', x, delimiter=',')
    np.savetxt(args.output / 'reference_actions.csv', expected, delimiter=',')
    record = dict(created_at=datetime.now().astimezone().isoformat(), checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=EXPECTED,
                  onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  training_interface=info['experiment_profile_contract'],
                  observation_contract=info['observation_contract'], control_contract=info['control_contract'],
                  training_environment=env, actor_config=actor_cfg,
                  input_shape=[1, 35], output_shape=[1, 6], samples=64, max_abs_error=error,
                  input=dict(name='observation', shape=[1, 35], dtype='float32'),
                  output=dict(name='action', shape=[1, 6], dtype='float32', order=list(env['actions']['pd']['policy_joint_order'])),
                  opset=13, random_seed=42, atol=1e-5,
                  graph_scope='deterministic actor only; preprocessing, PD and motor mapping stay outside graph',
                  exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python_version=platform.python_version(), numpy_version=np.__version__,
                  torch_version=torch.__version__, onnx_version=onnx.__version__,
                  ort_version=ort.__version__, export_only=True, hardware_validated=False,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    (args.output / 'delivery.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps(dict(path=str(path), max_abs_error=error, samples=64)))


if __name__ == '__main__':
    main()
