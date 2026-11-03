"""Strict physical-interface export; no compatibility with other 25-slot policies."""
import json
from pathlib import Path
import torch
from .policy import load_policy
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from .contract import CONTRACT, RESET_CONTRACT, RESET_V2_CONTRACT, LEGACY_RESET_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
from ct_mjlab.envs.fudan_ppo_walk.commands import LEGACY_COMMAND_SAMPLING

def history_metadata(reset_contract):
    if reset_contract not in (RESET_CONTRACT, RESET_V2_CONTRACT, LEGACY_RESET_CONTRACT):
        raise ValueError('Unsupported export reset/history contract')
    return dict(frames=5, order='oldest_to_newest', frame_size=25, clip=100.0, initial_fill='clean4_noisy1' if reset_contract == RESET_V2_CONTRACT else 'repeat_same_noisy_clipped_frame5', history_before_clip=reset_contract == RESET_V2_CONTRACT, runtime_sensor_noise='use_sensor_readings; do_not_add_simulation_noise_on_hardware')

class Policy(torch.nn.Module):

    def __init__(self, network):
        super().__init__()
        self.encoder = network.encoder
        self.actor = network.actor

    def forward(self, obs, obs_history):
        return self.actor(torch.cat((obs, self.encoder(obs_history)), -1))

def export(checkpoint, output):
    import onnx
    from onnx.reference import ReferenceEvaluator
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    network, metadata = load_policy(checkpoint)
    policy = Policy(network).eval()
    disturbances = from_record(metadata.get('disturbances'))
    path = output / 'policy.onnx'
    obs = torch.randn(1, 25)
    history = torch.randn(1, 125)
    torch.onnx.export(policy, (obs, history), path, opset_version=17, dynamo=False, input_names=['obs', 'obs_history'], output_names=['actions'])
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(None, {'obs': obs.numpy(), 'obs_history': history.numpy()})[0]
    with torch.no_grad():
        expected = policy(obs, history).numpy()
    import numpy as np
    error = float(np.abs(actual - expected).max())
    if error > 1e-05:
        raise RuntimeError(f'ONNX mismatch: {error}')
    (output / 'interface.json').write_text(json.dumps(dict(contract=metadata['contract'], reset_contract=metadata.get('reset_contract', LEGACY_RESET_CONTRACT), history_preprocessing=history_metadata(metadata.get('reset_contract', LEGACY_RESET_CONTRACT)), checkpoint_sha256=sha(checkpoint), onnx_sha256=sha(path), cpu_onnx_max_abs_error=error, input=dict(obs=[1, 25], obs_history=[1, 125]), output=dict(actions=[1, 6]), policy_dt_s=0.01, pd_dt_s=0.002, additional_action_delay_s=None if disturbances.action_delay else 0.0, additional_sensor_delay_s=None if disturbances.sensor_delay else 0.0, training_disturbances=disturbances.record(), training_command_sampling=metadata.get('command_sampling', LEGACY_COMMAND_SAMPLING), runtime_disturbances='simulation_only; queues and forces are outside ONNX; measure real hardware timing', h7='not_adapted_pending_owner_confirmation'), indent=2) + '\n')
