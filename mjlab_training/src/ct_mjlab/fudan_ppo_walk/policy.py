"""Load the frozen physical policy for GPU evaluation and ONNX export."""
import torch
from .contract import CONTRACT, RESET_CONTRACT, RESET_V2_CONTRACT, LEGACY_RESET_CONTRACT, make_network, validate_network_contract
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record

def load_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint['contract'] != CONTRACT:
        raise ValueError('Checkpoint physical interface differs')
    if checkpoint.get('reset_contract', LEGACY_RESET_CONTRACT) not in (RESET_CONTRACT, RESET_V2_CONTRACT, LEGACY_RESET_CONTRACT):
        raise ValueError('Checkpoint reset/history contract is unsupported')
    from_record(checkpoint.get('disturbances'))
    validate_network_contract()
    policy = make_network().to(device)
    policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    policy.eval()
    return (policy, checkpoint)
