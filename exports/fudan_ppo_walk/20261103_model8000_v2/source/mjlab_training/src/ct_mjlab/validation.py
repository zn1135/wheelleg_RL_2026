"""Shared final-validation backend and report identity; training is unchanged."""
STANDARD = 'wheelleg-unified-gpu-acceptance-v1'


def require_gpu(device):
    import torch
    device=torch.device(device)
    if device.type!='cuda':
        raise ValueError('Final validation requires CUDA GPU physics; CPU validation is retired')
    return dict(validation_standard=STANDARD,validation_engine='gpu',
                validation_backend='mujoco_warp_gpu',validation_device=str(device),
                acceptance_role='task_recording; finalize motion verdict with scripts/mjlab.sh accept',
                final_acceptance='pending_user_video_review')
