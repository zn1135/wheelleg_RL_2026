# SPDX-License-Identifier: MIT
# Motion-state selection follows RadiantForever/supermimic_wheelbipe/spec.py.
"""V12 motion-only dynamics; the old 64/254 layouts are recording containers.

The learned state is qpos without either wheel angle (23) plus every qvel (24).
Actual motor torque, contact forces and accumulated wheel angles never enter
the MLP, its regression targets, or its output normalization.  Expanded 64-D
paths exist only to reuse kinematics and compare common physical quantities.
"""

import torch
from torch import nn
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

from .networks import (RunningNorm, contact_parameters, mlp,
                       normalize_contact_inputs)
from .state import (MEMORY, PARAM_DIM, PHYS_DIM, PHYSICAL, SUBSTEPS,
                    rotation_quat, rotation_vector, unit_quat)


MOTION_DIM, MOTION_DELTA_DIM = 47, 46
WHEEL_Q = (9, 14)


def physical47_from_full64(physical):
    """Remove wheel q at 9/14 and all auxiliary channels at 49:64."""
    return torch.cat((physical[..., :9], physical[..., 10:14],
                      physical[..., 15:49]), -1)


def expand_motion47_to_full64(motion):
    """Fill wheel angles and unused auxiliary slots with zero, as in full_joints."""
    zero = torch.zeros_like(motion[..., :1])
    return torch.cat((motion[..., :9], zero, motion[..., 9:13], zero,
                      motion[..., 13:47],
                      torch.zeros_like(motion[..., :15])), -1)


def delta46(before, after):
    """Body-local pose/velocity increments and 16 non-wheel joint increments."""
    q = unit_quat(before[..., 3:7])
    inv = torch.cat((q[..., :1], -q[..., 1:]), -1)
    return torch.cat((quat_apply_inverse(q, after[..., :3]-before[..., :3]),
        rotation_vector(quat_mul(inv, unit_quat(after[..., 3:7]))),
        after[..., 7:23]-before[..., 7:23],
        quat_apply_inverse(q, after[..., 23:26]-before[..., 23:26]),
        after[..., 26:47]-before[..., 26:47]), -1)


def integrate47(before, delta):
    """Integrate one 2ms increment in the same coordinates as delta46."""
    q = unit_quat(before[..., 3:7])
    return torch.cat((before[..., :3]+quat_apply(q, delta[..., :3]),
        unit_quat(quat_mul(q, rotation_quat(delta[..., 3:6]))),
        before[..., 7:23]+delta[..., 6:22],
        before[..., 23:26]+quat_apply(q, delta[..., 22:25]),
        before[..., 26:47]+delta[..., 25:46]), -1)


class WorldModel(nn.Module):
    """Predict only five motion increments; no learned force/torque channels."""
    def __init__(self):
        super().__init__()
        # Lazy import keeps the motion/reward adapter contract free of cycles.
        from .motion_adapter import MotionStateAdapter
        self.adapter = MotionStateAdapter()
        width = MOTION_DIM+PARAM_DIM+18+1
        self.net = mlp(width, (512, 512), SUBSTEPS*MOTION_DELTA_DIM, normalized=True)
        self.in_norm = RunningNorm(width)
        self.delta_norm = RunningNorm(MOTION_DELTA_DIM)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def inputs(state, parameters, action, previous, previous_previous):
        physical = physical47_from_full64(state[..., PHYSICAL])
        # Translation invariance matches the existing physical model.
        physical = torch.cat((torch.zeros_like(physical[..., :2]), physical[..., 2:]), -1)
        return torch.cat((physical, contact_parameters(parameters), action, previous,
                          previous_previous, state[..., MEMORY][..., :1]*.01), -1)

    @torch.no_grad()
    def observe(self, w):
        inputs = self.inputs(w['state'], w['parameters'], w['action'],
                             w['previous'], w['previous_previous'])
        full_path = w['path'].reshape(*w['path'].shape[:-1], SUBSTEPS, PHYS_DIM)
        path = physical47_from_full64(full_path)
        before = torch.cat((physical47_from_full64(w['state'][..., None, PHYSICAL]),
                            path[..., :-1, :]), -2)
        # Deleted recording channels cannot invalidate a motion training sample.
        valid = (w['mask'].bool() & inputs.isfinite().all(-1)
                 & path.isfinite().all(dim=(-1, -2)))
        self.in_norm.update(inputs[valid])
        self.delta_norm.update(delta46(before, path)[valid])

    def step(self, state, parameters, action, next_command, context, previous,
             previous_previous, noise, *, return_path=False):
        inputs = self.inputs(state, parameters, action, previous, previous_previous)
        y = self.net(normalize_contact_inputs(inputs, self.in_norm, MOTION_DIM))
        increments = (y.reshape(*y.shape[:-1], SUBSTEPS, MOTION_DELTA_DIM)
                      *self.delta_norm.std()+self.delta_norm.mean)
        motion = physical47_from_full64(state[..., PHYSICAL])
        path = []
        for t in range(SUBSTEPS):
            motion = integrate47(motion, increments[..., t, :])
            path.append(expand_motion47_to_full64(motion))
        path = torch.stack(path, -2)
        result = self.adapter.advance(state, parameters, path, next_command, action,
                                      previous, previous_previous, context, noise)
        return (*result, path) if return_path else result


MotionWorldModel = WorldModel

__all__ = ['WorldModel', 'MotionWorldModel', 'MOTION_DIM', 'MOTION_DELTA_DIM',
           'WHEEL_Q', 'physical47_from_full64', 'expand_motion47_to_full64',
           'delta46', 'integrate47']
