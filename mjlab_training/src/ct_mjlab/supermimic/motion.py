# SPDX-License-Identifier: MIT
# Motion-state selection follows RadiantForever/supermimic_wheelbipe/spec.py.
"""Control-step motion dynamics; the old64/254 layouts are recording containers.

The learned state is qpos without either wheel angle (23) plus every qvel (24).
Actual motor torque, contact forces and accumulated wheel angles never enter
the MLP, its regression targets, or its output normalization.  Expanded 64-D
paths exist only to reuse kinematics and compare common physical quantities.
"""

import torch
from torch import nn
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_mul

from .networks import (RunningNorm, contact_parameters, mlp,
                       normalize_contact_inputs)
from .state import (MEMORY, PARAM_DIM, PHYS_DIM, PHYSICAL, SUBSTEPS, WRENCH,
                    rotation_quat, rotation_vector, unit_quat)


MOTION_DIM, MOTION_DELTA_DIM = 47, 46
WORLD_INPUT_DIM = MOTION_DIM+PARAM_DIM+6+1
WHEEL_Q = (9, 14)


def invariant_features47(motion):
    """Reference layout: height, yaw-free rotation6D, body v/w, joint q/qd."""
    q = unit_quat(motion[..., 3:7])
    rotation = matrix_from_quat(q)
    # Same heading definition and regularization as upstream yaw_mat().
    x, y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = (x.square()+y.square()+1e-9).sqrt()
    c, s = x/norm, y/norm
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    heading = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1)
    heading = heading.reshape(*rotation.shape[:-2], 3, 3)
    tilt = heading.transpose(-1, -2) @ rotation
    rotation6d = tilt[..., :, :2].reshape(*tilt.shape[:-2], 6)
    # MuJoCo free-joint linear velocity is world-frame; angular velocity is already body-frame.
    linear = quat_apply_inverse(q, motion[..., 23:26])
    return torch.cat((motion[..., 2:3], rotation6d, linear, motion[..., 26:29],
                      motion[..., 7:23], motion[..., 29:47]), -1)


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
    """Apply a motion increment in the same coordinates as delta46."""
    q = unit_quat(before[..., 3:7])
    return torch.cat((before[..., :3]+quat_apply(q, delta[..., :3]),
        unit_quat(quat_mul(q, rotation_quat(delta[..., 3:6]))),
        before[..., 7:23]+delta[..., 6:22],
        before[..., 23:26]+quat_apply(q, delta[..., 22:25]),
        before[..., 26:47]+delta[..., 25:46]), -1)


class WorldModel(nn.Module):
    """Predict one10ms motion increment; no learned intermediate states or forces."""
    def __init__(self):
        super().__init__()
        # Lazy import keeps the motion/reward adapter contract free of cycles.
        from .motion_adapter import MotionStateAdapter
        self.adapter = MotionStateAdapter()
        width = WORLD_INPUT_DIM
        self.net = mlp(width, (512, 512), MOTION_DELTA_DIM, normalized=True)
        self.in_norm = RunningNorm(width)
        self.delta_norm = RunningNorm(MOTION_DELTA_DIM)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def inputs(state, parameters, action):
        physical = physical47_from_full64(state[..., PHYSICAL])
        features = invariant_features47(physical)
        q = unit_quat(physical[..., 3:7])
        # xfrc_applied stores world force/torque; its input representation must rotate too.
        force = quat_apply_inverse(q, parameters[..., WRENCH.start:WRENCH.start+3])
        torque = quat_apply_inverse(q, parameters[..., WRENCH.start+3:WRENCH.stop])
        local_parameters = torch.cat((parameters[..., :WRENCH.start], force, torque), -1)
        # Action history is deliberately excluded from learned dynamics.
        return torch.cat((features, contact_parameters(local_parameters), action,
                          state[..., MEMORY][..., :1]*.01), -1)

    @torch.no_grad()
    def observe(self, w):
        inputs = self.inputs(w['state'], w['parameters'], w['action'])
        full_path = w['path'].reshape(*w['path'].shape[:-1], SUBSTEPS, PHYS_DIM)
        endpoint = physical47_from_full64(full_path[..., -1, :])
        before = physical47_from_full64(w['state'][..., PHYSICAL])
        # Deleted recording channels cannot invalidate a motion training sample.
        valid = (w['mask'].bool() & inputs.isfinite().all(-1)
                 & endpoint.isfinite().all(-1))
        self.in_norm.update(inputs[valid])
        self.delta_norm.update(delta46(before, endpoint)[valid])

    def step(self, state, parameters, action, next_command, context, previous,
             previous_previous, noise, *, return_path=False, imagination=False):
        inputs = self.inputs(state, parameters, action)
        y = self.net(normalize_contact_inputs(inputs, self.in_norm, MOTION_DIM))
        increment = y*self.delta_norm.std()+self.delta_norm.mean
        motion = physical47_from_full64(state[..., PHYSICAL])
        endpoint = expand_motion47_to_full64(integrate47(motion, increment))
        if imagination:
            endpoint = self.adapter.sanitize_endpoint(endpoint)
        # A singleton endpoint path preserves the return API without inventing substeps.
        path = endpoint.unsqueeze(-2)
        result = self.adapter.advance(state, parameters, path, next_command, action,
                                      previous, previous_previous, context, noise)
        return (*result, path) if return_path else result


MotionWorldModel = WorldModel

__all__ = ['WorldModel', 'MotionWorldModel', 'MOTION_DIM', 'MOTION_DELTA_DIM',
           'WORLD_INPUT_DIM', 'invariant_features47',
           'WHEEL_Q', 'physical47_from_full64', 'expand_motion47_to_full64',
           'delta46', 'integrate47']
