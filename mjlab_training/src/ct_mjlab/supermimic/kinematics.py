# SPDX-License-Identifier: MIT
"""Differentiable tree kinematics of the unchanged chuanliantui MJCF.

Like the upstream SuperMimic kinematics, this walks the body/joint tree.  All
passive coordinates are inputs; it does not solve or project loop constraints.
"""

import torch
from torch import nn

from ct_mjlab.envs.chuanliantui.model import load_model


def quat_mul(a, b):
    """Hamilton product, with MuJoCo's scalar-first quaternion convention."""
    a, b = torch.broadcast_tensors(a, b)
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def quat_rotate(q, v):
    """Rotate a vector by a unit wxyz quaternion, allowing broadcast batches."""
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(qv, v, dim=-1)
    return v + 2*(q[..., :1]*uv + torch.cross(qv, uv, dim=-1))


def _unit(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class RobotKinematics(nn.Module):
    """Map full qpos/qvel to all MuJoCo body frames, including world body 0.

    ``forward`` accepts matching arbitrary leading batch dimensions and returns
    ``(position, quaternion, linear_velocity, angular_velocity)`` in world axes.
    Positions/linear velocities refer to body origins, not inertial COMs.
    A COM has offset ``r = quat_rotate(quaternion, body_ipos)`` and therefore
    position ``position+r`` and velocity ``linear_velocity+cross(angular,r)``.
    Free-joint linear qvel is world-frame; its angular qvel is body-frame.
    """

    def __init__(self, model=None):
        super().__init__()
        model = load_model() if model is None else model
        self.nq, self.nv, self.nbody = model.nq, model.nv, model.nbody
        self.body_names = tuple(model.body(i).name for i in range(model.nbody))
        self.joint_names = tuple(model.joint(i).name for i in range(model.njnt))
        self.parents = tuple(map(int, model.body_parentid))
        self.body_joints = tuple(tuple(range(int(model.body_jntadr[i]),
            int(model.body_jntadr[i]+model.body_jntnum[i])))
            if model.body_jntnum[i] else () for i in range(model.nbody))
        self.joint_types = tuple(map(int, model.jnt_type))
        self.qpos_adr = tuple(map(int, model.jnt_qposadr))
        self.qvel_adr = tuple(map(int, model.jnt_dofadr))
        # MuJoCo enums: free=0, ball=1, slide=2, hinge=3. The robot has no balls.
        if any(kind not in (0, 2, 3) for kind in self.joint_types):
            raise ValueError('RobotKinematics supports free, slide and hinge joints')
        for name in ('body_pos', 'body_quat', 'body_ipos', 'body_iquat',
                     'jnt_pos', 'jnt_axis', 'qpos0'):
            self.register_buffer(name, torch.as_tensor(getattr(model, name).copy()))

    def forward(self, qpos, qvel):
        if qpos.shape[:-1] != qvel.shape[:-1] or qpos.shape[-1] != self.nq or qvel.shape[-1] != self.nv:
            raise ValueError(f'Expected matching batches of qpos{self.nq}/qvel{self.nv}')
        if qpos.device != qvel.device or qpos.dtype != qvel.dtype:
            raise ValueError('qpos/qvel must have the same device and dtype')
        body_pos, body_quat = self.body_pos.to(qpos), self.body_quat.to(qpos)
        jnt_pos, jnt_axis, reference = self.jnt_pos.to(qpos), self.jnt_axis.to(qpos), self.qpos0.to(qpos)
        zero = torch.zeros_like(qpos[..., :3])
        identity = torch.cat((torch.ones_like(qpos[..., :1]), zero), -1)
        positions, quaternions, velocities, angular = [zero], [identity], [zero], [zero]
        for i in range(1, self.nbody):
            parent = self.parents[i]
            offset = quat_rotate(quaternions[parent], body_pos[i])
            p = positions[parent] + offset
            q = quat_mul(quaternions[parent], body_quat[i])
            w = angular[parent]
            v = velocities[parent] + torch.cross(w, offset, dim=-1)
            for j in self.body_joints[i]:
                qa, va, kind = self.qpos_adr[j], self.qvel_adr[j], self.joint_types[j]
                if kind == 0:
                    p, q = qpos[..., qa:qa+3], _unit(qpos[..., qa+3:qa+7])
                    v = qvel[..., va:va+3]
                    w = quat_rotate(q, qvel[..., va+3:va+6])
                    continue
                axis = quat_rotate(q, jnt_axis[j])
                distance = qpos[..., qa:qa+1] - reference[qa]
                speed = qvel[..., va:va+1]
                if kind == 2:
                    shift = axis*distance
                    p, v = p+shift, v+torch.cross(w, shift, dim=-1)+axis*speed
                else:
                    # Rotate about the joint anchor, which need not be the body origin.
                    anchor_offset = quat_rotate(q, jnt_pos[j])
                    anchor, anchor_v = p+anchor_offset, v+torch.cross(w, anchor_offset, dim=-1)
                    half = distance*.5
                    rotation = torch.cat((half.cos(), jnt_axis[j]*half.sin()), -1)
                    q = quat_mul(q, rotation)
                    p = anchor-quat_rotate(q, jnt_pos[j])
                    w = w+axis*speed
                    v = anchor_v+torch.cross(w, p-anchor, dim=-1)
            positions.append(p)
            quaternions.append(q)
            velocities.append(v)
            angular.append(w)
        return tuple(torch.stack(values, -2) for values in
                     (positions, quaternions, velocities, angular))
