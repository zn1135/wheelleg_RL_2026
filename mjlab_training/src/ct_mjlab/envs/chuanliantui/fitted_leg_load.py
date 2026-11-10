"""User-adopted mirrored leg load; shared by mjlab and native MuJoCo playback."""
import hashlib
import json
from pathlib import Path

import numpy as np

PARAMETERS = Path(__file__).resolve().parents[5] / 'resources/robots/chuanliantui_new_1/mjlab/fitted_leg_load.json'
PROFILE = 'mirrored-local-load-dahl-20261109-v1'


def parameter_record(enabled=True):
    return dict(profile=PROFILE if enabled else 'legacy-linear-spring-v1',
        sha256=hashlib.sha256(PARAMETERS.read_bytes()).hexdigest() if enabled else None,
        mirrored_left=bool(enabled), initial_memory_force_n=0.)


def parameters():
    result = json.loads(PARAMETERS.read_text())
    assert result['sides'] == ['left', 'right']
    return result


def coefficients(bundle, array):
    b, p = bundle['baseline'], bundle['candidate']
    return dict(qref=array(b['reference_q_model_rad'][2:]), bias=array(b['bias_nm'][2:]),
        matrix=array(np.asarray(b['stiffness_nm_rad'])[2:, 2:]), delta_bias=array(p['bias_delta_nm']),
        u=array([-1./np.sqrt(2), 1./np.sqrt(2)]), w=array([1./np.sqrt(2), 1./np.sqrt(2)]),
        sign=array([-1., 1.]))


def conservative_load(q, c, p):
    equivalent = q * c['sign'][..., None]
    delta = equivalent - c['qref']
    x, y = delta @ c['u'], delta @ c['w']
    load = -(c['bias'] + delta @ c['matrix'].T + c['delta_bias'])
    load -= (p['length_stiffness_delta_nm_rad']*x + p['length_quadratic_delta_nm_rad2']*x*x)[..., None]*c['u']
    load -= p['cross_stiffness_delta_nm_rad']*(x[..., None]*c['w'] + y[..., None]*c['u'])
    return load*c['sign'][..., None]


def memory_step(position, velocity, previous, force, p):
    exp = np.exp if isinstance(position, np.ndarray) else lambda x: x.exp()
    delta = position-previous
    sign = np.sign(delta) if isinstance(delta, np.ndarray) else delta.sign()
    cap = p['seal_kinetic_n'] + (p['seal_static_n']-p['seal_kinetic_n'])*exp(
        -(velocity/p['seal_velocity_scale_m_s'])**2)
    target = sign*cap
    return target + (force-target)*exp(-p['seal_bristle_stiffness_n_m']*abs(delta)/cap)


class TorchLegLoad:
    def __init__(self, action):
        import torch
        self.env = action._env
        self.bundle = parameters()
        self.record = parameter_record()
        self.p = self.bundle['candidate']
        self.c = coefficients(self.bundle, lambda x: torch.as_tensor(x, device=action.device, dtype=torch.float32))
        self.qi = action._motor_q_adr[[0, 1, 3, 4]].reshape(2, 2)
        self.vi = action._motor_v_adr[[0, 1, 3, 4]].reshape(2, 2)
        self.sq, self.sv = action._spring_q_adr, action._spring_v_adr
        missing = tuple(f for f in ('dof_frictionloss', 'dof_damping') if f not in self.env.sim.expanded_fields)
        self.env.sim.expand_model_fields(missing)
        self.env.sim.model.dof_frictionloss[:, self.vi] = self.c['qref'].new_tensor(self.p['static_nm'])
        self.env.sim.model.dof_damping[:, self.vi] = self.c['qref'].new_tensor(self.p['damping_nms_rad'])
        self.env.sim.model.dof_damping[:, self.sv] = (
            self.env.sim.get_default_field('dof_damping')[self.sv] + self.p['seal_damping_ns_m'])
        self.previous = self.env.sim.data.qpos[:, self.sq].clone()
        self.force = torch.zeros_like(self.previous)
        self.pending = torch.ones(action.num_envs, device=action.device, dtype=torch.bool)

    def apply(self):
        import torch
        data = self.env.sim.data
        position = data.qpos[:, self.sq]
        # Reset events may write the pose after action.reset; latch it on the first physics tick.
        previous = torch.where(self.pending[:, None], position, self.previous)
        self.force.copy_(memory_step(position, data.qvel[:, self.sv], previous, self.force, self.p))
        self.previous.copy_(position)
        self.pending.zero_()
        data.qfrc_applied[:, self.vi] = conservative_load(data.qpos[:, self.qi], self.c, self.p)
        data.qfrc_applied[:, self.sv] = self.p.get('seal_force_bias_n', 0.)-self.force

    def reset(self, ids):
        self.force[ids] = 0.
        self.pending[ids] = True
        self.env.sim.data.qfrc_applied[ids[:, None], self.vi.flatten()[None]] = 0.
        self.env.sim.data.qfrc_applied[ids[:, None], self.sv[None]] = 0.


class NativeLegLoad:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.bundle = parameters()
        self.record = parameter_record()
        self.p = self.bundle['candidate']
        self.c = coefficients(self.bundle, np.asarray)
        joints = [[model.joint(side+suffix) for suffix in ('0', '00')] for side in ('lf', 'rf')]
        self.qi = np.array([[j.qposadr[0] for j in row] for row in joints])
        self.vi = np.array([[j.dofadr[0] for j in row] for row in joints])
        sliders = [model.joint(side+'_spring2_joint') for side in ('left', 'right')]
        self.sq = np.array([j.qposadr[0] for j in sliders])
        self.sv = np.array([j.dofadr[0] for j in sliders])
        model.dof_frictionloss[self.vi] = self.p['static_nm']
        model.dof_damping[self.vi] = self.p['damping_nms_rad']
        model.dof_damping[self.sv] += self.p['seal_damping_ns_m']
        self.reset()

    def reset(self):
        self.force = np.zeros(2)
        self.pending = True
        self.data.qfrc_applied[self.vi] = 0.
        self.data.qfrc_applied[self.sv] = 0.

    def apply(self):
        position = self.data.qpos[self.sq]
        previous = position if self.pending else self.previous
        self.force = memory_step(position, self.data.qvel[self.sv], previous, self.force, self.p)
        self.previous = position.copy()
        self.pending = False
        self.data.qfrc_applied[self.vi] = conservative_load(self.data.qpos[self.qi], self.c, self.p)
        self.data.qfrc_applied[self.sv] = self.p.get('seal_force_bias_n', 0.)-self.force
