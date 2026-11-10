"""Opt-in fitted leg load for frozen Fudan GPU policy evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class LengthFitPolicyLoad:
    def __init__(self, wrapper, path):
        self.env = wrapper.env
        self.action = wrapper.action
        source = Path(path).read_bytes()
        bundle = json.loads(source)
        assert bundle['format'] == 'right-leg-length-fit-v1'
        self.p = p = bundle['candidate']
        baseline = bundle['baseline']
        self.sides = bundle.get('sides', ['right'])
        assert self.sides in (['right'], ['left', 'right'])
        # This adapter transfers the current constant rotary-friction/Dahl candidate only.
        assert p['seal_friction_route'] == 'dahl' and not p.get('seal_bristle_aging', False)
        assert p['static_nm'] == p['kinetic_nm'] and not p.get('elastic_stiffness_nm_rad', 0.)
        assert np.all(np.asarray(baseline['stiffness_nm_rad'])[2:, :2] == 0.)
        self.record = dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(source).hexdigest(),
            applied_sides=self.sides, source=bundle['source'],
            initial_seal_force_n=0., physics_dt_s=self.env.physics_dt,
            scope='Right measured candidate; left, when enabled, is a mirrored extrapolation; no controller/training-default change')
        self.device = wrapper.device
        model = self.env.sim.mj_model
        joints = [[model.joint('robot/' + name) for name in
            (('lf0', 'lf00') if side == 'left' else ('rf0', 'rf00'))] for side in self.sides]
        self.qi = torch.tensor([[int(j.qposadr[0]) for j in row] for row in joints], device=self.device)
        self.vi = torch.tensor([[int(j.dofadr[0]) for j in row] for row in joints], device=self.device)
        sliders = [model.joint('robot/' + side + '_spring2_joint') for side in self.sides]
        self.sq = torch.tensor([int(j.qposadr[0]) for j in sliders], device=self.device)
        self.sv = torch.tensor([int(j.dofadr[0]) for j in sliders], device=self.device)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        # Left motor axes oppose right axes; mirror both angle and generalized torque.
        self.sign = tensor([-1. if side == 'left' else 1. for side in self.sides])
        self.qref = tensor(baseline['reference_q_model_rad'][2:])
        self.bias = tensor(baseline['bias_nm'][2:])
        self.stiffness = tensor(np.asarray(baseline['stiffness_nm_rad'])[2:, 2:])
        self.delta_bias = tensor(p['bias_delta_nm'])
        self.u, self.w = tensor([-1., 1.]) / np.sqrt(2), tensor([1., 1.]) / np.sqrt(2)
        self.env.sim.expand_model_fields(('dof_frictionloss', 'dof_damping'))
        self.env.sim.model.dof_frictionloss[:, self.vi] = tensor(p['static_nm'])
        self.env.sim.model.dof_damping[:, self.vi] = tensor(p['damping_nms_rad'])
        self.env.sim.model.dof_damping[:, self.sv] += p['seal_damping_ns_m']
        self.previous = self.env.sim.data.qpos[:, self.sq].clone()
        self.force = torch.zeros((wrapper.num_envs, len(self.sides)), device=self.device)
        self.initial_slider = self.previous.cpu().numpy().copy()
        self.audit = []
        self.post_step_error = 0.
        self.original_apply = self.action.apply_actions
        self.action.apply_actions = self.apply

    def apply(self):
        self.original_apply()
        data = self.env.sim.data
        q = data.qpos[:, self.qi] * self.sign[None, :, None]
        v = data.qvel[:, self.vi] * self.sign[None, :, None]
        delta = q - self.qref
        x, y = delta @ self.u, delta @ self.w
        load = -(self.bias + delta @ self.stiffness.T + self.delta_bias)
        load -= (self.p['length_stiffness_delta_nm_rad'] * x
                 + self.p['length_quadratic_delta_nm_rad2'] * x.square())[..., None] * self.u
        load -= self.p['cross_stiffness_delta_nm_rad'] * (x[..., None] * self.w + y[..., None] * self.u)
        load *= self.sign[None, :, None]
        position = data.qpos[:, self.sq]
        velocity = data.qvel[:, self.sv]
        displacement = position - self.previous
        cap = self.p['seal_kinetic_n'] + (self.p['seal_static_n'] - self.p['seal_kinetic_n']) * torch.exp(
            -(velocity / self.p['seal_velocity_scale_m_s']).square())
        target = displacement.sign() * cap
        old = self.force.clone()
        self.force = target + (old - target) * torch.exp(
            -self.p['seal_bristle_stiffness_n_m'] * displacement.abs() / cap)
        # State evolves from simulated displacement; no recorded trajectory is injected.
        self.previous.copy_(position)
        data.qfrc_applied[:, self.vi] = load
        data.qfrc_applied[:, self.sv] = self.p.get('seal_force_bias_n', 0.) - self.force
        self.audit.append(torch.cat((q, v, position[..., None], velocity[..., None],
            displacement[..., None], old[..., None], cap[..., None], self.force[..., None], load), -1).clone())

    def after_policy_step(self):
        actual = self.env.sim.data.qfrc_applied[:, self.sv]
        expected = self.p.get('seal_force_bias_n', 0.) - self.force
        self.post_step_error = max(self.post_step_error, float((actual - expected).abs().max()))

    def save(self, directory):
        a = torch.stack(self.audit).cpu().numpy()
        np.savez_compressed(directory / 'response-audit.npz', audit=a, initial_slider=self.initial_slider,
            motor_q=self.qi.cpu().numpy(), motor_v=self.vi.cpu().numpy(),
            slider_q=self.sq.cpu().numpy(), slider_v=self.sv.cpu().numpy(), sides=np.array(self.sides),
            dof_frictionloss=self.env.sim.model.dof_frictionloss.cpu().numpy(),
            dof_damping=self.env.sim.model.dof_damping.cpu().numpy())
        p = self.p
        q, velocity = a[..., :2].astype(float), a[..., 5].astype(float)
        dq, old, cap, force = [a[..., i].astype(float) for i in (6, 7, 8, 9)]
        expected_cap = p['seal_kinetic_n'] + (p['seal_static_n'] - p['seal_kinetic_n']) * np.exp(
            -(velocity / p['seal_velocity_scale_m_s']) ** 2)
        target = np.sign(dq) * cap
        factor = np.exp(-p['seal_bristle_stiffness_n_m'] * abs(dq) / cap)
        predicted = target + (old - target) * factor
        delta = q - self.qref.cpu().numpy()
        u, w = self.u.cpu().numpy().astype(float), self.w.cpu().numpy().astype(float)
        x, y = delta @ u, delta @ w
        load = -(self.bias.cpu().numpy() + delta @ self.stiffness.cpu().numpy().T + p['bias_delta_nm'])
        load -= (p['length_stiffness_delta_nm_rad'] * x + p['length_quadratic_delta_nm_rad2'] * x*x)[..., None] * u
        load -= p['cross_stiffness_delta_nm_rad'] * (x[..., None]*w + y[..., None]*u)
        load *= self.sign.cpu().numpy()[None, None, :, None]
        work = cap*abs(dq) + np.sign(dq)*(old-target)*cap/p['seal_bristle_stiffness_n_m']*(1-factor)
        loss = work - (force*force-old*old)/(2*p['seal_bristle_stiffness_n_m'])
        checks = dict(force_update_max_n=float(abs(predicted-force).max()),
            capacity_max_n=float(abs(expected_cap-cap).max()),
            state_continuity_max_n=float(abs(old[1:]-force[:-1]).max()),
            initial_force_max_n=float(abs(old[0]).max()),
            displacement_max_m=float(abs(np.diff(a[..., 4], axis=0)-dq[1:]).max()),
            force_bound_excess_n=float(np.maximum(abs(force)-p['seal_static_n'], 0.).max()),
            load_max_nm=float(abs(load-a[..., 10:12]).max()),
            post_step_applied_error_n=self.post_step_error, min_dissipation_j=float(loss.min()),
            physics_updates=len(a), worlds=a.shape[1], applied_sides=self.sides,
            mirror_motor_sign=self.sign.cpu().tolist())
        checks['passed'] = bool(np.isfinite(a).all() and checks['force_update_max_n'] < .001
            and checks['capacity_max_n'] < .001 and checks['state_continuity_max_n'] == 0.
            and checks['initial_force_max_n'] == 0. and checks['displacement_max_m'] < 1e-7
            and checks['force_bound_excess_n'] < .001 and checks['load_max_nm'] < 1e-4
            and self.post_step_error < .001 and checks['min_dissipation_j'] >= -1e-6)
        (directory / 'response-verification.json').write_text(json.dumps(checks, indent=2) + '\n')
        assert checks['passed'], checks
