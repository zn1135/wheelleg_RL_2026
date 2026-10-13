"""Check the active MJCF's closed-chain, wheel-only support and loaded height reach.

Use the native mjlab Python environment. No learning, asset edits or external lift.
Kinematic reach and force equilibrium are reported separately from PD simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from ct_mjlab.asset import cpu_model, POSES
from ct_mjlab.config import ASSET
from ct_mjlab.reference_training.gas_spring import gas_spring_force

LEFT = ('lf0', 'lf1', 'left_spring1_joint', 'left_spring2_joint', 'lf00', 'lf01', 'lf02', 'lf03')
RIGHT = ('rf0', 'rf1', 'right_spring1_joint', 'right_spring2_joint', 'rf00', 'rf01', 'rf02', 'rf03')
SIGN = (-1., -1., 1., 1., -1., -1., -1., -1.)
PAIRS = tuple((f'{side}_{a}', f'{side}_{b}') for side in ('left', 'right') for a, b in (
    ('rear_pin1', 'front_pin1'), ('rear_pin2', 'front_pin2'), ('spring_tip', 'gas_spring_lower')))
MOTORS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')


class Audit:
    def __init__(self):
        self.model = m = cpu_model()
        self.data = d = mujoco.MjData(m)
        self.left = np.array([m.joint(n).qposadr[0] for n in LEFT])
        self.right = np.array([m.joint(n).qposadr[0] for n in RIGHT])
        self.motor_q = np.array([m.joint(n).qposadr[0] for n in MOTORS])
        self.motor_v = np.array([m.joint(n).dofadr[0] for n in MOTORS])
        self.motor_u = np.array([m.actuator(n + '_motor').id for n in MOTORS])
        self.spring_q = np.array([m.joint(n).qposadr[0] for n in ('left_spring2_joint', 'right_spring2_joint')])
        self.spring_v = np.array([m.joint(n).dofadr[0] for n in ('left_spring2_joint', 'right_spring2_joint')])
        self.spring_u = np.array([m.actuator(n + '_gas_spring_motor').id for n in ('left', 'right')])
        self.pairs = [(m.site(a).id, m.site(b).id) for a, b in PAIRS]
        self.wheels = [m.body(n).id for n in ('lfwheel', 'rfwheel')]
        self.floor = m.geom('floor').id
        self.base = m.body('base_link').id
        self.collision = [i for i in range(m.ngeom) if i != self.floor and (m.geom_contype[i] or m.geom_conaffinity[i])]
        self.wheel_geoms = [next(i for i in self.collision if m.geom_bodyid[i] == b) for b in self.wheels]
        self.vertices = {}
        for i in self.collision:
            mesh = m.geom_dataid[i]
            assert m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            adr, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.vertices[i] = m.mesh_vert[adr:adr + count].astype(float).copy()
        for name, value in POSES['nominal'].items():
            d.qpos[m.joint(name).qposadr[0]] = value
        self.nominal = d.qpos.copy()
        self.lower, self.upper = [], []
        for name in LEFT:
            joint = m.joint(name)
            if m.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_SLIDE:
                low, high = -.2, .2
            elif m.jnt_limited[joint.id] and name != 'lf0':
                low, high = m.jnt_range[joint.id]
            else:
                low, high = m.qpos0[joint.qposadr[0]] - 3.14, m.qpos0[joint.qposadr[0]] + 3.14
            self.lower.append(low + 1e-6)
            self.upper.append(high - 1e-6)
        self.lower, self.upper = np.r_[self.lower, -.03], np.r_[self.upper, .03]

    def set_pose(self, x, height):
        d, m = self.data, self.model
        d.qpos[:] = self.nominal
        d.qpos[:3] = (0., 0., height)
        d.qpos[3:7] = (np.cos(x[-1] / 2), np.sin(x[-1] / 2), 0., 0.)
        d.qpos[self.left] = x[:-1]
        d.qpos[self.right] = x[:-1] * SIGN
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def bottom(self, geom):
        d = self.data
        vertices = self.vertices[geom] @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
        return vertices[np.argmin(vertices[:, 2])]

    def residual(self, x, height):
        self.set_pose(x, height)
        d = self.data
        loops = np.concatenate([d.site_xpos[a] - d.site_xpos[b] for a, b in self.pairs[:3]])
        bottoms = [self.bottom(g) for g in self.wheel_geoms]
        left_fraction = (d.subtree_com[self.base, 1] - bottoms[1][1]) / (bottoms[0][1] - bottoms[1][1])
        support_x = left_fraction * bottoms[0][0] + (1. - left_fraction) * bottoms[1][0]
        # Upright chassis and wheel axle under the actual whole-robot COM.
        return np.r_[loops * 10., bottoms[0][2], bottoms[1][2],
                     support_x - d.subtree_com[self.base, 0]]

    def solve(self, height):
        seed = np.clip(np.r_[self.nominal[self.left], 0.], self.lower + 1e-5, self.upper - 1e-5)
        result = least_squares(lambda x: self.residual(x, height), seed,
            bounds=(self.lower, self.upper), max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        self.set_pose(result.x, height)
        d, m = self.data, self.model
        loop = max(float(np.linalg.norm(d.site_xpos[a] - d.site_xpos[b])) for a, b in self.pairs)
        wheel_bottom = [float(self.bottom(g)[2]) for g in self.wheel_geoms]
        nonwheel = [i for i in self.collision if i not in self.wheel_geoms]
        clearances = {m.body(m.geom_bodyid[g]).name: float(self.bottom(g)[2]) for g in nonwheel}
        angles = {n: float(d.qpos[m.joint(n).qposadr[0]]) for n in LEFT + RIGHT}
        limit_margin = {n: float(min(angles[n] - m.jnt_range[m.joint(n).id, 0],
                                    m.jnt_range[m.joint(n).id, 1] - angles[n])) for n in ('lf1', 'rf1')}
        return dict(target_height_m=height, residual_norm=float(np.linalg.norm(result.fun)),
                    max_pin_error_m=loop, wheel_bottom_m=wheel_bottom,
                    nonwheel_min_clearance_m=min(clearances.values()), clearances=clearances,
                    knee_limit_margin_rad=limit_margin, joint_positions=angles,
                    root_roll_deg=float(np.degrees(result.x[-1])),
                    kinematic_reachable=loop < 1e-5 and max(abs(v) for v in wheel_bottom) < 1e-4
                        and min(clearances.values()) > 0. and min(limit_margin.values()) >= 0.)

    def force_equilibrium(self):
        m, d = self.model, self.data
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        jacobians = []
        for a, b in self.pairs:
            ja, jb = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, ja, None, a)
            mujoco.mj_jacSite(m, d, jb, None, b)
            jacobians.append(ja - jb)
        eq = np.vstack(jacobians)
        # CAD pins carry redundant planar rows at roundoff scale. Use their
        # independent row space so spurious tiny singular values cannot invent
        # large opposing lateral contact forces.
        _, singular, basis = np.linalg.svd(eq, full_matrices=False)
        eq = basis[singular > 1e-8]
        contact = []
        for body, geom in zip(self.wheels, self.wheel_geoms):
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, self.bottom(geom), body)
            contact.append(jac)
        motor = np.zeros((m.nv, 6))
        motor[self.motor_v, np.arange(6)] = 1.
        spring = np.zeros(m.nv)
        force = gas_spring_force(d.qpos[self.spring_q], np.zeros(2))
        spring[self.spring_v] = force
        matrix = np.column_stack((motor, np.vstack(contact).T, eq.T))
        rhs = d.qfrc_bias - d.qfrc_passive - spring
        bounds = (np.r_[[-40., -40., -3.9, -40., -40., -3.9], [-50., -50., 0., -50., -50., 0.], np.full(eq.shape[0], -np.inf)],
                  np.r_[[40., 40., 3.9, 40., 40., 3.9], [50., 50., np.inf, 50., 50., np.inf], np.full(eq.shape[0], np.inf)])
        values = np.linalg.lstsq(matrix, rhs, rcond=1e-8)[0]
        if np.any(values < bounds[0]) or np.any(values > bounds[1]):
            values = lsq_linear(matrix, rhs, bounds=bounds, tol=1e-11, max_iter=500).x
        torque = values[:6]
        contact_force = values[6:12].reshape(2, 3)
        return dict(motor_torque_nm=torque.tolist(), spring_force_n=force.tolist(),
                    wheel_vertical_support_n=contact_force[:, 2].tolist(),
                    wheel_horizontal_support_n=contact_force[:, :2].tolist(),
                    friction_ratio=(np.linalg.norm(contact_force[:, :2], axis=-1) / contact_force[:, 2]).tolist(),
                    robot_weight_n=float(m.body_mass.sum() * 9.81),
                    constraint_jacobian_rank=len(eq),
                    generalized_force_residual=float(np.linalg.norm(matrix @ values - rhs))), torque

    def pd_rollout(self, torque, seconds, wheel_balance=False):
        """Unassisted free base, original60/2 PD and physical gas springs."""
        m, d = self.model, self.data
        q = d.qpos.copy()
        targets = q[self.motor_q].copy()
        legs = np.array([0, 1, 3, 4])
        targets[legs] += torque[legs] / 60.
        assert (abs(targets[legs] - m.qpos0[self.motor_q[legs]]) <= 3.14).all()
        d.qvel[:] = 0.
        records = []
        max_loop = 0.
        for step in range(round(seconds / m.opt.timestep)):
            motor = torque.copy()
            motor[legs] = 60. * (targets[legs] - d.qpos[self.motor_q[legs]]) - 2. * d.qvel[self.motor_v[legs]]
            motor[[2, 5]] = torque[[2, 5]] - .1 * d.qvel[self.motor_v[[2, 5]]]
            if wheel_balance:
                rotation = d.xmat[self.base].reshape(3, 3)
                pitch = np.arctan2(rotation[0, 2], rotation[2, 2])
                correction = 20. * pitch + 4. * d.qvel[4] + .4 * d.qvel[0]
                motor[2] += correction
                motor[5] -= correction  # right wheel axis is-Y
            d.ctrl[self.motor_u] = np.clip(motor, [-40, -40, -3.9, -40, -40, -3.9], [40, 40, 3.9, 40, 40, 3.9])
            d.ctrl[self.spring_u] = gas_spring_force(d.qpos[self.spring_q], d.qvel[self.spring_v])
            mujoco.mj_step(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                raise RuntimeError('Nonfinite loaded PD rollout')
            if step % 10 == 0:
                mujoco.mj_forward(m, d)
                support = np.zeros(2)
                other = 0.
                for index in range(d.ncon):
                    c = d.contact[index]
                    if self.floor not in c.geom:
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, index, force)
                    world = c.frame.reshape(3, 3).T @ force[:3]
                    ground_sign = 1. if c.geom[0] == self.floor else -1.
                    vertical = max(0., ground_sign * world[2])
                    other_geom = c.geom[1] if c.geom[0] == self.floor else c.geom[0]
                    if other_geom in self.wheel_geoms:
                        support[self.wheel_geoms.index(other_geom)] += vertical
                    else:
                        other += vertical
                loop = max(float(np.linalg.norm(d.site_xpos[a] - d.site_xpos[b])) for a, b in self.pairs)
                max_loop = max(max_loop, loop)
                records.append(dict(time_s=d.time, height_m=float(d.qpos[2]),
                    tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[self.base, 8], -1., 1.)))),
                    wheel_support_n=support.tolist(), other_support_n=other,
                    max_motor_torque_nm=float(abs(d.ctrl[self.motor_u[legs]]).max())))
        tail = [r for r in records if r['time_s'] >= seconds - 1.]
        return dict(seconds=seconds, learning=False, external_force=False, finite=True,
                    wheel_balance_feedback=wheel_balance,
                    max_pin_error_m=max_loop, last1s_mean_height_m=float(np.mean([r['height_m'] for r in tail])),
                    last1s_max_tilt_deg=max(r['tilt_deg'] for r in tail),
                    last1s_mean_wheel_support_n=np.mean([r['wheel_support_n'] for r in tail], axis=0).tolist(),
                    last1s_mean_other_support_n=float(np.mean([r['other_support_n'] for r in tail])),
                    max_motor_torque_nm=max(r['max_motor_torque_nm'] for r in records), trajectory=records), q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--wheel-balance', action='store_true',
        help='Diagnostic wheel pitch/velocity feedback only; does not edit the training controller')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    m = audit.model
    report = dict(asset=str(ASSET), asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        nq=m.nq, nv=m.nv, neq=m.neq, nu=m.nu, mass_kg=float(m.body_mass.sum()),
        inertia_positive=bool((m.body_inertia[1:] > 0.).all()), gravity=m.opt.gravity.tolist(), cases=[])
    for height in (.27, .30, .34):
        case = audit.solve(height)
        equilibrium, torque = audit.force_equilibrium()
        case['force_equilibrium'] = equilibrium
        if (case['kinematic_reachable'] and equilibrium['generalized_force_residual'] < 1e-4
                and max(equilibrium['friction_ratio']) <= .5):
            case['pd_rollout'], pose = audit.pd_rollout(torque, args.seconds, args.wheel_balance)
            np.save(args.output / f'pose-{round(height*100):02d}.npy', pose)
        report['cases'].append(case)
        (args.output / 'xml-height-audit.json').write_text(json.dumps(report, indent=2) + '\n')
        print('XML_HEIGHT_CASE ' + json.dumps({k: v for k, v in case.items() if k not in ('joint_positions', 'clearances', 'pd_rollout')}), flush=True)
        if 'pd_rollout' in case:
            print('LOADED_PD ' + json.dumps({k: v for k, v in case['pd_rollout'].items() if k != 'trajectory'}), flush=True)


if __name__ == '__main__':
    main()
