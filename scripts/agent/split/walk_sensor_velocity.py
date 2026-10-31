#!/usr/bin/env python3
"""Fixed sensor-only rolling kinematics for the frozen walk velocity diagnosis.

No filtering, fitting, learned fusion, contact gate or simulated translation is
used. The passive knees are reconstructed from the two motor encoders per leg.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sim2sim.chuanliantui_closed_adapter import ClosedChainAdapter, _PlanarLegGeometry


WHEEL_RADIUS_M = .06
FORMULA_VERSION = 'sensor-wheel-rootbody-com-rolling-kinematics-v1'
MOTOR_NAMES = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rotate_y(angle, vector):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = vector
    return np.stack((c*x+s*z, np.full_like(c, y), -s*x+c*z), -1)


def _name(model, kind, name):
    accessor = getattr(model, kind)
    try:
        return accessor(name).name
    except KeyError:
        count = getattr(model, dict(body='nbody', joint='njnt', site='nsite')[kind])
        matches = [accessor(i).name for i in range(count)
                   if accessor(i).name.endswith('/'+name)]
        if len(matches) != 1:
            raise ValueError(f'Expected one {kind} named {name}, found {matches}')
        return matches[0]


class SensorWheelVelocity:
    """Use current Actor sensor slots and fixed XML geometry to estimate vx."""

    def __init__(self, model):
        self.legs = []
        geometry_description = {}
        # The old CAD solver expects q-ref, so it never receives centered Actor q.
        for side, sign, nominal_ids, observation_ids, velocity_ids in (
                (ClosedChainAdapter._SIDES[1], 1., (0, 1), (9, 10), (13, 14)),
                (ClosedChainAdapter._SIDES[0], -1., (3, 4), (11, 12), (16, 17))):
            front, knee = (_name(model, 'body', value) for value in side[4:6])
            rear = tuple(_name(model, 'body', value) for value in side[6])
            pins = tuple(tuple(_name(model, 'site', value) for value in pair)
                         for pair in side[7])
            mapped_side = (*side[:4], front, knee, rear, pins, side[8])
            geometry = _PlanarLegGeometry(model, mapped_side)
            wheel = _name(model, 'body', 'lfwheel' if sign > 0 else 'rfwheel')
            motor_joint_names = tuple(_name(model, 'joint', value)
                                     for value in (side[4], side[6][0]))
            refs = np.array([model.qpos0[model.joint(value).qposadr[0]]
                             for value in motor_joint_names], dtype=np.float64)
            knee_ref = model.qpos0[model.joint(_name(model, 'joint', side[5])).qposadr[0]]
            if knee_ref != 0:
                raise ValueError('The fixed CAD solver assumes zero passive-knee ref')
            h, u, v = (np.array(model.body_pos[model.body(value).id], dtype=np.float64)
                       for value in (front, knee, wheel))
            axis = np.array((0., sign, 0.))
            self.legs.append(dict(geometry=geometry, h=h, u=u, v=v, sign=sign,
                axis=axis, refs=refs, nominal_ids=nominal_ids,
                observation_ids=observation_ids, velocity_ids=velocity_ids,
                wheel_velocity_id=15 if sign > 0 else 18,
                front=front, knee=knee, wheel=wheel, motor_joint_names=motor_joint_names))
            geometry_description[side[0]] = dict(h=h.tolist(), u=u.tolist(), v=v.tolist(),
                axis=axis.tolist(), motor_refs_rad=refs.tolist(), passive_knee_ref_rad=float(knee_ref),
                cad_points={name: getattr(geometry, name).tolist() for name in (
                    'front_origin', 'front1_offset', 'front_pin1', 'front_pin2', 'rear_origin',
                    'rear1_offset', 'rear2_offset', 'rear3_offset', 'rear_pin1', 'rear_pin2')},
                cad_sign=float(geometry.sign), assembly_branches=[geometry.first_branch, geometry.second_branch])
        base = model.body(_name(model, 'body', 'base_link')).id
        self.com_offset = np.array(model.body_ipos[base], dtype=np.float64)
        description = dict(legs=geometry_description, rootbody_com_offset_m=self.com_offset.tolist())
        dependency = Path(inspect.getfile(_PlanarLegGeometry)).resolve()
        self.formula_spec = dict(version=FORMULA_VERSION, radius_m=WHEEL_RADIUS_M,
            formula='mean_left_right(-r_dot_rel - omega cross (r-c) - Omega_wheel cross rho)[x]',
            wheel_absolute_rotation='Omega_wheel=omega+axis*(dq_front+dq_knee+dq_wheel)',
            contact_vector='rho=-radius*projection_of_world_up_perpendicular_to_wheel_axis',
            passive_knee='CAD two-circle intersections and analytic motor Jacobian, q-ref inputs',
            sensor_allowlist=dict(actor_gyro_body_rad_s='slots 0:3 / .25',
                actor_projected_gravity='slots 3:6',
                leg_motor_angle_rad='slots 9:13 + nominal[:,[0,1,3,4]]',
                leg_motor_velocity_rad_s='slots [13,14,16,17] / .05',
                wheel_velocity_rad_s='slots [15,18] / .05',
                nominal_motor_order=list(MOTOR_NAMES),
                model_inputs='fixed body positions, axes, joint refs and base_link body_ipos'),
            forbidden_inputs=['root translation or translational velocity', 'true velocity labels',
                'measured passive-joint state', 'simulation contact state', 'randomized true COM offset'],
            assumptions=['flat ground and ideal circular 0.06m wheels',
                'equal left/right no-slip rolling constraints', 'rigid CAD closure and its fixed assembly branch',
                'sensor streams refer to the same instant'],
            filtering='none', fitted_parameters='none', fusion='none', contact_gating='none',
            dependency_sources=[dict(path=str(Path(__file__).resolve()), sha256=_sha(__file__)),
                dict(path=str(dependency), sha256=_sha(dependency))],
            geometry=description,
            geometry_sha256=hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest())

    def _leg_state(self, leg, motor_q, motor_dq):
        states = [leg['geometry'].knee_and_jacobian(*q) for q in motor_q]
        knees = np.array([value[0] for value in states])
        jacobians = np.array([value[1] for value in states])
        knee_rates = np.einsum('ni,ni->n', jacobians, motor_dq)
        front = motor_q[:, 0]
        upper = _rotate_y(leg['sign']*front, leg['u'])
        lower = _rotate_y(leg['sign']*(front+knees), leg['v'])
        point = leg['h']+upper+lower
        relative_velocity = (np.cross(leg['axis'], upper)*motor_dq[:, :1]
            + np.cross(leg['axis'], lower)*(motor_dq[:, :1]+knee_rates[:, None]))
        return point, relative_velocity, knees, knee_rates

    def predict(self, actor_observation, nominal):
        observation = np.asarray(actor_observation, dtype=np.float64)
        nominal = np.asarray(nominal, dtype=np.float64)
        if observation.ndim != 2 or observation.shape[1] != 25:
            raise ValueError(f'Expected Actor observations (N,25), got {observation.shape}')
        if nominal.shape != (observation.shape[0], 6):
            raise ValueError(f'Expected nominal motor positions (N,6), got {nominal.shape}')
        omega = observation[:, :3]/.25
        normal = -observation[:, 3:6]
        normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        if np.any(normal_norm < 1e-9):
            raise ValueError('IMU projected gravity has no direction')
        normal = normal/normal_norm
        estimates = []
        for leg in self.legs:
            motor_q = (observation[:, leg['observation_ids']]
                + nominal[:, leg['nominal_ids']]-leg['refs'])
            motor_dq = observation[:, leg['velocity_ids']]/.05
            wheel_dq = observation[:, leg['wheel_velocity_id']]/.05
            point, relative_velocity, _, knee_rate = self._leg_state(leg, motor_q, motor_dq)
            projected_normal = normal-(normal@leg['axis'])[:, None]*leg['axis']
            norm = np.linalg.norm(projected_normal, axis=-1, keepdims=True)
            if np.any(norm < 1e-9):
                raise ValueError('Wheel axis is parallel to the assumed ground normal')
            rho = -WHEEL_RADIUS_M*projected_normal/norm
            wheel_omega = omega+leg['axis']*(motor_dq[:, 0]+knee_rate+wheel_dq)[:, None]
            # Zero contact-point velocity gives root velocity; omega x c shifts it to rootbody COM.
            velocity = (-relative_velocity-np.cross(omega, point-self.com_offset)
                - np.cross(wheel_omega, rho))
            estimates.append(velocity[:, 0])
        return np.mean(estimates, axis=0)


def selfcheck(model):
    """Static synthetic inputs only: CAD derivatives and FK, never physics steps."""
    import mujoco
    estimator = SensorWheelVelocity(model)
    eps, maximum_jacobian_error, maximum_fk_error = 1e-6, 0., 0.
    synthetic_points = ((0., 0.), (.001, -.001), (-.001, .001))
    for leg in estimator.legs:
        for q in synthetic_points:
            q = np.asarray(q)
            knee, analytic = leg['geometry'].knee_and_jacobian(*q)
            numerical = []
            for index in range(2):
                delta = np.eye(2)[index]*eps
                high = leg['geometry'].knee_and_jacobian(*(q+delta))[0]
                low = leg['geometry'].knee_and_jacobian(*(q-delta))[0]
                numerical.append(np.arctan2(np.sin(high-low), np.cos(high-low))/(2*eps))
            maximum_jacobian_error = max(maximum_jacobian_error,
                float(np.max(np.abs(np.asarray(numerical)-analytic))))
            data = mujoco.MjData(model)
            data.qpos[:] = model.qpos0
            for name, value in zip(leg['motor_joint_names'], q+leg['refs']):
                data.qpos[model.joint(name).qposadr[0]] = value
            data.qpos[model.joint(leg['knee']).qposadr[0]] = knee
            mujoco.mj_kinematics(model, data)
            base_id = model.body(_name(model, 'body', 'base_link')).id
            base_rotation = data.xmat[base_id].reshape(3, 3)
            mj_point = base_rotation.T@(data.xpos[model.body(leg['wheel']).id]-data.xpos[base_id])
            point = estimator._leg_state(leg, q[None], np.zeros((1, 2)))[0][0]
            maximum_fk_error = max(maximum_fk_error, float(np.max(np.abs(point-mj_point))))
            # Differentiate the reconstructed wheel point with independent motor directions.
            for direction in np.eye(2):
                high = estimator._leg_state(leg, (q+eps*direction)[None], np.zeros((1, 2)))[0][0]
                low = estimator._leg_state(leg, (q-eps*direction)[None], np.zeros((1, 2)))[0][0]
                rate = estimator._leg_state(leg, q[None], direction[None])[1][0]
                maximum_jacobian_error = max(maximum_jacobian_error,
                    float(np.max(np.abs((high-low)/(2*eps)-rate))))
    nominal = np.array([[model.qpos0[model.joint(_name(model, 'joint', name)).qposadr[0]]
                         for name in MOTOR_NAMES]])
    observation = np.zeros((1, 25))
    observation[:, 5] = -1.
    zero = float(estimator.predict(observation, nominal)[0])
    observation[:, 15] = (.5/WHEEL_RADIUS_M)*.05
    observation[:, 18] = -(.5/WHEEL_RADIUS_M)*.05
    forward = float(estimator.predict(observation, nominal)[0])
    observation[:, [15, 18]] *= -1
    backward = float(estimator.predict(observation, nominal)[0])
    assert maximum_jacobian_error < 1e-6, maximum_jacobian_error
    assert maximum_fk_error < 1e-10, maximum_fk_error
    assert abs(zero) < 1e-12 and abs(forward-.5) < 1e-12 and abs(backward+.5) < 1e-12
    return dict(check='static mathematics and constructed-pose kinematics only', physics_steps=0,
        fitted_parameters=0, scoring_trajectories_read=0, synthetic_points_per_leg=len(synthetic_points),
        maximum_jacobian_absolute_error=maximum_jacobian_error, maximum_fk_error_m=maximum_fk_error,
        synthetic_vx_m_s=dict(zero=zero, forward=forward, backward=backward),
        source_sha256=_sha(__file__), formula_spec=estimator.formula_spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--model', type=Path,
        default=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
    args = parser.parse_args()
    if not args.selfcheck:
        parser.error('Use --selfcheck for the static synthetic-input check')
    import mujoco
    model = (mujoco.MjModel.from_binary_path(str(args.model)) if args.model.suffix == '.mjb'
             else mujoco.MjModel.from_xml_path(str(args.model)))
    print(json.dumps(selfcheck(model), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
