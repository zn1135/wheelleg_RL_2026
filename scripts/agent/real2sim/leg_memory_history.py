"""Estimate friction memory using only measurements at or before replay start."""
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import elastic_joint_resistance as elastic
from scripts.agent.real2sim import replay_leg_test_gpu as replay


def load(path, reference, initial_qpos):
    with np.load(path) as source:
        history = {key: source[key].copy() for key in source.files}
    for key, expected in (('recording_sha256', reference['stream_sha256']),
                          ('source_model_sha256', reference['source_model_sha256']),
                          ('config_sha256', reference['config_sha256'])):
        assert str(history[key]) == expected, 'History belongs to a different recording or calibration'
    t = history['time_board_s']
    assert np.all(np.diff(t) > 0) and t[-1] <= reference['window_start_board_s']
    assert np.max(abs(history['qpos'][-1]-initial_qpos)) < 1e-7
    assert np.isfinite(history['qpos']).all() and np.isfinite(history['qvel']).all()
    return history, dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         time_range_board_s=[float(t[0]), float(t[-1])], samples=len(t),
                         future_measurements_used=False, environment_steps=0)


def estimate(model, data, candidates, history, elastic_motors):
    mq = [model.joint(n).qposadr[0] for n in elastic.MOTORS]
    sq = [model.joint(side+'_spring2_joint').qposadr[0] for side in ('left', 'right')]
    sv = [model.joint(side+'_spring2_joint').dofadr[0] for side in ('left', 'right')]
    q = history['qpos']; v = history['qvel']
    motor_capacity = np.asarray([p['friction_nm'] for p in candidates])
    seal_capacity = np.asarray([p['spring_seal_static_n'] for p in candidates])
    seal_k = np.asarray([p['spring_seal_stiffness_n_m'] for p in candidates])
    seal_positive = np.asarray([p['spring_seal_positive_n'] for p in candidates])
    seal_negative = np.asarray([p['spring_seal_negative_n'] for p in candidates])
    seal_scale = np.asarray([p['spring_seal_velocity_scale_m_s'] for p in candidates])
    if elastic_motors:
        motor_k = np.asarray([p['stiffness_nm_rad'] for p in candidates])
        motor_positive = np.asarray([p.get('motor_dynamic_positive_nm', p['friction_nm']) for p in candidates])
        motor_negative = np.asarray([p.get('motor_dynamic_negative_nm', p['friction_nm']) for p in candidates])
        motor_scale = np.asarray([p.get('motor_velocity_scale_rad_s', [.02]*4) for p in candidates])
    motor_state = []; seal_state = []; initialization = []
    old_q = data.qpos.copy(); old_v = data.qvel.copy(); old_ctrl = data.ctrl.copy()
    base = model.body('base_link').id; old_quat = model.body_quat[base].copy()
    try:
        data.qpos[:] = q[0]; data.qvel[:] = 0.
        quat = history['initial_base_quat']; model.body_quat[base] = quat/np.linalg.norm(quat)
        for case, p in enumerate(candidates):
            params = {key: np.asarray(p[key]) for key in ('spring_reference_force_n', 'spring_stiffness_n_m',
                      'spring_force_limit_n', 'spring_damping_ns_m')}
            force = replay.spring_state(data.qpos[sq], 'linear-damped', np.zeros(2), params)[1]
            residual, info = elastic.initial_preload(model, data, history['initial_model_torque_nm'],
                                                     initial_spring_force=force, return_tangent=True)
            jacobian = np.asarray(info.pop('tangent'))[sv, :].T
            if elastic_motors:
                motor, seal, allocation = elastic.minimum_energy_preload(residual, jacobian, motor_k[case],
                    seal_k[case], motor_capacity[case], seal_capacity[case])
                motor_state.append(motor)
            else:
                seal, allocation = elastic.minimum_seal_energy_preload(residual, jacobian, motor_capacity[case],
                    np.asarray(p.get('motor_friction_bias_nm', [0.]*4)), seal_capacity[case])
            seal_state.append(seal); initialization.append(allocation)
    finally:
        model.body_quat[base] = old_quat; data.qpos[:] = old_q; data.qvel[:] = old_v; data.ctrl[:] = old_ctrl
        mujoco.mj_forward(model, data)
    seal_state = np.asarray(seal_state)
    motor_state = np.asarray(motor_state) if elastic_motors else None
    seal_memory_decay = np.zeros_like(seal_state)
    motor_memory_decay = np.zeros_like(motor_state) if elastic_motors else None
    minimum_loss = 0.
    for i in range(len(q)-1):
        kinetic = np.where(v[i, sv] >= 0, seal_positive, seal_negative)
        limit = kinetic+(seal_capacity-kinetic)*np.exp(-(v[i, sv]/seal_scale)**2)
        delta = q[i+1, sq]-q[i, sq]
        seal_state, loss = elastic.advance(seal_state, delta, limit, seal_k, capacity=seal_capacity)
        seal_memory_decay += seal_k*abs(delta)/limit; minimum_loss = min(minimum_loss, loss)
        if elastic_motors:
            kinetic = np.where(v[i, mq] >= 0, motor_positive, motor_negative)
            limit = kinetic+(motor_capacity-kinetic)*np.exp(-(v[i, mq]/motor_scale)**2)
            delta = q[i+1, mq]-q[i, mq]
            motor_state, loss = elastic.advance(motor_state, delta, limit, motor_k, capacity=motor_capacity,
                                                rule=candidates[0].get('motor_resistance_rule', 'dahl'))
            motor_memory_decay += motor_k*abs(delta)/limit; minimum_loss = min(minimum_loss, loss)
    reports = []
    for case in range(len(candidates)):
        reports.append(dict(initial_allocation=initialization[case], prior_seal_state_n=seal_state[case].tolist(),
                            prior_motor_state_nm=motor_state[case].tolist() if elastic_motors else None,
                            seal_initial_state_survival_factor=np.exp(-seal_memory_decay[case]).tolist(),
                            motor_initial_state_survival_factor=np.exp(-motor_memory_decay[case]).tolist() if elastic_motors else None,
                            minimum_analytic_dissipation_j=minimum_loss))
    return motor_state, seal_state, reports
