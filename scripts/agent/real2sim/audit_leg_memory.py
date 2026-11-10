"""Recompute causal initial states and every archived motor/seal force; zero integration."""
import hashlib
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import elastic_joint_resistance as elastic
from scripts.agent.real2sim import leg_memory_history


def require(condition, message):
    if not condition: raise ValueError(message)


def compare(actual, expected, label, tolerance=2e-5):
    actual = np.asarray(actual); expected = np.asarray(expected)
    require(actual.shape == expected.shape, label+' shape mismatch')
    error = float(np.max(abs(actual-expected)))
    require(np.isfinite(error) and error <= tolerance, f'{label} differs by {error}')
    return error


def audit(model, data, parameters, report, source, saved, original):
    require(report['friction_law'] == 'elastic' and report.get('saved_sample_stride') == 1,
            'Memory audit requires both elastic branches saved at every 1ms step')
    require(not any(key in parameters for key in ('coordinate_friction_nm', 'load_potential_coefficients_nm',
            'motor_friction_bias_nm', 'spring_friction_n', 'motor_friction_decay_m')), 'Unsupported mixed memory model')
    require(parameters.get('preload_rule') == 'minimum_energy_shared', 'Missing shared equilibrium rule')
    metadata = report.get('memory_history'); require(metadata is not None, 'Missing causal memory history')
    path = Path(metadata['path']); require(path.exists(), 'Missing archived history input')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256'], 'Memory-history hash mismatch')
    data.qpos[:] = original['qpos'][0, 0]; data.qvel[:] = original['qvel'][0, 0]
    history, verified_metadata = leg_memory_history.load(path, source, data.qpos)
    require(verified_metadata == metadata, 'Causal input metadata differs')
    motor, seal, observers = leg_memory_history.estimate(model, data, [parameters], history, True)
    archived_observer = report['memory_history_preloads'][0]
    compare(observers[0]['prior_motor_state_nm'], archived_observer['prior_motor_state_nm'], 'Observed motor preload')
    compare(observers[0]['prior_seal_state_n'], archived_observer['prior_seal_state_n'], 'Observed seal preload')
    require(observers[0]['initial_allocation']['equilibrium_max_error_nm'] <= .1,
            'History first pose cannot satisfy the assumed static balance')
    mq = [model.joint(n).qposadr[0] for n in elastic.MOTORS]
    mv = [model.joint(n).dofadr[0] for n in elastic.MOTORS]
    sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
    sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left', 'right')]
    from scripts.agent.real2sim.report_leg_response_joint import spring_force
    residual, info = elastic.initial_preload(model, data, original['submitted_model_torque'][0, 0, :4],
        initial_spring_force=spring_force(data.qpos[sq], np.zeros(2), parameters), return_tangent=True)
    jac = np.asarray(info['tangent'])[sv, :].T
    motor, seal, allocation = elastic.minimum_energy_preload(residual, jac, parameters['stiffness_nm_rad'],
        parameters['spring_seal_stiffness_n_m'], parameters['friction_nm'], parameters['spring_seal_static_n'],
        prior=np.r_[motor[0], seal[0]])
    require(np.max(abs(motor+jac@seal-residual)) <= 1e-5, 'Replay initial memory is not in static balance')
    archived = report['preload_info']['shared_preload_allocation']
    compare(motor, archived['motor_state_nm'], 'Allocated motor preload')
    compare(seal, archived['seal_state_n'], 'Allocated seal preload')
    q = saved['qpos']; v = saved['qvel']; applied = saved['submitted_applied_force']
    require(applied.shape == (len(q)-1, 2, model.nv), 'Applied-force archive shape mismatch')
    require(parameters.get('motor_resistance_rule', 'dahl') == 'dahl', 'Only Dahl recurrence audited here')
    motor = np.tile(motor, (2, 1)); seal = np.tile(seal, (2, 1))
    km = np.asarray(parameters['stiffness_nm_rad']); ks = np.asarray(parameters['spring_seal_stiffness_n_m'])
    fm = np.asarray(parameters['friction_nm']); fs = np.asarray(parameters['spring_seal_static_n'])
    mp = np.asarray(parameters.get('motor_dynamic_positive_nm', fm)); mn = np.asarray(parameters.get('motor_dynamic_negative_nm', fm))
    ms = np.asarray(parameters.get('motor_velocity_scale_rad_s', [.02]*4))
    sp = np.asarray(parameters['spring_seal_positive_n']); sn = np.asarray(parameters['spring_seal_negative_n'])
    ss = np.asarray(parameters['spring_seal_velocity_scale_m_s']); damping = np.asarray(parameters['damping_nms_rad'], dtype=np.float32)
    compare(saved['submitted_motor_friction_capacity'], np.zeros((len(q)-1, 2, 4)), 'Elastic native friction capacity')
    compare(saved['submitted_motor_damping'], np.broadcast_to(damping, (len(q)-1, 2, 4)), 'Physical motor damping')
    maximum_error = 0.; minimum_motor = np.zeros((2, 2)); minimum_seal = np.zeros((2, 2))
    expected = np.zeros((2, model.nv))
    for i in range(len(q)-1):
        expected[:] = 0.; expected[:, mv] = -motor; expected[:, sv] = -seal
        maximum_error = max(maximum_error, float(np.max(abs(applied[i]-expected))))
        motor_limit = np.where(v[i][:, mv] >= 0, mp, mn)
        motor_limit += (fm-motor_limit)*np.exp(-(v[i][:, mv]/ms)**2)
        seal_limit = np.where(v[i][:, sv] >= 0, sp, sn)
        seal_limit += (fs-seal_limit)*np.exp(-(v[i][:, sv]/ss)**2)
        dm = q[i+1][:, mq]-q[i][:, mq]; ds = q[i+1][:, sq]-q[i][:, sq]
        # 独立展开解析递推，不调用拟合器的状态更新函数。
        next_motor = np.sign(dm)*motor_limit+(motor-np.sign(dm)*motor_limit)*np.exp(-km*abs(dm)/motor_limit)
        next_seal = np.sign(ds)*seal_limit+(seal-np.sign(ds)*seal_limit)*np.exp(-ks*abs(ds)/seal_limit)
        require(np.all(abs(next_motor) <= fm+1e-7) and np.all(abs(next_seal) <= fs+1e-7), 'Friction memory exceeds capacity')
        for side in (0, 1):
            section = slice(2*side, 2*side+2)
            work = np.sum(motor[:, section]*dm[:, section]+damping[section]*dm[:, section]**2/source['dt_s'], axis=1)
            stored = np.sum((next_motor[:, section]**2-motor[:, section]**2)/(2*km[section]), axis=1)
            minimum_motor[:, side] = np.minimum(minimum_motor[:, side], work-stored)
        loss = seal*ds-(next_seal**2-seal**2)/(2*ks)+np.asarray(parameters['spring_damping_ns_m'])*v[i][:, sv]*ds
        minimum_seal = np.minimum(minimum_seal, loss); motor = next_motor; seal = next_seal
    require(maximum_error <= 2e-5, f'Applied memory force differs by {maximum_error}')
    minimum = np.min(np.minimum(minimum_motor, minimum_seal), axis=0)
    require(np.min(minimum) >= -1e-7, 'Recomputed memory/physical damping produces discrete energy')
    compare(minimum, report['best']['minimum_discrete_dissipation_j'], 'Per-side discrete memory dissipation', 2e-8)
    return dict(history=verified_metadata, applied_force_max_error=maximum_error,
                recomputed_minimum_discrete_dissipation_j=minimum.tolist(), initial_equilibrium_error_nm=allocation['equilibrium_max_error_nm'],
                scope='Force/history integrity only; response gates are evaluated separately', environment_steps=0)
