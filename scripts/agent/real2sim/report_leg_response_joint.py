"""Recompute saved fixed-candidate leg-response evidence without integrating physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from scripts.agent.real2sim import elastic_joint_resistance as elastic
from scripts.agent.real2sim import replay_leg_test_gpu as replay


ROOT = Path(__file__).resolve().parents[3]
BRANCHES = ['recorded_torque', 'recorded_target_feedback']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(value, parent=ROOT):
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (ROOT, parent):
        if (base / path).exists():
            return (base / path).resolve()
    return (ROOT / path).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-6):
    require(np.shape(actual) == np.shape(expected), f'{label}: shape mismatch')
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    require(np.isfinite(error) and error <= tolerance, f'{label}: maximum discrepancy {error}')
    return error


def metrics(error):
    return (np.sqrt(np.mean(error ** 2, axis=0)), np.max(np.abs(error), axis=0), error[-1])


def spring_force(q, v, parameters):
    """Independent diagnostic force curve; its fit is not a physical spring calibration."""
    compression = np.maximum(0., .010 - np.asarray(q))
    if 'spring_compression_knots_m' in parameters:
        knots = np.asarray(parameters['spring_compression_knots_m'], dtype=float)
        forces = np.asarray(parameters['spring_force_knots_n'], dtype=float)
        require(knots.ndim == 1 and len(knots) >= 2 and np.isfinite(knots).all()
                and knots.min() >= 0. and np.all(np.diff(knots) > 0.), 'Invalid compression knot grid')
        require(forces.shape == (2, len(knots)) and np.isfinite(forces).all()
                and forces.min() >= 0. and np.all(np.diff(forces, axis=1) >= 0.), 'Force knots must be nonnegative and monotone')
        # 端点外保持端点值；左右侧共用压缩量网格，各有自己的等效负载。
        force = np.stack([np.interp(compression[..., side], knots, forces[side])
                          for side in range(2)], axis=-1)
    else:
        force = (np.asarray(parameters['spring_reference_force_n'])
                 + np.asarray(parameters['spring_stiffness_n_m']) * (compression - .005))
    return np.clip(force - np.asarray(parameters.get('spring_damping_ns_m', 50.)) * np.asarray(v),
                   0., np.asarray(parameters['spring_force_limit_n']))


def native_motor_friction(slide, velocity, parameters):
    static = np.asarray(parameters['friction_nm'], dtype=float)
    positive = np.asarray(parameters['motor_dynamic_positive_nm'], dtype=float)
    negative = np.asarray(parameters['motor_dynamic_negative_nm'], dtype=float)
    scale = np.asarray(parameters['motor_velocity_scale_rad_s'], dtype=float)
    for value in (static, positive, negative, scale):
        require(value.shape == (4,) and np.isfinite(value).all(), 'Invalid native motor friction parameters')
    require(np.all(static >= positive) and np.all(static >= negative)
            and np.min(positive) > 0. and np.min(negative) > 0. and np.min(scale) > 0.,
            'Native static capacity must exceed both nonnegative kinetic capacities')
    kinetic = np.where(velocity >= 0., positive, negative)
    effective = kinetic + (static - kinetic) * np.exp(-(velocity / scale) ** 2)
    if 'motor_friction_decay_m' in parameters:
        decay = np.asarray(parameters['motor_friction_decay_m'], dtype=float)
        require(decay.shape == (2,) and np.isfinite(decay).all() and np.min(decay) > 0.,
                'Native friction compression decay lengths must be positive')
        compression = np.repeat(np.maximum(0., .010 - slide), 2, axis=-1)
        effective *= np.exp(-compression / np.repeat(decay, 2))
    require(np.isfinite(effective).all() and np.min(effective) >= 0.
            and np.all(effective <= static + 1e-12), 'Invalid effective native friction capacity')
    return effective


def directional_static_capacity(slide,velocity,parameters):
    nominal=np.asarray(parameters['friction_nm'],dtype=float)
    bias=np.asarray(parameters.get('motor_friction_bias_nm',[0.]*4),dtype=float)
    require(nominal.shape==bias.shape==(4,) and np.isfinite(nominal).all() and np.isfinite(bias).all()
            and np.all(nominal>=abs(bias)),'Directional breakaway interval can inject energy')
    capacity=np.broadcast_to(nominal,np.shape(velocity)).copy()
    if 'motor_static_excess_nm' in parameters:
        extra=np.asarray(parameters['motor_static_excess_nm']);decay=np.asarray(parameters['motor_static_decay_m'])
        scale=np.asarray(parameters['motor_static_velocity_scale_rad_s'])
        require(extra.shape==scale.shape==(4,) and decay.shape==(2,) and np.isfinite(extra).all()
                and np.min(extra)>=0 and np.isfinite(decay).all() and np.min(decay)>0
                and np.isfinite(scale).all() and np.min(scale)>0,'Invalid position-dependent static friction')
        compression=np.repeat(np.maximum(0.,.010-slide),2,axis=-1)
        capacity+=extra*np.exp(-compression/np.repeat(decay,2))*np.exp(-(velocity/scale)**2)
    power=(capacity*np.sign(velocity)+bias)*velocity
    require(np.isfinite(capacity).all() and np.min(power)>=-1e-9,'Ideal native resistance would create energy')
    return capacity,float(np.min(power))


def has_applied_state(parameters):
    return any(key in parameters for key in ('stiffness_nm_rad','coordinate_friction_nm','spring_seal_static_n','load_potential_coefficients_nm'))


def command_argv(path):
    record = json.loads(path.read_text())
    if isinstance(record, list):
        return record
    for key in ('argv', 'command'):
        if isinstance(record.get(key), list):
            return record[key]
    return None


def rc_path(directory, report, reference):
    if report.get('rc_inputs'):
        return locate(report['rc_inputs'], directory)
    candidates = [directory / 'command.json', directory.parent / (directory.name + '-command.json')]
    for path in candidates:
        if not path.exists():
            continue
        argv = command_argv(path)
        if argv and '--rc-inputs' in argv:
            if '--output' in argv:
                actual_output = locate(argv[argv.index('--output') + 1], directory.parent)
                require(actual_output == directory.resolve(), f'Command output mismatch: {path}')
            return locate(argv[argv.index('--rc-inputs') + 1], directory.parent)
    if (reference / 'rc-inputs.npz').exists():
        return reference / 'rc-inputs.npz'
    for parent in directory.parents:
        manifest_path = parent / 'manifest.json'
        if not manifest_path.exists():
            continue
        for item in json.loads(manifest_path.read_text()).get('references', []):
            if locate(item['reference'], parent) == reference.resolve():
                return locate(item['rc_inputs'], parent)
    raise ValueError(f'No archived RC-input path for {directory}')


def archived_rc_hash(directory, reference, rc):
    for path in (directory / 'command.json', directory.parent / (directory.name + '-command.json')):
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            continue
        for path_value, expected in record.get('inputs_sha256', {}).items():
            if locate(path_value, directory.parent) == rc.resolve():
                return expected
    for parent in directory.parents:
        path = parent / 'manifest.json'
        if path.exists():
            for item in json.loads(path.read_text()).get('references', []):
                if locate(item['reference'], parent) == reference.resolve():
                    return item.get('rc_inputs_sha256')
    return None


def command_input_hashes(directory):
    checked = {}
    for path in (directory / 'command.json', directory.parent / (directory.name + '-command.json')):
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if isinstance(record, dict):
            for path_value, expected in record.get('inputs_sha256', {}).items():
                source = locate(path_value, directory.parent)
                require(digest(source) == expected, f'Archived command input hash mismatch: {source}')
                checked[str(source)] = expected
    return checked


def protected_sources(experiment, config_paths):
    before = json.loads((experiment / 'before-sources.json').read_text())
    sources = {path: value for path, value in before['sources'].items()
               if (Path(path).suffix == '.xml' or Path(path).name == 'gas_spring.py')}
    require(bool(sources), 'Protected XML/spring hash manifest is empty')
    for path, expected in sources.items():
        require(digest(locate(path, experiment)) == expected, f'Protected source changed: {path}')
    firmware = before['firmware_sources']
    require(bool(firmware), 'Firmware hash manifest is empty')
    firmware_root = before.get('firmware_root', before.get('firmware_source_root'))
    if firmware_root:
        firmware_root = locate(firmware_root, experiment)
    else:
        matches = []
        for config in config_paths:
            for ancestor in config.parents:
                relative = str(config.relative_to(ancestor))
                if relative in firmware and digest(config) == firmware[relative]:
                    matches.append(ancestor)
        unique = set(matches)
        require(len(unique) == 1, 'Cannot infer a unique firmware source root from the hash manifest')
        firmware_root = unique.pop()
    for path, expected in firmware.items():
        require(digest(firmware_root / path) == expected, f'Firmware source changed: {path}')
    return dict(protected_sources=sources, firmware_root=str(firmware_root),
                unchanged_firmware_sources=len(firmware))


def audit_run(directory, fixed_parameters):
    report = json.loads((directory / 'report.json').read_text())
    require(report['physics_backend'] == 'mujoco_warp_gpu', 'Run was not a GPU replay')
    require(report['worlds'] == 2 and len(report['trials']) == 1, 'Expected one candidate and two input branches')
    require(report['objective_branches'] == BRANCHES, 'Incorrect replay branches')
    trial = report['trials'][0]
    parameters = trial['parameters']
    require('load_potential_coefficients_nm' not in parameters,
            'This verifier does not yet audit potential applied-force histories')
    if fixed_parameters is not None:
        require(parameters == fixed_parameters, 'Runs use different fixed candidate parameters')
    require(report['best']['parameters'] == parameters, 'Best-candidate metadata mismatch')
    reference = locate(report['reference'], directory)
    source = json.loads((reference / 'report.json').read_text())
    input_hashes = command_input_hashes(directory)
    if source.get('diagnostic_model_sha256'):
        require(digest(reference / 'fixed-base-model.xml') == source['diagnostic_model_sha256'], 'Reference diagnostic model hash mismatch')
    require(report['source_model_sha256'] == source['source_model_sha256']
            == digest(locate(source['source_model'], reference)), 'Source model hash mismatch')
    session = locate(source['session'], reference)
    require(report['stream_sha256'] == source['stream_sha256'] == digest(session / 'stream.jsonl'),
            'Recorded stream hash mismatch')
    meta = json.loads((session / 'meta.json').read_text())
    config = locate(meta['machine_config']['path'], session)
    require(digest(config) == source['config_sha256'] == meta['machine_config']['sha256'], 'Config hash mismatch')
    rc = rc_path(directory, report, reference)
    with np.load(rc) as cached:
        require(str(cached['stream_sha256']) == source['stream_sha256'], 'RC stream hash mismatch')
        require(np.isfinite(cached['time_s']).all() and np.isfinite(cached['dial_rate']).all(), 'Nonfinite RC cache')
    expected_rc_hash = report.get('rc_inputs_sha256') or archived_rc_hash(directory, reference, rc)
    if expected_rc_hash:
        require(digest(rc) == expected_rc_hash, 'RC file hash mismatch')
    compiled = mujoco.MjModel.from_xml_path(str(directory / 'candidate-model.xml'))
    require(not np.any(compiled.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 'Expected a fixed-base diagnostic')
    contact_mask = ((compiled.geom_contype[:, None] & compiled.geom_conaffinity[None, :])
                    | (compiled.geom_conaffinity[:, None] & compiled.geom_contype[None, :]))
    require(not np.any(contact_mask), 'Compiled diagnostic model permits contact pairs')
    # mj_saveLastXML 会舍入几何小数；闭链/预载用未舍入的同一诊断模板重建。
    template = directory / 'fixed-base-template.xml'
    if not template.exists():
        template = reference / 'fixed-base-model.xml'
    model = mujoco.MjModel.from_xml_path(str(template))
    require((model.nq, model.nv, model.nu) == (compiled.nq, compiled.nv, compiled.nu), 'Diagnostic template dimensions differ')
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    model.actuator_ctrlrange[:] = compiled.actuator_ctrlrange
    model.actuator_forcerange[:] = compiled.actuator_forcerange
    mq = [model.joint(name).qposadr[0] for name in replay.mirror.MOTORS[:4]]
    mv = [model.joint(name).dofadr[0] for name in replay.mirror.MOTORS[:4]]
    sq = [model.joint(side + '_spring2_joint').qposadr[0] for side in ('left', 'right')]
    sv = [model.joint(side + '_spring2_joint').dofadr[0] for side in ('left', 'right')]
    motor_ids = [model.actuator(name + '_motor').id for name in replay.mirror.MOTORS]
    spring_ids = [model.actuator(side + '_gas_spring_motor').id for side in ('left', 'right')]
    close(compiled.dof_armature[mv], parameters['armature_kg_m2'], 'Motor armature')
    model.dof_armature[mv] = parameters['armature_kg_m2']
    model.dof_frictionloss[mv] = 1e-6
    model.dof_solref[mv] = [.002, 1.]
    model.dof_solimp[mv] = [.999, .9999, .001, .5, 2.]
    if 'spring_extension_stop_m' in parameters:
        for side,name in enumerate(('left','right')):
            j=model.joint(name+'_spring2_joint').id
            model.jnt_limited[j]=True;model.jnt_range[j]=[-.09,parameters['spring_extension_stop_m'][side]]
            model.jnt_margin[j]=parameters.get('spring_extension_margin_m',[0.,0.])[side]
            model.jnt_solref[j]=[.002,1.];model.jnt_solimp[j]=[.999,.9999,.001,.5,2.]
        stop_ids=[model.joint(name+'_spring2_joint').id for name in ('left','right')]
        # 保存XML会舍入远处的主动轴大范围；只核对本候选实际修改的两个滑杆限位。
        close(compiled.jnt_range[stop_ids],model.jnt_range[stop_ids],'Diagnostic extension-stop ranges',1e-9)
        close(compiled.jnt_margin[stop_ids],model.jnt_margin[stop_ids],'Diagnostic extension-stop margins',1e-9)
        require(np.array_equal(compiled.jnt_limited[stop_ids],model.jnt_limited[stop_ids]),'Diagnostic extension-stop enable flags differ')
    mujoco.mj_setConst(model, mujoco.MjData(model))
    data = mujoco.MjData(model)
    limit = np.asarray(parameters['spring_force_limit_n'])
    require(np.all(model.actuator_forcerange[spring_ids, 0] <= 0.)
            and np.all(model.actuator_forcerange[spring_ids, 1] >= limit)
            and np.all(model.actuator_ctrlrange[spring_ids, 0] <= 0.)
            and np.all(model.actuator_ctrlrange[spring_ids, 1] >= limit), 'Candidate force is clipped by the diagnostic actuator range')
    close(compiled.actuator_gear[spring_ids], model.actuator_gear[spring_ids], 'Spring actuator gear')
    close(compiled.actuator_gainprm[spring_ids], model.actuator_gainprm[spring_ids], 'Spring actuator gain')
    close(compiled.actuator_biasprm[spring_ids], model.actuator_biasprm[spring_ids], 'Spring actuator bias')
    sign, zero = replay.mirror.load_mapping(config)
    with np.load(directory / 'trials.npz') as saved, np.load(reference / 'replay.npz') as original:
        q = saved['qpos']; v = saved['qvel']; times = saved['time_s']; real = saved['real_motor_q']
        require(q.shape[1:] == (2, model.nq) and v.shape[1:] == (2, model.nv), 'Saved-state shape mismatch')
        require(np.isfinite(q).all() and np.isfinite(v).all(), 'Nonfinite saved state')
        native_friction = None
        ideal_native_power=None
        if report['friction_law'] == 'coulomb' and 'motor_dynamic_positive_nm' in parameters:
            native_friction = native_motor_friction(q[:, :, sq], v[:, :, mv], parameters)
        elif 'motor_friction_decay_m' in parameters:
            raise ValueError('Compression-dependent native friction requires the native kinetic-capacity family')
        elif report['friction_law']=='coulomb' and ('motor_static_excess_nm' in parameters or 'motor_friction_bias_nm' in parameters):
            native_friction,ideal_native_power=directional_static_capacity(q[:,:,sq],v[:,:,mv],parameters)
        close(q[0], np.broadcast_to(original['qpos'][0, 0], q[0].shape), 'Initial pose', 1e-7)
        close(v[0], np.broadcast_to(original['qvel'][0, 0], v[0].shape), 'Initial velocity', 1e-7)
        steps = int(report['steps_per_world'])
        stride = int(report.get('saved_sample_stride', round(float(times[1] - times[0]) / source['dt_s'])))
        indices = np.arange(0, steps + 1, stride)
        close(times, original['time_s'][indices], 'Saved sample times', 1e-10)
        require(report['new_environment_steps'] == 2 * steps, 'GPU step count mismatch')
        absolute = times + source['window_start_board_s']
        measured = original['real_motor_state']
        expected = np.column_stack([np.interp(absolute, measured[:, 0], measured[:, 1 + index]) for index in range(4)])
        close(real, expected, 'Measured trajectory interpolation', 1e-12)
        predicted = q[:, :, mq] * sign[:4] + zero
        error = np.degrees(predicted - real[:, None, :])
        rmse, peak, final = metrics(error)
        for branch, prefix in enumerate(('', 'feedback_')):
            close(rmse[branch], trial['all_motor_rmse_deg' if not prefix else 'feedback_motor_rmse_deg'], prefix + 'RMSE')
            close(peak[branch], trial[prefix + 'peak_error_deg'], prefix + 'peak')
            close(final[branch], trial[prefix + 'final_error_deg'], prefix + 'terminal')
        formula = spring_force(q[:, :, sq], v[:, :, sv], parameters)
        close(formula, replay.spring_state(q[:, :, sq], 'linear-damped', v[:, :, sv], parameters)[1], 'Spring implementation', 1e-9)
        control_checks = dict(archived='submitted_control' in saved.files)
        if control_checks['archived']:
            controls = saved['submitted_control']
            require(controls.shape == (len(times) - 1, 2, model.nu), 'Submitted-control shape mismatch')
            recorded = original['submitted_model_torque'][indices[:-1], 0].astype(np.float32)
            require(np.array_equal(controls[:, 0, motor_ids], recorded), 'Recorded-torque motor inputs differ from the verified reference')
            force_error = close(controls[:, :, spring_ids], formula[:-1], 'Submitted pre-step spring force', 1e-4)
            control_checks.update(recorded_motor_input_exact=True, pre_step_spring_force_max_error_n=force_error)
        smooth_work_min=None
        if 'submitted_motor_friction_capacity' in saved.files:
            actual=saved['submitted_motor_friction_capacity']
            require(actual.shape==(len(times)-1,2,4),'Archived native friction-capacity shape mismatch')
            if report.get('smooth_motor_resistance'):
                close(actual,np.zeros_like(actual),'Smooth model must not retain native dry friction')
            elif native_friction is not None:
                control_checks['native_friction_capacity_max_error_nm']=close(actual,native_friction[:-1],'Submitted native friction capacities',2e-5)
        slider_friction=np.asarray(parameters.get('spring_friction_n',[0.,0.]),dtype=float)
        require(slider_friction.shape==(2,) and np.isfinite(slider_friction).all() and slider_friction.min()>=0.,
                'Invalid native slider friction capacity')
        if np.any(slider_friction):require('submitted_slider_friction_capacity' in saved.files,'Native slider friction capacities were not archived')
        if 'submitted_slider_friction_capacity' in saved.files:
            actual=saved['submitted_slider_friction_capacity']
            expected=np.broadcast_to(slider_friction,(len(times)-1,2,2))
            control_checks['slider_friction_capacity_max_error_n']=close(actual,expected,'Submitted native slider friction capacities',2e-5)
        if report.get('smooth_motor_resistance'):
            require('submitted_motor_damping' in saved.files,'Smooth-resistance force coefficients were not archived')
            positive=np.asarray(parameters['motor_smooth_positive_nm']);negative=np.asarray(parameters['motor_smooth_negative_nm'])
            scale=np.asarray(parameters['motor_smooth_velocity_scale_rad_s'])
            require(all(p.shape==(4,) and np.isfinite(p).all() and p.min()>0 for p in (positive,negative,scale)),
                    'Invalid positive smooth-resistance parameters')
            velocity=v[:-1,:,mv];amplitude=np.where(velocity>0,positive,np.where(velocity<0,negative,(positive+negative)/2))
            ratio=np.broadcast_to(1./scale,velocity.shape).copy()
            np.divide(np.tanh(velocity/scale),velocity,out=ratio,where=abs(velocity)>1e-10)
            expected=np.asarray(parameters['damping_nms_rad'],dtype=np.float32)+amplitude*ratio
            actual=saved['submitted_motor_damping']
            require(actual.shape==expected.shape and np.isfinite(actual).all() and actual.min()>=0
                    and np.allclose(actual,expected,rtol=3e-7,atol=2e-5),'Archived smooth damping differs from the independent resistance formula')
            control_checks['smooth_damping_max_relative_error']=float(np.max(abs(actual-expected)/np.maximum(abs(expected),1.)))
            if stride==1:
                displacement=q[1:,:,mq]-q[:-1,:,mq]
                work=np.sum(actual*v[1:,:,mv]*displacement,axis=2)
                smooth_work_min=float(np.min(work))
                require(smooth_work_min>=-1e-7,'Implicit smooth resistance would create discrete energy')
        if 'submitted_applied_force' in saved.files and not has_applied_state(parameters):
            applied=saved['submitted_applied_force'];expected=np.zeros_like(applied)
            expected[:,:,mv]=-np.asarray(parameters.get('motor_friction_bias_nm',[0.]*4))
            control_checks['constant_applied_friction_bias_max_error_nm']=close(applied,expected,'Submitted constant directional friction bias',2e-6)
        initial_preload = None
        has_elastic = any(np.any(np.asarray(parameters.get(key, [0.] * 4)) > 0.)
                          for key in ('stiffness_nm_rad', 'coordinate_stiffness_nm_rad'))
        if has_elastic:
            data.qpos[:] = original['qpos'][0, 0]; data.qvel[:] = original['qvel'][0, 0]
            initial_force = spring_force(data.qpos[sq], np.zeros(2), parameters)
            residual, preload = elastic.initial_preload(model, data, original['submitted_model_torque'][0, 0, :4],
                                                       initial_spring_force=initial_force)
            archived = report.get('preload_info')
            if archived is not None:
                close(preload['holding_torque_nm'], archived['holding_torque_nm'], 'Motor elastic initial holding torque')
            coordinate_info = report.get('coordinate_preload_info')
            if coordinate_info:
                close(preload['holding_torque_nm'], coordinate_info[0]['holding_torque_nm'], 'Coordinate elastic initial holding torque')
            initial_preload = dict(residual_nm=residual.tolist(), **preload)
        if 'spring_seal_static_n' in parameters:
            from scripts.agent.real2sim.audit_leg_memory import audit
            control_checks['motor_seal_memory']=audit(model,data,parameters,report,source,saved,original)
        closure = np.zeros(2)
        for position in q.reshape(-1, model.nq):
            data.qpos[:] = position
            mujoco.mj_kinematics(model, data)  # 只从保存的位置算连杆坐标，不推进物理。
            for side in range(2):
                closure[side] = max(closure[side], max(np.linalg.norm(data.site(a).xpos - data.site(b).xpos)
                                                       for a, b in replay.mirror.PIN_PAIRS[side]))
        close(closure, report['best_sampled_pin_error_m'], 'Saved-state pin closure', 1e-8)
        loss = np.asarray(trial['minimum_discrete_dissipation_j'], dtype=float)
        require(loss.shape == (2,) and np.isfinite(loss).all(), 'Missing/nonfinite per-side dissipation evidence')
        passive = bool(np.min(loss) >= -1e-7)
        coordinate_viscous = report.get('minimum_explicit_coordinate_viscous_work_j')
        if coordinate_viscous is not None:
            require(np.isfinite(coordinate_viscous), 'Nonfinite explicit coordinate viscous work')
        coordinate_work_recomputed = None
        if 'coordinate_damping_nms_rad' in parameters and stride == 1:
            transform = np.kron(np.eye(2), np.array([[.5, .5], [.5, -.5]]))
            velocity = v[:-1, :, mv] @ transform.T
            displacement = (q[1:, :, mq] - q[:-1, :, mq]) @ transform.T
            work = np.sum(np.asarray(parameters['coordinate_damping_nms_rad']) * velocity * displacement, axis=2)
            coordinate_work_recomputed = min(0., float(np.min(work)))
            if coordinate_viscous is not None:
                close(coordinate_work_recomputed, coordinate_viscous, 'Explicit coordinate viscous force-displacement work', 1e-7)
        passed = bool(np.all(rmse < 3.) and np.all(peak < 5.) and np.all(np.abs(final) < 1.)
                      and np.max(closure) < .0001 and not trial['feedback_blocked'] and passive)
        row = dict(run=str(directory.resolve()), reference=str(reference), duration_s=float(times[-1]),
                   data_role='fixed_parameter_development_crosscheck', integrity_passed=True,
                   response_thresholds_passed=passed, recording_sha256=source['stream_sha256'],
                   source_model_sha256=source['source_model_sha256'], config_sha256=source['config_sha256'],
                   geometry_template=str(template), geometry_template_sha256=digest(template),
                   rc_inputs=str(rc), rc_inputs_sha256=digest(rc), sample_stride=stride,
                   rc_file_hash_verified_against_archive=expected_rc_hash is not None,
                   archived_command_input_hashes=input_hashes,
                   joint_rmse_deg=rmse.tolist(), peak_error_deg=peak.tolist(), final_error_deg=final.tolist(),
                   closure_error_m=closure.tolist(), feedback_blocked=trial['feedback_blocked'],
                   control_checks=control_checks, initial_preload=initial_preload,
                   minimum_discrete_dissipation_j=loss.tolist(), discrete_passivity_passed=passive,
                   minimum_explicit_coordinate_viscous_work_j=coordinate_viscous,
                   recomputed_explicit_coordinate_viscous_work_j=coordinate_work_recomputed,
                   min_effective_native_motor_friction_nm=(np.min(native_friction, axis=(0, 1)).tolist()
                                                           if native_friction is not None else None),
                   max_effective_native_motor_friction_nm=(np.max(native_friction, axis=(0, 1)).tolist()
                                                           if native_friction is not None else None),
                   mathematical_native_friction_nonnegative=(True if native_friction is not None else None),
                   ideal_directional_native_resistance_min_power_w=ideal_native_power,
                   minimum_recomputed_smooth_viscous_work_j=smooth_work_min,
                   effective_native_friction_is_measured_parameter=False,
                   diagnostic_spring_force_limit_n=limit.tolist(),
                   force_curve_family=('monotone_piecewise_effective_load' if 'spring_compression_knots_m' in parameters
                                       else 'reference_force_plus_linear_compression'),
                   force_limit_is_measured_hardware_maximum=False)
        plot_data = (times.copy(), np.degrees(real), np.degrees(predicted))
    return row, parameters, config, plot_data


def completed_gpu_reports(experiment):
    reports = []
    seen = set()
    for path in sorted(experiment.rglob('report.json')):
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        report = json.loads(path.read_text())
        if (report.get('physics_backend') != 'mujoco_warp_gpu' or 'worlds' not in report
                or 'steps_per_world' not in report or not (path.parent / 'trials.npz').exists()):
            continue
        require(report['new_environment_steps'] == report['worlds'] * report['steps_per_world'],
                f'Completed GPU batch step count mismatch: {path}')
        reports.append(dict(report=str(path.resolve()), environment_steps=report['new_environment_steps'],
                            candidate_evaluations=len(report['trials'])))
    return reports


def plot_run(destination, plot_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times, real, predicted = plot_data
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(times, real[:, index], label='Measured', color='black')
        axis.plot(times, predicted[:, 0, index], label='Recorded torque', alpha=.85)
        axis.plot(times, predicted[:, 1, index], label='Recorded target feedback', alpha=.85)
        axis.set_ylabel(replay.mirror.MOTORS[index] + ' (deg)'); axis.grid(alpha=.25)
    axes[0].legend(); axes[-1].set_xlabel('Time (s)'); fig.tight_layout()
    fig.savefig(destination, dpi=130); plt.close(fig)


def selfcheck():
    error = np.array([[[1., -2., 3., -4.], [2., -3., 4., -5.]],
                      [[-1., 2., -3., 4.], [-2., 3., -4., 5.]]])
    rmse, peak, final = metrics(error)
    require(np.array_equal(rmse, np.abs(error[0])) and np.array_equal(peak, rmse)
            and np.array_equal(final, error[-1]), 'Metric selfcheck failed')
    parameters = dict(spring_reference_force_n=[150., 170.], spring_stiffness_n_m=[1000., 2000.],
                      spring_damping_ns_m=[50., 100.], spring_force_limit_n=[600., 600.])
    q = np.array([[.005, .005], [0., -.01]])
    v = np.array([[0., 0.], [.1, -.1]])
    close(spring_force(q, v, parameters), [[150., 170.], [150., 210.]], 'Spring formula selfcheck', 1e-12)
    parameters.update(spring_compression_knots_m=[0., .005, .01, .02],
                      spring_force_knots_n=[[100., 150., 160., 180.], [120., 170., 190., 200.]])
    close(spring_force([[.005, .005], [-.02, -.03]], v, parameters), [[150., 170.], [175., 210.]],
          'Piecewise force/damping/endpoint selfcheck', 1e-12)
    close(spring_force([.02, .03], [0., 0.], parameters), [100., 120.], 'Lower endpoint selfcheck', 1e-12)
    close(spring_force([.0025, .0025], [0., 0.], parameters), [155., 180.], 'Knot interpolation selfcheck', 1e-12)
    friction = dict(friction_nm=[10.] * 4, motor_dynamic_positive_nm=[2.] * 4,
                    motor_dynamic_negative_nm=[4.] * 4, motor_velocity_scale_rad_s=[.1] * 4,
                    motor_friction_decay_m=[.01, .02])
    close(native_motor_friction(np.array([0., 0.]), np.zeros(4), friction),
          10. * np.exp(-np.array([1., 1., .5, .5])), 'Native capacity compression decay selfcheck', 1e-12)
    close(native_motor_friction(np.array([.02, .02]), np.array([1., -1., 1., -1.]), friction),
          [2., 4., 2., 4.], 'Native directional kinetic capacity selfcheck', 1e-12)
    print(json.dumps(dict(selfcheck='passed', metric_axes=4, spring_formula_cases=12,
                         native_friction_cases=8, physics_steps=0)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--runs', type=Path, nargs='+', action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); return
    if args.experiment is None or not args.runs or args.output is None:
        parser.error('--experiment, --runs and --output are required')
    directories = [path.resolve() for group in args.runs for path in group]
    require(len(set(directories)) == len(directories), 'Duplicate run directories')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; errors = []; parameters = None; configs = []
    for index, directory in enumerate(directories):
        try:
            row, candidate, config, plot_data = audit_run(directory, parameters)
            if parameters is None:
                parameters = candidate
            configs.append(config)
            plot = f'{index:02d}-{directory.name}.png'
            plot_run(args.output / plot, plot_data)
            row['plot'] = plot
            rows.append(row)
        except (ValueError, KeyError, OSError, AssertionError) as error:
            errors.append(dict(run=str(directory), error=str(error)))
    protected = None; completed = []
    try:
        protected = protected_sources(args.experiment, configs)
        completed = completed_gpu_reports(args.experiment)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        errors.append(dict(check='protected sources or GPU accounting', error=str(error)))
    integrity = not errors and len(rows) == len(directories)
    response = integrity and all(row['response_thresholds_passed'] for row in rows)
    result = dict(integrity_passed=integrity, response_thresholds_passed=response,
                  status='candidate_unaccepted', training_model_adopted=False,
                  fixed_base_development_thresholds_passed=response,
                  independent_hardware_acceptance=False, motor_order=list(replay.mirror.MOTORS[:4]),
                  branch_order=BRANCHES, fixed_candidate_parameters=parameters, runs=rows, errors=errors,
                  protected_source_checks=protected, completed_gpu_batches=completed,
                  total_completed_gpu_environment_steps=sum(item['environment_steps'] for item in completed),
                  candidate_evaluations=sum(item['candidate_evaluations'] for item in completed),
                  new_environment_steps_in_this_report=0, policy_training_updates=0,
                  limitations=['All supplied trajectories are development/fixed-parameter crosschecks, not independent hardware acceptance',
                               'Fitted force curves represent effective load and are not physical gas-spring calibration',
                               'Configuration-dependent friction parameters are effective mathematical resistance, not measured motor properties',
                               'Spring force caps are diagnostic actuator limits, not measured physical maxima',
                               'Saved samples validate only their archived sampling frequency',
                               'Commanded torque is an input assumption; shaft torque and fixture motion were not independently measured'])
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if parameters is not None:
        (args.output / 'candidate-unaccepted.json').write_text(json.dumps(dict(status='candidate_unaccepted',
             fixed_base_development_thresholds_passed=response, independent_hardware_acceptance=False,
             parameters=parameters), indent=2) + '\n')
    text = ['# 固定候选复核', '', f'数据完整性：{integrity}；响应门槛：{response}。参数状态：candidate_unaccepted。', '',
            '以下均为固定机身、无接触的开发数据复查，不代表独立真机验收。', '',
            '| 轨迹 | 同力矩四轴RMSE (°) | 同目标四轴RMSE (°) | 最大峰值 (°) | 最大末端 (°) | 闭链间隙 (mm) | 门槛 |',
            '|---|---|---|---:|---:|---:|---|']
    for row in rows:
        torque = ', '.join(f'{value:.2f}' for value in row['joint_rmse_deg'][0])
        feedback = ', '.join(f'{value:.2f}' for value in row['joint_rmse_deg'][1])
        text.append(f"| [{Path(row['run']).name}]({row['plot']}) | {torque} | {feedback} | "
                    f"{np.max(row['peak_error_deg']):.2f} | {np.max(np.abs(row['final_error_deg'])):.2f} | "
                    f"{np.max(row['closure_error_m'])*1000.:.4f} | {row['response_thresholds_passed']} |")
    text.extend(['', '门槛逐轴、双分支同时要求：RMSE < 3°、峰值 < 5°、末端绝对误差 < 1°、闭链间隙 < 0.1 mm，反馈不锁止，逐侧最小离散耗散 ≥ -1e-7 J。', '',
                 '显式坐标阻尼功单列；若保存了每个物理步，则用步前速度对应的阻尼力乘该步实际位移复算，不用 B·Δq²/Δt 替代。', '',
                 f"本报告新增物理积分步数：0；实验目录内已完成GPU积分总步数：{result['total_completed_gpu_environment_steps']}。", '',
                 '弹簧力上限仅用于避免诊断执行器夹限，不能解释为产品的实测最大推力。'])
    if errors:
        text.extend(['', '复核错误：'] + [f'- {item}' for item in errors])
    (args.output / 'README.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(dict(integrity_passed=integrity, response_thresholds_passed=response,
                          runs=len(rows), errors=errors, physics_steps=0), indent=2))
    if not integrity:
        sys.exit(1)


if __name__ == '__main__':
    main()
