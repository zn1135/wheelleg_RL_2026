"""Finite response fitting with repaired loops/limits, 0.2ms GPU physics and held 1ms commands."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp

from scripts.agent.real2sim import fit_leg_response_gpu as prior
from scripts.agent.real2sim import refine_leg_response_joint as search
from scripts.agent.real2sim import report_leg_response_joint as audit


@wp.kernel
def forces(q: wp.array2d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32),
           params: wp.array2d(dtype=wp.float32), mv: wp.array(dtype=wp.int32),
           sq: wp.array(dtype=wp.int32), sv: wp.array(dtype=wp.int32), sid: wp.array(dtype=wp.int32),
           friction: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32)):
    world = wp.tid()
    for j in range(4):
        speed = v[world, mv[j]]
        dynamic = params[world, 5*j+1]
        if speed < 0.: dynamic = params[world, 5*j+2]
        ratio = speed/params[world, 5*j+3]
        friction[world, mv[j]] = dynamic+(params[world, 5*j]-dynamic)*wp.exp(-ratio*ratio)
    for side in range(2):
        offset = 20+4*side
        compression = wp.max(0., .010-q[world, sq[side]])
        value = params[world, offset]+params[world, offset+1]*(compression-.005)
        value = value-params[world, offset+2]*v[world, sv[side]]
        ctrl[world, sid[side]] = wp.clamp(value, 0., 600.)
        # 静态往返曲线的半差作为滑杆阻力容量，只依赖当前仿真压缩量。
        friction[world, sv[side]] = wp.max(0., params[world, offset+3]+params[world, 28+side]*(compression-.005))


@wp.kernel
def archive(q: wp.array2d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32),
            ctrl: wp.array2d(dtype=wp.float32), friction: wp.array2d(dtype=wp.float32),
            applied: wp.array2d(dtype=wp.float32), oa: wp.array3d(dtype=wp.float32),
            index: wp.array(dtype=wp.int32), oq: wp.array3d(dtype=wp.float32),
            ov: wp.array3d(dtype=wp.float32), oc: wp.array3d(dtype=wp.float32),
            of: wp.array3d(dtype=wp.float32), nv: int, nu: int):
    world = wp.tid(); k = index[0]
    for j in range(nv):
        oq[k, world, j] = q[world, j]; ov[k, world, j] = v[world, j]
        of[k, world, j] = friction[world, j]
        oa[k, world, j] = applied[world, j]
    for j in range(nu): oc[k, world, j] = ctrl[world, j]


@wp.kernel
def advance(index: wp.array(dtype=wp.int32)):
    index[0] = index[0]+1


REFERENCES = {
    'original': ('artifacts/real2sim/h7-response-align-20261104-01/final-elastic-replay',
                 'artifacts/real2sim/h7-response-align-20261104-01/rc-inputs.npz'),
    'wide': ('artifacts/real2sim/h7-joint-response-refit-20261104-01/input-wide',
             'artifacts/real2sim/h7-joint-response-refit-20261104-01/input-wide/rc-inputs.npz'),
    'slow': ('artifacts/real2sim/h7-joint-response-refit-20261104-01/input-near-slow',
             'artifacts/real2sim/h7-joint-response-refit-20261104-01/input-near-slow/rc-inputs.npz'),
    'fast': ('artifacts/real2sim/h7-joint-response-refit-20261104-01/input-near-fast',
             'artifacts/real2sim/h7-joint-response-refit-20261104-01/input-near-fast/rc-inputs.npz')}


def encode(profile, side):
    seal = max(.01, profile.get('spring_friction_n', [0., 0.])[side])
    return np.r_[search.encode(profile, side, 'independent'), search.logarithm(seal, .01, 250.)]


def decode(vectors):
    profile = search.decode(vectors[0, :13], vectors[1, :13], 'independent')
    profile['spring_friction_n'] = (.01*25000.**vectors[:, 13]).tolist()
    return profile


def canonical_xml(inputs):
    # 旧参考可能残留惯量候选；物理参数始终从共享源取，只沿用实录的机身姿态。
    source = Path(inputs['report']['source_model'])
    root = ET.parse(source).getroot()
    root.find('compiler').set('meshdir', str((source.parent/'../meshes').resolve()))
    base = root.find("worldbody/body[@name='base_link']"); base.remove(base.find('freejoint'))
    recorded_base = ET.parse(inputs['path']/'fixed-base-model.xml').getroot().find("worldbody/body[@name='base_link']")
    base.set('pos', recorded_base.get('pos', '0 0 .8')); base.set('quat', recorded_base.get('quat', '1 0 0 0'))
    return root


def input_data(name):
    path, rcpath = map(Path, REFERENCES[name]); report = json.loads((path/'report.json').read_text())
    assert audit.digest(Path(report['source_model'])) == report['source_model_sha256']
    assert audit.digest(Path(report['session'])/'stream.jsonl') == report['stream_sha256']
    with np.load(path/'replay.npz') as data: saved = {k: data[k].copy() for k in data.files}
    with np.load(rcpath) as data: rc = {k: data[k].copy() for k in data.files}
    assert str(rc['stream_sha256']) == report['stream_sha256'] and report['dt_s'] == .001
    config = Path(json.loads((Path(report['session'])/'meta.json').read_text())['machine_config']['path'])
    assert audit.digest(config) == report['config_sha256']
    sign, zero = prior.replay.mirror.load_mapping(config)
    absolute = report['window_start_board_s']+saved['time_s'][:-1]
    geometry = saved['real_geometry']; ix = np.maximum(0, np.searchsorted(geometry[:, 0], absolute, side='right')-1)
    target = geometry[ix, 4:6]; flags = geometry[:, 6].astype(int)
    assert np.all((flags&64) == (flags[0]&64))
    dial = rc['dial_rate'][np.maximum(0, np.searchsorted(rc['time_s'], absolute, side='right')-1)]
    return dict(path=path, report=report, saved=saved, sign=sign, zero=zero, target=target, dial=dial,
                soft=bool(flags[0]&64), recovering=bool(int(geometry[ix[0], 6])&32),
                gains=prior.replay.gains(config), rc_path=rcpath)


def controller(qp, qv, mq, mv, inputs, k, recovering, blocked):
    sign, zero = inputs['sign'][:4], inputs['zero']; fwq = qp[:, mq]*sign+zero; fwv = qv[:, mv]*sign
    kp, kd = inputs['gains']; tau = np.zeros_like(fwq); lengths = []
    for side in range(2):
        section = slice(2*side, 2*side+2)
        length, jac, speed = prior.batch_leg_state(fwq[:, section], fwv[:, section])
        blocked |= ~np.isfinite(length)
        if not inputs['soft']: blocked |= (length<.14)|((~recovering)&(length>.33))
        force = kp[side]*(inputs['target'][k, side]-length)-kd[side]*speed
        force = np.where(recovering, np.minimum(force, 0.), force)
        if inputs['soft']:
            force = np.where(length>=.33, np.minimum(force, 0.), np.where(length<=.14, np.maximum(force, 0.), force))
        tau[:, section] = np.clip(jac*force[:, None], -40., 40.); lengths.append(length)
    if inputs['dial'][k] == 0 and inputs['target'][k].max() <= .33:
        recovering &= np.max(np.column_stack(lengths), axis=1)>.33
    tau[blocked] = 0.
    return tau*sign


def run(profiles, inputs, output, full=False, solver_profile='incremental', regularization='original', extra_load_factory=None):
    import mujoco_warp as mw
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    root = canonical_xml(inputs)
    assert regularization in ('original', 'friction', 'loop', 'both')
    loop_impedance = '.95 .99 .001 .5 2' if regularization in ('loop', 'both') else '.999 .9999 .001 .5 2'
    friction_impedance = [.9, .95, .001, .5, 2.] if regularization in ('friction', 'both') else [.999, .9999, .001, .5, 2.]
    for eq in root.find('equality'):
        eq.set('solref', '.0004 1'); eq.set('solimp', loop_impedance)
    for joint in root.findall('.//joint'):
        if joint.get('range') is not None and joint.get('limited') != 'false':
            joint.set('solreflimit', '.0004 1'); joint.set('solimplimit', '.999 .9999 .001 .5 2')
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='.0002', integrator='implicitfast', jacobian='dense',
                         iterations='100', tolerance='1e-10', impratio='10', solver='Newton',
                         ls_iterations='100', ls_tolerance='.0001')
    assert solver_profile in ('incremental', 'rebuild', 'cg', 'fixed', 'ordered', 'split')
    if solver_profile in ('rebuild', 'ordered', 'split'):
        # 本诊断无接触；elliptic仅切到Newton完整重建路径，不改变接触力。
        option.set('cone', 'elliptic')
    elif solver_profile == 'cg':
        option.set('solver', 'CG'); option.set('iterations', '200')
    elif solver_profile == 'fixed':
        option.set('tolerance', '0')
    for side in ('left', 'right'):
        node = root.find(f"actuator/motor[@name='{side}_gas_spring_motor']")
        node.set('ctrlrange', '0 600'); node.set('forcerange', '0 600')
    xml = ET.tostring(root, encoding='unicode'); (output/'model.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model); model.geom_contype[:] = model.geom_conaffinity[:] = 0
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST; model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    mq = [model.joint(n).qposadr[0] for n in prior.replay.mirror.MOTORS[:4]]
    mv = [model.joint(n).dofadr[0] for n in prior.replay.mirror.MOTORS[:4]]
    assert np.array_equal(model.dof_armature[mv], np.full(4, .015795))
    sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
    sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left', 'right')]
    aid = [model.actuator(n+'_motor').id for n in prior.replay.mirror.MOTORS]
    sid = [model.actuator(s+'_gas_spring_motor').id for s in ('left', 'right')]
    model.dof_frictionloss[mv+sv] = 1e-6
    model.dof_solref[mv+sv] = [.002, 1.]; model.dof_solimp[mv+sv] = friction_impedance
    data = mujoco.MjData(model); data.qpos[:] = inputs['saved']['qpos'][0, 0]; data.qvel[:] = inputs['saved']['qvel'][0, 0]
    ncase = len(profiles); worlds = 2*ncase
    table = []
    for profile in profiles:
        row = []
        for j in range(4):
            row += [profile[key][j] for key in ('friction_nm', 'motor_dynamic_positive_nm',
                                               'motor_dynamic_negative_nm', 'motor_velocity_scale_rad_s', 'damping_nms_rad')]
        for side in range(2):
            row += [profile[key][side] for key in ('spring_reference_force_n', 'spring_stiffness_n_m',
                                                  'spring_damping_ns_m', 'spring_friction_n')]
        row += profile.get('spring_friction_stiffness_n_m', [0., 0.])
        assert profile['armature_kg_m2'] == [.015795]*4 and profile['spring_force_limit_n'] == [600., 600.]
        table.append(row)
    table = np.tile(np.asarray(table, dtype=np.float32), (2, 1))
    friction = np.tile(model.dof_frictionloss, (worlds, 1)).astype(np.float32)
    friction[:, sv] = table[:, [23, 27]]
    damping = np.tile(model.dof_damping, (worlds, 1)).astype(np.float32); damping[:, mv] = table[:, [4, 9, 14, 19]]
    controls = inputs['saved']['submitted_model_torque'][:, 0].astype(np.float32)
    steps = len(controls); stride = 1 if full else 10
    q = np.empty((steps//stride+1, worlds, model.nq), dtype=np.float32); v = np.empty_like(q)
    recovering = np.full(ncase, inputs['recovering']); blocked = np.zeros(ncase, dtype=bool)
    with wp.ScopedDevice('cuda:0'):
        gm = mw.put_model(model, batch_sizes={'dof_frictionloss': worlds, 'dof_damping': worlds})
        gm.dof_frictionloss.assign(friction); gm.dof_damping.assign(damping)
        gd = mw.put_data(model, data, nworld=worlds, nconmax=128, njmax=256)
        params = wp.array(table); ids = [wp.array(x, dtype=wp.int32) for x in (mv, sq, sv, sid)]
        extra_load = extra_load_factory(model, gd, ncase) if extra_load_factory is not None else None
        if solver_profile == 'ordered':
            from scripts.agent.real2sim import ordered_leg_constraints
            ordering = ordered_leg_constraints.buffers(gm, gd)
        if full:
            assert ncase == 1
            rawq = wp.empty((steps*5, worlds, model.nq)); rawv = wp.empty_like(rawq)
            rawctrl = wp.empty((steps*5, worlds, model.nu)); rawfric = wp.empty_like(rawq)
            rawapplied = wp.empty_like(rawq)
            counter = wp.zeros(1, dtype=wp.int32)
        with wp.ScopedCapture() as capture:
            for substep in range(5):
                wp.launch(forces, dim=worlds, inputs=[gd.qpos, gd.qvel, params, *ids, gm.dof_frictionloss, gd.ctrl])
                if extra_load is not None: extra_load()
                if full:
                    wp.launch(archive, dim=worlds, inputs=[gd.qpos, gd.qvel, gd.ctrl, gm.dof_frictionloss, gd.qfrc_applied, rawapplied,
                        counter, rawq, rawv, rawctrl, rawfric, model.nv, model.nu])
                if solver_profile == 'ordered':
                    mw.step1(gm, gd); ordered_leg_constraints.apply(gd, ordering); mw.step2(gm, gd)
                elif solver_profile == 'split': mw.step1(gm, gd); mw.step2(gm, gd)
                else: mw.step(gm, gd)
                if full: wp.launch(advance, dim=1, inputs=[counter])
        assert not np.any(gd.time.numpy())
        qp = gd.qpos.numpy(); qv = gd.qvel.numpy(); q[0] = qp; v[0] = qv
        ctrl = np.zeros((worlds, model.nu), dtype=np.float32)
        for k in range(steps):
            ctrl[:, aid] = controls[k]
            ctrl[ncase:, aid[:4]] = controller(qp[ncase:], qv[ncase:], mq, mv, inputs, k, recovering, blocked)
            gd.ctrl.assign(ctrl); wp.capture_launch(capture.graph)
            qp = gd.qpos.numpy(); qv = gd.qvel.numpy()
            assert np.isfinite(qp).all() and np.isfinite(qv).all()
            if (k+1)%stride == 0: q[(k+1)//stride] = qp; v[(k+1)//stride] = qv
            if (k+1)%5000 == 0: print(json.dumps(dict(output=str(output), ms=k+1, elapsed_s=time.monotonic()-started)), flush=True)
        assert abs(gd.time.numpy()[0]-steps*.001) < .1
        if full:
            assert counter.numpy()[0] == steps*5
            np.savez_compressed(output/'physics.npz', time_s=np.arange(steps*5)*.0002,
                qpos=rawq.numpy(), qvel=rawv.numpy(), ctrl=rawctrl.numpy(), friction_capacity=rawfric.numpy(),
                applied_force=rawapplied.numpy(),
                final_qpos=qp, final_qvel=qv)
    times = np.arange(0, steps+1, stride)*.001
    measured = inputs['saved']['real_motor_state']; absolute = times+inputs['report']['window_start_board_s']
    real = np.column_stack([np.interp(absolute, measured[:, 0], measured[:, j+1]) for j in range(4)])
    error = np.degrees(q[:, :, mq]*inputs['sign'][:4]+inputs['zero']-real[:, None, :])
    rmse = np.sqrt(np.mean(error**2, axis=0)); peak = abs(error).max(0); terminal = abs(error[-1])
    limited = np.flatnonzero(model.jnt_limited); limq = q[:, :, model.jnt_qposadr[limited]]
    excess = np.maximum(model.jnt_range[limited, 0]-limq, limq-model.jnt_range[limited, 1]).clip(0.).max((0, 2))
    # 已知平面闭链的销点三角形只依赖lf02/rf02被动铰角，正面积为原装配支路。
    branch = np.ones(worlds, dtype=bool)
    for prefix, side in [('lf', 'left'), ('rf', 'right')]:
        j = model.joint(prefix+'02'); position = model.body(prefix+'02').pos
        pin = model.site(side+'_rear_pin1').pos; angle = (q[:, :, j.qposadr[0]]-model.qpos0[j.qposadr[0]])*j.axis[1]
        rx = np.cos(angle)*pin[0]+np.sin(angle)*pin[2]; rz = -np.sin(angle)*pin[0]+np.cos(angle)*pin[2]
        area = position[2]*rx-position[0]*rz
        branch &= np.min(area, axis=0)>0.
    scores = np.zeros((ncase, 2)); results = []
    for i, profile in enumerate(profiles):
        rows = []
        for b in (i, ncase+i):
            rows.append(dict(rmse_deg=rmse[b].tolist(), peak_deg=peak[b].tolist(), terminal_deg=terminal[b].tolist(),
                             limit_excess_deg=float(np.degrees(excess[b])), original_assembly_branch=bool(branch[b])))
        for side in (0, 1):
            section = slice(2*side, 2*side+2); bx = [i, ncase+i]
            scores[i, side] = np.sqrt(np.mean(rmse[bx, section]**2))+.05*peak[bx, section].max()+.1*terminal[bx, section].max()
            if blocked[i] or not np.all(branch[bx]) or np.max(excess[bx])>np.radians(.1): scores[i, side] = 1e6
        results.append(dict(parameters=profile, recorded_torque=rows[0], recorded_target=rows[1], feedback_blocked=bool(blocked[i])))
    summary = dict(reference=str(inputs['path']), input_sha256={str(inputs['path']/f):audit.digest(inputs['path']/f)
                   for f in ('report.json', 'replay.npz', 'fixed-base-model.xml')}, rc_path=str(inputs['rc_path']),
                   rc_sha256=audit.digest(inputs['rc_path']), physics_dt_s=.0002, motor_command_dt_s=.001,
                   model_sha256=audit.digest(output/'model.xml'), saved_dt_s=stride*.001,
                   new_gpu_environment_steps=steps*5*worlds, results=results, side_scores=scores.tolist(),
                   physics_backend='mujoco_warp_gpu', shared_model_modified=False, hardware_access=False,
                   solver_profile=solver_profile, solver_iterations=model.opt.iterations,
                   solver_tolerance=model.opt.tolerance, solver_type=int(model.opt.solver), cone_type=int(model.opt.cone),
                   numerical_regularization=regularization, joint_friction_solimp=friction_impedance,
                   loop_solimp=[float(x) for x in loop_impedance.split()],
                   future_measured_position_feedback=False, elapsed_s=time.monotonic()-started)
    summary['empirical_load_override'] = extra_load_factory is not None
    search.write_json(output/'report.json', summary)
    np.savez_compressed(output/'trials.npz', time_s=times, qpos=q, qvel=v, real_motor_q=real)
    print(json.dumps(dict(completed=str(output), best_side_scores=scores.min(0).tolist())), flush=True)
    return scores, summary


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--population', type=int, default=24); parser.add_argument('--generations', type=int, default=3)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    assert 4 <= args.population <= 32 and 0 <= args.generations <= 8
    wp.init(); rng = np.random.default_rng(20261104)
    protected = audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), [])
    inputs = {name:input_data(name) for name in REFERENCES}
    seeds = [Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/independent-joint-02/combined-verification/trajectory-00/best-parameters.json'),
             Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/independent-joint-02/generation-03/trajectory-01/best-parameters.json'),
             Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/rated-independent-wide/best-parameters.json')]
    vectors = rng.uniform(0., 1., (args.population, 2, 14))
    for i, path in enumerate(seeds):
        profile = json.loads(path.read_text()); vectors[i] = [encode(profile, side) for side in (0, 1)]
    for i in range(len(seeds), args.population//2):
        vectors[i] = np.clip(vectors[i%len(seeds)]+rng.normal(0., .1, (2, 14)), 0., 1.)
    total = 0; history = []
    def evaluate(values, name):
        nonlocal total
        profiles = [decode(v) for v in values]; all_scores = []
        search.write_json(args.output/(name+'-parameters.json'), profiles)
        for session in ('original', 'wide'):
            scores, result = run(profiles, inputs[session], args.output/name/session, solver_profile='ordered')
            all_scores.append(scores); total += result['new_gpu_environment_steps']
        return np.mean(all_scores, axis=0)
    scores = evaluate(vectors, 'generation-00')
    for generation in range(args.generations+1):
        if generation:
            proposals = vectors.copy()
            for i in range(args.population):
                for side in (0, 1):
                    a, b, c = rng.choice(np.delete(np.arange(args.population), i), 3, replace=False)
                    mutant = np.clip(vectors[a, side]+.65*(vectors[b, side]-vectors[c, side]), 0., 1.)
                    mask = rng.uniform(size=14)<.7; mask[rng.integers(14)] = True
                    proposals[i, side] = np.where(mask, mutant, vectors[i, side])
            proposed_scores = evaluate(proposals, f'generation-{generation:02d}')
            improve = proposed_scores.mean(1)<scores.mean(1)
            vectors[improve] = proposals[improve]; scores[improve] = proposed_scores[improve]
        best = int(np.argmin(scores.mean(1))); selected = vectors[best].copy()
        entry = dict(generation=generation, best_side_scores=scores.min(0).tolist(), gpu_environment_steps=total)
        history.append(entry); search.write_json(args.output/'progress.json', history); print(json.dumps(entry), flush=True)
    selected = decode(selected); search.write_json(args.output/'selected-parameters.json', selected)
    final = {}
    for name in REFERENCES:
        _, result = run([selected], inputs[name], args.output/('final-'+name), full=True, solver_profile='ordered')
        final[name] = result['results'][0]; total += result['new_gpu_environment_steps']
    assert audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), []) == protected
    search.write_json(args.output/'report.json', dict(history=history, selected_parameters=selected, final=final,
        source_seeds={str(p):audit.digest(p) for p in seeds}, new_gpu_environment_steps=total,
        protected_source_checks=protected, accepted=False, physical_parameter_identification_unique=False,
        development_sessions=['original', 'wide'], regression_sessions=['slow', 'fast'],
        no_new_hardware_data=True, shared_model_modified=False, policy_training_updates=0))
    shutil.copy2(__file__, args.output/Path(__file__).name)


if __name__ == '__main__': main()
