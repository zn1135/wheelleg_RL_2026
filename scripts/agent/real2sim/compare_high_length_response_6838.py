"""Match recorded high stance, then compare short GPU PD-target responses; no learning."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.signal import coherence, csd, welch
from scipy.spatial.transform import Rotation

from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen
from scripts.agent.fudan_ppo_walk import inspect_slope5to20_mu08 as study

ROOT = frozen.ROOT
REAL = ROOT / 'artifacts/real2sim/20261117-high-length6838-01'
RAW = REAL / 'raw/高腿长抖腿'
ORDER = ('lf', 'lr', 'lw', 'rf', 'rr', 'rw')
LEGS = [0, 1, 3, 4]
STARTS = (376.593242, 377.593242, 378.593242, 381.273242, 382.273242, 383.273242)
SHA = '8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'


def inputs():
    pd_path = RAW / '6838高腿长对照_162432/full-pd.csv'
    phy_path = RAW / '6838高腿长完整录制/complete-cycle-162432.csv'
    pnames = ('h7_us', 'quat_w', 'quat_x', 'quat_y', 'quat_z', 'gyro_x_rad_s',
              'gyro_y_rad_s', 'gyro_z_rad_s', 'left_length_m', 'right_length_m')
    dnames = ('h7_us', *[prefix + n for prefix in ('target_', 'current_', 'velocity_rad_s_', 'pd_model_nm_') for n in ORDER])
    with pd_path.open() as f:
        d = np.asarray([[float(r[n]) for n in dnames] for r in csv.DictReader(f)
                        if r['rl_engaged'] == '1' and r['pd_valid'] == '1' and r['warmup'] == '0'])
    with phy_path.open() as f:
        p = np.asarray([[float(r[n]) for n in pnames] for r in csv.DictReader(f) if r['frame'] == 'physical'])
    p = p[np.unique(p[:, 0], return_index=True)[1]]
    assert np.all(np.diff(p[:, 0]) > 0) and np.all(np.diff(d[:, 0]) > 0)
    return p, d, {str(x.relative_to(ROOT)): frozen.sha(x) for x in (pd_path, phy_path)}


def lengths(model, data, poses):
    import mujoco
    hips = [model.joint(n).id for n in ('lf0', 'rf0')]
    wheels = [model.body(n).id for n in ('lfwheel', 'rfwheel')]
    base = model.body('base_link').id
    result = []
    for q in poses:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        vectors = [data.xmat[base].reshape(3, 3).T @ (data.xpos[w] - data.xanchor[h]) for h, w in zip(hips, wheels)]
        result.append([np.linalg.norm(v[[0, 2]]) for v in vectors])
    return np.asarray(result)


def run(out, support_probe=False):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'source.py')
    p, d, hashes = inputs()
    frozen.RUN = study.RUN
    assert frozen.sha(study.RUN / 'model_final.pt') == SHA
    for name, digest in json.loads((study.RUN / 'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert frozen.sha(study.RUN / 'source/ct_mjlab' / name.removeprefix('mjlab_training/src/ct_mjlab/')) == digest
        elif name.startswith('resources/'):
            assert frozen.sha(ROOT / name) == digest
    assert not any(n == 'ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    import torch
    from mjlab.managers.event_manager import RecomputeLevel
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.disturbances import from_record
    from ct_mjlab.fudan_ppo_walk.measurement import FIELDS
    from scripts.agent.real2sim.replay_usb_pd_targets import PoseSolver

    solver = PoseSolver()
    with np.load(ROOT / 'artifacts/fudan_ppo_walk/20261117-gyro-x6838-01/raw/traces/height_hold__50001.npz') as z:
        nominal = z['qpos'][np.argmin(abs(z['qpos_time'] - 5.))].copy()
    # 两种姿态共用实录速度；这样只改变起始站姿，不额外改变起始速度。
    qaddr = np.asarray([solver.qaddr[n] for n in ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')])
    vaddr = qaddr - 1
    poses, velocities, samples, targets, requests, refs = [], [], [], [], [], []
    arms = (('nominal_replay', False, False, 0), ('nominal_hold', False, True, 0),
            ('matched_replay', True, False, 0), ('matched_hold', True, True, 0),
            ('matched_delay17_replay', True, False, 17), ('matched_delay17_hold', True, True, 17))
    support_bias = np.zeros(2)
    if support_probe:
        static_balance(out, solver, p, d)
        calibration = json.loads((out / 'static-force-balance.json').read_text())['rows'][1]
        support_bias = np.asarray(calibration['fitted_spring_force_N']) - calibration['default_elastic_force_N']
        arms = (('matched_replay', True, False, 0), ('matched_hold', True, True, 0),
                ('matched_support_replay', True, False, 0), ('matched_support_hold', True, True, 0))
    for start in STARTS:
        i = int(np.argmin(abs(d[:, 0] / 1e6 - start)))
        block = d[i:i + 121]
        assert len(block) == 121 and np.max(abs(np.diff(block[:, 0]) - 5000)) < 50
        stamp = block[0, 0] / 1e6
        pi = int(np.argmin(abs(p[:, 0] - block[0, 0])))
        assert abs(p[pi, 0] - block[0, 0]) < 30
        quat = p[pi, 1:5].copy(); quat /= np.linalg.norm(quat)
        reference = np.stack([np.interp(block[:, 0], p[:, 0], p[:, j]) for j in (5, 6, 7, 8, 9)], -1)
        # p: 时间、四元数、gyroXYZ、左右实测腿长；PD是目标/角度/速度/请求各6路。
        refs.append(dict(start_h7_s=stamp, initial_physical_offset_us=float(p[pi, 0] - block[0, 0]),
                         recorded_lengths_m=p[pi, 8:10].tolist()))
        for arm, matched, hold, delay in arms:
            motors = block[0, 7:13].copy() if matched else nominal[qaddr].copy()
            q, error = solver.solve(motors)
            qnext, next_error = solver.solve(motors + block[0, 13:19] * .0001)
            v = np.zeros(solver.model.nv); v[6:] = (qnext[7:] - q[7:]) / .0001
            v[vaddr] = block[0, 13:19]; v[3:6] = p[pi, 5:8]
            q = solver.ground(q, quat if matched else nominal[3:7])
            q[2] -= .002  # 消除原复位2mm悬空；从模型刚好接地开始，不固定机身。
            assert max(error, next_error) < 1e-8
            poses.append(q); velocities.append(v)
            samples.append(dict(id=f'{arm}__{stamp:.6f}', arm=arm, matched=matched, hold=hold,
                                delay_ms=delay, start_h7_s=stamp, closure_error_m=error,
                                spring_bias_N=(support_bias.tolist() if '_support_' in arm else [0., 0.])))
            targets.append(np.repeat(block[:1, 1:7], 121, 0) if hold else block[:, 1:7])
            requests.append(d[max(0, i - 5):i + 1, 19:25])
    q0, v0, targets = np.asarray(poses), np.asarray(velocities), np.asarray(targets)
    np.savez_compressed(out / 'initial.npz', qpos=q0, qvel=v0, targets=targets, real_pd=d, real_physical=p)
    frozen.write(out / 'plan.json', dict(samples=samples, input_sha256=hashes, checkpoint_sha256=SHA,
        physical_references=refs, physics_dt_s=.001, pd_dt_s=.005, duration_s=.6, worlds=len(samples),
        support_probe=support_probe, support_calibration_window_h7_s=[377.593242, 378.593242] if support_probe else None,
        assumptions=['Root translation velocity unmeasured: zero for both postures',
                     'Root height/passive joints inferred from shared geometry, not measured',
                     'Both postures share measured motor/body angular velocities; wheel phase zero',
                     'Zero ground clearance only places initial lower wheel on ground; no support constraint',
                     'Real targets contain11Hz: forced response, never autonomous reproduction',
                     '17ms output delay is a hypothesis, not a calibrated motor delay'],
        validity='Initial geometry must agree; compare only before joint error>.05rad, leg error>5mm or tilt error>3deg',
        optimizer_updates=0, policy_forwards=0, wall_limit_s=240))
    torch.set_num_threads(1)
    meta = torch.load(study.RUN / 'model_final.pt', map_location='cpu', weights_only=False)
    cfg = FudanWalkCfg()
    for section, fields in json.loads((study.RUN / 'effective-training-config.json').read_text()).items():
        for key, value in fields.items():
            setattr(getattr(cfg, section), key, value)
    cfg.terrain.mode = 'plane'; cfg.observations.critic_height_scan = False
    conditions = replace(from_record(meta['disturbances']), action_delay=False, sensor_delay=False,
                         pushes=False, external_force_range=(0., 0.), external_torque_range=(0., 0.))
    old_cfg = physical.make_env_cfg
    def make_cfg(*args, **kwargs):
        c = old_cfg(*args, **kwargs); c.sim.nconmax, c.sim.njmax = 64, 256
        return c
    with patch.object(physical, 'make_env_cfg', make_cfg):
        env = physical.FudanEnv(len(samples), seed=50001, randomized=False, cfg=cfg,
                               disturbances=conditions, reset_contract=meta['reset_contract'])
    sim, act = env.env.sim, env.action
    sim.expand_model_fields(FIELDS)
    sim.model.geom_friction[..., 0] = .8
    sim.recompute_constants(RecomputeLevel.set_const)
    env.env.scene.env_origins.zero_()
    assert np.array_equal(act._motor_q_adr.cpu().numpy(), qaddr)
    assert np.array_equal(act._motor_v_adr.cpu().numpy(), vaddr)
    assert cfg.sim.dt == .001 and act.cfg.pd_decimation == 5
    assert act.cfg.leg_stiffness == 20 and act.cfg.leg_damping == 1 and act.cfg.wheel_stiffness == .1
    assert act.kp_scale.eq(1).all() and act.kd_scale.eq(1).all() and act.motor_strength.eq(1).all()
    act._spring_force_bias[:] = torch.tensor([s['spring_bias_N'] for s in samples], device=env.device)
    sim.data.qpos[:] = torch.tensor(q0, dtype=torch.float32, device=env.device)
    sim.data.qvel[:] = torch.tensor(v0, dtype=torch.float32, device=env.device)
    sim.data.ctrl.zero_(); sim.data.qacc_warmstart.zero_()
    sim.forward(); env.env.scene.update(dt=0.); sim.sense()
    initial_error = float(abs(sim.data.qpos.cpu().numpy() - q0).max())
    motor_ids = [sim.mj_model.actuator('robot/' + n + '_motor').id for n in ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')]
    ring = torch.zeros(18, len(samples), 6, device=env.device)
    for j, sample in enumerate(samples):
        for k in range(18):
            age = (18 - k) % 18 or 18
            i = np.searchsorted(d[:, 0] / 1e6, sample['start_h7_s'] - age * .001, side='right') - 1
            ring[k, j] = torch.tensor(d[max(i, 0), 19:25], device=env.device)
    delays = torch.tensor([s['delay_ms'] for s in samples], device=env.device)
    worlds = torch.arange(len(samples), device=env.device)
    rows_q, rows_v, rows_tau, rows_delivered, rows_spring = [sim.data.qpos.cpu().numpy().copy()], [sim.data.qvel.cpu().numpy().copy()], [], [], []
    max_pd_error = 0.; max_initial_request_error = 0.; head = 0; started = time.time()
    try:
        with torch.no_grad(), patch('torch.optim.Adam.step', side_effect=AssertionError('No learning')):
            for step in range(600):
                assert time.time() - started < 240
                target = torch.tensor(targets[:, step // 5], dtype=torch.float32, device=env.device)
                raw = target.clone(); raw[:, LEGS] = (raw[:, LEGS] - act._nominal_pos[:, LEGS]) / .5
                raw[:, [2, 5]] /= 10.
                act.process_actions(raw); update = act._pd_tick == 0; act.apply_actions()
                if update:
                    q = sim.data.qpos[:, act._motor_q_adr]; v = sim.data.qvel[:, act._motor_v_adr]
                    expected = 20 * (target - q) - v
                    expected[:, [2, 5]] = .1 * (target[:, [2, 5]] - v[:, [2, 5]])
                    expected = expected.clamp(-act._effort_limits, act._effort_limits)
                    max_pd_error = max(max_pd_error, float(abs(expected - act.torques).max()))
                    if step == 0:
                        for j, s in enumerate(samples):
                            if s['matched']:
                                i = np.argmin(abs(d[:, 0] / 1e6 - s['start_h7_s']))
                                max_initial_request_error = max(max_initial_request_error,
                                    float(abs(act.torques[j].cpu().numpy() - d[i, 19:25]).max()))
                env.env.scene.write_data_to_sim()
                rows_spring.append(sim.data.ctrl[:, act._spring_ids].cpu().numpy().copy())
                requested = sim.data.ctrl[:, motor_ids].clone(); ring[head] = requested
                delivered = ring[(head - delays) % 18, worlds].clone()
                delivered[:, [2, 5]] = requested[:, [2, 5]]  # 17ms只假设腿输出滞后，轮保持原模型。
                sim.data.ctrl[:, motor_ids] = delivered; head = (head + 1) % 18
                rows_tau.append(requested.cpu().numpy()); rows_delivered.append(delivered.cpu().numpy())
                sim.step(); env.env.scene.update(dt=.001)
                rows_q.append(sim.data.qpos.cpu().numpy().copy()); rows_v.append(sim.data.qvel.cpu().numpy().copy())
                if step % 100 == 0:
                    assert not sim.wp_data.overflow.numpy().any()
                    assert float(sim.data.xfrc_applied.abs().max()) == 0.
                    print(json.dumps(dict(step=step, total=600, worlds=len(samples))), flush=True)
    finally:
        env.close()
    q, v, tau, delivered = map(np.asarray, (rows_q, rows_v, rows_tau, rows_delivered))
    assert np.isfinite(q).all() and np.isfinite(v).all()
    assert max_pd_error < 2e-5 and max_initial_request_error < 2e-5
    error = 0.
    for j, s in enumerate(samples):
        lag = s['delay_ms']
        error = max(error, float(abs(delivered[lag:, j][:, LEGS] - tau[:len(tau) - lag, j][:, LEGS]).max()))
    assert error == 0.
    np.savez_compressed(out / 'gpu.npz', time=np.arange(601) * .001, qpos=q, qvel=v,
                        requested=tau, delivered=delivered, motor_q=qaddr, motor_v=vaddr,
                        spring_force=np.asarray(rows_spring))
    frozen.write(out / 'runtime-audit.json', dict(passed=True, backend='mujoco_warp_gpu', batched_physics_steps=600,
        worlds=len(samples), PD_updates_per_world=120, initial_qpos_float_error=initial_error,
        PD_formula_max_error_Nm=max_pd_error, initial_matched_request_error_Nm=max_initial_request_error,
        delivered_leg_delay_error_Nm=error, external_wrench=0, weights_unchanged=frozen.sha(study.RUN / 'model_final.pt') == SHA,
        optimizer_updates=0, policy_forwards=0, wall_s=time.time() - started, frozen_modules=frozen.LOADED))
    report(out, solver)


def transfer(u, y):
    f, cross = csd(u, y, fs=200., nperseg=256)
    _, power = welch(u, fs=200., nperseg=256)
    _, coh = coherence(u, y, fs=200., nperseg=256)
    i = np.argmin(abs(f - 11.))
    h = cross[i] / power[i]
    return dict(frequency_Hz=float(f[i]), gain=float(abs(h)), phase_deg=float(np.rad2deg(np.angle(h))), coherence=float(coh[i]))


def static_balance(out, solver, p, d):
    """Offline force balance only; no CPU integration and no parameter adoption."""
    import mujoco
    from scipy.linalg import null_space
    from scipy.optimize import lsq_linear
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.reset_geometry import mesh_vertices
    model, data = solver.model, solver.data
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    motor_names = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
    motor_v = [model.joint(n).dofadr[0] for n in motor_names]
    spring_v = [model.joint(n + '_spring2_joint').dofadr[0] for n in ('left', 'right')]
    result = []
    for start in STARTS[:3]:
        take = (d[:, 0] / 1e6 >= start) & (d[:, 0] / 1e6 < start + 1.)
        mean = d[take].mean(0)
        q, error = solver.solve(mean[7:13])
        ptake = (p[:, 0] / 1e6 >= start) & (p[:, 0] / 1e6 < start + 1.)
        quat = Rotation.from_quat(p[ptake, 1:5][:, [1, 2, 3, 0]]).mean().as_quat()[[3, 0, 1, 2]]
        q = solver.ground(q, quat); q[2] -= .002
        data.qpos[:] = q; data.qvel[:] = 0; data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        eq = data.efc_J.reshape(data.nefc, model.nv)[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        basis = null_space(eq, rcond=1e-8)
        spring = np.eye(model.nv)[:, spring_v]
        contact = []
        for name in ('lfwheel', 'rfwheel'):
            body = model.body(name).id
            points = np.concatenate([mesh_vertices(model, data, g) for g in solver.collision if model.geom_bodyid[g] == body])
            bottom = points[points[:, 2] < points[:, 2].min() + 1e-5].mean(0)
            bottom[2] = 0.
            jac = np.zeros((3, model.nv)); rot = np.zeros_like(jac)
            mujoco.mj_jac(model, data, jac, rot, bottom, body)
            contact.append(jac.T)
        motor_force = np.zeros(model.nv); motor_force[motor_v] = mean[19:25]
        needed = data.qfrc_bias - data.qfrc_passive - motor_force
        matrix = basis.T @ np.column_stack((spring, *contact))
        rhs = basis.T @ needed
        lower = [0., 0., -np.inf, -np.inf, 0., -np.inf, -np.inf, 0.]
        upper = [150., 150., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        fit = lsq_linear(matrix, rhs, bounds=(lower, upper), tol=1e-12)
        compression = .01 - q[np.asarray(spring_v) + 1]
        default = np.clip(compression * (150. / .09), 0., 150.)
        baseline = lsq_linear(matrix[:, 2:], rhs - matrix[:, :2] @ default,
                             bounds=(lower[2:], upper[2:]), tol=1e-12)
        result.append(dict(start_h7_s=start, rows=int(take.sum()), closure_error_m=error,
            constraint_nullity=basis.shape[1], fit_rank=int(np.linalg.matrix_rank(matrix)),
            compression_mm=(compression * 1000).tolist(), default_elastic_force_N=default.tolist(),
            fitted_spring_force_N=fit.x[:2].tolist(), fitted_contact_forces_N=fit.x[2:].reshape(2, 3).tolist(),
            projected_residual_default=float(np.linalg.norm(baseline.fun)),
            projected_residual_fitted=float(np.linalg.norm(fit.fun)),
            note='Mixed generalized-force coordinates: residual norm is diagnostic, not a calibrated error unit'))
    frozen.write(out / 'static-force-balance.json', dict(rows=result, integration_steps=0,
        assumptions=['Static acceleration and velocity zero; mean recorded motor requests approximate physical motor torque',
                     'Two wheel contacts, bottom points inferred from model; tangential force unconstrained',
                     'Closed-chain reactions projected out; static Coulomb friction not identified',
                     'Same baseline mass/geometry; spring fitting cannot uniquely separate missing loads or model errors'],
        scope='Idealized offline force balance, not measured gas-spring force, not a new accepted physical model'))


def report(out, solver=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if solver is None:
        frozen.RUN = study.RUN; sys.meta_path.insert(0, frozen.FrozenTrainingSource())
        from scripts.agent.real2sim.replay_usb_pd_targets import PoseSolver
        solver = PoseSolver()
    plan = json.loads((out / 'plan.json').read_text())
    with np.load(out / 'initial.npz') as z:
        d, p, targets = z['real_pd'].copy(), z['real_physical'].copy(), z['targets'].copy()
    with np.load(out / 'gpu.npz') as z:
        t, q, v, qaddr = z['time'].copy(), z['qpos'].copy(), z['qvel'].copy(), z['motor_q'].copy()
        spring_output = z['spring_force'].copy() if 'spring_force' in z.files else None
    static_balance(out, solver, p, d)
    leg = lengths(solver.model, solver.data, q.reshape(-1, q.shape[-1])).reshape(len(t), len(plan['samples']), 2)
    rows = []; paired = []; spectra = {}
    quat_real = np.stack([np.interp(d[:, 0], p[:, 0], p[:, j]) for j in range(1, 5)], -1)
    real_angles = Rotation.from_quat(quat_real[:, [1, 2, 3, 0]]).as_euler('xyz')[:, :2]
    physical = np.stack([np.interp(d[:, 0], p[:, 0], p[:, j]) for j in range(5, 10)], -1)
    for label, lo, hi in (('stable', 376.593242, 380.188240), ('jitter', 380.773242, 385.393247)):
        take = (d[:, 0] / 1e6 >= lo) & (d[:, 0] / 1e6 <= hi)
        x = d[take]; gyro = physical[take, 0]
        spectra[label] = {ORDER[k]: transfer(x[:, 1 + k], x[:, 7 + k]) for k in LEGS}
        spectra[label]['differential_target_to_gyroX'] = transfer(x[:, 1:7][:, LEGS] @ (np.array([1, -1, 1, -1]) / 4), gyro)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    chosen = (STARTS[0], STARTS[4])
    for j, sample in enumerate(plan['samples']):
        stamp = sample['start_h7_s'] + t
        realq = np.stack([np.interp(stamp * 1e6, d[:, 0], d[:, 7 + k]) for k in LEGS], -1)
        realg = np.interp(stamp * 1e6, p[:, 0], p[:, 5])
        reall = np.stack([np.interp(stamp * 1e6, p[:, 0], p[:, k]) for k in (8, 9)], -1)
        reala = np.stack([np.interp(stamp * 1e6, d[:, 0], real_angles[:, k]) for k in (0, 1)], -1)
        angles = Rotation.from_quat(q[:, j, 3:7][:, [1, 2, 3, 0]]).as_euler('xyz')[:, :2]
        qerr = abs(q[:, j, qaddr[LEGS]] - realq).max(-1)
        lerr = abs(leg[:, j] - reall).max(-1)
        aerr = abs(angles - reala).max(-1)
        invalid = (qerr > .05) | (lerr > .005) | (aerr > np.deg2rad(3))
        end = int(np.flatnonzero(invalid)[0]) if invalid.any() else len(t)
        row = dict(**sample, initial_leg_error_mm=float(lerr[0] * 1000), initial_joint_error_rad=float(qerr[0]),
            valid_prefix_s=float(t[max(0, end - 1)]), valid_samples=end,
            first_invalid=dict(joint_rad=float(qerr[min(end, len(t) - 1)]), leg_mm=float(lerr[min(end, len(t) - 1)] * 1000),
                               tilt_deg=float(np.rad2deg(aerr[min(end, len(t) - 1)]))),
            whole_window_is_calibration=False)
        early = t <= .2
        row['descriptive_first_200ms'] = dict(
            mean_leg_RMSE_mm=float(np.sqrt(np.mean((leg[early, j].mean(-1) - reall[early].mean(-1)) ** 2)) * 1000),
            individual_leg_RMSE_mm=(np.sqrt(np.mean((leg[early, j] - reall[early]) ** 2, axis=0)) * 1000).tolist(),
            gyroX_RMS=float(np.sqrt(np.mean(v[early, j, 3] ** 2))), real_gyroX_RMS=float(np.sqrt(np.mean(realg[early] ** 2))),
            scope='Includes posture drift after valid prefix; descriptive response only, not calibrated frequency response')
        if end > 1:
            row.update(joint_RMSE_deg=np.rad2deg(np.sqrt(np.mean((q[:end, j, qaddr[LEGS]] - realq[:end]) ** 2, axis=0))).tolist(),
                       gyroX_RMSE_rad_s=float(np.sqrt(np.mean((v[:end, j, 3] - realg[:end]) ** 2))))
        rows.append(row)
        for col, start in enumerate(chosen):
            if abs(sample['start_h7_s'] - start) > 1e-4 or sample['hold']:
                continue
            axes[0, col].plot(t, q[:, j, qaddr[1]], label=sample['arm'])
            axes[1, col].plot(t, leg[:, j].mean(-1) * 100, label=sample['arm'])
            axes[2, col].plot(t, v[:, j, 3], label=sample['arm'])
            if sample['arm'] == 'matched_replay':
                axes[0, col].plot(t, realq[:, 1], 'k--', label='Real LR joint')
                axes[0, col].plot(np.arange(121) * .005, targets[j, :, 1], color='.7', label='Real LR target')
                axes[1, col].plot(t, reall.mean(-1) * 100, 'k--', label='Real')
                axes[2, col].plot(t, realg, 'k--', label='Real')
                for ax in axes[:, col]:ax.axvline(row['valid_prefix_s'], color='red', linestyle=':', label='Matched validity ends')
    for j, sample in enumerate(plan['samples']):
        if sample['hold']:
            continue
        hold = next(k for k, s in enumerate(plan['samples']) if s['id'] == sample['id'].replace('_replay__', '_hold__'))
        assert np.array_equal(q[0, j], q[0, hold]) and np.array_equal(v[0, j], v[0, hold])
        n = min(rows[j]['valid_samples'], rows[hold]['valid_samples'])
        paired.append(dict(id=sample['id'], paired_initial_equal=True, valid_prefix_s=float(t[max(0, n - 1)]),
                           gyroX_delta_peak=float(abs(v[:max(1, n), j, 3] - v[:max(1, n), hold, 3]).max())))
    support_pairs = 0
    for j, sample in enumerate(plan['samples']):
        if '_support_' in sample['arm']:
            baseline = next(k for k, s in enumerate(plan['samples']) if s['id'] == sample['id'].replace('matched_support_', 'matched_'))
            assert np.array_equal(q[0, j], q[0, baseline]) and np.array_equal(v[0, j], v[0, baseline])
            support_pairs += 1
    axes[0, 0].set_ylabel('LR joint / target (rad)'); axes[1, 0].set_ylabel('Mean leg length (cm)')
    axes[2, 0].set_ylabel('gyroX (rad/s)')
    for i, ax in enumerate(axes.ravel()):
        ax.grid(alpha=.25)
        if i < 2:ax.set_title('Stable high stance' if i == 0 else 'Jitter high stance')
    for ax in axes[2]:ax.set_xlabel('Seconds from recorded initial state')
    axes[0, 0].legend(fontsize=7); fig.suptitle('Same recorded targets: initial posture matched, later drift is NOT valid calibration')
    fig.tight_layout(); fig.savefig(out / 'matched-response.png', dpi=160); plt.close(fig)
    frozen.write(out / 'comparison.json', dict(rows=rows, paired=paired, real_closed_loop_transfers=spectra,
        scope='Short matched-state forced response; empirical real closed-loop spectra are not independent plant identification',
        autonomous_11Hz_reproduced=False, optimizer_updates=0))
    if spring_output is not None:
        from ct_mjlab.envs.chuanliantui.gas_spring import SPRING_DAMPING
        sq = [solver.qaddr[n + '_spring2_joint'] for n in ('left', 'right')]
        bias = np.asarray([s['spring_bias_N'] for s in plan['samples']])
        expected = np.clip(np.clip((.01 - q[:-1, :, sq]) * (150. / .09), 0., 150.) + bias
                           - SPRING_DAMPING * v[:-1, :, np.asarray(sq) - 1], 0., 150.)
        force_error = float(abs(expected - spring_output).max())
        assert force_error < 1e-4
        frozen.write(out / 'spring-output-audit.json', dict(passed=True, formula_max_error_N=force_error,
            actual_initial_force_N=spring_output[0].tolist(), cap_fraction=float(np.mean(spring_output >= 150. - 1e-5)),
            copied_initial_arrays_verified=all(x['paired_initial_equal'] for x in paired),
            support_baseline_qpos_qvel_exact_pairs=support_pairs,
            optimizer_updates=0, physical_model_adopted=False))
    shutil.copy2(__file__, out / 'report-source.py')
    print(json.dumps([dict(id=r['id'], valid_prefix_s=r['valid_prefix_s'], initial_leg_error_mm=r['initial_leg_error_mm'])
                      for r in rows if r['arm'] == 'matched_replay']), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('run', 'report')); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support-probe', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':run(args.output.resolve(), args.support_probe)
    else:report(args.output.resolve())


if __name__ == '__main__':
    main()
