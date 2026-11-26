"""冻结台阶4200，比较高站姿/登台的观测噪声响应；不执行学习。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'logs/mjlab/fudan_ppo_step/20261117_fixed200_turn_v8_4096_resume3200_to4200_01'
MODEL_SHA = 'becd82418d671c9ea7113b84102fb3fa387055d57a6fe2edfec835dd853ad1f8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def setup_source():
    from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen
    frozen.RUN = RUN
    checks = {}
    for name, digest in json.loads((RUN / 'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            saved = RUN / 'source/ct_mjlab' / name.removeprefix('mjlab_training/src/ct_mjlab/')
            checks[name] = sha(saved) == digest
        elif name.startswith('resources/'):
            checks[name] = sha(ROOT / name) == digest
    assert checks and all(checks.values())
    sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    return frozen, checks


def design(step_scales):
    walk_scales = [.075] * 6 + [0.] * 3 + [.03] * 4 + [.1125] * 6 + [0.] * 6
    profiles = dict(clean=[0.] * 25, step=step_scales, walk6838=walk_scales)
    cases = [dict(key=f'hold_h{h:.2f}', height_command=h, speed=0., height=0.) for h in (.22, .30, .35)]
    cases += [dict(key=f'up200_v{v:.2f}', height_command=.35, speed=v, height=.2) for v in (.3, 1., 2.5)]
    groups = [dict(profile='clean', seed=71001)]
    groups += [dict(profile=p, seed=s) for s in (71001, 71002, 71003) for p in ('step', 'walk6838')]
    samples = [dict(c, **g, id=f'{g["profile"]}_s{g["seed"]}__{c["key"]}') for g in groups for c in cases]
    return profiles, cases, groups, samples


def run(out, deadline):
    import torch
    import mujoco
    frozen, source_checks = setup_source()
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg, recipe_record
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.runner import validate_source
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from mjlab.managers.event_manager import RecomputeLevel
    out.mkdir(parents=True, exist_ok=False)
    (out / 'traces').mkdir()
    shutil.copy2(__file__, out / 'diagnostic-source.py')
    assert sha(RUN / 'model_final.pt') == MODEL_SHA
    torch.set_num_threads(1)
    cfg = FudanStepCfg()
    saved = torch.load(RUN / 'model_final.pt', map_location='cpu', weights_only=False)
    assert saved['training_recipe'] == recipe_record(cfg) and saved['iterations'] == 4200 and not saved['partial_update']
    cfg.step.curriculum = False
    profiles, cases, groups, samples = design(saved['training_recipe']['normalization']['noise_scales'])
    n, total = len(cases), len(samples)
    protocol = dict(task='fudan_ppo_step', backend='mujoco_warp_gpu', samples=samples, groups=groups,
        noise_scales=profiles, duration_s=20., policy_dt=.01, pd_dt=.005, physics_dt=.001,
        checkpoint_sha256=MODEL_SHA, checkpoint=str(RUN / 'model_final.pt'), recipe=recipe_record(cfg),
        noise='Independent uniform each10ms after sensor delay; same draws across amplitudes; current/five-frame history matched',
        measurement=dict(hold_window_s=[15., 20.], parked_window='First stop command +[0.5,2.5)s, no substitution if missing',
            jitter='8-14Hz RMS after linear detrend; four active leg targets and actual gyroX; collision transients reported separately'),
        nominal_physics=True, friction=.8, action_delay_ms=15., sensor_delay_ms=15., pushes=False,
        pd_feedback_noise=False, optimizer_updates=0, source_checks=source_checks,
        scope='42 nominal GPU trajectories; tall standing and straight200mm ascent only; not full stair/hardware acceptance')
    write(out / 'plan.json', protocol)
    env = None
    started = time.time()
    try:
        with patch('torch.optim.Adam.step', side_effect=AssertionError('Diagnostic cannot train')), torch.inference_mode():
            policy, meta = load_policy(RUN / 'model_final.pt', 'cuda:0')
            policy.eval().requires_grad_(False)
            before = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            validate_source(meta, cfg)
            env = StepEnv(total, seed=35115, randomized=False, cfg=cfg)
            sim, action = env.env.sim, env.action
            ids = torch.arange(total, device=env.device)
            env.terrain.terrain_types[:] = torch.tensor([int(c['height'] > 0) for c in samples], device=env.device)
            env.terrain.terrain_levels[:] = torch.tensor([int(c['height'] > 0) for c in samples], device=env.device)
            env.env._step_fixed_yaw = torch.zeros(total, device=env.device)
            env.fixed_commands = torch.tensor([[c['speed'], 0., c['height_command']] for c in samples], device=env.device)
            sim.model.geom_friction[:, :, 0] = .8
            env.env._fudan_parameters['friction'].fill_(.8)
            sim.recompute_constants(RecomputeLevel.set_const)
            action.disturbances.action_steps.fill_(15)
            action.disturbances.sensor_steps.fill_(15)
            obs, _, _ = env.reset()
            env.contacts.reset(ids)
            # 同一工况各噪声分支复制积分、PD、延迟、历史及接触缓存的完整初态。
            source = torch.arange(n, device=env.device).repeat(len(groups) - 1)
            dest = torch.arange(n, total, device=env.device)
            env.env.scene.env_origins[dest] = env.env.scene.env_origins[source].clone()
            paired = {}
            for name in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied',
                         'mocap_pos', 'mocap_quat', 'time', 'eq_active'):
                value = getattr(sim.data, name)
                value[dest] = value[source].clone()
                paired['physics.' + name] = value
            owners = [('wrapper', env), ('pd', action), ('delay', action.disturbances),
                      ('motor', action.wheel_motor), ('contact', env.contacts)]
            for key in ('step_base', 'step_legs', 'step_wheels', 'step_scan', 'step_support'):
                sensor = env.env.scene[key]
                owners += [(key, sensor), (key + '.data', sensor.data)]
            for label, owner in owners:
                for name, value in vars(owner).items():
                    if not isinstance(value, torch.Tensor) or name in ('ids', '_env_ids', 'env_ids'):
                        continue
                    if name in ('_action_history', 'actions', 'sensors') and value.ndim > 1 and value.shape[1] == total:
                        value[:, dest] = value[:, source].clone()
                        paired[label + '.' + name] = value.transpose(0, 1)
                    elif value.ndim and value.shape[0] == total:
                        value[dest] = value[source].clone()
                        paired[label + '.' + name] = value
            obs[dest] = obs[source].clone()
            paired['actor.obs'] = obs
            sim.forward(); env.env.scene.update(dt=0.); sim.sense(); env._update_state()
            initial = {}
            for key, value in paired.items():
                array = value.cpu().numpy().copy().reshape(len(groups), n, *value.shape[1:])
                assert all(np.array_equal(array[0], other) for other in array[1:]), key
                initial[key] = array
            np.savez_compressed(out / 'same-initial-state.npz', **initial)
            origin = env.env.scene.env_origins.cpu().numpy().copy()
            initial_qpos = sim.data.qpos.cpu().numpy().copy()
            model = sim.mj_model
            model.geom_friction[:, 0] = .8
            mujoco.mj_saveModel(model, str(out / 'model.mjb'), None)
            weights = (sim.model.body_mass[:, env.robot.indexing.body_ids].sum(-1).expand(total) * 9.81).cpu().numpy()
            env.contacts.record_substeps = True
            pd_rows, pd_times = [], []
            pd_error, apply_count = [0.], [0]
            old_apply = action.apply_actions

            def apply():
                update = action._pd_tick == 0
                old_apply()
                if update:
                    q = sim.data.qpos[:, action._motor_q_adr]
                    v = sim.data.qvel[:, action._motor_v_adr]
                    target = action._nominal_pos + .5 * action._control_actions
                    legs, wheels = action._leg_action_ids, action._wheel_action_ids
                    target[:, legs] = torch.maximum(torch.minimum(target[:, legs], action._reference_pos[:, legs] + 3.14),
                                                   action._reference_pos[:, legs] - 3.14)
                    target[:, wheels] = (10 * action._control_actions[:, wheels]).clamp(-150., 150.)
                    expected = torch.zeros_like(q)
                    expected[:, legs] = 20 * (target[:, legs] - q[:, legs]) - v[:, legs]
                    expected[:, wheels] = .1 * (target[:, wheels] - v[:, wheels])
                    expected = expected.clamp(-action._effort_limits, action._effort_limits)
                    pd_error[0] = max(pd_error[0], float(abs(expected - action.torques).max()))
                    pd_rows.append(torch.stack((target, q, v, action.torques), 1).cpu().numpy())
                    pd_times.append(apply_count[0] * .001)
                apply_count[0] += 1

            action.apply_actions = apply
            generators = {s: torch.Generator(device=env.device).manual_seed(s) for s in (71001, 71002, 71003)}
            scales = torch.tensor([profiles[c['profile']] for c in samples], device=env.device)
            alive = torch.ones(total, dtype=torch.bool, device=env.device)
            reasons = ['duration'] * total
            stopped = [None] * total
            records, draws_saved, noisy_saved, history_saved = [], [], [], []
            ring = None
            max_noise_error = 0.
            contact_count, contact_error, external_force = 0, 0., 0.
            for tick in range(2000):
                if time.time() >= deadline:
                    raise TimeoutError('High stance diagnostic deadline expired')
                env.commands[:, 0] = torch.tensor([c['speed'] for c in samples], device=env.device)
                env.commands[env.crossed | ~alive, :2] = 0.
                env.fixed_commands.copy_(env.commands)
                obs, _ = env.observations()
                env.history[:, -25:] = env._history_frame
                draws = {s: torch.rand((n, 25), device=env.device, generator=g) * 2 - 1 for s, g in generators.items()}
                uniform = torch.cat([draws[g['seed']] for g in groups])
                noise = uniform * scales
                ring = noise[:, None].repeat(1, 5, 1) if ring is None else torch.cat((ring[:, 1:], noise[:, None]), 1)
                history = env.history.view(total, 5, 25)
                assert torch.equal(history[:, -1].clamp(-100., 100.), obs)
                noisy = (history[:, -1] + noise).clamp(-100., 100.)
                noisy_history = (history + ring).reshape(total, 125)
                max_noise_error = max(max_noise_error, float(abs((noisy - obs) - noise).max()))
                assert torch.equal(noisy[:n], obs[:n])
                assert torch.equal(noisy_history[:n], env.history[:n])
                assert torch.equal(noisy_history[:, -25:].clamp(-100., 100.), noisy)
                acts, _ = policy.act_inference(noisy, noisy_history)
                acts[~alive] = 0.
                command, valid = env.commands.clone(), alive.clone()
                for i in (alive & (env.commands[:, :2].abs().amax(-1) < 1e-6)).nonzero().flatten().tolist():
                    if stopped[i] is None:
                        stopped[i] = tick * .01
                _, _, _, done, info, _ = env.step(acts, auto_reset=False, resample=False)
                assert torch.isfinite(sim.data.qpos).all() and torch.isfinite(sim.data.qvel).all()
                row = dict(qpos=sim.data.qpos, qvel=sim.data.qvel, command=command, valid=valid,
                    position=env.position, wheel_pos=sim.data.xpos[:, env.wheel_ids], support=env.contacts.supported,
                    support_height=env.support_height, normal_support=env.normal_support, terrain_valid=info['terrain_valid'],
                    linear_body=env.linear, angular_body=env.angular, linear_world=env.robot.data.root_link_lin_vel_w,
                    gravity=env.gravity, force=torch.stack(env.contacts.step_forces, 1),
                    speed_peak=env.contacts.step_speed_peak, crossed=env.crossed, success=env.success,
                    task_failure=info['step_failure'], terminated=done)
                records.append({k: v.cpu().numpy().copy() for k, v in row.items()})
                draws_saved.append(uniform.cpu().numpy()); noisy_saved.append(torch.stack((obs, noisy), 1).cpu().numpy())
                history_saved.append(noisy_history.cpu().numpy())
                fatal = info['step_failure'] | (env.fail_ticks > cfg.env.fail_ticks) | info['time_outs']
                for i in (alive & fatal).nonzero().flatten().tolist():
                    reasons[i] = ('missing_terrain' if not info['terrain_valid'][i] else
                                  'fall' if env.fail_ticks[i] > cfg.env.fail_ticks else
                                  'timeout' if info['time_outs'][i] else 'task_failure')
                alive &= ~fatal
                if tick % 100 == 0:
                    assert not sim.wp_data.overflow.numpy().any()
                    external_force = max(external_force, float(sim.data.xfrc_applied.abs().max()))
                    count = int(sim.wp_data.nacon.numpy()[0])
                    terrain = model.body('terrain').id
                    pair = sim.wp_data.contact.geom.numpy()[:count]
                    hit = (model.geom_bodyid[pair] == terrain).any(-1)
                    mu = sim.wp_data.contact.friction.numpy()[:count, :2][hit]
                    if len(mu):
                        contact_count += len(mu); contact_error = max(contact_error, float(abs(mu - .8).max()))
                    print(json.dumps(dict(step=tick, total=2000, worlds=total, wall_s=time.time() - started)), flush=True)
                dead = (~alive).nonzero().flatten()
                if len(dead):
                    env.reset_idx(dead)
            assert all(torch.equal(v.cpu(), before[k]) for k, v in policy.state_dict().items())
            assert pd_error[0] < 1e-4 and apply_count[0] == 20000 and len(pd_rows) == 4000
            assert max_noise_error < 2e-5 and contact_count > 0 and contact_error < 1e-6 and external_force == 0.
            pd_values = np.asarray(pd_rows)
            assert (abs(pd_values[:, :, 3]) <= np.array([40., 40., 3.9, 40., 40., 3.9]) + 1e-4).all()
            np.savez_compressed(out / 'motor-pd.npz', time=np.asarray(pd_times), values=pd_values)
            np.savez_compressed(out / 'policy-noise.npz', uniform=np.asarray(draws_saved),
                                current=np.asarray(noisy_saved), history=np.asarray(history_saved))
            hashes = {}
            for i, sample in enumerate(samples):
                trace = {k: np.stack([r[k][i] for r in records]) for k in records[0]}
                trace.update(time=np.arange(1, 2001) * .01, origin=origin[i], initial=initial_qpos[i],
                             weight=weights[i], stop_reason=np.array(reasons[i]))
                path = out / 'traces' / (sample['id'] + '.npz')
                np.savez_compressed(path, **trace); hashes[sample['id']] = sha(path)
            write(out / 'runtime-audit.json', dict(passed=True, optimizer_updates=0, networks_unchanged=True,
                checkpoint_unchanged=sha(RUN / 'model_final.pt') == MODEL_SHA, pd_formula_max_error=pd_error[0],
                noise_max_error=max_noise_error, contacts_checked=contact_count, friction_max_error=contact_error,
                external_force_peak=external_force, same_initial_arrays=len(initial), physics_steps=20000,
                warmup_physics_steps=10, world_physics_transitions=20000 * total, policy_calls=2000,
                trace_sha256=hashes, first_stop_s=stopped, stop_reasons=reasons,
                loaded_source=frozen.LOADED, source_checks=source_checks, wall_s=time.time() - started))
    finally:
        if env is not None:
            env.close()


def band(value, rate):
    from scipy.signal import detrend
    value = detrend(np.asarray(value, dtype=float), axis=0)
    spectrum = np.fft.rfft(value, axis=0)
    frequency = np.fft.rfftfreq(len(value), 1 / rate)
    spectrum[(frequency < 8) | (frequency > 14)] = 0.
    return np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(value), axis=0) ** 2, axis=0))


def report(out):
    setup_source()
    from scripts.agent.fudan_ppo_step.evaluate import score
    plan = json.loads((out / 'plan.json').read_text())
    audit = json.loads((out / 'runtime-audit.json').read_text())
    assert audit['passed'] and sha(RUN / 'model_final.pt') == MODEL_SHA
    with np.load(out / 'motor-pd.npz') as data:
        pt, pd = data['time'].copy(), data['values'].copy()
    rows, stair_samples, cohort = [], [], hashlib.sha256()
    for i, sample in enumerate(plan['samples']):
        path = out / 'traces' / (sample['id'] + '.npz')
        assert sha(path) == audit['trace_sha256'][sample['id']]
        with np.load(path) as tr:
            trace = {k: tr[k] for k in tr.files}
        if sample['height']:
            stair_sample = dict(id=sample['id'], mode='up', height=.2, speed=sample['speed'],
                angle_deg=0., friction=.8, reverse=False)
            task = score(stair_sample, trace, duration=20., max_approach_speed=3.,
                         allow_temporary_support=True, landing_clearance_m=.15)
            stair_samples.append(stair_sample)
            init = trace['initial'].copy(); init[:3] -= trace['origin']; cohort.update(init.tobytes())
            cohort.update(np.array(.8, np.float32).tobytes())
            stop = audit['first_stop_s'][i]
            window = [stop + .5, stop + 2.5] if stop is not None else None
        else:
            task, window = None, plan['measurement']['hold_window_s']
        row = dict(sample, task_score=task, valid_duration_s=float(trace['valid'].sum() * .01),
            stopped_s=audit['first_stop_s'][i], stop_reason=audit['stop_reasons'][i], window_s=window)
        take = (trace['time'] >= window[0]) & (trace['time'] < window[1]) if window else np.zeros(len(trace['time']), bool)
        pm = (pt >= window[0]) & (pt < window[1]) if window else np.zeros(len(pt), bool)
        valid = window is not None and take.sum() >= round((window[1] - window[0]) / .01) - 1 and trace['valid'][take].all()
        row['measurement_complete'] = bool(valid)
        if valid:
            gyro, gravity = trace['angular_body'][take, 0], trace['gravity'][take]
            row.update(gyroX_band8to14_rad_s=float(band(gyro, 100.)),
                target_band8to14_rad=band(pd[pm, i, 0][:, [0, 1, 3, 4]], 200.).tolist(),
                actual_joint_band8to14_rad=band(pd[pm, i, 1][:, [0, 1, 3, 4]], 200.).tolist(),
                torque_band8to14_Nm=band(pd[pm, i, 3][:, [0, 1, 3, 4]], 200.).tolist(),
                horizontal_speed_peak=float(np.linalg.norm(trace['linear_world'][take, :2], axis=1).max()),
                roll_peak_deg=float(np.degrees(abs(np.arctan2(-gravity[:, 1], -gravity[:, 2]))).max()),
                pitch_peak_deg=float(np.degrees(abs(np.arcsin(np.clip(gravity[:, 0], -1., 1.)))).max()),
                nonwheel_force_peak_N=float(trace['force'][take].max()))
        rows.append(row)
    aggregates = []
    for key in sorted({r['key'] for r in rows}):
        for profile in plan['noise_scales']:
            chosen = [r for r in rows if r['key'] == key and r['profile'] == profile]
            valid = [r for r in chosen if r['measurement_complete']]
            summary = dict(case=key, profile=profile, total=len(chosen), measured=len(valid),
                task_passed=sum(bool(r['task_score'] and r['task_score']['passed']) for r in chosen))
            for name in ('gyroX_band8to14_rad_s', 'target_band8to14_rad', 'actual_joint_band8to14_rad',
                         'torque_band8to14_Nm', 'horizontal_speed_peak', 'roll_peak_deg', 'pitch_peak_deg'):
                if valid:
                    summary[name] = np.mean([r[name] for r in valid], axis=0).tolist()
            aggregates.append(summary)
    write(out / 'metrics.json', dict(rows=rows, aggregates=aggregates, optimizer_updates=0,
        warning='Missing post-stop windows remain missing; climb impacts are not classified as periodic jitter'))
    write(out / 'step-protocol.json', dict(task='fudan_ppo_step', backend='mujoco_warp_gpu', samples=stair_samples,
        duration_s=20., dt=.01, recipe=plan['recipe'], checkpoint_sha256=MODEL_SHA,
        cohort_sha256=cohort.hexdigest(), trace_sha256=audit['trace_sha256'],
        source_fingerprint=sha(out / 'diagnostic-source.py'), optimizer_updates=0))
    print(json.dumps(aggregates, ensure_ascii=False), flush=True)


def render(out, key, profiles=('clean', 'step', 'walk6838')):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    plan = json.loads((out / 'plan.json').read_text())
    selected = [next(c for c in plan['samples'] if c['profile'] == p and c['seed'] == 71001 and c['key'] == key)
                for p in profiles]
    folder = out / ('video-grid-' + key); folder.mkdir(exist_ok=False)
    panels = []
    try:
        for sample in selected:
            model = mujoco.MjModel.from_binary_path(str(out / 'model.mjb'))
            disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth, model.vis.global_.offheight = 480, 480
            data, camera = mujoco.MjData(model), mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 1.8, 120., -25.
            with np.load(out / 'traces' / (sample['id'] + '.npz')) as tr:
                trace = {k: tr[k].copy() for k in tr.files}
            panels.append((sample, model, data, camera, trace, mujoco.Renderer(model, width=480, height=480)))
        # 任何一栏首次失败就结束视频，避免把诊断复位后的姿态当连续轨迹。
        fps = 50
        valid_seconds = min(float(item[4]['valid'].sum()) * .01 for item in panels)
        frames = min(1000, int(valid_seconds * fps))
        path = folder / 'comparison-1x.mp4'
        with imageio.get_writer(path, fps=fps, codec='libx264', quality=7, macro_block_size=None) as writer:
            for f in range(frames):
                t, canvas = f / fps, Image.new('RGB', (1440, 480), '#18212b')
                for col, (sample, model, data, camera, trace, renderer) in enumerate(panels):
                    i = min(1999, max(0, round(t / .01) - 1))
                    data.qpos[:] = trace['qpos'][i]
                    mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
                    camera.lookat[:] = data.qpos[:3] + [0., 0., .03]
                    renderer.update_scene(data, camera=camera)
                    # 只在画面叠加地面标记，让跟随镜头中的漂移可见；不推进或改动物理。
                    origin = trace['origin']
                    centre = np.round((data.qpos[:2]-origin[:2])/.2)*.2+origin[:2]
                    for dx in np.arange(-1., 1.01, .2):
                        for dy in np.arange(-.8, .81, .2):
                            x, y = centre + [dx, dy]
                            top = (sample['height'] > 0 and origin[0]+1.2 <= x <= origin[0]+1.2+2.09
                                   and abs(y-origin[1]) < 1.148/2)
                            point = np.array([x, y, origin[2]+(.2 if top else 0.)+.002])
                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.array([.015,.015,.001]),
                                point, np.eye(3).ravel(), np.array([.9,.65,.2,1.]))
                            renderer.scene.ngeom += 1
                    picture = Image.fromarray(renderer.render().copy()); pen = ImageDraw.Draw(picture)
                    pen.rectangle((0, 0, 480, 86), fill='#18212b')
                    label = dict(clean='NO NOISE', step='STEP TRAIN NOISE', walk6838='6838 TRAIN NOISE',
                        gyro_x='ONLY GYRO X', gyro_yz='ONLY GYRO Y/Z', gravity='ONLY GRAVITY',
                        joint_angle='ONLY JOINT ANGLE', joint_velocity='ONLY JOINT SPEED')[sample['profile']]
                    pen.text((7, 5), f'STEP4200 | {label} | t={t:.2f}s | 1x', fill='white')
                    pen.text((7, 25), f'height cmd {sample["height_command"]:.2f}m | gyroX {trace["angular_body"][i,0]:+.3f}', fill='white')
                    pen.text((7, 44), f'vx cmd {trace["command"][i,0]:.2f} | valid {bool(trace["valid"][i])}', fill='white')
                    speed = np.linalg.norm(trace['linear_world'][i,:2])
                    drift = np.linalg.norm(trace['qpos'][i,:2]-trace['qpos'][0,:2])
                    pen.text((7, 63), f'speed {speed:.3f}m/s | XY displacement {drift:.3f}m', fill='white')
                    canvas.paste(picture, (480 * col, 0))
                writer.append_data(np.asarray(canvas))
                if f in (100, 400, 900):
                    canvas.save(folder / f'key-{f:03d}.png')
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps'] == fps and sum(1 for _ in reader) == frames
        write(folder / 'media.json', dict(video_sha256=sha(path), fps=fps, frames=frames, duration_s=frames/fps,
            case=key, physics_steps=0, optimizer_updates=0, full_decode_passed=True,
            traces={s['id']: sha(out / 'traces' / (s['id'] + '.npz')) for s in selected}))
    finally:
        for item in panels:
            item[-1].close()


def verify(out):
    import ast
    import imageio.v2 as imageio
    plan = json.loads((out / 'plan.json').read_text())
    audit = json.loads((out / 'runtime-audit.json').read_text())
    assert audit['passed'] and audit['optimizer_updates'] == 0 and audit['networks_unchanged']
    assert sha(RUN / 'model_final.pt') == MODEL_SHA
    def run_ast(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    assert run_ast(Path(__file__)) == run_ast(out / 'diagnostic-source.py')
    with np.load(out / 'same-initial-state.npz') as saved:
        for key in saved.files:
            assert all(np.array_equal(saved[key][0], other) for other in saved[key][1:]), key
    count = len(plan['groups'])
    n = len(plan['samples']) // count
    valid_rows = []
    for sample in plan['samples']:
        with np.load(out / 'traces' / (sample['id'] + '.npz')) as trace:
            valid_rows.append(trace['valid'].copy())
    valid = np.stack(valid_rows, axis=1)
    with np.load(out / 'policy-noise.npz') as saved:
        u, x, h = saved['uniform'], saved['current'], saved['history']
        scales = np.asarray([plan['noise_scales'][c['profile']] for c in plan['samples']], dtype=np.float32)
        expected = np.clip(x[:, :, 0] + u * scales, -100., 100.)
        noise_error = float(abs(expected - x[:, :, 1]).max())
        assert noise_error == 0.
        assert np.array_equal(np.clip(h[:, :, -25:], -100., 100.), x[:, :, 1])
        assert np.array_equal(x[:, :n, 0], x[:, :n, 1])
        for i, group in enumerate(plan['groups']):
            j = next(j for j, other in enumerate(plan['groups']) if other['seed'] == group['seed'])
            assert np.array_equal(u[:, i*n:(i+1)*n], u[:, j*n:(j+1)*n])
        for slot in range(4):
            # 首回合失败后诊断会复位；只核对实际用于评分的连续首回合。
            take = valid[4:] & valid[3:-1]
            assert np.array_equal(h[4:, :, slot*25:(slot+1)*25][take],
                                  h[3:-1, :, (slot+1)*25:(slot+2)*25][take])
    with np.load(out / 'motor-pd.npz') as saved:
        a = saved['values']; assert a.shape == (4000, n*count, 4, 6)
        expected = np.zeros_like(a[:, :, 0])
        legs, wheels = [0, 1, 3, 4], [2, 5]
        expected[:, :, legs] = 20*(a[:, :, 0][:, :, legs]-a[:, :, 1][:, :, legs])-a[:, :, 2][:, :, legs]
        expected[:, :, wheels] = .1*(a[:, :, 0][:, :, wheels]-a[:, :, 2][:, :, wheels])
        expected = np.clip(expected, -np.array([40.,40.,3.9,40.,40.,3.9]), np.array([40.,40.,3.9,40.,40.,3.9]))
        pd_error = float(abs(expected-a[:, :, 3]).max()); assert pd_error < 1e-4
    for sid, digest in audit['trace_sha256'].items():
        path = out / 'traces' / (sid + '.npz'); assert sha(path) == digest
        with np.load(path) as trace:
            assert all(np.isfinite(trace[k]).all() for k in ('qpos', 'qvel', 'angular_body', 'force'))
    media = []
    for folder in sorted(out.glob('video-*')):
        meta = json.loads((folder / 'media.json').read_text()); path = folder / 'comparison-1x.mp4'
        assert sha(path) == meta['video_sha256']
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps'] == meta['fps']
            frames = sum(1 for _ in reader)
        assert frames == meta['frames']
        for sid, digest in meta['traces'].items():
            assert sha(out / 'traces' / (sid + '.npz')) == digest
        media.append(dict(file=str(path), frames=frames, fps=meta['fps']))
    write(out / 'verification-final.json', dict(passed=True, source_run_ast_unchanged=True, physics_steps=0,
        optimizer_updates=0, noise_injection_error=noise_error, pd_formula_error=pd_error,
        paired_initial_arrays=True, full_history_shift=True, videos=media))
    write(out / 'evidence-final-sha256.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('run', 'report', 'render', 'verify'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float)
    p.add_argument('--case', default='hold_h0.35')
    args = p.parse_args()
    if args.mode == 'run':
        if args.deadline is None:
            p.error('GPU run requires --deadline')
        run(args.output, args.deadline)
    elif args.mode == 'report':
        report(args.output)
    elif args.mode == 'render':
        render(args.output, args.case)
    else:
        verify(args.output)


if __name__ == '__main__':
    main()
