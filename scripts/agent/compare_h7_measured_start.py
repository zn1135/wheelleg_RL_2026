#!/usr/bin/env python3
"""Offline measured-start-pose dynamics; frozen model, no hardware or training."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
I2N = np.array([0, 1, 4, 2, 3, 5])
SIGN = np.array([1., 1., -1., -1., 1., -1.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collision_bottoms(model, data, floor):
    result = []
    for g in range(model.ngeom):
        if g == floor or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        rotation = data.geom_xmat[g].reshape(3, 3)
        kind, size = model.geom_type[g], model.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            begin, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            bottom = (model.mesh_vert[begin:begin+count] @ rotation[2]).min()
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            bottom = -abs(rotation[2]) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            bottom = -size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            bottom = -size[0]*np.linalg.norm(rotation[2, :2])-size[1]*abs(rotation[2, 2])
        else:
            raise ValueError(f'Unsupported collision geometry: {kind}')
        result.append((float(bottom+data.geom_xpos[g, 2]), g))
    return sorted(result)


def initialize(sim, mirror, capture, frame, before, relaxed):
    env = sim.Playback('normal', (0., 0., float(frame[60]/5)))
    m, d = env.model, env.data
    pose = mirror.MirrorPose(sim.MODEL_ASSET, capture/'jetson-source/machine_config.c')
    pose.data.qpos[:3] = 0.
    pose.data.qpos[3:7] = before[22:26]/np.linalg.norm(before[22:26])
    # 使用策略采样时的已校正关节角，保留实机左右腿的不对称。
    angles = frame[67:71].astype(float)+env.nominal[sim.LEGS]
    for side in (0, 1):
        if not pose._solve_leg(side, angles[2*side:2*side+2]):
            raise ValueError('Measured pose cannot close')
    pose.data.qpos[pose.wheel_q] = before[18:20]*SIGN[4:]
    d.qpos[:] = pose.data.qpos
    d.qvel[env.v] = (frame[73:79].astype(float)*10)[I2N]
    d.qvel[3:6] = frame[61:64].astype(float)*2
    # 被动关节速度由闭链约束求解；未测得的机身线速度假定为零。
    passive = [int(m.joint(n).dofadr[0]) for n in
               sum(mirror.PASSIVE, ())+tuple(f'{s}_{p}_joint' for s in ('left', 'right')
                                             for p in ('spring1', 'spring2'))]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    blocks = []
    pairs = [p for side in mirror.PIN_PAIRS for p in side]
    pairs += [(f'{s}_spring_tip', f'{s}_gas_spring_lower') for s in ('left', 'right')]
    for a, b in pairs:
        ja, jb = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, ja, None, m.site(a).id)
        mujoco.mj_jacSite(m, d, jb, None, m.site(b).id)
        blocks.append(ja-jb)
    jac = np.vstack(blocks)
    velocity, _, rank, _ = np.linalg.lstsq(jac[:, passive], -jac@d.qvel, rcond=1e-10)
    if rank != len(passive):
        raise ValueError('Passive velocity reconstruction is singular')
    d.qvel[passive] = velocity
    bottoms = collision_bottoms(m, d, env.floor_geom)
    d.qpos[2] -= bottoms[0][0]
    changes, violations = [], []
    for j in range(m.njnt):
        value = float(d.qpos[m.jnt_qposadr[j]])
        low, high = map(float, m.jnt_range[j])
        if m.jnt_limited[j] and not low <= value <= high:
            violations.append({'joint': m.joint(j).name, 'excess_deg':
                               float(np.rad2deg(max(low-value, value-high)))})
            if relaxed:
                if m.joint(j).name not in ('lf1', 'rf1'):
                    raise ValueError('Unexpected limit violation')
                # 仅此运行的诊断假设：容纳实测初姿并留1度余量；源XML不改。
                new = [min(low, value-np.deg2rad(1)), max(high, value+np.deg2rad(1))]
                m.jnt_range[j] = new
                changes.append({'joint': m.joint(j).name, 'old_rad': [low, high], 'new_rad': new})
    env.steps = 11  # 从首次有效RL输出开始，不重复已经过去的200ms保持阶段。
    env.ground_start = True
    env.last_action[:] = frame[79:85][I2N]
    mujoco.mj_forward(m, d)
    return env, {'closure_m': pose.residual_m, 'joint_limit_violations': violations,
                 'limit_changes': changes, 'root_height_m': float(d.qpos[2]),
                 'lowest_collision_body': m.body(m.geom_bodyid[bottoms[0][1]]).name,
                 'initial_collision_bottoms_m': [
                     {'body': m.body(m.geom_bodyid[g]).name, 'floor_gap_m': height+d.qpos[2]}
                     for height, g in bottoms[:6]],
                 'velocity_closure_m_s': float(abs(jac@d.qvel).max())}


def simulate(env, sim, actor, policy_time, policy_obs, first_tick, seconds):
    poses, velocities, observations, actions, torques, rows = [], [], [], [], [], []
    direction = env.data.xmat[env.base_body].reshape(3, 3)[:2, 0].copy()
    direction /= np.linalg.norm(direction)
    for step in range(round(seconds/.02)+1):
        poses.append(env.data.qpos.copy())
        velocities.append(env.data.qvel.copy())
        row = env.metrics()
        row['forward_m'] = float(env.data.qpos[:2]@direction)
        rows.append(row)
        if step == round(seconds/.02):
            break
        tick = first_tick+step*20
        ix = max(0, np.searchsorted(policy_time, tick, side='right')-1)
        source = policy_obs[ix]
        env.command[:] = source[[0, 2, 3]]/[1, 1, 5]
        obs = env.observation()
        # 两组共用实录的命令/模式切换时序，传感器由各自仿真产生。
        obs[:4], obs[28:] = source[:4], source[28:]
        with torch.inference_mode():
            action = actor(torch.from_numpy(obs)[None]).numpy()[0]
        observations.append(obs.copy())
        actions.append(action.copy())
        env.last_action[:] = action
        for substep in range(sim.DECIMATION):
            torque, spring = env.controls(action)
            env.data.ctrl[env.motor_ids] = torque
            env.data.ctrl[env.spring_ids] = spring
            mujoco.mj_step(env.model, env.data)
            if max(env.data.warning.number):
                raise ValueError('MuJoCo numerical warning')
        torques.append(torque.copy())
        env.steps += 1
        mujoco.mj_forward(env.model, env.data)
        if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
            raise ValueError('Nonfinite state')
    return dict(qpos=np.asarray(poses), qvel=np.asarray(velocities), obs=np.asarray(observations),
                actions=np.asarray(actions), torque=np.asarray(torques),
                time_s=np.arange(len(poses))*.02), rows


def render(cases, output):
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 19)
    views = []
    for label, env, states in cases:
        m = env.model
        m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, width=640, height=480)
        camera = mujoco.MjvCamera()
        mujoco.mj_forward(m, env.data)
        quat = states['qpos'][0, 3:7]
        yaw = np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
        camera.lookat[:] = direction*.15+[0., 0., .17]
        camera.distance, camera.azimuth, camera.elevation = 1.9, np.rad2deg(yaw)+90, -18
        views.append((label, env, states, d, renderer, camera, direction))
    movie = output/'measured-start-pose-30s.mp4'
    try:
        with imageio.get_writer(movie, fps=25, codec='libx264', quality=7,
                               macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(750):
                index = round(frame/25/.02)
                canvas = Image.new('RGB', (1280, 608), '#18212b')
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 4), f'实录初姿仿真 | 6848 | {frame/25:.2f} 秒 | 原速 | 无绳索', font=font, fill='white')
                draw.text((12, 31), '机身姿态、腿角和动作历史取自实录；落地高度推算、机身线速度假定为零', font=font, fill='#a8cfff')
                for side, (label, env, states, d, renderer, camera, direction) in enumerate(views):
                    d.qpos[:], d.qvel[:] = states['qpos'][index], states['qvel'][index]
                    mujoco.mj_forward(env.model, d)
                    renderer.update_scene(d, camera=camera)
                    for x in np.arange(-.6, 1.21, .1):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        c, s = direction[:2]
                        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([.001, .30, .0005]), direction*x+[0, 0, .0008], rotation.ravel(),
                            np.array([.9, .15, .15, 1.] if abs(x)<1e-8 else [.6, .6, .6, .7]))
                        renderer.scene.ngeom += 1
                    canvas.paste(Image.fromarray(renderer.render().copy()), (side*640, 128))
                    draw.text((side*640+12, 62), label, font=font, fill='#ffcc8a')
                    x = float(states['qpos'][index, :2]@direction[:2])
                    draw.text((side*640+12, 91), f'相对投入前向位移 {x:+.3f}m；地面刻度10cm', font=font, fill='white')
                if frame in (0, 50, 749):
                    canvas.save(output/f'preview-{frame:04d}.png')
                writer.append_data(np.asarray(canvas))
                if frame and frame % 250 == 0:
                    print(f'Rendered {frame}/750', flush=True)
    finally:
        for view in views:
            view[4].close()
    return movie


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.out/Path(__file__).name)
    (args.out/'report.json').write_text('{"complete":false}\n')
    frozen = args.capture_root/'frozen-baseline'
    manifest = json.loads((frozen/'manifest.json').read_text())
    for item in manifest['files']:
        if digest(frozen/item['path']) != item['sha256']:
            raise ValueError('Frozen source changed: '+item['path'])
    sys.path.insert(0, str(frozen/'sim2sim'))
    sim = load_module('matched_pose_frozen_cpu', frozen/'sim2sim/mj_sim2sim_huananhu_multimode.py')
    mirror = load_module('matched_pose_mirror', ROOT/'sim2sim/real2sim_huananhu.py')
    checkpoint = ROOT/'deployment/H7_RL_ros/models/model6848-20261021/model_6848.pt'
    actor = sim.load_actor(checkpoint, '4a96a4c216fed4cedc2153667defbc0ae7974b9a31eb857022a31b4658bb3001')
    torch.set_num_threads(1)
    saved = np.load(args.capture_root/'chronology/frames.npz')
    f, ticks = saved['frames'], saved['board_tick_ms']
    first = np.flatnonzero((f[:, 152] == 0)&((f[:, 109].astype(int)&33) == 33))[0]
    frame, before = f[first], f[first-1]
    start = float(int(frame[50]) | (int(frame[51]) << 16))
    unique = {}
    for i in np.flatnonzero((f[:, 152] == 0)&((f[:, 47].astype(int)&1) != 0)):
        at = int(f[i, 50]) | (int(f[i, 51]) << 16)
        if start <= at <= start+31000:
            unique.setdefault(at, i)
    policy_time = np.array(sorted(unique))
    policy_obs = np.array([f[unique[t], 57:96] for t in policy_time])
    if policy_time[0] != start or np.max(np.diff(policy_time)) > 30:
        raise ValueError('Policy schedule missing at startup')
    output_times, output_torques = [], []
    with (args.capture_root/'output/outputs.csv').open() as stream:
        for r in csv.DictReader(stream):
            at = int(r['begin_ns'])/1e6
            if r['kind'] == 'control' and r['strategy'] == '0' and start <= at <= start+1100:
                output_times.append(at)
                output_torques.append((np.array([float(r[f'request_{j}_firmware_nm']) for j in range(6)])*SIGN)[I2N])
    output_times, output_torques = np.array(output_times), np.array(output_torques)
    first_nonzero = np.flatnonzero(np.any(output_torques != 0, axis=1))[0]
    # 首次策略采样比首次输出早约2ms；保留真实的初始零输出和500Hz请求时序。
    real_start_ms = float(output_times[first_nonzero])
    policy_start = start
    start = real_start_ms
    cases, reports, results = [], {}, {}
    for relaxed in (False, True):
        name = 'diagnostic-relaxed' if relaxed else 'original-limits'
        env, initial = initialize(sim, mirror, args.capture_root, frame, before, relaxed)
        obs = env.observation(); obs[:4], obs[28:] = policy_obs[0, :4], policy_obs[0, 28:]
        error = float(abs(obs-policy_obs[0]).max())
        if error > 1e-5:
            raise ValueError(f'Initial measured input mismatch: {error}')
        with torch.inference_mode():
            action = actor(torch.from_numpy(obs)[None]).numpy()[0]
        action_error = float(abs(action-frame[96:102]).max())
        if action_error > 1e-4:
            raise ValueError('Initial actor output mismatch')
        initial_torque = env.controls(action)[0]
        initial_torque_error = float(abs(initial_torque-output_torques[first_nonzero]).max())
        if initial_torque_error > 1e-3:
            raise ValueError(f'Initial applied torque mismatch: {initial_torque_error}')
        states, rows = simulate(env, sim, actor, policy_time, policy_obs, start, 30.)
        np.savez_compressed(args.out/(name+'.npz'), **states)
        (args.out/(name+'.json')).write_text(json.dumps(rows)+'\n')
        forward = np.array([r['forward_m'] for r in rows])
        reports[name] = {'initialization': initial, 'initial_obs_max_error': error,
            'initial_action_max_error': action_error, 'forward_0_to_30_m': float(forward[-1]-forward[0]),
            'initial_requested_torque_max_error_nm': initial_torque_error,
            'forward_2_to_30_m': float(forward[-1]-forward[100]),
            'max_closure_m': max(r['closed_chain_residual_m'] for r in rows),
            'base_contact_fraction_after_2s': float(np.mean([r['base_contact'] for r in rows[100:]])),
            'both_wheels_contact_fraction_after_2s': float(np.mean([r['both_wheels_contact'] for r in rows[100:]])),
            'max_roll_pitch_deg_after_2s': np.max(abs(np.array([r['roll_pitch_deg'] for r in rows[100:]])), axis=0).tolist()}
        results[name] = states
        # 从同一初态重新步进，核对保存的前1秒轨迹，包含模型约束和策略递归。
        repeat_env, _ = initialize(sim, mirror, args.capture_root, frame, before, relaxed)
        repeat, _ = simulate(repeat_env, sim, actor, policy_time, policy_obs, start, 1.)
        repeat_error = float(abs(repeat['qpos']-states['qpos'][:51]).max())
        if repeat_error > 1e-10:
            raise ValueError('Repeated startup trajectory differs')
        reports[name]['repeat_first_1s_qpos_max_error'] = repeat_error
        cases.append(('仅本次诊断放宽两膝限位' if relaxed else '原模型限位：初姿已有越限', env, states))
        print(name, json.dumps(reports[name]), flush=True)
    np.testing.assert_array_equal(results['original-limits']['qpos'][0], results['diagnostic-relaxed']['qpos'][0])
    np.testing.assert_array_equal(results['original-limits']['qvel'][0], results['diagnostic-relaxed']['qvel'][0])
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    real_mask = (ticks >= start)&(ticks <= start+2000)&((f[:,109].astype(int)&33) == 33)
    real_t = (ticks[real_mask]-start)/1000
    real_obs = f[real_mask, 57:96]
    quat = f[real_mask, 22:26].astype(float)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    pitch = np.rad2deg(np.arcsin(np.clip(2*(quat[:,0]*quat[:,2]-quat[:,1]*quat[:,3]), -1, 1)))
    axes[0].plot(real_t, pitch, 'k', label='real IMU pitch')
    for j in (0, 1):
        axes[1+j].plot(real_t, f[real_mask, 122+j], 'k', label='real '+('lf0' if j == 0 else 'lf00'))
    axes[3].plot(real_t, (f[real_mask,126+4]-f[real_mask,126+5])/2, 'k', label='real common wheel speed')
    for label, env, states in cases:
        name = 'diagnostic-relaxed' if env.model.jnt_range[env.model.joint('lf1').id,1] > .12 else 'original-limits'
        t = states['time_s']; obs = states['obs']
        axes[0].plot(t[:-1], np.rad2deg(np.arcsin(np.clip(obs[:,7], -1, 1))), label=name)
        for j in (0, 1): axes[1+j].plot(t, states['qpos'][:, env.q[j]])
        axes[3].plot(t, (states['qvel'][:,env.v[2]]-states['qvel'][:,env.v[5]])/2)
        measured_q = f[real_mask, 122:126]
        sim_q = states['qpos'][:, env.q[sim.LEGS]]
        common = (states['qvel'][:,env.v[2]]-states['qvel'][:,env.v[5]])/2
        sample = (t >= real_t[0])&(t <= 1.)
        real_q = np.array([np.interp(t[sample], real_t, measured_q[:,j]) for j in range(4)]).T
        reports[name]['first_1s_measured_comparison'] = {
            'leg_q_rmse_rad': np.sqrt(np.mean((sim_q[sample]-real_q)**2, axis=0)).tolist(),
            'rear_left_crosses_1rad_sim_s': float(t[np.flatnonzero(sim_q[:,1] <= 1.)[0]]),
            'rear_left_crosses_1rad_real_s': float(real_t[np.flatnonzero(measured_q[:,1] <= 1.)[0]]),
            'wheel_common_peak_sim_rad_s': float(common[t <= 1.].max()),
            'wheel_common_peak_real_rad_s': float(((f[real_mask,130]-f[real_mask,131])/2)[real_t <= 1.].max()),
            'caveat': 'Requested outputs and sensor response; rope onset and real root motion are unknown.'}
    for ax, ylabel in zip(axes, ('Pitch [deg]', 'lf0 [rad]', 'lf00 [rad]', 'Common wheel [rad/s]')):
        ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.set_xlim(0, 2)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel('Seconds since first nonzero output; rope onset unknown')
    fig.tight_layout(); fig.savefig(args.out/'startup-comparison.png', dpi=130); plt.close(fig)
    movie = render(cases, args.out)
    report = {'complete': True, 'scope': 'Measured-pose offline dynamics with explicit unmeasured-state assumptions',
        'cases': reports, 'start_policy_tick_ms': policy_start, 'first_nonzero_output_tick_ms': real_start_ms,
        'pose_frame_receive_s': float(saved['receive_s'][first-1]),
        'schedule': 'Recorded command/mode/flags; closed-loop simulated sensors and recurrent actions; no rope',
        'assumptions': ['Root xyz inferred from lowest collision geometry just touching a flat floor.',
            'Root linear velocity is zero; passive joint velocities solve closed-chain constraints.',
            'Wheel angles use signed feedback; attitude/joint velocities/history use recorded policy sample.',
            'Mode and flags follow recorded schedule; this is not a full firmware-state-machine emulator.',
            'Relaxed case changes only two in-memory joint ranges, not the source XML or hardware.',
            'Startup comparison uses measured angles/IMU/wheel velocities; tether onset unknown.',
            'No measured real translation: simulation displacement cannot be labeled a real-position error.'],
        'sources': {str(p): digest(p) for p in (Path(__file__), frozen/'manifest.json', checkpoint,
            args.capture_root/'chronology/frames.npz', ROOT/'sim2sim/real2sim_huananhu.py')},
        'video': str(movie.resolve()), 'video_sha256': digest(movie)}
    # 视频必须能解码首尾，才标为完整。
    reader = imageio.get_reader(movie)
    assert reader.get_data(0).shape == (608, 1280, 3)
    assert reader.get_data(749).shape == (608, 1280, 3)
    reader.close()
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'complete': True, 'cases': reports, 'video': str(movie)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
