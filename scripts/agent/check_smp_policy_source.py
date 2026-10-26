#!/usr/bin/env python3
"""Screen a frozen policy as a standup demonstration source on the current student interface."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg, inverted, reset_pose

ROOT = Path(__file__).resolve().parents[2]
DELAYS = (.002, .006, .008, .010)
DELAY_PAIRS = ((.002, .002),) + tuple((leg, wheel) for leg in DELAYS[1:] for wheel in DELAYS[1:])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_video(model, directory, report):
    """Render recorded acceptance states; never integrate physics or run a policy."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    if digest(MODEL_XML) != report['xml_sha256']:
        raise ValueError('Video XML differs from the recorded acceptance run')
    seed = next(iter(report['seeds']))
    result = report['seeds'][seed]
    trace = directory/f'policy-{seed}.npz'
    if digest(trace) != result['source_trace_sha256']:
        raise ValueError('Recorded acceptance trace hash mismatch')
    rows = result['rows']
    good = next((row for row in rows if row['candidate_qualified']), rows[0])
    bad = next((row for row in rows if not row['candidate_qualified']
                and row['pose_row'] == good['pose_row']), None)
    cases = [good] + ([] if bad is None else [bad])
    with np.load(trace) as source:
        ids = [row['env'] for row in cases]
        poses = source['qpos'][:, ids].copy()
        initial = source['initial_qpos'][ids].copy()
        features = source['features'][:, ids].copy()
        alive = source['alive'][:, ids].copy()
        times = source['time'].copy()
    if poses.shape[-1] != model.nq:
        raise ValueError('Recorded qpos does not match the rendering model')
    # 只平移世界XY到同一画面原点，不修改记录的关节、根高度或姿态。
    origins = initial[:, :2].copy()
    poses[:, :, :2] -= origins[None]
    initial[:, :2] -= origins
    width, height, header, fps = 640, 448, 112, 30
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height-header)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.22, 0., .20]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 110., -18.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 17)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height-header)
    movie = directory/f'acceptance-seed{seed}.mp4'
    length = min(int(alive[:, i].sum()) for i in range(len(cases)))
    duration = float(times[length-1])
    frame_count = round(duration*fps)
    anchor = poses[min(np.searchsorted(times, 2.), len(times)-1), :, :2].copy()
    indices = []
    try:
        with imageio.get_writer(movie, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for frame in range(frame_count):
                # 100Hz存档取最近一帧，以1倍速输出30fps；第0帧使用真实初态。
                t = frame/fps
                index = -1 if frame == 0 else min(length-1, max(0, round(t/.01)-1))
                indices.append(index)
                canvas = Image.new('RGB', (width*len(cases), height), '#18212b')
                draw = ImageDraw.Draw(canvas)
                for case, row in enumerate(cases):
                    data.qpos[:] = initial[case] if index < 0 else poses[index, case]
                    data.qvel[:] = 0.
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    # 10cm刻度及第2秒机身位置参考线只用于看清漂移。
                    for x in [*np.arange(-.5, 1.01, .1), anchor[case, 0]]:
                        reference = abs(x-anchor[case, 0]) < 1e-10
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([.0015 if reference else .0007, .36, .0005]),
                            np.array([x, 0., .0008]), np.eye(3).ravel(),
                            np.array([.9, .1, .1, 1.] if reference else [.6, .6, .6, .7]))
                        renderer.scene.ngeom += 1
                    canvas.paste(Image.fromarray(renderer.render().copy()), (case*width, header))
                    left = case*width+10
                    status = '末段平衡通过' if row['final_balance_v31'] else '末段平衡未通过'
                    color = '#a5e6aa' if row['final_balance_v31'] else '#ffb18a'
                    draw.text((left, 3), f'model3023 | seed {seed} env {row["env"]} | {t:.2f}s | 原速', font=font, fill='white')
                    draw.text((left, 29), f'腿/轮延迟 {row["leg_delay_s"]*1000:.0f}/{row["wheel_delay_s"]*1000:.0f}ms | {status}', font=font, fill=color)
                    if index >= 0:
                        gyro = float(np.linalg.norm(features[index, case, 7:9]))
                        drift = float(np.linalg.norm(poses[index, case, :2]-anchor[case]))
                        displacement = f'{drift:.3f}m' if t >= 2. else '起立中'
                        draw.text((left, 55), f'角速度 {gyro:.3f}rad/s (门槛0.5) | 第2秒后位移 {displacement}', font=font, fill='white')
                    draw.text((left, 81), '固定相机 | 地面10cm刻度 | 红线：第2秒位置 | 定点保持另判', font=font, fill='#a8cfff')
                if frame in (0, round(.6*fps), frame_count-1):
                    canvas.save(directory/f'acceptance-seed{seed}-{frame:04d}.png')
                writer.append_data(np.asarray(canvas))
                if frame and frame % 150 == 0:
                    print(f'video: {frame}/{frame_count} frames', flush=True)
    finally:
        renderer.close()
    # 编码后再读取首末帧，确认文件确实可播放且帧数完整。
    with imageio.get_reader(movie) as reader:
        assert reader.count_frames() == frame_count
        assert reader.get_data(0).shape == (height, width*len(cases), 3)
        reader.get_data(frame_count-1)
    metadata = dict(video=str(movie), video_sha256=digest(movie), fps=fps, frames=frame_count,
        duration_s=frame_count/fps, playback_speed=1, seed=int(seed), cases=cases,
        checkpoint_sha256=report['checkpoint_sha256'], xml_sha256=report['xml_sha256'],
        source_trace_sha256=result['source_trace_sha256'], frame_source_indices=indices,
        renderer_sha256=digest(__file__), optimizer_updates=0,
        scope='Render recorded qpos only; no physics integration or policy inference. Fixed camera; XY origins translated; overlay angular velocity sampled at 100Hz, acceptance evaluated at 500Hz. First seed, first qualified and same-pose rejected case if available; not all environments.')
    (directory/f'acceptance-seed{seed}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(f'Video saved and decoded: {movie}', flush=True)


def render_existing(args):
    report = json.loads((args.render_only/'source-check.json').read_text())
    if not report.get('completed'):
        raise ValueError('Acceptance run is not complete')
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                    num_envs=1, device=args.device)
    try:
        save_video(env.unwrapped.sim.mj_model, args.render_only, report)
    finally:
        env.close()


def paired_reset(env, env_ids):
    reset_pose(env, env_ids)
    if env_ids is None or len(env_ids) == env.num_envs:
        count = env.num_envs//len(DELAY_PAIRS)
        # 每组腿/轮延迟使用同一批关节/根姿态，世界XY只平移到各自原点。
        q = env.sim.data.qpos[:count].clone().repeat(len(DELAY_PAIRS), 1)
        q[:, :2] = env.scene.env_origins[:, :2]
        env.sim.data.qpos.copy_(q)
        env.sim.data.qvel.zero_()


def balance_substep(env):
    probe = getattr(env, '_source_probe', None)
    if probe is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene['robot'].data
    g = robot.projected_gravity_b
    tilt = torch.maximum(torch.atan2(-g[:, 1], -g[:, 2]).abs(), torch.asin(g[:, 0].clamp(-1, 1)).abs())
    base = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    wheels = -env.scene['wheel_contact'].data.force[:, :, 2]
    weight = env._source_weight
    balanced = ((tilt <= math.radians(35)) & (base <= 1.) &
        (wheels >= max(.1*weight, 5.)).all(-1) &
        (wheels.sum(-1) >= .8*weight) & (wheels.sum(-1) <= 1.2*weight) &
        (robot.root_link_ang_vel_b[:, :2].norm(dim=-1) <= .5))
    live = probe['live']
    probe['hold'] = torch.where(live, torch.where(balanced, probe['hold']+1, 0), probe['hold'])
    height = robot.root_link_pos_w[:, 2]-env.scene.env_origins[:, 2]
    strict = balanced & (tilt <= math.radians(15)) & ((height-.33648).abs() <= .03)
    probe['strict_run'] = torch.where(live, torch.where(strict, probe['strict_run']+1, 0), probe['strict_run'])
    probe['strict_ever'] |= live & (probe['strict_run'] >= 250)
    return balanced.float()


@torch.inference_mode()
def check(args):
    if args.num_envs < len(DELAY_PAIRS) or args.num_envs % len(DELAY_PAIRS):
        raise ValueError('num-envs must be divisible by ten leg/wheel delay groups')
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = digest(args.checkpoint)
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.auto_reset = False
    cfg.commands['standup'] = SmpCommandCfg()
    cfg.events['reset_pose'] = EventTermCfg(func=paired_reset, mode='reset')
    cfg.terminations['inverted'] = TerminationTermCfg(func=inverted)
    cfg.rewards = {}  # 纯推理，不调用SMP或PPO优化。
    cfg.metrics['source_balance'] = MetricsTermCfg(func=balance_substep, per_substep=True, reduce='last')
    cfg.actions['pd'].action_delay_s = None
    cfg.actions['pd'].randomize_delays = False
    cfg.actions['pd'].action_delay_choices_s = DELAYS
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                    num_envs=args.num_envs, device=args.device)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
        xml_sha256=digest(MODEL_XML), pose_bank_sha256=digest(MODEL_XML.with_name('huananhu_pose_bank.npz')),
        policy_dt=.01, physics_dt=.002, ground_start_hold_s=.2,
        delay_pairs_s=list(DELAY_PAIRS), optimizer_updates=0, seeds={},
        scope='Frozen actor under current XML, SMP normal-ground reset and zero commands. Nominal motors, no observation noise or external pushes; 2/2ms play and all nine 6/8/10ms leg/wheel delay combinations.')
    try:
        raw = env.unwrapped
        assert raw.step_dt == .01 and raw.physics_dt == .002
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train, device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        actor = runner.alg._raw_actor
        original_actor = {key: value.clone() for key, value in actor.state_dict().items()}
        action = raw.action_manager.get_term('pd')
        raw._source_weight = float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        reader = MotionStateReader(raw)
        count = args.num_envs//len(DELAY_PAIRS)
        with patch.object(runner.alg.optimizer, 'step', side_effect=AssertionError('Training forbidden')):
            for seed in map(int, args.seeds.split(',')):
                obs, _ = raw.reset(seed=seed)
                leg_delay = torch.tensor([round(leg/raw.physics_dt) for leg, wheel in DELAY_PAIRS],
                                         device=args.device).repeat_interleave(count)
                wheel_delay = torch.tensor([round(wheel/raw.physics_dt) for leg, wheel in DELAY_PAIRS],
                                           device=args.device).repeat_interleave(count)
                action.action_delay_steps.copy_(leg_delay)
                action.wheel_action_delay_steps.copy_(wheel_delay)
                qpos, qvel = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
                assert not qvel.any() and action.cfg.ground_start_hold_s == .2
                joints = raw.sim.data.qpos[:, raw.scene['robot'].indexing.joint_q_adr]
                distance = (raw._smp_ground_poses[None]-joints[:, None]).abs().amax(-1)
                assert (distance.min(-1).values < 1e-5).all()
                pose_rows = distance.argmin(-1).cpu().tolist()
                raw._source_probe = dict(live=torch.ones(args.num_envs, dtype=torch.bool, device=args.device),
                    hold=torch.zeros(args.num_envs, dtype=torch.long, device=args.device),
                    strict_run=torch.zeros(args.num_envs, dtype=torch.long, device=args.device),
                    strict_ever=torch.zeros(args.num_envs, dtype=torch.bool, device=args.device))
                probe = raw._source_probe
                buffers = {key: [] for key in ('root', 'motor_q', 'motor_dq', 'wheel_position_w',
                    'ground_height', 'features', 'alive', 'torque', 'action', 'base_contact', 'tilt', 'qpos')}
                durations = torch.zeros(args.num_envs, device=args.device)
                time_out = torch.zeros_like(probe['live'])
                for tick in range(2000):
                    live = probe['live'].clone()
                    actions = policy(obs)
                    obs, _, done, _ = env.step(actions)
                    state = reader.read()
                    g = raw.scene['robot'].data.projected_gravity_b
                    tilt = torch.maximum(torch.atan2(-g[:, 1], -g[:, 2]).abs(),
                                         torch.asin(g[:, 0].clamp(-1, 1)).abs())
                    values = dict(**state, features=reader.features(), alive=live, torque=action.torques,
                        action=actions, base_contact=raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
                        tilt=tilt, qpos=raw.sim.data.qpos)
                    for key, value in values.items():
                        buffers[key].append(value.cpu().numpy().copy())
                    durations += live.float()*raw.step_dt
                    if tick < 20:
                        assert not action.torques.any()
                    assert (raw.command_manager.get_term('standup').command[:, :2] == 0.).all()
                    time_out |= live & raw.reset_time_outs & ~raw.reset_terminated
                    probe['live'] &= ~done.bool()
                    if not probe['live'].any():
                        break
                    if done.any():
                        # 仅第一回合计数；已结束的世界复位以满足引擎要求，后续帧标为无效。
                        reset_ids = done.nonzero().flatten()
                        obs, _ = raw.reset(env_ids=reset_ids)
                        action.action_delay_steps.copy_(leg_delay)
                        action.wheel_action_delay_steps.copy_(wheel_delay)
                arrays = {key: np.asarray(values) for key, values in buffers.items()}
                rows = []
                for index in range(args.num_envs):
                    mask = arrays['alive'][:, index]
                    length = int(mask.sum())
                    after = np.arange(length)*.01 >= 2.
                    features = arrays['features'][:length, index]
                    speed_error = float(np.abs(features[after, 4]).mean()) if after.any() else None
                    yaw_error = float(np.abs(features[after, 9]).mean()) if after.any() else None
                    height_error = float(np.abs(features[after, 0]-.33648).mean()) if after.any() else None
                    tracking = bool(after.any() and speed_error <= .30 and yaw_error <= .30 and height_error <= .03
                        and arrays['tilt'][:length, index][after].max() <= math.radians(25)
                        and arrays['base_contact'][:length, index][after].max() <= 1.)
                    final_balance = bool(time_out[index] and probe['hold'][index] >= 500)
                    finite = bool(np.isfinite(arrays['qpos'][:length, index]).all())
                    accepted = bool(final_balance and probe['strict_ever'][index] and tracking and finite)
                    drift = (np.linalg.norm(arrays['root'][:length, index, :2][after]
                        -arrays['root'][:length, index, :2][after][0], axis=-1).max() if after.any() else None)
                    rows.append(dict(env=index, pose_row=pose_rows[index],
                        leg_delay_s=DELAY_PAIRS[index//count][0], wheel_delay_s=DELAY_PAIRS[index//count][1],
                        duration_s=round(float(durations[index]), 2), time_out=bool(time_out[index]),
                        strict_rise=bool(probe['strict_ever'][index]), final_balance_v31=final_balance,
                        tracking_pass=tracking, candidate_qualified=accepted, finite=finite,
                        mean_abs_vx_error=speed_error, mean_abs_yaw_error=yaw_error,
                        mean_abs_height_error=height_error, final_height_m=float(features[-1, 0]),
                        max_post2s_drift_m=None if drift is None else float(drift),
                        zero_hold_5cm=bool(accepted and drift is not None and drift <= .05)))
                metadata = dict(seed=seed, num_envs=args.num_envs, sample_dt=.01,
                    checkpoint_sha256=checkpoint_hash, xml_sha256=report['xml_sha256'],
                    initial='smp-normal-ground-zero-velocity', optimizer_updates=0)
                np.savez_compressed(args.output/f'policy-{seed}.npz', **arrays,
                    initial_qpos=qpos.cpu().numpy(), initial_qvel=qvel.cpu().numpy(),
                    time=(np.arange(len(arrays['root']))+1)*.01, metadata=json.dumps(metadata))
                report['seeds'][str(seed)] = dict(rows=rows, candidates=sum(row['candidate_qualified'] for row in rows),
                    strict_rise=sum(row['strict_rise'] for row in rows), final_balance=sum(row['final_balance_v31'] for row in rows),
                    source_trace_sha256=digest(args.output/f'policy-{seed}.npz'))
                (args.output/'source-check.json').write_text(json.dumps(report, indent=2)+'\n')
                print(f'seed={seed} candidates={report["seeds"][str(seed)]["candidates"]}/{args.num_envs}', flush=True)
        for key, value in original_actor.items():
            torch.testing.assert_close(actor.state_dict()[key], value, atol=0, rtol=0)
        assert digest(args.checkpoint) == checkpoint_hash
        report.update(actor_unchanged=True, completed=True, source_sha256=digest(__file__))
        (args.output/'source-check.json').write_text(json.dumps(report, indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml', asdict(cfg))
        if args.video:
            save_video(raw.sim.mj_model, args.output, report)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--video', action='store_true', help='Save a 1x acceptance video from the first seed')
    parser.add_argument('--render-only', type=Path, help='Render an existing acceptance directory without rerunning physics')
    parser.add_argument('--num-envs', type=int, default=160)
    parser.add_argument('--seeds', default='42,43')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.render_only is not None:
        render_existing(args)
    elif args.checkpoint is None or args.output is None:
        parser.error('--checkpoint and --output are required unless --render-only is used')
    else:
        check(args)


if __name__ == '__main__':
    main()
