"""Diagnose a frozen walk policy over 64 nominal first episodes and render fixed samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--seconds', type=float, default=20.)
parser.add_argument('--no-video', action='store_true', help='Keep trajectory/summary only for a short interface smoke')
args = parser.parse_args()
OUT, CHECKPOINT = args.output.resolve(), args.checkpoint.resolve()
OUT.mkdir(parents=True, exist_ok=False)
# 训练快照以原位置推导资产路径；先核对当前源码逐份相同，再用原位置加载。
SNAPSHOT = CHECKPOINT.parent/'snapshot'
assert (SNAPSHOT/'ct_mjlab').is_dir()
for source in (SNAPSHOT/'ct_mjlab').rglob('*.py'):
    current = ROOT/'mjlab_training/src/ct_mjlab'/source.relative_to(SNAPSHOT/'ct_mjlab')
    assert current.read_bytes() == source.read_bytes(), f'Source differs: {current}'
sys.path[:0] = [str(ROOT/'mjlab_training/src'), str(ROOT/'scripts/agent')]

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
import torch
from ct_mjlab.envs import task_registry
from scripts.agent.stand.render_chuanliantui_stand import renderer_for


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    task, count, seconds, seed = 'chuanliantui_walk', args.num_envs, args.seconds, 43
    assert count >= 4 and seconds >= 2.
    cfg, agent = task_registry.get_cfgs(task, play=True)
    cfg.auto_reset, cfg.seed = False, seed
    from ct_mjlab.envs.chuanliantui_stand_v2.action import StandV2PdAction
    assert cfg.actions['pd'].class_type is StandV2PdAction
    wrapped, _ = task_registry.make_env(task, env_cfg=cfg, train_cfg=agent,
                                      num_envs=count, device='cuda:0')
    raw = wrapped.unwrapped
    frames, records = [], []
    live = np.ones(count, dtype=bool)
    ended = np.full(count, seconds)
    reasons = ['duration']*count
    names = ['height', 'pitch_deg', 'roll_deg', 'tilt_deg', 'vx', 'yaw_rate',
             'cmd_vx', 'cmd_yaw', 'cmd_height', 'request_vx', 'request_yaw',
             'yaw_ready', 'tracking_active', 'position_locked', 'position_forward',
             'position_lateral', 'position_correction', 'heading_hold', 'heading_error_deg',
             'x', 'y', 'heading', 'has_stood', 'base_force', 'left_support', 'right_support']
    try:
        runner, _ = task_registry.make_alg_runner(wrapped, task, train_cfg=agent, device='cuda:0')
        runner.load(str(CHECKPOINT), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        before = {k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
        state = raw.command_manager.get_term('standup')
        state.speed_curriculum_frozen = True
        with torch.inference_mode():
            obs, _ = raw.reset(seed=seed)
            assert not state.ground_start.any()
            episode_height = state.command[:, 2].cpu().numpy().copy()
            movement = state.tracking_episode.cpu().numpy().copy()
            # 录像ID在推演前确定，不根据成功与否挑样本。
            static_ids = np.flatnonzero(~movement).tolist()
            moving_ids = np.flatnonzero(movement).tolist()
            video_ids = (static_ids[:1]+moving_ids[:3])
            assert len(video_ids) == 4
            mujoco.mj_saveModel(raw.sim.mj_model, str(OUT/'scene.mjb'), None)

            def capture():
                data = raw.scene['robot'].data
                g = data.projected_gravity_b
                pitch = torch.rad2deg(torch.asin(g[:, 0].clamp(-1., 1.)))
                roll = torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2]))
                tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1., 1.)))
                xy = data.root_link_pos_w[:, :2]-raw.scene.env_origins[:, :2]
                error = torch.atan2(torch.sin(state.heading_target-data.heading_w),
                                   torch.cos(state.heading_target-data.heading_w))
                support = -raw.scene['wheel_contact'].data.force[:, :, 2]
                base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
                row = torch.stack((data.root_link_pos_w[:, 2]-raw.scene.env_origins[:, 2],
                    pitch, roll, tilt, data.root_link_lin_vel_b[:, 0], data.root_link_ang_vel_b[:, 2],
                    state.command[:, 0], state.command[:, 1], state.command[:, 2],
                    state.tracking_velocity, state.tracking_yaw_rate, state.yaw_active,
                    state.tracking_active, state.position_locked, state.position_error_forward,
                    state.position_error_lateral, state.position_velocity, state.heading_hold,
                    torch.rad2deg(error), xy[:, 0], xy[:, 1], data.heading_w, state.has_stood,
                    base, support[:, 0], support[:, 1]), -1).cpu().numpy()
                pose = raw.sim.data.qpos.cpu().numpy().copy()
                assert np.isfinite(pose).all() and np.isfinite(row).all()
                if frames:
                    pose[~live], row[~live] = frames[-1][~live], records[-1][~live]
                frames.append(pose)
                records.append(row)

            capture()
            assert ((records[0][:,0]>.299)&(records[0][:,0]<.32)).all()
            for tick in range(round(seconds/raw.step_dt)):
                action = policy(obs)
                assert torch.isfinite(action).all()
                if agent.clip_actions is not None:
                    action = action.clamp(-agent.clip_actions, agent.clip_actions)
                obs, _, terminated, timeout, _ = raw.step(action)
                capture()
                done = (terminated | timeout).cpu().numpy()
                for i in np.flatnonzero(live & done):
                    ended[i] = (tick+1)*raw.step_dt
                    reasons[i] = 'failure' if terminated[i] else 'timeout'
                live &= ~done
                if tick+1 < round(seconds/raw.step_dt) and done.any():
                    raw.reset(env_ids=(terminated | timeout).nonzero(as_tuple=False).flatten())
                    obs = raw.get_observations()
                if (tick+1) % 500 == 0:
                    print(f'recorded {(tick+1)*raw.step_dt:.0f}/{seconds:.0f}s', flush=True)
        assert all(torch.equal(v, runner.alg.actor.state_dict()[k]) for k,v in before.items())
        speed_stages = {name: value['stage'] for name, value in state.direction_curriculum.items()}
        dt = raw.step_dt
    finally:
        raw.close()

    values, poses = np.asarray(records), np.asarray(frames)
    times = np.arange(len(values))*dt
    valid = times[:,None] <= ended[None,:]+1e-7
    warm = valid & (times[:,None] >= 2.)
    col = {n:values[:,:,i] for i,n in enumerate(names)}

    def stats(value, mask):
        x = value[mask]
        return None if not x.size else dict(samples=int(x.size), mean=float(x.mean()),
            mean_abs=float(np.abs(x).mean()), p95_abs=float(np.quantile(np.abs(x), .95)),
            max_abs=float(np.abs(x).max()))

    straight = warm & (np.abs(col['request_vx']) > .05)
    spin = warm & (np.abs(col['request_yaw']) > .05)
    # 稳态：已跟上限斜率的目标，并连续保持0.5秒后才统计。
    settle = np.zeros(count, dtype=int)
    straight_settled = np.zeros_like(valid)
    spin_settled = np.zeros_like(valid)
    for k in range(len(values)):
        reached = ((np.abs(col['cmd_vx'][k]-col['request_vx'][k]) < 1e-5) & straight[k]
                   | (np.abs(col['cmd_yaw'][k]-col['request_yaw'][k]) < 1e-5) & spin[k])
        settle = np.where(reached, settle+1, 0)
        straight_settled[k] = straight[k] & (settle > round(.5/dt))
        spin_settled[k] = spin[k] & (settle > round(.5/dt))
    parked = warm & (col['position_locked'] > .5)
    holding = warm & (col['heading_hold'] > .5)
    last = valid & (times[:,None] >= seconds-5.)
    report = dict(task=task, checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT),
        seed=seed, num_envs=count, duration_s=seconds, policy_dt_s=dt,
        parameter_updates=0, actor_parameters_unchanged=True, nominal_play=True,
        training_initial_nominal_height_m=.30, evaluation_initial_nominal_height_m=.30,
        initial_physical_height_m=values[0,:,0].tolist(),
        all_near_standing_start=True, speed_curriculum_stage=speed_stages,
        initial_target_height_m=episode_height.tolist(), movement_episode=movement.tolist(),
        episode_end_s=ended.tolist(), episode_end_reason=reasons, failed_count=reasons.count('failure'),
        surviving_count=int(live.sum()), natural_nonzero_vx_envs=int(straight.any(0).sum()),
        natural_nonzero_yaw_envs=int(spin.any(0).sum()), yaw_ready_envs=int((col['yaw_ready']>.5).any(0).sum()),
        video_env_ids=video_ids, selection='first static ID and first three movement IDs, selected before rollout',
        metrics=dict(pitch_deg_last5=stats(col['pitch_deg'],last), roll_deg_last5=stats(col['roll_deg'],last),
            tilt_deg_last5=stats(col['tilt_deg'],last), height_error_m_last5=stats(col['height']-col['cmd_height'],last),
            straight_vx_error=stats(col['vx']-col['cmd_vx'],straight),
            straight_settled_vx_error=stats(col['vx']-col['cmd_vx'],straight_settled),
            forward_vx_error=stats(col['vx']-col['cmd_vx'],straight & (col['request_vx'] > 0.)),
            backward_vx_error=stats(col['vx']-col['cmd_vx'],straight & (col['request_vx'] < 0.)),
            forward_settled_vx_error=stats(col['vx']-col['cmd_vx'],straight_settled & (col['request_vx'] > 0.)),
            backward_settled_vx_error=stats(col['vx']-col['cmd_vx'],straight_settled & (col['request_vx'] < 0.)),
            spin_yaw_error=stats(col['yaw_rate']-col['cmd_yaw'],spin),
            spin_settled_yaw_error=stats(col['yaw_rate']-col['cmd_yaw'],spin_settled),
            parking_forward_error=stats(col['position_forward'],parked),
            parking_lateral_error=stats(col['position_lateral'],parked),
            parking_correction=stats(col['position_correction'],parked),
            heading_error_deg=stats(col['heading_error_deg'],holding)),
        scope=f'{seconds:g}s nominal first-episode diagnostic; excludes resets; not formal acceptance, randomized training or CPU validation')
    np.savez_compressed(OUT/'trajectory.npz', qpos=poses, time_s=times, metrics=values,
                        metric_names=np.asarray(names), valid=valid, initial_target_height_m=episode_height)
    report['trajectory_sha256'] = digest(OUT/'trajectory.npz')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['metrics'],indent=2),flush=True)

    if args.no_video:return
    model = mujoco.MjModel.from_binary_path(str(OUT/'scene.mjb'))
    renderer, render = renderer_for(model, 640, 480)
    video = OUT/'walk-policy-20s.mp4'
    try:
        with imageio.get_writer(video, fps=30, codec='libx264', quality=8,
                                macro_block_size=1, pixelformat='yuv420p') as writer:
            for k in range(round(seconds*30)):
                t = k/30
                index = min(int(np.searchsorted(times,t)),len(times)-1)
                sheet = Image.new('RGB',(1280,960))
                for slot,i in enumerate(video_ids):
                    status = 'FAIL' if t >= ended[i] and reasons[i] == 'failure' else 'FIRST'
                    text = (f'{CHECKPOINT.stem} env{i} t={t:04.1f}s {status} pitch={col["pitch_deg"][index,i]:+.1f} '
                        f'v={col["vx"][index,i]:+.2f}/{col["cmd_vx"][index,i]:+.2f} '
                        f'w={col["yaw_rate"][index,i]:+.2f}/{col["cmd_yaw"][index,i]:+.2f}')
                    sheet.paste(render(poses[index,i],text),((slot%2)*640,(slot//2)*480))
                writer.append_data(np.asarray(sheet))
                if k in (0,150,300,570):
                    sheet.save(OUT/f'frame-{k:04d}.png')
                if (k+1)%150==0:
                    print(f'rendered {k+1}/600',flush=True)
    finally:
        renderer.close()
    with imageio.get_reader(video) as reader:
        meta = reader.get_meta_data()
        decoded = sum(1 for _ in reader)
    assert decoded == round(seconds*30) and meta['size'] == (1280,960) and meta['fps'] == 30
    (OUT/'video.json').write_text(json.dumps(dict(video=str(video),sha256=digest(video),
        fps=30,decoded_frames=decoded,duration_s=seconds,resolution=[1280,960],original_speed=True,
        checkpoint_sha256=report['checkpoint_sha256'],trajectory_sha256=report['trajectory_sha256'],
        env_ids=video_ids,parameter_updates=0),indent=2)+'\n')


if __name__ == '__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Evaluation cannot train')):
        main()
