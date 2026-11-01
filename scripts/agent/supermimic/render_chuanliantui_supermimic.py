#!/usr/bin/env python3
"""Draw saved first-episode physics poses as a 30 s, real-time four-panel video.

No physics integration, policy inference or learning is performed. MuJoCo EGL
draws the saved scene/qpos through render_chuanliantui_stand.renderer_for;
mj_forward inside that helper only recomputes geometry for a recorded pose.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SECONDS = 30.
DEFAULT_ENVS = (0, 16, 32, 80)
WIDTH, HEIGHT, FOOTER = 640, 360, 64


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_measurement(directory, env_ids):
    directory = directory.resolve()
    summary_path = directory/'summary.json'
    summary = json.loads(summary_path.read_text())
    checkpoint = Path(summary['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = directory/checkpoint
    paths = dict(summary=summary_path, checkpoint=checkpoint,
                 scene=directory/'scene.mjb', trajectory=directory/'trajectory.npz')
    hashes = {name:digest(path) for name,path in paths.items()}
    for name in ('checkpoint', 'scene', 'trajectory'):
        if hashes[name] != summary[f'{name}_sha256']:
            raise ValueError(f'{name} hash differs from the measurement summary')
    if summary['seconds'] < SECONDS-1e-7:
        raise ValueError('A complete 30-second measurement is required')
    if summary['physics_dt_s'] != .002 or summary['policy_dt_s'] != .01:
        raise ValueError('Expected the recorded 500 Hz physics / 100 Hz policy timing')
    if len(env_ids) != 4 or len(set(env_ids)) != 4 or min(env_ids) < 0 or max(env_ids) >= summary['num_envs']:
        raise ValueError('Select four distinct environment IDs present in the measurement')
    rows = {row['env']:row for row in summary['summary']['per_environment']}
    cases = [rows[env] for env in env_ids]
    layout = summary['layout']
    columns = [layout['valid'][0], layout['true_velocity'][0], layout['roll_pitch'][0]+1]
    with np.load(paths['trajectory'], allow_pickle=False) as archive:
        # Select four environments before keeping the large arrays in memory.
        poses = archive['qpos'][:, env_ids].copy()
        live = archive['qpos_live'][:, env_ids].copy().astype(bool)
        pose_times = archive['qpos_time_s'].copy()
        metrics = archive['values'][:, env_ids][:, :, columns].copy()
        metric_times = archive['time_s'].copy()
        requests = archive['requested_vx'][:, env_ids].copy()
        policy_times = archive['policy_time_s'].copy()
    if poses.ndim != 3 or poses.shape[:2] != live.shape or poses.shape[1] != 4:
        raise ValueError('Expected qpos[time,env,nq] and matching qpos_live')
    for times, count in ((pose_times,len(poses)), (metric_times,len(metrics)),
                         (policy_times,len(requests))):
        if times.shape != (count,) or not np.isfinite(times).all() or not (np.diff(times) > 0.).all():
            raise ValueError('Recorded time arrays must be finite and strictly increasing')
    if pose_times[0] > 1e-9 or pose_times[-1] < SECONDS-1e-7:
        raise ValueError('Saved qpos does not cover t=0 through t=30 seconds')
    if (not np.allclose(np.diff(pose_times),.002,atol=1e-8,rtol=0.)
            or not np.allclose(np.diff(metric_times),.002,atol=1e-8,rtol=0.)
            or metric_times[-1] < SECONDS-1e-7
            or policy_times[0] > 1e-9 or policy_times[-1] < SECONDS-.01-1e-7):
        raise ValueError('Recorded physics/command timestamps do not cover the planned video')
    if np.any(np.diff(live.astype(np.int8),axis=0) > 0):
        raise ValueError('qpos_live reactivates after the first episode ended')
    if not np.isfinite(requests).all():
        raise ValueError('Requested velocity history is nonfinite')
    poses.flags.writeable = False
    return dict(directory=directory, summary=summary, paths=paths, hashes=hashes,
        env_ids=env_ids, cases=cases, poses=poses, live=live, pose_times=pose_times,
        metrics=metrics, metric_times=metric_times, requests=requests, policy_times=policy_times)


def frame_plan(record, fps):
    """Choose only finite first-episode samples at or before each video timestamp."""
    times = np.arange(round(SECONDS*fps), dtype=np.float64)/fps
    pose_indices = np.empty((len(times),4), dtype=np.int64)
    metric_indices = np.empty_like(pose_indices)
    frozen = np.empty_like(pose_indices, dtype=bool)
    tails = []
    for column,row in enumerate(record['cases']):
        end = float(row['episode_end_s'])
        if not math.isfinite(end) or end < 0.:
            raise ValueError('Invalid first-episode end time')
        valid = (record['live'][:,column] & np.isfinite(record['poses'][:,column]).all(-1)
                 & (record['pose_times'] <= end+1e-8))
        candidates = np.flatnonzero(valid)
        if not len(candidates) or record['pose_times'][candidates[0]] > 1e-9:
            raise ValueError(f"env {row['env']} has no finite initial first-episode pose")
        capped = np.minimum(times,end)
        selected = np.searchsorted(record['pose_times'][candidates],capped+1e-9,side='right')-1
        pose_indices[:,column] = candidates[np.maximum(selected,0)]
        source_times = record['pose_times'][pose_indices[:,column]]
        finite_metrics = ((record['metrics'][:,column,0] > .5)
                          & np.isfinite(record['metrics'][:,column]).all(-1)
                          & (record['metric_times'] <= end+1e-8))
        metric_candidates = np.flatnonzero(finite_metrics)
        metric_indices[:,column] = -1
        if len(metric_candidates):
            selected = np.searchsorted(record['metric_times'][metric_candidates],source_times+1e-9,side='right')-1
            present = selected >= 0
            metric_indices[present,column] = metric_candidates[selected[present]]
        frozen[:,column] = times >= end-1e-9
        if (record['pose_times'][pose_indices[:,column]] > capped+1e-8).any():
            raise AssertionError('Rendering would show a future or post-episode pose')
        if not record['live'][pose_indices[:,column],column].all():
            raise AssertionError('Rendering would show a reset episode')
        tails.append(dict(env=row['env'], case=row['case'], first_episode_end_s=end,
            end_reason=row['episode_end_reason'], last_finite_first_episode_pose_s=float(
                record['pose_times'][candidates[-1]]), frozen_video_frames=int(frozen[:,column].sum())))
    requested = np.searchsorted(record['policy_times'],times+1e-9,side='right')-1
    requested = requested.clip(0,len(record['policy_times'])-1)
    return dict(times=times,pose_indices=pose_indices,metric_indices=metric_indices,
                frozen=frozen,request_indices=requested,tails=tails)


def precheck(record, plan, fps):
    return dict(measurement=str(record['directory']), env_ids=list(record['env_ids']),
        default_envs_preselected_before_final_measurement=list(DEFAULT_ENVS), selection='explicit' if tuple(record['env_ids']) != DEFAULT_ENVS else 'preselected defaults',
        hashes_verified=True, input_sha256=record['hashes'], planned_video_seconds=SECONDS,
        fps=fps, frames=len(plan['times']), first_episode_only=True,
        pose_selection='latest finite qpos_live sample at/before min(video_t, episode_end_s)',
        tails=plan['tails'], scope='CPU input/hash/index check only; no renderer or physics integration')


def draw_video(record, plan, output, fps):
    import imageio.v2 as imageio
    import mujoco
    from scripts.agent.shared.rendering import renderer_for

    model = mujoco.MjModel.from_binary_path(str(record['paths']['scene']))
    if record['poses'].shape[-1] != model.nq:
        raise ValueError('Saved qpos dimension differs from scene.mjb')
    renderer, frame = renderer_for(model, WIDTH, HEIGHT)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        font = ImageFont.load_default()
    video = output/'replay.mp4'
    preview_at = 3*fps
    storyboard_at = {round(t*fps) for t in (0.,2.,6.,12.,20.,SECONDS-1/fps)}
    storyboard = []
    try:
        with imageio.get_writer(video,fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
            for index,time in enumerate(plan['times']):
                canvas = Image.new('RGB',(WIDTH*2,HEIGHT*2+FOOTER),'#121a21')
                for column,row in enumerate(record['cases']):
                    pose_index = plan['pose_indices'][index,column]
                    tile = frame(record['poses'][pose_index,column],'')
                    draw = ImageDraw.Draw(tile)
                    ended = bool(plan['frozen'][index,column])
                    draw.rectangle((0,0,WIDTH,74),fill='#6b2525' if ended else '#19232c')
                    status = 'ENDED/FROZEN' if ended else 'RUNNING'
                    metric_index = plan['metric_indices'][index,column]
                    if metric_index >= 0:
                        _,vx,pitch = record['metrics'][metric_index,column]
                        actual = f'actual vx={vx:+.3f} m/s | pitch={pitch:+.2f} deg'
                    else:
                        actual = 'actual vx=n/a | pitch=n/a (initial pose)'
                    request = record['requests'][plan['request_indices'][index],column]
                    labels = (f"env {row['env']:03d} | {row['case']} | {status}",
                        f"t={time:05.2f}s | pose={record['pose_times'][pose_index]:06.3f}s | end={row['episode_end_s']:.2f}s ({row['episode_end_reason']})",
                        f'planned req={request:+.2f} m/s | {actual}')
                    for line,label in enumerate(labels):
                        draw.text((8,5+22*line),label,font=font,fill='white')
                    canvas.paste(tile,((column%2)*WIDTH,(column//2)*HEIGHT))
                draw = ImageDraw.Draw(canvas)
                labels = ('Saved 500 Hz mjlab physics poses | MuJoCo EGL drawing | no reintegration | not native RTX',
                    'First episode only. ENDED/FROZEN retains its last finite pose; planned request keeps its original timeline.',
                    f'1x real time | {fps} fps | video t={time:05.2f}s | env IDs: '+', '.join(map(str,record['env_ids'])))
                for line,label in enumerate(labels):
                    draw.text((8,HEIGHT*2+3+20*line),label,font=font,fill='white')
                writer.append_data(np.asarray(canvas))
                if index == preview_at:
                    canvas.save(output/'preview.png')
                if index in storyboard_at:
                    storyboard.append(canvas.resize((WIDTH,(HEIGHT*2+FOOTER)//2),Image.Resampling.LANCZOS))
    finally:
        renderer.close()
    sheet = Image.new('RGB',(WIDTH*3,(HEIGHT*2+FOOTER)), '#121a21')
    for index,tile in enumerate(storyboard):
        sheet.paste(tile,((index%3)*WIDTH,(index//3)*tile.height))
    sheet.save(output/'storyboard.png')
    reader = imageio.get_reader(video)
    try:
        metadata = reader.get_meta_data()
        count, first, last = 0, None, None
        for array in reader:
            if array.shape[:2] != (HEIGHT*2+FOOTER,WIDTH*2) or array.std() <= 1.:
                raise AssertionError('Decoded video frame has an unexpected size or is blank')
            frame_hash = hashlib.sha256(array.tobytes()).hexdigest()
            first = frame_hash if first is None else first
            last = frame_hash
            count += 1
    finally:
        reader.close()
    if count != len(plan['times']) or abs(float(metadata['fps'])-fps) > 1e-6 or abs(count/fps-SECONDS) > 1e-8:
        raise AssertionError('Decoded video frame count, fps or duration differs from the plan')
    if first == last:
        raise AssertionError('Decoded first and last video frames are identical')
    for name,path in record['paths'].items():
        if digest(path) != record['hashes'][name]:
            raise AssertionError(f'{name} changed during rendering')
    result = precheck(record,plan,fps)
    result.update(scope='Draw saved real-physics qpos only; no policy inference, integration or learning; not native RTX',
        renderer='render_chuanliantui_stand.renderer_for / MuJoCo EGL',physics_steps_executed=0,
        saved_qpos_read_only=not record['poses'].flags.writeable,
        renderer_script_sha256=digest(Path(__file__)),
        renderer_helper_sha256=digest(Path(__file__).resolve().parents[3] / 'scripts/agent/shared/rendering.py'),
        policy_calls=0,parameter_updates=0,decoded_frames=count,decoded_fps=float(metadata['fps']),
        decoded_duration_s=count/fps,decoded_first_frame_sha256=first,decoded_last_frame_sha256=last,
        video=str(video), output_sha256={name:digest(output/name) for name in ('replay.mp4','preview.png','storyboard.png')},
        metrics='actual vx is recorded LINK-body x velocity; pitch is recorded degrees; initial metrics unavailable',
        command_label='planned request from the original measurement timeline; frozen actual metrics remain first-episode data')
    write(output/'video.json',result)
    print(json.dumps(result,ensure_ascii=False))


def selfcheck():
    # Reset poses are deliberately huge; no selected frame may use one.
    times = np.arange(151)*.2
    poses = np.zeros((151,4,7));live = np.ones((151,4),dtype=bool)
    cases = []
    for column,end in enumerate((30.,1.,2.,4.)):
        live[times > end+1e-8,column] = False
        poses[~live[:,column],column] = 999.
        cases.append(dict(env=DEFAULT_ENVS[column],case='synthetic',episode_end_s=end,
                          episode_end_reason='duration' if end==30. else 'failure'))
    # A nonfinite terminal pose must fall back to the preceding finite live pose.
    poses[20,3] = np.nan
    metrics = np.ones((150,4,3))
    record = dict(cases=cases,poses=poses,live=live,pose_times=times,metrics=metrics,
        metric_times=times[1:],policy_times=np.arange(3000)*.01)
    for fps in (25,30):
        plan=frame_plan(record,fps)
        assert len(plan['times'])==30*fps
        for column in range(4):
            selected=poses[plan['pose_indices'][:,column],column]
            assert np.isfinite(selected).all() and not (selected==999.).any()
            frozen=plan['frozen'][:,column]
            if frozen.any():
                assert (selected[frozen]==selected[frozen][0]).all()
        assert math.isclose(plan['tails'][3]['last_finite_first_episode_pose_s'],3.8,abs_tol=1e-9)
    print(json.dumps(dict(passed=True,fps=[25,30],seconds=30,checks=[
        'preselected IDs','no reset qpos','failure freezes first episode','nonfinite terminal fallback',
        '750/900 frames at real-time timestamps'],scope='Synthetic CPU frame-selection check; no renderer')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--env-ids',type=int,nargs=4,default=DEFAULT_ENVS)
    parser.add_argument('--fps',type=int,choices=(25,30),default=25)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--selfcheck',action='store_true')
    args=parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.measurement is None or (not args.check_only and args.output is None):
        parser.error('--measurement is required, plus --output for rendering')
    record=load_measurement(args.measurement,args.env_ids)
    plan=frame_plan(record,args.fps)
    if args.check_only:
        print(json.dumps(precheck(record,plan,args.fps),ensure_ascii=False))
        return
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    draw_video(record,plan,args.output,args.fps)


if __name__=='__main__':
    main()
