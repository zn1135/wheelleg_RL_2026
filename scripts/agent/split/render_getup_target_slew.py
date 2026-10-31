#!/usr/bin/env python3
"""Render saved baseline/candidate getup first episodes, without new simulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from scripts.agent.stand.render_chuanliantui_stand import renderer_for


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.baseline, args.candidate]
    reports = [json.loads((p/'result.json').read_text()) for p in sources]
    assert all(r['seconds'] == 40. and r['num_envs'] == 4 and r['parameter_updates'] == 0 for r in reports)
    assert reports[0]['rate_rad_s'] == 0. and reports[0]['checkpoint_sha256'] == reports[1]['checkpoint_sha256']
    arrays = []
    for folder, report in zip(sources, reports):
        assert digest(folder/'trajectory.npz') == report['trajectory_sha256']
        with np.load(folder/'trajectory.npz') as q, np.load(folder/'trace.npz') as trace:
            arrays.append((q['qpos'], q['time_s'], trace['values'], list(trace['fields'])))
    assert np.array_equal(arrays[0][0][0], arrays[1][0][0])
    assert np.array_equal(arrays[0][1], arrays[1][1])
    models = [mujoco.MjModel.from_binary_path(str(p/'scene.mjb')) for p in sources]
    for field in ('body_mass','body_inertia','jnt_range','qpos0','geom_pos','geom_size','actuator_gainprm','actuator_biasprm','actuator_gear'):
        assert np.array_equal(getattr(models[0],field),getattr(models[1],field))
    renderer, frame = renderer_for(models[0], 480, 360)
    video = args.output/'baseline-vs-target-slew-40s.mp4'
    try:
        with imageio.get_writer(video, fps=30, codec='libx264', quality=8, macro_block_size=1, pixelformat='yuv420p') as writer:
            for k in range(1200):
                seconds = k/30.
                canvas = Image.new('RGB',(1920,720))
                for row,(report,(poses,times,values,fields)) in enumerate(zip(reports, arrays)):
                    for env in range(4):
                        result = report['rows'][env]
                        q_index = min(int(np.searchsorted(times,seconds)),len(times)-1)
                        measured_time = min(seconds,result['end_s']-.002)
                        v_index = min(max(round(measured_time/.002),0),len(values)-1)
                        h, tilt = values[v_index,env,[fields.index('height_m'),fields.index('tilt_deg')]]
                        label = 'BASE' if row==0 else f"SLEW {report['rate_rad_s']:g}rad/s"
                        status = ('END '+result['end_reason']) if seconds>=result['end_s'] else 'FIRST EPISODE'
                        canvas.paste(frame(poses[q_index,env],f'{label} env{env} t={seconds:04.1f}s {status} h={h:.2f} tilt={tilt:.0f}'),(env*480,row*360))
                writer.append_data(np.asarray(canvas))
                if k in (0,150,600,1170):
                    canvas.save(args.output/f'frame-{k:04d}.png')
                if (k+1)%300==0:
                    print(f'rendered {k+1}/1200',flush=True)
    finally:
        renderer.close()
    with imageio.get_reader(video) as reader:
        meta = reader.get_meta_data()
        count = sum(1 for _ in reader)
    assert count==1200 and meta['size']==(1920,720) and meta['fps']==30
    result=dict(video=str(video.resolve()),video_sha256=digest(video),decoded_frames=count,
        duration_s=40,fps=30,resolution=[1920,720],original_speed=True,parameter_updates=0,
        source_directories=[str(p.resolve()) for p in sources],
        source_trajectory_sha256=[r['trajectory_sha256'] for r in reports],
        definition='Top row original PD, bottom row target slew; each column one same-initial-state environment. Failed first episodes remain frozen.',
        source_sha256=digest(Path(__file__)))
    (args.output/'video.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
