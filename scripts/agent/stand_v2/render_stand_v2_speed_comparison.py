#!/usr/bin/env python3
"""Render matched old/new frozen transient poses, never re-simulates physics."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from scripts.agent.stand.render_chuanliantui_stand import renderer_for
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[json.loads((p/'result.json').read_text()) for p in (args.before,args.after)]
    for d in reports:
        assert d['parameter_updates']==0 and d['qpos_recorded']
        assert digest(d['checkpoint'])==d['checkpoint_sha256']
    assert reports[0]['acceleration_m_s2']==reports[1]['acceleration_m_s2']
    with np.load(args.before/'initial.npz') as a,np.load(args.after/'initial.npz') as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    poses,traces=[],[]
    # Load compressed arrays once; reading the archive per frame repeats decompression.
    for p in (args.before,args.after):
        with np.load(p/'qpos.npz') as archive:
            poses.append({k:archive[k] for k in archive.files})
        with np.load(p/'trajectories.npz') as archive:
            traces.append({k:archive[k] for k in archive.files})
    models=[mujoco.MjModel.from_binary_path(str(p/'scene.mjb')) for p in (args.before,args.after)]
    renderers=[renderer_for(m,800,600) for m in models]
    videos=[]
    try:
        assert np.array_equal(poses[0]['time_s'],poses[1]['time_s'])
        assert all(d['qpos'].shape==(4001,128,models[k].nq) for k,d in enumerate(poses))
        assert all(np.isfinite(d['qpos']).all() for d in poses)
        for speed,name in ((.5,'forward'),(-.5,'backward')):
            old={r['env']:r for r in reports[0]['moving']['episodes'] if r['speed_m_s']==speed}
            new={r['env']:r for r in reports[1]['moving']['episodes'] if r['speed_m_s']==speed}
            eligible=[i for i in sorted(old) if old[i]['segments'][0]['complete_gate_and_live']
                and new[i]['segments'][0]['complete_gate_and_live']
                and all(np.isnan(p['first_end_s'][i]) for p in poses)]
            env=eligible[0] if eligible else min(old)
            outcome=[old[env]['passed'],new[env]['passed']]
            video=args.output/f'{name}-3999-vs-active4999.mp4'
            with imageio.get_writer(video,fps=30,codec='libx264',quality=8,macro_block_size=1) as writer:
                for k in range(1200):
                    t=k/30.
                    canvas=Image.new('RGB',(1600,600))
                    for col,((renderer,frame),pose,trace,label) in enumerate(zip(renderers,poses,traces,('3999','NEW active4999'))):
                        end=pose['first_end_s'][env]
                        ended=np.isfinite(end) and t>=end
                        case_t=min(t,float(end)) if ended else t
                        index=max(0,int(np.searchsorted(pose['time_s'],case_t,side='right'))-1)
                        physical=max(0,min(round(float(pose['time_s'][index])/.002)-1,19999))
                        u,target,vx,gate=trace['command'][physical,env]
                        phase=('GETUP' if t<8 else 'START' if t<14 else 'HOLD' if t<18 else
                               'STOP1' if t<24 else 'START' if t<30 else 'REVERSE' if t<36 else 'STOP2')
                        status='ENDED/FROZEN' if ended else phase
                        text=(f'{label} env{env} t={t:.2f}s {status} | u={u:+.2f} vx={vx:+.2f} | '
                              f'40s {"PASS" if outcome[col] else "FAIL"}')
                        canvas.paste(frame(pose['qpos'][index,env],text),(col*800,0))
                    writer.append_data(np.asarray(canvas))
                    if k in (0,300,600,960,1140):canvas.save(args.output/f'{name}-frame-{k:04d}.png')
            reader=imageio.get_reader(video)
            try:
                decoded=0
                for pixels in reader:
                    assert pixels.shape==(600,1600,3)
                    decoded+=1
            finally:reader.close()
            assert decoded==1200
            videos.append(dict(video=str(video.resolve()),sha256=digest(video),fps=30,frames=1200,
                decoded_frames=decoded,seconds=40,env=env,initial_signed_target=speed,
                old_pipeline_pass=outcome[0],new_pipeline_pass=outcome[1],
                selection='Smallest common gate-complete first drive and full survival; else smallest speed-case ID',
                eligible_envs=eligible))
        record=dict(videos=videos,parameter_updates=0,physics_reintegration=False,
            checkpoint_sha256=[d['checkpoint_sha256'] for d in reports],
            source_pose_sha256=[digest(p/'qpos.npz') for p in (args.before,args.after)],
            scope='Exact saved100Hz poses, original40s speed, same initial case; examples are not aggregate success rates')
        (args.output/'video.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))
    finally:
        for renderer,_ in renderers:renderer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
