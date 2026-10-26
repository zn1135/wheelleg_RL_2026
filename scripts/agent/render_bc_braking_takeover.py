#!/usr/bin/env python3
"""Render actual full student/teacher histories; never splice or restore prefixes."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from check_lqr_student_closed_loop import digest
from check_bc_braking_takeover import MODES

NAMES=('学生独立运行','LQR保留旧停车参考','LQR重锁停车参考')


def render(args):
    r=json.loads((args.run/'result.json').read_text());assessment=json.loads((args.run/'assessment.json').read_text())
    assert digest(args.run/'trace.npz')==r['trace_sha256'] and digest(MODEL_XML)==r['xml_sha256']
    assert assessment['artifact_checks_passed']
    rows=[x for x in r['rows'] if x['mode']=='relock'];selected=[]
    for qualified in (True,False):
        row=next((x for x in rows if x['qualified']==qualified),None)
        if row is not None:selected.append(row['case'])
    n=r['base_cases'];ids=[case+m*n for case in selected for m in range(len(MODES))]
    with np.load(args.run/'trace.npz') as d:
        q=d['qpos'][:,ids].copy();v=d['qvel'][:,ids].copy();f=d['features'][:,ids].copy()
        initial=d['initial_qpos'][ids].copy();alive=d['alive'][:,ids].copy()
    q[:,:,:2]-=initial[None,:,:2];initial[:,:2]=0.
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1);renderer=None;movies=[]
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        renderer=mujoco.Renderer(model,width=640,height=360);data=mujoco.MjData(model)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',17)
        for item,case in enumerate(selected):
            movie=args.run/f'takeover-case{case}.mp4';frame_count=round(r['duration_s']*60)
            with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1,
                ffmpeg_params=['-movflags','+faststart']) as writer:
                for frame in range(frame_count):
                    t=frame/60;tick=min(len(q)-1,max(0,round(t/.01)-1));canvas=Image.new('RGB',(1920,500),'#18212b');draw=ImageDraw.Draw(canvas)
                    for panel,mode in enumerate(MODES):
                        j=item*len(MODES)+panel;row=r['rows'][case+panel*n]
                        length=int(alive[:,j].sum());at=min(tick,max(length-1,0))
                        data.qpos[:]=initial[j] if frame==0 else q[at,j];data.qvel[:]=0 if frame==0 else v[at,j]
                        camera.lookat[:]=[data.qpos[0],data.qpos[1],.2]
                        mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option)
                        x=640*panel;canvas.paste(Image.fromarray(renderer.render()),(x,140))
                        draw.text((x+10,6),NAMES[panel],font=font,fill='white')
                        owner='学生控制（各自实际轨迹）' if t<10 or panel==0 else '教师接管'
                        draw.text((x+10,32),f'{owner} | t={t:.2f}s | 延迟{row["leg_delay_s"]*1000:.0f}/{row["wheel_delay_s"]*1000:.0f}ms',font=font,fill='white')
                        draw.text((x+10,58),f'前向速度{f[at,j,4]:+.3f}m/s | yaw={f[at,j,9]:+.3f}rad/s',font=font,fill='white')
                        drift=np.linalg.norm(q[at,j,:2]-q[2499,j,:2])*100 if t>=25 else 0.
                        draw.text((x+10,84),f'总位移{np.linalg.norm(data.qpos[:2]):.2f}m | 末段位移{drift:.2f}cm',font=font,fill='white')
                        label='本回合停车合格' if row['qualified'] else '本回合停车未过'
                        if tick>=length:label='已终止后定格'
                        draw.text((x+10,110),label+' | 非同状态精确分叉',font=font,fill='white' if row['qualified'] else '#ffbbbb')
                    writer.append_data(np.asarray(canvas))
                    if frame%600==0:print(f'case={case} render={frame}/{frame_count}',flush=True)
            with imageio.get_reader(movie) as reader:
                assert reader.count_frames()==frame_count==1800
                for at in (0,600,900,1620,1799):assert reader.get_data(at).shape==(500,1920,3)
                imageio.imwrite(args.run/f'takeover-case{case}-27s.png',reader.get_data(1620))
            movies.append(dict(case=case,path=movie.name,sha256=digest(movie),frames=frame_count,fps=60,
                relock_qualified=r['rows'][case+2*n]['qualified'],source_envs=[case+m*n for m in range(len(MODES))]))
        (args.run/'videos.json').write_text(json.dumps(dict(movies=movies,trace_sha256=r['trace_sha256'],checkpoint_sha256=r['checkpoint_sha256'],
            renderer_sha256=digest(__file__),selection='First qualified relock case and first rejected relock case, each with same pose/delay template in the other modes.',
            prefix='Each panel shows its own actual0..30s; no common-prefix splice. Initial states match, prefix trajectories differ numerically.',
            display='Signed body forward velocity; negative means backward. Camera follows actual body; only world XY translation for display.'),indent=2)+'\n')
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);render(p.parse_args())
