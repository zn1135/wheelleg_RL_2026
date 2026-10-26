#!/usr/bin/env python3
"""Render fixed case0 in all four frozen-BC conditions from recorded states."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from check_lqr_student_closed_loop import digest


def render(args):
    names=('nominal_mean','nominal_sample','random_mean','random_sample')
    labels=('名义条件 · 无动作噪声','名义条件 · 动作噪声0.05',
            '训练随机化 · 无动作噪声','训练随机化 · 动作噪声0.05')
    clips=[];rows=[];hashes=[]
    for name in names:
        p=args.run/name;report=json.loads((p/'result.json').read_text());assert digest(p/'trace.npz')==report['trace_sha256']
        row=report['rows'][args.env];rows.append(row);hashes.append(report['trace_sha256'])
        with np.load(p/'trace.npz') as d:
            n=int(d['alive'][:,args.env].sum());q=d['qpos'][:n,args.env].copy();v=d['qvel'][:n,args.env].copy();initial=d['initial_qpos'][args.env].copy()
        q[:,:2]-=initial[:2];initial[:2]=0;clips.append((q,v,initial))
    for q,v,initial in clips:np.testing.assert_array_equal(initial,clips[0][2])
    assert len({(r['group_id'],r['leg_delay_s'],r['wheel_delay_s']) for r in rows})==1
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1);renderer=None
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        data=mujoco.MjData(model);renderer=mujoco.Renderer(model,width=640,height=360)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
        path=args.output/'comparison.mp4'
        with imageio.get_writer(path,fps=60,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(1200):
                t=frame/60;canvas=Image.new('RGB',(1280,920),'#18212b');draw=ImageDraw.Draw(canvas)
                for i,(q,v,initial) in enumerate(clips):
                    x,y=(i%2)*640,(i//2)*460;tick=max(0,round(t/.01)-1);ended=tick>=len(q);tick=min(tick,len(q)-1)
                    data.qpos[:]=initial if frame==0 else q[tick];data.qvel[:]=0 if frame==0 or ended else v[tick]
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.2]
                    mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()),(x,y+100))
                    row=rows[i];status='已终止，之后画面定格' if ended else f'本回合 起立{int(row["strict_rise"])} 平衡{int(row["final_balance_v31"])} 停车{int(row["qualified"])}'
                    draw.text((x+10,y+7),labels[i],font=font,fill='white')
                    draw.text((x+10,y+36),f't={t:.2f}s · 位移{np.linalg.norm(data.qpos[:2]):.2f}m · case{args.env}',font=font,fill='white')
                    draw.text((x+10,y+65),status,font=font,fill='#ffaaaa' if ended else '#c7e4f3')
                writer.append_data(np.asarray(canvas))
        reader=imageio.get_reader(path);assert reader.count_frames()==1200
        for n in (0,600,1199):
            pixels=reader.get_data(n)
            if n==600:imageio.imwrite(args.output/'poster.png',pixels)
        reader.close()
        (args.output/'video.json').write_text(json.dumps(dict(frames=1200,fps=60,seconds=20,environment=args.env,
            trace_sha256=hashes,video_sha256=digest(path),renderer_sha256=digest(__file__),rows=rows,
            scope='Fixed matched case, actual saved states, uncut20s; explicitly freezes after termination. No simulation or learning.'),indent=2)+'\n')
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--env',type=int,default=0);render(p.parse_args())
