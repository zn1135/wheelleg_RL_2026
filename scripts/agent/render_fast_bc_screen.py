#!/usr/bin/env python3
"""Recorded20s baseline versus low-noise frozen BC; no simulation or learning."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from check_lqr_student_closed_loop import digest


def render(args):
    reports=[json.loads((p/'result.json').read_text()) for p in (args.baseline,args.candidate)]
    clips=[]
    for p,r in zip((args.baseline,args.candidate),reports):
        assert digest(p/'trace.npz')==r['trace_sha256']
        with np.load(p/'trace.npz') as d:
            n=int(d['alive'][:,0].sum());q=d['qpos'][:n,0].copy();v=d['qvel'][:n,0].copy();initial=d['initial_qpos'][0].copy()
        q[:,:2]-=initial[:2];initial[:2]=0;clips.append((q,v,initial))
    np.testing.assert_array_equal(clips[0][2],clips[1][2])
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True);env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1);renderer=None
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        data=mujoco.MjData(model);renderer=mujoco.Renderer(model,width=640,height=360)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
        r=reports[1];titles=['冻结BC · 确定动作',f'冻结BC · 腿噪声{r["leg_std"]}／轮噪声{r["wheel_std"]}']
        movie=args.output/'comparison.mp4'
        with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(1200):
                t=frame/60;canvas=Image.new('RGB',(1280,460),'#18212b');draw=ImageDraw.Draw(canvas)
                for i,(q,v,initial) in enumerate(clips):
                    tick=max(0,round(t/.01)-1);ended=tick>=len(q);tick=min(tick,len(q)-1)
                    data.qpos[:]=initial if frame==0 else q[tick];data.qvel[:]=0 if frame==0 or ended else v[tick]
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.2];mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera=camera,scene_option=option);canvas.paste(Image.fromarray(renderer.render()),(i*640,100))
                    draw.text((i*640+10,7),titles[i],font=font,fill='white')
                    draw.text((i*640+10,36),f't={t:.2f}s · 位移{np.linalg.norm(data.qpos[:2]):.2f}m · case0',font=font,fill='white')
                    draw.text((i*640+10,65),'已终止，之后定格' if ended else '实际记录回放 · 参数更新0',font=font,fill='#c7e4f3')
                writer.append_data(np.asarray(canvas))
        reader=imageio.get_reader(movie);assert reader.count_frames()==1200
        for n in (0,600,1199):
            x=reader.get_data(n)
            if n==600:imageio.imwrite(args.output/'poster.png',x)
        reader.close()
        (args.output/'video.json').write_text(json.dumps(dict(frames=1200,fps=60,seconds=20,case=0,
            video_sha256=digest(movie),trace_sha256=[r['trace_sha256'] for r in reports],renderer_sha256=digest(__file__),
            scope='Actual fixed-case saved states, complete20s, no learning; nominal conditions only'),indent=2)+'\n')
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','candidate','output'):p.add_argument('--'+n,type=Path,required=True)
    render(p.parse_args())
