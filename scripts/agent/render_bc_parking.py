#!/usr/bin/env python3
"""Render the common student prefix and four real parking continuations at60fps."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from check_lqr_student_closed_loop import digest

MODES=('student-restored','warm','relock','balance')
NAMES=('恢复后学生继续控制','原教师历史接管','重新锁定停车参考','进入平衡并锁定参考')


def render(args):
    out=args.run;reports={m:json.loads((out/m/'result.json').read_text()) for m in MODES}
    traces={m:np.load(out/m/'trace.npz') for m in MODES}
    for m in MODES:assert digest(out/m/'trace.npz')==reports[m]['trace_sha256']
    with np.load(out/'student-paired/switch-check.npz') as d:invalid=np.flatnonzero(d['teacher_phase']!=3)
    selected=[0]+([int(invalid[0])] if len(invalid) and invalid[0]!=0 else [])
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True);env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1)
    renderer=None;movies=[]
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        renderer=mujoco.Renderer(model,width=640,height=360);data=mujoco.MjData(model);camera=mujoco.MjvCamera()
        camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
        b=np.load(out/'student-paired/trace.npz')
        for index in selected:
            frames=[]
            for mode,d in traces.items():
                row=reports[mode]['rows'][index];length=round(row['observed_s']/.01)
                q=d['qpos'][:length,index].copy();v=d['qvel'][:length,index].copy();f=d['features'][:length,index].copy()
                phase=d['teacher_phase'][:length,index].copy()
                q[:1000]=b['qpos'][:1000,index];v[:1000]=b['qvel'][:1000,index]
                f[:1000]=b['features'][:1000,index];phase[:1000]=b['teacher_phase'][:1000,index]
                initial=b['initial_qpos'][index].copy();q[:,:2]-=initial[:2];initial[:2]=0
                frames.append((q,v,f,phase,initial))
            movie=out/f'parking-restored-env{index}.mp4'
            with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
                for frame in range(1800):
                    t=frame/60;canvas=Image.new('RGB',(1280,920),'#18212b');draw=ImageDraw.Draw(canvas)
                    for panel,(q,v,f,phase,initial) in enumerate(frames):
                        tick=max(0,round(t/.01)-1);ended=tick>=len(q);tick=min(tick,len(q)-1)
                        data.qpos[:]=initial if frame==0 else q[tick];data.qvel[:]=0 if frame==0 or ended else v[tick]
                        camera.lookat[:]=[data.qpos[0],data.qpos[1],.2]
                        mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option)
                        x=640*(panel%2);y=460*(panel//2);canvas.paste(Image.fromarray(renderer.render()),(x,y+100))
                        draw.text((x+12,y+8),NAMES[panel],font=font,fill='white')
                        speed=np.linalg.norm(f[tick,4:6]);yaw=f[tick,9]
                        draw.text((x+12,y+37),f't={t:.2f}s | v={speed:.3f}m/s | yaw={yaw:.3f}rad/s',font=font,fill='white')
                        state='共用学生前10秒' if t<10 else ('学生控制' if panel==0 else '教师接管')
                        if panel and t>=10 and phase[tick]==4:state='教师阶段失败'
                        if ended:state='已终止后定格'
                        draw.text((x+12,y+66),f'{state} | 位移 {np.linalg.norm(data.qpos[:2]):.2f}m',font=font,fill='#ffcccc' if state=='教师阶段失败' else 'white')
                    writer.append_data(np.asarray(canvas))
            with imageio.get_reader(movie) as reader:
                count=reader.count_frames();assert count==1800
                reader.get_data(0);imageio.imwrite(out/f'parking-restored-env{index}-15s.png',reader.get_data(900));reader.get_data(1799)
            movies.append(dict(env=index,sha256=digest(movie),path=movie.name,frames=1800,fps=60,
                original_teacher_phase_at_switch=int(b['teacher_phase'][999,index])))
        (out/'videos.json').write_text(json.dumps(dict(renderer_sha256=digest(__file__),
            trace_sha256={m:r['trace_sha256'] for m,r in reports.items()},checkpoint_sha256=reports['student-restored']['checkpoint_sha256'],
            selection='Fixed env0 and first original-teacher-non-DONE case; not selected by outcomes',
            layout=list(NAMES),common_prefix='Canonical student0..10s, then each actual restored-state continuation; discarded warmup never shown',
            camera='Follows actual body at1.25m; displacement and measured velocity/yaw displayed',movies=movies),indent=2)+'\n');b.close()
    finally:
        for d in traces.values():d.close()
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);render(p.parse_args())
