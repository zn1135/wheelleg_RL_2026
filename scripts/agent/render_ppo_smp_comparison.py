#!/usr/bin/env python3
"""Uncut20s four-panel teacher/BC/task-only/SMP replay from recorded physical states."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from check_lqr_student_closed_loop import digest


def render(args):
    runs=[args.baseline,args.run/'taskonly/evaluation',args.run/'smp/evaluation']
    reports=[json.loads((p/'result.json').read_text()) for p in runs]
    for p,r in zip(runs,reports):assert digest(p/'trace.npz')==r['trace_sha256']
    manifest=json.loads(args.dataset.read_text());rows=[r['rows'][args.env] for r in reports]
    key=lambda r:(r['group_id'],r['leg_delay_s'],r['wheel_delay_s'])
    assert key(rows[0])==key(rows[1])==key(rows[2])
    teacher=next(e for e in manifest['episodes'] if key(e)==key(rows[0]))
    teacher_path=args.dataset.parent/teacher['path'];assert digest(teacher_path)==teacher['sha256']
    clips=[]
    with np.load(teacher_path) as d:clips.append((d['qpos'].copy(),d['qvel'].copy(),d['initial_qpos'].copy()))
    for p,row in zip(runs,rows):
        with np.load(p/'trace.npz') as d:
            n=int(d['alive'][:,args.env].sum())
            clips.append((d['qpos'][:n,args.env].copy(),d['qvel'][:n,args.env].copy(),d['initial_qpos'][args.env].copy()))
    for q,v,initial in clips:
        np.testing.assert_allclose(initial[2:],clips[0][2][2:],atol=1e-6)
        q[:,:2]-=initial[:2];initial[:2]=0
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1)
    renderer=None
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        renderer=mujoco.Renderer(model,width=640,height=360);data=mujoco.MjData(model)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
        titles=['LQR教师→学生PD','训练前BC','仅任务奖励PPO · 300轮','PPO＋SMP · 300轮']
        movie=args.output/'comparison.mp4'
        with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(1200):
                t=frame/60;canvas=Image.new('RGB',(1280,920),'#18212b');draw=ImageDraw.Draw(canvas)
                for i,(q,v,initial) in enumerate(clips):
                    x,y=(i%2)*640,(i//2)*460;index=max(0,round(t/.01)-1);ended=index>=len(q);index=min(index,len(q)-1)
                    data.qpos[:]=initial if frame==0 else q[index];data.qvel[:]=0 if frame==0 or ended else v[index]
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.20]
                    mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()),(x,y+100))
                    if ended:status='已终止，之后画面定格'
                    elif i==0:status='合格教师示范'
                    else:
                        row=rows[i-1];status=f'本回合 起立{int(row["strict_rise"])} 平衡{int(row["final_balance_v31"])} 停车{int(row["qualified"])}'
                    draw.text((x+10,y+7),titles[i],font=font,fill='white')
                    draw.text((x+10,y+36),f't={t:.2f}s · 位移{np.linalg.norm(data.qpos[:2]):.2f}m · case{args.env}',font=font,fill='white')
                    draw.text((x+10,y+65),status,font=font,fill='#ffaaaa' if ended else '#c7e4f3')
                writer.append_data(np.asarray(canvas))
        reader=imageio.get_reader(movie);assert reader.count_frames()==1200
        for n in (0,600,1199):
            frame=reader.get_data(n)
            if n==600:imageio.imwrite(args.output/'poster.png',frame)
        reader.close()
        result=dict(frames=1200,fps=60,seconds=20,environment=args.env,leg_delay_s=rows[0]['leg_delay_s'],wheel_delay_s=rows[0]['wheel_delay_s'],
            video_sha256=digest(movie),teacher_sha256=teacher['sha256'],rows=rows,
            trace_sha256=[r['trace_sha256'] for r in reports],renderer_sha256=digest(__file__),
            scope='Actual saved states; fixed original case, complete20s, explicitly freezes after termination; no resimulation or cherry-picked success montage')
        (args.output/'video.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','baseline','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--env',type=int,default=0);render(p.parse_args())
