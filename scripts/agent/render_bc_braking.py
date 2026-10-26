#!/usr/bin/env python3
"""Render full accepted/rejected braking traces without re-integrating physics."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from check_lqr_student_closed_loop import digest


def render(args):
    r=json.loads((args.run/'result.json').read_text())
    assert digest(args.run/'trace.npz')==r['trace_sha256'] and digest(MODEL_XML)==r['xml_sha256']
    good=next(row for row in r['rows'] if row['qualified'])
    bad=next((row for row in r['rows'] if not row['qualified'] and row['group_id']==good['group_id']),None)
    if bad is None:bad=next((row for row in r['rows'] if not row['qualified']),None)
    rows=[good]+([] if bad is None else [bad]);ids=[row['env'] for row in rows]
    with np.load(args.run/'trace.npz') as d:
        q=d['qpos'][:,ids].copy();v=d['qvel'][:,ids].copy();f=d['features'][:,ids].copy()
        initial=d['initial_qpos'][ids].copy();phase=d['teacher_phase'][:,ids].copy()
    q[:,:,:2]-=initial[None,:,:2];initial[:,:2]=0.
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1)
    renderer=None
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        renderer=mujoco.Renderer(model,width=640,height=360);data=mujoco.MjData(model)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
        movie=args.run/'braking-accepted-rejected.mp4';frames=round(r['duration_s']*60)
        with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1,
                               ffmpeg_params=['-movflags','+faststart']) as writer:
            for frame in range(frames):
                t=frame/60;tick=min(len(q)-1,max(0,round(t/.01)-1))
                canvas=Image.new('RGB',(640*len(rows),480),'#18212b');draw=ImageDraw.Draw(canvas)
                for panel,row in enumerate(rows):
                    data.qpos[:]=initial[panel] if frame==0 else q[tick,panel]
                    data.qvel[:]=0 if frame==0 else v[tick,panel]
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.2]
                    mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option)
                    x=640*panel;canvas.paste(Image.fromarray(renderer.render()),(x,120))
                    title='接收：平衡、平滑、停车均合格' if row['qualified'] else '拒收：'+','.join(row['rejection_reasons'])
                    draw.text((x+10,6),title,font=font,fill='white' if row['qualified'] else '#ffaaaa')
                    owner='冻结学生控制' if t<r['takeover_s'] else '同步阶段后LQR制动'
                    draw.text((x+10,33),f'{owner} | t={t:.2f}s | 延迟{row["leg_delay_s"]*1000:.0f}/{row["wheel_delay_s"]*1000:.0f}ms',font=font,fill='white')
                    draw.text((x+10,60),f'v={np.linalg.norm(f[tick,panel,4:6]):.3f}m/s | yaw={f[tick,panel,9]:.3f}rad/s',font=font,fill='white')
                    # 以末5秒段首为参考展示厘米漂移，不能用跟随镜头掩盖移动。
                    drift=np.linalg.norm(q[tick,panel,:2]-q[2499,panel,:2])*100 if t>=25 else 0.
                    draw.text((x+10,87),f'总位移{np.linalg.norm(data.qpos[:2]):.2f}m | 末段位移{drift:.2f}cm | 阶段{phase[tick,panel]}',font=font,fill='white')
                writer.append_data(np.asarray(canvas))
                if frame%600==0:print(f'render {frame}/{frames}',flush=True)
        with imageio.get_reader(movie) as reader:
            assert reader.count_frames()==frames
            for index in (0,600,900,1500,frames-1):reader.get_data(index)
            imageio.imwrite(args.run/'braking-27s.png',reader.get_data(1620))
        result=dict(video=movie.name,sha256=digest(movie),fps=60,frames=frames,duration_s=r['duration_s'],
            rows=rows,trace_sha256=r['trace_sha256'],checkpoint_sha256=r['checkpoint_sha256'],renderer_sha256=digest(__file__),
            selection='First qualified env, then first rejected of the same pose if present; otherwise first rejected.',
            scope='Actual full traces at1x speed; only translate world XY for display, no snapshot restore or simulated continuation.')
        (args.run/'videos.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);render(p.parse_args())
