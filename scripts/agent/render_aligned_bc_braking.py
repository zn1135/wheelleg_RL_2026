#!/usr/bin/env python3
"""Render actual 30s independent-student/teacher-takeover cases and source plots."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from check_lqr_student_closed_loop import digest


def render(args):
    a=json.loads((args.run/'assessment.json').read_text());assert a['artifact_checks_passed']
    reports={b:json.loads((args.run/b/'result.json').read_text()) for b in ('student','teacher')}
    selected=[]
    qualified=next((r['env'] for r in a['cases']['teacher'] if r['qualified'] and not a['cases']['student'][r['env']]['qualified']),None)
    if qualified is None:qualified=next((r['env'] for r in a['cases']['teacher'] if r['qualified']),None)
    failed=next((r['env'] for r in a['cases']['teacher'] if not r['qualified']),None)
    if qualified is not None:selected.append(('qualified-teacher',qualified))
    if failed is not None:selected.append(('rejected-teacher',failed))
    data={}
    for branch,r in reports.items():
        assert digest(args.run/branch/'trace.npz')==r['trace_sha256']
        with np.load(args.run/branch/'trace.npz') as d:
            data[branch]={k:d[k][:,[case for _,case in selected]].copy() for k in ('qpos','qvel','features','alive','teacher_x','teacher_requested','student_requested','source_dq','actuator_torque','observation')}
            data[branch]['initial_qpos']=d['initial_qpos'][[case for _,case in selected]].copy()
            data[branch]['time']=d['time'].copy()
            data[branch]['wheel_angle_handover_latched']=d['wheel_angle_handover_latched'][:,[case for _,case in selected]].copy()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,training=task_registry.get_cfgs('chuanliantui',play=True)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=training,num_envs=1);renderer=None;movies=[]
    try:
        model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
        renderer=mujoco.Renderer(model,width=640,height=360);physical=mujoco.MjData(model)
        camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-18;camera.distance=1.25
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',17)
        for j,(kind,case) in enumerate(selected):
            path=args.output/f'{kind}-case{case}.mp4'
            for branch in data:
                initial=data[branch]['initial_qpos'][j]
                data[branch]['qpos'][:,j,:2]-=initial[:2];initial[:2]=0.
            with imageio.get_writer(path,fps=60,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
                for frame in range(1800):
                    t=frame/60;k=min(2999,max(0,round(t/.01)-1));canvas=Image.new('RGB',(1280,490),'#18212b');draw=ImageDraw.Draw(canvas)
                    for panel,branch in enumerate(('student','teacher')):
                        d=data[branch];r=reports[branch]['rows'][case];verdict=a['cases'][branch][case]
                        length=int(d['alive'][:,j].sum());at=min(k,max(length-1,0))
                        physical.qpos[:]=d['initial_qpos'][j] if frame==0 else d['qpos'][at,j]
                        physical.qvel[:]=0 if frame==0 or k>=length else d['qvel'][at,j]
                        camera.lookat[:]=[physical.qpos[0],physical.qpos[1],.20]
                        mujoco.mj_forward(model,physical);renderer.update_scene(physical,camera=camera,scene_option=option)
                        x=panel*640;canvas.paste(Image.fromarray(renderer.render()),(x,130))
                        title='学生独立运行30秒' if branch=='student' else '学生10秒→LQR实际接管'
                        owner='学生控制' if branch=='student' or t<10. else '教师控制（共享位置参考不重锁）'
                        draw.text((x+10,5),title+' | 未新增训练',font=font,fill='white')
                        draw.text((x+10,31),f'{owner} | t={t:.2f}s | 腿/轮 {r["leg_delay_s"]*1000:.0f}/{r["wheel_delay_s"]*1000:.0f}ms',font=font,fill='white')
                        draw.text((x+10,57),f'机体前向速度 {d["features"][at,j,4]:+.4f}m/s | 总位移 {np.linalg.norm(physical.qpos[:2]):.2f}m',font=font,fill='white')
                        drift=np.linalg.norm(d['qpos'][at,j,:2]-d['qpos'][2499,j,:2])*100 if at>=2499 else 0.
                        text=('全回合停车合格' if verdict['qualified'] else '全回合停车未过')+f' | 25秒后位移 {drift:.2f}cm'
                        if k>=length:text='已终止后定格 | '+text
                        draw.text((x+10,83),text,font=font,fill='white' if verdict['qualified'] else '#ffbbbb')
                        draw.text((x+10,107),'同初态；展示各分支真实完整轨迹',font=font,fill='white')
                    writer.append_data(np.asarray(canvas))
                    if frame%600==0:print(f'{kind} frame={frame}/1800',flush=True)
            with imageio.get_reader(path) as reader:
                assert reader.count_frames()==1800
                for frame in (60,600,1200,1799):imageio.imwrite(args.output/f'{kind}-case{case}-{frame:04d}.png',reader.get_data(frame))
            movies.append(dict(path=path.name,kind=kind,case=case,frames=1800,fps=60,sha256=digest(path),
                student_qualified=a['cases']['student'][case]['qualified'],teacher_qualified=a['cases']['teacher'][case]['qualified']))
            # 同一已记录学生轨迹上的参考及教师请求；不把旁路请求当实际执行力矩。
            d=data['student'];time=d['time'];live=d['alive'][:,j]
            fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True)
            for branch,color in (('student','tab:blue'),('teacher','tab:orange')):
                b=data[branch];mask=b['alive'][:,j]
                axes[0].plot(time[mask],b['features'][mask,j,4],label=branch,color=color)
                axes[3].plot(time[mask],np.linalg.norm(b['qpos'][mask,j,:2],axis=-1),label=branch,color=color)
            axes[0].axhline(.01,color='grey',ls=':');axes[0].axhline(-.01,color='grey',ls=':')
            axes[0].set_ylabel('Body forward speed (m/s)')
            axes[1].plot(time[live],.03*(d['observation'][live,j,14]-d['observation'][live,j,15]),label='signed wheel travel, R=0.06m')
            axes[1].plot(time[live],d['teacher_x'][live,j,0],label='LQR velocity-error integral')
            axes[1].set_ylabel('Position / integral (m)')
            for owner,color in (('student','tab:blue'),('teacher','tab:orange')):
                values=d[owner+'_requested'][live,j]
                axes[2].plot(time[live],(values[:,2]-values[:,5])*.5,label=owner+' nominal request',color=color)
            axes[2].set_ylabel('Forward wheel request (Nm)')
            axes[3].set_ylabel('Root XY displacement (m)');axes[3].set_xlabel('Time (s)')
            for ax in axes:ax.axvline(10,color='black',ls='--',alpha=.6);ax.grid(alpha=.25);ax.legend()
            fig.suptitle(f'Case {case}: actual closed loops; teacher requests in panel3 are queries on student states')
            fig.tight_layout();fig.savefig(args.output/f'{kind}-case{case}-signals.png',dpi=150);plt.close(fig)
        (args.output/'videos.json').write_text(json.dumps(dict(movies=movies,optimizer_updates=0,
            checkpoint_sha256=a['checkpoint_sha256'],renderer_sha256=digest(__file__),
            selection='First qualified teacher with unqualified student if present, otherwise first qualified teacher; also first rejected teacher.',
            prefix='Own actual trajectories; no state restoration or prefix splicing. Prefix equality is separately measured in assessment.'),indent=2)+'\n')
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output'):p.add_argument('--'+name,type=Path,required=True)
    render(p.parse_args())
