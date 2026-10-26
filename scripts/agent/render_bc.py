#!/usr/bin/env python3
"""Full-length paired teacher/BC videos from saved states; never resimulates."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ct_mjlab.envs import task_registry
from check_lqr_student_closed_loop import digest


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
 r=json.loads((a.run/'result.json').read_text());m=json.loads(a.dataset.read_text());assert digest(a.run/'trace.npz')==r['trace_sha256']
 matched={}
 for row in r['rows']:
  good=next((e for e in m['episodes'] if e['group_id']==row['group_id'] and e['leg_delay_s']==row['leg_delay_s'] and e['wheel_delay_s']==row['wheel_delay_s']),None)
  if good:matched[row['env']]=good
 selected=[]
 for kind,qualified in (('passed',True),('failed',False)):
  rows=[row for row in r['rows'] if row['qualified']==qualified and row['env'] in matched]
  if rows:selected.append((kind,rows[0],matched[rows[0]['env']]))
 # 没有合格样本时，另保留首个最终平衡但综合失败的回合，不能只展示提前终止。
 if not any(kind=='passed' for kind,_,_ in selected):
  rows=[row for row in r['rows'] if row['final_balance_v31'] and not row['qualified'] and row['env'] in matched]
  if rows and all(row['env']!=rows[0]['env'] for _,row,_ in selected):
   selected.append(('balanced-unqualified',rows[0],matched[rows[0]['env']]))
 cfg,train=task_registry.get_cfgs('chuanliantui',play=True);env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=1)
 renderer=None;metadata=[]
 try:
  model=env.unwrapped.sim.mj_model;model.vis.global_.offwidth=640;model.vis.global_.offheight=360
  renderer=mujoco.Renderer(model,width=640,height=360);data=mujoco.MjData(model);camera=mujoco.MjvCamera()
  camera.azimuth=100;camera.elevation=-18;camera.distance=2.8
  option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
  font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',18)
  for kind,row,reference in selected:
   path=a.dataset.parent/reference['path'];assert digest(path)==reference['sha256']
   with np.load(path) as good,np.load(a.run/'trace.npz') as bad:
    n=round(row['observed_s']/.01);q=[good['qpos'].copy(),bad['qpos'][:n,row['env']].copy()];v=[good['qvel'].copy(),bad['qvel'][:n,row['env']].copy()]
    initial=[good['initial_qpos'].copy(),bad['initial_qpos'][row['env']].copy()]
    np.testing.assert_allclose(initial[0][2:],initial[1][2:],atol=1e-6)
    for i in range(2):q[i][:,:2]-=initial[i][:2];initial[i][:2]=0
   # 跟随各自真实机身，避免失败漂移把机器人缩成小点；同时显示位移。
   camera.distance=1.25
   movie=a.run/f'{kind}-comparison.mp4'
   with imageio.get_writer(movie,fps=60,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
    for frame in range(1200):
     t=frame/60;canvas=Image.new('RGB',(1280,460),'#18212b');draw=ImageDraw.Draw(canvas)
     label=('轮角39维BC累计' if 'wheel-angle-39-82' in r.get('observation_contract','') else
            '41维行为克隆累计' if 'reference-41-84' in r.get('observation_contract','') else
            '40维行为克隆' if 'wheel-40-83' in r.get('observation_contract','') else '行为克隆累计')
     student_title=('诊断策略（未新增训练）：独立控制' if r.get('diagnostic_only') else
                    f'{label}{r.get("actor_updates",5000)}次：独立控制')
     if r.get('ppo_iterations'):
      student_title=f'{"PPO＋SMP" if r["smp_reward_weight"] else "仅任务奖励PPO"} {r["ppo_iterations"]}轮：独立控制'
     for i,title in enumerate(('教师LQR→学生PD',student_title)):
      index=max(0,round(t/.01)-1);ended=index>=len(q[i]);index=min(index,len(q[i])-1)
      data.qpos[:]=initial[i] if frame==0 else q[i][index];data.qvel[:]=0 if frame==0 or ended else v[i][index]
      camera.lookat[:]=[data.qpos[0],data.qpos[1],.20]
      mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option)
      canvas.paste(Image.fromarray(renderer.render()),(640*i,100))
      draw.text((640*i+12,8),title,font=font,fill='white')
      draw.text((640*i+12,37),f't={t:.2f}s | 腿/轮延迟 {row["leg_delay_s"]*1000:.0f}/{row["wheel_delay_s"]*1000:.0f}ms',font=font,fill='white')
      state='已终止后定格' if ended else ('通过' if i==0 or row['qualified'] else '本回合未通过验收')
      if i==1 and not ended and row['strict_rise'] and row['final_balance_v31'] and not row['qualified']:
       state='起立/平衡通过，综合未过'
      draw.text((640*i+12,66),f'{state} | 位移 {np.linalg.norm(data.qpos[:2]):.2f}m',font=font,fill='#ffaaaa' if i==1 else 'white')
     writer.append_data(np.asarray(canvas))
   reader=imageio.get_reader(movie);count=reader.count_frames();reader.get_data(0);poster=reader.get_data(60);reader.get_data(count-1);reader.close();assert count==1200
   imageio.imwrite(a.run/f'{kind}-poster.png',poster)
   metadata.append(dict(kind=kind,row=row,teacher_episode_id=reference['episode_id'],teacher_sha256=reference['sha256'],
    video_sha256=digest(movie),frames=count,fps=60,terminal_note='BC view freezes with explicit label after recorded termination; not continued physical survival.'))
  (a.run/'video.json').write_text(json.dumps(dict(renderer_sha256=digest(__file__),trace_sha256=r['trace_sha256'],checkpoint_sha256=r['checkpoint_sha256'],
    selection='First qualified and first unqualified matching row; if none qualified, also first final-balanced unqualified row. Identical pose/delays, distinct cases only.',
    camera='Tracks each recorded body at fixed 1.25m distance; horizontal displacement is displayed',videos=metadata),indent=2)+'\n')
 finally:
  if renderer is not None:renderer.close()
  env.close()


if __name__=='__main__':main()
