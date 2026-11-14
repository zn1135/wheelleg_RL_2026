"""Render saved GPU height trajectories with actual/target height; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    folder=a.recording/'no_external';protocol=json.loads((folder/'protocol.json').read_text())
    run=json.loads((a.recording/'run.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=640;data=mujoco.MjData(model)
    cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-12.;cam.azimuth=90.
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    mapping=json.loads((folder/'fixtures/mapping.json').read_text());legs=np.asarray(mapping['motor_q'])[[0,1,3,4]]
    rows=[];videos=[]
    with mujoco.Renderer(model,width=960,height=640) as renderer:
        for sample in protocol['samples']:
            sid=sample['id'];target=sample['case']['segments'][0]['command'][2];end=sample['case']['duration']
            source=folder/'traces'/(sid+'.npz')
            with np.load(source) as saved:tr={k:saved[k].copy() for k in saved.files}
            assert tr['time'][-1]>=end-.002 and np.isfinite(tr['qpos']).all()
            final=(tr['time']>=end-5)&(tr['time']<end);qm=(tr['qpos_time']>=end-1)&(tr['qpos_time']<end)
            height=tr['position'][final,2];w,x,y,z=tr['qpos'][:,3:7].T
            yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
            rows.append(dict(sample_id=sid,target_height_m=target,actual_height_last5s_mean_m=float(height.mean()),
                actual_height_last5s_min_m=float(height.min()),actual_height_last5s_max_m=float(height.max()),
                height_error_last5s_mean_m=float((height-target).mean()),height_reward_last5s_mean=float((.01*np.exp(-(height-target)**2/.001)).mean()),
                leg_last1s_pp_deg=float(np.ptp(np.rad2deg(tr['qpos'][qm][:,legs]),axis=0).max()),
                horizontal_displacement_m=float(np.linalg.norm(tr['qpos'][-1,:2]-tr['qpos'][0,:2])),
                yaw_change_deg=float(np.rad2deg(yaw[-1]-yaw[0]))))
            path=a.output/(sid+'.mp4');frames=0
            with imageio.get_writer(path,fps=50,codec='libx264',quality=8) as writer:
                for i in range(0,len(tr['qpos'])-1,2):
                    tick=i*5;data.qpos[:]=tr['qpos'][i];mujoco.mj_forward(model,data)
                    # 相机只跟随水平位移，竖直观察中心固定，保留高度变化的直观对照。
                    cam.lookat[:]=[data.qpos[0],data.qpos[1],.20]
                    renderer.update_scene(data,camera=cam,scene_option=option)
                    canvas=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(canvas)
                    actual=tr['position'][tick,2];speed=np.linalg.norm(tr['linear_world'][tick,:2])
                    draw.rectangle((0,0,960,127),fill='#18212b')
                    draw.text((12,5),f'{Path(run["checkpoint"]).stem} | leg P60/D2 wheel P0.1 | no external load',font=font,fill='white')
                    draw.text((12,35),f'1x NORMAL | t={i*.01:.2f}s | cmd vx=0 yaw=0 | target h={target:.3f}m',font=font,fill='white')
                    draw.text((12,65),f'ACTUAL h={actual:.3f}m | height error={(actual-target)*100:+.1f}cm',font=font,fill='#ffbd65')
                    drift=np.linalg.norm(tr['qpos'][i,:2]-tr['qpos'][0,:2])
                    draw.text((12,95),f'speed={speed:.3f}m/s | XY from start={drift:.3f}m | horizontal following camera',font=font,fill='white')
                    frame=np.asarray(canvas);writer.append_data(frame);frames+=1
                    if i==round((end-1)*100):imageio.imwrite(path.with_suffix('.png'),frame)
            assert frames==round(end*50)
            videos.append(dict(path=str(path),frames=frames,fps=50,duration_s=end,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trajectory_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    result=dict(checkpoint=run['checkpoint'],checkpoint_sha256=run['sha256'],rows=rows,videos=videos,
        physics_steps=0,policy_updates=0,scope='Three nominal zero-command height samples; not training population or full behavior acceptance')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
