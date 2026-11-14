"""Render saved GPU torque-replay error beside measured-joint reconstruction."""
import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation,Slerp
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
from .replay_usb_pd_targets import PoseSolver,sha,write


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    replay=a.experiment/'wheel-torque-short-replay-01';original=a.experiment/'mujoco-target-replay-01'
    profiles=json.loads((replay/'plan.json').read_text())['profiles']
    selected=profiles.index(dict(mode='recorded_torque',initial='rolling',tau_ms=0))
    with np.load(replay/'traces.npz') as f:
        starts=f['starts'];window=int(np.flatnonzero(starts==8500)[0]);world=selected*len(starts)+window
        simulated=f['qpos'][:,world].copy()
    with np.load(original/'inputs.npz') as f:inputs={k:f[k].copy() for k in f.files}
    ticks=8500+np.arange(len(simulated));times=ticks*.002
    model=mujoco.MjModel.from_binary_path(str(original/'model.mjb'));disable_model_sameframe_shortcuts(model)
    qi=int(model.joint('robot/rf0').qposadr[0]);model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    # 只改变渲染颜色，标出右前主动杆；不修改或推进任何物理轨迹。
    body=int(model.body('robot/rf0').id)
    for i in np.flatnonzero(model.geom_bodyid==body):model.geom_matid[i]=-1;model.geom_rgba[i]=[1.,.38,.05,1.]
    reference_times=inputs['reference_indices']*.002
    quaternions=Slerp(reference_times,Rotation.from_quat(inputs['reference_qpos'][:,[4,5,6,3]]))(times).as_quat()[:,[3,0,1,2]]
    wheel_positions=np.stack([np.interp(times,reference_times,inputs['reference_qpos'][:,int(model.joint('robot/'+name).qposadr[0])]) for name in ('lfwheel','rfwheel')],axis=1)
    solver=PoseSolver();measured=[];closure=[]
    for k,tick in enumerate(ticks):
        motors=inputs['current'][tick].copy();motors[[2,5]]=wheel_positions[k]
        pose,error=solver.solve(motors);measured.append(solver.ground(pose,quaternions[k]));closure.append(error)
    measured=np.asarray(measured);actual=np.rad2deg(inputs['current'][ticks,3]);predicted=np.rad2deg(simulated[:,qi]);error=abs(predicted-actual)
    assert np.max(abs(np.rad2deg(measured[:,qi])-actual))<1e-9
    expected=json.loads((replay/'maximum-joint-errors.json').read_text())['rows'][selected]
    assert abs(float(error[-1])-expected['max_absolute_error_deg'])<1e-4
    np.savez_compressed(out/'displayed-poses.npz',time_s=times,measured_qpos=measured,simulated_qpos=simulated,measured_rf0_deg=actual,simulated_rf0_deg=predicted,error_deg=error)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    camera=mujoco.MjvCamera();camera.distance=1.3;camera.elevation=-12.
    # 两栏采用同一右侧视角，避免各自跟随姿态把角度差隐藏掉。
    yaw=Rotation.from_quat(quaternions[0,[1,2,3,0]]).as_euler('xyz')[2]
    camera.azimuth=-90+np.rad2deg(yaw)
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    data=mujoco.MjData(model);rendered=[]
    with mujoco.Renderer(model,width=640,height=480) as renderer:
        for k in range(len(times)):
            canvas=Image.new('RGB',(1280,672),'#18212b');draw=ImageDraw.Draw(canvas)
            draw.text((14,45),f'Log t={times[k]:.3f}s | elapsed={k*.002:.3f}s | rf0 angle difference={error[k]:.2f} deg',font=title,fill='#ffcd72')
            for col,(pose,label,angle) in enumerate(((measured[k],'MEASURED JOINTS - pose reconstruction',actual[k]),(simulated[k],'GPU REPLAY - recorded torque requests',predicted[k]))):
                data.qpos[:]=pose;mujoco.mj_forward(model,data);camera.lookat[:]=[pose[0],pose[1],.20]
                renderer.update_scene(data,camera=camera,scene_option=option);panel=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(panel)
                d.rectangle((0,0,640,90),fill='#18212b');d.text((12,5),label,font=font,fill='white')
                d.text((12,33),f'Right-front joint rf0 = {angle:+.2f} deg',font=title,fill='#ff9b43')
                d.text((12,65),'XY fixed, Z from geometry; body pose partly reconstructed' if col==0 else 'Original parameters; no added wheel lag; no policy feedback',font=small,fill='#bfc9d2')
                canvas.paste(panel,(640*col,90))
            draw=ImageDraw.Draw(canvas);draw.text((14,577),'Orange = right-front link. Left is reconstructed telemetry, not camera footage.',font=font,fill='white')
            draw.text((14,605),'Each side uses its own body pose. Real root translation/height were not measured.',font=font,fill='#ffcd72')
            draw.text((14,633),'Torque requests are replayed as recorded; simulated errors do not trigger new policy corrections.',font=small,fill='#bfc9d2')
            rendered.append(canvas)
    videos=[]
    for filename,stride,label in (('normal.mp4',10,'1x NORMAL'),('slow-10x.mp4',1,'0.1x SPEED / 10x SLOW MOTION')):
        frames=0
        with imageio.get_writer(out/filename,fps=50,codec='libx264',quality=8) as writer:
            # 运动段之后单独标注2秒定格，不把定格误称成持续运动。
            sequence=[(k,False) for k in range(0,len(times),stride)]+[(len(times)-1,True)]*100
            for k,hold in sequence:
                frame=rendered[k].copy();draw=ImageDraw.Draw(frame)
                draw.text((14,9),('END FRAME HOLD (2 seconds)' if hold else label)+' | source window 17.000-17.200s',font=title,fill='white')
                writer.append_data(np.asarray(frame));frames+=1
                if hold and frames==len(sequence):frame.save(out/'endpoint.png')
        reader=imageio.get_reader(out/filename);meta=reader.get_meta_data();decoded=sum(1 for _ in reader);reader.close()
        assert decoded==frames and meta['fps']==50 and abs(meta['duration']-frames/50)<.03
        videos.append(dict(file=filename,frames=frames,fps=50,duration_s=frames/50,motion_speed=.1 if stride==1 else 1.,end_hold_s=2,sha256=sha(out/filename),full_decode_passed=True))
    write(out/'report.json',dict(profile=profiles[selected],window_start_s=17.,window_end_s=17.2,
        measured_end_deg=float(actual[-1]),simulated_end_deg=float(predicted[-1]),absolute_error_deg=float(error[-1]),videos=videos,
        source_sha256={str(x):sha(x) for x in (replay/'traces.npz',original/'inputs.npz',original/'model.mjb')},
        real_root_position_measured=False,reference_attitude_interpolation='Slerp between archived50Hz IMU-derived poses',
        new_physics_steps=0,policy_forwards=0,training_updates=0))
    shutil.copy2(__file__,out/'render-source.py');print(json.dumps(dict(error_deg=float(error[-1]),videos=videos),indent=2))


if __name__=='__main__':main()
