"""Replay saved shared-noise and shared-delay experiments, with labeled slow motion."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--noise-root',type=Path,required=True)
    p.add_argument('--delay14000',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for group in ('noise','delay'):
        sid='armature_x2_velnoise_0.2_h26' if group=='noise' else 'armature_x2_veldelay_6ms_h26'
        panels=[];sources=[]
        with ExitStack() as stack:
            for number,pd in (('11700','P30/D2'),('14000','P30/D10'),('13179','P60/D2')):
                folder=a.delay14000 if group=='delay' and number=='14000' else a.noise_root/('model'+number)
                path=folder/'traces'/(sid+'.npz')
                with np.load(path) as d:trace={k:d[k].copy() for k in d.files}
                model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
                model.vis.global_.offwidth=480;model.vis.global_.offheight=400
                data=mujoco.MjData(model);renderer=stack.enter_context(mujoco.Renderer(model,width=480,height=400))
                cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-12.;cam.azimuth=90.
                opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0]
                panels.append((number,pd,trace,model,data,renderer,cam,opt))
                sources.append(dict(model=number,pd=pd,trajectory=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            name='shared_noise_normal' if group=='noise' else 'shared_delay_normal'
            target=a.output/(name+'.mp4');fps=50 if group=='noise' else 100
            writer=stack.enter_context(imageio.get_writer(target,fps=fps,codec='libx264',quality=8))
            slow=None;slow_path=a.output/'shared_delay_5x_slow.mp4'
            if group=='delay':slow=stack.enter_context(imageio.get_writer(slow_path,fps=20,codec='libx264',quality=8))
            condition='PD velocity noise std 0.2 rad/s; no added PD velocity lag' if group=='noise' else 'PD velocity lag 6 ms; no added PD velocity noise'
            for i in range(0,1500,2 if group=='noise' else 1):
                canvas=Image.new('RGB',(1440,448),'#18212b');header=ImageDraw.Draw(canvas)
                header.text((12,11),f'1x NORMAL | {condition} | all: armature x2, h0.26m',font=titlefont,fill='white')
                for col,(number,pd,tr,model,data,renderer,cam,opt) in enumerate(panels):
                    data.qpos[:]=tr['qpos'][i];mujoco.mj_forward(model,data);cam.lookat[:]=data.qpos[:3]
                    renderer.update_scene(data,camera=cam,scene_option=opt)
                    panel=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(panel);tick=i*5
                    speed=np.linalg.norm(tr['linear_world'][tick,:2]);pitch=tr['roll_pitch_deg'][tick,1]
                    draw.rectangle((0,0,480,85),fill='#18212b')
                    draw.text((10,4),f'{number} / {pd}',font=titlefont,fill='white')
                    draw.text((10,32),f't={i*.01:.2f}s  command vx={1 if i<700 else 0}',font=font,fill='white')
                    draw.text((10,57),f'speed={speed:.3f}m/s  pitch={pitch:+.2f}deg',font=font,fill='white')
                    canvas.paste(panel,(col*480,48))
                writer.append_data(np.asarray(canvas))
                if slow is not None and i>=1000:
                    enlarged=canvas.copy();draw=ImageDraw.Draw(enlarged);draw.rectangle((0,0,1440,47),fill='#681e16')
                    draw.text((12,11),'0.2x SPEED / 5x SLOW | source time 10-15s | PD velocity lag 6ms, NO added noise | armature x2',font=titlefont,fill='white')
                    slow.append_data(np.asarray(enlarged))
        metadata=dict(fps=fps,duration_s=15,playback_speed=1,condition=condition,columns=sources,
            source_pose_hz=100,physics_steps=0,model_forwards=0,following_camera=True,video_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        target.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if group=='delay':slow_path.with_suffix('.json').write_text(json.dumps({**metadata,'fps':20,'duration_s':25,'playback_speed':.2,'source_window_s':[10,15],'video_sha256':hashlib.sha256(slow_path.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
