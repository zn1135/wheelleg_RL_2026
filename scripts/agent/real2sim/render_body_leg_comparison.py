"""Normal-speed real/before/candidate replay of independent zero-command windows."""
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
    p.add_argument('--video',type=Path,required=True);p.add_argument('--candidates',nargs=2,required=True)
    p.add_argument('--height',type=int,choices=(22,26,30),default=26);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_binary_path(str(args.recording/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=480;model.vis.global_.offheight=400;data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.distance=1.25;camera.elevation=-20.;camera.azimuth=90.
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    protocol=json.loads((args.recording/'protocol.json').read_text());traces=[];sources=[]
    for name in args.candidates:
        sid=f'{name}_h{args.height}';path=args.recording/'traces'/f'{sid}.npz'
        with np.load(path) as tr:traces.append({k:tr[k].copy() for k in tr.files})
        sample=next(s for s in protocol['samples'] if s['id']==sid)
        assert sample['case']['duration']==30. and sample['case']['segments'][1]['command'][:2]==[0.,0.]
        sources.append(dict(sample_id=sid,candidate=sample['candidate'],trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    source=imageio.get_reader(args.video);source_fps=source.get_meta_data()['fps'];output=args.output/'real_before_candidate.mp4'
    with mujoco.Renderer(model,width=480,height=400) as renderer,imageio.get_writer(output,fps=30,codec='libx264',quality=8) as writer:
        for frame in range(240):
            canvas=Image.new('RGB',(1440,448),'#18212b');draw=ImageDraw.Draw(canvas)
            draw.text((12,12),'1x NORMAL | independent windows: real8-16s / simulation22-30s | real zero confirmed8-10,14-16s',font=font,fill='white')
            real_time=8.+frame/30;sim_time=22.+frame/30
            panel=Image.new('RGB',(480,400),'#18212b')
            real=Image.fromarray(source.get_data(round(real_time*source_fps))).crop((520,280,1020,640))
            real.thumbnail((480,330));panel.paste(real,((480-real.width)//2,70))
            text=ImageDraw.Draw(panel);text.text((10,6),f'REAL13679 P60/D2 | t={real_time:.2f}s',font=font,fill='white')
            text.text((10,32),'Height unknown; zero confirmed8-10/14-16s',font=font,fill='white');canvas.paste(panel,(0,48))
            for col,(name,trace) in enumerate(zip(args.candidates,traces),1):
                i=round(sim_time*100);data.qpos[:]=trace['qpos'][i];mujoco.mj_forward(model,data)
                camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera=camera,scene_option=option)
                panel=Image.fromarray(renderer.render().copy());text=ImageDraw.Draw(panel);tick=round(sim_time*500)
                text.rectangle((0,0,480,72),fill='#18212b')
                text.text((10,5),('BEFORE: ' if col==1 else 'CANDIDATE: ')+name,font=font,fill='white')
                text.text((10,29),f'13679 P60/D2 | h={args.height/100:.2f} | t={sim_time:.2f}s',font=font,fill='white')
                text.text((10,51),f'cmd=0 | pitch={trace["roll_pitch_deg"][tick,1]:+.2f}deg',font=font,fill='white')
                canvas.paste(panel,(480*col,48))
            writer.append_data(np.asarray(canvas))
    source.close()
    output.with_suffix('.json').write_text(json.dumps(dict(fps=30,frames=240,duration_s=8.,playback_speed=1.,
        real_window_s=[8,16],real_confirmed_zero_intervals_s=[[8,10],[14,16]],sim_window_s=[22,30],same_event_time_synchronized=False,real_crop_px=[520,280,1020,640],
        real_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),columns=sources,physics_steps=0,policy_forwards=0,
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
