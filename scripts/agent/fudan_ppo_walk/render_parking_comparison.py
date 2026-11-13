"""Normal-speed paired parking replay from savedGPU trajectories only."""
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
    p=argparse.ArgumentParser(__doc__);p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--group',choices=('no_external','configured_external'),default='no_external')
    p.add_argument('--samples',nargs='+',default=['parking_5']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    with ExitStack() as stack:
        sides=[]
        for directory in (args.before,args.after):
            run=json.loads((directory/'run.json').read_text());folder=directory/args.group
            protocol=json.loads((folder/'protocol.json').read_text())
            model=mujoco.MjModel.from_binary_path(str(folder/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
            model.vis.global_.offwidth=480;model.vis.global_.offheight=400
            data=mujoco.MjData(model);renderer=stack.enter_context(mujoco.Renderer(model,width=480,height=400))
            cam=mujoco.MjvCamera();cam.distance=1.5;cam.elevation=-12.;cam.azimuth=90.
            option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
            sides.append((folder,run,protocol,model,data,renderer,cam,option))
        for sid in args.samples:
            traces=[];sources=[];cases=[]
            for folder,run,protocol,*_ in sides:
                sample=next(s for s in protocol['samples'] if s['id']==sid);cases.append(sample['case'])
                path=folder/'traces'/(sid+'.npz')
                with np.load(path) as saved:traces.append({k:saved[k].copy() for k in saved.files})
                w,x,y,z=traces[-1]['qpos'][:,3:7].T
                traces[-1]['yaw_unwrapped']=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                sources.append(dict(checkpoint=run['checkpoint'],model_sha256=run['sha256'],trajectory=str(path),trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            assert cases[0]==cases[1] and len(traces[0]['qpos'])==len(traces[1]['qpos'])
            output=args.output/(sid+'.mp4');duration=cases[0]['duration']
            with imageio.get_writer(output,fps=50,codec='libx264',quality=8) as writer:
                for i in range(0,len(traces[0]['qpos'])-1,2):
                    canvas=Image.new('RGB',(960,448),'#18212b');header=ImageDraw.Draw(canvas)
                    header.text((12,11),f'1x NORMAL | {cases[0]["id"]} | {args.group} | following camera',font=titlefont,fill='white')
                    if 'wheel_motor_gain' in sides[0][1]:
                        first,second=sides[0][1],sides[1][1]
                        label=(f'Both: wheel P={first["wheel_p"]:g}, post-limit gain={first["wheel_motor_gain"]:g}'
                            if (first['wheel_p'],first['wheel_motor_gain'])==(second['wheel_p'],second['wheel_motor_gain'])
                            else f'Left: wheel P={first["wheel_p"]:g}, gain={first["wheel_motor_gain"]:g} | Right: wheel P={second["wheel_p"]:g}, gain={second["wheel_motor_gain"]:g}')
                        header.text((12,33),label,fill='white')
                    for col,((folder,run,protocol,model,data,renderer,cam,option),trace) in enumerate(zip(sides,traces)):
                        data.qpos[:]=trace['qpos'][i];mujoco.mj_forward(model,data);cam.lookat[:]=data.qpos[:3]
                        renderer.update_scene(data,camera=cam,scene_option=option)
                        panel=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(panel);tick=i*5
                        command=cases[0]['segments'][0]['command']
                        for segment in cases[0]['segments']:
                            if i*.01>=segment['start']:command=segment['command']
                        speed=np.linalg.norm(trace['linear_world'][tick,:2]);pitch=trace['roll_pitch_deg'][tick,1]
                        draw.rectangle((0,0,480,110),fill='#18212b')
                        draw.text((10,4),f'{run.get("display_label",Path(run["checkpoint"]).stem)} P{run["leg_p"]:g}/D{run["leg_d"]:g}',font=titlefont,fill='white')
                        draw.text((10,32),f't={i*.01:.2f}s cmd vx={command[0]:g} h={command[2]:.2f}',font=font,fill='white')
                        draw.text((10,57),f'speed={speed:.3f}m/s pitch={pitch:+.2f}deg',font=font,fill='white')
                        drift=np.linalg.norm(trace['qpos'][i,:2]-trace['qpos'][0,:2])
                        yaw=np.rad2deg(trace['yaw_unwrapped'][i]-trace['yaw_unwrapped'][0])
                        draw.text((10,82),f'XY from start={drift:.3f}m yaw change={yaw:+.1f}deg',font=font,fill='white')
                        canvas.paste(panel,(col*480,48))
                    writer.append_data(np.asarray(canvas))
            output.with_suffix('.json').write_text(json.dumps(dict(fps=50,duration_s=duration,playback_speed=1,columns=sources,
                physics_steps=0,model_forwards=0,video_sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
