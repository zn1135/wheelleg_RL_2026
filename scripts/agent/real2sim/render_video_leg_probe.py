"""Normal-speed side-by-side rendering of savedGPU parameter candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidates',nargs='+',help='Candidate names to compare; default retains historical inertia columns')
    p.add_argument('--heights',type=int,nargs='+',default=[22,26,30])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_binary_path(str(args.recording/'fixtures/model.mjb'));disable_model_sameframe_shortcuts(model)
    data=mujoco.MjData(model);model.vis.global_.offwidth=480;model.vis.global_.offheight=400
    camera=mujoco.MjvCamera();camera.distance=1.5;camera.elevation=-12.;camera.azimuth=90.
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    run=json.loads((args.recording/'probe-run.json').read_text())
    samples={s['id']:s for s in json.loads((args.recording/'protocol.json').read_text())['samples']}
    names=tuple(args.candidates or [c['name'] for c in run['parameter_candidates'] if c['name'] in ('baseline','armature_x2','armature_x4')])
    model_name=Path(run['checkpoint']).stem
    number=re.search(r'model(\d+)',model_name)
    label=(number.group(1) if number else model_name)+f' P{run["leg_p"]:g}/D{run["leg_d"]:g} wheelP{run.get("wheel_p",.1):g}'
    with mujoco.Renderer(model,width=480,height=400) as renderer:
        for h in args.heights:
            traces=[]
            for name in names:
                with np.load(args.recording/'traces'/f'{name}_h{h}.npz') as d:traces.append({k:d[k].copy() for k in d.files})
            target=args.output/f'armature_comparison_h{h}.mp4'
            with imageio.get_writer(target,fps=50,codec='libx264',quality=8) as writer:
                for i in range(0,len(traces[0]['qpos'])-1,2):
                    canvas=Image.new('RGB',(480*len(names),400))
                    for col,(name,tr) in enumerate(zip(names,traces)):
                        data.qpos[:]=tr['qpos'][i];mujoco.mj_forward(model,data)
                        camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera=camera,scene_option=option)
                        panel=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(panel);tick=i*5
                        speed=np.linalg.norm(tr['linear_world'][tick,:2]);pitch=tr['roll_pitch_deg'][tick,1]
                        draw.rectangle((0,0,480,66),fill='#18212b')
                        draw.text((8,5),f'{label} | {name} | h={h/100:.2f}m',fill='white')
                        case=samples[f'{name}_h{h}']['case'];command=case['segments'][0]['command']
                        for segment in case['segments']:
                            if i*.01>=segment['start']:command=segment['command']
                        draw.text((8,24),f't={i*.01:.2f}s cmd vx={command[0]:g} | v={speed:.3f}m/s',fill='white')
                        draw.text((8,43),f'pitch={pitch:+.2f}deg | normal speed / following camera',fill='white')
                        canvas.paste(panel,(col*480,0))
                    writer.append_data(np.asarray(canvas))
            target.with_suffix('.json').write_text(json.dumps(dict(fps=50,duration_s=float(traces[0]['qpos_time'][-1]),columns=names,height_m=h/100,
                video_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),physics_steps=0,model_forwards=0),indent=2)+'\n')


if __name__=='__main__':main()
