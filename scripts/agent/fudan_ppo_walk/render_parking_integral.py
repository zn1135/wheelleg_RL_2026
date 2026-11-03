"""Normal-speed three-way GPU trajectory comparison; kinematic rendering only."""
import argparse,json,time,signal,hashlib
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .diagnose_parking_integral import HEIGHTS,BRANCHES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    budget=json.loads((a.experiment/'budget.json').read_text());stop=budget['deadline_unix']-8.
    def guard():
        if time.time()>=stop:raise TimeoutError('Video soft stop before overall hard deadline')
    def interrupted(*args):raise TimeoutError('Overall video deadline')
    signal.signal(signal.SIGTERM,interrupted)
    (a.experiment/'videos').mkdir(exist_ok=True);index=[]
    for height in HEIGHTS:
        guard();model=mujoco.MjModel.from_binary_path(str(a.experiment/height/'baseline/fixtures/model.mjb'));data=mujoco.MjData(model)
        from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
        disable_model_sameframe_shortcuts(model)
        model.vis.global_.offwidth=384;model.vis.global_.offheight=288
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        camera=mujoco.MjvCamera();camera.distance=1.65;camera.elevation=-18.;camera.azimuth=110.
        path=a.experiment/'videos'/(height+'.mp4');frames=0;chapters=[];error=None
        protocol=json.loads((a.experiment/height/'baseline/protocol.json').read_text());samples={s['id']:s for s in protocol['samples']}
        try:
            with mujoco.Renderer(model,height=288,width=384) as renderer,imageio.get_writer(path,fps=50,codec='libx264',quality=7) as writer:
                for sid in ('diagnostic_0','diagnostic_13','diagnostic_14'):
                    guard();traces=[];hashes={}
                    for branch in BRANCHES:
                        source=a.experiment/height/branch/'traces'/(sid+'.npz');hashes[branch]=sha(source)
                        with np.load(source) as f:traces.append({k:f[k] for k in f.files})
                    sample=samples[sid];expected=round(sample['case']['duration']*100)+1
                    complete=all(len(t['qpos'])==expected for t in traces);start=frames
                    for tick in range(0,min(len(t['qpos']) for t in traces)-1,2):
                        guard();panels=[];t=tick*.01
                        for branch,trace in zip(BRANCHES,traces):
                            q=trace['qpos'][tick];data.qpos[:]=q;mujoco.mj_forward(model,data)
                            camera.lookat[:]=q[:3];camera.lookat[2]-=.025
                            renderer.update_scene(data,camera=camera,scene_option=option)
                            for x in range(int(np.floor(q[0]))-2,int(np.floor(q[0]))+3):
                                for y in range(int(np.floor(q[1]))-2,int(np.floor(q[1]))+3):
                                    if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                                    g=renderer.scene.geoms[renderer.scene.ngeom]
                                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([.05,.05,.003]),np.array([float(x),float(y),.003]),np.eye(3).reshape(9),np.array([.95,.7,.2,1.]))
                                    renderer.scene.ngeom+=1
                            panel=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(panel);draw.rectangle((0,0,384,61),fill='#18212b')
                            j=min(tick,len(trace['compensation'])-1);speed=float(np.linalg.norm(trace['linear_world'][tick*5,:2]))
                            draw.text((6,5),f'{branch} | GPU | {height} | t={t:.2f}s',fill='white')
                            draw.text((6,23),f'vx raw={trace["user_command"][j,0]:+.2f} used={trace["policy_command"][j,0]:+.3f}',fill='white')
                            draw.text((6,41),f'speed={speed:.3f}m/s I={trace["integral_before"][j,0]:+.3f}m',fill='white')
                            panels.append(panel)
                        canvas=Image.new('RGB',(1152,320),'#101820');draw=ImageDraw.Draw(canvas)
                        draw.text((8,8),f'8000 frozen model | {sid} | complete startup/braking | normal speed 1x | yellow ground marks 1m',fill='white')
                        for b,panel in enumerate(panels):canvas.paste(panel,(384*b,32))
                        writer.append_data(np.asarray(canvas));frames+=1
                    chapters.append(dict(sample_id=sid,start_s=start/50,duration_s=(frames-start)/50,expected_s=sample['case']['duration'],complete=complete,trajectory_sha256=hashes))
        except Exception as exc:error=repr(exc)
        item=dict(height=height,path=str(path),frames=frames,fps=50,chapters=chapters,error=error,complete=error is None and len(chapters)==3 and all(c['complete'] for c in chapters))
        if path.exists():
            item['sha256']=sha(path)
            if error is None:
                guard();reader=imageio.get_reader(path);count=reader.count_frames();reader.get_data(count-1);fps=reader.get_meta_data()['fps'];reader.close()
                item['decoded_frames']=count;item['decode_checked']=count==frames==1850 and fps==50
                item['complete'] &=item['decode_checked']
        index.append(item);(a.experiment/'video-index.json').write_text(json.dumps(index,indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if k!='chapters'}),flush=True)
        if error:raise RuntimeError(error)

if __name__=='__main__':main()
