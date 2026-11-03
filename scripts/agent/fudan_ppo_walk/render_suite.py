"""Normal-speed chaptered videos covering the complete nominal matrix and a random failure."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from ct_mjlab.fudan_ppo_walk.protocol import samples
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--score',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engine',required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads(a.score.read_text());scores={s['sample_id']:s for s in report['samples']}
    protocol_path=a.traces.parent/'protocol.json'
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    source_samples=protocol['samples'] if protocol.get('mode')=='diagnostic_not_formal_acceptance' else samples()
    nominal=[s for s in source_samples if not s['randomized']]
    random_fail=next((s for s in source_samples if s['randomized'] and scores[s['id']]['complete'] and not scores[s['id']]['passed']),None)
    random_success=next((s for s in source_samples if s['randomized'] and scores[s['id']]['complete'] and scores[s['id']]['passed']),None)
    model=mujoco.MjModel.from_binary_path(str(a.model));data=mujoco.MjData(model)
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=640
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    camera=mujoco.MjvCamera();camera.distance=1.8;camera.elevation=-18.;camera.azimuth=110.
    checkpoint_hash=sha(a.checkpoint);metadata=dict(engine=a.engine,checkpoint_sha256=checkpoint_hash,
        mode=protocol.get('mode','formal_987_samples'),
        final_acceptance='pending_user_video_review',nominal_cases_planned=len(nominal),videos=[],missing=[])
    with mujoco.Renderer(model,height=640,width=960) as renderer:
        for name,items in [('nominal_matrix',nominal),('random_failure',[] if random_fail is None else [random_fail]),('random_success',[] if random_success is None else [random_success])]:
            if not items:continue
            video=a.output/(name+'.mp4');chapters=[];frame_count=0
            with imageio.get_writer(video,fps=50,codec='libx264',quality=7) as writer:
                for sample in items:
                    path=a.traces/(sample['id']+'.npz')
                    if time.time()>=a.deadline or not path.exists() or not scores[sample['id']]['complete']:
                        metadata['missing'].append(sample['id']);continue
                    with np.load(path) as saved:trace={k:saved[k] for k in saved.files}
                    q=trace['qpos'];start_frame=frame_count;complete=True
                    for i in range(0,len(q)-1,2):
                        if time.time()>=a.deadline:complete=False;break
                        data.qpos[:]=q[i];mujoco.mj_forward(model,data)
                        camera.lookat[:]=q[i,:3];camera.lookat[2]-=.025
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        # Ground crosses are anchored at integer metre coordinates, not the moving camera.
                        for x in range(int(np.floor(q[i,0]))-2,int(np.floor(q[i,0]))+3):
                            for y in range(int(np.floor(q[i,1]))-2,int(np.floor(q[i,1]))+3):
                                if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                                g=renderer.scene.geoms[renderer.scene.ngeom]
                                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([.05,.05,.003]),
                                    np.array([float(x),float(y),.003]),np.eye(3).reshape(9),np.array([.95,.7,.2,1.]))
                                renderer.scene.ngeom+=1
                        canvas=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(canvas)
                        t=i*.01;tick=min(i*5,len(trace['time'])-1);cmd=sample['case']['segments'][0]['command']
                        for segment in sample['case']['segments']:
                            if t>=segment['start']:cmd=segment['command']
                        status='technical PASS' if scores[sample['id']]['passed'] else 'technical FAIL'
                        lines=[f'{a.engine} | {sample["id"]} | model {checkpoint_hash[:12]} | {status}',
                            f't={t:.2f}s  command vx={cmd[0]:+.2f}m/s yaw={cmd[1]:+.2f}rad/s height={cmd[2]:.2f}m',
                            f'actual vx={trace["linear_body"][tick,0]:+.3f}m/s yaw={trace["angular_body"][tick,2]:+.3f}rad/s h={trace["position"][tick,2]:.3f}m',
                            f'pitch={trace["roll_pitch_deg"][tick,1]:+.1f}deg body={trace["base_contact"][tick]:.1f}N xy=({q[i,0]:+.2f},{q[i,1]:+.2f})m | 1x speed | yellow marks 1m']
                        draw.rectangle((0,0,960,90),fill='#17212b')
                        for line,text in enumerate(lines):draw.text((10,6+20*line),text,fill='white')
                        writer.append_data(np.asarray(canvas));frame_count+=1
                    chapters.append(dict(sample_id=sample['id'],start_s=start_frame/50,duration_s=(frame_count-start_frame)/50,
                        complete=complete,technical_pass=scores[sample['id']]['passed'],trajectory_sha256=sha(path)))
                    if not complete:metadata['missing'].append(sample['id'])
            metadata['videos'].append(dict(path=str(video),sha256=sha(video),fps=50,frames=frame_count,chapters=chapters))
    (a.output/'index.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(engine=a.engine,videos=len(metadata['videos']),missing=len(metadata['missing']))))

if __name__=='__main__':main()
