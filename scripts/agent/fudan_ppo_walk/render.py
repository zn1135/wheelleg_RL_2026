"""Render saved complete trajectories at normal speed; does not step physics."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',required=True);a=p.parse_args()
    model=mujoco.MjModel.from_binary_path(str(a.model));data=mujoco.MjData(model)
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=640
    with np.load(a.trajectory) as saved:
        qpos=saved['qpos'].copy();velocity=saved['linear_body'].copy();height=saved['position'][:,2].copy()
    from ct_mjlab.fudan_ppo_walk.protocol import samples
    sample=next((s for s in samples() if s['id']==a.trajectory.stem),None)
    protocol=a.trajectory.parent.parent/'protocol.json'
    if sample is None and protocol.exists():
        sample=next((s for s in json.loads(protocol.read_text())['samples'] if s['id']==a.trajectory.stem),None)
    camera=mujoco.MjvCamera();camera.lookat[:]=qpos[:,:3].mean(0)
    camera.distance=max(2.,float(np.linalg.norm(np.ptp(qpos[:,:2],axis=0)))+1.5)
    camera.elevation=-20.;camera.azimuth=110.
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    with mujoco.Renderer(model,height=640,width=960) as renderer:
        with imageio.get_writer(a.output,fps=50,codec='libx264',quality=8) as writer:
            for i in range(0,len(qpos)-1,2):
                data.qpos[:]=qpos[i];mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=option)
                # 固定世界地面标记，每格1m；只添加渲染几何，不改变碰撞或仿真。
                for x in range(int(np.floor(qpos[:,0].min()))-1,int(np.ceil(qpos[:,0].max()))+2):
                    if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_BOX,np.array([.03,.3,.002]),
                        np.array([float(x),0.,.002]),np.eye(3).reshape(9),np.array([.9,.65,.2,1.]))
                    renderer.scene.ngeom+=1
                canvas=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(canvas)
                draw.rectangle((0,0,960,62),fill='#18212b');draw.text((10,8),f'{a.label} t={i*.01:.2f}s | normal speed | ground marks 1m',fill='white')
                cmd=None
                if sample is not None:
                    cmd=sample['case']['segments'][0]['command']
                    for seg in sample['case']['segments']:
                        if i*.01>=seg['start']:cmd=seg['command']
                tick=min(i*5,len(velocity)-1)
                draw.text((10,32),f'command vx/yaw/h={cmd} | actual vx={velocity[tick,0]:.3f}m/s h={height[tick]:.3f}m',fill='white')
                writer.append_data(np.asarray(canvas))
    a.output.with_suffix('.json').write_text(json.dumps(dict(trajectory=str(a.trajectory),trajectory_sha256=sha(a.trajectory),
        video_sha256=sha(a.output),fps=50,duration_s=(len(qpos)-1)*.01,label=a.label),indent=2)+'\n')

if __name__=='__main__':main()
