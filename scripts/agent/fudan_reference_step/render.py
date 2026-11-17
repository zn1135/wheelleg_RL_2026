"""Render saved trajectories at real-time speed; offline kinematics only."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .replay import mujoco_rotation, write, sha


def render(out, run, case):
    directory=out/'runs'/run
    meta=json.loads((directory/'metadata.json').read_text())
    with np.load(directory/(case+'.npz')) as saved:t={k:saved[k] for k in saved.files}
    target=out/'media'/f'{run}_{case}_fullview';target.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(directory/'model.mjb'));d=mujoco.MjData(m)
    m.vis.global_.offwidth=800;m.vis.global_.offheight=480
    renderer=mujoco.Renderer(m,height=480,width=800)
    cam=mujoco.MjvCamera();cam.azimuth=90.;cam.elevation=-15.;cam.distance=2.7
    edge=0. if meta['kind']=='single' else -10.3
    duration=t['time'][-1];fps=50;frames=round(duration*fps)
    vpath=target/'replay-1x.mp4'
    wids=[m.body(n).id for n in ('l_wheel_Link','r_wheel_Link')]
    base=m.body('base_Link_del').id
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==base]
    clearance=[];lengths=[]
    snap_indices=[]
    dist=t['wheel_pos'][...,0]-edge
    side=(t['wheel_side_force']>1).any(-1)&(t['time']>1)
    face=((t['base_force']>1)&(abs(t['base_contact_pos'][:,0]-edge)<.035))|((t['leg_force']>1)&(abs(t['leg_contact_pos'][:,0]-edge)<.035))
    idx=np.where((side|face)&(t['time']>1))[0];event=idx[0] if len(idx) else np.argmin(abs(dist.min(-1)))
    candidates=[max(0,event-100),event,min(len(dist)-1,event+100),int(np.argmax(t['wheel_pos'][...,2].min(-1))),len(dist)-1]
    for i in candidates:
        if i not in snap_indices:snap_indices.append(i)
    writer=imageio.get_writer(vpath,fps=fps,codec='libx264',quality=8,macro_block_size=None)
    keyframes=[]
    def draw(index):
        d.qpos[:]=t['qpos'][index];mujoco.mj_kinematics(m,d)
        cam.lookat[:]=[float(d.qpos[0])+.1,float(d.qpos[1]),float(d.qpos[2])+.12]
        renderer.update_scene(d,camera=cam)
        panel=Image.fromarray(renderer.render());pen=ImageDraw.Draw(panel)
        pen.rectangle((0,0,800,64),fill='#18212b')
        pen.text((8,5),f'Fudan original Upstairs ONNX | GPU MuJoCo | 1x | t={t["time"][index]:.3f}s',fill='white')
        pen.text((8,23),f'height={meta["height"]:.3f}m | vx={t["qvel"][index,0]:+.2f}m/s | pitch={t["pitch_deg"][index]:+.1f}deg',fill='white')
        pen.text((8,41),f'wheel support={t["wheel_support"][index].round(1)} N | body={t["base_force"][index]:.1f} N | leg={t["leg_force"][index]:.1f} N',fill='white')
        return panel
    try:
        for frame in range(frames):
            i=min(len(t['time'])-1,round(frame/fps/.002))
            writer.append_data(np.asarray(draw(i)))
        for k,i in enumerate(snap_indices):
            path=target/f'key-{k:02d}.png';draw(i).save(path);keyframes.append(dict(path=str(path),time_s=float(t['time'][i])))
        for i in range(0,len(t['time']),5):
            d.qpos[:]=t['qpos'][i];mujoco.mj_kinematics(m,d)
            lower=[]
            for g in geoms:
                mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];b=a+m.mesh_vertnum[mid]
                v=m.mesh_vert[a:b]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                ground=np.where((v[:,0]>=0)&(v[:,0]<=2.09)&(abs(v[:,1])<=meta['width']/2),meta['height'],0.) if meta['kind']=='single' else np.zeros(len(v))
                lower.append(float((v[:,2]-ground).min()))
            clearance.append([t['time'][i],min(lower)])
            hips=np.array([d.xanchor[m.joint(n).id] for n in ('lf0_Joint','rf0_Joint')])
            lengths.append([t['time'][i],*np.linalg.norm(d.xpos[wids]-hips,axis=1)])
    finally:writer.close();renderer.close()
    reader=imageio.get_reader(vpath);count=reader.count_frames();reader.get_data(count-1);info=reader.get_meta_data();reader.close()
    assert count==frames and info['fps']==fps
    tt=t['time'];fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True)
    axes[0].plot(tt,dist);axes[0].axhline(0,c='k',ls='--');axes[0].set_ylabel('wheel x - edge (m)')
    axes[1].plot(tt,t['wheel_pos'][...,2]);axes[1].axhline(meta['height']+.06,c='k',ls='--');axes[1].set_ylabel('wheel z (m)')
    axes[2].plot(tt,np.rad2deg(t['virtual_q'][:,[0,1,3,4]]));axes[2].plot(tt,t['pitch_deg'],c='k',label='body pitch');axes[2].set_ylabel('virtual joints (deg)');axes[2].legend()
    axes[3].plot(tt,t['wheel_support']);axes[3].plot(tt,t['base_force'],label='body');axes[3].plot(tt,t['leg_force'],label='leg');axes[3].set_ylabel('contact force (N)');axes[3].legend()
    axes[4].plot(tt,t['motor_torque']);axes[4].set_ylabel('actual motor request (Nm)');axes[4].set_xlabel('simulation time (s)')
    for ax in axes:ax.grid(alpha=.3);ax.axvline(tt[event],c='r',alpha=.4)
    fig.suptitle(run+' / '+case);fig.tight_layout();fig.savefig(target/'mechanism.png',dpi=130);plt.close(fig)
    np.savez_compressed(target/'geometry-audit.npz',body_clearance=np.asarray(clearance),hip_wheel_length=np.asarray(lengths))
    contact_window=np.asarray(clearance);lengths=np.asarray(lengths)
    audit=dict(video=str(vpath),frames=count,fps=fps,duration_s=count/fps,sha256=sha(vpath),keyframes=keyframes,
        body_clearance_min_m=float(contact_window[:,1].min()),hip_wheel_length_minmax_m=[lengths[:,1:].min(0).tolist(),lengths[:,1:].max(0).tolist()],
        note='Kinematic mesh vertex-to-top height; broad-phase hull shape and soft penetration can differ. Contact forces remain the dynamic evidence.')
    write(target/'media.json',audit);print(json.dumps(audit),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--case',required=True)
    a=p.parse_args();render(a.output,a.run,a.case)


if __name__=='__main__':main()
