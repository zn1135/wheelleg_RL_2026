"""Replay logged PD targets in MuJoCo Warp; measured motion is a kinematic reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS,_SIDES,closed_chain_residual
from ct_mjlab.envs.fudan_ppo_walk.reset_geometry import ResetGeometry,mesh_vertices

ORDER=('lf','lr','lw','rf','rr','rw')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


class PoseSolver:
    def __init__(self):
        source=ResetGeometry();self.model=source.model;self.data=source.data;self.qaddr=source.qaddr
        self.collision=source.collision
        self.data.qpos[:]=self.model.qpos0
        for name,value in source.baseline.items():self.data.qpos[self.qaddr[name]]=value
        self.data.qpos[:7]=[0,0,0,1,0,0,0]

    def solve(self,motors):
        model,data=self.model,self.data
        data.qpos[:7]=[0,0,0,1,0,0,0]
        for name,value in zip(ACTUATED_JOINTS,motors):data.qpos[self.qaddr[name]]=value
        for side,hip,knee,passive,pairs,_ in _SIDES:
            names=(knee,*passive[1:]);ids=[self.qaddr[n] for n in names]
            bounds=[model.joint(n).range if model.joint(n).limited else [model.qpos0[self.qaddr[n]]-np.pi,model.qpos0[self.qaddr[n]]+np.pi] for n in names]
            lower,upper=np.asarray(bounds).T;sites=[(model.site(a).id,model.site(b).id) for a,b in pairs]
            def residual(values):
                data.qpos[ids]=values;mujoco.mj_kinematics(model,data)
                return np.concatenate([data.site_xpos[a]-data.site_xpos[b] for a,b in sites])
            result=least_squares(residual,np.clip(data.qpos[ids],lower+1e-10,upper-1e-10),bounds=(lower,upper),
                xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=150)
            residual(result.x)
        mujoco.mj_kinematics(model,data)
        for side,parent in (('left','lf0'),('right','rf0')):
            delta=data.body(parent).xmat.reshape(3,3).T@(data.site(side+'_gas_spring_lower').xpos-data.site(side+'_gas_spring_upper').xpos)
            quaternion=model.body(side+'_spring1').quat;angle=2*np.arctan2(quaternion[2],quaternion[0])
            data.qpos[self.qaddr[side+'_spring1_joint']]=np.arctan2(delta[0],delta[2])-angle
            data.qpos[self.qaddr[side+'_spring2_joint']]=np.hypot(delta[0],delta[2])-model.body(side+'_spring2').pos[2]
        mujoco.mj_kinematics(model,data)
        error=closed_chain_residual(model,data)
        return data.qpos.copy(),error

    def ground(self,q,quat):
        self.data.qpos[:]=q;self.data.qpos[:7]=[0,0,0,*quat];mujoco.mj_kinematics(self.model,self.data)
        bottom=min(float(mesh_vertices(self.model,self.data,i)[:,2].min()) for i in self.collision)
        q=self.data.qpos.copy();q[2]=-bottom+.002
        return q


def prepare(experiment,out):
    out.mkdir(parents=True,exist_ok=False)
    with (experiment/'rl_valid.csv').open() as f:
        reader=csv.reader(f);names=next(reader);values=np.asarray(list(reader),dtype=float)
    d={n:values[:,i] for i,n in enumerate(names)};n=len(values)
    assert np.all(np.diff(d['sequence'])==1) and np.all(np.diff(d['h7_us'])>0)
    times=(d['h7_us']-d['h7_us'][0])/1e6;dt=.002;grid=np.arange(n)*dt
    current=np.stack([d['current_'+m] for m in ORDER],axis=1)
    target=np.stack([d['target_'+m] for m in ORDER],axis=1)
    velocity=np.stack([d['velocity_rad_s_'+m] for m in ORDER],axis=1)
    torque=np.stack([d['pd_model_nm_'+m] for m in ORDER],axis=1)
    gravity=np.stack([d[f'obs_{i}'] for i in (3,4,5)],axis=1)
    gravity/=np.linalg.norm(gravity,axis=1,keepdims=True)
    roll=np.arctan2(-gravity[:,1],-gravity[:,2]);pitch=np.arcsin(gravity[:,0].clip(-1,1))
    angular=np.stack([d[f'obs_{i}']*4 for i in (0,1,2)],axis=1)
    yaw_rate=(np.sin(roll)*angular[:,1]+np.cos(roll)*angular[:,2])/np.cos(pitch)
    yaw=np.r_[0,np.cumsum(.5*(yaw_rate[:-1]+yaw_rate[1:])*np.diff(times))]
    quats=Rotation.from_euler('xyz',np.c_[roll,pitch,yaw]).as_quat()[:,[3,0,1,2]]
    wheel_angle=np.vstack((np.zeros(2),np.cumsum(.5*(velocity[:-1][:,[2,5]]+velocity[1:][:,[2,5]])*np.diff(times)[:,None],axis=0)))
    solver=PoseSolver();motor0=current[0].copy();motor0[[2,5]]=0
    q0,error0=solver.solve(motor0);epsilon=1e-4;qnext,_=solver.solve(motor0+epsilon*velocity[0])
    v0=np.zeros(solver.model.nv);mujoco.mj_differentiatePos(solver.model,v0,epsilon,q0,qnext)
    q0=solver.ground(q0,quats[0]);v0[:3]=0;v0[3:6]=angular[0]
    for i,name in enumerate(ACTUATED_JOINTS):v0[solver.model.joint(name).dofadr[0]]=velocity[0,i]
    indices=np.arange(0,n-1,10);poses=[];errors=[]
    for k,i in enumerate(indices):
        motors=current[i].copy();motors[[2,5]]=wheel_angle[i]
        q,error=solver.solve(motors);q=solver.ground(q,quats[i]);poses.append(q);errors.append(error)
        if k%200==0:print(json.dumps(dict(stage='kinematic_reference',frame=k,total=len(indices))),flush=True)
    np.savez_compressed(out/'inputs.npz',time_s=grid,source_time_s=times,current=current,target=target,velocity=velocity,
        recorded_torque=torque,initial_qpos=q0,initial_qvel=v0,reference_qpos=np.asarray(poses),reference_indices=indices,
        reference_loop_error_m=np.asarray(errors),commands=np.c_[d['obs_6']/2,d['obs_7']/.25,d['obs_8']/5])
    report=dict(source_csv=str((experiment/'rl_valid.csv').resolve()),source_sha256=sha(experiment/'rl_valid.csv'),rows=n,
        duration_s=float(grid[-1]),maximum_timestamp_grid_error_us=float(abs(times-grid).max()*1e6),
        maximum_reference_loop_error_m=float(max(errors)),initial_loop_error_m=float(error0),
        initial_height_geometry_estimate_m=float(q0[2]),joint_names=solver.qaddr,
        assumptions=['Initial root XY/yaw set0 and linear velocity0; roll/pitch use published policy gravity, which may contain a bias',
            'Initial height inferred from model collision geometry and2mm clearance, not recorded height',
            'Reference root XY fixed; reference Z fitted to ground for display; yaw/wheel angles integrated and not absolute measurements',
            'Passive joints inferred from current model geometry; closure residual reported; no measured body contacts/friction/mass corrections'],
        source_to_control='One continuous recorded PD row per2ms; no additional action/sensor delay',optimizer_updates=0)
    write(out/'inputs-report.json',report)


def run(out):
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
    from ct_mjlab.envs.fudan_ppo_walk import height_reward
    torch.set_num_threads(1)
    with np.load(out/'inputs.npz') as f:a={k:f[k].copy() for k in f.files}
    cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    height_reward.configure(cfg,None)
    wrapper=FudanEnv(1,seed=30001,randomized=False,disturbances=DisturbanceConfig(),cfg=cfg)
    env=wrapper.env;sim=env.sim;action=wrapper.action;device=wrapper.device
    try:
        assert not action._wheel_speed_shaping and not action.wheel_motor.enabled
        assert action.kp_scale.eq(1).all() and action.kd_scale.eq(1).all() and action.motor_strength.eq(1).all()
        assert sim.mj_model.opt.timestep==.002
        q=torch.as_tensor(a['initial_qpos'],device=device,dtype=torch.float32)
        v=torch.as_tensor(a['initial_qvel'],device=device,dtype=torch.float32)
        assert sim.data.qpos.shape[1]==len(q) and sim.data.qvel.shape[1]==len(v)
        sim.data.qpos[0]=q;sim.data.qvel[0]=v;sim.data.ctrl.zero_();sim.forward();env.scene.update(dt=0.)
        targets=torch.as_tensor(a['target'],device=device,dtype=torch.float32)
        raw=targets.clone();legs=[0,1,3,4];wheels=[2,5]
        raw[:,legs]=(targets[:,legs]-action.nominal_positions[0,legs])/action.cfg.leg_position_scale
        raw[:,wheels]=targets[:,wheels]/action.cfg.wheel_velocity_scale
        motor_ids=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ACTUATED_JOINTS]
        for name,index in zip(ACTUATED_JOINTS,action._motor_q_adr.cpu().tolist()):
            assert sim.mj_model.joint('robot/'+name).qposadr[0]==index
        mujoco.mj_saveModel(sim.mj_model,str(out/'model.mjb'))
        rows=[sim.data.qpos[0].cpu().numpy().copy()];velocities=[sim.data.qvel[0,action._motor_v_adr].cpu().numpy().copy()]
        controls=[];max_error=torch.zeros((),device=device);started=time.time();failure=None
        for i in range(len(targets)-1):
            action.process_actions(raw[i:i+1]);action.apply_actions();env.scene.write_data_to_sim()
            actual=sim.data.ctrl[0,motor_ids]
            qnow=sim.data.qpos[0,action._motor_q_adr];vnow=sim.data.qvel[0,action._motor_v_adr]
            expected=torch.empty_like(actual);expected[legs]=(60*(targets[i,legs]-qnow[legs])-2*vnow[legs]).clamp(-40,40)
            expected[wheels]=(.1*(targets[i,wheels]-vnow[wheels])).clamp(-3.9,3.9)
            max_error=torch.maximum(max_error,abs(actual-expected).max());controls.append(actual.cpu().numpy().copy())
            sim.step();env.scene.update(dt=.002)
            state=sim.data.qpos[0].cpu().numpy().copy();rows.append(state)
            velocities.append(sim.data.qvel[0,action._motor_v_adr].cpu().numpy().copy())
            if not np.isfinite(state).all():failure=dict(step=i,error='nonfinite');break
            if i%1000==0:print(json.dumps(dict(stage='GPU_PD_targets',step=i,total=len(targets)-1)),flush=True)
        rows=np.asarray(rows);velocities=np.asarray(velocities);controls=np.asarray(controls)
        np.savez_compressed(out/'gpu-replay.npz',qpos=rows,motor_velocity=velocities,torque=controls,time_s=a['time_s'][:len(rows)])
        error=float(max_error);assert error<1e-4,error
        columns=action._motor_q_adr.cpu().numpy();difference=rows[:,columns[legs]]-a['current'][:len(rows),legs]
        report=dict(complete=failure is None and len(rows)==len(targets),failure=failure,backend='mujoco_warp_gpu',
            explicit_physics_steps=len(rows)-1,wall_s=time.time()-started,dt_s=.002,PD_target_formula_max_error_nm=error,
            first_torque_vs_recorded_max_error_nm=float(abs(controls[0]-a['recorded_torque'][0]).max()),
            leg_RMSE_deg=np.rad2deg(np.sqrt(np.mean(difference**2,axis=0))).tolist(),
            motor_qpos_indices=columns.tolist(),wheel_physics=wrapper.wheel_physics_profile,
            model_sha256=sha(out/'model.mjb'),inputs_sha256=sha(out/'inputs.npz'),optimizer_updates=0,policy_forwards=0,
            scope='Open-loop replay of logged targets; simulated state feeds PD; global initial state/contact assumptions prevent full real-robot fidelity claims')
        write(out/'gpu-report.json',report)
        print(json.dumps(report,indent=2),flush=True)
    finally:wrapper.close()


def render(out,follow_heading=False):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    with np.load(out/'inputs.npz') as f:a={k:f[k].copy() for k in f.files}
    with np.load(out/'gpu-replay.npz') as f:simulated=f['qpos'].copy()
    report=json.loads((out/'gpu-report.json').read_text());assert report['complete']
    model=mujoco.MjModel.from_binary_path(str(out/'model.mjb'));disable_model_sameframe_shortcuts(model)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480;data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.distance=1.5;camera.elevation=-16.;camera.azimuth=90.
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    qids=np.asarray(report['motor_qpos_indices']);legs=[0,1,3,4]
    output=out/('recorded-targets-vs-measured-side.mp4' if follow_heading else 'recorded-targets-vs-measured.mp4');count=0
    with mujoco.Renderer(model,width=640,height=480) as renderer,imageio.get_writer(output,fps=50,codec='libx264',quality=8) as writer:
        for k,i in enumerate(a['reference_indices']):
            if i>=len(simulated):break
            canvas=Image.new('RGB',(1280,528),'#18212b');draw=ImageDraw.Draw(canvas)
            cmd=a['commands'][i];draw.text((12,8),f'1x NORMAL | t={i*.002:.2f}s | logged cmd vx={cmd[0]:+.2f}, yaw={cmd[1]:+.2f}, h={cmd[2]:.2f}',font=font,fill='white')
            for column,(q,title) in enumerate(((a['reference_qpos'][k],'MEASURED JOINTS / kinematic reference'),(simulated[i],'MuJoCo GPU / logged PD targets'))):
                data.qpos[:]=q;mujoco.mj_forward(model,data);camera.lookat[:]=[q[0],q[1],.22]
                if follow_heading:
                    w,x,y,z=q[3:7]
                    camera.azimuth=90+np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                renderer.update_scene(data,camera=camera,scene_option=option);panel=Image.fromarray(renderer.render().copy());text=ImageDraw.Draw(panel)
                text.rectangle((0,0,640,92),fill='#18212b');text.text((10,5),title,font=font,fill='white')
                if column==0:
                    text.text((10,32),'XY fixed / Z auto-ground / heading integrated',font=font,fill='#ffca75')
                    text.text((10,58),'Only joint angles are direct measured positions',font=font,fill='#ffca75')
                else:
                    error=np.rad2deg(np.sqrt(np.mean((q[qids[legs]]-a['current'][i,legs])**2)))
                    plant='log-fit candidate' if 'physical_candidate' in report else 'current mechanical plant'
                    text.text((10,32),f'leg P60/D2 / wheel P0.1 / {plant}',font=font,fill='white')
                    text.text((10,58),f'sim h={q[2]:.3f}m / instantaneous leg RMS={error:.1f}deg',font=font,fill='white')
                canvas.paste(panel,(column*640,48))
            frame=np.asarray(canvas);writer.append_data(frame);count+=1
            if k in (0,250,750):imageio.imwrite(out/f'{"side-" if follow_heading else ""}frame-{k:04d}.png',frame)
    write(out/('side-video.json' if follow_heading else 'video.json'),dict(path=str(output),fps=50,frames=count,duration_s=count/50,sha256=sha(output),
        physics_steps=0,policy_forwards=0,source_gpu_report_sha256=sha(out/'gpu-report.json'),camera_follows_heading=follow_heading))
    if follow_heading:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for ax,j,name in zip(axes,legs,('lf','lr','rf','rr')):
        ax.plot(a['time_s'],np.rad2deg(a['target'][:,j]),label='Recorded target',lw=.8)
        ax.plot(a['time_s'],np.rad2deg(a['current'][:,j]),label='Measured',lw=.9)
        ax.plot(a['time_s'][:len(simulated)],np.rad2deg(simulated[:,qids[j]]),label='GPU simulated',lw=.8)
        ax.set_ylabel(name+' (deg)');ax.grid(alpha=.2);ax.legend(ncol=3)
    axes[0].set_title('Logged PD target replay / initial body state and contacts estimated')
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'leg-comparison.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','render'))
    p.add_argument('--follow-heading',action='store_true',help='Keep both panels in body-relative side view for joint comparison')
    p.add_argument('--experiment',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.experiment.resolve(),a.output.resolve())
    elif a.mode=='run':run(a.output.resolve())
    else:render(a.output.resolve(),a.follow_heading)


if __name__=='__main__':main()
