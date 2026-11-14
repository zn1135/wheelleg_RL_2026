"""Short-window GPU response check; bounds unmeasured base-velocity/contact initialization."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
from ct_mjlab.envs.fudan_ppo_walk import height_reward
from .replay_usb_pd_targets import PoseSolver,sha,write
from .validate_usb_physics import FIELDS


def gravity(q):
    w,x,y,z=q.T
    return np.stack((2*(w*y-x*z),-2*(w*x+y*z),-(1-2*(x*x+y*y))),axis=-1)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scan',action='store_true');p.add_argument('--profiles',type=Path);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    selection=json.loads((a.fit/'comparison.json').read_text());best=selection['selected_index']
    parameters=json.loads((a.fit/'actual-parameters.json').read_text())
    with np.load(a.experiment/'mujoco-target-replay-01/inputs.npz') as f:inputs={k:f[k].copy() for k in f.files}
    with (a.experiment/'rl_valid.csv').open() as file:
        reader=csv.reader(file);names=next(reader);records=np.asarray(list(reader),dtype=float)
    angular=records[:,[names.index(f'obs_{i}') for i in range(3)]]*4
    starts=np.arange(0,len(inputs['time_s'])-100,250);steps=100;count=len(starts)
    settings=[dict(name='rolling',dv=0.,clearance=-.0003),dict(name='rolling_vx_minus02',dv=-.2,clearance=-.0003),
        dict(name='rolling_vx_plus02',dv=.2,clearance=-.0003),dict(name='rolling_gap2mm',dv=0.,clearance=.002)]
    profiles=[dict(setting,plant=plant,index=idx) for setting in settings for plant,idx in (('baseline',0),('candidate',best))]
    if a.scan or a.profiles:
        candidates=[dict(name='baseline')]
        for key,values in (('wheel_armature_scale',(.5,2.)),('spring_bias_n',(-25.,25.)),
                           ('front_armature_scale',(.5,2.)),('rear_armature_scale',(.5,2.)),
                           ('all_leg_armature_scale',(.5,2.)),('spring_damping',(25.,100.))):
            candidates += [dict(name=f'{key}_{value:g}',**{key:value}) for value in values]
        if a.profiles:candidates=json.loads(a.profiles.read_text())
        assert len(candidates)<=24 and candidates[0]['name']=='baseline'
        profiles=[dict(setting,plant=c['name'],index=0,parameters=c) for setting in (settings[:1] if a.scan else settings) for c in candidates]
    n=count*len(profiles);cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    height_reward.configure(cfg,None)
    wrapper=FudanEnv(n,seed=30001,randomized=False,disturbances=DisturbanceConfig(),cfg=cfg)
    sim=wrapper.env.sim;action=wrapper.action;device=wrapper.device;began=time.time()
    write(out/'plan.json',dict(profiles=profiles,starts_s=(starts*.002).tolist(),window_s=.2,
        source_inputs_sha256=sha(a.experiment/'mujoco-target-replay-01/inputs.npz'),fit_selection_sha256=sha(a.fit/'fit-selection.json'),
        initialization='Measured motor q/qd and published attitude/rates; passive velocity from geometry. Root height from geometry. Root linear velocity from mean wheel no-slip contact Jacobian; sensitivity +/-0.2m/s and2mm gap.',
        scope='Repeated real-state initialization, not continuous free rollout or policy balance acceptance; real slip/body translation/contact load unmeasured',training_updates=0))
    shutil.copy2(__file__,out/'check-source.py')
    try:
        model=sim.mj_model;native=mujoco.MjData(model);solver=PoseSolver();q0=[];v0=[]
        wheels=[model.body('robot/'+name).id for name in ('lfwheel','rfwheel')]
        for start in starts:
            motor=inputs['current'][start].copy();motor[[2,5]]=0
            qa,_=solver.solve(motor);qb,_=solver.solve(motor+1e-4*inputs['velocity'][start])
            velocity=np.zeros(model.nv);mujoco.mj_differentiatePos(model,velocity,1e-4,qa,qb)
            pose=inputs['reference_qpos'][start//10].copy();velocity[:3]=0;velocity[3:6]=angular[start]
            native.qpos[:]=pose;mujoco.mj_kinematics(model,native);mujoco.mj_comPos(model,native)
            contact_velocity=[]
            for body in wheels:
                point=native.xpos[body].copy();point[2]-=.060
                jac=np.zeros((3,model.nv));mujoco.mj_jac(model,native,jac,None,point,body)
                contact_velocity.append(jac@velocity)
            velocity[:3]=-np.mean(contact_velocity,axis=0)
            q0.append(pose);v0.append(velocity)
        q0=np.asarray(q0);v0=np.asarray(v0);qs=[];vs=[]
        sim.expand_model_fields(FIELDS)
        for j,profile in enumerate(profiles):
            ids=slice(j*count,(j+1)*count);idx=profile['index']
            for name in FIELDS:
                getattr(sim.model,name)[ids]=torch.as_tensor(parameters[name][idx],device=device)
            action._spring_force_bias[ids]=torch.as_tensor(parameters['spring_bias_n'][idx],device=device)
            action.spring_damping[ids]=torch.as_tensor(parameters['spring_damping'][idx],device=device)
            physical=profile.get('parameters',{})
            base_id=model.body('robot/base_link').id
            sim.model.body_mass[ids,base_id]*=physical.get('base_mass_scale',1.)
            sim.model.body_inertia[ids,base_id]*=physical.get('base_mass_scale',1.)
            # 分别检查惯量和气弹簧；不把PD误差较小的组合直接当成真实机械参数。
            motor_dofs=action._motor_v_adr
            for motors,key in (([0,3],'front_armature_scale'),([1,4],'rear_armature_scale'),
                               ([0,1,3,4],'all_leg_armature_scale'),([2,5],'wheel_armature_scale')):
                sim.model.dof_armature[ids,motor_dofs[motors]]*=physical.get(key,1.)
            if 'leg_damping' in physical:sim.model.dof_damping[ids,motor_dofs[[0,1,3,4]]]=physical['leg_damping']
            if 'leg_frictionloss' in physical:sim.model.dof_frictionloss[ids,motor_dofs[[0,1,3,4]]]=physical['leg_frictionloss']
            if 'spring_bias_n' in physical:action._spring_force_bias[ids]=physical['spring_bias_n']
            if 'spring_damping' in physical:action.spring_damping[ids]=physical['spring_damping']
            pose=q0.copy();velocity=v0.copy();pose[:,2]+=profile['clearance']-.002
            w,x,y,z=pose[:,3:7].T;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            velocity[:,0]+=profile['dv']*np.cos(yaw);velocity[:,1]+=profile['dv']*np.sin(yaw)
            qs.append(pose);vs.append(velocity)
        sim.recompute_constants(RecomputeLevel.set_const)
        spring_enabled=torch.as_tensor([profile.get('parameters',{}).get('gas_spring_enabled',True) for profile in profiles],device=device).repeat_interleave(count)
        np.savez_compressed(out/'actual-parameters.npz',**{k:getattr(sim.model,k).cpu().numpy() for k in FIELDS},
            spring_bias_n=action._spring_force_bias.cpu().numpy(),spring_damping=action.spring_damping.cpu().numpy(),gas_spring_enabled=spring_enabled.cpu().numpy())
        sim.data.qpos.copy_(torch.as_tensor(np.concatenate(qs),device=device,dtype=torch.float32))
        sim.data.qvel.copy_(torch.as_tensor(np.concatenate(vs),device=device,dtype=torch.float32))
        sim.data.ctrl.zero_();sim.forward();wrapper.env.scene.update(dt=0.)
        qrecords=[sim.data.qpos.cpu().numpy().copy()];vrecords=[sim.data.qvel[:,action._motor_v_adr].cpu().numpy().copy()]
        legs=[0,1,3,4];wheels=[2,5];dofs=action._motor_v_adr;qids=action._motor_q_adr
        motor_ids=[model.actuator('robot/'+name+'_motor').id for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        error=torch.zeros((),device=device)
        for step in range(steps):
            if time.time()-began>180:raise TimeoutError('Short-window check exceeded180s')
            target=torch.as_tensor(np.tile(inputs['target'][starts+step],(len(profiles),1)),device=device,dtype=torch.float32)
            raw=target.clone();raw[:,legs]=(target[:,legs]-action.nominal_positions[:,legs])/.5;raw[:,wheels]=target[:,wheels]/10
            action.process_actions(raw);action.apply_actions();wrapper.env.scene.write_data_to_sim()
            spring_control=sim.data.ctrl[:,action._spring_ids]
            spring_control[~spring_enabled]=0.
            sim.data.ctrl[:,action._spring_ids]=spring_control
            assert not sim.data.ctrl[~spring_enabled][:,action._spring_ids].count_nonzero()
            expected=torch.empty_like(target)
            expected[:,legs]=(60*(target[:,legs]-sim.data.qpos[:,qids[legs]])-2*sim.data.qvel[:,dofs[legs]]).clamp(-40,40)
            expected[:,wheels]=(.1*(target[:,wheels]-sim.data.qvel[:,dofs[wheels]])).clamp(-3.9,3.9)
            error=torch.maximum(error,abs(sim.data.ctrl[:,motor_ids]-expected).max())
            sim.step();wrapper.env.scene.update(dt=.002)
            qrecords.append(sim.data.qpos.cpu().numpy().copy());vrecords.append(sim.data.qvel[:,dofs].cpu().numpy().copy())
        qrecords=np.asarray(qrecords);vrecords=np.asarray(vrecords);assert np.isfinite(qrecords).all() and float(error)<1e-4
        np.savez_compressed(out/'trajectories.npz',qpos=qrecords,motor_velocity=vrecords,starts=starts,initial_qvel=np.concatenate(vs))
        rows=[];qindices=qids.cpu().numpy();ticks=starts[None,:]+np.arange(steps+1)[:,None]
        for j,profile in enumerate(profiles):
            q=qrecords[:,j*count:(j+1)*count];v=vrecords[:,j*count:(j+1)*count]
            leg_error=np.rad2deg(q[:,:,qindices[legs]]-inputs['current'][ticks][:,:,legs])
            wheel_error=v[:,:,wheels]-inputs['velocity'][ticks][:,:,wheels]
            reference=inputs['reference_qpos'][(ticks[::10]//10).ravel(),3:7]
            g=gravity(q[::10,:,3:7].reshape(-1,4));angles=np.rad2deg(np.arccos(np.clip((g*gravity(reference)).sum(-1),-1,1))).reshape(11,count)
            values={}
            for label,mask in (('fit',starts<8000),('internal_validation',starts>=8000)):
                values[label]=dict(leg_RMSE_deg=np.sqrt(np.mean(leg_error[1:,mask]**2,axis=(0,1))).tolist(),
                    combined_leg_RMSE_deg=float(np.sqrt(np.mean(leg_error[1:,mask]**2))),
                    combined_wheel_RMSE_rad_s=float(np.sqrt(np.mean(wheel_error[1:,mask]**2))),
                    gravity_direction_RMSE_deg=float(np.sqrt(np.mean(angles[1:,mask]**2))))
            rows.append(dict(profile=profile,**values))
        write(out/'result.json',dict(complete=True,backend='mujoco_warp_gpu',worlds=n,steps_per_world=steps,
            wall_s=time.time()-began,maximum_PD_formula_error_nm=float(error),rows=rows,training_updates=0,
            source_to_measurement='Initial and future motor measurements are separate; future measured motion used only for error scoring',adopted=False))
        print(json.dumps(rows,indent=2),flush=True)
    finally:wrapper.close()


if __name__=='__main__':main()
