"""Bounded mechanical-parameter comparison using logged targets, with a time holdout."""
import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from mjlab.managers.event_manager import RecomputeLevel

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
from ct_mjlab.envs.fudan_ppo_walk import height_reward
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
from .replay_usb_pd_targets import sha,write


def profiles():
    values=[dict(name='baseline')]
    values += [dict(name=f'spring_bias_{n:+g}N',spring_bias_n=n) for n in (-25.,25.,50.,75.)]
    values += [dict(name=f'spring_damping_{n:g}',spring_damping=n) for n in (25.,100.)]
    values += [dict(name=f'front_armature_x{n:g}',front_armature_scale=n) for n in (.5,2.)]
    values += [dict(name=f'wheel_armature_x{n:g}',wheel_armature_scale=n) for n in (.5,2.)]
    values += [dict(name='state_vx015',initial_vx=.15,nuisance_only=True),
               dict(name='state_pitch_minus2deg',initial_pitch_delta_deg=-2.,nuisance_only=True)]
    return values


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profiles',type=Path)
    p.add_argument('--max-wall-seconds',type=float,default=600.)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    candidates=json.loads(a.profiles.read_text()) if a.profiles else profiles();assert candidates[0]['name']=='baseline'
    assert 1<=len(candidates)<=24
    with np.load(a.source/'inputs.npz') as f:data={k:f[k].copy() for k in f.files}
    plan=dict(profiles=candidates,fit_time_s=[0.,16.],validation_time_s=[16.,float(data['time_s'][-1])],
        inputs_sha256=sha(a.source/'inputs.npz'),PD=dict(leg_p=60.,leg_d=2.,wheel_p=.1),
        spring_force_bounds_N=[0,150],maximum_wall_seconds=a.max_wall_seconds,
        initialization='Same recorded joint state and estimated base state for every plant candidate; all independent GPU worlds placed at origin',
        selection='Only0-16s leg, published gravity direction and wheel errors; holdout after fit selection is frozen; nuisance-only states cannot become plant parameters',
        limitations='One previously inspected motion record; time holdout is internal validation, not new independent hardware data',
        source_baseline=str(a.source.resolve()),training_updates=0)
    write(out/'plan.json',plan);shutil.copy2(__file__,out/'fit-source.py')
    cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    height_reward.configure(cfg,None)
    wrapper=FudanEnv(len(candidates),seed=30001,randomized=False,disturbances=DisturbanceConfig(),cfg=cfg)
    env=wrapper.env;sim=env.sim;action=wrapper.action;device=wrapper.device;n=len(candidates)
    start=time.time();qrecords=[];vrecords=[];trecords=[]
    try:
        assert action.kp_scale.eq(1).all() and action.kd_scale.eq(1).all() and action.motor_strength.eq(1).all()
        assert not action._wheel_speed_shaping and not action.wheel_motor.enabled
        sim.expand_model_fields(('dof_armature','dof_damping','dof_frictionloss','body_mass','body_inertia','body_ipos','geom_friction'))
        dofs=action._motor_v_adr;legs=[0,1,3,4];wheels=[2,5];front=[0,3]
        rear=[1,4];base=sim.mj_model.body('robot/base_link').id
        q=torch.as_tensor(data['initial_qpos'],device=device,dtype=torch.float32).expand(n,-1).clone()
        v=torch.as_tensor(data['initial_qvel'],device=device,dtype=torch.float32).expand(n,-1).clone()
        for i,c in enumerate(candidates):
            sim.model.dof_armature[i,dofs[front]]*=c.get('front_armature_scale',1.)
            sim.model.dof_armature[i,dofs[rear]]*=c.get('rear_armature_scale',1.)
            sim.model.dof_armature[i,dofs[wheels]]*=c.get('wheel_armature_scale',1.)
            sim.model.body_mass[i,base]*=c.get('base_mass_scale',1.)
            sim.model.body_inertia[i,base]*=c.get('base_mass_scale',1.)
            sim.model.body_ipos[i,base,0]+=c.get('base_com_x_m',0.)
            if 'sliding_friction' in c:sim.model.geom_friction[i,:,0]=c['sliding_friction']
            if 'front_damping' in c:sim.model.dof_damping[i,dofs[front]]=c['front_damping']
            if 'front_frictionloss' in c:sim.model.dof_frictionloss[i,dofs[front]]=c['front_frictionloss']
            action._spring_force_bias[i]=c.get('spring_bias_n',0.)
            action.spring_damping[i]=c.get('spring_damping',50.)
            if 'initial_vx' in c:v[i,0]=c['initial_vx']
            if 'initial_vy' in c:v[i,1]=c['initial_vy']
            if 'initial_vz' in c:v[i,2]=c['initial_vz']
            if 'initial_height_delta_m' in c:q[i,2]+=c['initial_height_delta_m']
            if 'initial_pitch_delta_deg' in c:
                xyzw=data['initial_qpos'][3:7][[1,2,3,0]]
                rpy=Rotation.from_quat(xyzw).as_euler('xyz');rpy[1]+=np.deg2rad(c['initial_pitch_delta_deg'])
                q[i,3:7]=torch.as_tensor(Rotation.from_euler('xyz',rpy).as_quat()[[3,0,1,2]],device=device,dtype=torch.float32)
        sim.recompute_constants(RecomputeLevel.set_const)
        sim.data.qpos.copy_(q);sim.data.qvel.copy_(v);sim.data.ctrl.zero_();sim.forward();env.scene.update(dt=0.)
        motor_ids=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ACTUATED_JOINTS]
        mujoco.mj_saveModel(sim.mj_model,str(out/'template-model.mjb'))
        actual_parameters=dict(dof_armature=sim.model.dof_armature.cpu().tolist(),
            dof_damping=sim.model.dof_damping.cpu().tolist(),dof_frictionloss=sim.model.dof_frictionloss.cpu().tolist(),
            body_mass=sim.model.body_mass.cpu().tolist(),body_inertia=sim.model.body_inertia.cpu().tolist(),
            body_ipos=sim.model.body_ipos.cpu().tolist(),geom_friction=sim.model.geom_friction.cpu().tolist(),
            spring_bias_n=action._spring_force_bias.cpu().tolist(),spring_damping=action.spring_damping.cpu().tolist())
        write(out/'actual-parameters.json',actual_parameters)
        target=torch.as_tensor(data['target'],device=device,dtype=torch.float32)
        raw=target.clone();raw[:,legs]=(target[:,legs]-action.nominal_positions[0,legs])/.5;raw[:,wheels]=target[:,wheels]/10
        qrecords.append(sim.data.qpos.cpu().numpy().copy());vrecords.append(sim.data.qvel[:,dofs].cpu().numpy().copy())
        error=torch.zeros((),device=device);spring_min=150.;spring_max=0.;failure=None
        for step in range(len(target)-1):
            if step%500==0 and time.time()-start>a.max_wall_seconds:raise TimeoutError('Bounded GPU parameter comparison expired')
            action.process_actions(raw[step].expand(n,-1));action.apply_actions();env.scene.write_data_to_sim()
            ctrl=sim.data.ctrl[:,motor_ids];qn=sim.data.qpos[:,action._motor_q_adr];vn=sim.data.qvel[:,dofs]
            expected=torch.empty_like(ctrl);expected[:,legs]=(60*(target[step,legs]-qn[:,legs])-2*vn[:,legs]).clamp(-40,40)
            expected[:,wheels]=(.1*(target[step,wheels]-vn[:,wheels])).clamp(-3.9,3.9)
            error=torch.maximum(error,abs(ctrl-expected).max());trecords.append(ctrl.cpu().numpy().copy())
            sf=sim.data.ctrl[:,action._spring_ids].cpu().numpy();spring_min=min(spring_min,float(sf.min()));spring_max=max(spring_max,float(sf.max()))
            sim.step();env.scene.update(dt=.002)
            row=sim.data.qpos.cpu().numpy().copy();qrecords.append(row);vrecords.append(sim.data.qvel[:,dofs].cpu().numpy().copy())
            if not np.isfinite(row).all():failure=dict(step=step,candidate_ids=np.flatnonzero(~np.isfinite(row).all(1)).tolist());break
            if step%1000==0:print(json.dumps(dict(stage='parameter_comparison',step=step,total=len(target)-1,worlds=n)),flush=True)
        qrecords=np.asarray(qrecords);vrecords=np.asarray(vrecords);trecords=np.asarray(trecords)
        np.savez_compressed(out/'candidate-trajectories.npz',qpos=qrecords,motor_velocity=vrecords,torque=trecords,time_s=data['time_s'][:len(qrecords)])
        write(out/'execution.json',dict(complete=failure is None and len(qrecords)==len(target),failure=failure,
            explicit_GPU_steps_per_world=len(qrecords)-1,worlds=n,wall_s=time.time()-start,
            maximum_PD_formula_error_nm=float(error),spring_force_minmax_N=[spring_min,spring_max],optimizer_updates=0))
        assert failure is None and len(qrecords)==len(target)
        assert float(error)<1e-4 and spring_min>=0 and spring_max<=150
        qids=action._motor_q_adr.cpu().numpy()
        score(out,data,candidates,qrecords,vrecords,qids)
    finally:wrapper.close()


def metrics(data,q,v,qids,mask):
    legs=[0,1,3,4];wheels=[2,5]
    diff=np.rad2deg(q[mask][:,qids[legs]]-data['current'][mask][:,legs])
    wheel=v[mask][:,wheels]-data['velocity'][mask][:,wheels]
    w,x,y,z=q[mask,3:7].T
    tilt=np.rad2deg(np.maximum(abs(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))),abs(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))))
    indices=data['reference_indices'];indices=indices[mask[indices]]
    def gravity(quat):
        w,x,y,z=quat.T
        return np.stack((2*(w*y-x*z),-2*(w*x+y*z),-(1-2*(x*x+y*y))),axis=-1)
    reference=data['reference_qpos'][mask[data['reference_indices']],3:7]
    cosine=np.sum(gravity(reference)*gravity(q[indices,3:7]),axis=1)
    angle=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
    return dict(leg_RMSE_deg=np.sqrt(np.mean(diff**2,axis=0)).tolist(),leg_signed_mean_deg=diff.mean(0).tolist(),
        leg_combined_RMSE_deg=float(np.sqrt(np.mean(diff**2))),wheel_RMSE_rad_s=np.sqrt(np.mean(wheel**2,axis=0)).tolist(),
        wheel_combined_RMSE_rad_s=float(np.sqrt(np.mean(wheel**2))),max_abs_roll_pitch_deg=float(tilt.max()),
        published_gravity_direction_RMSE_deg=float(np.sqrt(np.mean(angle**2))))


def score(out,data,candidates,q,v,qids):
    fit=data['time_s']<16.;test=~fit
    fit_rows=[metrics(data,q[:,i],v[:,i],qids,fit) for i in range(len(candidates))];base=fit_rows[0]
    for c,r in zip(candidates,fit_rows):
        rear=np.asarray(r['leg_RMSE_deg'])[[1,3]];original=np.asarray(base['leg_RMSE_deg'])[[1,3]]
        penalty=float(np.maximum(rear-original-.3,0).sum())
        r['fit_score']=.35*r['leg_combined_RMSE_deg']/base['leg_combined_RMSE_deg']+.4*r['published_gravity_direction_RMSE_deg']/base['published_gravity_direction_RMSE_deg']+.25*r['wheel_combined_RMSE_rad_s']/base['wheel_combined_RMSE_rad_s']+penalty
        if c.get('nuisance_only') or r['max_abs_roll_pitch_deg']>60:r['fit_score']=None
    eligible=[i for i,r in enumerate(fit_rows) if r['fit_score'] is not None]
    best=min(eligible,key=lambda i:fit_rows[i]['fit_score'])
    write(out/'fit-selection.json',dict(index=best,profile=candidates[best],fit_metrics=fit_rows[best],
        selection_uses_only_time_before16s=True,adopted=False))
    validation=[metrics(data,q[:,i],v[:,i],qids,test) for i in range(len(candidates))]
    rows=[dict(profile=c,fit=f,validation=t) for c,f,t in zip(candidates,fit_rows,validation)]
    write(out/'comparison.json',dict(rows=rows,selected_profile=candidates[best],selected_index=best,
        initial_conditions_identical_for_plant_candidates=True,motor_qpos_indices=qids.tolist(),
        scope='Internal time holdout on one already-inspected real record; not independent hardware validation',adopted=False))
    print(json.dumps(dict(selected=candidates[best],rows=rows),indent=2),flush=True)


if __name__=='__main__':main()
