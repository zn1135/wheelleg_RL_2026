"""GPU short-window recorded-torque/PD-target comparison of wheel response tau."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from .replay_usb_pd_targets import sha,write
from .check_usb_pd_short_windows import gravity


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profiles',type=Path);p.add_argument('--rolling-only',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    source=a.experiment/'mujoco-target-replay-01/inputs.npz';initial=a.experiment/'short-window-final-01'
    with np.load(source) as f:inputs={k:f[k].copy() for k in f.files}
    plan=json.loads((initial/'plan.json').read_text());profiles=[];qs=[];vs=[]
    candidates=json.loads(a.profiles.read_text()) if a.profiles else [dict(tau_ms=t) for t in (0,25,30,35)]
    if a.profiles:
        assert candidates[0]==dict(name='baseline') and len(candidates)<=24
        assert all(all(0<=c.get(key,1)<=4 for key in ('armature_scale','damping_scale','frictionloss_scale')) for c in candidates)
    with np.load(initial/'trajectories.npz') as f:
        starts=f['starts'].copy();n=len(starts)
        for mode in ('recorded_torque','PD_targets'):
            for j,c in enumerate(plan['profiles']):
                if c['plant']!='baseline':continue
                if a.rolling_only and c['name']!='rolling':continue
                for candidate in candidates:
                    profile=dict(mode=mode,initial=c['name'],tau_ms=candidate.get('tau_ms',0))
                    if a.profiles:profile['parameters']=candidate
                    profiles.append(profile)
                    qs.append(f['qpos'][0,j*n:(j+1)*n].copy());vs.append(f['initial_qvel'][j*n:(j+1)*n].copy())
    write(out/'plan.json',dict(profiles=profiles,starts_s=(starts*.002).tolist(),window_s=.2,
        source_sha256=sha(source),initial_sha256=sha(initial/'trajectories.npz'),
        criterion='Relative diagnostic gate only: in each initialization and both time partitions, wheel RMS <=90% baseline and leg/gravity RMS <=105% baseline. Not an absolute hardware tolerance or whole-robot acceptance.',
        caveat='Logged torque is controller request, not measured shaft torque. Base velocity/contact and initial motor response state remain assumptions.',training_updates=0))
    shutil.copy2(__file__,out/'source.py')
    cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    count=len(profiles)*n;env=FudanEnv(count,seed=30001,randomized=False,disturbances=DisturbanceConfig(),cfg=cfg)
    try:
        sim=env.env.sim;action=env.action;device=env.device;legs=[0,1,3,4];wheels=[2,5]
        qids=action._motor_q_adr;vids=action._motor_v_adr
        motors=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        if a.profiles:
            sim.expand_model_fields(('dof_armature','dof_damping','dof_frictionloss'))
            for j,c in enumerate(profiles):
                for field in ('armature','damping','frictionloss'):
                    getattr(sim.model,'dof_'+field)[j*n:(j+1)*n,vids[wheels]]*=c['parameters'].get(field+'_scale',1.)
            sim.recompute_constants(RecomputeLevel.set_const)
        write(out/'actual-wheel-parameters.json',{name:getattr(sim.model,'dof_'+name)[:,vids[wheels]].cpu().tolist() for name in ('armature','damping','frictionloss')})
        sim.data.qpos.copy_(torch.tensor(np.concatenate(qs),device=device,dtype=torch.float32));sim.data.qvel.copy_(torch.tensor(np.concatenate(vs),device=device,dtype=torch.float32))
        sim.data.ctrl.zero_();sim.forward();env.env.scene.update(dt=0.)
        tau=np.repeat([c['tau_ms'] for c in profiles],n)
        action.wheel_motor.configure(time_constant_ms=tau)
        direct=torch.tensor(np.repeat([c['mode']=='recorded_torque' for c in profiles],n),device=device)
        # 仅使用窗口开始之前的请求初始化响应状态，未来实测运动只参与评分。
        warm={}
        for value in sorted(set(c['tau_ms'] for c in profiles)):
            stream=inputs['recorded_torque'][:,wheels];state=stream[0].copy();before=[]
            alpha=1 if value==0 else -np.expm1(-.002/(value*.001))
            for request in stream:
                before.append(state.copy());state+=alpha*(request-state)
            warm[value]=np.asarray(before)[starts]
        action.wheel_motor.state.copy_(torch.tensor(np.concatenate([warm[c['tau_ms']] for c in profiles]),device=device,dtype=torch.float32))
        previous=action.wheel_motor.state.cpu().numpy().astype(float);alpha=-np.expm1(-.002/np.maximum(tau*.001,1e-12))[:,None]
        positions=[sim.data.qpos.cpu().numpy().copy()];velocities=[sim.data.qvel[:,vids].cpu().numpy().copy()];requests=[];outputs=[];max_error=0.;began=time.time()
        for step in range(100):
            target=torch.tensor(np.tile(inputs['target'][starts+step],(len(profiles),1)),device=device,dtype=torch.float32)
            raw=target.clone();raw[:,legs]=(target[:,legs]-action.nominal_positions[:,legs])/.5;raw[:,wheels]/=10
            action.process_actions(raw);HuananhuPdAction.apply_actions(action)
            request=action._torques.clone();logged=torch.tensor(np.tile(inputs['recorded_torque'][starts+step],(len(profiles),1)),device=device,dtype=torch.float32)
            request[direct]=logged[direct]
            assert (abs(request[:,legs])<=40.0001).all() and (abs(request[:,wheels])<=3.9001).all()
            output=request.clone();output[:,wheels]=action.wheel_motor.apply(request[:,wheels],sim.data.qvel[:,vids[wheels]])
            action._entity.set_joint_effort_target(output,joint_ids=action._physical_joint_ids);env.env.scene.write_data_to_sim()
            written=sim.data.ctrl[:,motors].cpu().numpy();r=request.cpu().numpy();expected=r.copy();previous+=alpha*(r[:,wheels]-previous);expected[:,wheels]=previous
            max_error=max(max_error,float(abs(written-expected).max()));requests.append(r);outputs.append(written)
            sim.step();env.env.scene.update(dt=.002)
            positions.append(sim.data.qpos.cpu().numpy().copy());velocities.append(sim.data.qvel[:,vids].cpu().numpy().copy())
        positions=np.asarray(positions);velocities=np.asarray(velocities);assert np.isfinite(positions).all() and np.isfinite(velocities).all() and max_error<1e-5
        np.savez_compressed(out/'traces.npz',qpos=positions,motor_velocity=velocities,request=np.asarray(requests),output=np.asarray(outputs),starts=starts,tau_ms=tau)
        ticks=starts[None,:]+np.arange(101)[:,None];rows=[]
        for j,c in enumerate(profiles):
            q=positions[:,j*n:(j+1)*n];v=velocities[:,j*n:(j+1)*n]
            leg_error=np.rad2deg(q[:,:,qids.cpu().numpy()[legs]]-inputs['current'][ticks][:,:,legs]);wheel_error=v[:,:,wheels]-inputs['velocity'][ticks][:,:,wheels]
            # 姿态实录为100Hz持有；沿用已冻结的50Hz参考姿态取同一时刻。
            reference=inputs['reference_qpos'][(ticks[::10]//10).ravel(),3:7]
            g=gravity(q[::10,:,3:7].reshape(-1,4));angle=np.rad2deg(np.arccos(np.clip((g*gravity(reference)).sum(-1),-1,1))).reshape(11,n)
            def body_angles(values):return np.c_[np.arctan2(-values[:,1],-values[:,2]),np.arcsin(values[:,0].clip(-1,1))]
            axes_error=np.rad2deg((body_angles(g)-body_angles(gravity(reference))+np.pi)%(2*np.pi)-np.pi).reshape(11,n,2)
            row=dict(profile=c)
            for label,mask in (('first16s',starts<8000),('last8s',starts>=8000)):
                row[label]=dict(leg_RMS_deg=float(np.sqrt(np.mean(leg_error[1:,mask]**2))),wheel_RMS_rad_s=float(np.sqrt(np.mean(wheel_error[1:,mask]**2))),
                    wheel_axes_RMS_rad_s=np.sqrt(np.mean(wheel_error[1:,mask]**2,axis=(0,1))).tolist(),gravity_RMS_deg=float(np.sqrt(np.mean(angle[1:,mask]**2))),
                    roll_RMS_deg=float(np.sqrt(np.mean(axes_error[1:,mask,0]**2))),pitch_RMS_deg=float(np.sqrt(np.mean(axes_error[1:,mask,1]**2))),
                    leg_max_error_deg=float(abs(leg_error[1:,mask]).max()))
            rows.append(row)
        verdicts=[]
        for row in rows:
            c=row['profile']
            if c['tau_ms']==0 and c.get('parameters',{}).get('name','baseline')=='baseline':continue
            base=next(r for r in rows if r['profile']['mode']==c['mode'] and r['profile']['initial']==c['initial'] and r['profile']['tau_ms']==0 and r['profile'].get('parameters',{}).get('name','baseline')=='baseline');ratios={}
            for part in ('first16s','last8s'):ratios[part]={k:row[part][k]/max(base[part][k],1e-12) for k in ('leg_RMS_deg','wheel_RMS_rad_s','gravity_RMS_deg')}
            passed=all(v['wheel_RMS_rad_s']<=.9 and v['leg_RMS_deg']<=1.05 and v['gravity_RMS_deg']<=1.05 for v in ratios.values())
            verdicts.append(dict(profile=c,ratios=ratios,passed=passed))
        write(out/'result.json',dict(complete=True,backend='mujoco_warp_gpu',worlds=count,steps_per_world=100,wall_s=time.time()-began,
            maximum_response_error_Nm=max_error,rows=rows,relative_diagnostic_verdicts=verdicts,training_updates=0,policy_forwards=0,absolute_real2sim_pass=False))
        print(json.dumps(dict(worlds=count,passed=sum(v['passed'] for v in verdicts),total=len(verdicts),rolling_rows=[r for r in rows if r['profile']['initial']=='rolling']),indent=2))
    finally:env.close()


if __name__=='__main__':main()
