"""GPU early-response diagnostic: source-row alignment hypotheses, not real delay fitting."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction
from .replay_usb_pd_targets import sha,write


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(exist_ok=False);torch.set_num_threads(1)
    source=a.experiment/'mujoco-target-replay-01/inputs.npz';initial=a.experiment/'short-window-final-01/trajectories.npz'
    with np.load(source) as f:inputs={k:f[k].copy() for k in f.files}
    with np.load(initial) as f:
        all_starts=f['starts'];valid=all_starts>=5;starts=all_starts[valid];n=len(starts)
        q0=f['qpos'][0,:len(all_starts)][valid].copy();v0=f['initial_qvel'][:len(all_starts)][valid].copy()
    profiles=[dict(group='all',offset_ms=0)]+[dict(group=group,offset_ms=ms) for group in ('all','wheels','legs') for ms in (-10,-6,-2,2,6,10)]
    write(out/'plan.json',dict(profiles=profiles,starts_s=(starts*.002).tolist(),duration_s=.02,
        meaning='Negative offset uses earlier logged requests; positive offset uses future requests ONLY to diagnose timestamp/measurement alignment, never a deployable controller.',
        limitations='Velocity sample age and actual motor current/shaft torque unmeasured; lower error after shifting does not identify physical lag.',
        selection='Report group-wise fit/validation errors; no automatic adoption or training.',source_sha256=sha(source),initial_sha256=sha(initial)))
    shutil.copy2(__file__,out/'source.py')
    cfg=FudanWalkCfg();cfg.control.leg_stiffness=60.;cfg.control.leg_damping=2.;cfg.control.wheel_stiffness=.1
    env=FudanEnv(n*len(profiles),seed=30001,randomized=False,disturbances=DisturbanceConfig(),cfg=cfg)
    try:
        sim=env.env.sim;action=env.action;device=env.device;vids=action._motor_v_adr;qids=action._motor_q_adr
        motors=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]
        sim.data.qpos.copy_(torch.tensor(np.tile(q0,(len(profiles),1)),device=device,dtype=torch.float32));sim.data.qvel.copy_(torch.tensor(np.tile(v0,(len(profiles),1)),device=device,dtype=torch.float32))
        sim.data.ctrl.zero_();sim.forward();env.env.scene.update(dt=0.)
        velocities=[sim.data.qvel[:,vids].cpu().numpy().copy()];positions=[sim.data.qpos.cpu().numpy().copy()];written=[];max_error=0.;began=time.time()
        for step in range(10):
            action.process_actions(torch.zeros(env.num_envs,6,device=device));HuananhuPdAction.apply_actions(action)
            requests=[]
            for c in profiles:
                r=inputs['recorded_torque'][starts+step].copy();ids={'all':list(range(6)),'legs':[0,1,3,4],'wheels':[2,5]}[c['group']]
                shifted=inputs['recorded_torque'][starts+step+c['offset_ms']//2]
                r[:,ids]=shifted[:,ids];requests.append(r)
            request=torch.tensor(np.concatenate(requests),device=device,dtype=torch.float32)
            action._entity.set_joint_effort_target(request,joint_ids=action._physical_joint_ids);env.env.scene.write_data_to_sim()
            max_error=max(max_error,float(abs(sim.data.ctrl[:,motors]-request).max()));written.append(sim.data.ctrl[:,motors].cpu().numpy().copy())
            sim.step();env.env.scene.update(dt=.002);velocities.append(sim.data.qvel[:,vids].cpu().numpy().copy());positions.append(sim.data.qpos.cpu().numpy().copy())
        v=np.asarray(velocities);q=np.asarray(positions);assert np.isfinite(v).all() and np.isfinite(q).all() and max_error==0
        np.savez_compressed(out/'traces.npz',motor_velocity=v,qpos=q,request=np.asarray(written),starts=starts)
        rows=[]
        for j,c in enumerate(profiles):
            row=dict(profile=c,errors=[])
            for step in (1,5,10):
                error=v[step,j*n:(j+1)*n]-inputs['velocity'][starts+step]
                for part,mask in (('first16s',starts<8000),('last8s',starts>=8000)):
                    e=error[mask];row['errors'].append(dict(time_ms=step*2,part=part,
                        per_motor_RMS_rad_s=np.sqrt(np.mean(e**2,axis=0)).tolist(),legs_RMS_rad_s=float(np.sqrt(np.mean(e[:,[0,1,3,4]]**2))),wheels_RMS_rad_s=float(np.sqrt(np.mean(e[:,[2,5]]**2)))))
            rows.append(row)
        write(out/'result.json',dict(complete=True,backend='mujoco_warp_gpu',worlds=env.num_envs,steps_per_world=10,wall_s=time.time()-began,request_write_error_Nm=max_error,rows=rows,training_updates=0,policy_forwards=0,adopted=False))
        print(json.dumps(dict(complete=True,worlds=env.num_envs,steps_per_world=10,request_write_error_Nm=max_error)))
    finally:env.close()


if __name__=='__main__':main()
