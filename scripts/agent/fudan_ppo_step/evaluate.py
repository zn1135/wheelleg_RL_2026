"""Bounded GPU single-platform matrix and offline trajectory scoring; never learns."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_step.config import TASK, FudanStepCfg, recipe_record
from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.envs.fudan_ppo_step.runner import validate_source
from scripts.agent.motion.assess import phase_verdict
from .check import source_fingerprint


def samples():
    return [dict(id=f'step_{i:03d}',mode=mode,height=height,speed=speed,angle_deg=angle,friction=mu,reverse=reverse)
        for i,(mode,height,speed,angle,mu,reverse) in enumerate(itertools.product(
            ('up','down'),(.20,.21,.22),(.15,.30,.50),(0.,-10.,10.,-20.,20.),(.4,.8,1.2),(False,True)))]


def longest(mask):
    best=run=0
    for value in mask:
        run=run+1 if value else 0;best=max(best,run)
    return best


def score(sample, trace, *, dt=.01, duration=20., max_approach_speed=.8,
          allow_temporary_support=False, landing_clearance_m=.45):
    valid=np.asarray(trace['valid'],bool)
    n=int(valid.sum())
    if n==0 or not valid[:n].all() or valid[n:].any():
        return dict(sample_id=sample['id'],passed=False,reason='invalid_or_empty_trace')
    if 'terrain_valid' in trace and not np.asarray(trace['terrain_valid'][:n]).all():
        return dict(sample_id=sample['id'],passed=False,complete=False,mode=sample['mode'],reason='missing_terrain')
    fields=('position','wheel_pos','linear_world','linear_body','angular_body','gravity','command','force','support','support_height')
    finite=all(np.isfinite(trace[k][:n]).all() for k in fields)
    if not finite:return dict(sample_id=sample['id'],passed=False,reason='nonfinite')
    origin=trace['origin'];direction=1 if sample['mode']=='up' else -1
    approach_distance=1.2 if sample['mode']=='up' else .6
    wheel=trace['wheel_pos'][:n];dist=(wheel[...,0]-origin[0])*direction-approach_distance
    target=sample['height'] if sample['mode']=='up' else 0.
    ground=trace['support_height'][:n]
    lane=(abs(wheel[...,1]-origin[1])<1.148/2-.04).all(-1)
    on_top=(abs(ground-target)<.01).all(-1)
    loaded=trace['support'][:n].astype(bool).all(-1)
    reached=(dist.min(-1)>landing_clearance_m)&lane&on_top&loaded
    speed=np.linalg.norm(trace['linear_world'][:n,:2],axis=-1)
    approach=(dist.min(-1)>-.5)&(dist.min(-1)<.1)
    approach_speed=trace['speed_peak'][:n] if 'speed_peak' in trace else speed
    max_speed=float(approach_speed[approach].max()) if approach.any() else None
    forces=trace['force'][:n]
    if forces.shape[1:]!=(10,2):raise ValueError('Expected ten1ms body/leg samples per policy step')
    any_contact=(forces>1.).any(axis=(1,2))
    flat_force=forces.reshape(-1,2)
    strict_contact_violation=(longest((flat_force>1.).any(-1))*.001>=.20 or
               longest((flat_force>float(trace['weight'])*.1).any(-1))*.001>=.10)
    forbidden=strict_contact_violation and not allow_temporary_support
    gravity=trace['gravity'][:n]
    pitch=np.degrees(np.arcsin(np.clip(gravity[:,0],-1,1)))
    roll=np.degrees(np.arctan2(-gravity[:,1],-gravity[:,2]))
    stopped=(abs(trace['command'][:n,:2]).max(-1)<1e-6)
    normal_support=trace['normal_support'][:n] if 'normal_support' in trace else loaded
    stable=reached&stopped&~any_contact&normal_support&(speed<=.05)&(abs(pitch)<=5)&(abs(roll)<=5)
    count=round(2./dt);run=0;end=None
    for i,ok in enumerate(stable):
        run=run+1 if ok else 0
        if run>=count:end=i+1;break
    selected=np.zeros(n,bool)
    if end is not None:selected[end-count:end]=True
    motion=phase_verdict(vx=trace['linear_body'][:n,0],yaw=trace['angular_body'][:n,2],
        horizontal_speed=speed,pitch_deg=pitch,vx_target=0.,yaw_target=0.,selected=selected,
        valid=np.ones(n,bool),complete=end is not None,parking=True,
        command_vx=trace['command'][:n,0],command_yaw=trace['command'][:n,1])
    passed=bool(end is not None and motion['passed'] and not forbidden and max_speed is not None and max_speed<=max_approach_speed)
    return dict(sample_id=sample['id'],passed=passed,complete=bool(end is not None or n>=round(duration/dt)),
        max_approach_speed=max_speed,forbidden_support=bool(forbidden),parking=motion,
        strict_contact_violation=bool(strict_contact_violation),temporary_support_allowed=allow_temporary_support,
        nonwheel_contact_rate=(flat_force>1).mean(0).tolist(),nonwheel_peak_N=flat_force.max(0).tolist(),
        stable_stop_s=longest(stable)*dt,mode=sample['mode'])


def assess_recording(directory):
    from scripts.agent.motion.acceptance import envelope
    directory=Path(directory);protocol=json.loads((directory/'step-protocol.json').read_text())
    if protocol.get('backend')!='mujoco_warp_gpu' or protocol.get('task')!=TASK:
        raise ValueError('Step acceptance requires an identified Warp GPU recording')
    rows=[]
    step=protocol['recipe']['step']
    contact_policy=dict(allow_temporary_support=step.get('allow_temporary_support',False),
                       landing_clearance_m=step.get('landing_clearance_m',.45))
    for sample in protocol['samples']:
        path=directory/'traces'/f"{sample['id']}.npz"
        if not path.exists():rows.append(dict(sample_id=sample['id'],passed=False,complete=False,reason='missing_trace'));continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['trace_sha256'][sample['id']]:
            raise ValueError('Step trajectory fingerprint differs')
        with np.load(path,allow_pickle=False) as data:rows.append(score(sample,data,duration=protocol['duration_s'],
            max_approach_speed=step.get('max_approach_speed',.8) if contact_policy['allow_temporary_support'] else .8,
            **contact_policy))
    result=envelope(TASK,directory,rows,{'parking'},True,details=dict(
        checkpoint_sha256=protocol['checkpoint_sha256'],task_recipe=protocol['recipe'],
        scope='Single-platform traversal; flat walking regression must be supplied separately'))
    full=protocol['samples']==samples() and protocol['duration_s']==20.
    coverage=sorted({'step_'+row['mode'] for row in rows if 'mode' in row})
    if any(row.get('parking',{}).get('expected_samples',0)>0 for row in rows):coverage.append('parking')
    missing=[name for name in ('step_up','step_down','parking') if name not in coverage]
    if not full:missing.append('full_step_matrix')
    result.update(cohort_sha256=protocol['cohort_sha256'],coverage=coverage,
        missing_capabilities=missing+['flat_walk_regression'],step_matrix_complete=full,
        step_thresholds_met=bool(full and rows and all(row['passed'] for row in rows)),
        technical_thresholds_met=False,full_task_accepted=False)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--limit',type=int,help='Diagnostic subset only; cannot pass the full matrix')
    p.add_argument('--seconds',type=float,default=20.)
    a=p.parse_args()
    if a.batch_size<1 or a.seconds<=0 or a.limit is not None and a.limit<1:p.error('Positive bounds required')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'traces').mkdir()
    cases=samples()[:a.limit];cfg=FudanStepCfg();cfg.step.curriculum=False
    report=dict(task=TASK,backend='mujoco_warp_gpu',samples=cases,duration_s=a.seconds,dt=.01,
        recipe=recipe_record(cfg),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        optimizer_updates=0,physics_steps=0,world_physics_transitions=0,policy_batch_calls=0,
        trace_sha256={},source_fingerprint=source_fingerprint())
    cohort=hashlib.sha256();env=None;torch.set_num_threads(1)
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Evaluation cannot train')):
            policy,saved=load_policy(a.checkpoint,device='cuda:0')
            validate_source(saved,cfg)
            for start in range(0,len(cases),a.batch_size):
                if time.time()>=a.deadline:raise TimeoutError('Evaluation budget expired')
                batch=cases[start:start+a.batch_size];n=len(batch)
                env=StepEnv(n,seed=32001+start,randomized=False,cfg=cfg)
                ids=torch.arange(n,device=env.device)
                env.terrain.terrain_types[:]=torch.tensor([1 if c['mode']=='up' else 2 for c in batch],device=env.device)
                env.terrain.terrain_levels[:]=torch.tensor([cfg.step.heights.index(c['height']) for c in batch],device=env.device)
                env.env._step_fixed_yaw=torch.deg2rad(torch.tensor([c['angle_deg'] for c in batch],device=env.device))
                env.env._step_fixed_reverse=torch.tensor([c['reverse'] for c in batch],device=env.device)
                env.fixed_commands=torch.tensor([[c['speed']*(-1 if c['reverse'] else 1),0.,cfg.step.height_range[1]] for c in batch],device=env.device)
                mu=torch.tensor([c['friction'] for c in batch],device=env.device)
                env.env.sim.model.geom_friction[:,:,0]=mu[:,None]
                env.env._fudan_parameters['friction'].copy_(mu)
                obs,_,history=env.reset();env.contacts.reset(ids)
                env.contacts.record_substeps=True
                origin=env.env.scene.env_origins.cpu().numpy().copy()
                initial=env.env.sim.data.qpos.cpu().numpy().copy();initial[:,:3]-=origin
                cohort.update(initial.tobytes());cohort.update(mu.cpu().numpy().tobytes())
                cohort.update(env.action.disturbances.action_steps.cpu().numpy().tobytes())
                cohort.update(env.action.disturbances.sensor_steps.cpu().numpy().tobytes())
                records=[];alive=torch.ones(n,dtype=torch.bool,device=env.device)
                weights=env.env.sim.model.body_mass[:,env.robot.indexing.body_ids].sum(-1).expand(n).cpu().numpy()*9.81
                with torch.inference_mode():
                    for step in range(round(a.seconds/env.dt)):
                        if time.time()>=a.deadline:raise TimeoutError('Evaluation budget expired')
                        command=env.commands.clone();valid=alive.clone()
                        action,_=policy.act_inference(obs,history)
                        report['policy_batch_calls']+=1
                        action[~alive]=0.
                        obs,_,_,done,info,history=env.step(action,auto_reset=False)
                        alive &= ~done
                        record=dict(position=env.position,wheel_pos=env.env.sim.data.xpos[:,env.wheel_ids],
                            linear_world=env.robot.data.root_link_lin_vel_w,linear_body=env.linear,
                            angular_body=env.angular,gravity=env.gravity,command=command,valid=valid,
                            support=env.contacts.supported,support_height=env.support_height,
                            normal_support=env.normal_support,
                            terrain_valid=info['terrain_valid'],
                            force=torch.stack(env.contacts.step_forces,dim=1),
                            speed_peak=env.contacts.step_speed_peak,
                            contact_position=torch.stack(env.contacts.step_positions,dim=1))
                        records.append({k:v.detach().cpu().numpy().copy() for k,v in record.items()})
                        if not alive.any():break
                        dead=(~alive).nonzero().flatten()
                        if len(dead):
                            # 终止样本仍留在分母；复位其物理体，防止污染其他仍有效的样本。
                            env.reset_idx(dead);env.commands[dead,:2]=0.
                            obs,_=env.observations()
                            env.history[:, -25:]=env._history_frame
                            env.history[dead]=env._history_frame[dead].repeat(1,5)
                            history=env.history
                for i,sample in enumerate(batch):
                    arrays={key:np.stack([r[key][i] for r in records]) for key in records[0]}
                    path=a.output/'traces'/f"{sample['id']}.npz"
                    np.savez_compressed(path,**arrays,origin=origin[i],weight=weights[i],initial=initial[i],
                                        time=np.arange(1,len(records)+1)*env.dt)
                    report['trace_sha256'][sample['id']]=hashlib.sha256(path.read_bytes()).hexdigest()
                report['physics_steps']+=env.contacts.total_ticks
                report['world_physics_transitions']+=env.contacts.total_ticks*n
                env.close();env=None
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        report['cohort_sha256']=cohort.hexdigest()
        (a.output/'step-protocol.json').write_text(json.dumps(report,indent=2)+'\n')
    verdict=assess_recording(a.output)
    (a.output/'diagnostic.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verdict.items() if k not in ('samples','details')}))


if __name__=='__main__':main()
