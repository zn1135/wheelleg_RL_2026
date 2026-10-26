#!/usr/bin/env python3
"""Fast frozen-BC screening from cached initial states; one candidate gets full40 replay."""
import argparse,copy,gc,json,math,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.os import dump_yaml
from ct_mjlab.bc import load_actor
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import huananhu_randomization,multimode_observations
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import configure_smp
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT
from check_lqr_student_closed_loop import digest


def nominal_wrench_state(env,env_ids):
    # Only initialize the wrench buffer; leave nominal model fields unchanged.
    env._huananhu_randomization=huananhu_randomization.HuananhuRandomization(env)


def support(env):
    p=getattr(env,'_fast_probe',None)
    if p is None:return torch.zeros(env.num_envs,device=env.device)
    data=env.scene['robot'].data;g=data.projected_gravity_b
    tilt=torch.maximum(torch.atan2(-g[:,1],-g[:,2]).abs(),torch.asin(g[:,0].clamp(-1,1)).abs())
    base=env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    wheel=-env.scene['wheel_contact'].data.force[:,:,2];weight=env._fast_weight
    balanced=((tilt<=math.radians(35)) & (base<=1.) &
        (wheel>=torch.maximum(.1*weight,torch.full_like(weight,5.))[:,None]).all(-1) &
        (wheel.sum(-1)>=.8*weight) & (wheel.sum(-1)<=1.2*weight) &
        (data.root_link_ang_vel_b[:,:2].norm(dim=-1)<=.5))
    strict=balanced & (tilt<=math.radians(15)) & ((data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]-.33648).abs()<=.03)
    p['hold']=torch.where(p['live'],torch.where(balanced,p['hold']+1,0),p['hold'])
    p['strict_run']=torch.where(p['live'],torch.where(strict,p['strict_run']+1,0),p['strict_run'])
    p['strict_ever']|=p['live'] & (p['strict_run']>=250)
    return balanced.float()


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    previous=json.loads((args.source/'comparison.json').read_text())
    assert digest(args.checkpoint)==previous['checkpoint_sha256'] and digest(args.prior)==previous['prior_sha256']
    cached=torch.load(args.source/'nominal_mean/initial-state.pt',map_location='cpu',weights_only=False)
    actor,saved=load_actor(args.checkpoint,'cuda:0');actor.requires_grad_(False)
    weights={k:v.clone() for k,v in actor.state_dict().items()};results={}
    def evaluate(name,kind,leg_std,wheel_std,seconds,indices):
        print('Starting',name,flush=True);wall=time.monotonic();directory=args.output/name;directory.mkdir()
        nominal,train=task_registry.get_cfgs('chuanliantui',play=True)
        randomized,_=task_registry.get_cfgs('chuanliantui',play=False)
        cfg=copy.deepcopy(randomized if kind=='physics' else nominal)
        cfg.events={k:v for k,v in cfg.events.items() if v.mode!='interval'}
        if kind=='physics':cfg.observations=copy.deepcopy(nominal.observations)
        if kind in ('sensors','sensor_noise','sensor_delay'):
            cfg.observations=copy.deepcopy(randomized.observations)
            if kind!='sensors':
                for group in cfg.observations.values():
                    term=group.terms['proprioception']
                    term.params['cfg']=replace(term.params['cfg'],noise=kind=='sensor_noise',delay=kind=='sensor_delay')
        if kind=='push':
            cfg.events['nominal_wrench_state']=EventTermCfg(func=nominal_wrench_state,mode='startup')
            cfg.events['clear_wrench']=EventTermCfg(func=huananhu_randomization.reset_disturbances,mode='reset')
            cfg.events.update({k:v for k,v in randomized.events.items() if v.mode=='interval'})
        configure_smp(cfg,train,args.prior,play=kind!='physics',policy_contract=HANDOVER_OBSERVATION_CONTRACT)
        cfg.seed=42;cfg.auto_reset=False;cfg.metrics['fast_support']=MetricsTermCfg(func=support,per_substep=True,reduce='last')
        cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False
        cfg.actions['pd'].action_delay_choices_s=(.002,.004)
        env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=len(indices));raw=env.unwrapped
        try:
            raw.reset(seed=42);action=raw.action_manager.get_term('pd');ids=torch.tensor(indices)
            q=cached['qpos'][ids].to(raw.device).clone();v=cached['qvel'][ids].to(raw.device).clone()
            q[:,:2]=raw.scene.env_origins[:,:2];raw.sim.data.qpos.copy_(q);raw.sim.data.qvel.copy_(v);raw.sim.forward();action.reset()
            leg=cached['leg_delay'][ids].to(raw.device);wheel=cached['wheel_delay'][ids].to(raw.device)
            action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
            raw.command_manager.get_term('standup').ground_start[:]=True;raw.command_manager.compute(dt=0.)
            multimode_observations.reset_observations(raw,torch.arange(len(indices),device=raw.device))
            obs=raw.observation_manager.compute(update_history=True)
            body=raw.scene['robot'].indexing.body_ids
            if kind=='physics':
                mass=raw.sim.model.body_mass
                raw._fast_weight=mass[:,body].sum(-1)*9.81
            else:
                raw._fast_weight=torch.full((len(indices),),float(raw.sim.get_default_field('body_mass')[body].sum())*9.81,device=raw.device)
            raw._fast_probe=dict(live=torch.ones(len(indices),dtype=torch.bool,device=raw.device),
                hold=torch.zeros(len(indices),dtype=torch.long,device=raw.device),strict_run=torch.zeros(len(indices),dtype=torch.long,device=raw.device),
                strict_ever=torch.zeros(len(indices),dtype=torch.bool,device=raw.device))
            std=torch.tensor([leg_std,leg_std,wheel_std,leg_std,leg_std,wheel_std],device=raw.device)
            generator=torch.Generator(device=raw.device).manual_seed(20261026);reader=MotionStateReader(raw)
            buffers={k:[] for k in ('qpos','qvel','motor_dq','features','observation','action','mean','noise','alive')}
            for tick in range(round(seconds/.01)):
                live=raw._fast_probe['live'].clone();inputs=obs['actor'].clone();mean=actor.mlp(inputs)
                noise=torch.randn(mean.shape,device=raw.device,generator=generator)*std;output=mean+noise
                obs,_,done,_=env.step(output);state=reader.read()
                values=dict(qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,motor_dq=state['motor_dq'],features=reader.features(),
                    observation=inputs,action=output,mean=mean,noise=noise,alive=live)
                for k,x in values.items():buffers[k].append(x.cpu().numpy().copy())
                if tick<20:assert not action.torques.any()
                raw._fast_probe['live']&=~done.bool()
                if not raw._fast_probe['live'].any():break
                if done.any():
                    raw.reset(env_ids=done.nonzero().flatten());action.action_delay_steps.copy_(leg);action.wheel_action_delay_steps.copy_(wheel)
                    obs=raw.observation_manager.compute(update_history=False)
            d={k:np.asarray(v) for k,v in buffers.items()};rows=[];sample_time=(np.arange(len(d['alive']))+1)*.01
            for i in range(len(indices)):
                n=int(d['alive'][:,i].sum());late=(sample_time[:n]>max(2.,seconds-1. if seconds<20 else 15.))
                rms=float(np.sqrt(np.square(d['motor_dq'][:n,i][:,[0,1,3,4]][late]).mean())) if late.any() else None
                speed=float(np.linalg.norm(d['features'][:n,i,4:6][late],axis=-1).mean()) if late.any() else None
                balance=bool(n==round(seconds/.01) and raw._fast_probe['hold'][i]>=500)
                drift=float(np.linalg.norm(d['qpos'][:n,i,:2][late]-d['qpos'][:n,i,:2][late][0],axis=-1).max()) if late.any() else None
                yaw=float(np.abs(d['features'][:n,i,9][late]).mean()) if late.any() else None
                strict=bool(raw._fast_probe['strict_ever'][i])
                rows.append(dict(env=i,original_env=indices[i],strict_rise=strict,end_balance=balance,observed_s=n*.01,
                    leg_rms=rms,speed=speed,drift=drift,yaw=yaw,
                    qualified=bool(seconds==20 and strict and balance and rms is not None and rms<=.02 and speed<=.01 and drift<=.02 and yaw<=.02)))
            for tick in np.linspace(0,len(d['alive'])-1,8).round().astype(int):
                actual=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
                np.testing.assert_allclose(actual,d['mean'][tick],atol=2e-6,rtol=2e-6)
            np.testing.assert_array_equal(d['action'],d['mean']+d['noise'])
            for k,x in actor.state_dict().items():torch.testing.assert_close(x,weights[k],atol=0,rtol=0)
            np.savez(directory/'trace.npz',**d,time=sample_time,initial_qpos=q.cpu().numpy(),initial_qvel=v.cpu().numpy())
            dump_yaml(directory/'env.yaml',asdict(cfg))
            result=dict(kind=kind,leg_std=leg_std,wheel_std=wheel_std,seconds=seconds,num_envs=len(indices),
                strict_rise=sum(x['strict_rise'] for x in rows),end_balance=sum(x['end_balance'] for x in rows),
                qualified=sum(x['qualified'] for x in rows),mean_leg_rms=float(np.mean([x['leg_rms'] for x in rows if x['leg_rms'] is not None])),
                mean_speed=float(np.mean([x['speed'] for x in rows if x['speed'] is not None])),rows=rows,
                actual_weight_n=raw._fast_weight.cpu().tolist(),trace_sha256=digest(directory/'trace.npz'),
                wall_seconds=time.monotonic()-wall,optimizer_updates=0,actor_unchanged=True,
                scope='4s screening is not20s acceptance; push screen12s. Full candidate only20s. Support uses each environment actual mass.')
            (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[name]=result
            print('Finished',name,result['strict_rise'],result['end_balance'],result['mean_leg_rms'],flush=True)
        finally:env.close()
        gc.collect();torch.cuda.empty_cache()
    subset=[pair*10+i for pair in range(4) for i in range(5)]
    cases=(
        ('baseline','nominal',0.,0.,4),('physics','physics',0.,0.,4),('sensors','sensors',0.,0.,4),
        ('leg001','nominal',.001,0.,4),('leg003','nominal',.003,0.,4),('leg005','nominal',.005,0.,4),
        ('wheel005','nominal',0.,.005,4),('combined','nominal',.001,.005,4),('push','push',0.,0.,12),
        ('sensor_noise','sensor_noise',0.,0.,4),('sensor_delay','sensor_delay',0.,0.,4))
    selected=set(args.conditions.split(',')) if args.conditions else {x[0] for x in cases[:-2]}
    if not selected.issubset({x[0] for x in cases}):raise ValueError('Unknown screening condition')
    for name,kind,leg,wheel,seconds in cases:
        if name in selected:evaluate(name,kind,leg,wheel,seconds,subset)
    eligible=[name for name in ('combined','leg005','leg003','leg001','wheel005')
        if name in results and results[name]['strict_rise']==20 and results[name]['end_balance']==20 and results[name]['mean_leg_rms']<=.02]
    if eligible:
        winner=eligible[0];c=results[winner];evaluate('full_candidate','nominal',c['leg_std'],c['wheel_std'],20,list(range(40)))
    report=dict(completed=True,optimizer_updates=0,checkpoint_sha256=digest(args.checkpoint),prior_sha256=digest(args.prior),
        cached_initial_sha256=digest(args.source/'nominal_mean/initial-state.pt'),source_sha256=digest(__file__),
        eligible=eligible,selected=eligible[0] if eligible else None,total_wall_seconds=time.monotonic()-started,
        groups={k:{x:v[x] for x in ('strict_rise','end_balance','qualified','mean_leg_rms','mean_speed','seconds','num_envs')} for k,v in results.items()},
        scope='Screening5 fixed poses x4 delays; one passing candidate tested on original40. No training or default configuration changes.')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--conditions',help='Comma-separated incremental conditions; omit for the original screening matrix')
    run(p.parse_args())
