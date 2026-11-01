"""Frozen candidate checks under actual actuator delays and training randomization."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.supermimic.__main__ import make_env,collect
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.networks import act,RunningNorm
from ct_mjlab.supermimic.state import capture_state,capture_parameters,NOISE
from ct_mjlab.supermimic.buffer import ReplayBuffer
from scripts.agent.supermimic.diagnose_v13_wheel_response import Snapshot
from scripts.agent.supermimic.diagnose_v13_v5_response import outputs,sha
from scripts.agent.supermimic.assess_chuanliantui_supermimic import make_cases,actor_digest,verify_training_sources
from scripts.agent.supermimic.record_supermimic_physical_prediction import ObservedWalkRewardState
from ct_mjlab.supermimic.analytic_reward import capture


class ConditionSnapshot(Snapshot):
    def __init__(self,raw):
        super().__init__(raw)
        # Scheduling the same exogenous push in every branch requires its own state.
        for obj in (raw.command_manager.get_term('standup'),getattr(raw,'_standup_randomization',None)):
            if obj is None:continue
            for key,value in vars(obj).items():
                if isinstance(value,torch.Tensor):self.tensors.append((value,value.clone()))
                elif type(value) in (int,float,bool):self.scalars.append((obj,key,value))
        self.tensors.append((raw.sim.data.xfrc_applied,raw.sim.data.xfrc_applied.clone()))


def branch(raw,action,motor_v):
    pd=raw.action_manager.get_term('pd');pd.process_actions(action);path=[];parameters=None
    for _ in range(5):
        pd.apply_actions();raw.scene.write_data_to_sim()
        if parameters is None:parameters=capture_parameters(raw).clone()
        raw.sim.step();raw.scene['robot'].update(raw.physics_dt)
        q=raw.sim.data.qpos.clone();q[:,:3]-=raw.scene.env_origins
        path.append(torch.cat((q,raw.sim.data.qvel.clone(),raw.sim.data.qfrc_actuator[:,motor_v].clone(),torch.zeros(raw.num_envs,9,device=raw.device)),-1))
    return torch.stack(path,1),parameters


def anchors(buffer,alive_ring,generator):
    length=24
    if buffer.size<length:return None
    n=buffer.num_envs
    env=torch.randint(n,(96,),device=buffer.device,generator=generator)
    start=torch.randint(buffer.size-length+1,(96,),device=buffer.device,generator=generator)
    row=(buffer.ptr-buffer.size+start[:,None]+torch.arange(length,device=buffer.device))%buffer.capacity
    w={k:v[row,env[:,None]].clone() for k,v in buffer.data.items()}
    first_alive=alive_ring[row,env[:,None]]
    prefix=torch.cumprod(torch.cat((torch.ones(96,1,device=buffer.device),(~w['done'][:,:-1]).float()),1),1)
    w['mask']=prefix*first_alive.float()
    return {k:v.cpu() for k,v in w.items()}


def summary(samples,worlds,initial,alive,ends,conditions,seed):
    rows=[];horizons=[]
    for label,world in worlds.items():
        for lag in range(6):
            true_all=[];pred_all=[];eligible_all=[];live_total=0;possible_total=0
            for s in samples:
                group=s['parameters'][:,25].long()==lag
                possible_total+=int(group.sum())*2
                for wheel in (2,5):
                    plus,minus=f'{wheel}_+0.10',f'{wheel}_-0.10'
                    real=s['truth'][plus][:,0]-s['truth'][minus][:,0]
                    predicted=s['predictions'][label][plus][:,0]-s['predictions'][label][minus][:,0]
                    valid=s['live']&group&(s['action'][:,wheel].abs()+.1<3)
                    floor=torch.maximum(5*s['noise'][:,0],torch.tensor(1e-5))
                    true_all.append(real[valid]);pred_all.append(predicted[valid]);eligible_all.append(real[valid].abs()>floor[valid])
                    live_total+=int(valid.sum())
            real=torch.cat(true_all);predicted=torch.cat(pred_all);eligible=torch.cat(eligible_all)
            observable=predicted.abs()>1e-5
            rows.append(dict(model=label,delay_ms=lag*2,possible_cases=possible_total,valid_cases=live_total,
                eligible=int(eligible.sum()),correct=int((eligible&observable&(real*predicted>0)).sum()),
                opposite=int((eligible&observable&(real*predicted<0)).sum()),weak=int((eligible&~observable).sum()),
                true_near_zero=int((~eligible).sum()),spurious_nonzero=int((~eligible&observable).sum()),
                response_slope_mse=float(((predicted-real)/.2).square().mean()) if len(real) else None))
        learner=Learner('cpu');learner.world=world.cpu()
        for length in (1,8,24):
            total=0.;positions=0;windows=0
            for s in samples:
                if s['anchor_windows'] is None:continue
                w={k:v[:,:length] for k,v in s['anchor_windows'].items()}
                loss,m=learner.world_loss(w);count=m['valid_world_positions']
                total+=float(loss)*count;positions+=count;windows+=len(w['state'])
            horizons.append(dict(model=label,horizon_ms=length*10,normalized_mse=total/positions if positions else None,
                valid_positions=positions,requested_windows=windows))
    scales={label:w.delta_norm.std().cpu() for label,w in worlds.items()}
    assert torch.equal(scales['parent'],scales['candidate'])
    groups={}
    for lag in range(6):
        mask=initial['parameters'][:,25].long()==lag
        groups[str(lag*2)]=dict(initial_envs=int(mask.sum()),failed=int((mask&~alive.cpu()).sum()))
    return dict(conditions=conditions,seed=seed,rows=rows,horizons=horizons,delay_groups=groups,
        initial_envs=len(alive),failed_first_episodes=int((~alive).sum()),first_episode_ends_s=ends.cpu().tolist(),
        samples=[dict(time_s=s['time_s'],live=int(s['live'].sum()),total=len(s['live']),
            baseline_repeat_or_native_noise_max=s['noise'][s['live']].max(0).values.tolist() if s['live'].any() else None) for s in samples],
        fixed_delta_scale=scales['parent'].tolist(),optimizer_updates=0,normalization_updates=0,
        scope='Same live first-episode states/actions for both frozen models. Failed/reset samples excluded and reported; not a control recovery test.')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','candidate','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--conditions',choices=('delay_only','training'),required=True)
    p.add_argument('--seed',type=int,default=71);p.add_argument('--num-envs',type=int,default=144)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    assert args.num_envs%36==0
    verify_training_sources(args.checkpoint,args.output,'supermimic')
    candidate=torch.load(args.candidate,map_location='cpu',weights_only=False)
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert candidate['parent_sha256']==sha(args.checkpoint)
    env,cfg=make_env(SimpleNamespace(num_envs=args.num_envs,seed=args.seed,device='cuda:0'),play=args.conditions=='delay_only')
    raw=env.unwrapped;raw.cfg.auto_reset=False
    worlds={};samples=[]
    try:
        command=raw.command_manager.get_term('standup');learner=Learner('cuda:0')
        learner.load(args.checkpoint,command,inference=True);learner.actor.eval().requires_grad_(False)
        for label,weights in [('parent',parent['world']),('candidate',candidate['world'])]:
            w=WorldModel().cuda().eval().requires_grad_(False);w.load_state_dict(weights);worlds[label]=w
        frozen={k:actor_digest(v) for k,v in worlds.items()};actor_before=actor_digest(learner.actor)
        raw._split_walk_rewards=ObservedWalkRewardState(raw);command.speed_curriculum_frozen=True;raw.reset(seed=args.seed)
        pd=raw.action_manager.get_term('pd')
        if args.conditions=='delay_only':
            delays=torch.arange(args.num_envs,device=raw.device)%6
            pd.action_delay_steps.copy_(delays);pd.wheel_action_delay_steps.copy_(delays)
        assert torch.equal(pd.action_delay_steps,pd.wheel_action_delay_steps)
        initial=dict(parameters=capture_parameters(raw).cpu(),qpos=raw.sim.data.qpos.cpu(),qvel=raw.sim.data.qvel.cpu(),
            contact_training_recipe='actual_make_env_training_mixture' if args.conditions=='training' else 'nominal',
            action_randomization=bool(cfg.actions['pd'].motor_randomization),observations=repr(cfg.observations['actor'].terms['proprioception'].params))
        targets=torch.empty(args.num_envs,device=raw.device);acc=torch.empty_like(targets)
        for case in make_cases(args.num_envs):targets[case['env_ids']]=case['speed'];acc[case['env_ids']]=case['acceleration']
        def commands(t):
            command.tracking_episode.fill_(True);command.zero_hold.zero_();command.tracking_yaw_rate.zero_()
            command.tracking_velocity.copy_(targets if 2<=t<16 else torch.zeros_like(targets))
            command.tracking_interval_steps.fill_(1000000);command.tracking_acceleration.copy_(acc)
            command.episode_height.fill_(.26);command._apply_height(None)
        commands(0);raw.observation_manager.reset();raw.obs_buf=raw.observation_manager.compute(update_history=True);obs=raw.get_observations()
        buffer=ReplayBuffer(32,args.num_envs,raw.device);life_ring=torch.zeros(32,args.num_envs,device=raw.device,dtype=torch.bool)
        alive=torch.ones(args.num_envs,device=raw.device,dtype=torch.bool);ends=torch.full((args.num_envs,),20.01,device=raw.device)
        generator=torch.Generator(device=raw.device).manual_seed(args.seed+1000)
        for tick in range(2001):
            commands(tick*.01);sample=None
            if tick in (40,80,300,1000,2000):
                state=capture_state(raw);action=act(learner.actor,obs['history']);context=capture(raw,raw._split_walk_rewards)['context']
                previous=raw.action_manager.action.clone();previous2=raw.action_manager.prev_action.clone()
                snapshot=ConditionSnapshot(raw);truth={};pred={k:{} for k in worlds};actions={};paths={};parameters=None
                for key,wheel,eps in [('base',2,0.),('repeat',2,0.)]+[(f'{wheel}_{sign}0.10',wheel,float(sign+'0.10')) for wheel in (2,5) for sign in ('-','+')]:
                    snapshot.restore();assert torch.equal(capture_state(raw),state)
                    a=action.clone();a[:,wheel]=(a[:,wheel]+eps).clamp(-3,3)
                    path,params=branch(raw,a,worlds['parent'].adapter.motor_v-25)
                    if parameters is None:parameters=params
                    else:assert torch.equal(params,parameters),'Exogenous force changed between branches'
                    paths[key]=path.cpu();truth[key]=outputs(worlds['parent'],path).cpu();actions[key]=a.cpu()
                    for label,w in worlds.items():
                        result=w.step(state,parameters,a,context[:,:3],context,previous,previous2,state[:,NOISE],return_path=True)
                        pred[label][key]=outputs(w,result[-1]).cpu()
                snapshot.restore();del snapshot
                sample=dict(time_s=tick*.01,state=state.cpu(),parameters=parameters.cpu(),action=action.cpu(),context=context.cpu(),
                    previous=previous.cpu(),previous_previous=previous2.cpu(),live=alive.cpu(),actions=actions,paths=paths,truth=truth,predictions=pred)
            life_ring[buffer.ptr].copy_(alive)
            obs,_,done=collect(env,learner,buffer,obs,exploration=0.)
            done=done.bool()
            if sample is not None:
                native=buffer.data['path'][(buffer.ptr-1)%buffer.capacity].reshape(args.num_envs,5,64)
                native_out=outputs(worlds['parent'],native).cpu()
                sample['noise']=torch.maximum((sample['truth']['base']-sample['truth']['repeat']).abs(),(sample['truth']['base']-native_out).abs())
                sample['anchor_windows']=anchors(buffer,life_ring,generator)
                samples.append(sample);print(json.dumps(dict(time_s=sample['time_s'],live=int(sample['live'].sum()),total=args.num_envs)),flush=True)
            newly=alive&done;ends[newly]=(tick+1)*.01;alive&=~done
            if done.any():
                raw.reset(env_ids=done.nonzero().flatten());obs=raw.get_observations()
        assert actor_before==actor_digest(learner.actor) and frozen=={k:actor_digest(v) for k,v in worlds.items()}
    finally:env.close()
    torch.save(dict(schema='v13-v5-condition-response-v1',initial=initial,samples=samples,
        conditions=args.conditions,seed=args.seed,actor_checkpoint_sha256=sha(args.checkpoint),candidate_sha256=sha(args.candidate)),args.output/'recording.pt')
    for w in worlds.values():w.cpu()
    report=summary(samples,worlds,initial,alive,ends,args.conditions,args.seed)
    report.update(checkpoint_sha256=sha(args.checkpoint),candidate_sha256=sha(args.candidate),recording_sha256=sha(args.output/'recording.pt'),script_sha256=sha(Path(__file__)),actor_frozen=True,worlds_frozen=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True,failed=report['failed_first_episodes'],horizons=report['horizons']),ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Frozen check only')),patch.object(RunningNorm,'update',side_effect=AssertionError('Frozen normalization')):
        main()
