#!/usr/bin/env python3
"""Four matched40 frozen-BC replays; training conditions and action sampling toggled separately."""
import argparse,copy,gc,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import eval_bc
from ct_mjlab.bc import load_actor
from ct_mjlab.smp.ppo import configure_smp
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT
from check_lqr_student_closed_loop import digest


GROUPS=(('nominal_mean',False,False),('nominal_sample',False,True),
        ('random_mean',True,False),('random_sample',True,True))


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    actor,saved=load_actor(args.checkpoint,args.device);actor.requires_grad_(False)
    weights={k:v.clone() for k,v in actor.state_dict().items()}
    std=weights['distribution.std_param'];torch.testing.assert_close(std,torch.full_like(std,.05),atol=0,rtol=0)
    get_cfgs=eval_bc.task_registry.get_cfgs;make_env=eval_bc.task_registry.make_env
    dataset_class=eval_bc.ActionDataset;cache={};initials={};reports={}
    def dataset(*pos,**kw):
        key=repr((pos,kw))
        if key not in cache:cache[key]=dataset_class(*pos,**kw)
        return cache[key]
    for name,randomized,sampled in GROUPS:
        print('Starting',name,flush=True);directory=args.output/name;holder={};means=[];noises=[];rewards=[];states=[]
        generator=torch.Generator(device=args.device).manual_seed(20261026)
        class FrozenActor:
            def state_dict(self):return actor.state_dict()
            def mlp(self,obs):
                if not means:holder['initial_observation']=obs.cpu().clone()
                mean=actor.mlp(obs)
                noise=torch.randn(mean.shape,device=mean.device,generator=generator)*std if sampled else torch.zeros_like(mean)
                means.append(mean.cpu().numpy().copy());noises.append(noise.cpu().numpy().copy())
                return mean+noise
        def configs(task,*,play=False):
            cfg,train=get_cfgs(task,play=not randomized)
            holder['task_rewards']=copy.deepcopy(cfg.rewards)
            return cfg,train
        def environment(*pos,**kw):
            cfg=kw['env_cfg'];cfg.rewards=holder['task_rewards']
            configure_smp(cfg,kw['train_cfg'],args.prior,play=not randomized,policy_contract=HANDOVER_OBSERVATION_CONTRACT)
            env,result=make_env(*pos,**kw);raw=env.unwrapped;holder['raw']=raw
            names=raw.reward_manager.active_terms;holder['names']=names;step=env.step
            def measured(actions):
                action=raw.action_manager.get_term('pd')
                if not states:
                    initial=dict(qpos=raw.sim.data.qpos.cpu().clone(),qvel=raw.sim.data.qvel.cpu().clone(),
                        observation=holder['initial_observation'],rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),kp=action.kp_scale.cpu().clone(),kd=action.kd_scale.cpu().clone(),
                        strength=action.motor_strength.cpu().clone(),nominal=action._nominal_pos.cpu().clone(),
                        spring_damping=action.spring_damping.cpu().clone(),spring_bias=action._spring_force_bias.cpu().clone(),
                        leg_delay=action.action_delay_steps.cpu().clone(),wheel_delay=action.wheel_action_delay_steps.cpu().clone())
                    domain=getattr(raw,'_huananhu_randomization',None)
                    if domain is not None:
                        initial.update(mass_scales=domain.mass_scales.cpu().clone(),com_offset=domain.com_offset.cpu().clone(),
                            materials={k:v.cpu().clone() for k,v in domain.materials.items()})
                    initials[name]=initial;torch.save(initial,directory/'initial-state.pt')
                result=step(actions)
                values=raw.reward_manager._step_reward*raw.step_dt
                torch.testing.assert_close(values.sum(-1),result[1],atol=2e-6,rtol=2e-5)
                rewards.append(values.cpu().numpy().copy())
                command=raw.command_manager.get_term('standup');probe=raw._source_probe
                states.append(dict(strict=probe['strict_ever'].cpu().numpy().copy(),
                    balance=probe['hold'].cpu().numpy().copy(),anchor_valid=command.zero_hold_anchor_valid.cpu().numpy().copy(),
                    anchor_error=(raw.scene['robot'].data.root_link_pos_w[:,:2]-command.zero_hold_anchor_xy).norm(dim=-1).cpu().numpy().copy(),
                    recover=command.recover.cpu().numpy().copy(),stable_time=command.stable_time.cpu().numpy().copy()))
                return result
            env.step=measured
            return env,result
        case=SimpleNamespace(dataset=args.dataset,corrections=args.corrections,pose_dataset=args.pose_dataset,
            checkpoint=args.checkpoint,device=args.device,output=directory,split='validation',
            collect_corrections=False,full_precision=False)
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        with patch.object(eval_bc,'load_actor',return_value=(FrozenActor(),saved)), \
                patch.object(eval_bc,'ActionDataset',side_effect=dataset), \
                patch.object(eval_bc.task_registry,'get_cfgs',side_effect=configs), \
                patch.object(eval_bc.task_registry,'make_env',side_effect=environment), \
                patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Training forbidden')):
            eval_bc.run(case)
        for k,v in actor.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
        report=json.loads((directory/'result.json').read_text())
        with np.load(directory/'trace.npz') as d:
            for tick in np.linspace(0,len(means)-1,16).round().astype(int):
                expected=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).to(args.device)).cpu().numpy()
                np.testing.assert_allclose(expected,means[tick],atol=2e-6,rtol=2e-6)
            np.testing.assert_array_equal(d['action'],np.asarray(means)+np.asarray(noises))
            if not sampled:assert not np.asarray(noises).any()
            for row in report['rows']:
                i=row['env'];n=int(d['alive'][:,i].sum());first=None;lost=None;run_bad=0
                for tick,state in enumerate(states[:n]):
                    if first is None and state['strict'][i]:first=(tick+1)*.01
                    if first is not None:
                        run_bad=run_bad+1 if state['balance'][i]==0 else 0
                        if run_bad>=10 and lost is None:lost=(tick-8)*.01
                tilt=np.maximum(np.abs(np.arcsin(d['features'][:n,i,1].clip(-1,1))),
                    np.abs(np.arctan2(-d['features'][:n,i,2],-d['features'][:n,i,3])))
                row.update(first_strict_support_s=first,first_sustained_balance_loss_after_strict_s=lost,
                    late5s_mean_tilt_deg=float(np.rad2deg(tilt[1500:]).mean()) if n>1500 else None)
        np.savez_compressed(directory/'probe.npz',action_mean=np.asarray(means),action_noise=np.asarray(noises),
            reward=np.asarray(rewards),reward_names=np.asarray(holder['names']),**{k:np.asarray([s[k] for s in states]) for k in states[0]})
        report.update(randomized=randomized,action_sampling=sampled,action_std=std.cpu().tolist(),
            action_noise_seed=20261026,actor_unchanged=True,optimizer_updates=0,started_at=started,
            ended_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),probe_sha256=digest(directory/'probe.npz'),
            overrides='Original40 poses, four balanced2/4ms delay pairs, auto_reset=False; all other play/train config retained',
            scope='Frozen original BC; Gaussian samples use independent RNG with the same distribution as PPO, not a bitwise training rollout replay')
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n');reports[name]=report
        print('Finished',name,{k:report[k] for k in ('strict_rise','final_balance','qualified')},flush=True)
        holder.clear();gc.collect();torch.cuda.empty_cache()
    def equal(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        else:torch.testing.assert_close(a,b,atol=0,rtol=0)
    equal(initials['nominal_mean'],initials['nominal_sample']);equal(initials['random_mean'],initials['random_sample'])
    for name in initials:
        equal(initials['nominal_mean']['qpos'],initials[name]['qpos']);equal(initials['nominal_mean']['qvel'],initials[name]['qvel'])
    with np.load(args.output/'nominal_sample/probe.npz') as a,np.load(args.output/'random_sample/probe.npz') as b:
        n=min(len(a['action_noise']),len(b['action_noise']));np.testing.assert_array_equal(a['action_noise'][:n],b['action_noise'][:n])
    summary=dict(completed=True,optimizer_updates=0,checkpoint_sha256=digest(args.checkpoint),prior_sha256=digest(args.prior),
        source_sha256=digest(__file__),matched_initial_qpos_qvel=True,paired_initial_parameters_observations_rng=True,
        independent_identical_sampling_noise=True,groups={name:{k:r[k] for k in ('strict_rise','final_balance','qualified','randomized','action_sampling','trace_sha256','probe_sha256')} for name,r in reports.items()},
        scope='One matched seed/cohort; randomization is a bundle of current training physics, sensor errors and pushes, not attribution to an individual parameter.')
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','pose-dataset','checkpoint','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
