#!/usr/bin/env python3
"""Matched initialization and actual zero-SMP reward checks; never optimize."""
import argparse,copy,json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import configure_smp,checkpoint_smp_weight
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False);initial=[];rows=[]
    for weight in (0.,1.):
        cfg,train=task_registry.get_cfgs('chuanliantui');cfg.seed=train.seed=42
        configure_smp(cfg,train,args.prior,policy_contract=HANDOVER_OBSERVATION_CONTRACT,reward_weight=weight)
        if weight==0.:
            baseline=copy.deepcopy(asdict(cfg));agent=copy.deepcopy(asdict(train))
        else:
            target=asdict(cfg);baseline['rewards']['smp_motion']['params'].pop('reward_scale')
            # 默认地形工厂每次创建新lambda对象，比较其代码／闭包而非内存地址。
            left=baseline['scene']['terrain']['spec_fn'];right=target['scene']['terrain']['spec_fn']
            assert left.__code__==right.__code__ and left.__defaults__==right.__defaults__
            assert left.__closure__==right.__closure__==None
            baseline['scene']['terrain']['spec_fn']=right
            assert baseline==target
            other=asdict(train);agent['experiment_name']=other['experiment_name'];assert agent==other
        env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=20,device='cuda:0')
        try:
            runner,_=task_registry.make_alg_runner(env,'chuanliantui',train_cfg=train,device='cuda:0')
            runner.initialize_from_bc(args.bc);raw=env.unwrapped;obs=env.get_observations()
            state=dict(qpos=raw.sim.data.qpos.clone().cpu(),qvel=raw.sim.data.qvel.clone().cpu(),
                actor={k:v.cpu().clone() for k,v in runner.alg._raw_actor.state_dict().items()},
                critic={k:v.cpu().clone() for k,v in runner.alg._raw_critic.state_dict().items()},
                observations={k:v.cpu().clone() for k,v in obs.items()},
                leg_delay=raw.action_manager.get_term('pd').action_delay_steps.cpu().clone(),
                wheel_delay=raw.action_manager.get_term('pd').wheel_action_delay_steps.cpu().clone(),
                rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state().cpu().clone())
            initial.append(state)
            path=args.output/f'initial-{int(weight)}.pt';runner.save(str(path));assert checkpoint_smp_weight(path)==weight
            runner.load(str(path));raw_score=[]
            names=raw.reward_manager.active_terms;index=names.index('smp_motion')
            with patch.object(runner.alg.optimizer,'step',side_effect=AssertionError('Training forbidden')):
                for _ in range(24):
                    obs,reward,done,extras=env.step(runner.alg.act(obs));runner.alg.process_env_step(obs,reward,done,extras)
                    values=raw.reward_manager._step_reward*raw.step_dt
                    torch.testing.assert_close(values.sum(-1),reward,atol=2e-6,rtol=2e-5)
                    torch.testing.assert_close(values[:,index],raw._smp_reward.last_reward*raw.step_dt*weight,atol=1e-7,rtol=1e-6)
                    if weight==0.:assert not values[:,index].any()
                    raw_score.append(float(raw._smp_reward.last_reward.mean()))
                runner.alg.compute_returns(obs)
            assert not runner.alg.optimizer.state
            saved=torch.load(path,map_location='cpu',weights_only=False)
            saved['infos']['smp']['weight']=1.-weight;bad=args.output/f'mismatch-{int(weight)}.pt';torch.save(saved,bad)
            try:runner.load(str(bad))
            except ValueError:pass
            else:raise AssertionError('Cross-reward checkpoint accepted')
            rows.append(dict(weight=weight,mean_raw_smp_score=sum(raw_score)/len(raw_score),optimizer_updates=0))
        finally:env.close()
    def equal(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        else:torch.testing.assert_close(a,b,atol=0,rtol=0)
    equal(initial[0],initial[1]);torch.save(initial[0],args.output/'matched-initial-state.pt')
    report=dict(passed=True,optimizer_updates=0,num_envs=20,rollout_steps_each=24,
        identical_actor_critic_initial_physical_state_observations_delays_and_rng=True,
        configs_only_differ_in_smp_reward_and_log_directory=True,actual_zero_smp_contribution=True,
        state_refresh_and_prior_scoring_kept_in_both=True,cross_reward_resume_rejected=True,rows=rows)
    (args.output/'check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bc','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
