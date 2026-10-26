#!/usr/bin/env python3
"""Original40 nominal replay through the real PPO actor initialized from wheel39 BC; no updates."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import eval_bc
from ct_mjlab.bc import load_actor
from ct_mjlab.smp.ppo import configure_smp,WHEEL_PROFILE,checkpoint_smp_weight,checkpoint_bc_finetune
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    actor,saved=load_actor(args.checkpoint,args.device)
    ppo_path=getattr(args,'ppo_checkpoint',None)
    weight=checkpoint_smp_weight(ppo_path)
    original=eval_bc.task_registry.make_env;holder={};contributions=[]
    class Actor:
        @property
        def mlp(self):return holder['runner'].alg._raw_actor.mlp
        def state_dict(self):return holder['runner'].alg._raw_actor.state_dict()
    def make_env(*pos,**kw):
        # BC验收原本不算奖励；此处恢复任务项，再加入SMP，核对完整PPO奖励。
        kw['env_cfg'].rewards=eval_bc.task_registry.get_cfgs('chuanliantui',play=True)[0].rewards
        configure_smp(kw['env_cfg'],kw['train_cfg'],args.prior,play=True,policy_contract=HANDOVER_OBSERVATION_CONTRACT,
                      reward_weight=weight,bc_finetune=checkpoint_bc_finetune(ppo_path))
        env,cfg=original(*pos,**kw)
        runner,_=eval_bc.task_registry.make_alg_runner(env,'chuanliantui',train_cfg=kw['train_cfg'],device=args.device)
        if ppo_path is None:
            runner.initialize_from_bc(args.checkpoint)
            for key,value in actor.state_dict().items():torch.testing.assert_close(runner.alg._raw_actor.state_dict()[key],value,atol=0,rtol=0)
        else:
            runner.load(str(ppo_path),load_cfg={'actor':True},map_location=args.device)
            assert runner.bc_init['sha256']==digest(args.checkpoint)
        holder['runner']=runner
        assert not runner.alg.optimizer.state
        step=env.step
        holder['reward_names']=env.unwrapped.reward_manager.active_terms
        assert 'smp_motion' in holder['reward_names'] and len(holder['reward_names'])>1
        def measured_step(actions):
            result=step(actions)
            values=env.unwrapped.reward_manager._step_reward*env.unwrapped.step_dt
            torch.testing.assert_close(values.sum(-1),result[1],atol=2e-6,rtol=2e-5)
            contributions.append(values.cpu().numpy().copy())
            return result
        env.step=measured_step
        return env,cfg
    args.split='validation';args.collect_corrections=False;args.full_precision=False
    with patch.object(eval_bc,'load_actor',return_value=(Actor(),saved)), \
            patch.object(eval_bc.task_registry,'make_env',side_effect=make_env), \
            patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Training forbidden')):
        eval_bc.run(args)
    assert not holder['runner'].alg.optimizer.state
    path=args.output/'result.json';report=json.loads(path.read_text())
    report.update(diagnostic_only=True,ppo_profile=holder['runner'].experiment_profile,
        policy_source='Actual PPO actor initialized from BC; zero PPO optimizer updates',
        ppo_actor_weights_identical_to_bc=True,prior_sha256=digest(args.prior),
        entry_script_sha256=digest(__file__),
        scope='40 original heldout initial states and four2/4ms delays, nominal20s actual PPO actor/PD; no policy learning, randomization, hardware or parking improvement claim.')
    if ppo_path is not None:
        checkpoint=torch.load(ppo_path,map_location='cpu',weights_only=False)
        report.update(diagnostic_only=False,ppo_iterations=checkpoint['iter']+1,smp_reward_weight=weight,
            checkpoint_sha256=digest(ppo_path),bc_checkpoint_sha256=digest(args.checkpoint),
            ppo_actor_weights_identical_to_bc=False,policy_source='Trained PPO actor, evaluation optimizer updates0',
            scope='40 original heldout poses, nominal20s trained PPO actor, 2/4ms delays; no evaluation updates or hardware claim.')
    path.write_text(json.dumps(report,indent=2)+'\n')
    values=np.asarray(contributions);names=holder['reward_names'];index=names.index('smp_motion')
    np.savez_compressed(args.output/'rewards.npz',contributions=values,names=np.asarray(names))
    with np.load(args.output/'trace.npz') as trace:alive=trace['alive']
    shares={}
    for name,segment in (('startup_hold',slice(0,20)),('rise',slice(20,200)),('late5s',slice(1500,2000))):
        # 只统计原始回合，失败后的复位轨迹不能计入奖励占比。
        part=values[segment][alive[segment]];smp=part[:,index];task=np.delete(part,index,axis=-1)
        positive=task.clip(min=0).sum();penalty=-task.clip(max=0).sum()
        positive_total=float(smp.sum()+positive);absolute_total=positive_total+float(penalty)
        shares[name]=dict(valid_first_episode_steps=len(part),
            smp_mean_step=float(smp.mean()) if len(part) else None,
            task_mean_step=float(task.sum(-1).mean()) if len(part) else None,
            smp_share_of_positive=float(smp.sum()/positive_total) if positive_total else None,
            smp_share_of_absolute=float(smp.sum()/absolute_total) if absolute_total else None,
            contributions={key:float(part[:,i].sum()) for i,key in enumerate(names)})
    (args.output/'reward-shares.json').write_text(json.dumps(dict(optimizer_updates=0,
        scope='Frozen PPO mean actions, nominal40; evaluation only, reward weights unchanged',
        rewards_sha256=digest(args.output/'rewards.npz'),segments=shares),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','checkpoint','prior','pose-dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',default=[]);p.add_argument('--device',default='cuda:0')
    p.add_argument('--ppo-checkpoint',type=Path,help='Evaluate trained PPO; --checkpoint remains its BC source for pose/schema validation')
    run(p.parse_args())
