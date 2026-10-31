#!/usr/bin/env python3
"""Exercise actual wheel39 PPO entry dispatch; intercept learn before any rollout/update."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import SmpOnPolicyRunner,WHEEL_PROFILE
from ct_mjlab.smp.finetune import PROFILE as FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT


def check(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    from ct_mjlab.scripts import train,play,check as finite
    from ct_mjlab.scripts.eval import evaluate_seed
    profile=(PROFILE_V2 if args.bc_finetune=='v2' else FINETUNE_PROFILE) if getattr(args,'bc_finetune',False) else WHEEL_PROFILE
    if args.bc_finetune == 'isaac-standup': profile = PROFILE_ISAAC
    if args.bc_finetune == 'isaac-standup-v2': profile = PROFILE_ISAAC_V2
    if args.bc_finetune == 'isaac-standup-v3': profile = PROFILE_ISAAC_V3
    original=task_registry.make_alg_runner;calls=[]
    def make_runner(*pos,**kw):
        if kw.get('log_dir')=='default':kw['log_dir']=args.output/('dispatch-'+('resume' if kw.get('resume') else 'bc'))
        runner,cfg=original(*pos,**kw)
        assert runner.observation_contract==HANDOVER_OBSERVATION_CONTRACT
        assert runner.experiment_profile==profile and not runner.alg.optimizer.state
        runner._entry_expected_steps=(torch.load(kw['resume'],map_location='cpu',weights_only=False)
            ['infos']['env_state']['common_step_counter'] if kw.get('resume') else 0)
        calls.append(runner)
        return runner,cfg
    class BeforeLearn(Exception):pass
    def no_learn(runner,**kw):
        assert kw['num_learning_iterations']==1 and runner.bc_init
        assert runner.env.unwrapped.common_step_counter==runner._entry_expected_steps and not runner.alg.optimizer.state
        raise BeforeLearn()
    base=dict(task='chuanliantui',seed=42,num_envs=2,device='cuda:0',smp_prior=args.prior,
              bc_finetune=getattr(args,'bc_finetune',False))
    # 真实创建、初始化／恢复并写参数；learn入口即停止，不允许调用优化器。
    with patch.object(task_registry,'make_alg_runner',side_effect=make_runner), \
            patch.object(SmpOnPolicyRunner,'learn',no_learn), \
            patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Training forbidden')):
        for bc,resume in ((args.bc,None),(None,args.checkpoint)):
            ppo_init = args.ppo_init if resume is None else None
            try:train.train(SimpleNamespace(**base,bc_init=None if ppo_init else bc,ppo_init=ppo_init,
                                           resume=resume,iterations=1,run_name='interface-check'))
            except BeforeLearn:pass
            else:raise AssertionError('train dispatch did not reach intercepted learn')
        finite.check(SimpleNamespace(**base,checkpoint=args.checkpoint,steps=4))
        class Viewer:
            def __init__(self,env,policy):self.env,self.policy=env,policy
            def run(self):
                with torch.inference_mode():
                    obs=self.env.get_observations()
                    for _ in range(4):
                        obs,reward,_,_=self.env.step(self.policy(obs))
                        assert obs['actor'].shape==(2,39) and torch.isfinite(reward).all()
        with patch('mjlab.viewer.ViserPlayViewer',Viewer):
            play.play(SimpleNamespace(**base,checkpoint=args.checkpoint,viewer='viser',viser_port=None,knockdown_every=None))
        result,rows,_=evaluate_seed(SimpleNamespace(num_envs=2,start_mode='force',smp_prior=args.prior,
            device='cuda:0',checkpoint=args.checkpoint,capture=False,steps=4,knockdown_at=None,knockdown_speed=(18.,28.)),42)
        assert len(rows)==2 and result['extras']['ground_start_envs']==2
    report=dict(passed=True,optimizer_updates=0,ppo_profile=profile,observation_contract=HANDOVER_OBSERVATION_CONTRACT,
        real_train_bc_initialization_and_resume_dispatch=not bool(args.ppo_init),
        real_train_ppo_transfer_and_resume_dispatch=bool(args.ppo_init),learn_intercepted_before_sampling=True,
        check_entry_steps=4,play_entry_steps=4,eval_entry_steps=4,num_envs=2,
        viewer_boundary_mocked=True,visual_viewer_not_tested=True,default_logs_not_created=True)
    (args.output/'check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bc','checkpoint','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bc-finetune',nargs='?',const='v1',default=False,choices=('v1','v2','isaac-standup','isaac-standup-v2','isaac-standup-v3'))
    p.add_argument('--ppo-init',type=Path)
    check(p.parse_args())
