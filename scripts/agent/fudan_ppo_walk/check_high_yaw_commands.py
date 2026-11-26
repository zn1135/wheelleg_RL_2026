"""Check actual command resampling on CPU/CUDA; no physics, inference or learning."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import commands as sampling
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);checks={};actual={}
    def verify(name,value):checks[name]=bool(value);assert checks[name],name
    def environment(count,device,cfg=None):
        cfg=FudanWalkCfg() if cfg is None else cfg
        env=SimpleNamespace(cfg=cfg,num_envs=count,device=device,speed_limit=2.,commands=torch.zeros(count,3,device=device))
        sampling.initialize(env);return env
    with ExitStack() as stack:
        for method in ('torch.optim.Adam.step','torch.nn.Linear.forward','mjlab.envs.ManagerBasedRlEnv.step'):
            stack.enter_context(patch(method,side_effect=AssertionError('No physics, inference or learning')))
        cfg=sampling.command_config()
        verify('range_and_rule_recorded',cfg['yaw_range']==[-10.,10.] and cfg['zero_vx_above_yaw']==2.
               and cfg['version']=='fudan-high-yaw-parking-commands-v2')
        for device in ('cpu','cuda:0'):
            env=environment(10000,device);ids=torch.arange(env.num_envs,device=device);torch.manual_seed(721)
            sampling.reset_modes(env,ids);FudanEnv.resample(env,ids)
            regular=env.command_mode==0;high=regular & (env.commands[:,1].abs()>2.)
            low=regular & (env.commands[:,1].abs()<=2.)
            verify(device+'_high_yaw_has_no_vx',env.commands[high,0].eq(0).all() and high.any())
            verify(device+'_low_yaw_keeps_driving_turns',low.any() and env.commands[low,0].ne(0).any())
            verify(device+'_yaw_spans_both_sides',env.commands[:,1].min() < -9.5 and env.commands[:,1].max()>9.5)
            verify(device+'_yaw_bounded',env.commands[:,1].abs().le(10.).all())
            verify(device+'_standstill_still_zero',env.commands[env.command_mode==1,:2].eq(0).all())
            stop=env.command_mode==2
            verify(device+'_stopping_keeps_linear_only',env.commands[stop,1].eq(0).all()
                and env.commands[stop,0].abs().ge(.25).all() and env.commands[stop,0].abs().le(1.).all())
            actual[device]=dict(total=env.num_envs,high_yaw_fraction=float(high.float().mean()),
                yaw_minmax=[float(env.commands[:,1].min()),float(env.commands[:,1].max())])
            boundary=environment(9,device);values=torch.tensor([-10.,-2.0001,-2.,-1.,0.,1.,2.,2.0001,10.],device=device)
            calls=[0]
            def supplied_uniform(tensor,*args,**kwargs):
                calls[0]+=1
                return tensor.copy_(values) if calls[0]==2 else tensor.fill_(.26 if calls[0]==3 else 1.)
            with patch.object(torch.Tensor,'uniform_',supplied_uniform):FudanEnv.resample(boundary,torch.arange(9,device=device))
            verify(device+'_strict_signed_boundary',torch.equal(boundary.commands[:,0],(values.abs()<=2.).float()))
            before=boundary.commands.clone();ids=torch.tensor([1,7],device=device)
            FudanEnv.resample(boundary,ids)
            others=torch.tensor([0,2,3,4,5,6,8],device=device)
            verify(device+'_partial_resample_preserves_others',torch.equal(boundary.commands[others],before[others]))
            legacy_cfg=FudanWalkCfg();legacy_cfg.commands.yaw_range=[-2.,2.]
            legacy=environment(16,device,legacy_cfg);legacy.command_sampling=dict(sampling.LEGACY_COMMAND_SAMPLING)
            torch.manual_seed(725);expected=torch.empty(16,3,device=device)
            expected[:,0]=torch.empty(16,device=device).uniform_(-2.,2.)
            expected[:,1]=torch.empty(16,device=device).uniform_(-2.,2.)
            expected[:,2]=torch.empty(16,device=device).uniform_(.22,.30)
            state=torch.get_rng_state() if device=='cpu' else torch.cuda.get_rng_state()
            torch.manual_seed(725);FudanEnv.resample(legacy,torch.arange(16,device=device))
            after=torch.get_rng_state() if device=='cpu' else torch.cuda.get_rng_state()
            verify(device+'_old_recipe_commands_and_rng_preserved',torch.equal(legacy.commands,expected) and torch.equal(after,state))
        from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg
        step=FudanStepCfg();step_sampling=sampling.command_config(step)
        verify('step_yaw_and_sampling_unchanged',step.commands.yaw_range==[-.5,.5]
            and step_sampling['version']=='fudan-parking-commands-v1' and 'zero_vx_above_yaw' not in step_sampling)
        from scripts.agent.fudan_ppo_walk.reset_v2_fixture import fixture
        from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
        env=fixture(32,RESET_CONTRACT);sampling.initialize(env)
        env.command_mode.zero_();FudanEnv.resample(env,torch.arange(32))
        obs,_=env.observations();high=env.commands[:,1].abs()>2.
        verify('actor_receives_zero_vx_and_high_yaw',high.any() and obs[high,6].eq(0).all()
               and torch.equal(obs[high,7],env.commands[high,1]*.25))
        env.command_mode[:]=2;env.command_elapsed[:]=300;env.stop_command_step[:]=300
        env.commands[:,:2]=1.;env.step(torch.zeros(32,6),auto_reset=False)
        verify('parking_timer_still_sets_both_commands_zero',env.commands[:,:2].eq(0).all())
        env.commands[:,0]=.3;env.commands[:,1]=8.;before=env.commands.clone()
        env.step(torch.zeros(32,6),auto_reset=False,resample=False)
        verify('fixed_evaluation_commands_are_not_resampled',torch.equal(env.commands,before))
    report=dict(passed=all(checks.values()),checks=checks,command_sampling=cfg,actual=actual,
        physics_steps=0,model_forwards=0,optimizer_updates=0,scope='Actual CPU/CUDA resampling and boundaries; not10rad/s behavior acceptance')
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
