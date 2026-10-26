#!/usr/bin/env python3
"""Audit matched initial states and both explicitly budgeted PPO runs; no training."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from assess_smp_ppo import assess


def check(args):
    root=args.run;initial=[];states=[];runs={}
    for group,weight in (('taskonly',0.),('smp',1.)):
        directory=root/group;run=json.loads((directory/'run.json').read_text())
        assert run['completed'] and run['authorized_iterations']==300 and run['num_envs']==2048 and run['seed']==42
        assert run['policy_steps']==7200 and run['checkpoint_iteration_index']==299
        path=Path(run['run_dir'])/'model_299.pt'
        assess(SimpleNamespace(checkpoint=path,prior=args.prior,expected_iterations=300,output=directory/'budget-check.json'))
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['infos']['smp']['weight']==weight
        first=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        assert not first['optimizer_state_dict']['state'] and first['infos']['env_state']['common_step_counter']==0
        assert saved['infos']['bc_init']==first['infos']['bc_init']
        assert any(not torch.equal(v,saved['actor_state_dict'][k]) for k,v in first['actor_state_dict'].items())
        initial.append(first);states.append(torch.load(directory/'initial-state.pt',map_location='cpu',weights_only=False))
        runs[group]=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            smp_weight=weight,ppo_iterations=300,optimizer_steps=6000,actor_changed=True)
    def equal(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
        else:assert a==b
    for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):equal(initial[0][key],initial[1][key])
    equal(states[0],states[1]);equal(initial[0]['infos']['bc_init'],initial[1]['infos']['bc_init'])
    report=dict(passed=True,total_authorized_iterations=600,actual_iterations_per_group=300,
        actual_optimizer_steps_per_group=6000,actual2048_initial_actor_critic_optimizer_physics_observations_delays_rng_identical=True,
        only_reward_contribution_differs=True,groups=runs,
        scope='One matched seed pair; budget, initialization and logs audited, not statistical superiority or behavior acceptance')
    (root/'training-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    check(p.parse_args())
