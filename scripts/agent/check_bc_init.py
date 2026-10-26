#!/usr/bin/env python3
"""Check real BC actor initialization into fresh SMP PPO; no parameter updates."""
import argparse,json,tempfile
from pathlib import Path
import torch
from ct_mjlab.bc import load_actor
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import configure_smp


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 cfg,train=task_registry.get_cfgs('chuanliantui',play=True);cfg.seed=42;configure_smp(cfg,train,args.prior,play=True)
 env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=2,device='cuda:0')
 try:
  runner,_=task_registry.make_alg_runner(env,'chuanliantui',train_cfg=train,device='cuda:0')
  critic={k:v.clone() for k,v in runner.alg._raw_critic.state_dict().items()}
  runner.initialize_from_bc(args.checkpoint)
  actor,saved=load_actor(args.checkpoint,'cuda:0');obs=env.get_observations()
  torch.testing.assert_close(actor(obs),runner.alg._raw_actor(obs),atol=0,rtol=0)
  for k,v in critic.items():torch.testing.assert_close(runner.alg._raw_critic.state_dict()[k],v,atol=0,rtol=0)
  assert not runner.alg.optimizer.state
  torch.testing.assert_close(runner.alg._raw_actor.distribution.std_param,torch.full((6,),.05,device='cuda:0'))
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'untrained_roundtrip.pt';runner.save(str(path));before=dict(runner.bc_init)
   runner.load(str(path));assert runner.bc_init==before
  r=dict(passed=True,optimizer_updates=0,bc_updates=saved['iteration'],actor_outputs_identical=True,critic_unchanged=True,ppo_optimizer_empty=True,bc_init_metadata_roundtrip=True,bc_init=runner.bc_init)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
 finally:env.close()


if __name__=='__main__':main()
