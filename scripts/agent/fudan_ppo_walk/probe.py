"""Bounded GPU sample-batch probe with zero optimizer updates."""
import argparse,json,time
from pathlib import Path
import torch
from ct_mjlab.fudan_ppo_walk.env import FudanEnv


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--num-envs',type=int,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.time();env=None;report=dict(passed=False,optimizer_updates=0,startup_policy_steps=0,policy_steps=0)
    try:
        env=FudanEnv(a.num_envs,seed=30021)
        if time.time()>=a.deadline:raise TimeoutError('Probe budget expired before startup')
        env.reset();report['startup_policy_steps']=1
        for _ in range(48):
            if time.time()>=a.deadline:raise TimeoutError('Probe budget expired')
            env.step(torch.zeros(a.num_envs,6,device=env.device));report['policy_steps']+=1
        report.update(passed=True,cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),num_envs=a.num_envs,
            transitions_per_ppo_iteration=a.num_envs*48,minibatch=a.num_envs*12)
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        report['total_policy_steps']=report['startup_policy_steps']+report['policy_steps']
        report['physics_substeps']=5*report['total_policy_steps']
        report['wall_s']=time.time()-start;a.output.write_text(json.dumps(report,indent=2)+'\n')
        if env is not None:env.close()

if __name__=='__main__':main()
