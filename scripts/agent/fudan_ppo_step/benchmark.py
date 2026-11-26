"""GPU sampling benchmark, optional real PPO-buffer/minibatch memory check; no updates."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import SOURCE, FudanStepCfg, recipe_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.env import StepEnv
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.runner import validate_source
from .check import source_fingerprint


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path(SOURCE))
    p.add_argument('--ppo-memory-check',action='store_true',help='Allocate actual rollout storage and backprop one minibatch; never step optimizers')
    p.add_argument('--all-levels',action='store_true',help='Cover all platform heights instead of only the initial level')
    a=p.parse_args()
    if a.steps<1 or a.num_envs<1:p.error('Positive steps and num-envs required')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,optimizer_updates=0,backend='mujoco_warp_gpu',num_envs=a.num_envs)
    env=None;runner=None
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Benchmark cannot train')):
            torch.set_num_threads(1)
            cfg=FudanStepCfg()
            if a.all_levels:cfg.step.curriculum=False
            env=StepEnv(a.num_envs,seed=31002,cfg=cfg)
            if a.all_levels:
                env.terrain.terrain_levels[:]=torch.arange(a.num_envs,device=env.device).remainder(len(cfg.step.heights))
            policy,saved=load_policy(a.checkpoint,device='cuda:0')
            validate_source(saved,cfg)
            if a.ppo_memory_check:
                from ct_mjlab.envs.chuanliantui.fudan_ppo_step.runner import StepRunner
                runner=StepRunner(env,log_dir=a.output/'memory-only')
                runner.policy.load_state_dict(policy.state_dict())
                size=a.num_envs*runner.num_steps_per_env//runner.ppo_config['num_mini_batches']
                inputs=torch.zeros(size,25,device=env.device);history_probe=torch.zeros(size,125,device=env.device)
                critic_probe=torch.zeros(size,144,device=env.device)
                before={k:v.detach().cpu().clone() for k,v in runner.policy.state_dict().items()}
                action=runner.policy.act(inputs,history_probe)
                loss=runner.policy.evaluate(critic_probe).square().mean()-runner.policy.get_actions_log_prob(action).mean()*.001
                loss.backward();runner.policy.zero_grad(set_to_none=True)
                runner.policy.encode(history_probe).square().mean().backward();runner.policy.zero_grad(set_to_none=True)
                assert all(torch.equal(v.cpu(),before[k]) for k,v in runner.policy.state_dict().items())
                assert not runner.alg.optimizer.state and not runner.alg.extra_optimizer.state
                del inputs,history_probe,critic_probe,action,loss
                report.update(ppo_memory_checked=True,minibatch=size,rollout_steps=runner.num_steps_per_env,
                    memory_probe_weights_unchanged=True)
            obs,_,history=env.reset()
            timings=[];free=[]
            with torch.inference_mode():
                for i in range(20+a.steps):
                    if time.time()>=a.deadline:raise TimeoutError('Sampling benchmark budget expired')
                    torch.cuda.synchronize();start=time.perf_counter()
                    action,_=policy.act_inference(obs,history)
                    obs,_,_,_,_,history=env.step(action)
                    torch.cuda.synchronize()
                    if i>=20:timings.append(time.perf_counter()-start)
                    if i%10==0:free.append(torch.cuda.mem_get_info())
            model=env.env.sim.mj_model
            report.update(passed=True,steps=a.steps,warmup_steps=20,
                physics_steps=env.contacts.total_ticks,reset_warmup_physics_steps=cfg.control.decimation,
                measured_seconds=sum(timings),transitions_per_second=a.num_envs*a.steps/sum(timings),
                p95_policy_step_s=float(torch.tensor(timings).quantile(.95)),
                ngeom=model.ngeom,terrain_collision_geoms=len(env.env._terrain_friction_ids),
                robot_collision_geoms=sum(bool(model.geom_contype[g] or model.geom_conaffinity[g])
                    for g in range(model.ngeom) if model.geom_bodyid[g]!=model.body('terrain').id),
                free_memory_min_bytes=min(x[0] for x in free),total_gpu_memory_bytes=free[0][1],
                memory_includes_other_processes=True,recipe=recipe_record(cfg),
                constraint_capacity=env.env.sim.wp_data.njmax,
                constraint_peak=int(env.contacts.constraint_peak),
                global_contact_capacity=env.env.sim.wp_data.naconmax,
                global_contact_peak=int(env.contacts.global_contact_peak),
                capacity_overflow=bool(env.contacts.capacity_overflow.any()),
                all_platform_levels_sampled=a.all_levels,
                cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                source_fingerprint=source_fingerprint(),scope='Rollout sampling only; no training throughput or behavioral acceptance')
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        if runner is not None:runner.writer.close()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='recipe'}),flush=True)


if __name__=='__main__':main()
