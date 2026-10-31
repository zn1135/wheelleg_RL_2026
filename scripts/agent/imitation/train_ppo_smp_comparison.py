#!/usr/bin/env python3
"""Run one explicitly budgeted comparison group and preserve its true initial state."""
import argparse,json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ct_mjlab.scripts.train import train
from ct_mjlab.smp.ppo import SmpOnPolicyRunner


def run(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    original=SmpOnPolicyRunner.learn
    def learn(runner,**kw):
        assert kw['num_learning_iterations']==args.iterations and runner.bc_init
        start=runner.alg.completed_iterations if args.bc_finetune else 0
        assert bool(runner.alg.optimizer.state)==bool(args.resume)
        assert runner.env.unwrapped.common_step_counter==start*24
        raw=runner.env.unwrapped;obs=runner.env.get_observations()
        runner.save(str(args.output/'initial.pt'))
        torch.save(dict(qpos=raw.sim.data.qpos.cpu().clone(),qvel=raw.sim.data.qvel.cpu().clone(),
            obs={k:v.cpu().clone() for k,v in obs.items()},rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            leg_delay=raw.action_manager.get_term('pd').action_delay_steps.cpu().clone(),
            wheel_delay=raw.action_manager.get_term('pd').wheel_action_delay_steps.cpu().clone()),args.output/'initial-state.pt')
        report=dict(group=args.group,authorized_iterations=args.iterations,num_envs=args.num_envs,seed=args.seed,
            bc_finetune=args.bc_finetune,
            start_completed_iterations=start,resume=str(args.resume) if args.resume else None,
            ppo_init=str(args.ppo_init) if args.ppo_init else None,
            phase_mix=args.phase_mix,
            wheel_std=args.wheel_std,
            run_dir=str(Path(runner.logger.log_dir).resolve()),
            started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
        result=original(runner,**kw)
        report.update(completed=True,ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            checkpoint_iteration_index=runner.current_learning_iteration,policy_steps=raw.common_step_counter)
        if hasattr(runner,'phase_mixing'):report['phase_mixing']=runner.phase_mixing
        if args.bc_finetune:
            report.update(completed_training_iterations=runner.alg.completed_iterations,
                          actor_updates=runner.alg.actor_updates,critic_updates=runner.alg.critic_updates)
            assert runner.alg.completed_iterations==start+args.iterations
        assert runner.current_learning_iteration==start+args.iterations-1 and raw.common_step_counter==(start+args.iterations)*24
        (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
        return result
    with patch.object(SmpOnPolicyRunner,'learn',learn):
        train(SimpleNamespace(task='chuanliantui',seed=args.seed,num_envs=args.num_envs,device='cuda:0',
            iterations=args.iterations,bc_init=None if args.resume or args.ppo_init else args.bc,
            ppo_init=args.ppo_init,smp_prior=args.prior,smp_weight=0. if args.group=='taskonly' else 1.,
            phase_mix=args.phase_mix,
            wheel_std=args.wheel_std,
            bc_finetune=args.bc_finetune,resume=args.resume,
            run_name='wheel39-'+('wheelstd'+str(args.wheel_std)+'-' if args.wheel_std is not None else '')+('phase-mix-' if args.phase_mix else '')+('bcfinetune-' if args.bc_finetune else '')+args.group+'-'+('continue-' if args.resume else '')+str(args.iterations)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=('taskonly','smp'),required=True)
    for name in ('prior','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bc',type=Path)
    p.add_argument('--iterations',type=int,required=True);p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bc-finetune',nargs='?',const='v1',default=False,choices=('v1','v2','isaac-standup','isaac-standup-v2','isaac-standup-v3'))
    p.add_argument('--resume',type=Path)
    p.add_argument('--ppo-init',type=Path)
    p.add_argument('--phase-mix',action='store_true')
    p.add_argument('--wheel-std',type=float,choices=(.005,.01,.02,.05))
    args=p.parse_args()
    if args.iterations<1:raise SystemExit('Explicit positive budget required')
    if args.resume and not args.bc_finetune:raise SystemExit('This comparison wrapper resumes BC refinement only')
    if args.ppo_init and (args.resume or args.bc or args.bc_finetune not in ('isaac-standup-v2','isaac-standup-v3')):
        raise SystemExit('PPO transfer requires isaac-standup-v2 and no --bc/--resume')
    if not (args.bc or args.resume or args.ppo_init):raise SystemExit('Choose --bc, --resume or --ppo-init')
    if args.phase_mix and args.bc_finetune not in ('isaac-standup-v2','isaac-standup-v3'):raise SystemExit('Phase mixing requires isaac-standup-v2/v3')
    if args.wheel_std is not None and args.bc_finetune not in ('isaac-standup-v2','isaac-standup-v3'):raise SystemExit('Wheel noise change requires isaac-standup-v2/v3')
    run(args)
