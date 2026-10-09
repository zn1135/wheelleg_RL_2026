"""Entrypoints for the independent mjlab chuanliantui task."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import torch
from .config import ROOT, manifest, runner_cfg

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('train','play','eval','export','check'):
        p=sub.add_parser(name)
        p.add_argument('--task',choices=['chuanliantui'],default='chuanliantui')
        p.add_argument('--seed',type=int,default=42)
        if name in ('train','play','check'):
            p.add_argument('--num-envs',type=int,default=32 if name=='train' else 1)
            p.add_argument('--device',default='cuda:0')
        if name in ('play','eval','export'):p.add_argument('--checkpoint',type=Path,required=name=='export')
        if name in ('play','eval'):
            p.add_argument('--seconds',type=float,default=30)
            p.add_argument('--render',action='store_true');p.add_argument('--report',type=Path)
        if name=='play':p.add_argument('--backend',choices=['mjlab','cpu'],default='mjlab')
        if name=='export':p.add_argument('--output',type=Path,required=True)
        if name=='train':
            p.add_argument('--iterations',type=int,default=20000);p.add_argument('--run-name',default='')
            p.add_argument('--resume',type=Path)
        if name=='check':p.add_argument('--steps',type=int,default=32)
    args=parser.parse_args();torch.manual_seed(args.seed)
    if args.command in ('play','eval') and args.seconds<=0:
        parser.error('--seconds must be positive')
    if args.command=='play' and args.backend=='mjlab' and args.report:
        parser.error('--report is available with --backend cpu; use eval for behavior reports')
    if args.command=='export':
        from .policy import export
        print(json.dumps(export(args.checkpoint,args.output),indent=2));return
    if args.command=='eval' or args.command=='play' and args.backend=='cpu':
        from .policy import load_actor
        from .cpu import evaluate
        actor=load_actor(args.checkpoint) if args.checkpoint else None
        print(json.dumps(evaluate(actor,args.seconds,args.render,args.report),indent=2));return
    from .registry import make_task
    env=make_task(args.task,num_envs=args.num_envs,device=args.device,seed=args.seed,randomize=args.command=='train')
    if args.command=='train':
        from .policy import runner_class
        if '/' in args.run_name or '\\' in args.run_name:raise ValueError('run-name must be a label')
        name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+('_'+args.run_name if args.run_name else '')
        path=ROOT/'logs/mjlab/chuanliantui'/name;path.mkdir(parents=True,exist_ok=False)
        cfg=runner_cfg(args.seed);cfg['run_name']=args.run_name
        (path/'interface.json').write_text(json.dumps(manifest(),indent=2)+'\n')
        (path/'runner.json').write_text(json.dumps(cfg,indent=2)+'\n')
        runner=runner_class()(env,cfg,str(path),device=args.device)
        if args.resume:runner.load(str(args.resume))
        runner.learn(args.iterations,init_at_random_ep_len=False);return
    if args.command=='check':
        for _ in range(args.steps):
            obs,rew,done,extras=env.step(torch.zeros(args.num_envs,6,device=args.device))
            assert obs['actor'].shape==(args.num_envs,35) and obs['critic'].shape==(args.num_envs,78)
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(rew).all()
        print(json.dumps(dict(kind='finite physics/interface check; no training',steps=args.steps,**manifest()),indent=2));return
    from .policy import load_actor
    actor=load_actor(args.checkpoint,args.device) if args.checkpoint else None
    import mujoco
    import time
    viewer=None
    if args.render:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(env.sim.mj_model,env.sim.mj_data)
    try:
        for _ in range(round(args.seconds/.02)):
            start=time.monotonic()
            with torch.inference_mode():action=actor(env.obs) if actor else torch.zeros(args.num_envs,6,device=args.device)
            env.step(action)
            if viewer:
                if not viewer.is_running():break
                d=env.sim.mj_data;d.qpos[:]=env.sim.data.qpos[0].cpu().numpy();d.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
                mujoco.mj_forward(env.sim.mj_model,d);viewer.sync();time.sleep(max(0,.02-time.monotonic()+start))
    finally:
        if viewer:viewer.close()
        env.close()

if __name__=='__main__':main()
