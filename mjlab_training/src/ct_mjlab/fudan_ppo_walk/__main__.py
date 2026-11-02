"""Independent Fudan task entry, with an explicit time or iteration budget."""
import argparse
import json
from pathlib import Path
import time
import torch
from .env import FudanEnv
from .contract import RESET_CONTRACT
from .runner import FudanRunner,snapshot
from .disturbances import from_task_config,from_record


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--num-envs',type=int,required=True)
    train.add_argument('--log-dir',type=Path,required=True)
    budget=train.add_mutually_exclusive_group(required=True)
    budget.add_argument('--deadline',type=float);budget.add_argument('--iterations',type=int)
    train.add_argument('--precheck',type=Path,required=True)
    train.add_argument('--resume',type=Path)
    train.add_argument('--migrate-history-v2',action='store_true')
    export=sub.add_parser('export');export.add_argument('--checkpoint',type=Path,required=True);export.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);torch.set_num_threads(1)
    if args.command=='export':
        from .export import export as export_policy
        return export_policy(args.checkpoint,args.output)
    source=(torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume is not None else {})
    disturbances=from_task_config()
    proof=json.loads(args.precheck.read_text())
    if (not proof['passed'] or proof.get('reset_contract')!=RESET_CONTRACT
            or proof.get('kind')!='physical-visible-precheck' or proof.get('num_envs')!=20
            or proof.get('physics_steps',0)<=0):
        raise ValueError('A passed current reset/history physical precheck is required; older prechecks cannot approve this recipe')
    if from_record(proof.get('disturbances'))!=disturbances:
        raise ValueError('Physical precheck disturbance settings differ from the requested training settings')
    if args.migrate_history_v2 and args.resume is None:raise ValueError('--migrate-history-v2 requires --resume')
    if args.num_envs%4:raise ValueError('Four equal reference minibatches required')
    if args.iterations is not None and args.iterations<=0:raise ValueError('Iteration budget must be positive')
    if args.deadline is not None and time.time()>=args.deadline:raise ValueError('Training budget already expired')
    torch.manual_seed(1);args.log_dir.mkdir(parents=True,exist_ok=False);snapshot(args.log_dir)
    env=None
    try:
        env=FudanEnv(args.num_envs,seed=1,randomized=True,disturbances=disturbances)
        runner=FudanRunner(env,log_dir=args.log_dir)
        if args.resume is not None:
            options={}
            if args.migrate_history_v2:options['migrate_history_v2']=True
            if from_record(source.get('disturbances'))!=disturbances:options['allow_disturbance_change']=True
            restored=runner.load(args.resume,**options)
            print(json.dumps(restored, indent=2), flush=True)
        # 次数预算不叠加旧墙钟期限；恰好完成指定轮数后保存停止。
        report=(runner.learn_until(float('inf'),max_iterations=runner.iteration+args.iterations) if args.iterations is not None
            else runner.learn_until(args.deadline))
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        (args.log_dir/'failure.json').write_text(json.dumps(dict(error=repr(exc),time=time.time()),indent=2)+'\n')
        if env is not None:
            import numpy as np
            np.savez_compressed(args.log_dir/'failure_state.npz',qpos=env.env.sim.data.qpos.cpu().numpy(),qvel=env.env.sim.data.qvel.cpu().numpy())
        raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
