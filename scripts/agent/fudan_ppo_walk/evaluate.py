"""Final GPU validation on the frozen 987-case matrix; diagnostic subsets stay separate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from ct_mjlab.fudan_ppo_walk.protocol import samples,manifest
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples,gpu_row,arrays,update_evaluation_commands
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT,RESET_V2_CONTRACT
from ct_mjlab.fudan_ppo_walk.disturbances import from_task_config,from_record


def command(case,t):
    result=case['segments'][0]['command']
    for segment in case['segments']:
        if t>=segment['start']-1e-9:result=segment['command']
    return result


def evaluation_items(diagnostic=False,diagnostic_height=None):
    items=samples()
    if diagnostic_height is not None and not diagnostic:
        raise ValueError('--diagnostic-height requires --diagnostic')
    if diagnostic:
        nominal=items[::21]
        # 只筛选固定高度诊断，不改变正式987例或原20例诊断。
        selected=nominal[:20] if diagnostic_height is None else [s for s in nominal
            if all(segment['command'][2]==diagnostic_height for segment in s['case']['segments'])]
        items=[dict(s,seed=30001+i,id=f'diagnostic_{i}') for i,s in enumerate(selected)]
    return items


def protocol_record(items,diagnostic):
    result=manifest()
    if diagnostic:
        result.update(mode='diagnostic_not_formal_acceptance',samples=items)
    else:result['mode']='formal_987_samples'
    return result


def gpu(args):
    from ct_mjlab.fudan_ppo_walk.env import FudanEnv
    policy,metadata=load_policy(args.checkpoint,'cuda:0')
    disturbances=from_task_config()
    if metadata.get('reset_contract') not in (RESET_CONTRACT,RESET_V2_CONTRACT):
        raise ValueError('This GPU evaluator requires PD-centred reset v2/v3; use the preserved trial source for legacy models')
    items=evaluation_items(args.diagnostic,args.diagnostic_height)
    start=time.time();args.output.mkdir(parents=True,exist_ok=False)
    protocol=protocol_record(items,args.diagnostic)
    protocol.update(validation_engine='gpu',validation_backend='mujoco_warp_gpu',
        disturbances=disturbances.record(),training_disturbances=from_record(metadata.get('disturbances')).record())
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    trace_dir=args.output/'traces';trace_dir.mkdir()
    fixture=args.output/'fixtures';fixture.mkdir()
    for offset in range(0,len(items),args.batch_size):
        if time.time()>=args.deadline:break
        batch=items[offset:offset+args.batch_size]
        env=FudanEnv(len(batch),seed=50001+offset,randomized=True,reset_contract=metadata['reset_contract'],disturbances=disturbances)
        obs,_=configure_samples(env,batch,fixture)
        env.action.record_substeps=True
        rows=[gpu_row(env.env).cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()]
        term=[np.zeros(len(batch),dtype=bool)]
        max_steps=round(max(s['case']['duration'] for s in batch)/.01)
        try:
            with torch.inference_mode():
                for step in range(max_steps):
                    if time.time()>=args.deadline:break
                    obs=update_evaluation_commands(env,[command(sample['case'],step*.01) for sample in batch],obs)
                    actions,_=policy.act_inference(obs,env.history)
                    obs,_,_,done,_,_=env.step(actions,auto_reset=False,resample=False)
                    rows.extend(torch.stack(env.action.substep_rows).cpu().numpy())
                    poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
                    term.extend([(env.fail_ticks>100).cpu().numpy().copy()]*5)
                    if step%100==0:print(json.dumps(dict(engine='mjlab',batch_start=offset,step=step,total=max_steps)),flush=True)
        finally:
            rows=np.asarray(rows);poses=np.asarray(poses);term=np.asarray(term)
            origins=env.env.scene.env_origins.cpu().numpy()
            for i,sample in enumerate(batch):
                end=min(len(rows),round(sample['case']['duration']/.002)+1)
                q=poses[:min(len(poses),round(sample['case']['duration']/.01)+1),i].copy();q[:,:3]-=origins[i]
                np.savez_compressed(trace_dir/(sample['id']+'.npz'),**arrays(rows[:end,i],q,terminated=term[:end,i]))
            env.close()
    (args.output/'execution.json').write_text(json.dumps(dict(engine='mjlab',validation_engine='gpu',
        validation_backend='mujoco_warp_gpu',wall_s=time.time()-start,
        complete_traces=len(list(trace_dir.glob('*.npz'))),planned=len(items)),indent=2)+'\n')



def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('engine',nargs='?',choices=('gpu',),default='gpu');p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--diagnostic-height',type=float,choices=(.22,.26,.30))
    a=p.parse_args();torch.set_num_threads(1);gpu(a)

if __name__=='__main__':main()
