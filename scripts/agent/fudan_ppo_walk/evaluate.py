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
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config,from_record
from ct_mjlab.fudan_ppo_walk.selected_policy import selection_record,reference_starts,policy_inputs,reference_angles


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
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    from ct_mjlab.envs.chuanliantui.fitted_leg_load import parameter_record
    from ct_mjlab.envs.fudan_ppo_walk import posture_reward
    from types import SimpleNamespace
    policy,metadata=load_policy(args.checkpoint,'cuda:0')
    posture=posture_reward.from_record(metadata.get('posture_reward'))
    disturbances=from_sim2sim_config()
    if metadata.get('reset_contract') not in (RESET_CONTRACT,RESET_V2_CONTRACT):
        raise ValueError('This GPU evaluator requires PD-centred reset v2/v3; use the preserved trial source for legacy models')
    items=evaluation_items(args.diagnostic,args.diagnostic_height)
    start=time.time();args.output.mkdir(parents=True,exist_ok=False)
    protocol=protocol_record(items,args.diagnostic)
    protocol.update(validation_engine='gpu',validation_backend='mujoco_warp_gpu',
        disturbances=disturbances.record(),training_disturbances=from_record(metadata.get('disturbances')).record(),
        posture_reward=posture)
    from ct_mjlab.envs.fudan_ppo_walk.wheel_physics import record as wheel_record
    protocol['physical_wheel_model']=wheel_record(FudanWalkCfg.sim.wheel_model)
    fitted = getattr(args,'leg_load_profile','legacy')=='fitted' and getattr(args,'response_parameters',None) is None
    protocol['physical_leg_load']=parameter_record(fitted)
    if getattr(args,'response_parameters',None) is not None:
        import hashlib
        protocol['response_parameters']=dict(path=str(args.response_parameters.resolve()),
            sha256=hashlib.sha256(args.response_parameters.read_bytes()).hexdigest(),
            sides=json.loads(args.response_parameters.read_text()).get('sides',['right']),initial_seal_force_n=0.)
    if getattr(args,'selection',None) is not None:
        protocol['selected_policy']=args.selection
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    trace_dir=args.output/'traces';trace_dir.mkdir()
    fixture=args.output/'fixtures';fixture.mkdir()
    for offset in range(0,len(items),args.batch_size):
        if time.time()>=args.deadline:break
        batch=items[offset:offset+args.batch_size]
        task=FudanWalkCfg();task.sim=SimpleNamespace(**section_dict(task.sim));task.sim.fitted_leg_load=fitted
        posture_reward.configure(task,posture)
        env=FudanEnv(len(batch),seed=50001+offset,randomized=True,reset_contract=metadata['reset_contract'],disturbances=disturbances,cfg=task)
        obs,_=configure_samples(env,batch,fixture)
        response=None
        if getattr(args,'response_parameters',None) is not None:
            from scripts.agent.real2sim.length_fit_policy_load import LengthFitPolicyLoad
            response=LengthFitPolicyLoad(env,args.response_parameters)
        starts=reference_starts(batch,obs) if getattr(args,'selection',None) is not None else None
        target=obs.new_full((len(batch),),np.deg2rad(args.selection['parking_pitch_deg'])) if starts is not None else None
        env.action.record_substeps=True
        rows=[gpu_row(env.env).cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()]
        term=[np.zeros(len(batch),dtype=bool)]
        motor_velocities=[];references=[]
        max_steps=round(max(s['case']['duration'] for s in batch)/.01)
        try:
            with torch.inference_mode():
                for step in range(max_steps):
                    if time.time()>=args.deadline:break
                    obs=update_evaluation_commands(env,[command(sample['case'],step*.01) for sample in batch],obs)
                    actor_obs,history=policy_inputs(obs,env.history,step*.01,starts,target) if starts is not None else (obs,env.history)
                    actions,_=policy.act_inference(actor_obs,history)
                    obs,_,_,done,_,_=env.step(actions,auto_reset=False,resample=False)
                    if response is not None:response.after_policy_step()
                    motor_velocities.append(env.env.sim.data.qvel[:,env.action._motor_v_adr].cpu().numpy().copy())
                    references.append(reference_angles(step*.01,starts,target).cpu().numpy() if starts is not None else np.zeros(len(batch)))
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
                trace=arrays(rows[:end,i],q,terminated=term[:end,i])
                count=min(len(motor_velocities),round(sample['case']['duration']/.01))
                trace.update(policy_time=np.arange(count)*.01,motor_qvel_rad_s=np.asarray(motor_velocities).reshape(-1,len(batch),6)[:count,i],
                    pitch_reference_rad=np.asarray(references).reshape(-1,len(batch))[:count,i])
                np.savez_compressed(trace_dir/(sample['id']+'.npz'),**trace)
            try:
                if response is not None:
                    response_folder=args.output/f'response-batch-{offset}';response_folder.mkdir()
                    response.save(response_folder)
            finally:env.close()
    (args.output/'execution.json').write_text(json.dumps(dict(engine='mjlab',validation_engine='gpu',
        validation_backend='mujoco_warp_gpu',wall_s=time.time()-start,
        complete_traces=len(list(trace_dir.glob('*.npz'))),planned=len(items)),indent=2)+'\n')



def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('engine',nargs='?',choices=('gpu',),default='gpu');p.add_argument('--checkpoint',type=Path,
        help='Explicit checkpoint uses its original inputs; omit to use the adopted8000+2deg recipe')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--diagnostic-height',type=float,choices=(.22,.26,.30))
    p.add_argument('--response-parameters',type=Path,help='opt-in length-fit plant bundle with right or mirrored-both sides; no training-default change')
    p.add_argument('--leg-load-profile',choices=('fitted','legacy'),default='legacy',help='original linear-spring plant by default; fitted explicitly replays the archived mirrored load')
    a=p.parse_args();a.selection=None
    if a.checkpoint is None:
        a.selection=selection_record();a.checkpoint=Path(a.selection['checkpoint'])
    torch.set_num_threads(1);gpu(a)

if __name__=='__main__':main()
