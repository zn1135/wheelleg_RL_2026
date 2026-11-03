"""Bounded GPU parking estimator intervention; no training or production policy changes."""
import time
PROCESS_STARTED=time.time()
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import signal
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples,update_evaluation_commands,gpu_row,arrays
from .evaluate import command

BRANCHES=('baseline','true_vx_at_parking','true_xyz_at_parking')
CASES={'diagnostic_0':(2.,7.),'diagnostic_13':(10.,15.),'diagnostic_14':(10.,15.)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervention(estimate,label,active,branch):
    result=estimate.clone()
    if branch=='true_vx_at_parking':result[active,0]=2.*label[active,0]
    elif branch=='true_xyz_at_parking':result[active]=2.*label[active]
    elif branch!='baseline':raise ValueError(branch)
    return result


def check():
    estimate=torch.arange(12,dtype=torch.float32).reshape(4,3)
    label=torch.full((4,3),-.051);active=torch.tensor([False,True,False,True])
    for branch in BRANCHES:
        used=intervention(estimate,label,active,branch)
        assert torch.equal(used[~active],estimate[~active])
        if branch=='baseline':assert torch.equal(used,estimate)
        elif branch=='true_vx_at_parking':
            assert torch.equal(used[:,1:],estimate[:,1:]) and torch.equal(used[active,0],2*label[active,0])
        else:assert torch.equal(used[active],2*label[active])
    return dict(passed=True,branches=3,label_scale=2.,physics_steps=0,model_forwards=0,optimizer_updates=0)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=PROCESS_STARTED;deadline=start+args.budget_seconds;stop=deadline-15.
    def guard():
        if time.time()>=stop:raise TimeoutError('Diagnostic compute budget exhausted')
    def terminated(*a):raise TimeoutError('External diagnostic deadline reached')
    signal.signal(signal.SIGTERM,terminated)
    def forbidden(*a,**kw):raise AssertionError('Optimizer updates forbidden')
    source=args.recording
    items=json.loads((source/'protocol.json').read_text())['samples']
    assert len(items)==15 and all(not x['randomized'] for x in items)
    assert {x['id'] for x in items}=={f'diagnostic_{i}' for i in range(15)}
    cfg=from_task_config()
    assert not (cfg.action_delay or cfg.sensor_delay or cfg.pushes),'Freeze zero-added-disturbance baseline'
    files=[args.checkpoint,*Path('mjlab_training/src/ct_mjlab/fudan_ppo_walk').glob('*.py'),
        *Path('mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk').glob('*.py'),
        *Path('mjlab_training/src/ct_mjlab/rsl_rl').rglob('*.py'),
        Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'),Path(__file__)]
    protected={str(p):sha(p) for p in files}
    budget=dict(started_unix=start,deadline_unix=deadline,compute_stop_unix=stop,budget_seconds=args.budget_seconds,
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),recording=str(source),
        branches=BRANCHES,windows=CASES,source_sha256=protected,optimizer_updates=0,
        scope='Frozen7000-v2 GPU counterfactual diagnostic; not deployable policy or final acceptance')
    (args.output/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    results=[];env=None;encoder_calls=actor_calls=physical_steps=0;error=None
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=forbidden))
        stack.enter_context(patch.object(torch.optim.SGD,'step',side_effect=forbidden))
        try:
            guard();torch.set_num_threads(1)
            policy,metadata=load_policy(args.checkpoint,'cuda:0')
            assert metadata['iterations']==7000 and metadata['reset_contract']==RESET_V2_CONTRACT
            original={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
            for branch in BRANCHES:
                guard();folder=args.output/branch;folder.mkdir()
                torch.manual_seed(50001)
                env=FudanEnv(15,seed=50001,randomized=True,reset_contract=metadata['reset_contract'],disturbances=cfg)
                guard();obs,_=configure_samples(env,items,folder/'fixtures')
                # Same layout and seeded parameters as the original15-environment replay.
                for sample in items:
                    name=sample['id']+'_initial.npz'
                    with np.load(source/'fixtures'/name) as old,np.load(folder/'fixtures'/name) as new:
                        for key in old.files:assert np.array_equal(old[key],new[key]),(branch,name,key)
                env.action.record_substeps=True
                rows=[gpu_row(env.env).cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()]
                records=[];post_torques=[];schema=None;completed=0;branch_error=None
                starts=torch.tensor([CASES[s['id']][0] if s['id'] in CASES else float('inf') for s in items],device=env.device)
                try:
                    with torch.inference_mode():
                        for step in range(1500):
                            guard();t=step*.01
                            obs=update_evaluation_commands(env,[command(s['case'],t) for s in items],obs)
                            raw,estimate=policy.act_inference(obs,env.history);encoder_calls+=1;actor_calls+=1
                            active=(t>=starts)&env.commands[:,:2].eq(0).all(-1)
                            used=intervention(estimate,env.linear,active,branch)
                            applied=raw
                            if branch!='baseline' and active.any():
                                applied=policy.actor(torch.cat((obs,used),-1));actor_calls+=1
                            current=gpu_row(env.env)
                            values=dict(obs=obs,history=env.history,estimate_m_s=estimate/2.,used_m_s=used/2.,
                                label_m_s=env.linear,instantaneous_m_s=current[:,3:6],command=env.commands,
                                original_action=raw,applied_action=applied,
                                wheel_target_rad_s=(applied[:,[2,5]].clamp(-100,100)*10.).clamp(-150,150),
                                wheel_velocity_rad_s=env.env.sim.data.qvel[:,env.action._motor_v_adr][:,[2,5]],
                                position=current[:,:3],roll_pitch_deg=current[:,12:14],
                                intervention_active=(active if branch!='baseline' else torch.zeros_like(active))[:,None].float())
                            if schema is None:
                                schema={};offset=0
                                for key,value in values.items():schema[key]=[offset,offset+value.shape[1]];offset+=value.shape[1]
                            records.append(torch.cat(list(values.values()),-1).cpu().numpy())
                            guard();obs,_,_,_,_,_=env.step(applied,auto_reset=False,resample=False)
                            completed+=1;physical_steps+=15*5
                            rows.extend(torch.stack(env.action.substep_rows).cpu().numpy())
                            poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
                            post_torques.append(env.action.torques.cpu().numpy().copy())
                            if step%300==0:print(json.dumps(dict(branch=branch,step=step,elapsed_s=time.time()-start)),flush=True)
                except BaseException as exc:
                    branch_error=repr(exc);raise
                finally:
                    records=np.asarray(records[:completed]);rows=np.asarray(rows);poses=np.asarray(poses)
                    trace=folder/'traces';trace.mkdir()
                    origins=env.env.scene.env_origins.cpu().numpy()
                    if completed:
                        data={k:records[:,:,a:b] for k,(a,b) in schema.items()}
                        np.savez_compressed(folder/'policy.npz',time=np.arange(completed)*.01,**data,
                            last_pd_torques_after_step=np.asarray(post_torques))
                        for i,sample in enumerate(items):
                            end=min(len(rows),round(sample['case']['duration']/.002)+1)
                            q=poses[:min(len(poses),round(sample['case']['duration']/.01)+1),i].copy();q[:,:3]-=origins[i]
                            np.savez_compressed(trace/(sample['id']+'.npz'),**arrays(rows[:end,i],q))
                    record=dict(branch=branch,complete=completed==1500,policy_steps=completed,error=branch_error,
                        initial_states_identical_to_source=True,finished_unix=time.time())
                    results.append(record);(folder/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
                    (folder/'protocol.json').write_text(json.dumps(dict(samples=items,branch=branch,validation_engine='gpu',
                        disturbances=cfg.record(),intervention_windows=CASES),indent=2)+'\n')
                    env.close();env=None
            torch.cuda.synchronize()
            assert all(torch.equal(value,policy.state_dict()[key].cpu()) for key,value in original.items())
        except BaseException as exc:
            error=repr(exc)
        finally:
            if env is not None:env.close()
            unchanged=all(sha(p)==value for p,value in protected.items())
            report=dict(complete=len(results)==3 and all(x['complete'] for x in results),error=error,
                branches=results,started_unix=start,finished_unix=time.time(),wall_s=time.time()-start,
                encoder_batch_calls=encoder_calls,actor_batch_calls=actor_calls,
                explicit_environment_physics_substeps=physical_steps,optimizer_updates=0,protected_files_unchanged=unchanged,
                budget_exceeded=time.time()>deadline,backend_preparation='GPU environment initialization/capture included in wall clock, not explicit substep counter')
            (args.output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)
    if error:raise RuntimeError(error)


def analyze(root):
    report=json.loads((root/'execution.json').read_text());assert report['complete'] and not report['budget_exceeded']
    results={};datasets={}
    for branch in BRANCHES:
        with np.load(root/branch/'policy.npz') as saved:datasets[branch]={k:saved[k].copy() for k in saved.files}
        d=datasets[branch];results[branch]={}
        for name,(lo,hi) in CASES.items():
            i=int(name.split('_')[-1]);mask=(d['time']>=lo)&(d['time']<hi)
            assert mask.sum()==500
            with np.load(root/branch/'traces'/f'{name}.npz') as saved:
                t=saved['time'];select=(t>=lo)&(t<hi)
                speed=np.linalg.norm(saved['linear_world'][select,:2],axis=1)
                first=round(lo/.002);last=round(hi/.002)
                displacement=saved['position'][first:last+1,:2]-saved['position'][first,:2]
                row=dict(mean_speed_m_s=float(speed.mean()),peak_speed_m_s=float(speed.max()),
                    net_dx_m=float(displacement[-1,0]),max_displacement_m=float(np.linalg.norm(displacement,axis=1).max()),
                    pitch_peak_deg=float(np.abs(saved['roll_pitch_deg'][select,1]).max()),
                    base_contact_peak_N=float(saved['base_contact'][select].max()))
            row.update(estimate_vx_m_s=float(d['estimate_m_s'][mask,i,0].mean()),
                label_vx_m_s=float(d['label_m_s'][mask,i,0].mean()),
                estimator_bias_m_s=float((d['estimate_m_s'][mask,i,0]-d['label_m_s'][mask,i,0]).mean()),
                wheel_target_rad_s=d['wheel_target_rad_s'][mask,i].mean(0).tolist(),
                wheel_velocity_rad_s=d['wheel_velocity_rad_s'][mask,i].mean(0).tolist())
            results[branch][name]=row
    comparisons={}
    for branch in BRANCHES[1:]:
        comparisons[branch]={}
        for name,(lo,hi) in CASES.items():
            i=int(name.split('_')[-1]);before=datasets[branch]['time']<lo
            baseline=results['baseline'][name]['max_displacement_m'];changed=results[branch][name]['max_displacement_m']
            comparisons[branch][name]=dict(drift_reduction_fraction=1-changed/baseline,
                pre_intervention_max_position_difference_m=float(np.abs(datasets[branch]['position'][before,i]-datasets['baseline']['position'][before,i]).max()))
    result=dict(results=results,comparisons=comparisons,windows=CASES,execution=report,
        conclusion_scope='Nominal0.30m GPU intervention; true velocity is privileged diagnostic input, not a deployed repair')
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--check-only',action='store_true');p.add_argument('--analyze',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--recording',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--budget-seconds',type=float,default=300.)
    args=p.parse_args()
    if args.check_only:print(json.dumps(check(),indent=2));return
    if args.analyze:analyze(args.analyze);return
    if not args.checkpoint or not args.recording or not args.output:p.error('checkpoint, recording and output required')
    if not 0<args.budget_seconds<=300:p.error('This diagnostic is bounded to at most300 seconds')
    run(args)


if __name__=='__main__':main()
