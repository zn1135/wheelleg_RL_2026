"""Frozen 8000-v2, three feedback branches, exact archived fixtures, GPU only."""
import argparse,json,time,signal,shutil
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.fudan_ppo_walk.measurement import FIELDS,update_evaluation_commands,gpu_row,arrays
from ct_mjlab.envs.fudan_ppo_walk.parking_integral import ParkingIntegral,feedback_velocity,finite,REASONS
from .evaluate import command

BRANCHES=('baseline','truth_integral','encoder_integral')
HEIGHTS=('h022','h026','h030')


def compare_initial(expected,actual,folder,sample_id):
    """Keep every differing component before rejecting an initial state; no tolerance."""
    differences={}
    for key,value in actual.items():
        target=np.asarray(expected[key],dtype=value.dtype)
        if np.array_equal(value,target):continue
        out=folder/'initial-differences';out.mkdir(parents=True,exist_ok=True)
        path=out/(sample_id+'_'+key+'.npz')
        delta=value.astype(np.float64)-target.astype(np.float64)
        np.savez_compressed(path,expected=target,actual=value,difference=delta)
        differences[key]=dict(indices=np.argwhere(value!=target).tolist(),max_abs_difference=float(np.abs(delta).max()),
            expected_dtype=str(np.asarray(expected[key]).dtype),actual_dtype=str(value.dtype),arrays=str(path))
    result=dict(all_arrays_identical=not differences,live_inputs_identical_at_simulation_dtype=not differences,differences=differences)
    if differences:(folder/'initial-differences'/(sample_id+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def restore_initial(wrapper,items,source,folders,checks,height):
    """Restore archived nominal fixtures, then refresh inputs without a physics step."""
    env=wrapper.env;action=wrapper.action;mapping=json.loads((source/'mapping.json').read_text())
    assert mapping['reset_contract']==wrapper.reset_contract==RESET_V2_CONTRACT
    assert not any(mapping['disturbances'][k] for k in ('action_delay','sensor_delay','pushes'))
    assert not any(getattr(wrapper.disturbance_config,k) for k in ('action_delay','sensor_delay','pushes'))
    assert all(not sample['randomized'] for sample in items)
    assert env.num_envs==len(items)*len(BRANCHES)
    for name,key in (('_motor_q_adr','motor_q'),('_motor_v_adr','motor_v'),('_spring_ids','spring_ids'),
            ('_spring_q_adr','spring_q'),('_spring_v_adr','spring_v')):
        assert getattr(action,name).cpu().tolist()==mapping[key],key
    for name,key in (('_wheel_response','wheel_response'),('reference_positions','reference')):
        value=getattr(action,name);target=value.new_tensor(mapping[key])
        assert torch.equal(value,target.expand_as(value)),key
    archived=[]
    for sample in items:
        with np.load(source/(sample['id']+'_initial.npz')) as f:archived.append({k:f[k].copy() for k in f.files})
    records=archived*len(BRANCHES)
    def batch(key,target):return torch.as_tensor(np.stack([r[key] for r in records]),device=target.device,dtype=target.dtype)
    for key in FIELDS:
        target=getattr(env.sim.model,key);target.copy_(batch(key,target))
    env.sim.recompute_constants(RecomputeLevel.set_const)
    action.nominal_offset.zero_();action._nominal_pos.copy_(batch('nominal',action._nominal_pos))
    for name,key in (('kp_scale','kp'),('kd_scale','kd'),('motor_strength','strength')):
        target=getattr(action,name);target.copy_(batch(key,target))
    local=batch('qpos',env.sim.data.qpos);world=local.clone();world[:,:3]+=env.scene.env_origins
    env.sim.data.qpos.copy_(world);env.sim.data.qvel.copy_(batch('qvel',env.sim.data.qvel))
    # Do not solve the closed chain again: the archived passive joints are part of the initial state.
    action.reset_joint_positions=local[:,wrapper.robot.indexing.joint_q_adr].clone()
    action.reset_root_height=local[:,2].clone()
    for name in ('_raw_actions','_control_actions','_torques','_action_history','_spring_force_bias','_motor_response_torque',
            'action_delay_steps','wheel_action_delay_steps','sample_velocity'):
        value=getattr(action,name,None)
        if value is not None:value.zero_()
    action.substep_rows.clear()
    for name in ('ctrl','qacc','qacc_warmstart','qfrc_applied','xfrc_applied','time'):
        value=getattr(env.sim.data,name,None)
        if value is not None:value.zero_()
    for name in ('previous','previous_vel','fail_ticks','sums','episode_steps','command_mode','command_elapsed','stop_command_step'):
        getattr(wrapper,name).zero_()
    env.episode_length_buf.zero_();env.common_step_counter=0;wrapper.total_steps=0
    base=mapping['base']
    mass=env.sim.model.body_mass[:,base].clone()
    env._fudan_parameters=dict(base_mass=mass,com=torch.zeros_like(env.sim.model.body_ipos[:,base]),
        friction=torch.full_like(mass,.5),restitution=torch.full_like(mass,.5))
    env.sim.forward();env.scene.update(dt=0.);env.sim.sense()
    action.last_sample_q.copy_(env.sim.data.qpos[:,action._motor_q_adr])
    if hasattr(action,'disturbances'):
        state=action.disturbances;state.reset(slice(None),action.sensor_frame())
        state.action_steps.copy_(batch('action_delay_steps',state.action_steps))
        state.sensor_steps.copy_(batch('sensor_delay_steps',state.sensor_steps))
        state.push_seeds[:]=np.array([r['push_seed'] for r in records])
        action.push_force_scale.copy_(batch('push_force_scale',action.push_force_scale))
    wrapper.previous_position.copy_(wrapper.robot.data.root_link_pos_w);wrapper._update_state()
    wrapper.commands.copy_(wrapper.commands.new_tensor([s['case']['segments'][0]['command'] for s in items]*len(BRANCHES)))
    wrapper.noise_mask=torch.zeros(env.num_envs,dtype=torch.bool,device=wrapper.device)
    clean=wrapper.frame().clone();obs,critic=wrapper.observations()
    wrapper.history=torch.cat((clean.repeat(1,4),wrapper._history_frame),-1)
    wrapper._history_needs_fill.zero_()
    # Each branch receives its own unmodified archival files, with no repeated-name generation.
    actual=dict(qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,nominal=action.nominal_positions,
        kp=action.kp_scale,kd=action.kd_scale,strength=action.motor_strength)
    actual.update({key:getattr(env.sim.model,key) for key in FIELDS})
    if hasattr(action,'disturbances'):
        actual.update(action_delay_steps=state.action_steps,sensor_delay_steps=state.sensor_steps,
            push_seed=torch.as_tensor(state.push_seeds),push_force_scale=action.push_force_scale)
    arrays_cpu={key:value.detach().cpu().numpy() for key,value in actual.items()}
    for b,branch in enumerate(BRANCHES):
        folder=folders[branch]
        for path in source.iterdir():
            if path.is_file():shutil.copy2(path,folder/'fixtures'/path.name)
        for i,sample in enumerate(items):
            idx=b*len(items)+i;expected=dict(archived[i],qpos=world[idx].cpu().numpy())
            result=compare_initial(expected,{k:v[idx] for k,v in arrays_cpu.items()},folder,sample['id'])
            checks.append(dict(height=height,branch=branch,sample_id=sample['id'],qpos_comparison='archive cast to runtime dtype plus environment origin',**result))
    if any(not r['all_arrays_identical'] for r in checks):raise AssertionError('Initial state mismatch; see initial-differences arrays/JSON')
    for b in range(1,len(BRANCHES)):
        lo=b*len(items);hi=lo+len(items)
        assert torch.equal(obs[:len(items)],obs[lo:hi]) and torch.equal(wrapper.history[:len(items)],wrapper.history[lo:hi]),'Branch observation/history mismatch'
    return obs,critic


def policy_tick(env,policy,c,tick,raw,obs):
    """One actor/physics/history step; feedback is sampled at its endpoint."""
    row=gpu_row(env.env);finite(row,obs,env.history)
    true=env.linear.clone();actual,mean=c.begin(tick,raw,row[:,12:14]);n=len(raw)//len(BRANCHES)
    assert torch.equal(actual[:n],raw[:n])
    moving=(raw[:,:2]!=0).any(-1);assert torch.equal(actual[moving],raw[moving])
    old_history=env.history[:,:100].clone();obs=update_evaluation_commands(env,actual,obs)
    assert torch.equal(old_history,env.history[:,:100])
    action,estimate=policy.act_inference(obs,env.history)  # Actor always uses its original encoder.
    finite(action,estimate,true)
    integral_before=c.integral.clone();applied=c.compensation.clone();active=c.active.clone();reason=c.reason.clone();gate_ticks=c.ticks.clone()
    obs,_,_,done,_,_=env.step(action,auto_reset=False,resample=False)
    substeps=torch.stack(env.action.substep_rows);end_row=gpu_row(env.env)
    # A feedback-only encoder call on the newly advanced history; no actor call or extra history frame.
    end_true=env.linear.clone();end_estimate=policy.encode(env.history)
    finite(substeps,end_row,end_true,end_estimate,obs,env.history)
    feedback=feedback_velocity(end_estimate[:,0],encoder=True);feedback[n:2*n]=end_true[n:2*n,0]
    feedback_raw=end_estimate[:,0].clone();feedback_raw[n:2*n]=end_true[n:2*n,0]
    c.observe_contacts(substeps[:,:,14:17]);c.end(raw,feedback,end_row[:,12:14])
    values=dict(user_command=raw,policy_command=actual,feedback_raw=feedback_raw[:,None],feedback_m_s=feedback[:,None],
        truth_m_s=true,encoder_raw=estimate,encoder_m_s=estimate/2.,feedback_truth_m_s=end_true,
        feedback_encoder_raw=end_estimate,feedback_encoder_m_s=end_estimate/2.,
        integral_before=integral_before[:,None],integral_after=c.integral[:,None].clone(),
        compensation=applied[:,None],compensation_after=c.compensation[:,None].clone(),active=active[:,None].float(),reason=reason[:,None].float(),reason_after=c.reason[:,None].float(),
        gate_ticks=gate_ticks[:,None].float(),contact_mean20=mean,contact_end_mean20=c.contacts.mean(0),
        action=action,wheel_action=action[:,[2,5]],policy_position=row[:,:3],policy_roll_pitch_deg=row[:,12:14],contacts=row[:,14:17],
        integral_saturated=(c.integral.abs()>=c.cfg['integral_limit']-1e-7)[:,None].float())
    return obs,done,substeps,values


def run(a):
    budget=json.loads((a.output/'budget.json').read_text());stop=budget['replay_stop_unix']-15.
    def guard():
        if time.time()>=stop:raise TimeoutError('Replay soft stop; reserve time for trace saving')
    def stopped(*args):raise TimeoutError('Replay interrupted at deadline')
    signal.signal(signal.SIGTERM,stopped)
    def forbidden(*args,**kw):raise AssertionError('Optimizer updates forbidden')
    torch.set_num_threads(1);guard()
    policy,metadata=load_policy(a.checkpoint,'cuda:0')
    assert metadata['iterations']==8000 and metadata['reset_contract']==RESET_V2_CONTRACT
    cfg=from_task_config();assert not(cfg.action_delay or cfg.sensor_delay or cfg.pushes)
    before={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    initial_checks=[];executions=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=forbidden))
        stack.enter_context(patch.object(torch.optim.SGD,'step',side_effect=forbidden))
        prepared=[]
        for height in HEIGHTS:
            guard();source=a.recording/height/'gpu15'
            protocol=json.loads((source/'protocol.json').read_text());items=protocol['samples']
            assert len(items)==15 and all(not s['randomized'] for s in items)
            folders={}
            for branch in BRANCHES:
                folder=a.output/height/branch;folder.mkdir(parents=True,exist_ok=False);folders[branch]=folder
                (folder/'protocol.json').write_text(json.dumps(dict(protocol,validation_engine='gpu',validation_backend='mujoco_warp_gpu',integral_branch=branch,score_commands='original_user_commands'),indent=2)+'\n')
                (folder/'traces').mkdir();(folder/'fixtures').mkdir()
            try:
                env=FudanEnv(45,seed=50001,randomized=True,reset_contract=metadata['reset_contract'],disturbances=cfg)
                stack.callback(env.close)
                guard();obs,_=restore_initial(env,items,source/'fixtures',folders,initial_checks,height)
                (a.output/height/'environment-config.txt').write_text(repr(env.env.cfg)+'\n')
                prepared.append((height,items,env,obs))
            except Exception as exc:
                (a.output/'initialization-result.json').write_text(json.dumps(dict(passed=False,height=height,error=repr(exc),checked=len(initial_checks),expected=135),indent=2)+'\n')
                raise
            finally:
                (a.output/'initial-consistency.json').write_text(json.dumps(initial_checks,indent=2)+'\n')
        assert len(initial_checks)==135 and all(r['all_arrays_identical'] for r in initial_checks)
        (a.output/'initialization-result.json').write_text(json.dumps(dict(passed=True,checked=135,expected=135,policy_batches_before_checks=0),indent=2)+'\n')
        # No policy/physics rollout starts until all three heights' 135 initial states pass.
        for height,items,env,obs in prepared:
            started=time.time();rows=[];poses=[];terms=[];logs=[];schema={};completed=0;error=None
            try:
                c=ParkingIntegral(45,env.device,enabled=[False]*15+[True]*30)
                c.reset(torch.arange(45,device=env.device))
                env.action.record_substeps=True
                row=gpu_row(env.env);finite(row)
                c.observe_contacts(row[:,14:17][None])
                rows=[row.cpu().numpy()];poses=[env.env.sim.data.qpos.cpu().numpy().copy()];terms=[np.zeros(45,dtype=bool)]
                with torch.inference_mode():
                    for tick in range(1500):
                        guard();t=tick*.01
                        raw=torch.tensor([command(s['case'],t) for s in items]*3,device=env.device,dtype=obs.dtype)
                        obs,done,substeps,values=policy_tick(env,policy,c,tick,raw,obs)
                        if not schema:
                            offset=0
                            for k,v in values.items():schema[k]=[offset,offset+v.shape[1]];offset+=v.shape[1]
                        logs.append(torch.cat(list(values.values()),-1).cpu().numpy())
                        rows.extend(substeps.cpu().numpy());poses.append(env.env.sim.data.qpos.cpu().numpy().copy())
                        terms.extend([(env.fail_ticks>100).cpu().numpy().copy()]*5);completed=tick+1
                        assert c.updates==completed
                        if tick%300==0:print(json.dumps(dict(height=height,step=tick,total=1500)),flush=True)
            except Exception as exc:
                error=repr(exc)
            finally:
                if rows:
                    row_array=np.asarray(rows);pose_array=np.asarray(poses);term_array=np.asarray(terms);log_array=np.asarray(logs) if logs else np.empty((0,45,0))
                    origins=env.env.scene.env_origins.cpu().numpy()
                    for b,branch in enumerate(BRANCHES):
                        folder=a.output/height/branch
                        for i,sample in enumerate(items):
                            idx=b*15+i;end=min(len(row_array),round(sample['case']['duration']/.002)+1)
                            q=pose_array[:min(len(pose_array),round(sample['case']['duration']/.01)+1),idx].copy();q[:,:3]-=origins[idx]
                            trace=arrays(row_array[:end,idx],q,terminated=term_array[:end,idx])
                            count=min(completed,round(sample['case']['duration']/.01))
                            trace.update(policy_time=np.arange(count)*.01,feedback_sample_time=(np.arange(count)+1)*.01,integral_commit_time=(np.arange(count)+1)*.01)
                            for key,(lo,hi) in schema.items():trace[key]=log_array[:count,idx,lo:hi]
                            np.savez_compressed(folder/'traces'/(sample['id']+'.npz'),**trace)
                        execution=dict(engine='mjlab',validation_engine='gpu',validation_backend='mujoco_warp_gpu',policy_batches=completed,
                            complete_traces=sum(completed*.01>=s['case']['duration'] for s in items),planned=15,error=error,wall_s=time.time()-started,
                            feedback_phase='Actor/encoder at t; feedback truth and extra encoder-only output from endpoint t+.01; commit at t+.01; next begin uses integral',
                            actor_batches=completed,actor_encoder_batches=completed,feedback_encoder_batches=completed,
                            startup='No extra warmup; one t=0 contact sample; 10 real2ms samples required; first good sample starts gate timer at0; first active after50 intervals; zero initial integral; no autoresets',
                            reason_codes=REASONS,optimizer_updates=0)
                        (folder/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
                executions.append(dict(height=height,steps=completed,error=error))
                (a.output/'initial-consistency.json').write_text(json.dumps(initial_checks,indent=2)+'\n')
                (a.output/'replay-executions.json').write_text(json.dumps(executions,indent=2)+'\n')
            if error:raise RuntimeError(error)
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in policy.state_dict().items())
    (a.output/'frozen-policy.json').write_text(json.dumps(dict(identical=True,actor_encoder_original=True,optimizer_updates=0))+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
