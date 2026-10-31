#!/usr/bin/env python3
"""V11 physical-path adapters, BPTT/tail oracles and mocked updates; CPU only."""
import argparse
import copy
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from ct_mjlab.supermimic.analytic_reward import differentiable_reward
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, Learner, frozen
from ct_mjlab.supermimic.networks import RunningNorm, WorldModel, Critic, act, make_actor
from ct_mjlab.supermimic.state import (StateAdapter, STATE_DIM, PARAM_DIM, PHYS_DIM,
    SUBSTEPS, SENSORS, NOISE, MEMORY, MASSES, SOLREF, WRENCH, physical_delta, integrate_physical)

V11 = 'chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'


def fixture(batch=3):
    adapter = StateAdapter()
    state = torch.zeros(batch,STATE_DIM)
    state[:,:25] = adapter.kin.qpos0.float()
    state[:,2] = .30
    state[:,3:7] = torch.tensor([1.,0.,0.,0.])
    params = torch.zeros(batch,PARAM_DIM)
    params[:,:6] = state[:,adapter.motor_q]
    params[:,6:12],params[:,12:18],params[:,18:24] = 10.,.5,1.
    params[:,MASSES] = 1.
    params[:,SOLREF] = torch.tensor([.02,1.])
    state[:,60] = state[:,63] = -params[:,MASSES].sum(-1)*9.81/2
    state[:,MEMORY] = torch.tensor([0.,1.,1.,50.,0.,0.,0.,50.,1.,50.])
    state[:,SENSORS] = adapter.raw_sensors(state[:,:PHYS_DIM],params)[:,None].expand(-1,9,-1).flatten(1)
    context = torch.zeros(batch,21)
    context[:,2],context[:,6],context[:,19],context[:,20] = .30,1.,.04,.01
    context[:,7:19:2],context[:,8:19:2] = -math.inf,math.inf
    return adapter,state,params,context


def history_gradient_check():
    torch.manual_seed(17)
    actor = make_actor('cpu'); before = copy.deepcopy(actor.state_dict())
    h = torch.randn(2,125,requires_grad=True)
    direction = torch.randn_like(h); direction[:,100:] = 0.
    with frozen(actor.encoder):
        act(actor,h).sum().backward()
        assert h.grad is not None and torch.isfinite(h.grad).all() and h.grad[:,:100].abs().max()>0
        assert all(p.grad is None for p in actor.encoder.parameters())
        analytical = float((h.grad*direction).sum())
        with torch.no_grad():
            numerical = float((act(actor,h+.01*direction).sum()-act(actor,h-.01*direction).sum())/.02)
    assert math.isclose(analytical,numerical,rel_tol=.02,abs_tol=1e-6)
    assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
    return dict(old_history_gradient=True,analytical=analytical,finite_difference=numerical,parameters_unchanged=True)


def adapter_checks():
    adapter,s,p,c = fixture()
    path = s[:,None,:PHYS_DIM].expand(-1,SUBSTEPS,-1).clone()
    path[...,49:55] = torch.arange(1,7)*.25
    raw = torch.tensor([[.2,-.1,.3,-.2,.1,-.3]]).expand(3,-1)
    result = adapter.advance(s,p,path,torch.zeros(3,3),raw,raw*.5,raw*.25,c,s[:,NOISE])
    ns,frame,reward,failure,features = result
    assert ns.shape==(3,STATE_DIM) and frame.shape==(3,25) and features.shape==(3,31)
    torch.testing.assert_close(features[:,22:28],path[:,-1,49:55],atol=0,rtol=0)
    torch.testing.assert_close(features[:,12:16],path[:,-1,adapter.leg_q],atol=0,rtol=0)
    torch.testing.assert_close(features[:,18:22],path[:,-1,adapter.leg_v],atol=0,rtol=0)
    _,expected = differentiable_reward(features,raw,raw*.5,raw*.25,c)
    torch.testing.assert_close(reward,expected,atol=0,rtol=0)
    changed = path.clone();changed[...,adapter.leg_q[0]] += .2
    torque_terms = differentiable_reward(adapter.reward_features(changed),raw,raw*.5,raw*.25,c)[0]['torques']
    torch.testing.assert_close(torque_terms,differentiable_reward(features,raw,raw*.5,raw*.25,c)[0]['torques'],atol=0,rtol=0)
    # Each sensor group selects its own physical-substep delay and saved noise.
    ring = torch.arange(9*18,dtype=s.dtype).reshape(1,9,18).expand(3,-1,-1).clone()*.001
    delayed=s.clone();delayed[:,SENSORS]=ring.flatten(1);delayed[:,NOISE]=.002
    p[:,26:30]=torch.tensor([0.,2.,4.,8.])
    selected=torch.cat((ring[:,-1,:3],ring[:,-3,3:6],ring[:,-5,6:12],ring[:,-9,12:18]),-1)+.002
    expected_frame=torch.cat((selected[:,:3]*.25,selected[:,3:6],torch.zeros(3,3),selected[:,6:10],
        selected[:,12:18][:,(0,1,4,2,3,5)]*.05,raw),-1).clamp(-100,100)
    torch.testing.assert_close(adapter.actor_frame(delayed,p,torch.zeros(3,3),raw),expected_frame,atol=0,rtol=0)
    differentiable_path=path.clone();differentiable_path[...,2]+=.02
    differentiable_path.requires_grad_()
    path_reward=adapter.advance(s,p,differentiable_path,torch.zeros(3,3),raw,raw*.5,raw*.25,c,s[:,NOISE])[2]
    path_gradient=torch.autograd.grad(path_reward.sum(),differentiable_path)[0]
    assert torch.isfinite(path_gradient).all()
    assert path_gradient[:,-2,2].abs().max()>0 and path_gradient[:,-1,49:55].abs().max()>0
    delta=physical_delta(path[:,:-1],path[:,1:]);restored=integrate_physical(path[:,:-1],delta)
    torch.testing.assert_close(restored,path[:,1:],atol=1e-6,rtol=1e-6)
    # Age/down memory changes hard death; timeout is deliberately not a death signal.
    down=path.clone();down[...,60]=down[...,63]=0.
    young=s.clone();young[:,MEMORY.start]=98.;young[:,MEMORY.start+6]=9.
    armed=young.clone();armed[:,MEMORY.start]=99.
    assert not adapter.advance_memory(young,p,down)[1].any()
    assert adapter.advance_memory(armed,p,down)[1].all()
    timeout=s.clone();timeout[:,MEMORY.start]=4000.
    assert not adapter.advance_memory(timeout,p,path)[1].any()
    # Real world has no independent observation/feature/failure head.
    world=WorldModel().eval();a=torch.randn(3,6,requires_grad=True)*.03
    original=copy.deepcopy(world.state_dict())
    outputs=world.step(s,p,a,torch.zeros(3,3),c,raw,raw*.5,s[:,NOISE],return_path=True)
    direct=world.adapter.advance(s,p,outputs[-1],torch.zeros(3,3),a,raw,raw*.5,c,s[:,NOISE])
    for left,right in zip(outputs[:5],direct):torch.testing.assert_close(left,right,atol=0,rtol=0)
    with frozen(world):
        outputs=world.step(s,p,a,torch.zeros(3,3),c,raw,raw*.5,s[:,NOISE],return_path=True)
        torch.autograd.grad(outputs[2].sum(),a)
        state_input=s.clone();state_input[:,2]+=.02;state_input.requires_grad_()
        value=world.step(state_input,p,a,torch.zeros(3,3),c,raw,raw*.5,s[:,NOISE])[2]
        state_gradient=torch.autograd.grad(value.sum(),state_input)[0]
        assert torch.isfinite(state_gradient).all() and state_gradient[:,2].abs().max()>0
    assert all(torch.equal(v,original[k]) for k,v in world.state_dict().items())
    return dict(shared_path_observation_reward_failure=True,actual_torque_not_PD=True,
        sensor_group_delays_noise_and_motor_order=True,physical_delta_roundtrip=True,
        armed_down_memory_and_timeout_boundary=True,world_step_state_unchanged=True,
        physical_path_height_and_actual_torque_gradients=True,frozen_world_input_gradients=True)


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(125,3);self.mlp=nn.Linear(28,6)
        for p in self.parameters():nn.init.zeros_(p)


class PathOracle(nn.Module):
    def __init__(self,fail_tick=None):
        super().__init__();self.adapter=StateAdapter();self.tick=0;self.fail_tick=fail_tick
        self.rewards=[];self.failures=[]
    def step(self,state,parameters,action,command,context,previous,previous_previous,noise):
        path=state[:,None,:PHYS_DIM].expand(-1,SUBSTEPS,-1).clone()
        if self.fail_tick is not None and self.tick>=self.fail_tick:
            path[...,3:7]=torch.tensor([0.,1.,0.,0.])
        result=self.adapter.advance(state,parameters,path,command,action,previous,previous_previous,context,noise)
        self.rewards.append(result[2]);self.failures.append(result[3]);self.tick+=1
        return result


class ToyCritic(nn.Module):
    inputs = staticmethod(Critic.inputs)
    def __init__(self):
        super().__init__();self.norm=RunningNorm(STATE_DIM+PARAM_DIM+21)
    def forward(self,state,parameters,context):return .05+.01*state[...,MEMORY.start]


def toy_window():
    adapter,s,p,c=fixture();batch,horizon=3,4;lengths=(1,2,4)
    w={name:torch.zeros(batch,horizon,dim) for name,dim in ReplayBuffer.FIELDS.items() if dim}
    w.update(done=torch.zeros(batch,horizon,dtype=torch.bool),failure=torch.zeros(batch,horizon,dtype=torch.bool),
        reward=torch.zeros(batch,horizon),mask=torch.zeros(batch,horizon))
    w['state'][:]=s[:,None];w['parameters'][:]=p[:,None];w['context'][:]=c[:,None]
    w['previous'][:]=.07;w['previous_previous'][:]=.03
    for i,length in enumerate(lengths):w['mask'][i,:length]=1.;w['done'][i,length-1]=True
    return w,lengths


def run_oracle(w,fail_tick):
    learner=Learner.__new__(Learner);learner.actor=ToyActor();learner.world=PathOracle(fail_tick);learner.critic=ToyCritic()
    result=learner.imagination(w,gamma=.9,lam=.8)
    return learner,result


def boundary_checks():
    reports=[]
    for event in (None,0,1):
        w,lengths=toy_window();learner,result=run_oracle(w,event)
        rewards=torch.stack(learner.world.rewards).detach();failure=torch.stack(learner.world.failures)
        expected=torch.zeros_like(result[3]);objectives=[]
        for col,length in enumerate(lengths):
            alive=1.;objective=0.;target=.05+.01*length
            for t in range(length):
                objective+=.9**t*alive*float(rewards[t,col]);alive*=float(~failure[t,col])
            objective+=.9**length*alive*(.05+.01*length);objectives.append(objective)
            for t in reversed(range(length)):
                target=float(rewards[t,col])+.9*float(~failure[t,col])*(.2*(.05+.01*(t+1))+.8*target)
                expected[t,col]=target
        torch.testing.assert_close(result[0],torch.tensor(-sum(objectives)/3),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(result[3][result[4].bool()],expected[result[4].bool()],atol=1e-6,rtol=1e-6)
        assert learner.critic.norm.count==sum(length+1 for length in lengths)
        for mode in ('done','autoreset','padding','derived_flags'):
            changed=copy.deepcopy(w)
            if mode=='done':changed['done']=~changed['done']
            elif mode=='autoreset':changed['next_frame'][w['done'],6:9]=123.
            elif mode=='padding':
                for key in ('state','parameters','next_state','next_frame'):changed[key][~w['mask'].bool()]=456.
            else:changed['context'][:,1:,5:7]=1.
            _,other=run_oracle(changed,event)
            torch.testing.assert_close(other[0],result[0],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(other[3][result[4].bool()],result[3][result[4].bool()],atol=1e-6,rtol=1e-6)
        changed=copy.deepcopy(w);changed['context'][2,2:,3]=1.
        _,other=run_oracle(changed,event);assert not other[4][:,2].any()
        torch.testing.assert_close(other[0],torch.tensor(-sum(objectives[:2])/2),atol=1e-6,rtol=1e-6)
        disturbed=copy.deepcopy(w)
        for tick in range(4):disturbed['parameters'][:,tick,WRENCH]=float(tick)*.1
        _,disturbance_result=run_oracle(disturbed,event)
        for column,length in enumerate(lengths):
            torch.testing.assert_close(disturbance_result[2][:length,column,WRENCH],
                disturbed['parameters'][column,:length,WRENCH],atol=0,rtol=0)
            torch.testing.assert_close(disturbance_result[2][length,column,WRENCH],
                disturbed['parameters'][column,length-1,WRENCH],atol=0,rtol=0)
        reports.append(dict(hard_failure_tick=event,lengths=lengths,tail_bootstrap_and_lambda=True,
            timeout_done_does_not_force_death=True,autoreset_and_padding_ignored=True,
            derived_failure_reward_flags_not_teacher_forced=True,external_tracking_parking_constant_window_gate=True,
            external_wrench_and_critic_parameter_sequence_aligned=True))
    return reports


def synthetic_buffer(changing_flags=False):
    adapter,state,params,context=fixture(20);buffer=ReplayBuffer(24,20,'cpu')
    previous=torch.full((20,6),.02);previous_previous=previous*.5
    history=torch.zeros(20,125)
    for tick in range(16):
        action=torch.full((20,6),.015+.001*tick)
        path=state[:,None,:PHYS_DIM].expand(-1,SUBSTEPS,-1).clone()
        for sub in range(SUBSTEPS):
            path[:,sub,0]+=(sub+1)*.0001
            path[:,sub,adapter.motor_q[0]]+=(sub+1)*.00003
            path[:,sub,25]=.01+.001*tick
        context[:,3]=float(tick%2) if changing_flags else 0.
        next_state,frame,reward,failure,features=adapter.advance(state,params,path,torch.zeros(20,3),action,
            previous,previous_previous,context,state[:,NOISE])
        buffer.add(history=history,privileged=torch.zeros(20,71),action=action,next_frame=frame,
            next_privileged=torch.zeros(20,71),velocity=state[:,25:28]*2,reward=reward,
            done=torch.zeros(20,dtype=torch.bool),failure=failure,features=features,context=context,
            previous=previous,previous_previous=previous_previous,state=state,parameters=params,
            next_state=next_state,path=path.flatten(-2))
        history=torch.cat((history[:,25:],frame),-1);state=next_state;previous_previous,previous=previous,action
    return buffer


def mocked_update_checks():
    reports=[]
    for mode in ('warmup','valid','invalid'):
        learner=Learner('cpu');buffer=synthetic_buffer(mode=='invalid')
        before={name:[p.detach().clone() for p in getattr(learner,name).parameters()] for name in ('actor','world','critic')}
        names=('opt_world','opt_encoder','opt_actor','opt_critic')
        with ExitStack() as stack:
            calls=[stack.enter_context(patch.object(getattr(learner,n),'step')) for n in names]
            metrics=learner.update(buffer,batch=8,window=4,horizon=4,policy=mode!='warmup',encoder=mode!='warmup')
        expected=[4,0,0,0] if mode=='warmup' else [4,1,0,0] if mode=='invalid' else [4,1,1,1]
        assert [p.call_count for p in calls]==expected
        assert all(math.isfinite(value) for value in metrics.values())
        if mode=='valid':assert metrics['actor_grad']>0
        assert all(not getattr(learner,n).state for n in names)
        for name,values in before.items():
            assert all(torch.equal(a,b) for a,b in zip(values,getattr(learner,name).parameters()))
        reports.append(dict(mode=mode,mocked_step_calls=expected,actual_optimizer_steps=0,parameters_unchanged=True))
    # Reward/features/failure are diagnostics, not independent model-head losses.
    learner=Learner('cpu');w=synthetic_buffer().sample(4,3);learner.world.observe(w)
    loss,_=learner.world_loss(w);changed=copy.deepcopy(w)
    changed['reward']+=3.;changed['features']+=2.;changed['failure']=~changed['failure']
    other,_=learner.world_loss(changed);torch.testing.assert_close(loss,other,atol=0,rtol=0)
    return dict(updates=reports,world_loss_only_physical_path=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();torch.set_num_threads(1)
    assert ALGORITHM_CONTRACT==V11 and not torch.cuda.is_initialized()
    report=dict(algorithm_contract=ALGORITHM_CONTRACT,state_dim=STATE_DIM,parameter_dim=PARAM_DIM,
        critic_input_dim=STATE_DIM+PARAM_DIM+21,device='cpu',optimizer_steps=0,real_simulator_steps=0,
        shared_physical_path=adapter_checks(),history_gradient=history_gradient_check(),
        imagined_boundaries=boundary_checks(),mocked_updates=mocked_update_checks(),
        scope='CPU synthetic/interface checks only; no fitting, simulation, convergence or behavior evidence.')
    from ct_mjlab.supermimic import learner,networks,state,kinematics
    report['source_sha256']={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (learner,networks,state,kinematics)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
