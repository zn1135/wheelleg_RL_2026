#!/usr/bin/env python3
"""CPU dual-ONNX switching/outer-loop/observation checks, separate from single-task training."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sim2sim'))
import mj_sim2sim_chuanliantui_two_policy as cpu
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.switch import conditions
from scripts.agent.split.check_chuanliantui_two_policy import rejected


def command_checks(config):
    c=cpu.CommandController(config,.4,1.)
    for _ in range(50):c.update([0.,0.],0.,.1,0.,0.,True)
    assert c.anchor is None
    for _ in range(49):c.update([.1,0.],0.,.04,0.,0.,True)
    assert c.anchor is None
    c.update([.123,0.],0.,.04,0.,0.,True)
    assert np.array_equal(c.anchor,[.123,0.])
    assert c.update([.133,0.],0.,0.,0.,0.,True)[0]==0.
    c.update([1.123,0.],0.,0.,0.,0.,True)
    assert abs(c.position_velocity+.1)<1e-12 and abs(c.vx)<=.01000001
    for _ in range(20):c.update([1.123,0.],0.,0.,0.,.3,True)
    assert abs(c.vx+.1)<1e-8 and c.anchor is not None
    # At pi the forward projection reverses, while a lateral offset stays diagnostic.
    c.update([1.123,.5],np.pi,0.,0.,0.,True)
    assert c.forward_error<-.99 and abs(c.lateral_error+.5)<1e-8
    c.update([1.,0.],0.,0.,.2,0.,True);assert c.anchor is None
    c.heading_reference=np.pi-.01;c.yaw=0.
    c.update([1.,0.],-np.pi+.01,0.,.2,0.,True)
    assert c.yaw<0 and abs(c.yaw)<=.0100001
    for request, limit in ((2., .4), (-2., -1.)):
        for _ in range(200):c.update([1.,0.],0.,0.,request,0.,True)
        assert abs(c.vx-limit) < 1e-8
    c.update([1.,0.],0.,0.,0.,0.,False)
    assert c.anchor is None and c.heading_reference is None and c.vx==c.yaw==0.


def boundary_checks():
    gz=torch.tensor([-1.,-1.,-1.,-1.,1.],dtype=torch.float64)
    height=torch.full((5,),.22,dtype=torch.float64)
    omega=torch.zeros(5,dtype=torch.float64);base=torch.zeros_like(omega)
    support=torch.tensor([[50.,50.],[65.,65.],[9.,91.],[50.,50.],[50.,50.]],dtype=torch.float64)
    finite=torch.tensor([True,True,True,False,True])
    ready,down,emergency,_=conditions(gz,height,omega,base,support,torch.full((5,),100.),finite)
    assert ready.tolist()==[True,False,False,False,False]
    assert down.tolist()==[False,False,True,True,True]
    assert emergency.tolist()==[False,False,False,True,True]


class CapturePolicy:
    def __init__(self,infos,offset):self.infos=infos;self.offset=offset;self.frames=[]
    def __call__(self,history):
        self.frames.append(history.copy())
        return np.full(6,self.offset+len(self.frames)*.001,dtype=np.float32)


def routing_checks(initial,policies):
    getup=CapturePolicy(policies[0].infos,-.1);walk=CapturePolicy(policies[1].infos,.2)
    env=cpu.TwoPolicyPlayback(initial,getup,walk,height=.28)
    qpos,qvel=env.data.qpos.copy(),env.data.qvel.copy()
    yes=torch.tensor([True]);no=~yes
    def infer_only(self,controller):
        h=self.update_history(self.observation());action=controller(h)
        self.last_action=action.copy()
        return action,np.zeros(6)
    # Mock only physics stepping for deterministic handover/history assertions.
    with patch.object(cpu.StandV2Playback,'step',infer_only):
        with patch.object(env,'switch_conditions',return_value=(yes,no,no,yes)):
            for _ in range(49):env.step()
            assert not env.selector.walk.any() and len(getup.frames)==49
            assert env.command[2]==.22
            previous=env.history.copy();action_before=env.last_action.copy()
            env.step();assert env.selector.walk.all() and len(walk.frames)==1
            assert env.command[2]==.28
            assert np.array_equal(env.history[:-1],previous[1:])
            assert np.array_equal(env.history[-1,-6:],action_before)
        # 外部切到walk后还需满1s复位等待；不再依赖getup recovered锁存。
        env.steps = 99; env.recovered = False
        with patch.object(env,'switch_conditions',return_value=(yes,no,no,yes)):
            env.step(); assert env.command_loop.ready_ticks == 0
            env.steps = 100
            for _ in range(50):env.step()
            assert env.command_loop.ready_ticks == 50
        env.command_loop.anchor=np.array([4.,4.]);env.command_loop.heading_reference=2.
        with patch.object(env,'switch_conditions',return_value=(no,yes,no,yes)):
            for n in range(9):
                env.step()
                assert env.command_loop.ready_ticks == 51+n
                assert env.command_loop.anchor is not None
                assert env.command_loop.heading_reference == 2.
            assert env.selector.walk.all()
            env.step();assert not env.selector.walk.any()
            assert env.command[2]==.22
            assert env.command_loop.anchor is None and env.command_loop.heading_reference is None
            assert not np.count_nonzero(getup.frames[-1][-25+6:-25+8])
        assert np.array_equal(env.data.qpos,qpos) and np.array_equal(env.data.qvel,qvel)
    env.reset();assert env.history is None and not env.selector.walk.any()
    return dict(mock_physics_only=True,enter_50_exit_10_ticks=True,no_handover_physical_reset=True,
        shared_fifo_and_previous_action_preserved=True,outer_loop_reference_cleared=True,walk_native_settle_and_dip_readiness=True,
        physical_reset_clears_history_and_selector=True,role_height_routed=True)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    walk = cpu.OnnxPolicy(args.fixtures/'interface-only-walk.pt',
                          args.fixtures/'interface-only-walk.onnx', 'walk')
    getup_checkpoint = args.fixtures/'interface-only-getup.pt'
    getup_graph = args.fixtures/'interface-only-getup.onnx'
    getup_infos = torch.load(getup_checkpoint, map_location='cpu', weights_only=False)['infos']
    filtered_getup_rejected = bool(getup_infos.get('action_filter_contract'))
    if filtered_getup_rejected:
        # CPU尚未实现起立滤波：保留正式拒载，只用合成策略检查切换/外环/物理有限性。
        rejected(lambda: cpu.OnnxPolicy(getup_checkpoint, getup_graph, 'getup'))
        getup = CapturePolicy(getup_infos, 0.)
    else:
        getup = cpu.OnnxPolicy(getup_checkpoint, getup_graph, 'getup')
    policies = [getup, walk]
    old = torch.load(args.previous_v7, map_location='cpu', weights_only=False)['infos']
    rejected(lambda:cpu.validate(old, command_cfg('walk')))
    rejected(lambda:cpu.OnnxPolicy(args.previous_v7, args.fixtures/'interface-only-walk.onnx', 'walk'))
    boundary_checks();command_checks(policies[1].infos['command_config'])
    parity={}
    # Neither playback construction nor stepping may construct a training environment.
    with patch.object(task_registry,'make_env',side_effect=AssertionError('CPU playback used training physics')):
        for role in ('getup','walk'):
            initial=args.fixtures/f'{role}-initial.npz'
            env=cpu.TwoPolicyPlayback(initial,*policies)
            with np.load(initial) as saved:
                expected=saved['actor'][0]
                # Compare identical physical inputs and the same role-specific command.
                env.command[:]=saved['command'][0]
            actual=env.observation()
            error=float(np.abs(actual-expected).max());assert error<1e-6,(role,error,actual,expected)
            parity[role]=error
        routing=routing_checks(args.fixtures/'getup-initial.npz',policies)
        saved_speed = walk.infos['standup_curriculum']['speed_curriculum']
        forward_max = walk.infos['command_config']['forward_speed_limits'][saved_speed['forward']['stage']]
        backward_max = walk.infos['command_config']['backward_speed_limits'][saved_speed['backward']['stage']]
        for values in ({'height':.21},{'height':.31},{'height':float('nan')},{'vx':.2,'yaw':.2},{'vx':forward_max+.001},{'vx':-backward_max-.001}):
            try:cpu.TwoPolicyPlayback(args.fixtures/'getup-initial.npz',*policies,**values)
            except ValueError:pass
            else:raise AssertionError(f'Out-of-profile CPU command accepted: {values}')
        env=cpu.TwoPolicyPlayback(args.fixtures/'getup-initial.npz',*policies)
        rows=[]
        for _ in range(32):
            action,torque=env.step()
            assert np.isfinite(action).all() and np.isfinite(torque).all()
            assert env.history.shape==(5,25) and np.array_equal(env.history[-1],env.history.reshape(125)[-25:])
            rows.append(env.metrics(action,torque))
    record=dict(verified=True,parameter_updates=0,actual_cpu_steps=32,observation_max_error=parity,
        walk_checkpoint_v8_accepted=True, walk_checkpoint_v7_rejected=True, filtered_getup_rejected=filtered_getup_rejected,
        getup_fixture='synthetic' if filtered_getup_rejected else 'onnx',
        outer_loop_boundaries=True,routing=routing,rows=rows,
        source_sha256=cpu.sha256(Path(__file__)),cpu_source_sha256=cpu.sha256(Path(cpu.__file__)),
        scope='Untrained interface fixtures; real independent CPU stepping plus isolated mock handover checks, not behavior acceptance')
    (args.output/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--previous-v7',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('CPU check cannot train')):
        main(parser.parse_args())
