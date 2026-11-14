"""Compare the held-torque PD path with the unchanged historical controller."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
from ct_mjlab.envs.chuanliantui import huananhu_action as module
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, timing_record
from ct_mjlab.fudan_ppo_walk.measurement import arrays

ROOT=Path(__file__).resolve().parents[3]


def action(decimation):
    a=SimpleNamespace(cfg=SimpleNamespace(pd_decimation=decimation,leg_position_scale=.5,
        leg_stiffness=20.,leg_damping=1.,wheel_velocity_scale=10.,max_wheel_velocity=150.,
        wheel_stiffness=.1,wheel_damping=0.,motor_response_time_constant_s=0.),
        _history_head=0,_action_history=torch.zeros(1,2,6),_control_actions=torch.zeros(2,6),
        action_delay_steps=torch.zeros(2,dtype=torch.long),wheel_action_delay_steps=torch.zeros(2,dtype=torch.long),
        _env_ids=torch.arange(2),_motor_q_adr=torch.arange(6),_motor_v_adr=torch.arange(6),
        _leg_action_ids=[0,1,3,4],_wheel_action_ids=[2,5],_reference_pos=torch.zeros(1,6),
        _nominal_pos=torch.zeros(2,6),kp_scale=torch.ones(2,6),kd_scale=torch.ones(2,6),
        motor_strength=torch.ones(2,6),_effort_limits=torch.tensor([40.,40.,3.9,40.,40.,3.9]),
        _torques=torch.zeros(2,6),_motor_response_torque=torch.zeros(2,6),_wheel_speed_shaping=False,
        _spring_q_adr=[6,7],_spring_v_adr=[6,7],_spring_ids=[6,7],
        _spring_force_bias=torch.zeros(2,2),spring_damping=torch.full((2,2),50.),leg_load=None,_pd_tick=0)
    a._env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2,8),qvel=torch.zeros(2,8),ctrl=torch.zeros(2,8))))
    def submit(value,joint_ids):a._env.sim.data.ctrl[:,joint_ids]=value
    a._entity=SimpleNamespace(set_joint_effort_target=submit);a._physical_joint_ids=list(range(6))
    a._ground_hold=lambda:torch.zeros(2,dtype=torch.bool)
    return a


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=subprocess.check_output(['git','show','HEAD:mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_action.py'],cwd=ROOT,text=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='HuananhuPdAction')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_actions')
    scope=dict(vars(module));exec(compile(ast.Module(body=[fn],type_ignores=[]),'historical_pd','exec'),scope)
    old=scope['apply_actions'];new=module.HuananhuPdAction.apply_actions
    torch.manual_seed(411);frames=[]
    for tick in range(30):
        frames.append((torch.randn(2,8)*.1,torch.randn(2,8),torch.randn(2,6)))
    passed=[]
    for decimation in (1,5):
        actual=action(decimation);expected=action(1);held=None;last_spring=None;spring_changes=0
        for tick,(q,v,target) in enumerate(frames):
            for a in (actual,expected):
                a._env.sim.data.qpos.copy_(q);a._env.sim.data.qvel.copy_(v);a._control_actions.copy_(target)
            old(expected);new(actual)
            if tick%decimation==0:held=expected._env.sim.data.ctrl[:,:6].clone()
            assert torch.equal(actual._env.sim.data.ctrl[:,:6],held),(decimation,tick)
            assert torch.equal(actual.torques if hasattr(actual,'torques') else actual._torques,held)
            assert torch.equal(actual._env.sim.data.ctrl[:,6:],expected._env.sim.data.ctrl[:,6:])
            if last_spring is not None and not torch.equal(last_spring,actual._env.sim.data.ctrl[:,6:]):spring_changes+=1
            last_spring=actual._env.sim.data.ctrl[:,6:].clone()
        assert spring_changes>0
        passed.append(dict(pd_decimation=decimation,physics_ticks=len(frames),pd_updates=len(frames)//decimation,
            historical_pd_exact=True,passive_spring_updates_between_pd=True))
    cfg=FudanWalkCfg();cfg.sim.dt=.001;cfg.control.decimation=10;cfg.control.pd_decimation=5
    assert timing_record(cfg)==dict(physics_dt=.001,pd_dt=.005,policy_dt=.01)
    trace=arrays(torch.zeros(11,18),torch.zeros(2,7),physics_dt=.001)
    assert abs(trace['time'][-1]-.01)<1e-12
    report=dict(passed=True,cases=passed,timing=timing_record(cfg),optimizer_updates=0,physics_steps=0,
        scope='Historical PD numerical equivalence, exact torque hold cadence, spring updates and trace clock; not physical acceptance')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
