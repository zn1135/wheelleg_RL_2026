#!/usr/bin/env python3
"""Check shared reference clock, readonly student observation and partial reset."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations as obs
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose
from ct_mjlab.teacher.reference import controller_tensors
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def check(args):
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    assert cfg.observations['actor'].terms['proprioception'].func is obs.actor_observation
    assert cfg.observations['critic'].terms['proprioception'].func is obs.critic_observation
    cfg.auto_reset=False;cfg.rewards={};cfg.terminations={}
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    obs.configure_reference_observations(cfg)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=4,device='cuda:0')
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        state=raw._bc_reference_observer
        o=raw.observation_manager.compute(update_history=True)
        assert o['actor'].shape==(4,41) and o['critic'].shape==(4,84)
        for tick in range(30):
            ref=state.update();before=[value.clone() for value in controller_tensors(ref.teacher.controller)]
            q=raw.sim.data.qpos.clone();v=raw.sim.data.qvel.clone();ctrl=raw.sim.data.ctrl.clone();fifo=action._action_history.clone()
            a=obs.reference_actor_observation(raw);c=obs.reference_critic_observation(raw)
            torch.testing.assert_close(a[:,40:41],c[:,83:84],atol=0,rtol=0)
            for value,snapshot in zip(controller_tensors(ref.teacher.controller),before):torch.testing.assert_close(value,snapshot,atol=0,rtol=0)
            assert torch.equal(q,raw.sim.data.qpos) and torch.equal(v,raw.sim.data.qvel)
            assert torch.equal(ctrl,raw.sim.data.ctrl) and torch.equal(fifo,action._action_history)
            env.step(torch.zeros(4,6,device=raw.device))
        # 人工支持计数只检查499/500边界；实际接触由正式采集检查。
        state.synchronized.zero_();state.hold[:]=torch.tensor([499,500,700,0],device=raw.device)
        state.valid.zero_();state.update()
        assert torch.equal(state.synchronized,torch.tensor([False,True,True,False],device=raw.device))
        before=[value.clone() for value in controller_tensors(state.teacher.controller)]
        state.reset(torch.tensor([1],device=raw.device));state.update()
        keep=torch.tensor([0,2,3],device=raw.device)
        for value,snapshot in zip(controller_tensors(state.teacher.controller),before):
            if value.ndim and value.shape[0]==4:torch.testing.assert_close(value[keep],snapshot[keep],atol=0,rtol=0)
        assert not state.synchronized[1] and state.synchronized[2]
        result=dict(passed=True,optimizer_updates=0,actor_dim=41,critic_dim=84,
            default_ppo_observation_functions_unchanged=True,
            repeated_actor_critic_queries_do_not_advance_reference=True,
            observation_does_not_write_physics_or_pd_fifo=True,partial_reset_does_not_advance_other_observers=True,
            first_supported_anchor_499_500_checked=True,checker_sha256=digest(__file__),
            scope='Shared observer clock and reset on real simulator with synthetic support boundary; integrated teacher data checked separately')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);check(p.parse_args())
