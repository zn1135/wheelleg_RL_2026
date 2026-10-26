#!/usr/bin/env python3
"""Check real QueryAction boundary masks and partial reset without physics or learning."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose
from eval_bc import QueryCfg
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def check(args):
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.actions['pd']=QueryCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'})
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=4,device='cuda:0')
    try:
        raw=env.unwrapped;raw.reset(seed=42);action=raw.action_manager.get_term('pd')
        with np.load(args.run/'trace.npz') as d:
            q=torch.from_numpy(d['qpos'][999,:4].copy()).to(raw.device)
            v=torch.from_numpy(d['qvel'][999,:4].copy()).to(raw.device)
        q[:,:2]=raw.scene.env_origins[:,:2]
        raw.sim.data.qpos.copy_(q);raw.sim.data.qvel.copy_(v);raw.sim.forward();action.reset()
        raw.episode_length_buf[:]=1000;raw.command_manager.get_term('standup').ground_start[:]=True
        # 人工设置499/500子步边界，仅检查采集控制逻辑；真实承重由主采集验收。
        hold=torch.tensor([499,500,700,0],device=raw.device)
        raw._source_probe=dict(live=torch.ones(4,dtype=torch.bool,device=raw.device),hold=hold)
        action.teacher._control_ticks[:]=10000
        c=action.teacher.controller;c.standup.phase[:]=4;c.standup.fault[:]=3
        inputs=torch.zeros(4,6,device=raw.device);inputs[:,2]=.1
        history=action._action_history.clone();head=action._history_head
        q0=raw.sim.data.qpos.clone();v0=raw.sim.data.qvel.clone()
        action.process_actions(inputs)
        expected=torch.tensor([False,True,True,False],device=raw.device)
        assert torch.equal(action.balance_synchronized,expected)
        assert (c.standup.phase[expected]==3).all() and (c.standup.fault[expected]==0).all()
        assert (c.standup.phase[~expected]==4).all()
        assert torch.equal(action.raw_action,inputs) and torch.equal(history,action._action_history) and head==action._history_head
        torch.testing.assert_close(raw.sim.data.qpos.clone(),q0,atol=0,rtol=0)
        torch.testing.assert_close(raw.sim.data.qvel.clone(),v0,atol=0,rtol=0)
        c.pos[2]=.2
        action.process_actions(inputs)
        assert abs(float(c.pos[2]))>.1,'Do not repeatedly relock the parking reference'
        # 局部复位只清该环境的同步历史；下一回合重新满足门槛后才同步。
        action.reset(torch.tensor([1],device=raw.device));raw.episode_length_buf[1]=0;hold[1]=0
        action.process_actions(inputs)
        assert not action.balance_synchronized[1] and action.balance_synchronized[2]
        raw.episode_length_buf[1]=1000;hold[1]=500;action.teacher._control_ticks[1]=10000
        c.standup.phase[1]=4;c.standup.fault[1]=3
        action.process_actions(inputs)
        assert action.balance_synchronized[1] and c.standup.phase[1]==3 and c.standup.fault[1]==0
        result=dict(passed=True,optimizer_updates=0,threshold_499_not_synced_500_synced=True,
            failed_teacher_enters_done_on_ready_mask=True,no_repeated_reference_relock=True,
            partial_reset_isolated=True,student_action_fifo_and_physics_unchanged_by_query=True,
            checker_sha256=digest(__file__),query_source_sha256=digest(Path(__file__).with_name('eval_bc.py')),
            scope='Real action/controller on recorded training states with synthetic support counters; no integrated behavior claim.')
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);check(p.parse_args())
