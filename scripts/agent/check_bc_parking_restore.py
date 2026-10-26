#!/usr/bin/env python3
"""Restore canonical10s state into a fresh env; unchanged student must reproduce next5 ticks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from ct_mjlab.bc import load_actor
from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose,inverted
from check_bc_parking import ParkingCfg,restore_switch
from check_smp_policy_source import balance_substep
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def check(args):
    frozen=torch.load(args.run/'student-paired/switch-state.pt',map_location='cpu',weights_only=True)
    actor,_=load_actor(args.checkpoint,'cuda:0');count=len(frozen['episode'])
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True);cfg.seed=42;cfg.auto_reset=False;cfg.episode_length_s=31.
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    cfg.terminations['inverted']=TerminationTermCfg(func=inverted);cfg.rewards={}
    cfg.metrics['source_balance']=MetricsTermCfg(func=balance_substep,per_substep=True,reduce='last')
    cfg.actions['pd']=ParkingCfg(**{k:v for k,v in vars(cfg.actions['pd']).items() if k!='class_type'},parking_mode='student')
    cfg.actions['pd'].action_delay_s=None;cfg.actions['pd'].randomize_delays=False;cfg.actions['pd'].action_delay_choices_s=(.002,.004)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=count,device='cuda:0')
    try:
        raw=env.unwrapped;raw.reset(seed=42)
        raw._source_weight=float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe=dict(live=torch.ones(count,dtype=torch.bool,device=raw.device),hold=torch.zeros(count,dtype=torch.long,device=raw.device),
            strict_run=torch.zeros(count,dtype=torch.long,device=raw.device),strict_ever=torch.zeros(count,dtype=torch.bool,device=raw.device))
        obs,_,_,_=env.step(torch.zeros(count,6,device=raw.device))
        action=raw.action_manager.get_term('pd');restore_switch(raw,action,obs,frozen)
        qpos=[];qvel=[];actions=[];observations=[]
        for _ in range(5):
            observations.append(obs['actor'].cpu().numpy().copy());output=actor.mlp(obs['actor']);actions.append(output.cpu().numpy().copy())
            obs,_,_,_=env.step(output);qpos.append(raw.sim.data.qpos.clone().cpu().numpy());qvel.append(raw.sim.data.qvel.clone().cpu().numpy())
        with np.load(args.run/'student-paired/trace.npz') as source:
            differences={k:float(abs(np.asarray(value)-source[k][1000:1005]).max()) for k,value in
                (('qpos',qpos),('qvel',qvel),('student_action',actions),('observation',observations))}
        passed=(differences['qpos']<=1e-5 and differences['qvel']<=1e-3 and differences['student_action']<=1e-3 and differences['observation']<=1e-3)
        result=dict(passed=passed,optimizer_updates=0,frames_replayed=5,num_envs=count,differences=differences,
            snapshot_sha256=digest(args.run/'student-paired/switch-state.pt'),source_sha256=digest(__file__),
            scope='Same frozen10s physics/control/teacher/sensor/RNG state and unchanged student action, fresh env; finite5-tick replay check.')
        (args.run/'restore-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        assert passed,'Canonical restore does not reproduce student continuation within declared tolerances'
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);check(p.parse_args())
