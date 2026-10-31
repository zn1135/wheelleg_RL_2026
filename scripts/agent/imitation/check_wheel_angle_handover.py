#!/usr/bin/env python3
"""Check first-rise wheel-angle clearing, history isolation and legacy compatibility; no learning."""
import argparse,hashlib,json
from dataclasses import replace
from pathlib import Path
import torch
from mjlab.managers.event_manager import EventTermCfg
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations as observations
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.ppo import SmpCommandCfg,reset_pose
from ct_mjlab import wheel_angles


@torch.inference_mode()
def check(args):
    cfg,train=task_registry.get_cfgs('chuanliantui',play=True)
    assert not cfg.observations['actor'].terms['proprioception'].params['cfg'].wheel_angle_handover_reset
    cfg.auto_reset=False;cfg.rewards={};cfg.terminations={}
    cfg.commands['standup']=SmpCommandCfg();cfg.events['reset_pose']=EventTermCfg(func=reset_pose,mode='reset')
    observations.configure_wheel_angle_observations(cfg,handover_reset=True)
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=20,device=args.device)
    try:
        raw=env.unwrapped;raw.reset(seed=42);state=observations.get_state(raw);command=state.command
        command.ground_start[:]=True;command.recovered[:]=False
        ids=torch.arange(20,device=raw.device);chosen=ids[::2];other=ids[1::2]
        adr=state._motor_q_adr[4:6];origin=state.wheel_angle_origin.clone()
        increment=torch.tensor([2.,-3.],device=raw.device)
        raw.sim.data.qpos[:,adr]=origin+increment;raw.sim.forward();state.sample_substep()
        before_actor=state.actor();before_critic=state.critic()
        torch.testing.assert_close(before_actor[:,14:16],increment.expand(20,2),atol=2e-6,rtol=2e-6)
        q=raw.sim.data.qpos.clone();v=raw.sim.data.qvel.clone();fifo=state.action._action_history.clone()
        history=state.delay.history.clone();command.recovered[chosen]=True
        actor=state.actor();critic=state.critic()
        assert actor.shape==(20,39) and critic.shape==(20,82)
        assert not actor[chosen,14:16].any() and not critic[chosen,14:16].any()
        assert not state.delay.history[:,chosen,10:12].any()
        for after,before in ((actor,before_actor),(critic,before_critic)):
            keep=torch.ones(after.shape[-1],dtype=torch.bool,device=raw.device);keep[14:16]=False
            # 恢复信号本身会改模式标志；这里只比较轮角变更之外的传感器和动作字段。
            torch.testing.assert_close(after[:,:28][:,keep[:28]],before[:,:28][:,keep[:28]],atol=0,rtol=0)
        torch.testing.assert_close(state.delay.history[:,other],history[:,other],atol=0,rtol=0)
        keep=torch.ones(18,dtype=torch.bool,device=raw.device);keep[10:12]=False
        torch.testing.assert_close(state.delay.history[:,:,keep],history[:,:,keep],atol=0,rtol=0)
        for now,old in ((raw.sim.data.qpos,q),(raw.sim.data.qvel,v),(state.action._action_history,fifo)):
            torch.testing.assert_close(now.clone(),old,atol=0,rtol=0)
        # 清零后继续累计；恢复信号反复变化或速度命令改变都不能再次锁定。
        locked=state.wheel_angle_origin.clone();raw.sim.data.qpos[:,adr]+=increment
        command.recovered[:]=False;state.sample_substep();command.recovered[chosen]=True
        command._command[:,0]=.3;state.sample_substep()
        torch.testing.assert_close(state.actor()[chosen,14:16],increment.expand(len(chosen),2),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(state.wheel_angle_origin,locked,atol=0,rtol=0)
        # 让critic先读取也只能锁一次；延迟历史只清两轮位置。
        command.recovered[other]=True;state.cfg=replace(state.cfg,noise=True,delay=True)
        state.delay.lags[:,2]=2;state.noise[:]=0.;state.noise[:,10:12]=.01
        critic=state.critic();assert not critic[other,14:16].any()
        torch.testing.assert_close(state.actor()[other,14:16],torch.full((len(other),2),.01,device=raw.device),atol=0,rtol=0)
        stable_origin=state.wheel_angle_origin.clone();stable_history=state.delay.history.clone()
        state.actor();state.critic();state.actor()
        torch.testing.assert_close(state.wheel_angle_origin,stable_origin,atol=0,rtol=0)
        torch.testing.assert_close(state.delay.history,stable_history,atol=0,rtol=0)
        state.cfg=replace(state.cfg,noise=False,delay=False);state.delay.lags[:]=0
        keep_origin=state.wheel_angle_origin[1:].clone();keep_history=state.delay.history[:,1:].clone()
        command.recovered[0]=False;observations.reset_observations(raw,ids[:1],cfg=state.cfg)
        assert not state.wheel_angle_handover_latched[0] and not state.actor()[0,14:16].any()
        torch.testing.assert_close(state.wheel_angle_origin[1:],keep_origin,atol=0,rtol=0)
        torch.testing.assert_close(state.delay.history[:,1:],keep_history,atol=0,rtol=0)
        # 旧v1明确不启用清零，恢复信号不会改变原点。
        state.cfg=replace(state.cfg,wheel_angle_handover_reset=False)
        command.recovered[:]=True;old_origin=state.wheel_angle_origin.clone()
        state.actor();state.sample_substep();state.critic()
        torch.testing.assert_close(state.wheel_angle_origin,old_origin,atol=0,rtol=0)
        state.cfg=replace(state.cfg,wheel_angle_handover_reset=True)
        raw.reset(seed=43);assert not state.wheel_angle_handover_latched.any()
        obs=raw.observation_manager.compute(update_history=True)
        for tick in range(50):
            obs,_,_,_=env.step(torch.zeros(20,6,device=raw.device))
            assert all(torch.isfinite(value).all() for value in obs.values())
            if tick<20:assert not state.action.torques.any()
        result=dict(passed=True,optimizer_updates=0,num_envs=20,finite_physics_s=.5,
            observation_contract=wheel_angles.HANDOVER_OBSERVATION_CONTRACT,actor_dim=39,critic_dim=82,
            phase_signal_in_arithmetic_checks='explicitly driven recovered flags; actual teacher transition checked separately',
            first_recovery_zero=True,continues_accumulating=True,no_repeated_reanchor=True,
            actor_or_critic_first_consistent=True,only_wheel_position_history_cleared=True,
            delay_noise_and_partial_reset_checked=True,physics_and_action_fifo_unchanged_at_reanchor=True,
            legacy_v1_recovery_does_not_reanchor=True,default_ppo_flag_false=True,
            xml_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),
            scope='Observation arithmetic and finite simulation only; no policy training or parking success claim')
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');check(p.parse_args())
