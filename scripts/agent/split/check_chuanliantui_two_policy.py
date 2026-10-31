#!/usr/bin/env python3
"""Formal finite/contract/routing check for separate 25/125 get-up and walk tasks; no learning."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import mdp, observations, rewards
from ct_mjlab.envs.chuanliantui_split.switch import PolicySwitch
from ct_mjlab.envs.chuanliantui_split.export import export, load_policy
from ct_mjlab.envs.chuanliantui_split.runner import validate, contracts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensors_equal(a[k],b[k]) for k in a)
    if isinstance(a, (tuple,list)):
        return len(a)==len(b) and all(tensors_equal(x,y) for x,y in zip(a,b))
    return a == b


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Incompatible checkpoint was accepted')


def switch_check():
    s=PolicySwitch(2,'cpu');yes=torch.ones(2,dtype=torch.bool);no=~yes
    for _ in range(49):s.update(yes,no,no)
    assert not s.walk.any()
    s.update(yes,no,no);assert s.walk.all()
    for _ in range(9):s.update(no,yes,no)
    assert s.walk.all()
    s.update(no,yes,no);assert not s.walk.any()
    for _ in range(49):s.update(yes,no,no)
    s.update(no,no,no);assert not s.ready_ticks.any()
    for _ in range(50):s.update(yes,no,no)
    s.update(no,no,yes);assert not s.walk.any()
    s.ready_ticks[:] = 17;s.reset(torch.tensor([0]));assert s.ready_ticks.tolist()==[0,17]
    s.update(yes,no,no,torch.tensor([True,False]));assert s.ready_ticks.tolist()==[1,17]


def seed_optimizer_state(optimizer):
    # Construct resume fixtures without calling optimizer.step or changing a policy.
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state[p] = dict(step=torch.tensor(7.), exp_avg=torch.full_like(p,.003),
                                      exp_avg_sq=torch.full_like(p,.004))


def getup_regression(args):
    if args.getup_oracle is None:return None
    name='ct_mjlab.envs.chuanliantui_split._getup_oracle'
    spec=importlib.util.spec_from_file_location(name,args.getup_oracle)
    oracle=importlib.util.module_from_spec(spec);__import__('sys').modules[name]=oracle
    spec.loader.exec_module(oracle)
    cfg,train=task_registry.get_cfgs('chuanliantui_getup',play=True)
    env,_=task_registry.make_env('chuanliantui_getup',env_cfg=cfg,train_cfg=train,
        num_envs=4,seed=43,device=args.device)
    try:
        raw=env.unwrapped;captures=[]
        validate(torch.load(args.getup_checkpoint,map_location='cpu',weights_only=False)['infos'],
                 cfg.commands['standup'])
        # Hold physical inputs fixed to isolate command-path and RNG equivalence.
        for cls in (oracle.SplitCommand,mdp.SplitCommand):
            torch.manual_seed(43)
            state=cls(copy.deepcopy(cfg.commands['standup']),raw)
            ids=torch.arange(4,device=args.device);state.reset(ids)
            frames=[]
            for step in range(128):
                raw.common_step_counter=step
                if step==64:state.reset(ids[:1])
                state._update_command(None)
                frames.append(state.command.clone())
            captures.append(dict(command_frames=frames,cpu_rng=torch.get_rng_state(),
                device_rng=torch.cuda.get_rng_state(args.device) if str(args.device).startswith('cuda') else None))
        assert tensors_equal(*captures)
    finally:env.close()
    return dict(getup_commands_and_rng_bitwise_equal=True,command_ticks=128,num_envs=4,
        fixed_physical_inputs=True,running_getup_checkpoint_contract_accepted=True,
        oracle_sha256=digest(args.getup_oracle))


@torch.inference_mode()
def getup_termination_check(raw):
    # 只执行Isaac源码中的张量终止函数，不导入旧仿真运行环境。
    source_path=Path(__file__).resolve().parents[3]/'wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup.py'
    tree=ast.parse(source_path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ChuanliantuiStandup')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_termination')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source_path),'exec'),namespace)
    s=raw.command_manager.get_term('standup');n=raw.num_envs;device=raw.device
    ids=torch.arange(n,device=device)
    class FixtureScene(dict):
        pass
    position=raw.scene.env_origins.clone();position[:,2]+=.15
    gravity=torch.zeros((n,3),device=device)
    forces=torch.zeros((n,2,3),device=device)
    scene=FixtureScene(robot=SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=position,projected_gravity_b=gravity)),
        wheel_contact=SimpleNamespace(data=SimpleNamespace(force=forces)))
    scene.env_origins=raw.scene.env_origins
    fake=SimpleNamespace(scene=scene,command_manager=raw.command_manager,sim=raw.sim,
        device=device,step_dt=raw.step_dt,max_episode_length=raw.max_episode_length,
        episode_length_buf=torch.zeros(n,dtype=torch.long,device=device),
        common_step_counter=raw.common_step_counter+20000)
    zero=torch.zeros(n,device=device);no=zero.bool();rows=[]
    # 每组首次失败拍：从未触地不终止；连续19拍不终止，第20拍终止。
    cases=[('never_landed',None),('first_contact_inverted',1),('airborne_before_stood',21),
        ('single_wheel_resets_clock',None),('gravity_zero',None),('gravity_negative',None),
        ('late_first_contact_inverted',6),('airborne_after_stood',21),('support_exactly_1N',21)]
    with patch.object(mdp.stand,'normal_wheel_support',return_value=no),patch.object(
            mdp.stand,'base_contact_force',return_value=zero):
        for name,expected_first in cases:
            s.completed.zero_();s.reset(ids)
            source=SimpleNamespace(has_landed=no.clone(),has_stood=no.clone(),standing_time=zero.clone(),
                fail_buf=zero.clone(),wheels_airborne_steps=torch.zeros_like(s.wheels_airborne_steps),
                episode_length_buf=fake.episode_length_buf.clone(),dt=raw.step_dt,max_episode_length=4000,
                cfg=SimpleNamespace(standup=SimpleNamespace(success_base_contact_force_threshold=.1,
                    success_projected_gravity_z=-.9,success_requires_base_contact_free=True,
                    success_duration_s=.5,wheels_airborne_timeout_s=.2),env=SimpleNamespace(fail_to_terminal_time_s=1.)),
                projected_gravity=gravity,base_height=torch.full_like(zero,.15))
            source._has_wheel_contact=lambda:(-forces[:,:,2]>1.).any(-1)
            source._both_wheels_airborne=lambda:~source._has_wheel_contact()
            source._base_link_has_contact=lambda threshold:no
            source._current_standup_success_height=lambda:s.success_height
            if name=='airborne_after_stood':s.has_stood.fill_(True);source.has_stood.fill_(True)
            first=None
            for tick in range(1,46):
                forces.zero_();gravity.zero_();gravity[:,2]=-1.
                if name in ('first_contact_inverted','gravity_zero','gravity_negative') or tick==1:
                    forces[:,0,2]=-50.
                if name=='single_wheel_resets_clock' and tick in (21,41):forces[:,1,2]=-50.
                if name=='support_exactly_1N' and tick>1:forces[:,0,2]=-1.
                if name in ('never_landed','late_first_contact_inverted'):
                    forces.zero_()
                    if name=='late_first_contact_inverted' and tick>=6:forces[:,0,2]=-50.
                if name in ('never_landed','first_contact_inverted','late_first_contact_inverted'):gravity[:,2]=.01
                if name=='gravity_zero':gravity[:,2]=0.
                if name=='gravity_negative':gravity[:,2]=-.0001
                fake.common_step_counter+=1;namespace['check_termination'](source)
                actual=mdp.failure(fake)
                assert torch.equal(actual,source.reset_buf),(name,tick)
                assert torch.equal(s.wheels_airborne_steps,source.wheels_airborne_steps),(name,tick)
                if actual.any() and first is None:first=tick
                count=s.wheels_airborne_steps.clone()
                assert torch.equal(mdp.failure(fake),actual) and torch.equal(count,s.wheels_airborne_steps)
            assert first==expected_first,(name,first,expected_first)
            rows.append(dict(case=name,first_failure_tick=first,isaac_equal=True))
    s.completed.zero_();s.reset(ids);s._update_command(None)
    return dict(synthetic_physics_only=True,cases=rows,same_tick_cache=True,
        isaac_source_sha256=digest(source_path),parameter_updates=0)


@torch.inference_mode()
def getup_success_height_check(raw):
    s=raw.command_manager.get_term('standup')
    assert (s.cfg.pre_unlock_success_height,s.cfg.post_unlock_success_height)==(.28,.20)
    class FixtureScene(dict):
        pass
    n=raw.num_envs
    position=raw.scene.env_origins.clone()
    gravity=torch.tensor([0.,0.,-1.],device=raw.device).repeat(n,1)
    forces=torch.zeros((n,2,3),device=raw.device);forces[:,:,2]=-50.
    scene=FixtureScene(robot=SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=position,projected_gravity_b=gravity)),
        wheel_contact=SimpleNamespace(data=SimpleNamespace(force=forces)))
    scene.env_origins=raw.scene.env_origins
    fake=SimpleNamespace(scene=scene,command_manager=raw.command_manager,sim=raw.sim,
        device=raw.device,step_dt=raw.step_dt,max_episode_length=raw.max_episode_length,
        episode_length_buf=raw.episode_length_buf,common_step_counter=raw.common_step_counter+10000)
    yes=torch.ones(n,dtype=torch.bool,device=raw.device);zero=torch.zeros(n,device=raw.device)
    rows=[]
    with patch.object(mdp.stand,'normal_wheel_support',return_value=yes),patch.object(
            mdp.stand,'base_contact_force',return_value=zero):
        for unlocked,threshold in ((False,.28),(True,.20)):
            for height in sorted({.20,threshold-.001,threshold,threshold+.001}):
                s.unlocked=unlocked;s.has_stood.zero_();s.has_landed.fill_(True);s.standing_time.zero_()
                position[:,2]=scene.env_origins[:,2]+height
                for tick in range(50):
                    fake.common_step_counter+=1;mdp.failure(fake)
                    if tick<49:assert not s.has_stood.any()
                expected=height>=threshold
                assert bool(s.has_stood.all())==expected
                assert not s.has_stood.any() if not expected else (s.standing_time>=.5).all()
                rows.append(dict(unlocked=unlocked,height_m=height,success_height_m=threshold,stood=expected))
    s.unlocked=False;s.completed.zero_()
    s.reset(torch.arange(n,device=raw.device));s._update_command(None)
    return dict(synthetic_physics_only=True,continuous_ticks=50,cases=rows,parameter_updates=0)


@torch.inference_mode()
def getup_height_check(raw, runner, args):
    s=raw.command_manager.get_term('standup')
    ids=torch.arange(raw.num_envs,device=raw.device)
    assert not s.unlocked and s.target_height==.30
    assert (s.command[:,2]==.30).all()
    assert observations.get_state(raw).actor()[0,8]==s.command[0,2]*5.
    assert observations.get_state(raw).critic()[0,3]==s.command[0,2]*5.
    assert s.cfg.unlock_window_episodes==4096 and s.cfg.unlock_recovered_rate==.30
    success_heights=getup_success_height_check(raw)
    terminations=getup_termination_check(raw)
    obs=raw.get_observations();policy=runner.get_inference_policy(device=args.device)
    for _ in range(64):
        obs,reward,_,_=runner.env.step(policy(obs))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        assert not s.unlocked and (s.command[:,2]==.30).all()
    before=raw.sim.data.qpos.clone()
    # 未解锁的计数和模型必须能完整恢复，不能在复位或load时强制跳到0.22m。
    s.completed_episodes,s.recovered_episodes=37,11
    locked=args.output/'interface-only-getup-locked.pt'
    runner.save(str(locked))
    s.unlocked=True;s.completed_episodes=s.recovered_episodes=0
    runner.load(str(locked),load_cfg={'actor':True},map_location=args.device)
    assert not s.unlocked and (s.completed_episodes,s.recovered_episodes)==(37,11)
    assert (s.command[:,2]==.30).all()
    s.reset(ids[:1]);s._update_command(None)
    assert not s.unlocked and (s.command[:,2]==.30).all()
    saved=torch.load(locked,map_location='cpu',weights_only=False)
    bad=copy.deepcopy(saved['infos']);bad['command_config']['unlock_recovered_rate']=.31
    rejected(lambda:validate(bad,s.cfg))
    bad=copy.deepcopy(saved['infos']);bad['command_config']['unlock_window_episodes']=2048
    rejected(lambda:validate(bad,s.cfg))
    bad=copy.deepcopy(saved['infos']);bad['command_config']['pre_unlock_success_height']=.20
    rejected(lambda:validate(bad,s.cfg))
    bad=copy.deepcopy(saved['infos']);bad.pop('termination_contract')
    rejected(lambda:validate(bad,s.cfg))
    old=copy.deepcopy(saved['infos']);old['experiment_profile_contract']='chuanliantui-getup-physical25-v1'
    rejected(lambda:validate(old,s.cfg))
    # 4095回合不切换；4096回合的1228/1229个成功分别低于/达到30%。
    s.completed.zero_();s.has_stood.zero_()
    s.completed_episodes,s.recovered_episodes=4094,1228
    s.completed[0]=True;s.update_curriculum(ids[:1])
    assert not s.unlocked and s.completed_episodes==4095
    s.update_curriculum(ids[:1])
    assert not s.unlocked and s.completed_episodes==0
    s.completed_episodes,s.recovered_episodes=4095,1228
    s.has_stood[0]=True;s.update_curriculum(ids[:1]);s._update_command(None)
    assert s.unlocked and (s.command[:,2]==.22).all()
    assert abs(s.last_recovered_rate-1229/4096)<1e-12
    s.reset(ids);s._update_command(None)
    assert s.unlocked and (s.command[:,2]==.22).all()
    assert torch.equal(before,raw.sim.data.qpos)
    # 正式CPU入口拒绝未解锁模型，避免固定0.22m命令与策略阶段不匹配。
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sim2sim'))
    from mj_sim2sim_chuanliantui_two_policy import OnnxPolicy
    rejected(lambda:OnnxPolicy(locked,args.output/'not-created.onnx','getup'))
    return dict(initial_height_m=.30,unlocked_height_m=.22,window_episodes=4096,
        recovered_threshold=.30,threshold_boundaries=True,partial_reset_preserves_stage=True,
        locked_checkpoint_resume=True,changed_curriculum_and_v1_rejected=True,
        locked_cpu_playback_rejected=True,success_height_checks=success_heights,
        termination_checks=terminations,old_termination_contract_rejected=True,
        physical_steps=64,parameter_updates=0)


@torch.inference_mode()
def walking_reset_check(raw,args):
    import mujoco
    import numpy as np
    from PIL import Image
    from scripts.agent.stand.check_chuanliantui_stand import minimum_collision_z
    from scripts.agent.stand.render_chuanliantui_stand import renderer_for
    from ct_mjlab.envs.chuanliantui.model import load_model,closed_chain_residual
    s=raw.command_manager.get_term('standup');robot=raw.scene['robot']
    model=load_model();data=mujoco.MjData(model);rows=[]
    assert abs(mdp.resets.standing_pose(.30)['height']-.30)<1e-9
    nominal=raw.action_manager.get_term('pd')._nominal_pos.clone()
    for i in range(raw.num_envs):
        data.qpos[:7]=raw.sim.data.qpos[i,robot.indexing.free_joint_q_adr].cpu().numpy()
        data.qpos[:3]-=raw.scene.env_origins[i].cpu().numpy()
        for local,name in enumerate(robot.joint_names):
            data.qpos[model.joint(name).qposadr[0]]=float(raw.sim.data.qpos[i,robot.indexing.joint_q_adr[local]])
        mujoco.mj_forward(model,data)
        residual=closed_chain_residual(model,data);clearance=minimum_collision_z(model,data)
        assert residual<2e-7 and abs(clearance-.002)<2e-7
        assert .299<data.qpos[2]<.32 and not s.ground_start[i]
        assert not raw.sim.data.qvel[i].any()
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                q=data.qpos[model.jnt_qposadr[j]]
                assert model.jnt_range[j,0]-1e-7<=q<=model.jnt_range[j,1]+1e-7
        rows.append(dict(env=i,height_m=float(data.qpos[2]),clearance_m=clearance,closed_chain_residual_m=residual))
    # 保存20个真实初态的静态预览，不用随机策略行为冒充初态检查。
    renderer,frame=renderer_for(raw.sim.mj_model,400,300)
    try:
        sheet=Image.new('RGB',(2000,300*((raw.num_envs+4)//5)))
        for i,row in enumerate(rows):
            sheet.paste(frame(raw.sim.data.qpos[i].cpu().numpy(),f'walk env{i} h={row["height_m"]:.3f}m nominal=0.30m'),((i%5)*400,(i//5)*300))
        sheet.save(args.output/'walk-reset-precheck.png')
    finally:renderer.close()
    before=raw.sim.data.qpos.clone();heights=s.episode_height.clone()
    raw.reset(env_ids=torch.tensor([0],device=raw.device))
    assert torch.equal(raw.sim.data.qpos[1:],before[1:])
    assert torch.equal(s.episode_height[1:],heights[1:])
    assert torch.equal(raw.action_manager.get_term('pd')._nominal_pos,nominal)
    return dict(nominal_height_m=.30,rows=rows,partial_reset_preserves_other_envs=True,
        pd_nominal_unchanged=True,clearance_m=.002,parameter_updates=0)


@torch.inference_mode()
def walking_height_check(raw):
    s=raw.command_manager.get_term('standup')
    heights=s.episode_height.clone()
    assert ((heights>=.22)&(heights<=.30)).all() and heights.unique().numel()>1
    assert torch.equal(s.command[:,2],heights)
    ids=torch.arange(raw.num_envs,device=raw.device)
    kinds=set()
    for _ in range(64):
        s._sample_yaw_speed_target(ids)
        assert not ((s.tracking_velocity!=0.)&(s.tracking_yaw_rate!=0.)).any()
        kinds.update(s.tracking_transition.tolist())
        assert torch.equal(s.episode_height,heights)
    assert kinds=={0,1,2}
    # Partial command resets redraw only those environments; physical state is untouched.
    selected=ids[:1];physical=raw.sim.data.qpos.clone()
    s.reset(selected)
    assert torch.equal(s.episode_height[1:],heights[1:]) and s.episode_height[0]!=heights[0]
    assert torch.equal(physical,raw.sim.data.qpos)
    for value in (.22,.30):
        s.episode_height[selected]=value;s._update_command(selected)
        assert s.command[0,2]==value
        assert observations.get_state(raw).actor()[0,8]==s.command[0,2]*5.
        assert observations.get_state(raw).critic()[0,3]==s.command[0,2]*5.
    frozen=s.episode_height.clone()
    for _ in range(4):s._update_command(selected)
    assert torch.equal(frozen,s.episode_height) and torch.equal(s.sampled[:,2],frozen)
    # Height reward must read the per-environment command, rather than scalar 0.22.
    from ct_mjlab.envs.chuanliantui_split.rewards import WalkRewardState as RewardState
    s.reward_active.fill_(True);s.numerical_failure.zero_();s.standing_time.fill_(.5)
    s.height_above_ground.fill_(.30)
    rewards=RewardState(raw)
    s._command[:,2]=.30;at_target=rewards.compute(raw)['base_height'].clone()
    rewards.step=-1;s._command[:,2]=.22;away=rewards.compute(raw)['base_height'].clone()
    assert (at_target>away).all()
    s.reset(ids);s._update_command(None)
    return dict(height_range=[.22,.30],sampling='uniform_each_episode',
        partial_reset=True,constant_between_resets=True,actor_critic_height_command=True,
        height_reward_reads_command=True,motion_kinds=sorted(kinds),curved_requests=0)


@torch.inference_mode()
def walking_speed_check(raw):
    from scripts.agent.split.check_walk_direction_curriculum import sampling_checks, accounting_checks
    s=raw.command_manager.get_term('standup')
    saved=copy.deepcopy(s.direction_curriculum)
    rows=sampling_checks(raw)
    promotion=accounting_checks(raw)
    s.direction_curriculum=saved
    s.reset(torch.arange(raw.num_envs,device=raw.device))
    return dict(stages=rows,promotion_checks=promotion,minimum_applies_to_straight_target_only=True)


@torch.inference_mode()
def walking_reward_check(raw):
    s=raw.command_manager.get_term('standup')
    removed={'recovered','base_link_airborne'}
    assert len(raw.cfg.rewards)==18 and set(raw.cfg.rewards)==set(rewards.WALK_WEIGHTS)
    assert not removed.intersection(raw.cfg.rewards)
    assert all(term.func is rewards.reward for term in raw.cfg.rewards.values())
    assert raw.cfg.rewards['orientation'].weight==-100.
    assert raw.cfg.rewards['tracking_ang_vel'].weight==1.
    state=rewards.WalkRewardState(raw)
    expected_legs=raw.scene['robot'].find_joints(('lf0','lf00','rf0','rf00'),preserve_order=True)[0]
    expected_knees=raw.scene['robot'].find_joints(('lf1','rf1'),preserve_order=True)[0]
    assert list(state.legs)==list(expected_legs) and list(state.knees)==list(expected_knees)
    data=raw.scene['robot'].data
    joint_pos=data.joint_pos.clone()
    joint_pos[:,state.legs[0]]=state.soft_limits[:,0,1]+.02
    joint_pos[:,state.knees]=state.knee_limits[...,0]+.01
    n=raw.num_envs
    state.robot=SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([.12,.03,.1],device=raw.device).repeat(n,1),
        root_link_ang_vel_b=torch.tensor([.1,.2,.3],device=raw.device).repeat(n,1),
        projected_gravity_b=torch.tensor([.1,0.,-.99],device=raw.device).repeat(n,1),
        body_link_pos_w=data.body_link_pos_w.clone(),root_link_quat_w=data.root_link_quat_w.clone(),
        joint_pos=joint_pos,joint_vel=torch.full_like(data.joint_vel,.2)))
    action=torch.full_like(raw.action_manager.action,.2)
    fake=SimpleNamespace(command_manager=raw.command_manager,scene=raw.scene,common_step_counter=0,
        action_manager=SimpleNamespace(action=action,prev_action=action*.5,prev_prev_action=-action*.5),
        sim=SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torch.full_like(raw.sim.data.qfrc_actuator,.2))))
    names=('has_landed','has_stood','standing_time','reward_active','numerical_failure','tracking_active','position_locked')
    previous={name:getattr(s,name).clone() for name in names}
    unlocked=s.unlocked
    try:
        s.reward_active.fill_(True);s.numerical_failure.zero_();s.tracking_active.zero_()
        with patch.object(rewards,'base_contact_force',return_value=torch.full((n,),.5,device=raw.device)):
            s.has_landed.zero_();s.has_stood.zero_();s.standing_time.zero_();s.unlocked=False
            first={key:value.clone() for key,value in state.compute(fake).items()}
            for name in ('tracking_lin_vel','tracking_ang_vel','base_height','dof_pos_limits',
                         'knee_limit_margin','base_link_contact','lin_vel_z','ang_vel_xy','dof_vel',
                         'torques','action_rate','action_smooth'):
                assert (first[name]>0.).all(),name
            assert not first['tracking_lin_vel_tight'].any()
            s.has_landed.fill_(True);s.has_stood.fill_(True);s.standing_time.fill_(.5);s.unlocked=True
            state.step=-1
            assert tensors_equal(first,state.compute(fake))
            # Driven lower-leg motors must be counted; passive-knee velocity must not be.
            saved_q=state.robot.data.joint_pos.clone();saved_dq=state.robot.data.joint_vel.clone()
            state.robot.data.joint_vel.zero_()
            state.robot.data.joint_vel[:,state.legs[1::2]]=2.
            state.step=-1;motor_cost=state.compute(fake)['dof_vel'].clone()
            assert torch.allclose(motor_cost,torch.full_like(motor_cost,8.))
            state.robot.data.joint_vel[:,state.knees]=3.
            state.step=-1;assert torch.equal(motor_cost,state.compute(fake)['dof_vel'])
            assert torch.isneginf(state.soft_limits[:,1::2,0]).all()
            assert torch.isposinf(state.soft_limits[:,1::2,1]).all()
            state.robot.data.joint_pos[:,state.legs]=0.
            state.robot.data.joint_pos[:,state.legs[0]]=state.soft_limits[:,0,1]+.02
            state.step=-1;limit_cost=state.compute(fake)['dof_pos_limits'].clone()
            assert torch.allclose(limit_cost,torch.full_like(limit_cost,.02),atol=1e-5)
            state.robot.data.joint_pos[:,state.legs[1::2]]=1000.
            state.step=-1;assert torch.equal(limit_cost,state.compute(fake)['dof_pos_limits'])
            state.robot.data.joint_pos[:,state.knees]=state.knee_limits[...,1]+.01
            state.step=-1;separate=state.compute(fake)
            assert torch.equal(limit_cost,separate['dof_pos_limits'])
            assert (separate['knee_limit_margin']>0.).all()
            state.robot.data.joint_pos.copy_(saved_q);state.robot.data.joint_vel.copy_(saved_dq)
            saved_gravity=state.robot.data.projected_gravity_b.clone()
            angles=torch.tensor([0.,5.,10.,90.,180.],device=raw.device)[torch.arange(n,device=raw.device)%5]
            theta=torch.deg2rad(angles)
            state.robot.data.projected_gravity_b[:,0]=theta.sin()
            state.robot.data.projected_gravity_b[:,1]=0.
            state.robot.data.projected_gravity_b[:,2]=-theta.cos()
            state.step=-1
            weighted_orientation=state.compute(fake)['orientation']*raw.cfg.rewards['orientation'].weight
            # v6：0～5°免罚，超出后按1−cos增长，加权后到−5饱和。
            free=math.cos(rewards.WALK_ORIENTATION_FREE_RAD)
            expected_orientation=(-100.*(free-theta.cos())).clamp(-5.,0.)
            assert torch.allclose(weighted_orientation,expected_orientation,atol=1e-5)
            assert (weighted_orientation>=-5.).all() and (weighted_orientation<=0.).all()
            state.robot.data.projected_gravity_b.copy_(saved_gravity)
            s.tracking_active.fill_(True);state.step=-1
            active=state.compute(fake)
            s.position_locked.zero_();state.step=-1
            active=state.compute(fake)
            assert torch.allclose(active['tracking_lin_vel'],5.*first['tracking_lin_vel'])
            s.position_locked.fill_(True);state.step=-1
            assert torch.allclose(state.compute(fake)['tracking_lin_vel'],first['tracking_lin_vel'])
            s.position_locked.zero_();state.step=-1
            active=state.compute(fake)
            assert (active['tracking_lin_vel_tight']>0.).all()
            s.numerical_failure.fill_(True);state.step=-1
            assert all(not value.any() for value in state.compute(fake).values())
    finally:
        s.unlocked=unlocked
        for name,value in previous.items():getattr(s,name).copy_(value)
    return dict(count=18,removed=sorted(removed),getup_phase_independent=True,
        driven_motor_velocity_and_soft_limits=True,passive_knees_independent=True,
        unlimited_motor_limits_preserved=True,
        orientation_weight=-100.,orientation_free_deg=5.,weighted_orientation_range=[-5.,0.],yaw_tracking_weight=1.,
        penalties_active_before_first_stand=True,height_precision_fixed=True,
        moving_velocity_boost_only_parking_unboosted=True,tight_gate_retained=True,numerical_mask_retained=True)


@torch.inference_mode()
def command_fixture(role,args):
    task='chuanliantui_'+role
    cfg,train=task_registry.get_cfgs(task,play=True)
    env,_=task_registry.make_env(task,env_cfg=cfg,train_cfg=train,num_envs=4,seed=43,device=args.device)
    try:
        with patch.object(mdp,'_update_walk_state'):
            raw=env.unwrapped;s=raw.command_manager.get_term('standup')
            before=raw.sim.data.qpos.clone()
            s.has_stood.fill_(True);s.observation_recovered.fill_(True);s.fallen.zero_()
            s.walk_ready.fill_(True)
            s.tracking_ready_steps.fill_(50);s.tracking_episode.fill_(True)
            s.tracking_velocity.fill_(1.);s.tracking_yaw_rate.zero_()
            s.tracking_interval_steps.fill_(100000);s.tracking_acceleration.fill_(1.)
            s._command[:,:2]=0.
            raw.common_step_counter+=1;s._update_command(None)
            if role=='getup':
                assert not torch.count_nonzero(s.command[:,:2])
            else:
                assert (s.command[:,0]>0.).all() and not s.command[:,1].any() and s.tracking_active.all()
                previous=s.command.clone();s._update_command(None)
                assert torch.equal(previous,s.command)
                s.tracking_velocity.zero_();s.tracking_yaw_rate.fill_(.3)
                raw.common_step_counter+=1;s._update_command(None)
                assert not s.command[:,0].any() and (s.command[:,1]>0.).all()
                s.tracking_velocity.zero_();s.tracking_yaw_rate.zero_()
                xy=torch.zeros((4,2),device=raw.device);vx=torch.zeros(4,device=raw.device)
                with patch.object(s,'_position_measurement',return_value=(xy,vx)):
                    for _ in range(50):raw.common_step_counter+=1;s._update_command(None)
                    assert s.position_locked.all()
                    # Use current heading to create a known +3 cm forward displacement.
                    q=raw.scene['robot'].data.root_link_quat_w
                    heading=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
                    xy[:,0]=.03*heading.cos();xy[:,1]=.03*heading.sin()
                    raw.common_step_counter+=1;s._update_command(None)
                    assert (s.position_velocity<0.).all() and (s.position_velocity.abs()<.1).all()
                s.walk_ready.zero_();raw.common_step_counter+=1;s._update_command(None)
                assert not s.position_locked.any() and not torch.count_nonzero(s.command[:,:2])
            assert torch.equal(before,raw.sim.data.qpos)
            return dict(role=role,mock_readiness_only=True,physical_steps=0,
                        command_gates_checked=True,parameter_updates=0)
    finally:env.close()


def role_check(role,args):
    task='chuanliantui_'+role
    cfg,train=task_registry.get_cfgs(task,play=True)
    env,_=task_registry.make_env(task,env_cfg=cfg,train_cfg=train,num_envs=args.num_envs,seed=43,device=args.device)
    try:
        raw=env.unwrapped;state=raw.command_manager.get_term('standup')
        assert cfg.commands['standup'].ground_start_probability == (1. if role=='getup' else 0.)
        assert state.ground_start.all() if role=='getup' else not state.ground_start.any()
        runner,_=task_registry.make_alg_runner(env,task,train_cfg=train,device=args.device)
        reset_checks=walking_reset_check(raw,args) if role=='walk' else None
        reward_checks=walking_reward_check(raw) if role=='walk' else None
        height_checks=walking_height_check(raw) if role=='walk' else getup_height_check(raw,runner,args)
        speed_checks=walking_speed_check(raw) if role=='walk' else None
        if role=='getup':
            # CPU初态从零历史/零上次动作开始，不能复用64步推演后的动作缓存。
            with torch.inference_mode():raw.reset(seed=43)
        raw.observation_manager.reset()
        raw.obs_buf=raw.observation_manager.compute(update_history=True)
        obs=env.get_observations();policy=runner.get_inference_policy(device=args.device)
        assert obs['actor'].shape==(args.num_envs,25) and obs['critic'].shape==(args.num_envs,71)
        assert obs['history'].shape==(args.num_envs,125)
        # Old task-mode and extra-flag access must be unnecessary for either network.
        with patch.object(state,'mode',side_effect=AssertionError('Mode input used')), patch.object(
                state,'extra_flags',side_effect=AssertionError('Flag input used')):
            physical=observations.get_state(raw)
            assert physical.actor().shape[-1]==25 and physical.critic().shape[-1]==71
        import numpy as np
        np.savez_compressed(args.output/f'{role}-initial.npz',qpos=raw.sim.data.qpos.cpu().numpy(),
            qvel=raw.sim.data.qvel.cpu().numpy(),actor=obs['actor'].cpu().numpy(),
            history=obs['history'].cpu().numpy(),command=state.command.cpu().numpy())
        before=copy.deepcopy(runner.alg.actor.state_dict())
        nonzero_commands=0
        contribution_sum=torch.zeros(len(raw.cfg.rewards),device=raw.device)
        contribution_abs=torch.zeros_like(contribution_sum)
        with torch.inference_mode():
            for _ in range(args.steps):
                height_before=state.command[:,2].clone()
                obs,reward,done,_=env.step(policy(obs))
                assert all(torch.isfinite(obs[k]).all() for k in ('actor','critic','history'))
                assert torch.isfinite(reward).all() and torch.equal(obs['actor'],obs['history'][:,-25:])
                nonzero_commands+=int(torch.count_nonzero(state.command[:,:2]))
                if role=='walk':
                    rates=raw.reward_manager._step_reward
                    contribution_sum+=rates.sum(0)
                    contribution_abs+=rates.abs().sum(0)
                if role=='getup':
                    assert not torch.count_nonzero(state.command[:,:2])
                    assert {key:term.weight for key,term in raw.cfg.rewards.items()}==rewards.GETUP_WEIGHTS
                    for name in rewards.GETUP_WEIGHTS:
                        assert torch.equal(rewards.reward(raw,name=name),rewards.getup_reward(raw,name=name))
                else:
                    continuing=~done.bool()
                    assert torch.equal(state.command[continuing,2],height_before[continuing])
        assert tensors_equal(before,runner.alg.actor.state_dict())
        for optimizer in (runner.alg.optimizer,runner.alg.encoder_optimizer):seed_optimizer_state(optimizer)
        if role=='walk':
            # 仅接口夹具：故意保存不同的非零档位/计数，避免全零恢复掩盖漏载。
            state.direction_curriculum=dict(
                forward=dict(stage=3,segments=73,successes=51,last_success_fraction=.82),
                backward=dict(stage=1,segments=19,successes=12,last_success_fraction=.85))
        checkpoint=args.output/f'interface-only-{role}.pt';runner.save(str(checkpoint))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for p in list(runner.alg.actor.parameters())+list(runner.alg.critic.parameters()):
            with torch.no_grad():p.add_(.01)
        runner.alg.optimizer.state.clear()
        runner.alg.encoder_optimizer.state.clear()
        runner.load(str(checkpoint),map_location=args.device)
        assert tensors_equal(saved['actor_state_dict'],runner.alg.actor.state_dict())
        assert tensors_equal(saved['critic_state_dict'],runner.alg.critic.state_dict())
        for key,optimizer in [('optimizer_state_dict',runner.alg.optimizer),
                              ('encoder_optimizer_state_dict',runner.alg.encoder_optimizer)]:
            assert tensors_equal(saved[key],optimizer.state_dict())
        actor_before=copy.deepcopy(runner.alg.actor.state_dict())
        rejected(lambda:runner.load(str(args.old_checkpoint),map_location=args.device))
        assert tensors_equal(actor_before,runner.alg.actor.state_dict())
        other='walk' if role=='getup' else 'getup'
        rejected(lambda:load_policy(checkpoint,other))
        for name in ('position_gain','position_velocity_limit','position_tolerance',
                     'position_lock_speed','position_lock_duration'):
            bad=dict(infos=copy.deepcopy(saved['infos']))
            bad['infos']['command_config'][name] += .01
            path=args.output/f'bad-{role}-{name}.pt';torch.save(bad,path)
            rejected(lambda:runner.load(str(path),map_location=args.device))
            assert tensors_equal(actor_before,runner.alg.actor.state_dict())
        if role=='walk':
            assert saved['infos']['experiment_profile_contract'] == 'chuanliantui-walk-physical25-v8'
            assert saved['infos']['command_contract'] == contracts(state.cfg)['command_contract']
            for key,value in (('experiment_profile_contract','chuanliantui-walk-physical25-v2'),
                              ('experiment_profile_contract','chuanliantui-walk-physical25-v3'),
                              ('experiment_profile_contract','chuanliantui-walk-physical25-v4'),
                              ('experiment_profile_contract','chuanliantui-walk-physical25-v5'),
                              ('experiment_profile_contract','chuanliantui-walk-physical25-v6'),
                              ('experiment_profile_contract','chuanliantui-walk-physical25-v7'),
                              ('command_contract','split-walk-episode-height-stop-straight-spin-speed-min1-moving75-walk-ready-armed1s-down100ms-v5'),
                              ('command_contract','split-walk-episode-height-stop-straight-spin-speed-min1-v3'),
                              ('command_contract','split-walk-episode-height-stop-straight-spin-speed-min1-moving75-v4'),
                              ('reward_recipe','split-walk-relative-velocity-parking-noboost-tilt5-v4'),
                              ('reward_recipe','split-walk-physical-motor-regularization-v3'),
                              ('command_contract','split-walk-episode-height-stop-straight-spin-v2'),
                              ('reward_recipe','split-walk-motion-without-getup-phases-v2'),
                              ('reward_recipe','split-retained-stand-v2-formulas-v1')):
                bad=copy.deepcopy(saved['infos']);bad[key]=value
                rejected(lambda:validate(bad,state.cfg))
            bad=copy.deepcopy(saved['infos']);bad['reward_weights'].pop('nominal_state')
            rejected(lambda:validate(bad,state.cfg))
            bad=copy.deepcopy(saved['infos']);bad['reward_weights']['recovered']=1.
            rejected(lambda:validate(bad,state.cfg))
            bad=copy.deepcopy(saved['infos']);bad['command_config']['walking_speed_min']=.1
            rejected(lambda:validate(bad,state.cfg))
            bad=copy.deepcopy(saved['infos']);bad['command_config'].pop('forward_speed_min')
            rejected(lambda:validate(bad,state.cfg))
            bad=copy.deepcopy(saved['infos']);bad['command_config']['speed_limits']=(.5,1.,1.5)
            rejected(lambda:validate(bad,state.cfg))
            bad=dict(infos=copy.deepcopy(saved['infos']))
            bad['infos']['experiment_profile_contract']='chuanliantui-walk-physical25-v1'
            bad['infos'].pop('command_contract');bad['infos']['command_config'].pop('height_range')
            rejected(lambda:validate(bad['infos'],state.cfg))
            bad['infos']=copy.deepcopy(saved['infos']);bad['infos']['command_config']['height_range']=(.22,.29)
            rejected(lambda:validate(bad['infos'],state.cfg))
            bad=dict(infos=copy.deepcopy(saved['infos']));bad['infos']['standup_curriculum']['position_hold']['position_gain']+=.1
            path=args.output/'bad-nested-position.pt';torch.save(bad,path)
            rejected(lambda:load_policy(path,'walk'))
        numeric=export(checkpoint,args.output/f'interface-only-{role}.onnx',samples=16)
        # Isolated terminal-clock fixtures: these do not advance physics or optimize.
        yes=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device);no=~yes
        if role=='getup':
            # Successful get-up keeps the same policy balancing; no success-shortened episode.
            assert set(cfg.terminations)=={'failure','time_out'}
            assert not cfg.commands['standup'].post_stand_tracking
        else:
            state.walk_down_ticks.zero_();state.completed.zero_();raw.episode_length_buf.fill_(100)
            with patch.object(mdp.stand,'failure',return_value=no), patch.object(
                    mdp,'physical_conditions',return_value=(no,yes,no,yes)):
                for _ in range(9):
                    raw.common_step_counter+=1;assert not mdp.failure(raw).any()
                raw.common_step_counter+=1;assert mdp.failure(raw).all() and state.completed.all()
        return dict(task=task,steps=args.steps,num_envs=args.num_envs,checkpoint=str(checkpoint),
            shapes={k:list(obs[k].shape) for k in ('actor','critic','history')},
            finite=True,parameter_updates=0,optimizer_resume_fixture_step=7,
            optimizer_states_exact=True,old_and_wrong_role_rejected=True,
            nonzero_xy_command_samples=nonzero_commands,single_task_termination_checked=True,
            walking_reset_checks=reset_checks,walking_height_checks=height_checks,walking_speed_checks=speed_checks,
            walking_reward_checks=reward_checks,onnx=numeric,
            untrained_reward_contributions_before_dt={name:dict(
                mean=float(contribution_sum[i]/(args.steps*args.num_envs)),
                mean_abs=float(contribution_abs[i]/(args.steps*args.num_envs)))
                for i,name in enumerate(raw.reward_manager._term_names)} if role=='walk' else None)
    finally:
        env.close()


def play_check(args):
    from ct_mjlab.scripts.play import play
    calls=[]
    class FiniteViewer:
        def __init__(self,env,policy,*a,**kw):self.env,self.policy=env,policy
        def run(self):
            s=self.env.unwrapped.command_manager.get_term('standup')
            if s.cfg.role=='walk':
                assert s.speed_curriculum_frozen
                saved=torch.load(args.output/'interface-only-walk.pt',map_location='cpu',weights_only=False)
                assert s.direction_curriculum == {k:saved['infos']['standup_curriculum']['speed_curriculum'][k]
                                                  for k in ('forward','backward')}
                before=copy.deepcopy(s.direction_curriculum)
            obs=self.env.get_observations()
            with torch.inference_mode():
                for _ in range(32):
                    obs,reward,_,_=self.env.step(self.policy(obs))
                    assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            if s.cfg.role=='walk':assert s.direction_curriculum==before
            calls.append(32)
    for role in ('getup','walk'):
        ns=SimpleNamespace(task='chuanliantui_'+role,checkpoint=args.output/f'interface-only-{role}.pt',
            seed=43,num_envs=2,device=args.device,smp_prior=None,knockdown_every=None,viewer='native',viser_port=None)
        with patch('mjlab.viewer.NativeMujocoViewer',FiniteViewer):play(ns)
    assert calls==[32,32]
    return dict(actual_single_task_play_entries=True,finite_viewer_steps=calls,trained_behavior_pass=False)


def main(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    if args.commands_only:
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training authorized')):
            report=dict(verified=True,parameter_updates=0,command_fixtures=[command_fixture(r,args) for r in ('getup','walk')],source_sha256=digest(__file__))
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training authorized')):
        getup_compatibility=getup_regression(args)
        switch_check()
        roles=[role_check(role,args) for role in ('getup','walk')]
        playback=play_check(args)
    report=dict(verified=True,parameter_updates=0,roles=roles,external_switch_tensor_checks=True,play=playback,
        getup_compatibility=getup_compatibility,
        scope='Finite physics, mock-condition routing, strict role/config resume, and untrained ONNX/play interface fixtures only; no trained behavior approval',
        old_checkpoint_sha256=digest(args.old_checkpoint),source_sha256=digest(__file__))
    (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=20)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--commands-only',action='store_true',help='Only exercise isolated ready/command fixtures; no physics rollout')
    parser.add_argument('--getup-oracle',type=Path)
    parser.add_argument('--getup-checkpoint',type=Path)
    args=parser.parse_args()
    if (args.getup_oracle is None)!=(args.getup_checkpoint is None):
        parser.error('--getup-oracle and --getup-checkpoint must be supplied together')
    main(args)
