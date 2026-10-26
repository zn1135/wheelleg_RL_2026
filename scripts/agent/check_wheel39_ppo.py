#!/usr/bin/env python3
"""Real wheel39 SMP/PPO initialization, sampling and checkpoint checks; no updates."""
import argparse,copy,json,shutil
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import torch
from ct_mjlab.bc import load_actor
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations as observations
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
from ct_mjlab.smp.ppo import (configure_smp,policy_observation_contract,observation_contract,
    OBSERVATION_CONTRACT,PROFILE,WHEEL_PROFILE)
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT,HANDOVER_WHEEL_ANGLES
from check_lqr_student_closed_loop import digest,ROOT


def rejected(call):
    try:call()
    except ValueError as error:return str(error)
    raise AssertionError('Incompatible same-dimension interface was accepted')


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs('chuanliantui');cfg.seed=train.seed=42
    old,old_train=task_registry.get_cfgs('chuanliantui');configure_smp(old,old_train,args.prior)
    contract=policy_observation_contract(args.checkpoint,bc=True)
    assert contract==HANDOVER_OBSERVATION_CONTRACT
    configure_smp(cfg,train,args.prior,policy_contract=contract)
    assert observation_contract(old)==OBSERVATION_CONTRACT and observation_contract(cfg)==contract
    assert asdict(cfg.actions['pd'])==asdict(old.actions['pd']) and cfg.decimation==old.decimation
    assert cfg.sim.mujoco.timestep==old.sim.mujoco.timestep==.002
    assert {k:asdict(v) for k,v in cfg.rewards.items()}=={k:asdict(v) for k,v in old.rewards.items()}
    assert cfg.events['reset_pose'].func is old.events['reset_pose'].func
    assert cfg.actions['pd'].action_delay_choices_s==(.002,.004)
    immutable={str(p):digest(p) for p in (args.checkpoint,args.prior,
        ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')}
    env,_=task_registry.make_env('chuanliantui',env_cfg=cfg,train_cfg=train,num_envs=20,device='cuda:0')
    try:
        raw=env.unwrapped;runner,_=task_registry.make_alg_runner(env,'chuanliantui',train_cfg=train,device='cuda:0')
        motor_ids=torch.tensor([raw.sim.mj_model.actuator('robot/'+name+'_motor').id
            for name in ACTUATED_JOINTS],device=raw.device)
        critic={k:v.clone() for k,v in runner.alg._raw_critic.state_dict().items()}
        runner.initialize_from_bc(args.checkpoint)
        actor,saved=load_actor(args.checkpoint,'cuda:0');obs=env.get_observations()
        torch.testing.assert_close(actor(obs),runner.alg._raw_actor(obs),atol=0,rtol=0)
        for key,value in actor.state_dict().items():torch.testing.assert_close(runner.alg._raw_actor.state_dict()[key],value,atol=0,rtol=0)
        for key,value in critic.items():torch.testing.assert_close(runner.alg._raw_critic.state_dict()[key],value,atol=0,rtol=0)
        assert not runner.alg.optimizer.state and not hasattr(raw.action_manager.get_term('pd'),'teacher')
        assert not hasattr(raw,'_bc_reference_observer')
        path=args.output/'untrained-interface.pt';runner.save(str(path))
        source=torch.load(path,map_location='cpu',weights_only=False)
        assert source['infos']['observation_contract']==contract and source['infos']['experiment_profile_contract']==WHEEL_PROFILE
        assert source['infos']['wheel_angles']==HANDOVER_WHEEL_ANGLES
        assert policy_observation_contract(path)==contract
        errors={};bad=args.output/'rejected.pt'
        for name,change in (
            ('legacy_profile',lambda x:x['infos'].update(observation_contract=OBSERVATION_CONTRACT,experiment_profile_contract=PROFILE)),
            ('wrong_wheel_origin',lambda x:x['infos']['wheel_angles'].update(origin='episode_reset')),
            ('wrong_xml',lambda x:x['infos'].update(xml_sha256='different')),
            ('wrong_prior',lambda x:x['infos']['smp'].update(prior_sha256='different')),
            ('adaptive_scale',lambda x:x['infos']['smp_normalizer']['mean_abs'].fill_(2.))):
            corrupt=copy.deepcopy(source);change(corrupt);torch.save(corrupt,bad)
            errors[name]=rejected(lambda:runner.load(str(bad)))
        errors['old_bc']=rejected(lambda:runner.initialize_from_bc(args.legacy))
        from ct_mjlab.scripts.export import load_actor as export_actor
        errors['legacy_export']=rejected(lambda:export_actor(path))
        # 合成相位与轮角变化，只检查清零算术；不把它当实际起立成功。
        state=observations.get_state(raw);command=state.command
        ids=torch.arange(20,device=raw.device);chosen=ids[::2];other=ids[1::2]
        command.recovered.zero_();state.wheel_angle_handover_latched.zero_()
        q=raw.sim.data.qpos.clone();vel=raw.sim.data.qvel.clone();action=state.action
        raw.sim.data.qpos[:,state._motor_q_adr[4:6]]+=torch.tensor([1.,-2.],device=raw.device)
        raw.sim.forward();state.sample_substep();history=state.delay.history.clone()
        fifo=action._action_history.clone();command.recovered[chosen]=True
        critic_obs=state.critic()
        assert not critic_obs[chosen,14:16].any() and state.wheel_angle_handover_latched[chosen].all()
        assert not state.delay.history[:,chosen,10:12].any()
        torch.testing.assert_close(state.delay.history[:,other],history[:,other],atol=0,rtol=0)
        keep=torch.ones(18,dtype=torch.bool,device=raw.device);keep[10:12]=False
        torch.testing.assert_close(state.delay.history[:,:,keep],history[:,:,keep],atol=0,rtol=0)
        torch.testing.assert_close(action._action_history,fifo,atol=0,rtol=0)
        locked=state.wheel_angle_origin.clone();state.actor();state.critic();state.sample_substep()
        torch.testing.assert_close(state.wheel_angle_origin,locked,atol=0,rtol=0)
        raw.sim.data.qpos.copy_(q);raw.sim.data.qvel.copy_(vel);raw.sim.forward()
        raw.reset(seed=43);assert not state.wheel_angle_handover_latched.any()
        torch.testing.assert_close(state.wheel_angle_origin,state.wheel_angles(),atol=0,rtol=0)
        assert not state.critic()[:,14:16].any()
        obs=env.get_observations();prior=raw._smp_reward
        frozen={k:v.clone() for k,v in prior.prior.state_dict().items()}
        with patch.object(runner.alg.optimizer,'step',side_effect=AssertionError('Training forbidden')):
            for tick in range(24):
                obs,rewards,dones,extras=env.step(runner.alg.act(obs))
                runner.alg.process_env_step(obs,rewards,dones,extras)
                assert obs['actor'].shape==(20,39) and obs['critic'].shape==(20,82)
                assert torch.isfinite(rewards).all() and all(torch.isfinite(v).all() for v in obs.values())
                torch.testing.assert_close((raw.reward_manager._step_reward*raw.step_dt).sum(-1),rewards,atol=2e-6,rtol=2e-5)
                if tick<20:assert not raw.sim.data.actuator_force[:,motor_ids].any()
            runner.alg.compute_returns(obs)
        before_origin=state.wheel_angle_origin[1:].clone();before_latch=state.wheel_angle_handover_latched[1:].clone()
        history=state.delay.history[:,1:].clone();motion=prior.history.frames[1:].clone();counts=prior.history.count[1:].clone()
        raw.reset(env_ids=ids[:1])
        assert not state.wheel_angle_handover_latched[0]
        torch.testing.assert_close(state.wheel_angle_origin[1:],before_origin,atol=0,rtol=0)
        torch.testing.assert_close(state.wheel_angle_handover_latched[1:],before_latch,atol=0,rtol=0)
        torch.testing.assert_close(state.delay.history[:,1:],history,atol=0,rtol=0)
        torch.testing.assert_close(prior.history.frames[1:],motion,atol=0,rtol=0)
        torch.testing.assert_close(prior.history.count[1:],counts,atol=0,rtol=0)
        for key,value in frozen.items():torch.testing.assert_close(prior.prior.state_dict()[key],value,atol=0,rtol=0)
        # 恢复策略／评分RNG，环境新回合参考由reset生成，不能套用旧物理回合原点。
        raw.reset(seed=44);origin=state.wheel_angle_origin.clone()
        runner.load(str(path));assert runner.bc_init['sha256']==digest(args.checkpoint)
        assert torch.equal(prior.generator.get_state().cpu(),source['infos']['smp_generator'].cpu())
        torch.testing.assert_close(state.wheel_angle_origin,origin,atol=0,rtol=0)
        for key,value in critic.items():torch.testing.assert_close(runner.alg._raw_critic.state_dict()[key],value,atol=0,rtol=0)
        assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
        obs=env.get_observations();torch.testing.assert_close(actor(obs),runner.alg._raw_actor(obs),atol=0,rtol=0)
        for name,sha in immutable.items():assert digest(name)==sha
        result=dict(passed=True,optimizer_updates=0,num_envs=20,rollout_steps=24,returns_computed=True,
            observation_contract=contract,ppo_profile=WHEEL_PROFILE,actor_dim=39,critic_dim=82,actions=6,
            bc_total_updates=saved['total_actor_updates'],actor_weights_and_outputs_identical=True,critic_unchanged=True,
            ppo_optimizer_empty=True,standard_deviation=.05,ground_hold_actual_zero=True,
            initial_reset_and_first_recovery_reference=True,only_wheel_position_history_cleared=True,
            no_repeated_reanchor=True,partial_reset_no_cross_env_history_leak=True,
            actor_critic_optimizer_and_smp_rng_roundtrip=True,physical_episode_reference_not_restored_from_checkpoint=True,
            reward_weights_physics_and_control_unchanged=True,all_reward_contributions_sum_once_dt=True,prior_and_immutable_inputs_unchanged=True,
            rejects=errors,bc_init=runner.bc_init,checkpoint_sha256=digest(path),immutable_sha256=immutable,
            scope='Real20-env randomized finite sampling/returns plus explicit phase arithmetic; no optimization, parking improvement or deployment claim.')
        (args.output/'check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','legacy','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    check(p.parse_args())
