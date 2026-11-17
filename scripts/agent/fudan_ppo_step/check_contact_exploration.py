"""台阶搭沿探索配方的参考准备与零学习检查，不启动训练。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.alignment/'alignment.json').read_text())
    frames=[];provenance=[]
    for row in source['tracks']:
        if not row['all_frames_within_limits']:continue
        path=args.alignment/'tracks'/f'{row["id"]}.npz'
        assert digest(path)==row['track_sha256']
        with np.load(path) as data:
            active=np.isin(data['phase'],[1,2,5])&(abs(data['motor_v']).max(-1)>.5)
            indices=np.flatnonzero(active)
            selected=indices[np.linspace(0,len(indices)-1,32).round().astype(int)]
            frames.append(np.concatenate((data['motor_q'][selected],data['motor_v'][selected]),-1))
            provenance.append(dict(track=row['id'],sha256=digest(path),selected_indices=selected.tolist()))
    assert len(frames)==5
    reference=np.concatenate(frames).astype(np.float32)
    path=args.output/'motion.npz';np.savez_compressed(path,features=reference)
    report=dict(kind='weak_motion_hint_not_success_demonstration',source_alignment=str(args.alignment.resolve()),
        source_alignment_sha256=digest(args.alignment/'alignment.json'),motion_sha256=digest(path),
        frames=len(reference),feature_order=['lf','lr','rf','rr','lf_vel','lr_vel','rf_vel','rr_vel'],
        sources=provenance,physics_steps=0,optimizer_updates=0,
        note='Unordered state/velocity hints; no action labels, reference phase input or forced playback')
    (args.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(motion_sha256=digest(path),frames=len(reference))))


def static_checks(output):
    import torch
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,SOURCE,SOURCE_SHA
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.fudan_ppo_step.runner import StepRunner,initialize_policy
    from ct_mjlab.envs.fudan_ppo_step.reference_motion import MotionHint
    from ct_mjlab.fudan_ppo_walk.contract import make_network
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    from .evaluate import score
    from .check import scoring_checks,contact_logic_checks
    cfg=FudanStepCfg();assert digest(ROOT/SOURCE)==SOURCE_SHA
    hint=MotionHint(cfg.step,'cpu')
    frames=hint.frames*hint.scale
    assert torch.allclose(hint.score(frames[:,:4],frames[:,4:]),torch.ones(len(frames)),atol=1e-5)
    assert (hint.score(frames[:,:4]+10,frames[:,4:])<.001).all()
    fake=object.__new__(StepEnv);fake.cfg=cfg
    fake.terrain=SimpleNamespace(terrain_types=torch.tensor([0,1,2]))
    fake.contacts=SimpleNamespace(failed=torch.ones(3,dtype=torch.bool))
    assert fake._contact_failure().tolist()==[True,False,False]
    cfg.step.allow_temporary_support=False
    assert fake._contact_failure().all()
    cfg.step.allow_temporary_support=True
    online=object.__new__(StepEnv);online.cfg=cfg;online.dt=.01
    online.terrain=SimpleNamespace(terrain_types=torch.tensor([1]),terrain_levels=torch.tensor([7]))
    online.env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1,3)),
        sim=SimpleNamespace(data=SimpleNamespace(xpos=torch.tensor([[[1.5,.21,.26],[1.5,-.21,.26]]]))))
    online.wheel_ids=torch.tensor([0,1]);online.support_height=torch.tensor([[.2,.2]])
    online.contacts=SimpleNamespace(supported=torch.ones(1,2,dtype=torch.bool),failed=torch.tensor([True]),
        step_contact=torch.tensor([False]),step_speed_peak=torch.tensor([0.]))
    online.crossed=torch.tensor([True]);online.landing_ticks=torch.tensor([10]);online.parking_ticks=torch.tensor([199])
    online.parking_time=torch.tensor([1.99]);online.max_approach_speed=torch.tensor([.3])
    online.robot=SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=torch.zeros(1,3)))
    online.position=torch.tensor([[1.5,0.,.5]]);online.normal_support=torch.tensor([True])
    online.commands=torch.tensor([[0.,0.,.35]]);online.gravity=torch.tensor([[0.,0.,-1.]])
    online._progress();assert online.success.item() and online.parking_ticks.item()==200
    online.contacts.step_contact[:]=True;online._progress();assert not online.success.item()
    n=800;wheel=np.zeros((n,2,3));wheel[:,:,0]=np.minimum(np.arange(n)[:,None]*.003,1.8)
    wheel[:,0,1]=.21;wheel[:,1,1]=-.21;wheel[:,:,2]=.26
    command=np.zeros((n,3));command[:600,0]=.3;command[:,2]=.3
    velocity=np.zeros((n,3));velocity[:600,0]=.3
    gravity=np.zeros((n,3));gravity[:,2]=-1.
    trace=dict(valid=np.ones(n,bool),origin=np.zeros(3),position=np.zeros((n,3)),wheel_pos=wheel,
        linear_world=velocity,linear_body=velocity,angular_body=np.zeros((n,3)),gravity=gravity,
        command=command,force=np.zeros((n,10,2)),support=np.ones((n,2),bool),
        normal_support=np.ones(n,bool),support_height=np.full((n,2),.2),weight=np.array(131.4))
    sample=dict(id='contact_then_stand',mode='up',height=.2)
    trace['force'][100:180,:,0]=50.
    assert not score(sample,trace,duration=8.)['passed']
    permitted=score(sample,trace,duration=8.,allow_temporary_support=True,landing_clearance_m=.15)
    assert permitted['passed'] and permitted['strict_contact_violation']
    trace['force'][650:675,:,0]=50.
    assert not score(sample,trace,duration=8.,allow_temporary_support=True)['passed']
    policy=make_network();transfer=initialize_policy(policy,ROOT/SOURCE,cfg)
    teacher,_=load_policy(ROOT/SOURCE,device='cpu');teacher.eval().requires_grad_(False)
    runner=object.__new__(StepRunner);runner.policy=policy;runner.flat_teacher=teacher
    runner.env=SimpleNamespace(cfg=cfg)
    generator=torch.Generator().manual_seed(716)
    obs=torch.randn(8,25,generator=generator);history=torch.randn(8,125,generator=generator)
    flat=torch.tensor([True]*4+[False]*4)
    assert runner.flat_preservation_loss(obs,history,flat).item()<1e-10
    saved={k:v.clone() for k,v in policy.state_dict().items()}
    with torch.no_grad():list(policy.actor.modules())[-1].bias.add_(.01)
    obs.requires_grad_();history.requires_grad_()
    loss=runner.flat_preservation_loss(obs,history,flat);loss.backward()
    assert loss>0 and obs.grad[~flat].abs().max()==0 and obs.grad[flat].abs().max()>0
    assert all(p.grad is None for p in teacher.parameters())
    assert not any(p.grad is not None for p in policy.encoder.parameters())
    policy.load_state_dict(saved);policy.zero_grad(set_to_none=True)
    return dict(reference_frames=len(frames),weak_reference_peak_one=True,
        legacy_contact_statistics=contact_logic_checks(),legacy_scoring=scoring_checks(),
        stair_contacts_do_not_terminate=True,flat_contact_rule_retained=True,
        temporary_support_then_clear_stop_passes=True,continued_support_during_stop_fails=True,
        online_stop_after_historical_support_passes=True,online_stop_contact_rejected=True,
        flat_teacher_matches_source=True,flat_only_actor_gradient=True,teacher_and_encoder_frozen_for_anchor=True,
        inherited_std=transfer['std'],optimizer_updates=0,physics_steps=0)


def gpu_checks(args,report):
    import torch
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,FudanStepCfgPPO,SOURCE,recipe_record
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.fudan_ppo_step.runner import StepRunner
    from .check import contact_friction
    if args.num_envs!=20 or args.steps<200:raise ValueError('Use20 environments and at least200 policy steps')
    cfg=FudanStepCfg();env=None;runner=None;torch.set_num_threads(1);torch.manual_seed(716)
    try:
        env=StepEnv(20,cfg=cfg,randomized=True,seed=716)
        runner=StepRunner(env,train_cfg=FudanStepCfgPPO(),log_dir=args.output/'runner',device='cuda:0')
        transfer=runner.initialize_from_ppo(ROOT/SOURCE)
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        runner._load_flat_teacher()
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        original={k:v.clone() for k,v in runner.policy.state_dict().items()}
        ids=torch.arange(20,device=env.device);flat=env.modes==0
        flat_max_v=flat_max_yaw=0.
        for _ in range(16):
            env.resample(ids)
            flat_max_v=max(flat_max_v,float(env.commands[flat,0].abs().max()))
            flat_max_yaw=max(flat_max_yaw,float(env.commands[flat,1].abs().max()))
        assert flat_max_v>2 and flat_max_yaw>1
        obs,critic,history=env.reset()
        env.contacts.failed.fill_(True)
        with patch.object(env.env,'step',return_value=None):
            _,_,_,done,info,_=env.step(torch.zeros(20,6,device=env.device),auto_reset=False,resample=False)
        assert done[flat].all() and not done[~flat].any()
        assert not info['step_failure'][~flat].any()
        obs,critic,history=env.reset()
        rewards=[];friction=[]
        with torch.inference_mode():
            for step in range(args.steps+runner.num_steps_per_env):
                if time.time()>=args.deadline:raise TimeoutError('V7 no-learning precheck deadline')
                if step<args.steps:
                    action,_=runner.policy.act_inference(obs,history)
                else:action=runner.alg.act(obs,history,critic)
                traversing=(~flat)&~env.crossed.clone()
                obs,critic,reward,done,info,history=env.step(action)
                if step>=args.steps:runner.alg.process_env_step(reward,done,info,obs)
                assert obs.shape==(20,25) and critic.shape==(20,141) and history.shape==(20,125)
                assert torch.isfinite(reward).all() and not env.contacts.capacity_overflow.any()
                assert set(info['terms'])==set(cfg.rewards.scales)
                assert (info['terms']['reference_motion'][flat]==0).all()
                assert (info['terms']['collision'][~flat]==0).all()
                assert (info['terms']['reference_motion']<=.2*env.dt+1e-6).all()
                for name in ('base_height','base_height_enhance','nominal_state','lin_vel_z'):
                    assert (info['terms'][name][traversing]==0).all()
                friction.append(contact_friction(env));rewards.append(float(reward.mean()))
            runner.alg.compute_returns(torch.cat((critic,runner.policy.encode(history)),-1))
        # 执行真实PPO反向及平地模仿回调，在Adam更新边界拦截，权重保持不变。
        with patch.object(runner,'flat_preservation_loss',wraps=runner.flat_preservation_loss) as anchor:
            with patch('torch.optim.Adam.step',side_effect=AssertionError('V7 backward boundary')):
                try:runner.alg.update()
                except AssertionError as exc:assert str(exc)=='V7 backward boundary'
                else:raise AssertionError('PPO update must stop before optimizer step')
            assert anchor.call_count==1
        assert all(p.grad is None for p in runner.flat_teacher.parameters())
        assert all(torch.equal(v,original[k]) for k,v in runner.policy.state_dict().items())
        assert not runner.alg.optimizer.state and not runner.alg.extra_optimizer.state
        saved=runner.save('model_precheck.pt')
        from ct_mjlab.fudan_ppo_walk.policy import load_policy
        loaded,_=load_policy(saved,device='cuda:0')
        with torch.inference_mode():
            before=runner.policy.act_inference(obs,history)[0];after=loaded.act_inference(obs,history)[0]
        assert torch.equal(before,after)
        report.update(kind='step_gpu_precheck',num_envs=20,recipe=recipe_record(cfg),optimizer_updates=0,
            physics_steps=int(env.contacts.total_ticks),rollout_steps_without_update=runner.num_steps_per_env,
            migration=transfer,all_networks_unchanged_after_rollout=True,saved_policy_load_and_fixed_inference_equal=True,
            critic_height_samples=77,flat_max_sampled_speed=flat_max_v,flat_max_sampled_yaw=flat_max_yaw,
            stair_contact_flag_not_applied=True,flat_contact_flag_applied=True,
            reference_reward_bounded=True,all_reward_terms_logged=True,contact_friction=friction,
            constraint_peak=int(env.contacts.constraint_peak),capacity_overflow=False,
            reward_mean_range=[min(rewards),max(rewards)],flat_teacher_frozen=True,
            teacher_reload_preserves_rng=True,ppo_backward_with_flat_anchor_checked=True,
            scope='V7 contact/reward/migration/rollout checks only; no optimizer step or stair success claim')
    finally:
        if runner is not None:runner.writer.close()
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare-reference');p.add_argument('--alignment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args)


if __name__=='__main__':main()
