#!/usr/bin/env python3
"""Same-state B/A reward equality, real RewardManager and checkpoint isolation; no learning."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.managers.reward_manager import RewardManager
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2 import rewards
from ct_mjlab.envs.chuanliantui_stand_v2.export import export
from ct_mjlab.envs.chuanliantui_stand_v2.runner import task_contracts
from scripts.agent.stand_v2.check_stand_v2_speed_commands import equal_tree
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        'ct_mjlab.envs.chuanliantui_stand_v2._a_reward_oracle', args.oracle)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand_v2')
    cfg.commands['standup'].post_stand_tracking = True
    agent.actor.distribution_cfg['std_range'] = (.05, 1e6)
    assert list(cfg.rewards) == list(oracle.WEIGHTS) and len(cfg.rewards) == 19
    env, _ = task_registry.make_env('chuanliantui_stand_v2', env_cfg=cfg, train_cfg=agent,
        num_envs=128, seed=42, device=args.device)
    tight_path = args.output/'interface-only-untrained-tight-v6.pt'
    try:
        raw, s = env.unwrapped, env.unwrapped.command_manager.get_term('standup')
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui_stand_v2', train_cfg=agent, device=args.device)
        current, previous = rewards.RewardState(raw), oracle.RewardState(raw)
        counts = dict(active=0, inactive=0, zero_role=0, active_zero_command=0, reward_samples=0)
        def compute(port):
            port.step = -1
            return {k: v.clone() for k,v in port.compute(raw).items()}
        def compare():
            # 同一物理状态分别计算，避免GPU复跑轨迹数值分歧混入奖励差异。
            s.cfg.tracking_tight_term = False
            old, off = compute(previous), compute(current)
            assert old.keys() == off.keys() and len(off) == 19
            assert all(torch.equal(old[k], off[k]) for k in old)
            old_active = s.cfg.tracking_weight_while_active
            s.cfg.tracking_weight_while_active = True
            a = compute(previous)
            s.cfg.tracking_tight_term = True
            b = compute(current)
            assert list(b) == [*a, 'tracking_lin_vel_tight']
            assert all(torch.equal(a[k], b[k]) for k in a)
            expected = torch.exp(-(s.command[:, 0]-raw.scene['robot'].data.root_link_lin_vel_b[:, 0]).square()
                /s.cfg.tracking_tight_kernel_denominator).masked_fill(~s.tracking_active, 0.)
            expected.masked_fill_(s.numerical_failure | ~s.reward_active, 0.)
            assert torch.equal(expected, b['tracking_lin_vel_tight'])
            assert not b['tracking_lin_vel_tight'][~s.tracking_active].any()
            counts['active'] += int(s.tracking_active.sum())
            counts['inactive'] += int((~s.tracking_active).sum())
            counts['zero_role'] += int((~s.tracking_episode).sum())
            counts['active_zero_command'] += int((s.tracking_active & (s.command[:, 0] == 0.)).sum())
            counts['reward_samples'] += raw.num_envs
            s.cfg.tracking_tight_term = False
            s.cfg.tracking_weight_while_active = old_active
        def recipe(speed, active, tight):
            s.cfg.speed_curriculum, s.cfg.tracking_weight_while_active, s.cfg.tracking_tight_term = speed, active, tight
            raw.cfg.rewards.pop('tracking_lin_vel_tight', None)
            if tight:
                rewards.configure_tight_reward(raw.cfg, s.cfg)
            raw.reward_manager = RewardManager(raw.cfg.rewards, raw, scale_by_dt=raw.cfg.scale_rewards_by_dt)
            if hasattr(raw, '_isaac_v2_rewards'):
                raw._isaac_v2_rewards.step = -1
            names = [*oracle.WEIGHTS, *(['tracking_lin_vel_tight'] if tight else [])]
            assert raw.reward_manager.active_terms == names and list(raw.cfg.rewards) == names
            assert list(raw.reward_manager.reset()) == ['Episode_Reward/'+k for k in names]
        rows = []
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer updates')):
            for label, speed, active, path, ticks in (
                ('v4-fixed',False,False,args.checkpoint,800),
                ('v6-speed-old-reward',True,False,args.speed_checkpoint,1200),
                ('A-v5',True,True,args.active_checkpoint,1200)):
                recipe(speed, active, False)
                runner.load(str(path), map_location=args.device)
                before = [tensor_digest(m.state_dict()) for m in (runner.alg.actor,runner.alg.critic)]
                optimizers = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer,runner.alg.encoder_optimizer)]
                s.speed_curriculum_frozen = True
                raw.reset()
                obs, policy = env.get_observations(), runner.get_inference_policy(device=args.device)
                original = raw.reward_manager.compute
                captured = {}
                def reward_compute(*, dt):
                    assert torch.equal(captured['command'],s.command)
                    assert torch.equal(captured['active'],s.tracking_active)
                    compare()
                    return original(dt=dt)
                raw.reward_manager.compute = reward_compute
                for tick in range(ticks):
                    captured['command'], captured['active'] = s.command.clone(), s.tracking_active.clone()
                    obs,_,_,_ = env.step(policy(obs))
                    if (tick+1) % 400 == 0:
                        print(f'{label}: frozen reward pairs {(tick+1)/100:.0f}s',flush=True)
                raw.reward_manager.compute = original
                assert before == [tensor_digest(m.state_dict()) for m in (runner.alg.actor,runner.alg.critic)]
                for a,o in zip(optimizers,(runner.alg.optimizer,runner.alg.encoder_optimizer)):
                    equal_tree(a,o.state_dict())
                rows.append(dict(label=label,ticks=ticks,off_bit_equal=True,other_terms_bit_equal=True))
            # 合成覆盖: 恰好50%零速角色、active零命令、准备屏蔽和非有限屏蔽。
            ids = torch.arange(raw.num_envs,device=raw.device)
            s.tracking_episode[:] = ids < 64
            s.tracking_active.copy_(s.tracking_episode)
            s._command[:, 0] = 0.
            s.reward_active[:] = True
            s.numerical_failure[:] = False
            compare()
            s.reward_active[:16] = False
            s.numerical_failure[16:32] = True
            compare()
            # 实际manager对e=0/0.05/0.1/0.2/0.5施加权重与dt一次，不受宽核限幅。
            s.reward_active[:] = True
            s.numerical_failure.zero_()
            s.tracking_active[:] = True
            vx = raw.scene['robot'].data.root_link_lin_vel_b[:, 0]
            errors = torch.tensor([0.,.05,.1,.2,.5],device=raw.device).repeat(26)[:128]
            s._command[:, 0] = vx+errors
            recipe(True,True,True)
            raw.reward_manager.compute(dt=raw.step_dt)
            values = compute(current)
            expected = values['tracking_lin_vel_tight']*s.cfg.tracking_tight_weight
            idx = raw.reward_manager.active_terms.index('tracking_lin_vel_tight')
            torch.testing.assert_close(raw.reward_manager._step_reward[:,idx],expected,atol=3e-8,rtol=1e-7)
            assert expected[0] == .5
            # 两个优化器、actor/critic和课程精确往返。
            model = [tensor_digest(m.state_dict()) for m in (runner.alg.actor,runner.alg.critic)]
            opts = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer,runner.alg.encoder_optimizer)]
            curriculum = copy.deepcopy(s.checkpoint_state())
            runner.save(str(tight_path))
            infos = torch.load(tight_path,map_location='cpu',weights_only=False)['infos']
            assert all(infos[k] == v for k,v in task_contracts(True,True,True,True).items())
            runner.load(str(tight_path),map_location=args.device)
            assert model == [tensor_digest(m.state_dict()) for m in (runner.alg.actor,runner.alg.critic)]
            equal_tree(curriculum,s.checkpoint_state())
            for a,o in zip(opts,(runner.alg.optimizer,runner.alg.encoder_optimizer)):equal_tree(a,o.state_dict())
            # v4/v6命令/A/B全部16种原生加载组合，12种跨配方拒绝。
            versions = [(False,False,False,args.checkpoint),(True,False,False,args.speed_checkpoint),
                        (True,True,False,args.active_checkpoint),(True,True,True,tight_path)]
            rejects=[]
            for i,(speed,active,tight,_) in enumerate(versions):
                recipe(speed,active,tight)
                for j,(_,_,_,path) in enumerate(versions):
                    before = tensor_digest(runner.alg.actor.state_dict())
                    try:runner.load(str(path),map_location=args.device)
                    except ValueError as error:
                        assert i != j and ('recipe_version' in str(error) or 'profile' in str(error))
                        assert before == tensor_digest(runner.alg.actor.state_dict())
                        rejects.append(str(error))
                    else:assert i == j
            assert len(rejects) == 12
            recipe(True,True,True)
            for field,value in (('tracking_tight_kernel_denominator',.03),('tracking_tight_weight',.6)):
                old = getattr(s.cfg,field)
                setattr(s.cfg,field,value)
                try:runner.load(str(tight_path),map_location=args.device)
                except ValueError as error:assert 'tracking_tight_config' in str(error)
                else:raise AssertionError('Tight parameter mismatch accepted')
                setattr(s.cfg,field,old)
            s.cfg.tracking_weight_while_active = False
            try:rewards.configure_tight_reward(raw.cfg,s.cfg)
            except ValueError as error:assert 'requires' in str(error)
            else:raise AssertionError('Missing prerequisite accepted')
            s.cfg.tracking_weight_while_active = True
            onnx = export(tight_path,args.output/'interface-only-tight-v6.onnx',samples=16)
            assert onnx['recipe_version'] == rewards.TIGHT_TRACKING_RECIPE_VERSION
            assert all(counts[k] > 0 for k in counts)
            report = dict(passed=True,parameter_updates=0,oracle_sha256=digest(args.oracle),
                frozen_rollouts=rows,counts=counts,off_manager_terms_and_log_keys_equal=True,
                applied_command_active_same_tick=True,formula_bit_equal=True,weighted_peak=.5,
                checkpoint_actor_critic_optimizers_curriculum_roundtrip=True,
                cross_version_rejections=len(rejects),tight_parameter_rejections=2,
                missing_active_weight_rejected=True,export=onnx,
                interface_only_checkpoint=str(tight_path.resolve()),
                scope='Same-state rewards/registration/checkpoint compatibility, not behavior acceptance')
    finally:env.close()
    # B父模型走原生构建路径，确认加载前已注册第20项。
    cfg,agent = task_registry.get_cfgs('chuanliantui_stand_v2')
    cfg.commands['standup'].post_stand_tracking = cfg.commands['standup'].speed_curriculum = True
    cfg.commands['standup'].tracking_weight_while_active = cfg.commands['standup'].tracking_tight_term = True
    env,_ = task_registry.make_env('chuanliantui_stand_v2',env_cfg=cfg,train_cfg=agent,num_envs=4,seed=42,device=args.device)
    try:
        assert len(env.unwrapped.reward_manager.active_terms) == 20
        runner,_ = task_registry.make_alg_runner(env,'chuanliantui_stand_v2',train_cfg=agent,device=args.device)
        runner.load(str(tight_path),map_location=args.device)
        report['native_B_registration_and_load'] = True
        # 非默认参数也必须真正影响公式、manager权重和保存/恢复。
        raw, s = env.unwrapped, env.unwrapped.command_manager.get_term('standup')
        s.cfg.tracking_tight_kernel_denominator, s.cfg.tracking_tight_weight = .03, .75
        rewards.configure_tight_reward(raw.cfg,s.cfg)
        raw.reward_manager = RewardManager(raw.cfg.rewards,raw,scale_by_dt=raw.cfg.scale_rewards_by_dt)
        s.tracking_active[:] = s.reward_active[:] = True
        s.numerical_failure.zero_()
        s._command[:,0] = raw.scene['robot'].data.root_link_lin_vel_b[:,0]+.1
        raw._isaac_v2_rewards = rewards.RewardState(raw)
        raw.reward_manager.compute(dt=raw.step_dt)
        expected = torch.exp(-(s.command[:,0]-raw.scene['robot'].data.root_link_lin_vel_b[:,0]).square()/.03)*.75
        torch.testing.assert_close(raw.reward_manager._step_reward[:,-1],expected,atol=6e-8,rtol=1e-7)
        custom = args.output/'interface-only-custom-tight-v6.pt'
        runner.save(str(custom))
        metadata = torch.load(custom,map_location='cpu',weights_only=False)['infos']
        assert metadata['tracking_tight_config'] == dict(kernel_denominator=.03,weight=.75)
        s.cfg.tracking_tight_kernel_denominator, s.cfg.tracking_tight_weight = .02, .5
        rewards.restore_tracking_reward_cfg(s.cfg,metadata)
        runner.load(str(custom),map_location=args.device)
        report['nondefault_kernel_weight_formula_and_roundtrip'] = True
    finally:env.close()
    # 使用真实play参数解析/加载，仅把无限图形循环替换为16步冻结推理。
    from ct_mjlab.scripts.play import main as play_main
    def viewer(env,policy):
        def run():
            assert len(env.unwrapped.reward_manager.active_terms) == 20
            obs = env.get_observations()
            for _ in range(16):obs,_,_,_ = env.step(policy(obs))
        return SimpleNamespace(run=run)
    argv = ['play','--task','chuanliantui_stand_v2','--checkpoint',str(tight_path),
        '--num-envs','4','--viewer','viser','--post-stand-tracking','--speed-curriculum',
        '--tracking-weight-while-active','--tracking-tight-term']
    with patch.object(sys,'argv',argv),patch('mjlab.viewer.ViserPlayViewer',side_effect=viewer), \
         patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No learning')):
        play_main()
    report['real_play_cli_load_and_16_frozen_steps'] = True
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'export'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--speed-checkpoint',type=Path,required=True)
    p.add_argument('--active-checkpoint',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
