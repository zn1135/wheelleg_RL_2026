#!/usr/bin/env python3
"""Finite parking command, compatibility and frozen interface checks; no learning."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand.mdp import StandCommand
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp, rewards, observations
from ct_mjlab.envs.chuanliantui_stand_v2.export import export
from ct_mjlab.envs.chuanliantui_stand_v2.runner import task_contracts
from scripts.agent.stand_v2.check_stand_v2_yaw_control import load_oracle
from scripts.agent.stand_v2.check_stand_v2_speed_commands import equal_tree
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE = REPO/'artifacts/stand-v2-position-hold-20261029-214533/before/mjlab_training/src/ct_mjlab/envs/chuanliantui_stand_v2/mdp.py'


def disabled_equivalence(raw, cfg, oracle):
    saved_step = raw.common_step_counter
    devices = [torch.cuda.current_device()] if str(raw.device).startswith('cuda') else []
    for yaw in (False, True):
        outputs = []
        with torch.random.fork_rng(devices=devices), patch.object(StandCommand, '_update_command'):
            for cls in (oracle.StandV2Command, mdp.StandV2Command):
                torch.manual_seed(214533)
                c = copy.deepcopy(cfg); c.position_hold = False; c.yaw_control = yaw
                s = cls(c, raw); s.speed_curriculum_frozen = True
                ids = torch.arange(raw.num_envs, device=raw.device)
                s._resample_command(ids)
                s.has_stood[:] = s.observation_recovered[:] = True
                frames = []
                for tick in range(800):
                    raw.common_step_counter = tick
                    if tick == 300: s.observation_recovered[:3] = False
                    if tick == 310: s.observation_recovered[:3] = True
                    if tick == 500: s._resample_command(ids[:3])
                    s._update_command(None)
                    frames.append(torch.cat((s.command, s.tracking_velocity[:, None],
                        s.tracking_yaw_rate[:, None], s.tracking_active[:, None],
                        s.heading_target[:, None], s.heading_hold[:, None]), -1).clone())
                outputs.append((torch.stack(frames), torch.rand(32, device=raw.device)))
        assert all(torch.equal(a, b) for a,b in zip(*outputs)), f'Disabled path changed: yaw={yaw}'
    raw.common_step_counter = saved_step
    return dict(yaw_off_and_on_800_ticks_bit_equal=True, rng_bit_equal=True)


def controlled(raw, s):
    ids = torch.arange(raw.num_envs, device=raw.device)
    xy = torch.zeros((raw.num_envs, 2), device=raw.device)
    vx = torch.zeros(raw.num_envs, device=raw.device)
    heading = torch.zeros_like(vx)
    s._resample_command(ids)
    s.has_stood[:] = s.observation_recovered[:] = True
    s.fallen.zero_(); s.tracking_episode.zero_(); s.tracking_velocity.zero_()
    s.tracking_ready_steps.fill_(50); s.tracking_acceleration.fill_(1.)
    s.tracking_interval_steps.fill_(100000); s._command[:, :2] = 0.
    s.speed_curriculum_frozen = False
    speed_before = (s.speed_segments, s.speed_successes)

    def tick():
        # 本组是固定请求的确定性场景，恢复门控会清倒计时，每拍重设以隔离随机采样。
        s.tracking_interval_steps.fill_(100000)
        raw.common_step_counter += 1
        s._update_command(None)

    with patch.object(StandCommand, '_update_command'), \
         patch.object(type(raw.scene['robot'].data), 'heading_w', new_callable=PropertyMock, return_value=heading), \
         patch.object(s, '_position_measurement', side_effect=lambda: (xy, vx)):
        vx.fill_(.05)
        for _ in range(55): tick()
        assert not s.position_locked.any() and not s.position_ready_steps.any()
        vx.zero_()
        for _ in range(49):
            xy[:, 0] += .001
            tick()
        assert not s.position_locked.any()
        vx.fill_(.06); tick()
        assert not s.position_ready_steps.any()
        vx.zero_()
        for _ in range(49): tick()
        xy[:, 0] = .2; tick()
        assert s.position_locked.all() and torch.equal(s.position_anchor, xy)
        anchor = s.position_anchor.clone()
        s._update_command(None)
        assert torch.equal(s.position_anchor, anchor) and (s.position_ready_steps == 50).all()
        xy[:, 0] += .01; tick()
        assert not s.position_velocity.any()
        xy[:, 0] = anchor[:, 0]+.05; tick()
        torch.testing.assert_close(s.position_velocity, torch.full_like(vx, -.05))
        torch.testing.assert_close(s.command[:, 0], torch.full_like(vx, -.01))
        assert s.tracking_active.all() and not s.tracking_episode.any()
        assert (s.speed_segments, s.speed_successes) == speed_before and not s.segment_started.any()
        xy[:, 0] = anchor[:, 0]+1.; tick()
        torch.testing.assert_close(s.position_velocity, torch.full_like(vx, -.1))
        vx.fill_(.2); tick()
        assert s.position_locked.all(), 'Correction speed must not discard an existing anchor'
        xy.copy_(anchor); xy[:, 1] += .1; heading.fill_(math.pi/2); tick()
        torch.testing.assert_close(s.position_error_forward, torch.full_like(vx, .1), atol=1e-6, rtol=0)
        xy.copy_(anchor); xy[:, 0] += .1
        for angle in (math.pi-1e-5, -math.pi+1e-5):
            heading.fill_(angle); tick()
            torch.testing.assert_close(s.position_velocity, torch.full_like(vx, .1), atol=1e-6, rtol=0)
        heading.zero_(); xy.copy_(anchor); xy[:, 1] += .3; tick()
        assert not s.position_velocity.any()
        torch.testing.assert_close(s.position_error_lateral, torch.full_like(vx, .3))
        # 原地转期间raw vx仍为0，航向变化不丢失停车基准。
        s.tracking_episode.fill_(True); s.tracking_yaw_rate.fill_(.3)
        for _ in range(50):
            tick()
            assert not s.tracking_velocity.any() and (s.tracking_yaw_rate == .3).all()
        assert s.position_locked.all() and (s.command[:, 1] > 0.).all()
        s.tracking_velocity.fill_(.3); tick()
        assert not s.position_locked.any() and not s.position_ready_steps.any()
        s.tracking_velocity.zero_(); vx.zero_()
        for _ in range(50): tick()
        assert s.position_locked.all()
        s.observation_recovered.zero_(); tick()
        assert not s.position_locked.any() and not s.command[:, :2].any()
        s.observation_recovered.fill_(True)
        for _ in range(98): tick()
        assert not s.position_locked.any()
        tick(); assert s.position_locked.all()
        # 真实奖励读取小幅实际指令；静止角色锁定也启用现有窄核和宽核倍率。
        s.tracking_episode.zero_(); s.reward_active.fill_(True); s.numerical_failure.zero_()
        xy.copy_(s.position_anchor); xy[:, 0] += .05; tick()
        s._command[:, 0] = .05
        port = rewards.RewardState(raw)
        values = port.compute(raw)
        error = .05-raw.scene['robot'].data.root_link_lin_vel_b[:, 0]
        torch.testing.assert_close(values['tracking_lin_vel_tight'], torch.exp(-error.square()/.02))
        torch.testing.assert_close(values['tracking_lin_vel'], 5.*torch.exp(-error.square()/.25))
        torch.testing.assert_close(observations.actor_observation(raw)[:, 6], s.command[:, 0]*2.)
    untouched = {name: getattr(s, name)[3:].clone() for name in
        ('position_anchor','position_locked','position_ready_steps','position_velocity','_command')}
    s.reset(ids[:3]); s.compute(dt=0., env_ids=ids[:3])
    assert not s.position_locked[:3].any() and not s.position_ready_steps[:3].any()
    for name, value in untouched.items(): assert torch.equal(getattr(s,name)[3:], value), name
    return dict(lock_after_50_low_speed_ticks=True, threshold_strict_and_counter_resets=True,
        anchor_after_coast=True, deadzone_limit_slew=True, current_heading_projection_and_pi=True,
        lateral_diagnostic_only=True, spin_and_static_roles=True, movement_and_recovery_clear=True,
        partial_physical_reset=True, small_command_actual_reward_and_gate=True, parking_excluded_from_speed_promotion=True)


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    infos = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['infos']
    assert infos['command_contract'] == mdp.SPEED_COMMAND_CONTRACT
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand_v2', play=True)
    c = cfg.commands['standup']; c.post_stand_tracking = c.speed_curriculum = True
    rewards.restore_tracking_reward_cfg(c, infos)
    for name,value in infos['standup_curriculum']['speed_curriculum']['config'].items(): setattr(c,name,value)
    env,_ = task_registry.make_env('chuanliantui_stand_v2', env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, seed=43, device=args.device)
    output = args.output/'interface-only-untrained-position-v8.pt'
    try:
        raw=env.unwrapped; s=raw.command_manager.get_term('standup')
        runner,_=task_registry.make_alg_runner(env,'chuanliantui_stand_v2',train_cfg=agent,device=args.device)
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No learning')):
            runner.load(str(args.checkpoint),map_location=args.device)
            model_hashes=[tensor_digest(m.state_dict()) for m in (runner.alg.actor,runner.alg.critic)]
            opts=[copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer,runner.alg.encoder_optimizer)]
            legacy=disabled_equivalence(raw,s.cfg,load_oracle(args.oracle))
            s.cfg.yaw_control=True; runner.save(str(args.output/'interface-only-untrained-yaw-v7.pt'))
            s.cfg.position_hold=True; s.position_config()
            commands=controlled(raw,s)
            runner.save(str(output)); saved=torch.load(output,map_location='cpu',weights_only=False)
            expected=task_contracts(True,True,True,True,yaw=True,position=True,**s.yaw_config(),**s.position_config())
            assert all(saved['infos'][k]==v for k,v in expected.items())
            state=copy.deepcopy(s.checkpoint_state()); runner.load(str(output),map_location=args.device)
            equal_tree(state,s.checkpoint_state())
            rejected=[]
            def rejects(path):
                with patch('ct_mjlab.envs.chuanliantui_stand_v2.runner.DirectMotorOnPolicyRunner.load',side_effect=AssertionError('Mismatch reached weights')):
                    try: runner.load(str(path),map_location=args.device)
                    except ValueError as error: rejected.append(str(error))
                    else: raise AssertionError('Incompatible checkpoint accepted')
            rejects(args.checkpoint); rejects(args.output/'interface-only-untrained-yaw-v7.pt')
            s.cfg.position_hold=False; rejects(output); s.cfg.position_hold=True
            for name,value in s.position_config().items():
                setattr(s.cfg,name,value+.01); rejects(output); setattr(s.cfg,name,value)
            bad=copy.deepcopy(saved); bad['infos']['standup_curriculum']['position_hold']['position_gain']+=.1
            bad_path=args.output/'invalid-nested-position.pt'; torch.save(bad,bad_path); rejects(bad_path)
            try: export(bad_path,args.output/'must-not-export.onnx',samples=16)
            except ValueError: pass
            else: raise AssertionError('Nested mismatch exported')
            for name,bad_value in [('position_gain',0.),('position_velocity_limit',-1.),('position_tolerance',-.01),('position_lock_speed',float('nan')),('position_lock_duration',0.)]:
                old=getattr(s.cfg,name); setattr(s.cfg,name,bad_value)
                try: s.position_config()
                except ValueError: pass
                else: raise AssertionError(name)
                setattr(s.cfg,name,old)
            s.cfg.yaw_control=False
            try: s.position_config()
            except ValueError: pass
            else: raise AssertionError('Missing yaw prerequisite accepted')
            s.cfg.yaw_control=True; s.speed_curriculum_frozen=True; raw.reset()
            obs=env.get_observations(); policy=runner.get_inference_policy(device=args.device)
            locked=correction=violations=0
            for tick in range(4000):
                previous=s.command[:,0].clone(); obs,_,done,_=env.step(policy(obs))
                invalid=s.position_locked & (~s.yaw_active | (s.tracking_velocity!=0.))
                violations+=int(invalid.sum())
                violations+=int(((~done.bool()) & s.tracking_active & ((s.command[:,0]-previous).abs()>s.tracking_acceleration*.01+3e-7)).sum())
                locked+=int(s.position_locked.sum()); correction+=int((s.position_velocity!=0.).sum())
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['history']).all()
                if (tick+1)%1000==0: print(f'Frozen parking check {(tick+1)//100}/40s',flush=True)
            assert locked>0 and correction>0 and violations==0
            assert model_hashes==[tensor_digest(m.state_dict()) for m in (runner.alg.actor,runner.alg.critic)]
            for a,o in zip(opts,(runner.alg.optimizer,runner.alg.encoder_optimizer)): equal_tree(a,o.state_dict())
            onnx=export(output,args.output/'interface-only-untrained-position-v8.onnx',samples=16)
            assert onnx['command_contract']==mdp.POSITION_COMMAND_CONTRACT and 'position_hold_scope' in onnx
            report=dict(passed=True,parameter_updates=0,checkpoint_sha256=digest(args.checkpoint),
                oracle_sha256=digest(args.oracle),disabled=legacy,controlled=commands,rejections=rejected,
                frozen=dict(num_envs=args.num_envs,seconds=40,locked_env_steps=locked,correction_env_steps=correction,violations=violations),
                actor_critic_encoder_two_optimizers_unchanged=True,export=onnx,
                scope='Finite interface checks with frozen B5999; no trained parking behavior claim')
    finally: env.close()
    from ct_mjlab.scripts.play import main as play_main
    def viewer(env,policy):
        def run():
            assert env.unwrapped.command_manager.get_term('standup').cfg.position_hold
            obs=env.get_observations()
            for _ in range(16): obs,_,_,_=env.step(policy(obs))
        return SimpleNamespace(run=run)
    argv=['play','--task','chuanliantui_stand_v2','--checkpoint',str(output),'--num-envs','4','--device',args.device,
        '--viewer','viser','--post-stand-tracking','--speed-curriculum','--tracking-weight-while-active','--tracking-tight-term','--yaw-control','--position-hold']
    with patch.object(sys,'argv',argv),patch('mjlab.viewer.ViserPlayViewer',side_effect=viewer),patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No learning')): play_main()
    report['real_play_cli_16_steps']=True
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='export'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,default=DEFAULT_ORACLE); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--num-envs',type=int,default=32)
    args=parser.parse_args()
    if args.num_envs<8: parser.error('Need at least 8 environments')
    main(args)
