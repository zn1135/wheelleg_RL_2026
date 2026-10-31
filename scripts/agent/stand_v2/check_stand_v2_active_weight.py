#!/usr/bin/env python3
"""Reward-bit equality, command timing and version isolation; frozen policies only."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand.mdp import StandCommand
from ct_mjlab.envs.chuanliantui_stand_v2 import rewards
from ct_mjlab.envs.chuanliantui_stand_v2.export import export
from ct_mjlab.envs.chuanliantui_stand_v2.runner import task_contracts
from scripts.agent.stand_v2.check_stand_v2_speed_commands import equal_tree
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest


def equal_rewards(left, right, omit=None):
    names = [k for k in left if k != omit]
    assert left.keys() == right.keys()
    assert torch.equal(torch.stack([left[k] for k in names]), torch.stack([right[k] for k in names]))


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('ct_mjlab.envs.chuanliantui_stand_v2._previous_rewards', args.oracle)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    assert previous.TRACKING_RECIPE_VERSION == rewards.TRACKING_RECIPE_VERSION
    assert previous.WEIGHTS == {k: v for k, v in rewards.WEIGHTS.items() if k != 'tracking_lin_vel_tight'}
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand_v2')
    cfg.commands['standup'].post_stand_tracking = True
    agent.actor.distribution_cfg['std_range'] = (.05, 1e6)
    env, _ = task_registry.make_env('chuanliantui_stand_v2', env_cfg=cfg, train_cfg=agent,
        num_envs=128, seed=42, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui_stand_v2', train_cfg=agent, device=args.device)
        raw, s = env.unwrapped, env.unwrapped.command_manager.get_term('standup')
        legacy_port, current_port = previous.RewardState(raw), rewards.RewardState(raw)
        def compare(v4=False):
            original = s.cfg.tracking_weight_while_active
            try:
                s.cfg.tracking_weight_while_active = False
                legacy_port.step = current_port.step = -1
                old, off = legacy_port.compute(raw), current_port.compute(raw)
                equal_rewards(old, off)
                s.cfg.tracking_weight_while_active = True
                current_port.step = -1
                on = current_port.compute(raw)
                equal_rewards(off, on, omit='tracking_lin_vel')
                changed = s.tracking_active & (s.command[:, 0] == 0.)
                expected = torch.where(changed, off['tracking_lin_vel']*5., off['tracking_lin_vel'])
                assert torch.equal(expected, on['tracking_lin_vel'])
                if v4:
                    assert torch.equal(s.tracking_active, s.command[:, 0] != 0.)
                    equal_rewards(off, on)
                return int(changed.sum())
            finally:
                s.cfg.tracking_weight_while_active = original

        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No learning')):
            s.cfg.speed_curriculum = False
            runner.load(str(args.checkpoint), map_location=args.device)
            parent_before = tensor_digest(runner.alg.actor.state_dict())
            original_optimizers = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer, runner.alg.encoder_optimizer)]
            # Controlled current-pair cases: active zero, zero-role, recovery and ramp.
            ids = torch.arange(128, device=raw.device)
            s.cfg.speed_curriculum = True
            s.speed_curriculum_frozen = True
            s._resample_command(ids)
            s.tracking_episode[:] = ids < 64
            s.has_stood[:] = s.observation_recovered[:] = s.reward_active[:] = True
            s.numerical_failure[:] = s.fallen[:] = False
            s.tracking_ready_steps[:] = 50
            s.tracking_velocity.zero_()
            s.tracking_interval_steps[:] = 100000
            s._command[:, 0] = 0.
            with patch.object(StandCommand, '_update_command'):
                raw.common_step_counter += 1
                s._update_command(None)
                assert torch.equal(s.tracking_active, ids < 64) and not s.command[:, 0].any()
                assert compare() == 64
                s.fallen[:16] = True
                raw.common_step_counter += 1
                s._update_command(None)
                assert not s.tracking_active[:16].any() and not s.command[:16, 0].any()
                assert compare() == 48
                s.tracking_velocity[:] = .3
                raw.common_step_counter += 1
                s._update_command(None)
                assert (s.command[16:64, 0] > 0.).all()
                assert compare() == 0
            # Verify v4 -> old fixed nonzero command is equivalent to tracking_active every tick.
            rows = []
            original_compute = raw.reward_manager.compute
            captured = {}
            mode_v4 = True
            observed_zero = 0
            samples = 0
            def reward_compute(*, dt):
                nonlocal observed_zero, samples
                assert torch.equal(captured['command'], s.command)
                assert torch.equal(captured['active'], s.tracking_active)
                observed_zero += compare(v4=mode_v4)
                samples += 1
                return original_compute(dt=dt)
            raw.reward_manager.compute = reward_compute
            for speed, checkpoint, ticks in ((False,args.checkpoint,800),(True,args.speed_checkpoint,1200)):
                mode_v4 = not speed
                s.cfg.speed_curriculum = speed
                s.cfg.tracking_weight_while_active = False
                runner.load(str(checkpoint), map_location=args.device)
                model = tensor_digest(runner.alg.actor.state_dict())
                optimizers = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer, runner.alg.encoder_optimizer)]
                raw.reset()
                policy, obs = runner.get_inference_policy(device=args.device), env.get_observations()
                samples = observed_zero = 0
                for tick in range(ticks):
                    captured['command'], captured['active'] = s.command.clone(), s.tracking_active.clone()
                    obs, _, _, _ = env.step(policy(obs))
                    if (tick+1) % 400 == 0:
                        print(f'Frozen {"v6" if speed else "v4"} reward equality {(tick+1)/100:.0f}s',flush=True)
                assert samples == ticks
                if speed:
                    assert observed_zero > 0
                assert model == tensor_digest(runner.alg.actor.state_dict())
                for before, optimizer in zip(optimizers,(runner.alg.optimizer,runner.alg.encoder_optimizer)):
                    equal_tree(before,optimizer.state_dict())
                rows.append(dict(recipe='v6-speed' if speed else 'v4-fixed',ticks=ticks,
                    env_reward_samples=ticks*128,active_zero_env_samples=observed_zero,
                    old_off_bit_equal=True,other_terms_bit_equal=True,command_active_same_applied_tick=True,
                    v4_mask_equivalent=not speed,parameter_updates=0))
            raw.reward_manager.compute = original_compute
            # Save new reward checkpoint after an explicit old4999 load; it is not trained.
            s.cfg.speed_curriculum = True
            s.cfg.tracking_weight_while_active = True
            before = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer,runner.alg.encoder_optimizer)]
            curriculum = copy.deepcopy(s.checkpoint_state())
            model_before = tensor_digest(runner.alg.actor.state_dict())
            new_path = args.output/'interface-only-untrained-active-v5.pt'
            runner.save(str(new_path))
            saved = torch.load(new_path,map_location='cpu',weights_only=False)
            assert all(saved['infos'][k]==v for k,v in task_contracts(True,True,True).items())
            runner.load(str(new_path),map_location=args.device)
            assert tensor_digest(runner.alg.actor.state_dict()) == model_before
            equal_tree(curriculum,s.checkpoint_state())
            for old,opt in zip(before,(runner.alg.optimizer,runner.alg.encoder_optimizer)):
                equal_tree(old,opt.state_dict())
            rejects=[]
            for speed,active,path in ((False,False,new_path),(True,False,new_path),
                                      (False,True,args.checkpoint),(True,True,args.speed_checkpoint),
                                      (True,True,args.checkpoint)):
                s.cfg.speed_curriculum,s.cfg.tracking_weight_while_active=speed,active
                try:
                    runner.load(str(path),load_cfg={'actor':True},map_location=args.device)
                except ValueError as error:
                    assert 'recipe_version' in str(error) or 'experiment_profile_contract' in str(error)
                    rejects.append(str(error))
                else:
                    raise AssertionError('Recipe mismatch accepted')
            s.cfg.speed_curriculum=s.cfg.tracking_weight_while_active=True
            onnx = export(new_path,args.output/'interface-only-active-v5.onnx',samples=16)
            assert onnx['recipe_version']==rewards.ACTIVE_TRACKING_RECIPE_VERSION
            # Both original models remain natively loadable with flag false.
            for speed,path in ((False,args.checkpoint),(True,args.speed_checkpoint)):
                s.cfg.speed_curriculum,s.cfg.tracking_weight_while_active=speed,False
                runner.load(str(path),map_location=args.device)
            report=dict(passed=True,parameter_updates=0,oracle_sha256=digest(args.oracle),
                checkpoint_sha256=digest(args.checkpoint),speed_checkpoint_sha256=digest(args.speed_checkpoint),
                synthetic_active_zero_and_recovery_and_ramp=True,zero_role_rewards_unchanged=True,
                main_base_weight=.2,active_effective_weight=1.,kernel_denominator=.25,
                frozen_rollouts=rows,checkpoint_and_optimizer_roundtrip=True,
                mismatch_rejections=rejects,old_native_loads=True,export=onnx,
                interface_only_checkpoint=str(new_path.resolve()),
                scope='Same-state reward bit equality/timing/compatibility; no behavioral improvement claimed')
            (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k!='export'},indent=2))
    finally:
        env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--speed-checkpoint',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
