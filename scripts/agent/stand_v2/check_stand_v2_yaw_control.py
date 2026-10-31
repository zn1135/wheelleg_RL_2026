#!/usr/bin/env python3
"""Finite yaw/heading-hold, legacy compatibility and frozen inference checks; no learning."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand.mdp import StandCommand
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp, observations, rewards
from ct_mjlab.envs.chuanliantui_stand_v2.export import export
from ct_mjlab.envs.chuanliantui_stand_v2.runner import task_contracts
from scripts.agent.stand_v2.check_stand_v2_speed_commands import equal_tree
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest


REPO = Path(__file__).resolve().parents[3]
DEFAULT_ORACLE = REPO / 'artifacts/stand-v2-tight-effect-20261029/source/mjlab_training/src/ct_mjlab/envs/chuanliantui_stand_v2/mdp.py'
STATE_FIELDS = ('_command', 'tracking_velocity', 'tracking_yaw_rate', 'tracking_episode',
                'tracking_ready_steps', 'tracking_active', 'tracking_interval_steps',
                'heading_target', 'heading_hold', 'yaw_active')


def load_oracle(path):
    name = 'ct_mjlab.envs.chuanliantui_stand_v2._yaw_legacy_oracle'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy_equivalence(raw, cfg, oracle):
    """Compare the disabled flag with the saved pre-yaw source, including RNG use."""
    device = torch.device(raw.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    original_step = raw.common_step_counter
    outputs = []
    with torch.random.fork_rng(devices=devices), patch.object(StandCommand, '_update_command'):
        for command_type in (oracle.StandV2Command, mdp.StandV2Command):
            torch.manual_seed(20261029)
            raw.common_step_counter = 0
            legacy_cfg = copy.deepcopy(cfg)
            legacy_cfg.yaw_control = False
            s = command_type(legacy_cfg, raw)
            s.speed_curriculum_frozen = True
            ids = torch.arange(raw.num_envs, device=raw.device)
            s._resample_command(ids)
            s.has_stood[:] = s.observation_recovered[:] = True
            frames = []
            for tick in range(800):
                if tick == 300:
                    s.observation_recovered[:3] = False
                if tick == 310:
                    s.observation_recovered[:3] = True
                if tick == 500:
                    s._resample_command(ids[:3])
                raw.common_step_counter += 1
                s._update_command(None)
                frames.append(torch.cat((s.command, s.tracking_velocity[:, None],
                    s.tracking_ready_steps[:, None], s.tracking_interval_steps[:, None],
                    s.tracking_episode[:, None]), -1).clone())
            outputs.append((torch.stack(frames), torch.rand(32, device=raw.device)))
    raw.common_step_counter = original_step
    assert torch.equal(outputs[0][0], outputs[1][0]), 'Disabled yaw changed legacy commands'
    assert torch.equal(outputs[0][1], outputs[1][1]), 'Disabled yaw changed legacy RNG consumption'
    return dict(ticks=800, command_and_gate_bit_equal=True, rng_bit_equal=True)


def controlled_commands(raw, s):
    """Use the real command class with synthetic heading/gates, without changing physics."""
    n = raw.num_envs
    ids = torch.arange(n, device=raw.device)
    heading = torch.zeros(n, device=raw.device)
    requested_yaw = torch.where(ids % 2 == 0, .3, -.3)
    s._resample_command(ids)
    s.has_stood[:] = s.observation_recovered[:] = True
    s.fallen.zero_()
    s.tracking_episode.copy_(ids < n//2)
    s.zero_hold.copy_(~s.tracking_episode)
    s.tracking_acceleration.fill_(1.)
    s._command[:, :2] = 0.
    s.speed_curriculum_frozen = True
    slope_max = 0.

    def targets(selected):
        s.tracking_velocity[selected] = .3
        s.tracking_yaw_rate[selected] = requested_yaw[selected]
        s.tracking_interval_steps[selected] = 100000

    def tick():
        nonlocal slope_max
        previous = s.command[:, :2].clone()
        raw.common_step_counter += 1
        s._update_command(None)
        change = (s.command[:, :2]-previous).abs()
        active = torch.stack((s.tracking_active, s.yaw_active), -1)
        excess = change-torch.tensor((.01, s.cfg.yaw_acceleration*.01), device=raw.device)
        slope_max = max(slope_max, float(excess[active].max()) if active.any() else 0.)
        assert not ((excess > 3e-7) & active).any(), 'Active command exceeded slew limit'
        assert (s.command[:, 1].abs() <= s.cfg.yaw_rate_limit+1e-6).all()

    with patch.object(StandCommand, '_update_command'), \
         patch.object(type(raw.scene['robot'].data), 'heading_w', new_callable=PropertyMock, return_value=heading), \
         patch.object(s, '_sample_speed_target', side_effect=targets):
        for _ in range(49):
            tick()
            assert not s.command[:, :2].any() and not s.yaw_active.any()
        tick()
        assert s.yaw_active.all() and s.tracking_active[:n//2].all()
        assert not s.tracking_active[n//2:].any()
        torch.testing.assert_close(s.command[:n//2, 1], requested_yaw[:n//2].sign()*.01)
        assert not s.command[n//2:, 0].any() and s.heading_hold[n//2:].all()
        before = {name: getattr(s, name).clone() for name in STATE_FIELDS}
        s._update_command(None)
        for name, value in before.items():
            assert torch.equal(value, getattr(s, name)), f'Repeated read advanced {name}'
        for _ in range(30):
            tick()
        torch.testing.assert_close(s.command[:n//2, 1], requested_yaw[:n//2])

        # 静止角色允许航向纠偏，平移命令仍须保持零。
        heading[:n//2] = .8
        heading[n//2:] = .2
        tick()
        assert (s.command[n//2:, 1] < 0.).all() and not s.command[n//2:, 0].any()
        s.tracking_yaw_rate.zero_()
        tick()
        assert s.heading_hold.all()
        torch.testing.assert_close(s.heading_target[:n//2], heading[:n//2])
        assert not s.heading_target[n//2:].any(), 'Hold reference followed heading drift'
        for _ in range(35):
            tick()
        assert s.command[:n//2, 1].abs().max() < 1e-6

        # 穿过±pi时走短方向；实际命令先经过限幅与限斜率。
        s.heading_target[:2] = torch.tensor((math.pi-.02, -math.pi+.02), device=raw.device)
        heading[:2] = torch.tensor((-math.pi+.02, math.pi-.02), device=raw.device)
        s._command[:2, 1] = 0.
        for _ in range(8):
            tick()
        torch.testing.assert_close(s.command[:2, 1], torch.tensor((-.06, .06), device=raw.device), atol=2e-6, rtol=0.)
        s.heading_target[:2] = torch.tensor((1., -1.), device=raw.device)
        heading[:2] = torch.tensor((-1., 1.), device=raw.device)
        for _ in range(60):
            tick()
        torch.testing.assert_close(s.command[:2, 1], torch.tensor((.5, -.5), device=raw.device))

        s.observation_recovered[:] = False
        tick()
        assert not s.command[:, :2].any() and not s.tracking_yaw_rate.any()
        assert not s.heading_hold.any() and not s.yaw_active.any()
        assert not s.tracking_ready_steps.any()
        heading.fill_(1.2)
        requested_yaw.zero_()
        s.observation_recovered[:] = True
        for _ in range(49):
            tick()
            assert not s.command[:, :2].any()
        tick()
        assert s.heading_hold.all() and s.yaw_active.all()
        assert not s.command[:, 1].any()
        torch.testing.assert_close(s.heading_target, heading)

    # 环境复位在reset后另调compute(dt=0)生成新命令；这里使用真实父类更新。
    before = {name: getattr(s, name)[3:].clone() for name in STATE_FIELDS}
    s.reset(ids[:3])
    s.compute(dt=0., env_ids=ids[:3])
    for name, value in before.items():
        assert torch.equal(value, getattr(s, name)[3:]), f'Partial reset changed other environments: {name}'
    assert not s.command[:3, :2].any() and not s.heading_hold[:3].any()
    assert not s.yaw_active[:3].any() and not s.tracking_yaw_rate[:3].any()
    return dict(gate_49_50=True, rate_sign_and_slew=True, repeat_read=True, static_heading_correction=True,
        release_latches_current_heading=True, wrap_pi_short_direction=True, correction_saturation=True,
        lost_gate_clears_and_relocks=True, partial_reset=True, max_slew_excess=slope_max)


def sampling_and_reward(raw, s):
    ids = torch.arange(raw.num_envs, device=raw.device)
    counts = torch.zeros(4, dtype=torch.long, device=raw.device)
    roles = 0
    signs = set()
    for _ in range(64):
        s._resample_command(ids)
        roles += int(s.tracking_episode.sum())
        s._sample_speed_target(ids)
        kind = s.tracking_transition
        counts += torch.bincount(kind, minlength=4)
        assert torch.equal(s.tracking_velocity != 0., (kind == 1) | (kind == 3))
        assert torch.equal(s.tracking_yaw_rate != 0., (kind == 2) | (kind == 3))
        assert (s.tracking_yaw_rate.abs() <= s.cfg.yaw_rate_limit).all()
        signs.update(s.tracking_yaw_rate.sign().cpu().tolist())
    total = raw.num_envs*64
    assert .40 < roles/total < .60 and signs == {-1., 0., 1.}
    assert ((counts/total > .18) & (counts/total < .32)).all()

    s.has_stood[:] = s.reward_active[:] = s.tracking_active[:] = True
    s.numerical_failure.zero_()
    actual_yaw = raw.scene['robot'].data.root_link_ang_vel_b[:, 2]
    s._command[:, 0] = .2
    s._command[:, 1] = actual_yaw+.2
    actor = observations.actor_observation(raw)
    torch.testing.assert_close(actor[:, 6], s.command[:, 0]*2.)
    torch.testing.assert_close(actor[:, 7], s.command[:, 1]*.25)
    port = rewards.RewardState(raw)
    def values():
        port.step = -1
        return {k: v.clone() for k, v in port.compute(raw).items()}
    original = values()
    expected = torch.exp(-(s.command[:, 1]-actual_yaw).square()/.25)
    torch.testing.assert_close(original['tracking_ang_vel'], expected)
    s.tracking_yaw_rate.fill_(-.5)
    s.heading_target.add_(1.)
    changed_target = values()
    assert all(torch.equal(value, changed_target[name]) for name, value in original.items())
    s.cfg.yaw_control = False
    disabled = values()
    assert all(torch.equal(value, disabled[name]) for name, value in original.items())
    s.cfg.yaw_control = True
    return dict(samples=total, moving_role_fraction=roles/total, category_counts=counts.cpu().tolist(),
        observation_command_scales=True, yaw_reward_formula=True, reward_uses_applied_command=True,
        same_command_rewards_unchanged=True)


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = load_oracle(args.oracle)
    infos = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['infos']
    assert infos['command_contract'] == mdp.SPEED_COMMAND_CONTRACT
    assert infos['recipe_version'] == rewards.TIGHT_TRACKING_RECIPE_VERSION
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand_v2', play=True)
    command = cfg.commands['standup']
    command.post_stand_tracking = command.speed_curriculum = True
    rewards.restore_tracking_reward_cfg(command, infos)
    for name, value in infos['standup_curriculum']['speed_curriculum']['config'].items():
        setattr(command, name, value)
    env, _ = task_registry.make_env('chuanliantui_stand_v2', env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, seed=43, device=args.device)
    output_checkpoint = args.output/'interface-only-untrained-yaw-v7.pt'
    try:
        raw, s = env.unwrapped, env.unwrapped.command_manager.get_term('standup')
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui_stand_v2', train_cfg=agent, device=args.device)
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer updates')):
            # 先完整加载旧B，随后显式打开新接口；该产物没有转向训练。
            runner.load(str(args.checkpoint), map_location=args.device)
            models = [tensor_digest(m.state_dict()) for m in (runner.alg.actor, runner.alg.critic)]
            optimizers = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer, runner.alg.encoder_optimizer)]
            legacy = legacy_equivalence(raw, s.cfg, oracle)
            s.cfg.yaw_control = True
            s.yaw_config()
            controlled = controlled_commands(raw, s)
            formula = sampling_and_reward(raw, s)

            # 每个参数变化都必须被checkpoint严格检查；不能靠形状相同混载。
            curriculum = copy.deepcopy(s.checkpoint_state())
            runner.save(str(output_checkpoint))
            saved = torch.load(output_checkpoint, map_location='cpu', weights_only=False)['infos']
            expected = task_contracts(True, True, True, True, yaw=True, **s.yaw_config())
            assert all(saved[k] == value for k, value in expected.items())
            assert saved['standup_curriculum']['yaw_control'] == s.yaw_config()
            runner.load(str(output_checkpoint), map_location=args.device)
            equal_tree(curriculum, s.checkpoint_state())
            rejections = []
            for enabled, path in ((True, args.checkpoint), (False, output_checkpoint)):
                s.cfg.yaw_control = enabled
                try:
                    runner.load(str(path), map_location=args.device)
                except ValueError as error:
                    rejections.append(str(error))
                else:
                    raise AssertionError('Cross-command checkpoint accepted')
            s.cfg.yaw_control = True
            for name, value in (('yaw_rate_limit', .3), ('yaw_acceleration', 2.), ('heading_gain', 2.)):
                previous = getattr(s.cfg, name)
                setattr(s.cfg, name, value)
                try:
                    runner.load(str(output_checkpoint), map_location=args.device)
                except ValueError as error:
                    assert 'yaw' in str(error).lower()
                    rejections.append(str(error))
                else:
                    raise AssertionError(f'Yaw configuration mismatch accepted: {name}')
                setattr(s.cfg, name, previous)
            for name, value in (('yaw_rate_limit', .05), ('yaw_acceleration', 0.), ('heading_gain', float('nan'))):
                previous = getattr(s.cfg, name)
                setattr(s.cfg, name, value)
                try:
                    s.yaw_config()
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Invalid yaw configuration accepted: {name}')
                setattr(s.cfg, name, previous)
            for name in ('post_stand_tracking', 'speed_curriculum'):
                setattr(s.cfg, name, False)
                try:
                    s.yaw_config()
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Missing yaw prerequisite accepted: {name}')
                setattr(s.cfg, name, True)
            assert task_contracts(False)['command_contract'] == mdp.ZERO_COMMAND_CONTRACT
            assert task_contracts(True)['command_contract'] == mdp.TRACKING_COMMAND_CONTRACT
            for name, value in (('yaw_rate_limit', .05), ('yaw_acceleration', 0.), ('heading_gain', float('nan'))):
                config = s.yaw_config()
                config[name] = value
                try:
                    task_contracts(True, True, True, True, yaw=True, **config)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Invalid contract parameter accepted: {name}')
            # 顶层参数正确、嵌套课程参数被篡改，也必须在加载任何权重之前拒绝。
            tampered = torch.load(output_checkpoint, map_location='cpu', weights_only=False)
            tampered['infos']['standup_curriculum']['yaw_control']['heading_gain'] += .1
            tampered_path = args.output/'interface-only-invalid-nested-yaw.pt'
            torch.save(tampered, tampered_path)
            with patch('ct_mjlab.envs.chuanliantui_stand_v2.runner.DirectMotorOnPolicyRunner.load',
                       side_effect=AssertionError('Nested mismatch reached weight loading')):
                try:
                    runner.load(str(tampered_path), map_location=args.device)
                except ValueError as error:
                    assert 'yaw' in str(error).lower()
                    rejections.append(str(error))
                else:
                    raise AssertionError('Nested yaw configuration mismatch accepted')
            try:
                export(tampered_path, args.output/'must-not-export-invalid-yaw.onnx', samples=16)
            except ValueError as error:
                assert 'yaw' in str(error).lower()
            else:
                raise AssertionError('Export accepted nested yaw configuration mismatch')

            # 真实物理仅冻结推理40秒，验证生产路径能实际发出转向命令。
            s.speed_curriculum_frozen = True
            raw.reset()
            obs, policy = env.get_observations(), runner.get_inference_policy(device=args.device)
            yaw_samples = static_correction_samples = gate_violations = slew_violations = 0
            for tick in range(4000):
                previous = s.command[:, :2].clone()
                obs, _, done, _ = env.step(policy(obs))
                nonzero_yaw = s.command[:, 1] != 0.
                invalid = nonzero_yaw & (~s.has_stood | s.recover | s.fallen | (s.tracking_ready_steps < 50))
                gate_violations += int(invalid.sum())
                delta = (s.command[:, :2]-previous).abs()
                # 复位与安全清零不属于正常限斜率过程。
                live = ~done.bool()
                slew_violations += int((live & s.yaw_active & (delta[:, 1] > s.cfg.yaw_acceleration*.01+3e-7)).sum())
                slew_violations += int((live & s.tracking_active & (delta[:, 0] > s.tracking_acceleration*.01+3e-7)).sum())
                assert not s.command[~s.tracking_episode, 0].any()
                yaw_samples += int(nonzero_yaw.sum())
                static_correction_samples += int((nonzero_yaw & ~s.tracking_episode).sum())
                torch.testing.assert_close(obs['actor'][:, 7], s.command[:, 1]*.25)
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['history']).all()
                if (tick+1) % 1000 == 0:
                    print(f'Frozen yaw interface check {(tick+1)//100}/40s', flush=True)
            assert yaw_samples > 0 and static_correction_samples > 0
            assert gate_violations == slew_violations == 0
            assert models == [tensor_digest(m.state_dict()) for m in (runner.alg.actor, runner.alg.critic)]
            for expected_opt, optimizer in zip(optimizers, (runner.alg.optimizer, runner.alg.encoder_optimizer)):
                equal_tree(expected_opt, optimizer.state_dict())
            onnx = export(output_checkpoint, args.output/'interface-only-untrained-yaw-v7.onnx', samples=16)
            assert onnx['command_contract'] == mdp.YAW_COMMAND_CONTRACT
            assert onnx['yaw_control_config'] == s.yaw_config()
            report = dict(passed=True, parameter_updates=0, checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=digest(args.checkpoint), oracle=str(args.oracle.resolve()), oracle_sha256=digest(args.oracle),
                legacy_disabled=legacy, controlled=controlled, sampling_and_rewards=formula,
                frozen_rollout=dict(conditions='nominal', num_envs=raw.num_envs, seed=43, seconds=40.,
                    nonzero_yaw_samples=yaw_samples, static_heading_correction_samples=static_correction_samples,
                    gate_violations=gate_violations, slew_violations=slew_violations),
                checkpoint_actor_critic_two_optimizers_unchanged=True, curriculum_roundtrip=True,
                cross_version_and_parameter_rejections=rejections, invalid_config_and_prerequisites_rejected=True,
                nested_yaw_mismatch_rejected_before_weight_load_and_export=True,
                export=onnx, interface_only_checkpoint=str(output_checkpoint.resolve()),
                scope='Finite command/heading/interface checks with frozen B5999; not trained yaw behavior acceptance')
    finally:
        env.close()

    # 使用真实CLI加载与16步推理，替换无限viewer，不打开服务。
    from ct_mjlab.scripts.play import main as play_main
    def viewer(env, policy):
        def run():
            state = env.unwrapped.command_manager.get_term('standup')
            assert state.cfg.yaw_control and state.speed_curriculum_frozen
            obs = env.get_observations()
            for _ in range(16):
                obs, _, _, _ = env.step(policy(obs))
        return SimpleNamespace(run=run)
    argv = ['play', '--task', 'chuanliantui_stand_v2', '--checkpoint', str(output_checkpoint),
        '--num-envs', '4', '--device', args.device, '--viewer', 'viser', '--post-stand-tracking',
        '--speed-curriculum', '--tracking-weight-while-active', '--tracking-tight-term', '--yaw-control']
    with patch.object(sys, 'argv', argv), patch('mjlab.viewer.ViserPlayViewer', side_effect=viewer), \
         patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer updates')):
        play_main()
    report['real_play_cli_load_and_16_frozen_steps'] = True
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k != 'export'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Existing B5999 speed/tight checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--oracle', type=Path, default=DEFAULT_ORACLE,
        help='Saved pre-yaw mdp.py; defaults to the B5999 evaluation source snapshot')
    args = parser.parse_args()
    if args.num_envs < 8 or args.num_envs % 2:
        parser.error('num-envs must be even and at least8')
    main(args)
