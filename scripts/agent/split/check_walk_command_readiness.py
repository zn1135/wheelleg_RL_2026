#!/usr/bin/env python3
"""Walk-native command gate: synthetic boundaries, archived regression and noisy inference; no learning."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import mdp, runner, rewards, observations
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.switch import conditions
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp as stand, rewards as stand_rewards
from scripts.agent.split.check_chuanliantui_two_policy import digest, rejected, tensors_equal


def archived(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # 读取训练快照时禁止写入logs里的pyc缓存。
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@torch.inference_mode()
def fixtures(device):
    env, _ = task_registry.make_env('chuanliantui_walk', play=True, num_envs=4, seed=43, device=device)
    raw = env.unwrapped
    s = raw.command_manager.get_term('standup')
    yes = torch.ones(4, dtype=torch.bool, device=device); no = ~yes
    ids = torch.arange(4, device=device)
    down = no.clone(); emergency = no.clone(); finite = yes.clone()
    calls = 0

    def physical(_):
        nonlocal calls
        calls += 1
        # 掉载只改轮载荷，其余物理输入保持直立，使用生产down判据。
        support = torch.full((4, 2), 50., device=device)
        support[down, 0] = 0.
        ready, actual_down, _, _ = conditions(-torch.ones(4, device=device),
            torch.full((4,), .25, device=device), torch.zeros(4, device=device),
            torch.zeros(4, device=device), support, torch.full((4,), 100., device=device), finite)
        return ready, actual_down, emergency, finite

    def tick(age, reverse=False):
        raw.common_step_counter += 1
        raw.episode_length_buf.fill_(age)
        previous_calls = calls
        if reverse:
            s._update_command(None); failed = mdp.failure(raw)
        else:
            failed = mdp.failure(raw); s._update_command(None)
        count = s.walk_down_ticks.clone(); wait = s.tracking_ready_steps.clone()
        mdp.failure(raw); s._update_command(None)
        assert torch.equal(count, s.walk_down_ticks) and torch.equal(wait, s.tracking_ready_steps)
        assert calls == previous_calls + 1
        return failed

    try:
        # 屏蔽继承终止函数只为隔离门控；真正walk终止兼容性另见正式双策略检查。
        with patch.object(stand, 'failure', return_value=no), patch.object(mdp, 'physical_conditions', side_effect=physical):
            s.has_stood.zero_(); s.observation_recovered.zero_(); s.fallen.fill_(True)
            for age in range(1, 100):
                tick(age)
                assert not s.walk_ready.any() and not s.tracking_ready_steps.any()
                assert not s.command[:, :2].any()
            tick(100)
            assert s.walk_ready.all() and (s.tracking_ready_steps == 1).all()
            # 连续1～9拍掉载期间，等待继续累计；恢复后down计数归零。
            down[:] = True
            for n in range(1, 10):
                tick(100+n, reverse=n % 2 == 0)
                assert s.walk_ready.all() and (s.walk_down_ticks == n).all()
                assert (s.tracking_ready_steps == n+1).all()
            down.zero_(); tick(110)
            assert not s.walk_down_ticks.any()
            for age in range(111, 149):tick(age)
            assert (s.tracking_ready_steps == 49).all() and not s.yaw_active.any()
            tick(149)
            assert (s.tracking_ready_steps == 50).all() and s.yaw_active.all()
            s.tracking_episode.zero_(); s.tracking_velocity.zero_()
            s.position_locked.fill_(True); s.position_anchor.copy_(raw.scene['robot'].data.root_link_pos_w[:, :2])
            s.heading_hold.fill_(True); s.heading_target.fill_(.3)
            anchor = s.position_anchor.clone(); heading = s.heading_target.clone()
            down.fill_(True)
            for n in range(1, 10):
                tick(149+n)
                assert s.position_locked.all() and torch.equal(s.position_anchor, anchor)
                assert s.heading_hold.all() and torch.equal(s.heading_target, heading)
            down.zero_(); tick(159)
            # 明确构造运行中的速度段和停车/航向参考，确认掉载不触发lost。
            s.tracking_episode.fill_(True); s.tracking_velocity.fill_(1.)
            s.tracking_interval_steps.fill_(200); s.segment_started.fill_(True)
            s.segment_direction.zero_(); s.segment_stage.fill_(s.direction_curriculum['forward']['stage']); s.segment_steps.fill_(70)
            s.segment_error.zero_(); s.segment_failed.zero_(); s.speed_curriculum_frozen = False
            segments = s.direction_curriculum['forward']['segments']
            s.heading_hold.fill_(True); s.heading_target.fill_(.3)
            down[:] = True
            waiting = s.tracking_ready_steps.clone()
            for n in range(1, 10):
                assert not tick(159+n).any()
                assert s.walk_ready.all() and s.tracking_active.all() and s.yaw_active.all()
                assert torch.equal(s.tracking_ready_steps, waiting+n)
                assert s.segment_started.all() and s.direction_curriculum['forward']['segments'] == segments
            s.position_anchor.fill_(3.); s.position_locked.fill_(True)
            assert tick(169).all()
            assert not s.walk_ready.any() and not s.tracking_ready_steps.any()
            assert not s.tracking_active.any() and not s.yaw_active.any()
            assert not s.position_locked.any() and not s.position_anchor.any()
            assert not s.heading_hold.any() and not s.heading_target.any()
            assert not s.tracking_velocity.any() and not s.tracking_yaw_rate.any()
            assert not s._speed_previous_command.any() and not s._yaw_previous_command.any()
            assert not s.command[:, :2].any() and s.direction_curriculum['forward']['segments'] == segments+4
            assert s.direction_curriculum['forward']['successes'] == 0
            # 单拍、1～9拍测试已覆盖；急停与非有限状态无需等100ms。
            for kind in ('emergency', 'nonfinite'):
                down.zero_(); emergency.zero_(); finite.fill_(True)
                for age in range(160, 211):tick(age)
                if kind == 'emergency':emergency[0] = True
                else:finite[0] = False
                assert tick(211)[0] and not s.walk_ready[0] and not s.command[0, :2].any()
            down.zero_(); emergency.zero_(); finite.fill_(True)
            tick(212)
            s.walk_down_ticks[:] = torch.tensor([1, 2, 3, 4], device=device)
            before = {name: getattr(s, name)[1:].clone() for name in
                ('walk_down_ticks', 'walk_ready', 'tracking_ready_steps', 'position_anchor', 'heading_target')}
            s.reset(ids[:1])
            assert not s.walk_ready[0] and s.walk_down_ticks[0] == s.tracking_ready_steps[0] == 0
            assert all(torch.equal(getattr(s, name)[1:], value) for name, value in before.items())
            # 同拍复位不再次累计，下一拍按各自episode时钟重新武装。
            s._update_command(None)
            assert not s.walk_ready[0]
            raw.common_step_counter += 1; raw.episode_length_buf[0] = 0
            mdp.failure(raw); s._update_command(None)
            assert not s.walk_ready[0] and s.walk_ready[1:].all()
        return dict(before_1s_false=True, wait_50_ticks=True, support_dips_1_to_9_ticks_preserved=True, parking_and_heading_anchors_preserved=True,
            sustained_down_10_ticks_clears_commands_and_references=True, segment_lost_only_at_sustained_down=True,
            emergency_and_nonfinite_immediate=True, partial_reset_isolated=True,
            evaluation_orders_and_duplicate_calls_count_once=True)
    finally:
        env.close()


@torch.inference_mode()
def regression(snapshot, device, contract_before=None, shared_action_before=None, shared_rewards_before=None):
    root = Path('mjlab_training/src/ct_mjlab')
    unchanged = {}
    # 旧任务实现及其配置/奖励/观测源码逐字节核对，不把共享代码静默替换成当前版本。
    for folder in ('chuanliantui', 'chuanliantui_stand', 'chuanliantui_stand_v2'):
        for path in (snapshot/'envs'/folder).glob('*.py'):
            relative = path.relative_to(snapshot)
            baseline = (shared_action_before if shared_action_before is not None
                        and str(relative) == 'envs/chuanliantui/huananhu_action.py' else path)
            if shared_rewards_before is not None and str(relative) == 'envs/chuanliantui_stand_v2/rewards.py':
                baseline = shared_rewards_before
            assert baseline.read_bytes() == (root/relative).read_bytes(), str(relative)
            unchanged[str(relative)] = digest(baseline)
    for name in ('network.py', 'observations.py'):
        relative = Path('envs/chuanliantui_split')/name
        assert (snapshot/relative).read_bytes() == (root/relative).read_bytes(), str(relative)
        unchanged[str(relative)] = digest(snapshot/relative)
    old_env_cfg = archived(snapshot/'envs/chuanliantui_split/env_cfg.py',
                           'ct_mjlab.envs.chuanliantui_split._direction_cfg_before')
    assert runner.contracts(old_env_cfg.command_cfg('getup')) == runner.contracts(command_cfg('getup'))
    if contract_before is None:
        old_runner = archived(snapshot/'envs/chuanliantui_split/runner.py',
                              'ct_mjlab.envs.chuanliantui_split._ready_runner_before')
        assert old_runner.contracts(command_cfg('getup')) == runner.contracts(command_cfg('getup'))
    else:
        baseline = json.loads(contract_before.read_text())['getup']
        assert baseline == json.loads(json.dumps(runner.contracts(command_cfg('getup'))))
    old_split = archived(snapshot/'envs/chuanliantui_split/mdp.py',
                         'ct_mjlab.envs.chuanliantui_split._ready_before')
    old_stand = archived(snapshot/'envs/chuanliantui_stand_v2/mdp.py',
                         'ct_mjlab.envs.chuanliantui_stand_v2._ready_before')
    old_rewards = archived(snapshot/'envs/chuanliantui_stand_v2/rewards.py',
                           'ct_mjlab.envs.chuanliantui_stand_v2._ready_rewards_before')
    split_rewards_before = (contract_before.parent/'rewards.py' if shared_rewards_before is not None
                            else snapshot/'envs/chuanliantui_split/rewards.py')
    old_split_rewards = archived(split_rewards_before,
                                'ct_mjlab.envs.chuanliantui_split._ready_rewards_before')
    assert old_split_rewards.GETUP_WEIGHTS == rewards.GETUP_WEIGHTS
    assert old_split_rewards.GETUP_REWARD_RECIPE == rewards.GETUP_REWARD_RECIPE
    report = {}
    for task, classes, weight_modules in (
        ('chuanliantui_getup', (old_split.SplitCommand, mdp.SplitCommand), (old_split_rewards, rewards)),
        ('chuanliantui_stand_v2', (old_stand.StandV2Command, stand.StandV2Command), (old_rewards, stand_rewards))):
        env, _ = task_registry.make_env(task, play=True, num_envs=4, seed=43, device=device)
        raw = env.unwrapped; original = raw.command_manager.get_term('standup'); captures = []
        cfg = copy.deepcopy(original.cfg)
        if task.endswith('stand_v2'):
            cfg.post_stand_tracking = cfg.speed_curriculum = cfg.yaw_control = cfg.position_hold = True
        try:
            for cls, reward_module in zip(classes, weight_modules):
                torch.manual_seed(43)
                s = cls(copy.deepcopy(cfg), raw); raw.command_manager._terms['standup'] = s
                if hasattr(raw, '_isaac_v2_rewards'):del raw._isaac_v2_rewards
                ids = torch.arange(4, device=device); s.reset(ids); frames = []
                raw.common_step_counter = 0
                for step in range(160):
                    raw.common_step_counter = step+1
                    raw.episode_length_buf.fill_(step+1)
                    if step == 80:s.reset(ids[:1])
                    # 相同物理输入和锁存输入，覆盖就绪、失稳、复位与重新就绪路径。
                    good = step < 80 or step >= 90
                    s.has_stood.fill_(good); s.observation_recovered.fill_(good); s.fallen.fill_(not good)
                    s.standing_time.fill_(.5 if good else 0.)
                    s.reward_active.fill_(True); s.numerical_failure.zero_()
                    s._update_command(None)
                    values = {name: reward_module.reward(raw, name=name).clone() for name in raw.cfg.rewards}
                    frames.append(dict(command=s.command.clone(), rewards=values))
                captures.append(dict(frames=frames, cpu_rng=torch.get_rng_state(),
                    device_rng=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None))
            assert tensors_equal(*captures), task
            report[task] = dict(fixed_input_ticks=160, envs=4, commands_rewards_rng_bitwise_equal=True)
        finally:
            raw.command_manager._terms['standup'] = original
            env.close()
    report['unchanged_legacy_source_sha256'] = unchanged
    report['shared_action_baseline'] = str(shared_action_before) if shared_action_before else str(snapshot)
    return report


def getup_load_check(built, checkpoint, baseline_contracts, fixture, device):
    infos = torch.load(checkpoint, map_location='cpu', weights_only=False)['infos']
    expected = runner.contracts(command_cfg('getup'))
    mismatches = [key for key, value in expected.items() if infos.get(key) != value]
    if not mismatches:
        built.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=device)
        return dict(requested_checkpoint_accepted=True)
    # 并行会话已有配方改动不能由walk检查回退；只接受能证明在本任务前已有的差异。
    assert baseline_contracts is not None
    baseline = json.loads(baseline_contracts.read_text())['getup']
    assert baseline == json.loads(json.dumps(expected))
    assert any(infos.get(key) != value for key, value in baseline.items())
    rejected(lambda:built.load(str(checkpoint), map_location=device))
    built.save(str(fixture)); built.load(str(fixture), map_location=device)
    return dict(requested_checkpoint_accepted=False, preexisting_incompatibility=mismatches,
                expected_recipe=expected['reward_recipe'], requested_recipe=infos.get('reward_recipe'),
                current_untrained_fixture_strict_round_trip=True, fixture=str(fixture))


@torch.inference_mode()
def noise_check(args, std):
    cfg, train = task_registry.get_cfgs('chuanliantui_walk', play=False); cfg.seed = 43
    env, _ = task_registry.make_env('chuanliantui_walk', env_cfg=cfg, train_cfg=train, play=False,
                                  num_envs=args.num_envs, device=args.device)
    try:
        built, _ = task_registry.make_alg_runner(env, 'chuanliantui_walk', train_cfg=train, device=args.device)
        saved = torch.load(args.walk_checkpoint, map_location=args.device, weights_only=False)
        rejected(lambda: built.load(str(args.walk_checkpoint), load_cfg={'actor': True}, map_location=args.device))
        # 只读旧权重做诊断；正式runner仍严格拒绝v7，不修改checkpoint契约。
        built.alg.actor.load_state_dict(saved['actor_state_dict'], strict=True)
        actor_before = copy.deepcopy(built.alg.actor.state_dict())
        policy = built.get_inference_policy(device=args.device)
        raw = env.unwrapped; s = raw.command_manager.get_term('standup'); s.speed_curriculum_frozen = True
        obs = env.get_observations(); torch.manual_seed(0)
        sums = dict(support=0., has_stood=0., walk_ready=0., tracking_active=0., nonzero_vx_cmd=0.)
        count = 0; done_count = 0
        for tick in range(args.steps):
            action = policy(obs) + std*torch.randn(raw.num_envs, 6, device=args.device)
            obs, reward, done, _ = env.step(action)
            assert torch.isfinite(reward).all()
            done_count += int(done.sum())
            if tick < round(2./raw.step_dt):continue
            values = dict(support=stand.normal_wheel_support(raw), has_stood=s.has_stood,
                walk_ready=s.walk_ready, tracking_active=s.tracking_active,
                nonzero_vx_cmd=s.command[:, 0].abs() > .05)
            for name, value in values.items():sums[name] += float(value.float().mean())
            count += 1
        fractions = {name: value/count for name, value in sums.items()}
        assert fractions['tracking_active'] > .5, fractions
        assert fractions['nonzero_vx_cmd'] > .05, fractions
        assert tensors_equal(actor_before, built.alg.actor.state_dict())
        return dict(noise_std=std, num_envs=args.num_envs, seconds=args.steps*raw.step_dt,
                    warmup_seconds=2., measured_ticks=count, fractions=fractions,
                    episode_terminations=done_count, actor_weights_bitwise_unchanged=True)
    finally:
        env.close()


def main(args):
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    assert args.num_envs >= 128 and args.steps >= 1000
    report = dict(verified=False, parameter_updates=0, snapshot=str(args.snapshot),
        walk_checkpoint=str(args.walk_checkpoint), getup_checkpoint=str(args.getup_checkpoint),
        walk_checkpoint_sha256=digest(args.walk_checkpoint), getup_checkpoint_sha256=digest(args.getup_checkpoint),
        old_gate_user_reference=dict(warmup_seconds=1., noise0=dict(support=.992, has_stood=.969,
            tracking_active=.778, nonzero_vx_cmd=.388), noise02=dict(support=.766, has_stood=0.,
            tracking_active=0., nonzero_vx_cmd=0.)), scope='仅软件/合成检查，未训练、未行为验收')
    try:
        report['fixtures'] = fixtures(args.device)
        report['regression'] = regression(args.snapshot, args.device, args.getup_contract_before, args.shared_action_before, args.shared_rewards_before)
        rejected(lambda: runner.validate(torch.load(args.walk_checkpoint, map_location='cpu',
            weights_only=False)['infos'], command_cfg('walk')))
        # 真正通过runner严格载入getup，而不只检查元数据。
        env, _ = task_registry.make_env('chuanliantui_getup', play=True, num_envs=4, device=args.device)
        try:
            built, _ = task_registry.make_alg_runner(env, 'chuanliantui_getup', device=args.device)
            getup_status = getup_load_check(built, args.getup_checkpoint, args.getup_contract_before, args.output/'current-untrained-getup.pt', args.device)
        finally:env.close()
        report['contracts'] = dict(previous_walk_rejected=True, getup=getup_status,
                                   walk=runner.contracts(command_cfg('walk')))
        report['noise'] = [noise_check(args, std) for std in (0., .2)]
        report['verified'] = True
    finally:
        report['source_sha256'] = digest(__file__)
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'regression'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walk-checkpoint', type=Path, required=True)
    parser.add_argument('--getup-checkpoint', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--shared-rewards-before', type=Path, help='Pre-task shared rewards when getup changed independently after the training snapshot')
    parser.add_argument('--shared-action-before', type=Path, help='Baseline for an existing independent shared action change; other legacy files still use snapshot')
    parser.add_argument('--getup-contract-before', type=Path, help='Pre-task contract JSON when getup has newer independent changes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--steps', type=int, default=1000)
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Readiness check cannot train')):
        main(parser.parse_args())
