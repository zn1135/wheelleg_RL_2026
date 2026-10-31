#!/usr/bin/env python3
"""Walk v8 direction sampling, independent promotion/resume and frozen inference; never learns."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import runner
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from scripts.agent.split.check_chuanliantui_two_policy import digest, rejected, tensors_equal
from scripts.agent.split.check_walk_command_readiness import regression, getup_load_check

DIRECTIONS = ('forward', 'backward')


def fresh_state():
    return {d:dict(stage=0, segments=0, successes=0, last_success_fraction=0.) for d in DIRECTIONS}


@torch.inference_mode()
def sampling_checks(raw):
    s = raw.command_manager.get_term('standup'); ids = torch.arange(raw.num_envs, device=raw.device)
    assert s.cfg.forward_speed_limits == (.4, .6, .8, 1., 1.5)
    assert s.cfg.backward_speed_limits == (1., 1.5)
    assert s.cfg.forward_speed_min == .3 and s.cfg.backward_speed_min == 1.
    assert s.cfg.speed_error_threshold == .1
    previous = copy.deepcopy(s.direction_curriculum); rows = []
    try:
        for forward in range(5):
            for backward in range(2):
                s.direction_curriculum['forward']['stage'] = forward
                s.direction_curriculum['backward']['stage'] = backward
                samples = []; kinds = set()
                for _ in range(128):
                    s._sample_speed_target(ids)
                    target = s.tracking_velocity.clone(); kind = s.tracking_transition
                    kinds.update(kind.tolist()); samples.append(target)
                    assert not target[kind != 1].any() and not s.tracking_yaw_rate[kind == 1].any()
                    for index, direction in enumerate(DIRECTIONS):
                        selected = (kind == 1) & (s.segment_direction == index)
                        state = s.direction_curriculum[direction]
                        maximum = getattr(s.cfg, direction+'_speed_limits')[state['stage']]
                        minimum = getattr(s.cfg, direction+'_speed_min')
                        values = target[selected].abs()
                        assert ((values >= minimum) & (values <= maximum)).all()
                        assert torch.equal(s.segment_started[selected], values >= .8*maximum)
                        assert (s.segment_stage[selected] == state['stage']).all()
                values = torch.cat(samples)
                assert kinds == {0, 1, 2} and (values > 0).any() and (values < 0).any()
                summary = {}
                for direction, selected in (('forward', values > 0), ('backward', values < 0)):
                    x = values[selected].abs(); stage = s.direction_curriculum[direction]['stage']
                    lower = getattr(s.cfg, direction+'_speed_min')
                    upper = getattr(s.cfg, direction+'_speed_limits')[stage]
                    if upper == lower:assert (x == lower).all()
                    else:
                        assert float(x.min()) < lower+.1*(upper-lower)
                        assert float(x.max()) > upper-.1*(upper-lower)
                        assert abs(float(x.mean())-(lower+upper)/2.) < .05*(upper-lower)
                    summary[direction] = dict(stage=stage, count=x.numel(), minimum=float(x.min()),
                                              maximum=float(x.max()), mean=float(x.mean()))
                fraction = float((values > 0).sum() / (values != 0).sum())
                assert .43 < fraction < .57, fraction
                rows.append(dict(**summary, forward_fraction=fraction, motion_kinds=sorted(kinds)))
    finally:
        s.segment_started.zero_(); s.direction_curriculum = previous
    return rows


@torch.inference_mode()
def accounting_checks(raw):
    s = raw.command_manager.get_term('standup'); ids = torch.arange(raw.num_envs, device=raw.device)
    previous = copy.deepcopy(s.direction_curriculum); frozen = s.speed_curriculum_frozen
    assert s.cfg.speed_promotion_segments == 256 and s.cfg.speed_promotion_fraction == .8
    s.speed_curriculum_frozen = False

    def finish(direction, count, error=0., failed=False, measured=50, stage=None):
        remaining = count
        while remaining:
            selected = ids[:min(remaining, len(ids))]; state = s.direction_curriculum[direction]
            s.segment_started[selected] = True
            s.segment_direction[selected] = DIRECTIONS.index(direction)
            s.segment_stage[selected] = state['stage'] if stage is None else stage
            s.segment_failed[selected] = failed; s.segment_steps[selected] = measured
            s.segment_error[selected] = error*measured
            s._finish_speed_segments(selected)
            assert not s.segment_started[selected].any()
            remaining -= len(selected)

    rows = []
    try:
        for failing, passing in (('forward', 'backward'), ('backward', 'forward')):
            s.direction_curriculum = fresh_state()
            finish(failing, 256, failed=True); finish(passing, 255, error=.099)
            assert s.direction_curriculum[passing]['stage'] == 0
            finish(passing, 1, error=.099)
            assert s.direction_curriculum[passing]['stage'] == 1
            assert s.direction_curriculum[failing]['stage'] == 0
            assert s.direction_curriculum[passing]['last_success_fraction'] == 1.
            assert s.direction_curriculum[failing]['last_success_fraction'] == 0.
            # 晋级后尚未结束的旧档段不能灌入新档，另一方向原有计数不能被清空。
            finish(failing, 7, failed=True)
            finish(passing, 13, stage=0)
            assert s.direction_curriculum[passing]['segments'] == 0
            assert s.direction_curriculum[failing]['segments'] == 7
            rows.append(copy.deepcopy(s.direction_curriculum))
        s.direction_curriculum = fresh_state()
        finish('forward', 1, error=.099); finish('forward', 1, error=.101)
        finish('forward', 1, error=0., measured=49)
        finish('forward', 1, error=0., failed=True)
        assert s.direction_curriculum['forward']['segments'] == 4
        assert s.direction_curriculum['forward']['successes'] == 1
        # 80%包含等号，低于80%不晋级。
        for successes, expected in ((204, 0), (205, 1)):
            s.direction_curriculum = fresh_state()
            finish('forward', successes); finish('forward', 256-successes, failed=True)
            assert s.direction_curriculum['forward']['stage'] == expected
        for direction in DIRECTIONS:
            s.direction_curriculum = fresh_state()
            last = len(getattr(s.cfg, direction+'_speed_limits'))-1
            s.direction_curriculum[direction]['stage'] = last
            finish(direction, 256)
            assert s.direction_curriculum[direction]['stage'] == last
        s.direction_curriculum = fresh_state(); s.speed_curriculum_frozen = True
        before = copy.deepcopy(s.direction_curriculum)
        finish('forward', 256); finish('backward', 256)
        assert s.direction_curriculum == before
        return dict(independent_failure_success_batches=rows, boundary_0099_pass_0101_fail=True,
                    minimum_measurement_50_ticks=True, failed_segment_never_success=True,
                    promotion_205_of_256_pass_204_fail=True, stale_stage_ignored=True,
                    last_stage_clamped=True, frozen_stages_and_counters_unchanged=True)
    finally:
        s.direction_curriculum = previous; s.speed_curriculum_frozen = frozen
        s.segment_started.zero_()


@torch.inference_mode()
def resume_checks(env, args):
    raw = env.unwrapped; s = raw.command_manager.get_term('standup')
    built, _ = task_registry.make_alg_runner(env, 'chuanliantui_walk', device=args.device)
    s.direction_curriculum = dict(forward=dict(stage=3, segments=123, successes=77, last_success_fraction=.83),
                                 backward=dict(stage=1, segments=41, successes=31, last_success_fraction=.9))
    saved = copy.deepcopy(s.checkpoint_state()); path = args.output/'untrained-resume.pt'
    built.save(str(path)); s.direction_curriculum = fresh_state()
    built.load(str(path), map_location=args.device)
    assert s.checkpoint_state() == saved
    infos = torch.load(path, map_location='cpu', weights_only=False)['infos']
    runner.validate(infos, s.cfg)
    for field, value in (('forward_speed_limits', (.4,.7,.8,1.,1.5)),
                         ('backward_speed_limits', (1.,1.6)), ('speed_error_threshold', .3),
                         ('forward_speed_min', .2), ('backward_speed_min', .9)):
        for nested in (False, True):
            bad = copy.deepcopy(infos)
            config = bad['standup_curriculum']['speed_curriculum']['config'] if nested else bad['command_config']
            config[field] = value
            rejected(lambda: runner.validate(bad, s.cfg))
        bad = copy.deepcopy(saved); bad['speed_curriculum']['config'][field] = value
        rejected(lambda: s.load_checkpoint_state(bad))
    for direction in DIRECTIONS:
        for field, value in (('stage', 99), ('segments', -1), ('successes', 999), ('last_success_fraction', float('nan'))):
            bad = copy.deepcopy(infos); bad['standup_curriculum']['speed_curriculum'][direction][field] = value
            rejected(lambda: runner.validate(bad, s.cfg))
    for checkpoint in (args.previous_v7, args.previous_fixture):
        rejected(lambda: built.load(str(checkpoint), map_location=args.device))
    # 验证冻结后按两个保存档位采样，而非落回stage 0或使用一个公共档位。
    s.speed_curriculum_frozen = True; ids = torch.arange(raw.num_envs, device=raw.device)
    requests = []
    for _ in range(128):s._sample_speed_target(ids); requests.append(s.tracking_velocity.clone())
    x = torch.cat(requests)
    assert ((x[x>0] >= .3) & (x[x>0] <= 1.)).all() and (x > .8).any()
    assert ((x[x<0].abs() >= 1.) & (x[x<0].abs() <= 1.5)).all() and (x < -1.2).any()
    s._finish_speed_segments(ids)
    assert s.direction_curriculum == {d:saved['speed_curriculum'][d] for d in DIRECTIONS}
    return dict(saved_state=saved['speed_curriculum'], runner_round_trip_exact=True,
                config_and_counter_tampering_rejected=True, trained_and_interface_v7_rejected=True,
                frozen_loaded_direction_stages_used=True)


@torch.inference_mode()
def smoke(args):
    cfg, train = task_registry.get_cfgs('chuanliantui_walk', play=False)
    cfg.seed = 43
    env, _ = task_registry.make_env('chuanliantui_walk', env_cfg=cfg, train_cfg=train,
                                  num_envs=32, device=args.device)
    try:
        built, _ = task_registry.make_alg_runner(env, 'chuanliantui_walk', train_cfg=train, device=args.device)
        saved = torch.load(args.previous_v7, map_location=args.device, weights_only=False)
        built.alg.actor.load_state_dict(saved['actor_state_dict'], strict=True)
        weights = copy.deepcopy(built.alg.actor.state_dict()); policy = built.get_inference_policy(device=args.device)
        raw = env.unwrapped; s = raw.command_manager.get_term('standup')
        obs = env.get_observations(); samples = []; terminations = 0
        for tick in range(600):
            obs, reward, done, _ = env.step(policy(obs))
            assert torch.isfinite(reward).all(); terminations += int(done.sum())
            target = s.tracking_velocity[s.tracking_active & (s.tracking_transition == 1)]
            assert ((target[target>0] >= .3) & (target[target>0] <= .4)).all()
            assert (target[target<0] == -1.).all()
            samples.extend(target.cpu().tolist())
        assert samples and any(x>0 for x in samples) and any(x<0 for x in samples)
        assert all(s.direction_curriculum[d]['segments'] > 0 for d in DIRECTIONS), s.direction_curriculum
        assert all(s.direction_curriculum[d]['stage'] == 0 for d in DIRECTIONS)
        assert tensors_equal(weights, built.alg.actor.state_dict())
        return dict(training_config=True, policy_steps=600, envs=32, exploration_std=0.,
                    direct_v7_actor_load=True, parameter_updates=0, stages_and_counters=s.direction_curriculum,
                    nonzero_target_samples=len(samples), episode_terminations=terminations,
                    forward_bounds=[min(x for x in samples if x>0), max(x for x in samples if x>0)],
                    backward_bounds=[min(x for x in samples if x<0), max(x for x in samples if x<0)])
    finally:env.close()


def main(args):
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    report = dict(verified=False, parameter_updates=0, scope='仅软件/合成检查，未训练、未行为验收')
    try:
        baseline = json.loads(args.before_contracts.read_text())
        assert baseline['getup'] == json.loads(json.dumps(runner.contracts(command_cfg('getup'))))
        env, _ = task_registry.make_env('chuanliantui_walk', play=True, num_envs=32, seed=43, device=args.device)
        try:
            report['sampling'] = sampling_checks(env.unwrapped)
            report['accounting'] = accounting_checks(env.unwrapped)
            report['resume'] = resume_checks(env, args)
        finally:env.close()
        report['regression'] = regression(args.snapshot, args.device, args.before_contracts, args.shared_action_before, args.shared_rewards_before)
        env, _ = task_registry.make_env('chuanliantui_getup', play=True, num_envs=4, device=args.device)
        try:
            built, _ = task_registry.make_alg_runner(env, 'chuanliantui_getup', device=args.device)
            report['getup_load'] = getup_load_check(built, args.getup_checkpoint, args.before_contracts, args.output/'current-untrained-getup.pt', args.device)
        finally:env.close()
        report['smoke'] = smoke(args)
        hashes = json.loads(args.production_hashes.read_text())
        changed = [name for name, value in hashes.items() if digest(name) != value]
        assert set(changed) == {f'mjlab_training/src/ct_mjlab/envs/chuanliantui_split/{name}.py'
                                for name in ('mdp','runner','env_cfg')}, changed
        report['production_files_changed'] = changed
        report['reward_weights_and_recipe_unchanged'] = baseline['walk']['reward_weights'] == dict(
            runner.contracts(command_cfg('walk'))['reward_weights']) and baseline['walk']['reward_recipe'] == runner.contracts(command_cfg('walk'))['reward_recipe']
        assert report['reward_weights_and_recipe_unchanged']
        report['verified'] = True
    finally:
        report['source_sha256'] = digest(__file__)
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sampling','regression')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('previous-v7','previous-fixture','getup-checkpoint','snapshot','before-contracts','production-hashes','output'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--shared-rewards-before', type=Path)
    parser.add_argument('--shared-action-before', type=Path)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Direction curriculum check cannot train')):
        main(parser.parse_args())
