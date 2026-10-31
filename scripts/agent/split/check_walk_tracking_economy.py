#!/usr/bin/env python3
"""Check walk v8 tracking boost, relative vx kernel, 5° tilt band, 75% moving episodes and std floor; no learning."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.modules.distribution import GaussianDistribution

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_split import rewards, runner
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg, make_getup_ppo_cfg, make_walk_ppo_cfg
from scripts.agent.split.check_getup_nominal_state import source_module
from scripts.agent.split.check_chuanliantui_two_policy import rejected, digest

CHANGED = ('tracking_lin_vel', 'orientation')


def fake_inputs(raw, state, vx, gravity):
    n = raw.num_envs
    data = raw.scene['robot'].data
    state.robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.stack((vx, torch.full_like(vx, .03), torch.full_like(vx, .1)), -1),
        root_link_ang_vel_b=torch.tensor([.1, .2, .3], device=raw.device).repeat(n, 1),
        projected_gravity_b=gravity, body_link_pos_w=data.body_link_pos_w.clone(),
        root_link_quat_w=data.root_link_quat_w.clone(), joint_pos=data.joint_pos.clone(),
        joint_vel=torch.full_like(data.joint_vel, .2)))
    action = torch.full_like(raw.action_manager.action, .2)
    return SimpleNamespace(command_manager=raw.command_manager, scene=raw.scene, common_step_counter=0,
        action_manager=SimpleNamespace(action=action, prev_action=action*.5, prev_prev_action=-action*.5),
        sim=SimpleNamespace(data=SimpleNamespace(qfrc_actuator=torch.full_like(raw.sim.data.qfrc_actuator, .2))))


def tilt_gravity(degrees, device):
    theta = torch.deg2rad(degrees.to(device))
    return torch.stack((theta.sin(), torch.zeros_like(theta), -theta.cos()), -1), theta


@torch.inference_mode()
def reward_check(raw, before):
    s = raw.command_manager.get_term('standup')
    n = raw.num_envs
    assert set(rewards.WALK_WEIGHTS) == set(before.WALK_WEIGHTS) | {'nominal_state'}
    assert {k:v for k,v in rewards.WALK_WEIGHTS.items() if k != 'nominal_state'} == before.WALK_WEIGHTS
    current, previous = rewards.WalkRewardState(raw), before.WalkRewardState(raw)
    saved = {name: getattr(s, name).clone() for name in
             ('reward_active', 'numerical_failure', 'tracking_active', 'position_locked')}
    command = s._command.clone()
    rows = []
    try:
        s.reward_active.fill_(True); s.numerical_failure.zero_()
        angles = torch.tensor([0., 3., 5., 6., 8., 10., 15., 19., 25., 90., 180.])
        angles = angles[torch.arange(n) % len(angles)]
        gravity, theta = tilt_gravity(angles, raw.device)
        cmd = torch.tensor([0., .3, .5, -.5, 1., -1., 1.5, -1.5], device=raw.device)[torch.arange(n, device=raw.device) % 8]
        vx = cmd*.6
        with patch.object(rewards, 'base_contact_force', return_value=torch.full((n,), .5, device=raw.device)), \
             patch.object(before, 'base_contact_force', return_value=torch.full((n,), .5, device=raw.device)):
            for active, locked in ((False, False), (True, False), (True, True)):
                s.tracking_active.fill_(active); s.position_locked.fill_(locked); s._command[:, 0] = cmd
                fake = fake_inputs(raw, current, vx, gravity); previous.robot = current.robot
                current.step = previous.step = -1
                new, old = current.compute(fake), previous.compute(fake)
                # 其余15项与v5逐位相同。
                assert all(torch.equal(new[k], old[k]) for k in before.WALK_WEIGHTS if k not in CHANGED)
                boost = 5. if active and not locked else 1.
                error = (cmd-vx).square()
                expected = boost*torch.exp(-error/cmd.square().clamp_min(.25))
                assert torch.allclose(new['tracking_lin_vel'], expected, atol=1e-6)
                slow = cmd.abs() <= .5
                if not locked:
                    assert torch.equal(new['tracking_lin_vel'][slow], old['tracking_lin_vel'][slow])
                else:
                    assert torch.allclose(new['tracking_lin_vel'][slow], old['tracking_lin_vel'][slow]/5.)
                weighted = new['orientation']*raw.cfg.rewards['orientation'].weight
                free = math.cos(math.radians(5.))
                assert torch.allclose(weighted, (-100.*(free-theta.cos())).clamp(-5., 0.), atol=1e-5)
                assert not weighted[angles.to(raw.device) <= 5.].any()
                rows.append(dict(tracking_active=active, position_locked=locked, boost=boost))
            standing = torch.zeros(n, device=raw.device)
            s.tracking_active.fill_(True); s.position_locked.zero_(); s._command[:, 0] = 1.
            fake = fake_inputs(raw, current, standing, gravity); current.step = -1
            standing_1ms = float(current.compute(fake)['tracking_lin_vel'][0])
            assert math.isclose(standing_1ms, 5.*math.exp(-1.), rel_tol=1e-5)
            weighted_by_angle = {}
            for degrees in (0., 5., 6., 8., 10., 15., 19., 25., 180.):
                g, _ = tilt_gravity(torch.full((n,), degrees), raw.device)
                fake = fake_inputs(raw, current, standing, g); current.step = -1
                weighted_by_angle[degrees] = float(current.compute(fake)['orientation'][0]*-100.)
            values = list(weighted_by_angle.values())
            assert all(a >= b for a, b in zip(values, values[1:])) and values[-1] == -5.
            for flag, value in (('reward_active', False), ('numerical_failure', True)):
                getattr(s, flag).fill_(value); current.step = -1
                assert not any(v.any() for v in current.compute(fake).values())
                getattr(s, flag).fill_(not value)
    finally:
        s._command.copy_(command)
        for name, value in saved.items():
            getattr(s, name).copy_(value)
    return dict(cases=rows, unchanged_terms=sorted(set(before.WALK_WEIGHTS)-set(CHANGED)),
                weighted_orientation_before_dt=weighted_by_angle,
                tracking_lin_vel_standing_at_1m_s_active=standing_1ms,
                tracking_lin_vel_standing_at_1m_s_v5=5.*math.exp(-4.))


@torch.inference_mode()
def episode_check(raw):
    s = raw.command_manager.get_term('standup')
    assert s.cfg.moving_episode_probability == .75
    for value in (-.1, 1.1):
        bad = command_cfg('walk'); bad.moving_episode_probability = value
        rejected(lambda: bad.build(raw))
    ids = torch.arange(raw.num_envs, device=raw.device)
    torch.manual_seed(0)
    moving = []
    for _ in range(400):
        s._resample_command(ids)
        assert torch.equal(s.zero_hold, ~s.tracking_episode)
        moving.append(s.tracking_episode.float().mean())
    fraction = float(torch.stack(moving).mean())
    samples = 400*raw.num_envs
    assert abs(fraction-.75) < 4*math.sqrt(.75*.25/samples), fraction
    s.reset(ids)
    return dict(samples=samples, moving_fraction=fraction)


def ppo_check(env, raw):
    walk, getup = make_walk_ppo_cfg(), make_getup_ppo_cfg()
    assert walk.actor.distribution_cfg['std_range'] == (.2, 1e6)
    assert 'std_range' not in getup.actor.distribution_cfg
    cfg = {k: v for k, v in walk.actor.distribution_cfg.items() if k != 'class_name'}
    dist = GaussianDistribution(6, **cfg)
    dist.std_param.data.fill_(.05); dist.update(torch.zeros(1, 6))
    assert torch.allclose(dist.std, torch.full((1, 6), .2))
    dist.std_param.data.fill_(.5); dist.update(torch.zeros(1, 6))
    assert torch.allclose(dist.std, torch.full((1, 6), .5))
    _, train = task_registry.get_cfgs('chuanliantui_walk', play=True)
    built, _ = task_registry.make_alg_runner(env, 'chuanliantui_walk', train_cfg=train, device=raw.device)
    assert built.alg.actor.distribution.std_range[0] == .2
    return dict(walk_std_floor=.2, getup_std_range='default', runner_actor_floor=.2)


def main(args):
    torch.set_num_threads(1); args.output.mkdir(parents=True, exist_ok=False)
    before = source_module(args.before_rewards, 'ct_mjlab.envs.chuanliantui_split._walk_rewards_v5')
    walk_contract = runner.contracts(command_cfg('walk'))
    assert walk_contract['experiment_profile_contract'] == 'chuanliantui-walk-physical25-v8'
    assert walk_contract['command_contract'] == 'split-walk-episode-height-stop-straight-spin-moving75-walk-ready-armed1s-down100ms-dirsplit-fwd30-40-bwd100-err10-v6'
    assert walk_contract['reward_recipe'] == rewards.WALK_REWARD_RECIPE
    assert walk_contract['command_config']['moving_episode_probability'] == .75
    assert walk_contract['exploration'] == dict(actor_std_min=.2)
    for version in range(1, 8):
        bad = dict(walk_contract, experiment_profile_contract=f'chuanliantui-walk-physical25-v{version}')
        rejected(lambda: runner.validate(bad, command_cfg('walk')))
    old_walk = torch.load(args.previous_walk, map_location='cpu', weights_only=False)['infos']
    rejected(lambda: runner.validate(old_walk, command_cfg('walk')))
    getup = torch.load(args.getup_checkpoint, map_location='cpu', weights_only=False)['infos']
    getup_accepted = True
    try:runner.validate(getup, command_cfg('getup'))
    except ValueError:
        assert args.before_contracts is not None
        baseline = json.loads(args.before_contracts.read_text())['getup']
        assert baseline == json.loads(json.dumps(runner.contracts(command_cfg('getup'))))
        assert getup['reward_recipe'] != baseline['reward_recipe']
        getup_accepted = False
    env, _ = task_registry.make_env('chuanliantui_walk', play=True, num_envs=args.num_envs, seed=43, device=args.device)
    try:
        raw = env.unwrapped
        assert raw.cfg.rewards['orientation'].weight == -100.
        rewards_report = reward_check(raw, before)
        episodes = episode_check(raw)
        ppo = ppo_check(env, raw)
        # 零动作物理推演只核对奖励有限，不代表行为。
        env.reset(); total = {}
        with torch.inference_mode():
            for _ in range(args.steps):
                _, reward, _, _ = env.step(torch.zeros(raw.num_envs, 6, device=raw.device))
                assert torch.isfinite(reward).all()
        for name, term in raw.reward_manager._episode_sums.items():
            total[name] = float(term.mean())
    finally:
        env.close()
    report = dict(verified=True, parameter_updates=0, contract=dict(
        profile=walk_contract['experiment_profile_contract'], reward_recipe=walk_contract['reward_recipe'],
        command_contract=walk_contract['command_contract'], exploration=walk_contract['exploration']),
        previous_walk_rejected=True, getup_checkpoint_still_accepted=getup_accepted,
        getup_rejection_preexisting=not getup_accepted,
        rewards=rewards_report, episodes=episodes, ppo=ppo, zero_action_steps=args.steps,
        zero_action_episode_sums=total, before_rewards_sha256=digest(args.before_rewards),
        source_sha256=digest(__file__),
        scope='Synthetic reward/contract/sampling/std checks and finite zero-action rollout; no trained behavior acceptance')
    (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-rewards', type=Path, required=True, help='v5 walk rewards.py (training snapshot)')
    parser.add_argument('--previous-walk', type=Path, required=True)
    parser.add_argument('--getup-checkpoint', type=Path, required=True)
    parser.add_argument('--before-contracts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Check cannot train')):
        main(parser.parse_args())
