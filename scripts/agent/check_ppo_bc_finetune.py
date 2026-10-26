#!/usr/bin/env python3
"""Finite PPO refinement checks and original40 sampled replay; optimizer steps forbidden."""
import argparse, copy, hashlib, json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.os import dump_yaml

from ct_mjlab.bc import load_actor
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.finetune import ACTION_STD, PROFILE, RECIPE, RECIPE_V2, ProtectedPPO, parking_scores
from ct_mjlab.smp.ppo import configure_smp, checkpoint_bc_finetune, policy_observation_contract
from ct_mjlab.wheel_angles import HANDOVER_OBSERVATION_CONTRACT
from screen_frozen_bc_conditions import support


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = RECIPE_V2 if args.variant == 'v2' else RECIPE
    immutable = {str(p): digest(p) for p in (args.checkpoint, args.prior, MODEL_XML)}
    old, old_train = task_registry.get_cfgs('chuanliantui')
    configure_smp(old, old_train, args.prior, policy_contract=HANDOVER_OBSERVATION_CONTRACT)
    old_snapshot = copy.deepcopy((asdict(old), asdict(old_train)))
    cfg, train = task_registry.get_cfgs('chuanliantui')
    configure_smp(cfg, train, args.prior, policy_contract=HANDOVER_OBSERVATION_CONTRACT, bc_finetune=args.variant)
    assert (asdict(old), asdict(old_train)) == old_snapshot
    assert cfg.events['reset_pose'].func == old.events['reset_pose'].func
    assert cfg.actions['pd'].ground_start_hold_s == .2 and cfg.sim.mujoco.timestep == .002 and cfg.decimation == 5
    assert cfg.actions['pd'].randomize_delays and cfg.actions['pd'].action_delay_choices_s == (.002, .004)
    assert set(cfg.events) == {'reset_pose', 'reset_observations'}
    assert not cfg.actions['pd'].motor_randomization
    assert cfg.actions['pd'].spring_force_bias_range == cfg.actions['pd'].nominal_leg_offset_range == (0., 0.)
    for group in cfg.observations.values():
        obs_cfg = group.terms['proprioception'].params['cfg']
        assert not obs_cfg.noise and not obs_cfg.delay and obs_cfg.wheel_angle_handover_reset
    assert 'zero_command_hold' not in cfg.rewards
    assert sum(cfg.rewards[n].weight for n in ('zero_command_brake', 'zero_command_position')) == 3.
    for name in old.rewards:
        if name != 'zero_command_hold':
            assert asdict(cfg.rewards[name]) == asdict(old.rewards[name])
    speed = torch.tensor([0., .01, .05, .5], requires_grad=True)
    brake, position = parking_scores(speed, torch.full((4,), 1.37),
                                    velocity_scale=recipe.get('braking_velocity_scale', .05))
    assert (brake[:-1] > brake[1:]).all() and (position > 0).all()
    brake.sum().backward()
    assert (speed.grad[1:] < 0).all()
    cfg.seed = train.seed = 42
    cfg.auto_reset = False
    cfg.metrics['finetune_support'] = MetricsTermCfg(func=support, per_substep=True, reduce='last')
    dump_yaml(args.output/'env.yaml', asdict(cfg)); dump_yaml(args.output/'agent.yaml', asdict(train))
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=40)
    try:
        raw = env.unwrapped
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        assert isinstance(runner.alg, ProtectedPPO)
        runner.initialize_from_bc(args.checkpoint)
        alg = runner.alg
        actor, _ = load_actor(args.checkpoint, raw.device)
        initial = copy.deepcopy(alg.save())
        # 收集真实24步，但拦截所有Adam写入；只检查梯度和分支逻辑。
        def rollout():
            with torch.inference_mode():
                raw.reset(seed=42)
                obs = env.get_observations()
                torch.testing.assert_close(alg.actor(obs), actor(obs), atol=0, rtol=0)
                for tick in range(24):
                    obs, reward, done, extras = env.step(alg.act(obs))
                    alg.process_env_step(obs, reward, done, extras)
                    if tick < 20:
                        assert not raw.action_manager.get_term('pd').torques.any()
                alg.compute_returns(obs)
            return obs
        captures = []
        def capture_grad():
            ap = list(alg.actor.mlp.parameters()); cp = list(alg.critic.parameters())
            for p in ap+cp:
                assert p.grad is None or torch.isfinite(p.grad).all()
            captures.append(dict(actor=any(p.grad is not None and p.grad.count_nonzero() for p in ap),
                                 critic=any(p.grad is not None and p.grad.count_nonzero() for p in cp)))
        rollout()
        with patch.object(alg.optimizer, 'step', side_effect=capture_grad):
            warmup = alg.update()
        assert len(captures) == 20 and all(not x['actor'] and x['critic'] for x in captures)
        assert alg.completed_iterations == 1 and alg.actor_updates == 0
        alg.completed_iterations = 20  # 仅进入预热后的代码分支，不执行前20轮训练。
        rollout(); captures.clear()
        with patch.object(alg.optimizer, 'step', side_effect=capture_grad):
            actor_branch = alg.update()
        assert any(x['actor'] for x in captures) and all(x['critic'] for x in captures)
        assert not alg.optimizer.state
        for name in ('actor_state_dict', 'critic_state_dict'):
            for key, value in initial[name].items():
                torch.testing.assert_close(alg.save()[name][key], value, atol=0, rtol=0)
        # 固定BC参考对偏离动作产生拉回梯度；不修改生产模型权重。
        inputs = torch.randn(64, 39, device=raw.device)
        mean = (alg.reference(inputs).detach()+.01).requires_grad_()
        scale = torch.tensor(recipe.get('bc_action_scale', ACTION_STD), device=raw.device)
        loss = .5*((mean-alg.reference(inputs))/scale).square().mean()
        loss.backward(); assert (mean.grad > 0).all()
        alg.load(initial)
        path = args.output/'untrained-finetune.pt'; runner.save(str(path))
        assert checkpoint_bc_finetune(path) and policy_observation_contract(path) == HANDOVER_OBSERVATION_CONTRACT
        # 恢复到新的runner，避免同一对象自载掩盖缺失字段。
        resumed, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        resumed.load(str(path))
        for key, value in initial['bc_finetune']['reference_state_dict'].items():
            torch.testing.assert_close(resumed.alg.reference.state_dict()[key], value, atol=0, rtol=0)
        assert resumed.alg.completed_iterations == 0 and not resumed.alg.optimizer.state
        bad = torch.load(path, map_location='cpu', weights_only=False)
        bad['bc_finetune']['recipe'] = {**recipe, 'critic_warmup_iterations': 0}
        rejected = args.output/'rejected.pt'; torch.save(bad, rejected)
        try:
            resumed.load(str(rejected))
        except ValueError:
            pass
        else:
            raise AssertionError('Changed refinement recipe was accepted')
        # 原40组：直接使用真实PPO高斯采样接口，完整20秒，不加载巨大的BC配对数据。
        cached = torch.load(args.source/'nominal_mean/initial-state.pt', map_location=raw.device, weights_only=False)
        with torch.inference_mode(), patch.object(alg.optimizer, 'step', side_effect=AssertionError('No learning')):
            raw.reset(seed=42); action = raw.action_manager.get_term('pd')
            q = cached['qpos'].clone(); q[:, :2] = raw.scene.env_origins[:, :2]
            raw.sim.data.qpos.copy_(q); raw.sim.data.qvel.copy_(cached['qvel']); raw.sim.forward(); action.reset()
            action.action_delay_steps.copy_(cached['leg_delay']); action.wheel_action_delay_steps.copy_(cached['wheel_delay'])
            cmd = raw.command_manager.get_term('standup'); cmd.ground_start[:] = True; cmd.compute(dt=0.)
            multimode_observations.reset_observations(raw, torch.arange(40, device=raw.device))
            obs = raw.observation_manager.compute(update_history=True)
            body = raw.scene['robot'].indexing.body_ids
            raw._fast_weight = torch.full((40,), float(raw.sim.get_default_field('body_mass')[body].sum())*9.81, device=raw.device)
            raw._fast_probe = dict(live=torch.ones(40, dtype=torch.bool, device=raw.device),
                hold=torch.zeros(40, dtype=torch.long, device=raw.device), strict_run=torch.zeros(40, dtype=torch.long, device=raw.device),
                strict_ever=torch.zeros(40, dtype=torch.bool, device=raw.device))
            torch.manual_seed(20261026)
            reader = MotionStateReader(raw)
            buffers = {k: [] for k in ('qpos', 'qvel', 'motor_dq', 'features', 'action', 'mean', 'alive', 'rewards')}
            for tick in range(2000):
                live = raw._fast_probe['live'].clone()
                output = alg.actor(obs, stochastic_output=True)
                mean = alg.actor.output_distribution_params[0].clone()
                torch.testing.assert_close(alg.actor.output_std[0], torch.tensor(ACTION_STD, device=raw.device), atol=0, rtol=0)
                obs, reward, done, _ = env.step(output)
                values = raw.reward_manager._step_reward*raw.step_dt
                torch.testing.assert_close(values.sum(-1), reward, atol=2e-6, rtol=2e-5)
                state = reader.read()
                data = dict(qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel, motor_dq=state['motor_dq'],
                    features=reader.features(), action=output, mean=mean, alive=live, rewards=values)
                for key, value in data.items(): buffers[key].append(value.cpu().numpy().copy())
                if tick < 20: assert not action.torques.any()
                raw._fast_probe['live'] &= ~done.bool()
                if done.any():
                    raw.reset(env_ids=done.nonzero().flatten())
                    action.action_delay_steps.copy_(cached['leg_delay']); action.wheel_action_delay_steps.copy_(cached['wheel_delay'])
                    obs = raw.observation_manager.compute(update_history=False)
            d = {k: np.asarray(v) for k, v in buffers.items()}; rows = []
            for i in range(40):
                n = int(d['alive'][:, i].sum()); late = slice(1500, n)
                rms = float(np.sqrt(np.square(d['motor_dq'][late, i][:, [0, 1, 3, 4]]).mean())) if n > 1500 else None
                speed_mean = float(np.linalg.norm(d['features'][late, i, 4:6], axis=-1).mean()) if n > 1500 else None
                drift = float(np.linalg.norm(d['qpos'][late, i, :2]-d['qpos'][1500, i, :2], axis=-1).max()) if n > 1500 else None
                yaw = float(np.abs(d['features'][late, i, 9]).mean()) if n > 1500 else None
                balance = bool(n == 2000 and raw._fast_probe['hold'][i] >= 500)
                strict = bool(raw._fast_probe['strict_ever'][i])
                rows.append(dict(env=i, strict_rise=strict, end_balance=balance, observed_s=n*.01,
                    leg_rms=rms, speed=speed_mean, drift=drift, yaw=yaw,
                    qualified=bool(strict and balance and rms is not None and rms <= .02
                                   and speed_mean <= .01 and drift <= .02 and yaw <= .02)))
        np.savez(args.output/'trace.npz', **d, initial_qpos=q.cpu().numpy(), initial_qvel=cached['qvel'].cpu().numpy(),
                 reward_names=np.asarray(raw.reward_manager.active_terms))
        for name in ('actor_state_dict', 'critic_state_dict'):
            for key, value in initial[name].items():
                torch.testing.assert_close(alg.save()[name][key], value, atol=0, rtol=0)
        for p, sha in immutable.items(): assert digest(p) == sha
        result = dict(passed=True, optimizer_updates=0, recipe=recipe,
            strict_rise=sum(x['strict_rise'] for x in rows), end_balance=sum(x['end_balance'] for x in rows),
            qualified=sum(x['qualified'] for x in rows), smooth=sum(x['leg_rms'] is not None and x['leg_rms'] <= .02 for x in rows),
            mean_leg_rms=float(np.mean([x['leg_rms'] for x in rows if x['leg_rms'] is not None])),
            mean_speed=float(np.mean([x['speed'] for x in rows if x['speed'] is not None])),
            rows=rows, leg_std=.001, wheel_std=.005, seconds=20, num_envs=40,
            trace_sha256=digest(args.output/'trace.npz'), immutable_sha256=immutable,
            checks=dict(legacy_cfg_unchanged=True, actor_frozen_during_warmup=True,
                actor_grad_after_warmup=True, reference_restoring_gradient=True,
                new_runner_checkpoint_roundtrip=True, changed_recipe_rejected=True,
                actual_action_std_fixed=True, zero_torque_first_20_steps=True,
                reward_sum_once_dt=True, no_optimizer_writes=True),
            warmup_branch=warmup, actor_branch=actor_branch,
            scope='Finite40-env sampler/reward/gradient checks and original40 nominal20s replay; all optimizer steps intercepted, no trained PPO or parking improvement claim.')
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'prior', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--variant', choices=('v1', 'v2'), default='v1')
    check(p.parse_args())
