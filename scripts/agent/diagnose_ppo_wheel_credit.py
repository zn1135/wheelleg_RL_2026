#!/usr/bin/env python3
"""Frozen 60s continuation and same-state wheel pulse/critic comparison."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.smp.features import MotionStateReader
from ct_mjlab.smp.ppo import (configure_smp, checkpoint_bc_finetune,
                            checkpoint_smp_weight, policy_observation_contract)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_fields(owner, count):
    # 只复制按环境保存的张量；环境索引必须保持每个world自身的编号。
    return {name: value.clone() for name, value in vars(owner).items()
            if isinstance(value, torch.Tensor) and count in value.shape
            and name not in ('env_ids', '_env_ids')}


def repeat_first(value, count):
    axes = [i for i, size in enumerate(value.shape) if size == count]
    assert len(axes) == 1, value.shape
    axis = axes[0]
    return value.narrow(axis, 0, count//4).repeat(
        *[4 if i == axis else 1 for i in range(value.ndim)])


def group_difference(value):
    grouped = value.reshape(4, 40, -1)
    return float((grouped[1:]-grouped[:1]).abs().max())


@torch.inference_mode()
def run(args):
    from mjlab.utils.torch import configure_torch_backends
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_paths = (args.checkpoint, args.prior, MODEL_XML, Path(__file__))
    hashes = {str(p): digest(p) for p in frozen_paths}
    cached = torch.load(args.source/'nominal_mean/initial-state.pt', map_location='cpu', weights_only=False)
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    configure_smp(cfg, train, args.prior, play=True,
                  policy_contract=policy_observation_contract(args.checkpoint),
                  reward_weight=checkpoint_smp_weight(args.checkpoint),
                  bc_finetune=checkpoint_bc_finetune(args.checkpoint))
    cfg.seed = train.seed = 42
    cfg.auto_reset = False
    cfg.episode_length_s = 61.  # 仅诊断延长观察；原20秒验收不变。
    cfg.actions['pd'].randomize_delays = False
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train, num_envs=160)
    configure_torch_backends(allow_tf32=False)
    try:
        raw = env.unwrapped
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train)
        runner.load(str(args.checkpoint), load_cfg={'actor': True, 'critic': True})
        actor, critic = runner.alg._raw_actor, runner.alg.critic
        original = {name: {k: v.clone() for k, v in module.state_dict().items()}
                    for name, module in (('actor', actor), ('critic', critic))}
        raw.reset(seed=42)
        raw.scene.env_origins.zero_()
        q = cached['qpos'].to(raw.device).repeat(4, 1); q[:, :2] = 0.
        raw.sim.data.qpos.copy_(q)
        raw.sim.data.qvel.copy_(cached['qvel'].to(raw.device).repeat(4, 1))
        raw.sim.forward()
        pd = raw.action_manager.get_term('pd'); pd.reset()
        pd.action_delay_steps.copy_(cached['leg_delay'].to(raw.device).repeat(4))
        pd.wheel_action_delay_steps.copy_(cached['wheel_delay'].to(raw.device).repeat(4))
        command = raw.command_manager.get_term('standup')
        command.ground_start[:] = True; command.compute(dt=0.)
        multimode_observations.reset_observations(raw, torch.arange(160, device=raw.device))
        obs = raw.observation_manager.compute(update_history=True)
        reader = MotionStateReader(raw)
        observer = multimode_observations.get_state(raw)
        smp = raw._smp_reward
        continuation = {k: [] for k in ('qpos', 'features', 'motor_dq', 'alive')}
        live = torch.ones(160, dtype=torch.bool, device=raw.device)
        snapshot = None
        # 先完整回放，不施加扰动；所有参数冻结，Adam.step一旦被调用即失败。
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Frozen diagnostic')):
            for tick in range(1 if args.restore else 6000):
                obs, _, done, _ = env.step(actor(obs))
                live &= ~done.bool()
                state = reader.read()
                for name, value in dict(qpos=raw.sim.data.qpos, features=reader.features(),
                                        motor_dq=state['motor_dq'], alive=live).items():
                    continuation[name].append(value[:40].cpu().numpy().copy())
                if tick == 1499:
                    owners = dict(command=command, pd=pd, action_manager=raw.action_manager,
                                  observer=observer, delay=observer.delay, history=smp.history,
                                  smp=smp, isaac=raw._isaac_standup_rewards,
                                  robot_data=raw.scene['robot'].data, env=raw,
                                  termination=raw.termination_manager)
                    snapshot = {name: batch_fields(owner, 160) for name, owner in owners.items()}
                    # MuJoCo积分状态、外力、执行器输入及求解器热启动都保留。
                    snapshot['physics'] = {name: getattr(raw.sim.data, name).clone()
                        for name in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qfrc_applied',
                                     'xfrc_applied', 'mocap_pos', 'mocap_quat', 'time')}
                    snapshot['contacts'] = {name: {k: v.clone() for k, v in
                        (raw.scene[name]._history_state or {}).items()}
                        for name in ('wheel_contact', 'base_contact')}
                    scalars = dict(common=raw.common_step_counter, sim=raw._sim_step_counter,
                                   pd_head=pd._history_head, sensor_head=observer.delay.head)
                    source_obs = {k: v.clone() for k, v in obs.items()}
                    torch.save(dict(state=snapshot, scalars=scalars, observations=source_obs),
                               args.output/'state-at15s.pt')
                if (tick+1) % 1000 == 0:
                    print(f'continuation t={(tick+1)*.01:.0f}s live={int(live[:40].sum())}/40', flush=True)
                if done.any():
                    raise RuntimeError('Physical failure during frozen continuation; preserve log, no causal claim')

            if args.restore:
                saved = torch.load(args.restore, map_location=raw.device, weights_only=False)
                snapshot, scalars, source_obs = saved['state'], saved['scalars'], saved['observations']
                owners = dict(command=command, pd=pd, action_manager=raw.action_manager,
                              observer=observer, delay=observer.delay, history=smp.history,
                              smp=smp, isaac=raw._isaac_standup_rewards,
                              robot_data=raw.scene['robot'].data, env=raw,
                              termination=raw.termination_manager)
                with np.load(args.restore.parent/'continuation.npz') as d:
                    arrays = {k: d[k].copy() for k in d.files}
            else:
                arrays = {k: np.asarray(v) for k, v in continuation.items()}
                np.savez_compressed(args.output/'continuation.npz', **arrays)
            periods = {}
            for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 40), (40, 60), (55, 60)):
                interval = slice(lo*100, hi*100)
                f, pos = arrays['features'][interval], arrays['qpos'][interval, :, :2]
                periods[f'{lo}-{hi}s'] = dict(mean_speed_m_s=float(np.linalg.norm(f[..., 4:6], axis=-1).mean()),
                    signed_vx_m_s=float(f[..., 4].mean()),
                    mean_xy_excursion_m=float(np.linalg.norm(pos-pos[:1], axis=-1).max(0).mean()),
                    leg_rms_rad_s=float(np.sqrt(np.square(arrays['motor_dq'][interval][..., [0, 1, 3, 4]]).mean())))

            # 将15秒时第一组40个真实状态复制到四组；另留一组零扰动量化数值分歧。
            for name, owner in owners.items():
                for key, value in snapshot[name].items():
                    getattr(owner, key).copy_(repeat_first(value, 160))
            for name, value in snapshot['physics'].items():
                getattr(raw.sim.data, name).copy_(repeat_first(value, 160))
            for name, fields in snapshot['contacts'].items():
                for key, value in fields.items():
                    raw.scene[name]._history_state[key].copy_(repeat_first(value, 160))
            raw.common_step_counter = scalars['common']; raw._sim_step_counter = scalars['sim']
            pd._history_head = scalars['pd_head']; observer.delay.head = scalars['sensor_head']
            raw._isaac_standup_rewards.step = -1
            raw.sim.forward(); raw.sim.sense()
            # compute(False)默认返回旧缓存；重建后必须先使缓存失效。
            raw.observation_manager._obs_buffer = None
            obs = raw.observation_manager.compute(update_history=False)
            initial = dict(qpos=group_difference(raw.sim.data.qpos), qvel=group_difference(raw.sim.data.qvel),
                ctrl=group_difference(raw.sim.data.ctrl), actor=group_difference(obs['actor']),
                critic=group_difference(obs['critic']),
                pd_fifo=float((pd._action_history[:, 40:]-pd._action_history[:, :40].repeat(1, 3, 1)).abs().max()),
                sensor_fifo=float((observer.delay.history[:, 40:]-observer.delay.history[:, :40].repeat(1, 3, 1)).abs().max()),
                smp_frames=group_difference(smp.history.frames),
                source_actor=float((obs['actor'][:40]-source_obs['actor'][:40]).abs().max()),
                source_critic=float((obs['critic'][:40]-source_obs['critic'][:40]).abs().max()))
            assert max(initial[k] for k in initial if not k.startswith('source_')) < 1e-5, initial
            base_value = critic(obs).squeeze(-1).clone()
            sigma = actor.distribution.std_param
            generator = torch.Generator(device=raw.device).manual_seed(714)
            score_errors = smp.prior.score_errors

            def common_score(windows, *, noise):
                assert len(windows) == 160
                return score_errors(windows, noise=noise[:, :40].repeat(1, 4, 1))

            pulse = torch.tensor([0., 0., args.pulse, -args.pulse], device=raw.device).repeat_interleave(40)
            fields = {k: [] for k in ('reward', 'value', 'vx', 'torque', 'action', 'qpos', 'done')}
            with patch.object(smp.prior, 'score_errors', side_effect=common_score):
                for tick in range(500):
                    output = actor(obs)
                    if tick == 0:
                        output[:, 2] += pulse; output[:, 5] -= pulse
                    else:
                        noise = torch.randn(40, 6, generator=generator, device=raw.device)
                        output += noise.repeat(4, 1)*sigma
                    obs, reward, done, _ = env.step(output)
                    state = dict(reward=reward, value=critic(obs).squeeze(-1),
                                 vx=raw.scene['robot'].data.root_link_lin_vel_b[:, 0],
                                 torque=pd.torques[:, [2, 5]], action=output,
                                 qpos=raw.sim.data.qpos, done=done)
                    for name, value in state.items(): fields[name].append(value.cpu().numpy().copy())
                    if done.any(): raise RuntimeError('Fork terminated; no unrestricted return comparison')
                    if (tick+1) % 100 == 0: print(f'wheel pulse horizon={(tick+1)*.01:.0f}s', flush=True)
            fork = {k: np.asarray(v) for k, v in fields.items()}
            fork['initial_value'] = base_value.cpu().numpy()
            np.savez_compressed(args.output/'wheel-pulse.npz', **fork)
            gamma = runner.alg.gamma
            directions = {}
            for steps in (1, 20, 100, 200, 500):
                observed = (fork['reward'][:steps]*gamma**np.arange(steps)[:, None]).sum(0).reshape(4, 40)
                bootstrapped = observed+gamma**steps*fork['value'][steps-1].reshape(4, 40)
                delta = observed[2]-observed[3]
                predicted = bootstrapped[2]-bootstrapped[3]
                null = observed[0]-observed[1]
                directions[str(steps)] = dict(seconds=steps*.01,
                    return_mean_per_branch=observed.mean(1).tolist(),
                    plus_minus_mean=float(delta.mean()), plus_minus_standard_error=float(delta.std(ddof=1)/np.sqrt(40)),
                    null_mean=float(null.mean()), null_mean_absolute=float(np.abs(null).mean()),
                    bootstrapped_plus_minus_mean=float(predicted.mean()),
                    sign_agreement_one_step=float((np.sign(delta)==np.sign(
                        (fork['reward'][0]+gamma*fork['value'][0]).reshape(4, 40)[2]-
                        (fork['reward'][0]+gamma*fork['value'][0]).reshape(4, 40)[3])).mean()))
            for name, module in (('actor', actor), ('critic', critic)):
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value, original[name][key], atol=0, rtol=0)
            for path, old_hash in hashes.items(): assert digest(path) == old_hash
            report = dict(completed=True, optimizer_updates=0, hashes=hashes, periods=periods,
                continuation_source=str(args.restore.parent/'continuation.npz') if args.restore else str(args.output/'continuation.npz'),
                restored_state_sha256=digest(args.restore) if args.restore else None,
                initial_fork_max_difference=initial, pulse_raw_action=args.pulse,
                pulse_wheel_velocity_target_rad_s=args.pulse*10, gamma=gamma,
                branches=['baseline-a', 'baseline-b', 'plus', 'minus'], directions=directions,
                null_final_qpos_max=float(np.abs(fork['qpos'][-1, :40]-fork['qpos'][-1, 40:80]).max()),
                scope='Frozen cached40 poses; 60s horizon and zero diagnostic origins differ from20s acceptance. '
                      'Same15s physics/control/sensor/SMP state; one10ms wheel pulse, then common marginal-policy noise. '
                      'Duplicate null branch measures GPU divergence. Finite discounted returns are not infinite-horizon Q. '
                      'No learning or behavior-acceptance claim.')
            (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pulse', type=float, default=.02)
    parser.add_argument('--restore', type=Path, help='Reuse a saved15s state and its completed60s continuation')
    run(parser.parse_args())
