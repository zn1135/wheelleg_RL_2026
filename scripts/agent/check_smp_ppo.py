#!/usr/bin/env python3
"""Finite SMP/PPO integration and score probes; never learn/update/optimizer.step."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.features import motion_features
from ct_mjlab.smp.ppo import configure_smp, SmpOnPolicyRunner

ROOT = Path(__file__).resolve().parents[2]


def score_probes(term, manifests):
    dataset = MotionDataset(manifests)
    prior = term.prior
    generator = torch.Generator().manual_seed(823)
    windows = dataset.sample('validation', 384, generator).to(prior.feature_mean.device)
    noise = torch.randn(3, len(windows), 300, generator=generator).to(windows.device)
    bad_gravity = windows.clone()
    bad_gravity[:, :, 1:4] *= -1  # 保留其余字段，仅探测倒置重力方向。
    bad_speed = windows.clone()
    bad_speed[:, :, 4:10] = 8.
    bad_speed[:, :, 18:24] = 100.
    shuffled = windows[:, torch.randperm(10, generator=generator)]
    with torch.no_grad():
        errors = {name: prior.score_errors(value, noise=noise) for name, value in (
            ('demonstration', windows), ('inverted_gravity_probe', bad_gravity),
            ('extreme_velocity_probe', bad_speed), ('shuffled_time_probe', shuffled))}
        # 固定示范尺度仅供配对比较，不写入在线策略误差统计。
        scale = errors['demonstration'].mean(0).clamp_min(1e-4)
        scores = {name: torch.exp(-6*(value/scale).mean(-1)) for name, value in errors.items()}
    result = {name: dict(error_mean=float(value.mean()), reward_mean=float(scores[name].mean()))
              for name, value in errors.items()}
    result['paired_lower_score_fraction'] = {
        name: float((score < scores['demonstration']).float().mean())
        for name, score in scores.items() if name != 'demonstration'}
    result['scope'] = 'synthetic perturbations of heldout windows; not physical bad-motion acceptance'
    return result


def render_initial(raw, output):
    import mujoco
    from PIL import Image, ImageDraw
    model = raw.sim.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    canvas = Image.new('RGB', (5*320, 4*240), 'white')
    try:
        for index in range(20):
            data.qpos[:] = raw.sim.data.qpos[index].cpu().numpy()
            data.qvel[:] = raw.sim.data.qvel[index].cpu().numpy()
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.body('robot/base_link').xpos
            camera.distance, camera.azimuth, camera.elevation = 1.2, 120., -20.
            renderer.update_scene(data, camera=camera)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text((8, 8), f'Env {index+1} / t=0', fill='white')
            canvas.paste(tile, ((index % 5)*320, (index//5)*240))
        canvas.save(output)
    finally:
        renderer.close()


@torch.inference_mode()
def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env_cfg, train_cfg = task_registry.get_cfgs('chuanliantui')
    original, _ = task_registry.get_cfgs('chuanliantui')
    env_cfg.seed = train_cfg.seed = 42
    configure_smp(env_cfg, train_cfg, args.prior)
    assert 'smp_motion' not in original.rewards
    assert original.actions['pd'].ground_start_hold_s == .2
    env, _ = task_registry.make_env('chuanliantui', env_cfg=env_cfg, train_cfg=train_cfg,
                                    num_envs=20, device=args.device)
    raw, report = env.unwrapped, {}
    try:
        term = raw._smp_reward
        root = raw.scene['robot'].data
        initial_qpos, initial_qvel = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
        assert torch.equal(initial_qvel, torch.zeros_like(initial_qvel))
        torch.testing.assert_close(root.root_link_quat_w,
            root.root_link_quat_w.new_tensor([1., 0., 0., 0.]).expand(20, -1))
        torch.testing.assert_close(root.root_link_pos_w[:, 2]-raw.scene.env_origins[:, 2],
                                  torch.full((20,), .25, device=args.device))
        assert raw.command_manager.get_term('standup').ground_start.all()
        assert (raw.command_manager.get_term('standup').command[:, :2] == 0).all()
        command = raw.command_manager.get_term('standup')
        action = raw.action_manager.get_term('pd')
        assert action.cfg.ground_start_hold_s == .2 and action._ground_hold().all()
        assert not command.policy_active.any() and not command.extra_flags()[:, 1].any()
        # 不推进物理：检查0.20秒边界前后，观测标志与实际电机门控一致。
        action.process_actions(torch.zeros(20, 6, device=args.device))
        action.apply_actions()
        assert not action.torques.any()
        for age in (19, 20):
            raw.episode_length_buf.fill_(age)
            assert bool(command.policy_active.all()) == (age == 20)
            assert bool(action._ground_hold().all()) == (age == 19)
            action.apply_actions()
            assert bool((action.torques.abs().amax(-1) > 1e-3).all()) == (age == 20)
        raw.episode_length_buf.zero_()
        report['startup_control'] = dict(ground_start_hold_s=.2,
            inactive_in_all_initial_observations=True, first_requested_torques_zero=True,
            boundary_019s_inactive_020s_active=True,
            non_smp_ground_start_hold_s=original.actions['pd'].ground_start_hold_s)
        # 覆盖1.5秒之后、站稳后、反复重采样；不需要启动长训练验证命令分支。
        ages = raw.episode_length_buf.clone()
        stable = command.stable_time.clone()
        recovered = command.recovered.clone()
        for age, stood in ((151, False), (501, True), (1501, False)):
            raw.episode_length_buf.fill_(age)
            command.stable_time.fill_(.6 if stood else 0.)
            command.recovered.fill_(stood)
            command._resample_command(torch.arange(20, device=args.device))
            command._update_command(torch.arange(20, device=args.device))
            assert (command.command[:, :2] == 0).all()
            assert command.zero_hold.all() and command.standing.all() and not command.heading.any()
        raw.episode_length_buf.copy_(ages)
        command.stable_time.copy_(stable)
        command.recovered.copy_(recovered)
        report['initial'] = dict(root_height_m=.25, quaternion_wxyz=[1, 0, 0, 0],
            all_initial_velocities_zero=True, same_negative_branch_as_teacher=True,
            sampled_distinct_poses=len(torch.unique(initial_qpos, dim=0)),
            qpos=initial_qpos.cpu().tolist())
        if args.render_initial:
            render_initial(raw, args.output.with_name('initial-20.png'))
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train_cfg,
                                                  device=args.device)
        assert isinstance(runner, SmpOnPolicyRunner)
        before = {key: value.detach().clone() for key, value in term.prior.denoiser.state_dict().items()}
        obs = env.get_observations()
        smp_values, task_values = [], []
        # 走真实PPO采样和returns路径，但禁止优化器更新。
        with patch.object(runner.alg.optimizer, 'step', side_effect=AssertionError('Training forbidden')):
            with torch.inference_mode():
                for tick in range(train_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    assert obs['actor'].shape == (20, 39) and obs['critic'].shape == (20, 82)
                    assert torch.isfinite(rewards).all()
                    assert all(torch.isfinite(obs[name]).all() for name in ('actor', 'critic'))
                    if tick < 9:
                        assert not term.last_valid.any()
                    else:
                        assert term.last_valid.all()
                    smp_values.append(float(term.last_reward.mean()*raw.step_dt))
                    task_values.append(float((rewards-term.last_reward*raw.step_dt).mean()))
                runner.alg.compute_returns(obs)
        assert not runner.alg.optimizer.state
        assert int(term.prior.error_normalizer.count) == 0
        assert int(term.prior.error_normalizer.pending_count) == 0
        torch.testing.assert_close(term.prior.error_normalizer.mean_abs, torch.ones(3, device=args.device))
        assert not any(param.requires_grad or param.grad is not None for param in term.prior.parameters())
        for key, value in term.prior.denoiser.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        # 真实在线读取与离线公共函数完全同源。
        torch.testing.assert_close(term.history.frames[:, -1],
                                  motion_features(**term.reader.read()), atol=0, rtol=0)
        saved_frames = term.history.frames[[1, 3]].clone()
        saved_count = term.history.count[[1, 3]].clone()
        raw.reset(env_ids=torch.tensor([0, 2], device=args.device))
        assert not command.policy_active[[0, 2]].any() and action._ground_hold()[[0, 2]].all()
        assert not term.history.count[[0, 2]].any()
        torch.testing.assert_close(term.history.frames[[1, 3]], saved_frames, atol=0, rtol=0)
        torch.testing.assert_close(term.history.count[[1, 3]], saved_count, atol=0, rtol=0)
        with torch.inference_mode():
            for _ in range(10):
                env.step(torch.zeros(20, 6, device=args.device))
        assert term.last_valid.all()
        # 强制倒置其中一个世界，确认终止后只清空它的历史。
        inverted_root = raw.scene['robot'].data.default_root_state[[0]].clone()
        inverted_root[:, :3] = raw.scene.env_origins[[0]]
        inverted_root[:, 2] += .6
        inverted_root[:, 3:7] = inverted_root.new_tensor([0., 1., 0., 0.])
        raw.scene['robot'].write_root_state_to_sim(inverted_root,
                                                  env_ids=torch.tensor([0], device=args.device))
        raw.sim.forward()
        with torch.inference_mode():
            _, _, dones, _ = env.step(torch.zeros(20, 6, device=args.device))
        assert dones[0] and not term.history.count[0] and term.history.count[1] == 10
        with tempfile.TemporaryDirectory() as directory:
            from dataclasses import asdict
            from mjlab.utils.os import dump_yaml
            dump_yaml(Path(directory)/'env.yaml', asdict(env_cfg))
            dump_yaml(Path(directory)/'agent.yaml', asdict(train_cfg))
            path = str(Path(directory)/'untrained_roundtrip.pt')
            runner.save(path)
            normalizer = {key: value.clone() for key, value in term.prior.error_normalizer.state_dict().items()}
            rng = term.generator.get_state().clone()
            term.prior.error_normalizer.mean_abs.fill_(123.)
            term.generator.manual_seed(1)
            runner.load(path)
            for key, value in normalizer.items():
                torch.testing.assert_close(term.prior.error_normalizer.state_dict()[key], value, atol=0, rtol=0)
            assert torch.equal(term.generator.get_state(), rng)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            saved['infos']['smp_normalizer']['mean_abs'].fill_(2.)
            invalid = Path(directory)/'adaptive_scale.pt'
            torch.save(saved, invalid)
            try:
                runner.load(str(invalid))
            except ValueError:
                pass
            else:
                raise AssertionError('Adaptive error statistics accepted by fixed-unit profile')
            from ct_mjlab.smp.ppo import ADAPTIVE_STAND_PROFILE
            saved['infos']['experiment_profile_contract'] = ADAPTIVE_STAND_PROFILE
            saved['infos']['smp_normalizer']['mean_abs'].fill_(1.)
            torch.save(saved, invalid)
            try:
                runner.load(str(invalid))
            except ValueError:
                pass
            else:
                raise AssertionError('Old adaptive profile accepted for fixed-unit resume')
            term.contract['ground_start_hold_s'] = 0.
            try:
                runner.load(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched startup hold accepted')
            term.contract['ground_start_hold_s'] = .2
            term.contract['prior_sha256'] = 'wrong'
            try:
                runner.load(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched prior accepted')
            term.contract['prior_sha256'] = term.digest
            from ct_mjlab.scripts.eval import evaluate_seed
            result, rows, _ = evaluate_seed(SimpleNamespace(
                num_envs=2, start_mode='force', smp_prior=args.prior, device=args.device,
                checkpoint=Path(path), capture=False, steps=4, knockdown_at=None,
                knockdown_speed=(18., 28.)), 42)
            assert len(rows) == 2 and result['extras']['ground_start_envs'] == 2
        # 初始随机策略在真实训练随机化下采样2秒，只测奖励明细，不保存新策略。
        obs, _ = raw.reset(seed=42)
        contributions, episode_ages = [], []
        names = raw.reward_manager.active_terms
        smp_index = names.index('smp_motion')
        with patch.object(runner.alg.optimizer, 'step', side_effect=AssertionError('Training forbidden')):
            for _ in range(200):
                age = (raw.episode_length_buf.clone()+1)*raw.step_dt
                obs, rewards, _, _ = env.step(runner.alg.act(obs))
                values = raw.reward_manager._step_reward*raw.step_dt
                torch.testing.assert_close(values.sum(-1), rewards, atol=2e-6, rtol=2e-5)
                contributions.append(values.cpu().numpy().copy())
                episode_ages.append(age.cpu().numpy().copy())
        values, ages = np.asarray(contributions), np.asarray(episode_ages)
        smp = values[:, :, smp_index]
        task = np.delete(values, smp_index, axis=-1)
        positive, penalty = task.clip(min=0).sum(-1), -task.clip(max=0).sum(-1)
        shares = {}
        for name, mask in (('startup_hold', ages <= .200001), ('rise', ages > .200001)):
            s, p, n = smp[mask], positive[mask], penalty[mask]
            shares[name] = dict(environment_steps=int(mask.sum()),
                smp_share_of_positive=float(s.sum()/(s.sum()+p.sum())),
                smp_share_of_absolute=float(s.sum()/(s.sum()+p.sum()+n.sum())),
                mean_step_smp=float(s.mean()), mean_step_task_net=float((p-n).mean()),
                contributions={key: float(values[:, :, i][mask].sum()) for i, key in enumerate(names)})
        np.savez_compressed(args.output.with_name('initial-policy-rewards.npz'),
                            contributions=values, episode_age_s=ages)
        report['initial_random_policy_reward_shares'] = shares
        assert not runner.alg.optimizer.state and int(term.prior.error_normalizer.count) == 0
        assert not term.prior.error_normalizer.pending_count and not term.prior.error_normalizer.pending_sum.any()
        for key, value in term.prior.denoiser.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        if args.dataset:
            report['score_probes'] = score_probes(term, args.dataset)
        report.update(passed=True, num_envs=20, physics_dt=raw.physics_dt, policy_dt=raw.step_dt,
            observation_shapes={'actor': [20, 39], 'critic': [20, 82]}, actions=6,
            ppo_sample_steps=24, optimizer_updates=0, prior_frozen=True,
            normalizer_count_after_rollout=0, fixed_unit_scale=True,
            adaptive_checkpoint_statistics_rejected=True, partial_reset_and_warmup=True,
            old_adaptive_profile_rejected=True, initial_policy_diagnostic_seconds=2.,
            inverted_autoreset=True, checkpoint_normalizer_rng_roundtrip=True,
            eval_entry_finite_steps=4, configuration_yaml_serialization=True,
            mismatched_prior_rejected=True, task_configuration_unchanged_without_opt_in=True,
            mismatched_startup_hold_rejected=True,
            commands_zero_before_and_after_standing_and_resampling=True,
            mean_scaled_smp_reward=float(np.mean(smp_values)), mean_task_reward=float(np.mean(task_values)),
            prior_sha256=term.digest,
            sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [Path(__file__), ROOT/'mjlab_training/src/ct_mjlab/smp/ppo.py',
                         ROOT/'mjlab_training/src/ct_mjlab/utils/task_registry.py',
                         ROOT/'mjlab_training/src/ct_mjlab/scripts/train.py',
                         ROOT/'mjlab_training/src/ct_mjlab/scripts/check.py',
                         ROOT/'mjlab_training/src/ct_mjlab/scripts/eval.py',
                         ROOT/'mjlab_training/src/ct_mjlab/scripts/play.py']},
            scope='finite rollout/returns/serialization and synthetic score probes; no PPO training or behavior acceptance')
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key not in ('initial', 'sources')}, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, action='append')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--render-initial', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
