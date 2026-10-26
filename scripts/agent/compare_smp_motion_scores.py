#!/usr/bin/env python3
"""Compare real teacher, passive and failed-policy motion; no parameter updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def rollout(args):
    # 子进程可加载原300轮冻结源码，避免用新零速配置改写历史失败轨迹。
    from ct_mjlab.smp.features import MotionStateReader
    if args.kind == 'passive':
        from dataclasses import dataclass
        from mjlab.envs import ManagerBasedRlEnv
        from ct_mjlab.teacher.env import TeacherAction, TeacherActionCfg, make_teacher_env_cfg
        from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force

        class PassiveAction(TeacherAction):
            def apply_actions(self):
                self._entity.set_joint_effort_target(torch.zeros_like(self.torques), joint_ids=self.motor_ids)
                self._env.sim.data.ctrl[:, self.spring_ids] = gas_spring_force(
                    self._env.sim.data.qpos[:, self.sqadr], self._env.sim.data.qvel[:, self.svadr],
                    torch.zeros_like(self._env.sim.data.qpos[:, self.sqadr]))

        @dataclass(kw_only=True)
        class PassiveCfg(TeacherActionCfg):
            def build(self, env):
                return PassiveAction(self, env)

        cfg = make_teacher_env_cfg(initial='ground', num_envs=args.num_envs, seed=42)
        cfg.actions = {'passive': PassiveCfg(entity_name='robot', initial='ground')}
        raw = env = ManagerBasedRlEnv(cfg, device=args.device)
        obs, _ = env.reset(seed=42)
        policy = lambda obs: torch.zeros(args.num_envs, 3, device=args.device)
    else:
        from ct_mjlab.envs import task_registry
        from ct_mjlab.smp.ppo import configure_smp
        cfg, train_cfg = task_registry.get_cfgs('chuanliantui', play=True)
        cfg.seed = 42
        cfg.auto_reset = False
        configure_smp(cfg, train_cfg, args.prior, play=True)
        env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train_cfg,
                                        num_envs=args.num_envs, device=args.device)
        raw = env.unwrapped
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train_cfg,
                                                  device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = raw.reset(seed=42)
    try:
        reader = MotionStateReader(raw)
        live = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        command = (np.zeros((args.num_envs, 3), dtype=np.float32) if args.kind == 'passive'
                   else raw.command_manager.get_term('standup').command.cpu().numpy())
        signature = hashlib.sha256(
            np.zeros(args.num_envs, dtype=np.int32).tobytes()
            + np.ones(args.num_envs, dtype=np.bool_).tobytes()
            + command.astype(np.float32).tobytes()
            + raw.sim.data.qpos.cpu().numpy().astype(np.float32).tobytes()).hexdigest()
        buffers = {name: [] for name in ('root', 'motor_q', 'motor_dq', 'wheel_position_w',
                                         'ground_height', 'features', 'alive', 'time')}
        stride = round(.01/raw.step_dt)
        for tick in range(round((3. if args.kind == 'passive' else 6.)/raw.step_dt)):
            actions = policy(obs)
            if args.kind == 'passive':
                obs, _, _, _, _ = env.step(actions)
                done = torch.zeros_like(live)
            else:
                obs, _, done, _ = env.step(actions)
                done = done.bool()
            if (tick+1) % stride == 0:
                state = reader.read()
                for name, value in state.items():
                    buffers[name].append(value.cpu().numpy().copy())
                buffers['features'].append(reader.features().cpu().numpy().copy())
                buffers['alive'].append(live.cpu().numpy().copy())
                buffers['time'].append((tick+1)*raw.step_dt)
            live &= ~done
            if not live.any():
                break
            if done.any():
                obs, _ = raw.reset(env_ids=done.nonzero().flatten())
        from ct_mjlab.envs.chuanliantui.model import MODEL_XML
        import ct_mjlab.smp.ppo as ppo_module
        metadata = dict(kind=args.kind, seed=42, num_envs=args.num_envs, sample_dt=.01,
                        physics_dt=raw.physics_dt, initial_state_signature=signature,
                        checkpoint_sha256=digest(args.checkpoint), prior_sha256=digest(args.prior),
                        xml_sha256=digest(MODEL_XML), runtime_smp_ppo_sha256=digest(ppo_module.__file__),
                        ended_environments=int((~live).sum()), optimizer_updates=0)
        np.savez_compressed(args.output, **{key: np.asarray(value) for key, value in buffers.items()},
                            metadata=json.dumps(metadata))
    finally:
        env.close()


def windows_from_rollout(path):
    from ct_mjlab.smp.features import motion_features
    with np.load(path) as data:
        metadata = json.loads(str(data['metadata']))
        state = {key: torch.from_numpy(data[key].copy()).float() for key in
                 ('root', 'motor_q', 'motor_dq', 'wheel_position_w', 'ground_height')}
        features = motion_features(**state)
        torch.testing.assert_close(features, torch.from_numpy(data['features']), atol=2e-6, rtol=2e-6)
        alive = torch.from_numpy(data['alive'].copy())
        time = torch.from_numpy(data['time'].copy())
    windows = features.permute(1, 0, 2).unfold(1, 10, 1).transpose(-1, -2)
    valid = alive.T.unfold(1, 10, 1).all(-1)
    valid &= (time[:-9] >= (.2 if metadata['kind'] == 'failed-policy' else 1.))[None]
    if metadata['kind'] == 'passive':
        # 真正被动物理落稳的低姿态窗口；不固定根或修改速度伪造“趴着不动”。
        valid &= (windows[:, :, :, 0] < .2).all(-1)
        valid &= (windows[:, :, :, 4:7].norm(dim=-1) < .1).all(-1)
        valid &= (windows[:, :, :, 7:10].norm(dim=-1) < .25).all(-1)
        valid &= (windows[:, :, :, 18:24].abs().amax(-1) < .5).all(-1)
    selected = windows[valid]
    if not len(selected):
        raise RuntimeError(f'No qualified {metadata["kind"]} windows; do not substitute synthetic motion')
    return selected, metadata


@torch.inference_mode()
def rescore_cached(args):
    """Evaluate a new prior on preserved physical negatives without old prior weights."""
    from ct_mjlab.smp.dataset import MotionDataset
    from ct_mjlab.smp.prior import load_prior
    dataset = MotionDataset(args.dataset)
    prior, saved = load_prior(args.prior, args.device)
    prior.freeze()
    assert saved['dataset']['fingerprint'] == dataset.fingerprint
    original = {key: value.clone() for key, value in prior.state_dict().items()}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    args.output.mkdir(parents=True, exist_ok=False)

    def teacher_windows(split):
        groups = {'teacher_rise': [], 'teacher_standing': []}
        for episode in dataset.splits[split]:
            if episode['behavior'] != 'standup':
                continue
            frames, time = episode['features'], episode['time']
            windows = frames.unfold(0, 10, 1).transpose(-1, -2)
            groups['teacher_rise'].append(windows[(time[:-9] >= .2) & (time[9:] <= 2.)])
            groups['teacher_standing'].append(windows[time[:-9] >= 2.])
        return {name: torch.cat(values) for name, values in groups.items()}

    groups, metadata = teacher_windows('validation'), {}
    input_hashes = {str(path): digest(path) for path in args.dataset}
    for kind in ('passive', 'failed-policy'):
        path = args.cached_rollouts/f'{kind}.npz'
        groups[kind], metadata[kind] = windows_from_rollout(path)
        assert metadata[kind]['kind'] == kind and metadata[kind]['optimizer_updates'] == 0
        assert metadata[kind]['xml_sha256'] == dataset.xml_sha256
        input_hashes[str(path)] = digest(path)
    reference = json.loads(args.legacy_evaluation.read_text())
    assert metadata['failed-policy']['initial_state_signature'] == reference['per_seed']['42']['initial_state_signature']
    assert metadata['failed-policy']['checkpoint_sha256'] == digest(args.checkpoint)
    assert metadata['failed-policy']['ended_environments'] == 64
    student_good, origins = [], {}
    if args.student_source:
        from ct_mjlab.smp.features import motion_features
        source = json.loads((args.student_source/'source-check.json').read_text())
        assert source['completed'] and source['actor_unchanged'] and source['optimizer_updates'] == 0
        assert source['xml_sha256'] == dataset.xml_sha256
        pools = {'student_rise': [], 'student_balancing_with_drift': []}
        locations = {name: [] for name in pools}
        for seed, result in source['seeds'].items():
            path = args.student_source/f'policy-{seed}.npz'
            assert digest(path) == result['source_trace_sha256']
            input_hashes[str(path)] = digest(path)
            with np.load(path) as trace:
                state = {key: torch.from_numpy(trace[key].copy()).float() for key in
                         ('root', 'motor_q', 'motor_dq', 'wheel_position_w', 'ground_height')}
                features = motion_features(**state)
                torch.testing.assert_close(features, torch.from_numpy(trace['features']), atol=2e-6, rtol=2e-6)
                windows = features.permute(1, 0, 2).unfold(1, 10, 1).transpose(-1, -2)
                valid = torch.from_numpy(trace['alive'].copy()).T.unfold(1, 10, 1).all(-1)
                valid &= torch.tensor([row['candidate_qualified'] for row in result['rows']])[:, None]
                time = torch.from_numpy(trace['time'].copy())
                masks = {'student_rise': valid & (time[:-9] >= .2)[None] & (time[9:] <= 2.)[None],
                         'student_balancing_with_drift': valid & (time[:-9] >= 2.)[None]}
                for name, mask in masks.items():
                    pools[name].append(windows[mask])
                    indices = mask.nonzero()
                    locations[name].append(torch.cat((torch.full((len(indices), 1), int(seed)), indices), -1))
        for name, values in pools.items():
            groups[name] = torch.cat(values)
            origins[name] = torch.cat(locations[name])
            if not len(groups[name]):
                raise ValueError('No qualified student-source windows')
        student_good = list(pools)
    # 旧尺度只作历史参照，不冒充新先验在线训练得到的尺度。
    scales = {'unit': torch.ones(3, device=args.device),
              'legacy_ppo300_reference': checkpoint['infos']['smp_normalizer']['mean_abs'].to(args.device)}
    calibration = teacher_windows('train')
    generator = torch.Generator().manual_seed(2839)
    selected_calibration = torch.cat([pool[torch.randperm(len(pool), generator=generator)[:256]]
                                      for pool in calibration.values()]).to(args.device)
    cal_noise = torch.randn(4, 3, len(selected_calibration), 300, generator=generator).to(args.device)
    cal_errors = torch.stack([prior.score_errors(selected_calibration, noise=draw) for draw in cal_noise])
    scales['train_demo_balanced_diagnostic'] = cal_errors.mean((0, 1)).clamp_min(1e-4)
    generator = torch.Generator().manual_seed(823)
    noise = torch.randn(4, 3, 256, 300, generator=generator).to(args.device)
    report = dict(comparison_completed=True, optimizer_updates=0, prior_sha256=digest(args.prior),
        dataset_fingerprint=dataset.fingerprint, runtime_metadata=metadata, inputs=input_hashes,
        fixed_scales={key: value.cpu().tolist() for key, value in scales.items()},
        calibration=dict(split='train', rise_windows=256, standing_windows=256,
                         scope='diagnostic only; no reward/normalizer changes'), groups={}, ranking_auc={})
    arrays = {'calibration_windows': selected_calibration.cpu().numpy(),
              'calibration_errors': cal_errors.cpu().numpy()}
    scores = {}
    for name, pool in groups.items():
        indices = torch.randperm(len(pool), generator=generator)[:256]
        selected = pool[indices].to(args.device)
        errors = torch.stack([prior.score_errors(selected, noise=draw[:, :len(selected)]) for draw in noise])
        scores[name] = {key: torch.exp(-6*(errors/scale).mean(-1)).mean(0) for key, scale in scales.items()}
        report['groups'][name] = dict(available_windows=len(pool), evaluated_windows=len(selected),
            height_mean_m=float(selected[:, :, 0].mean()), raw_error_mean=float(errors.mean()),
            raw_errors_by_noise=errors.mean((0, 1)).cpu().tolist(),
            scores={key: dict(mean=float(value.mean()), median=float(value.median()),
                             p10=float(value.quantile(.1)), p90=float(value.quantile(.9)))
                    for key, value in scores[name].items()})
        arrays[name+'_windows'], arrays[name+'_errors'] = selected.cpu().numpy(), errors.cpu().numpy()
        arrays[name+'_pool_indices'] = indices.numpy()
        if name in origins:
            arrays[name+'_seed_env_start_frame'] = origins[name][indices].numpy()
    for scale in scales:
        report['ranking_auc'][scale] = {}
        for good in ('teacher_rise', 'teacher_standing', *student_good):
            for bad in ('passive', 'failed-policy'):
                x, y = scores[good][scale][:, None], scores[bad][scale][None]
                report['ranking_auc'][scale][good+'_vs_'+bad] = float(
                    ((x > y).float()+.5*(x == y).float()).mean())
    if 'student_rise' in origins:
        report['student_rise_time_bins'] = []
        start_time = (torch.from_numpy(arrays['student_rise_seed_env_start_frame'][:, 2]).to(args.device)+1)*.01
        for left, right in ((.2, .6), (.6, 1.), (1., 1.5), (1.5, 2.)):
            mask = (start_time >= left-1e-6) & (start_time < right-1e-6)
            if mask.any():
                x, y = scores['student_rise']['unit'][mask, None], scores['passive']['unit'][None]
                report['student_rise_time_bins'].append(dict(start_s=left, stop_s=right,
                    windows=int(mask.sum()), unit_mean=float(x.mean()),
                    unit_auc_vs_passive=float(((x > y).float()+.5*(x == y).float()).mean())))
    for key, value in original.items():
        torch.testing.assert_close(prior.state_dict()[key], value, atol=0, rtol=0)
    np.savez_compressed(args.output/'scored-windows.npz', **arrays)
    report.update(source_sha256=digest(__file__), prior_parameters_and_statistics_unchanged=True,
        student_source=str(args.student_source) if args.student_source else None,
        scope='New prior, new heldout teacher positives, preserved real physical negatives and optional physically qualified student trajectories; historical scale is a reference, not new PPO statistics. No training or behavioral acceptance.')
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'runtime_metadata'}, indent=2))


@torch.inference_mode()
def compare(args):
    from ct_mjlab.smp.dataset import MotionDataset
    from ct_mjlab.smp.prior import load_prior
    args.output.mkdir(parents=True, exist_ok=False)
    # 冻结包按原目录层级还原，避免模型按parents定位到历史logs下的错误资产路径。
    legacy = args.output.resolve()/'legacy-runtime'
    shutil.copytree(args.policy_source/'ct_mjlab', legacy/'mjlab_training/src/ct_mjlab',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (legacy/'resources').symlink_to(ROOT/'resources', target_is_directory=True)
    for kind, source in (('passive', ROOT/'mjlab_training/src'), ('failed-policy', legacy/'mjlab_training/src')):
        env = dict(os.environ, PYTHONPATH=str(source), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        command = [sys.executable, str(Path(__file__).resolve()), '--kind', kind,
                   '--prior', str(args.prior.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                   '--device', args.device, '--num-envs', '64', '--output', str(args.output/f'{kind}.npz')]
        with (args.output/f'{kind}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    dataset = MotionDataset(args.dataset)
    groups = {'teacher_rise': [], 'teacher_standing': []}
    for episode in dataset.splits['validation']:
        if episode['behavior'] != 'standup':
            continue
        f, time = episode['features'], episode['time']
        windows = f[torch.arange(len(f)-9)[:, None]+torch.arange(10)]
        groups['teacher_rise'].append(windows[(time[:-9] >= .2) & (time[9:] <= 2.)])
        groups['teacher_standing'].append(windows[time[:-9] >= 2.])
    groups = {key: torch.cat(value) for key, value in groups.items()}
    metadata = {}
    for kind in ('passive', 'failed-policy'):
        groups[kind], metadata[kind] = windows_from_rollout(args.output/f'{kind}.npz')
    reference = json.loads(args.legacy_evaluation.read_text())
    assert metadata['failed-policy']['initial_state_signature'] == reference['per_seed']['42']['initial_state_signature']
    assert metadata['failed-policy']['ended_environments'] == 64
    prior, saved = load_prior(args.prior, args.device)
    prior.freeze()
    original_stats = {key: value.clone() for key, value in prior.error_normalizer.state_dict().items()}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['infos']['smp']['prior_sha256'] == digest(args.prior)
    assert saved['dataset']['fingerprint'] == dataset.fingerprint
    assert all(meta['xml_sha256'] == dataset.xml_sha256 for meta in metadata.values())
    scales = {'unit': torch.ones(3, device=args.device),
              'ppo300': checkpoint['infos']['smp_normalizer']['mean_abs'].to(args.device)}
    generator = torch.Generator().manual_seed(823)
    noise = torch.randn(4, 3, 256, 300, generator=generator).to(args.device)
    report = dict(comparison_completed=True, optimizer_updates=0, runtime_metadata=metadata,
                  prior_sha256=digest(args.prior), policy_sha256=digest(args.checkpoint),
                  fixed_scales={key: value.cpu().tolist() for key, value in scales.items()}, groups={})
    arrays, scores = {}, {}
    for name, pool in groups.items():
        selected = pool[torch.randperm(len(pool), generator=generator)[:256]].to(args.device)
        errors = torch.stack([prior.score_errors(selected, noise=draw[:, :len(selected)]) for draw in noise])
        scores[name] = {key: torch.exp(-6*(errors/value).mean(-1)).mean(0) for key, value in scales.items()}
        report['groups'][name] = dict(available_windows=len(pool), evaluated_windows=len(selected),
            height_mean_m=float(selected[:, :, 0].mean()), raw_error_mean=float(errors.mean()),
            raw_errors_by_noise=errors.mean((0, 1)).cpu().tolist(),
            scores={key: dict(mean=float(value.mean()), median=float(value.median()),
                             p10=float(value.quantile(.1)), p90=float(value.quantile(.9)))
                    for key, value in scores[name].items()})
        arrays[name+'_windows'] = selected.cpu().numpy()
        arrays[name+'_errors'] = errors.cpu().numpy()
    report['ranking_auc'] = {}
    for good in ('teacher_rise', 'teacher_standing'):
        for bad in ('passive', 'failed-policy'):
            x, y = scores[good]['ppo300'][:, None], scores[bad]['ppo300'][None]
            report['ranking_auc'][good+'_vs_'+bad] = float(((x > y).float()+.5*(x == y).float()).mean())
    for key, value in original_stats.items():
        torch.testing.assert_close(prior.error_normalizer.state_dict()[key], value, atol=0, rtol=0)
    np.savez_compressed(args.output/'scored-windows.npz', **arrays)
    report['source_sha256'] = digest(__file__)
    report['scope'] = 'real motion score comparison under fixed scales; not PPO training or proof of reward sufficiency'
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'runtime_metadata'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, action='append')
    parser.add_argument('--policy-source', type=Path)
    parser.add_argument('--legacy-evaluation', type=Path)
    parser.add_argument('--cached-rollouts', type=Path,
                        help='Rescore preserved real negatives with a new prior; no rollout or training')
    parser.add_argument('--student-source', type=Path,
                        help='Add qualified student trajectories from check_smp_policy_source.py')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--kind', choices=('passive', 'failed-policy'))
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind:
        rollout(args)
    elif args.cached_rollouts:
        rescore_cached(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()
