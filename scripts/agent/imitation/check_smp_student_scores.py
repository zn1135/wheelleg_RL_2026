#!/usr/bin/env python3
"""Matched-pose physical negatives and frozen SMP ranking; never trains a policy."""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.features import MotionStateReader, motion_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def passive(args):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from ct_mjlab.envs import task_registry
    from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg

    class PassiveAction(HuananhuPdAction):
        def apply_actions(self):
            super().apply_actions()  # 保留原物理弹簧，只把六个主动电机置零。
            self._torques.zero_()
            self._motor_response_torque.zero_()
            self._entity.set_joint_effort_target(self._torques, joint_ids=self._physical_joint_ids)

    @dataclass(kw_only=True)
    class PassiveCfg(HuananhuPdActionCfg):
        class_type: type = field(default_factory=lambda: PassiveAction)

    def verify_force(env):
        force = env.sim.data.actuator_force[:, env._passive_actuators]
        env._passive_max_force = max(env._passive_max_force, float(force.abs().max()))
        return torch.zeros(env.num_envs, device=env.device)

    data = MotionDataset([args.initials])
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    cfg.auto_reset = False
    cfg.rewards, cfg.terminations = {}, {}
    cfg.metrics = {'force': MetricsTermCfg(func=verify_force, per_substep=True, reduce='last')}
    cfg.actions['pd'] = PassiveCfg(**{k: v for k, v in vars(cfg.actions['pd']).items() if k != 'class_type'})
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                   num_envs=len(data.episodes), device=args.device)
    try:
        raw = env.unwrapped
        # 初始化测量字段后再reset，避免复位时的指标调用缺字段。
        from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
        raw._passive_actuators = torch.tensor([raw.sim.mj_model.actuator('robot/'+name+'_motor').id
                                               for name in ACTUATED_JOINTS], device=args.device)
        raw._passive_max_force = 0.
        raw.reset(seed=42)
        states = []
        for e in data.episodes:
            with np.load(e['path']) as trace:
                states.append((trace['initial_qpos'].copy(), trace['initial_qvel'].copy()))
        qpos = torch.tensor(np.asarray([q for q, v in states]), device=args.device)
        qvel = torch.tensor(np.asarray([v for q, v in states]), device=args.device)
        qpos[:, :2] = raw.scene.env_origins[:, :2]
        raw.sim.data.qpos.copy_(qpos)
        raw.sim.data.qvel.copy_(qvel)
        raw.sim.forward()
        reader = MotionStateReader(raw)
        buffers = {k: [] for k in ('root', 'motor_q', 'motor_dq', 'wheel_position_w', 'ground_height', 'features')}
        for tick in range(300):
            env.step(torch.zeros(raw.num_envs, 6, device=args.device))
            for k, value in dict(reader.read(), features=reader.features()).items():
                buffers[k].append(value.cpu().numpy().copy())
        assert raw.physics_dt == .002 and abs(raw.step_dt-.01) < 1e-8
        assert raw._passive_max_force == 0., raw._passive_max_force
        metadata = dict(kind='passive', physics_dt=raw.physics_dt, action_dt=raw.step_dt,
            xml_sha256=data.xml_sha256, max_actual_motor_force_nm=raw._passive_max_force,
            episode_ids=[e['episode_id'] for e in data.episodes],
            group_ids=[e['group_id'] for e in data.episodes], optimizer_updates=0,
            script_sha256=digest(__file__))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **{k: np.asarray(v) for k, v in buffers.items()},
            time=(np.arange(300)+1)*.01, initial_qpos=qpos.cpu().numpy(), initial_qvel=qvel.cpu().numpy(),
            metadata=json.dumps(metadata))
        print(json.dumps(metadata, indent=2))
    finally:
        env.close()


def verified_features(trace):
    features = motion_features(**{k: torch.from_numpy(trace[k].copy()).float() for k in
        ('root', 'motor_q', 'motor_dq', 'wheel_position_w', 'ground_height')})
    torch.testing.assert_close(features, torch.from_numpy(trace['features']), atol=2e-6, rtol=2e-6)
    return features


def windows(features, time, left, right):
    starts = ((time[:-9] >= left-1e-8) & (time[9:] <= right+1e-8)).nonzero().flatten()[::10]
    return features[starts[:, None]+torch.arange(10)]


@torch.inference_mode()
def score(args):
    from ct_mjlab.smp.prior import load_prior
    dataset = MotionDataset([args.dataset])
    plan = json.loads(args.plan.read_text())
    assert plan['dataset_fingerprint'] == dataset.fingerprint
    prior, saved = load_prior(args.prior, args.device)
    assert saved['dataset']['fingerprint'] == dataset.fingerprint
    assert saved['iteration'] == 10000
    prior.freeze()
    assert int(prior.error_normalizer.count) == 0
    torch.testing.assert_close(prior.error_normalizer.mean_abs, torch.ones_like(prior.error_normalizer.mean_abs))
    before = {k: v.clone() for k, v in prior.state_dict().items()}
    pools = {k: {} for k in ('success_late', 'success_rise', 'oscillation', 'passive')}
    calibration = {'rise': [], 'standing': []}
    for split, episodes in dataset.splits.items():
        for e in episodes:
            rise = windows(e['features'], e['time'], .2, 2.)
            late = windows(e['features'], e['time'], 15., 20.)
            if split == 'train':
                calibration['rise'].append(rise)
                calibration['standing'].append(windows(e['features'], e['time'], 2.01, 20.))
            else:
                pools['success_rise'].setdefault(e['group_id'], []).append(rise)
                pools['success_late'].setdefault(e['group_id'], []).append(late)
    report = json.loads((args.oscillation/'result.json').read_text())
    assert report['completed'] and report['compensation'] == 'none' and report['optimizer_updates'] == 0
    assert report['xml_sha256'] == dataset.xml_sha256
    assert digest(args.oscillation/'trace.npz') == report['trace_sha256']
    id_to_group = {e['episode_id']: e['group_id'] for e in dataset.splits['validation']}
    initial_states = {}
    for e in dataset.splits['validation']:
        with np.load(e['path']) as trace:
            initial_states[e['episode_id']] = (trace['initial_qpos'].copy(), trace['initial_qvel'].copy())
    used_failures = []
    with np.load(args.oscillation/'trace.npz') as trace:
        features = verified_features(trace)
        time = torch.from_numpy(trace['time'].copy())
        for r in report['rows']:
            q, dq = initial_states[r['episode_id']]
            np.testing.assert_allclose(trace['initial_qpos'][r['env'], 2:], q[2:], atol=1e-6)
            np.testing.assert_allclose(trace['initial_qvel'][r['env']], dq, atol=1e-6)
            if (r['full_20s'] and r['finite'] and not r['final_balance_v31'] and
                    r['wheel_delay_s'] == .01 and r['last5s_leg_velocity_rms_rad_s'] > .1):
                index, group = r['env'], id_to_group[r['episode_id']]
                assert trace['alive'][:, index].all()
                pools['oscillation'].setdefault(group, []).append(windows(features[:, index], time, 15., 20.))
                used_failures.append(r)
    with np.load(args.passive) as trace:
        metadata = json.loads(str(trace['metadata']))
        assert metadata['max_actual_motor_force_nm'] == 0 and metadata['physics_dt'] == .002
        assert metadata['xml_sha256'] == dataset.xml_sha256
        features, time = verified_features(trace), torch.from_numpy(trace['time'].copy())
        for index, group in enumerate(metadata['group_ids']):
            q, dq = initial_states[metadata['episode_ids'][index]]
            np.testing.assert_allclose(trace['initial_qpos'][index, 2:], q[2:], atol=1e-6)
            np.testing.assert_allclose(trace['initial_qvel'][index], dq, atol=1e-6)
            pool = windows(features[:, index], time, 1., 3.)
            settled = ((pool[:, :, 0] < .2).all(-1) &
                       (pool[:, :, 4:7].norm(dim=-1) < .1).all(-1) &
                       (pool[:, :, 7:10].norm(dim=-1) < .25).all(-1) &
                       (pool[:, :, 18:24].abs().amax(-1) < .5).all(-1))
            if settled.any():
                pools['passive'][group] = [pool[settled]]
    available = {name: {group: sum(map(len, values)) for group, values in groups.items()}
                 for name, groups in pools.items()}
    common = sorted(set.intersection(*(set(groups) for groups in pools.values())))
    if not common:
        raise ValueError('No common heldout pose with all real trajectory classes')
    selected, locations = {}, {}
    for name, groups in pools.items():
        samples, origins = [], []
        for group in common:
            pool = torch.cat(groups[group])
            if len(pool) < plan['windows_per_pose']:
                raise ValueError(f'Too few nonoverlapping windows: {name}/{group}')
            indices = torch.linspace(0, len(pool)-1, plan['windows_per_pose']).round().long()
            samples.append(pool[indices])
            origins.extend([group]*len(indices))
        selected[name], locations[name] = torch.cat(samples).to(args.device), origins
    generator = torch.Generator().manual_seed(2839)
    cal = []
    for values in calibration.values():
        pool = torch.cat(values)
        cal.append(pool[torch.randperm(len(pool), generator=generator)[:256]])
    cal = torch.cat(cal).to(args.device)
    cal_noise = torch.randn(4, 3, len(cal), 300, generator=generator).to(args.device)
    cal_errors = torch.stack([prior.score_errors(cal, noise=draw) for draw in cal_noise])
    scales = {'unit': torch.ones(3, device=args.device),
              'train_demo_balanced_diagnostic': cal_errors.mean((0, 1)).clamp_min(1e-4)}
    count = len(next(iter(selected.values())))
    noise = torch.randn(4, 3, count, 300, generator=torch.Generator().manual_seed(823)).to(args.device)
    scores, arrays, summaries = {}, {}, {}
    for name, batch in selected.items():
        errors = torch.stack([prior.score_errors(batch, noise=draw) for draw in noise])
        assert torch.isfinite(errors).all()
        scores[name] = {key: torch.exp(-6*(errors/scale).mean(-1)).mean(0) for key, scale in scales.items()}
        summaries[name] = dict(evaluated_windows=count, height_mean_m=float(batch[:, :, 0].mean()),
            raw_errors_by_noise=errors.mean((0, 1)).cpu().tolist(),
            scores={key: dict(mean=float(v.mean()), median=float(v.median()),
                             p10=float(v.quantile(.1)), p90=float(v.quantile(.9)),
                             zero_count=int((v == 0).sum())) for key, v in scores[name].items()})
        arrays[name+'_windows'] = batch.cpu().numpy()
        arrays[name+'_errors'] = errors.cpu().numpy()
        for key, v in scores[name].items():
            arrays[name+'_'+key] = v.cpu().numpy()
    rankings = {}
    for scale in scales:
        rankings[scale] = {}
        for good in ('success_late', 'success_rise'):
            for bad in ('oscillation', 'passive'):
                x, y = scores[good][scale], scores[bad][scale]
                auc = float(((x[:, None] > y).float()+.5*(x[:, None] == y).float()).mean())
                rankings[scale][good+'_vs_'+bad] = dict(auc=auc, mean_gap=float(x.mean()-y.mean()),
                    passed=bool(auc >= plan['ranking_threshold']['auc_min'] and x.mean() > y.mean()),
                    pose_mean_wins=sum(float(x[i:i+10].mean()) > float(y[i:i+10].mean())
                                       for i in range(0, count, 10)))
    for k, value in before.items():
        torch.testing.assert_close(prior.state_dict()[k], value, atol=0, rtol=0)
    inputs = [args.dataset, args.plan, args.prior, args.passive, args.oscillation/'trace.npz', args.oscillation/'result.json']
    result = dict(comparison_completed=True, optimizer_updates=0, prior_unchanged=True,
        normalizer_count=int(prior.error_normalizer.count), iteration=saved['iteration'],
        dataset_fingerprint=dataset.fingerprint, prior_sha256=digest(args.prior),
        common_validation_pose_groups=common, omitted_pose_groups=sorted(set(id_to_group.values())-set(common)),
        available_windows=available, locations=locations, oscillatory_failure_rows=used_failures,
        fixed_scales={k: v.cpu().tolist() for k, v in scales.items()},
        calibration=dict(split='train', rise=256, standing=256, noise_draws=4),
        groups=summaries, ranking=rankings,
        primary_passed=all(rankings['unit']['success_late_vs_'+bad]['passed'] for bad in ('oscillation', 'passive')),
        inputs_sha256={str(p): digest(p) for p in inputs}, script_sha256=digest(__file__),
        scope='Heldout matched poses, nonoverlapping windows, equal samples per pose and fixed shared noise; diagnostic scales only. No PPO, reward changes or hardware validation.')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'scored-windows.npz', **arrays, calibration_errors=cal_errors.cpu().numpy())
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(primary_passed=result['primary_passed'], poses=len(common), groups=summaries, ranking=rankings), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('passive')
    p.add_argument('--initials', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    s = commands.add_parser('score')
    for name in ('dataset', 'plan', 'prior', 'passive', 'oscillation', 'output'):
        s.add_argument('--'+name, type=Path, required=True)
    for command in (p, s):
        command.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    (passive if args.command == 'passive' else score)(args)


if __name__ == '__main__':
    main()
