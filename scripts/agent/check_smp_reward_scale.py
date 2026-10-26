#!/usr/bin/env python3
"""Measure actual reward contributions on live student-PD teacher motion; no learning."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.ppo import configure_smp
from check_lqr_student_closed_loop import BridgeCfg, digest, teacher_failed
from check_smp_policy_source import balance_substep

ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = MotionDataset([args.initials])
    records = []
    for episode in dataset.episodes:
        with np.load(episode['path']) as trace:
            records.append((trace['initial_qpos'].copy(), trace['initial_qvel'].copy()))
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    configure_smp(cfg, train, args.prior, play=False)
    cfg.seed, cfg.auto_reset = 42, False
    cfg.actions['pd'] = BridgeCfg(**{k: v for k, v in vars(cfg.actions['pd']).items() if k != 'class_type'})
    cfg.actions['pd'].action_delay_s = None
    cfg.actions['pd'].randomize_delays = False
    cfg.actions['pd'].action_delay_choices_s = (.002, .004)
    cfg.terminations['teacher_failed'] = TerminationTermCfg(func=teacher_failed)
    cfg.metrics['source_balance'] = MetricsTermCfg(func=balance_substep, per_substep=True, reduce='last')
    pairs = [(leg, wheel) for leg in (.002, .004) for wheel in (.002, .004)]
    count = len(records)*len(pairs)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                   num_envs=count, device=args.device)
    try:
        raw = env.unwrapped
        raw.reset(seed=42)
        action, term = raw.action_manager.get_term('pd'), raw._smp_reward
        before = {k: v.clone() for k, v in term.prior.state_dict().items()}
        qpos = torch.tensor(np.asarray([q for pair in pairs for q, v in records]), device=args.device)
        qvel = torch.tensor(np.asarray([v for pair in pairs for q, v in records]), device=args.device)
        qpos[:, :2] = raw.scene.env_origins[:, :2]
        raw.sim.data.qpos.copy_(qpos)
        raw.sim.data.qvel.copy_(qvel)
        raw.sim.forward()
        action.action_delay_steps[:] = torch.tensor([round(a/.002) for a, b in pairs], device=args.device).repeat_interleave(len(records))
        action.wheel_action_delay_steps[:] = torch.tensor([round(b/.002) for a, b in pairs], device=args.device).repeat_interleave(len(records))
        raw._source_weight = float(raw.sim.get_default_field('body_mass')[raw.scene['robot'].indexing.body_ids].sum())*9.81
        raw._source_probe = dict(live=torch.ones(count, dtype=torch.bool, device=args.device),
            hold=torch.zeros(count, dtype=torch.long, device=args.device),
            strict_run=torch.zeros(count, dtype=torch.long, device=args.device),
            strict_ever=torch.zeros(count, dtype=torch.bool, device=args.device))
        names = raw.reward_manager.active_terms
        index = names.index('smp_motion')
        buffers = {k: [] for k in ('contributions', 'score', 'valid', 'alive', 'qpos', 'qvel', 'phase', 'features')}
        for tick in range(2000):
            live = raw._source_probe['live'].clone()
            _, rewards, done, _ = env.step(torch.zeros(count, 6, device=args.device))
            # 管理器保存的是每秒奖励率；这里乘一次dt，核对策略实际收到的奖励。
            contributions = raw.reward_manager._step_reward*raw.step_dt
            torch.testing.assert_close(contributions.sum(-1), rewards, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(contributions[:, index], term.last_reward*raw.step_dt, atol=1e-7, rtol=1e-6)
            assert torch.isfinite(contributions).all()
            values = dict(contributions=contributions, score=term.last_reward, valid=term.last_valid,
                alive=live, qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel,
                phase=action.teacher.controller.standup.phase, features=term.history.frames[:, -1])
            for key, value in values.items():
                buffers[key].append(value.cpu().numpy().copy())
            raw._source_probe['live'] &= ~done.bool()
            if tick < 20:
                assert not action.torques.any()
            if tick < 1999 and done.any():
                raise RuntimeError('Teacher terminated before 20s; preserve failure instead of mixing reset episodes')
            if tick % 500 == 0:
                print(f't={(tick+1)*.01:.2f}s live={int(live.sum())}/{count}', flush=True)
        arrays = {k: np.asarray(v) for k, v in buffers.items()}
        times = (np.arange(2000)+1)*.01
        smp = arrays['contributions'][:, :, index]
        task = np.delete(arrays['contributions'], index, axis=-1)
        positive = task.clip(min=0).sum(-1)
        negative = -task.clip(max=0).sum(-1)
        intervals = {}
        for name, left, right in (('startup_hold', 0., .2), ('rise', .2, 2.),
                                   ('standing', 2., 20.), ('late', 15., 20.)):
            mask = np.broadcast_to(((times > left+1e-8) & (times <= right+1e-8))[:, None], smp.shape)
            s, p, n = smp[mask], positive[mask], negative[mask]
            intervals[name] = dict(environment_steps=int(mask.sum()), mean_score=float(arrays['score'][mask].mean()),
                smp_sum=float(s.sum()), task_positive_sum=float(p.sum()), task_penalty_abs_sum=float(n.sum()),
                smp_share_of_positive=float(s.sum()/(s.sum()+p.sum())),
                smp_share_of_absolute=float(s.sum()/(s.sum()+p.sum()+n.sum())),
                mean_step_smp=float(s.mean()), mean_step_task_net=float((p-n).mean()),
                contributions={key: float(arrays['contributions'][:, :, i][mask].sum()) for i, key in enumerate(names)})
        for key, value in before.items():
            torch.testing.assert_close(term.prior.state_dict()[key], value, atol=0, rtol=0)
        np.savez_compressed(args.output/'trace.npz', **arrays, time=times,
                            initial_qpos=qpos.cpu().numpy(), initial_qvel=qvel.cpu().numpy())
        report = dict(passed=True, optimizer_updates=0, num_envs=count, seconds=20,
            physics_dt=raw.physics_dt, action_dt=raw.step_dt, profile=term.contract,
            prior_sha256=digest(args.prior), dataset_fingerprint=dataset.fingerprint,
            reward_weights={name: raw.reward_manager.get_term_cfg(name).weight for name in names},
            intervals=intervals, prior_parameters_and_statistics_unchanged=True,
            normalizer_count=int(term.prior.error_normalizer.count),
            strict_rise=int(raw._source_probe['strict_ever'].sum()),
            final_balance=int((raw._source_probe['hold'] >= 500).sum()),
            delay_pairs_s=pairs, episode_ids=[e['episode_id'] for e in dataset.episodes],
            xml_sha256=dataset.xml_sha256, trace_sha256=digest(args.output/'trace.npz'),
            source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
                ROOT/'mjlab_training/src/ct_mjlab/smp/ppo.py', ROOT/'scripts/agent/check_lqr_student_closed_loop.py']},
            scope='Actual RewardManager sums during nominal live teacher-to-student-PD motion; no PPO training or training randomization. Positive share excludes negative penalties from denominator; signed net is reported separately.')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml', asdict(cfg))
        print(json.dumps(dict(intervals=intervals, strict_rise=report['strict_rise'], final_balance=report['final_balance']), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--initials', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
