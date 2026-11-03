"""Collect qualified, complete LQR/VMC episodes at 100Hz; never train a model."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from ct_mjlab.scripts.teacher import Recorder, ROOT, command_schedule
from ct_mjlab.teacher.env import make_teacher_env_cfg
from ct_mjlab.teacher.mechanics import DT
from ct_mjlab.envs.chuanliantui.model import MODEL_XML, ACTUATED_JOINTS
from ct_mjlab.smp.features import (FEATURE_SCHEMA, FEATURE_DIM, FEATURE_GROUPS, SAMPLE_DT,
                                  MotionStateReader, motion_features)
from ct_mjlab.smp.dataset import behavior_name

DATA_SCHEMA = 'chuanliantui_smp_episodes_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_grid(args):
    if not args.commands:
        return None
    if args.sequence:
        raise ValueError('commands固定命令网格不能与sequence一起使用')
    rows = json.loads(args.commands.read_text())
    if not isinstance(rows,list) or not rows or args.num_envs % len(rows):
        raise ValueError('commands须为非空数组，num-envs须是命令条数的整数倍')
    for row in rows:
        if set(row) != {'vx','yaw','leg_length'}:
            raise ValueError('commands每项须包含vx/yaw/leg_length')
        command_schedule(SimpleNamespace(**row,sequence=None))
    return [rows[i % len(rows)] for i in range(args.num_envs)]


def qualify(result, *, tracking_tilt, tracking_contact, segments):
    reasons = []
    measurements = [result[k] for k in ('mean_abs_vx_error','mean_abs_yaw_error','mean_abs_height_error')]
    if any(value is None or not math.isfinite(value) for value in (*measurements,tracking_tilt)):
        return False, ['nonfinite_or_missing_tracking_measurement']
    if not result['final_balance_v31']:
        reasons.append('final_balance_v31')
    if result['phase'] != 'balance' or result['ever_failed']:
        reasons.append('controller_failure_or_incomplete_standup')
    if segments != 1:
        reasons.append('command_sequence_not_behavior_validated')
    if (result['mean_abs_vx_error'] > .30 or
            result['mean_abs_yaw_error'] > .30 or result['mean_abs_height_error'] > .03 or
            tracking_tilt > math.radians(25) or tracking_contact):
        reasons.append('tracking_or_post_settle_contact')
    return not reasons, reasons


class DemoRecorder(Recorder):
    def __init__(self, env, output, args, schedule):
        super().__init__(env, output, args, schedule)
        self.state_reader = MotionStateReader(env)

    def sample(self, command):
        super().sample(command)
        if self.count % 10 == 0:
            state = self.state_reader.read()
            features = motion_features(**state)
            if not torch.isfinite(features).all():
                raise RuntimeError('Nonfinite motion features')
            self.frames[-1].update({k: state[k].cpu().numpy().copy()
                                   for k in ('wheel_position_w','ground_height')})
            self.frames[-1]['features'] = features.cpu().numpy().copy()


def collect(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    if args.num_envs < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('num-envs和seconds必须为正')
    steps = round(args.seconds / DT)
    if abs(steps * DT-args.seconds) > 1e-8 or steps % 10:
        raise ValueError('采集时长须为0.01秒的整数倍')
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds须为不重复的逗号分隔整数')
    schedule = command_schedule(args)
    grid = command_grid(args)
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = ROOT / 'mjlab_training/src/ct_mjlab'
    source_hashes = {}
    source_files = sorted(source_root.rglob('*.py')) + [ROOT/'third_party/mimickit_smp'/name
                                                     for name in ('LICENSE','NOTICE.md')]
    for path in source_files:
        relative = path.relative_to(ROOT)
        target = args.output / 'snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = digest(path)
    for path in (MODEL_XML, MODEL_XML.with_name('huananhu_pose_bank.npz')):
        shutil.copy2(path, args.output / 'snapshot' / path.name)
    (args.output / 'snapshot/worktree.patch').write_text(subprocess.check_output(
        ['git','diff','HEAD','--','mjlab_training','scripts/agent','resources'], cwd=ROOT, text=True))
    source_id = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    episodes, rejected = [], []
    metadata = dict(schema=DATA_SCHEMA, feature_schema=FEATURE_SCHEMA, feature_dim=FEATURE_DIM,
                    feature_groups=FEATURE_GROUPS, sample_dt=SAMPLE_DT, physics_dt=DT,
                    motor_order=ACTUATED_JOINTS, xml_sha256=digest(MODEL_XML),
                    source_sha256=source_hashes, source_id=source_id,
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    created_at=datetime.now().astimezone().isoformat(), schedule=schedule,
                    initial=args.initial, seeds_requested=seeds, inverted_recovery=False,
                    ground_start_hold_s=args.ground_start_hold if args.initial == 'ground' else 0.,
                    commands_per_env=grid,velocity_integral=args.velocity_integral,
                    timing='States/features are post-step; torques drove preceding 1ms tick. No PPO action labels.')
    configure_torch_backends()
    cfg = make_teacher_env_cfg(initial=args.initial, num_envs=args.num_envs, seed=seeds[0],
                               velocity_integral=args.velocity_integral,
                               ground_start_hold_s=args.ground_start_hold)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        for seed in seeds:
            run_dir = args.output / f'rollout-{seed}'
            run_dir.mkdir()
            args.seed = seed
            # 上一回合缓存由推理模式生成，复位也在同一模式内更新这些张量。
            with torch.inference_mode():
                env.reset(seed=seed)
            recorded_schedule = [{'time':0.,'commands_per_env':grid}] if grid else schedule
            recorder = DemoRecorder(env, run_dir, args, recorded_schedule)
            initial_qvel = env.sim.data.qvel.cpu().numpy().copy()
            command = torch.tensor([schedule[0][k] for k in ('vx','yaw','leg_length')],
                                   device=args.device).repeat(args.num_envs, 1)
            if grid:
                command[:] = command.new_tensor([[row[k] for k in ('vx','yaw','leg_length')] for row in grid])
            completed = False
            try:
                index = 0
                with torch.inference_mode():
                    for tick in range(steps):
                        while index+1 < len(schedule) and schedule[index+1]['time'] <= tick*DT:
                            index += 1
                            command[:] = command.new_tensor([schedule[index][k] for k in ('vx','yaw','leg_length')])
                        env.step(command)
                        recorder.count += 1
                        recorder.sample(command)
                completed = True
            except Exception as exc:
                recorder.run_error = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                recorder.save(completed)
            report = json.loads((run_dir/'result.json').read_text())
            with np.load(run_dir/'trace.npz') as trace:
                for i, result in enumerate(report['results']):
                    episode_schedule = [dict(time=0.,**grid[i])] if grid else schedule
                    behavior = behavior_name(args.initial,episode_schedule)
                    group = hashlib.sha256(json.dumps([source_id,metadata['xml_sha256'],seed,args.initial,
                                                       episode_schedule,args.seconds,args.velocity_integral,
                                                       metadata['ground_start_hold_s']],sort_keys=True).encode()).hexdigest()
                    accepted, reasons = qualify(result, tracking_tilt=float(recorder.tracking_tilt[i]),
                                                tracking_contact=bool(recorder.tracking_contact[i]), segments=len(schedule))
                    if behavior == 'standup':
                        from ct_mjlab.teacher.standup import DONE, FAILED
                        phases = trace['phase'][:,i]
                        done = np.flatnonzero(phases == DONE)
                        result['standup_completed_s'] = float(trace['time'][done[0]]) if len(done) else None
                        if not len(done) or not np.any((phases[:done[0]] != DONE) & (phases[:done[0]] != FAILED)):
                            reasons.append('no_recorded_standup_transition')
                            accepted = False
                    episode_id = hashlib.sha256(f'{group}:{i}'.encode()).hexdigest()
                    record = dict(episode_id=episode_id, group_id=group, seed=seed, env=i, behavior=behavior,
                                  schedule=episode_schedule,
                                  accepted=accepted, rejection_reasons=reasons, quality=result)
                    relative = Path('accepted' if accepted else 'rejected') / f'{episode_id}.npz'
                    record['path'] = str(relative)
                    target = args.output / relative
                    target.parent.mkdir(exist_ok=True)
                    data = {key: trace[key][:, i] for key in trace.files if key not in ('time','initial_qpos')}
                    np.savez_compressed(target, **data, time=trace['time'],
                                        initial_qpos=recorder.initial_qpos[i], initial_qvel=initial_qvel[i],
                                        metadata=np.asarray(json.dumps({**metadata,**record},ensure_ascii=False)))
                    record['sha256'] = digest(target)
                    (episodes if accepted else rejected).append(record)
            print(f'采集seed={seed}: 合格{sum(r["seed"]==seed for r in episodes)}/{args.num_envs}', flush=True)
            # 每个种子完成即保存；异常中断不会把此前完整回合丢掉。
            (args.output/'dataset.json').write_text(json.dumps({**metadata,'episodes':episodes},indent=2,ensure_ascii=False)+'\n')
            (args.output/'report.json').write_text(json.dumps(dict(accepted=len(episodes),rejected=len(rejected),
                rejected_episodes=rejected, complete=seed == seeds[-1],
                seeds_completed=[*dict.fromkeys(r['seed'] for r in episodes+rejected)]),indent=2,ensure_ascii=False)+'\n')
    except BaseException as exc:
        (args.output/'report.json').write_text(json.dumps(dict(accepted=len(episodes),rejected=len(rejected),
            rejected_episodes=rejected,complete=False,error=f'{type(exc).__name__}: {exc}',
            seeds_completed=[*dict.fromkeys(r['seed'] for r in episodes+rejected)]),indent=2,ensure_ascii=False)+'\n')
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', default='42,43')
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--initial', choices=('standing','ground'), default='standing')
    parser.add_argument('--ground-start-hold', type=float, default=.2,
                        help='Zero active motor torque before ground-start takeover (seconds; default: 0.2)')
    parser.add_argument('--vx', type=float, default=0.)
    parser.add_argument('--yaw', type=float, default=0.)
    parser.add_argument('--leg-length', type=float, default=.30)
    parser.add_argument('--sequence', type=Path)
    parser.add_argument('--commands', type=Path, help='Fixed per-environment command grid JSON; repeats to num-envs')
    parser.add_argument('--velocity-integral', action='store_true', help='Simulation velocity-error integral in LQR position slot')
    parser.add_argument('--output', type=Path, required=True)
    collect(parser.parse_args())


if __name__ == '__main__':
    main()
