#!/usr/bin/env python3
"""Collect screened LQR demonstrations through the unchanged student PD interface."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.bc import LABEL_REFERENCE, WHEEL_PAIR_SCHEMA
from ct_mjlab.envs.chuanliantui.huananhu_poses import POSE_JOINT_NAMES
from ct_mjlab.envs.chuanliantui import multimode_observations
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS, MODEL_XML
from ct_mjlab.smp.dataset import MotionDataset
from ct_mjlab.smp.features import FEATURE_SCHEMA, FEATURE_DIM, FEATURE_GROUPS, SAMPLE_DT, MotionStateReader
from ct_mjlab.smp.ppo import SmpCommandCfg, inverted, reset_pose
from ct_mjlab.teacher.standup import DONE, FAILED
from check_lqr_student_closed_loop import BridgeCfg, teacher_failed, digest
from check_smp_policy_source import balance_substep

ROOT = Path(__file__).resolve().parents[2]
# 在查看新轨迹之前固定筛选门槛；停车／平滑与v3.1平衡分开检查。
QUALITY = dict(last5s_leg_velocity_rms_rad_s=.02, last5s_mean_speed_m_s=.01,
               last5s_xy_excursion_m=.02, mean_height_error_m=.03,
               post_settle_tilt_rad=float(np.deg2rad(25)))


def collect_quality(env):
    probe = getattr(env, '_collection_probe', None)
    if probe is None:
        return torch.zeros(env.num_envs, device=env.device)
    gravity = env.scene['robot'].data.projected_gravity_b
    tilt = torch.maximum(torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs(),
                         torch.asin(gravity[:, 0].clamp(-1, 1)).abs())
    settled = env.sim.data.time[:].reshape(-1) >= 2.
    contact = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    probe['contact'] |= settled & (contact > 1.)
    probe['tilt'] = torch.maximum(probe['tilt'], torch.where(settled, tilt, 0.))
    return tilt


@torch.inference_mode()
def collect(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
    training_cfg, _ = task_registry.get_cfgs('chuanliantui', play=False)
    delays = training_cfg.actions['pd'].action_delay_choices_s
    assert delays == (.002, .004)
    pairs = [(leg, wheel) for leg in delays for wheel in delays]
    seeds = [int(seed) for seed in args.seeds.split(',')]
    assert len(seeds) == len(set(seeds)) >= 2 and args.poses_per_seed > 0
    with np.load(MODEL_XML.with_name('huananhu_pose_bank.npz')) as bank:
        poses = np.concatenate((bank['negative_left_pos'], bank['negative_right_pos']), -1)
    total = args.poses_per_seed*len(seeds)
    assert total <= len(poses)
    rows = np.random.default_rng(args.pose_seed).permutation(len(poses))[:total]
    plan = dict(pose_seed=args.pose_seed, seeds=seeds, poses_per_seed=args.poses_per_seed,
        pose_rows=rows.tolist(), delay_pairs_s=pairs, quality_thresholds=QUALITY,
        pose_bank_sha256=digest(MODEL_XML.with_name('huananhu_pose_bank.npz')))
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    source_hashes = {}
    sources = list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py')) + [Path(__file__),
        ROOT/'scripts/agent/check_lqr_student_closed_loop.py', ROOT/'scripts/agent/check_smp_policy_source.py']
    for path in sources:
        name = str(path.relative_to(ROOT)); source_hashes[name] = digest(path)
        target = args.output/'snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_id = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    metadata = dict(schema='chuanliantui_smp_episodes_v1', feature_schema=FEATURE_SCHEMA,
        feature_dim=FEATURE_DIM, feature_groups=FEATURE_GROUPS, sample_dt=SAMPLE_DT,
        physics_dt=.002, action_dt=.01, motor_order=ACTUATED_JOINTS, xml_sha256=digest(MODEL_XML),
        pose_bank_sha256=plan['pose_bank_sha256'], source_sha256=source_hashes, source_id=source_id,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        created_at=datetime.now().astimezone().isoformat(), initial='ground', ground_start_hold_s=.2,
        schedule=[dict(time=0., vx=0., yaw=0., leg_length=.30)], velocity_integral=True,
        inverted_recovery=False, source_controller='live_lqr_vmc_through_student_pd',
        action_pair_schema=WHEEL_PAIR_SCHEMA,
        label_reference=LABEL_REFERENCE,
        observation_contract=multimode_observations.WHEEL_OBSERVATION_CONTRACT,
        wheel_odometry=multimode_observations.WHEEL_ODOMETRY,
        observation_noise=False, observation_delay=False,
        delay_pairs_s=pairs, quality_thresholds=QUALITY, seeds_requested=seeds,
        group_definition='Same pose-bank row across delays/seeds shares a group; no duplicate-pose split leakage',
        timing='100Hz post-step states/features; 100Hz target generation, original 500Hz student PD/FIFO; no PPO policy')
    cfg.seed, cfg.auto_reset = seeds[0], False
    cfg.commands['standup'] = SmpCommandCfg()
    cfg.events['reset_pose'] = EventTermCfg(func=reset_pose, mode='reset')
    cfg.terminations['inverted'] = TerminationTermCfg(func=inverted)
    cfg.terminations['teacher_failed'] = TerminationTermCfg(func=teacher_failed)
    cfg.rewards = {}
    cfg.metrics['source_balance'] = MetricsTermCfg(func=balance_substep, per_substep=True, reduce='last')
    cfg.metrics['collection_quality'] = MetricsTermCfg(func=collect_quality, per_substep=True, reduce='last')
    multimode_observations.configure_wheel_odometry(cfg)
    cfg.actions['pd'] = BridgeCfg(**{key: value for key, value in vars(cfg.actions['pd']).items() if key != 'class_type'})
    cfg.actions['pd'].action_delay_s = None
    cfg.actions['pd'].randomize_delays = False
    cfg.actions['pd'].action_delay_choices_s = delays
    count = args.poses_per_seed*len(pairs)
    env, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                  num_envs=count, device=args.device)
    accepted, rejected, completed_seeds = [], [], []
    def save_report(complete, error=None):
        (args.output/'dataset.json').write_text(json.dumps({**metadata, 'episodes':accepted}, indent=2)+'\n')
        (args.output/'report.json').write_text(json.dumps(dict(complete=complete, error=error,
            accepted=len(accepted), rejected=len(rejected), seeds_completed=completed_seeds,
            unique_accepted_pose_rows=sorted({row['pose_row'] for row in accepted}),
            rejected_episodes=rejected, optimizer_updates=0), indent=2)+'\n')
    try:
        raw = env.unwrapped
        assert raw.physics_dt == .002 and raw.step_dt == .01
        robot, action = raw.scene['robot'], raw.action_manager.get_term('pd')
        order = [POSE_JOINT_NAMES.index(name) for name in robot.joint_names]
        reader = MotionStateReader(raw)
        for batch, seed in enumerate(seeds):
            selected = rows[batch*args.poses_per_seed:(batch+1)*args.poses_per_seed]
            raw.reset(seed=seed)
            joints = torch.as_tensor(poses[selected][:, order], dtype=torch.float32, device=raw.device).repeat(len(pairs), 1)
            root = robot.data.default_root_state.clone()
            root[:, :3] = raw.scene.env_origins; root[:, 2] += .25
            root[:, 3:7] = root.new_tensor([1., 0., 0., 0.]); root[:, 7:] = 0.
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            robot.write_root_state_to_sim(root)
            raw.sim.forward()
            action.reset()
            action.action_delay_steps[:] = torch.tensor([round(pair[0]/.002) for pair in pairs], device=raw.device).repeat_interleave(args.poses_per_seed)
            action.wheel_action_delay_steps[:] = torch.tensor([round(pair[1]/.002) for pair in pairs], device=raw.device).repeat_interleave(args.poses_per_seed)
            raw.command_manager.get_term('standup').ground_start[:] = True
            initial_qpos, initial_qvel = raw.sim.data.qpos.cpu().numpy().copy(), raw.sim.data.qvel.cpu().numpy().copy()
            # 手工写入指定初态后重建观测缓存；不能沿用reset随机抽到的另一个姿态。
            raw.command_manager.compute(dt=0.)
            multimode_observations.reset_observations(raw, torch.arange(count, device=raw.device))
            obs = raw.observation_manager.compute(update_history=True)
            raw._source_weight = float(raw.sim.get_default_field('body_mass')[robot.indexing.body_ids].sum())*9.81
            raw._source_probe = dict(live=torch.ones(count, dtype=torch.bool, device=raw.device),
                hold=torch.zeros(count, dtype=torch.long, device=raw.device),
                strict_run=torch.zeros(count, dtype=torch.long, device=raw.device),
                strict_ever=torch.zeros(count, dtype=torch.bool, device=raw.device))
            raw._collection_probe = dict(contact=torch.zeros(count, dtype=torch.bool, device=raw.device),
                                         tilt=torch.zeros(count, device=raw.device))
            buffers = {key:[] for key in ('root','motor_q','motor_dq','wheel_position_w','ground_height',
                'features','qpos','qvel','torque','actuator_torque','control_active',
                'action','observation','source_motor_q','source_motor_dq',
                'command','phase','fault','alive','done')}
            for tick in range(2000):
                live = raw._source_probe['live'].clone()
                observation = obs['actor'].clone()  # 本拍动作生成前，包含轮式里程的40维输入。
                obs, _, done, _ = env.step(torch.zeros(count, 6, device=raw.device))
                state = reader.read()
                values = dict(**state, features=reader.features(), qpos=raw.sim.data.qpos,
                    qvel=raw.sim.data.qvel, torque=action.torques, action=action.raw_action,
                    actuator_torque=raw.sim.data.actuator_force[:,action.teacher.motor_actuators],
                    control_active=action.teacher.control_active,
                    observation=observation, source_motor_q=action._source_q,
                    source_motor_dq=action._source_dq,
                    command=action.teacher.command, phase=action.teacher.controller.standup.phase,
                    fault=action.teacher.controller.standup.fault, alive=live, done=done.bool())
                for key, value in values.items(): buffers[key].append(value.cpu().numpy().copy())
                if tick < 20:
                    assert not action.torques.any()
                    assert not values['actuator_torque'].any()
                raw._source_probe['live'] &= ~done.bool()
                if tick % 500 == 0:
                    print(f'seed={seed} t={(tick+1)*.01:.2f}s live={int(raw._source_probe["live"].sum())}/{count}', flush=True)
            arrays = {key:np.asarray(value) for key, value in buffers.items()}
            times = (np.arange(2000)+1)*SAMPLE_DT
            late, settled = times > 15., times >= 2.
            for index in range(count):
                pose_row = int(selected[index % args.poses_per_seed]); pair = pairs[index//args.poses_per_seed]
                root = arrays['root'][:, index]; dq = arrays['motor_dq'][:, index][:, [0,1,3,4]]
                quality = dict(final_balance_v31=bool(raw._source_probe['hold'][index] >= 500),
                    strict_rise=bool(raw._source_probe['strict_ever'][index]), full_20s=bool(arrays['alive'][:,index].all()),
                    mean_height_error_m=float(np.abs(arrays['features'][settled,index,0]-.33648).mean()),
                    last5s_leg_velocity_rms_rad_s=float(np.sqrt(np.square(dq[late]).mean())),
                    last5s_mean_speed_m_s=float(np.linalg.norm(root[late,7:9],axis=-1).mean()),
                    last5s_xy_excursion_m=float(np.linalg.norm(root[late,:2]-root[late,:2][0],axis=-1).max()),
                    post_settle_tilt_rad=float(raw._collection_probe['tilt'][index]),
                    post_settle_base_contact=bool(raw._collection_probe['contact'][index]),
                    ever_failed=bool((arrays['phase'][:,index] == FAILED).any()),
                    final_phase=int(arrays['phase'][-1,index]))
                reasons = [key for key in ('strict_rise','final_balance_v31','full_20s') if not quality[key]]
                reasons += [key for key, limit in QUALITY.items() if not np.isfinite(quality[key]) or quality[key] > limit]
                if quality['post_settle_base_contact']: reasons.append('post_settle_base_contact')
                if quality['ever_failed'] or quality['final_phase'] != DONE: reasons.append('controller_failed_or_not_balancing')
                if not all(np.isfinite(arrays[key][:,index]).all() for key in ('root','motor_q','motor_dq','features','qpos','qvel','torque','action')):
                    reasons.append('nonfinite')
                if not np.any((arrays['phase'][:200,index] != DONE) & (arrays['phase'][:200,index] != FAILED)):
                    reasons.append('no_recorded_rise')
                group = hashlib.sha256(json.dumps([metadata['xml_sha256'],plan['pose_bank_sha256'],pose_row,metadata['schedule']],sort_keys=True).encode()).hexdigest()
                identity = hashlib.sha256(json.dumps([group,seed,pair,source_id]).encode()).hexdigest()
                record = dict(episode_id=identity, group_id=group, seed=seed, env=index, pose_row=pose_row,
                    behavior='standup', schedule=metadata['schedule'], leg_delay_s=pair[0], wheel_delay_s=pair[1],
                    accepted=not reasons, rejection_reasons=reasons, quality=quality)
                relative = Path('rejected' if reasons else 'accepted')/f'{identity}.npz'
                target = args.output/relative; target.parent.mkdir(exist_ok=True)
                record['path'] = str(relative)
                np.savez_compressed(target, **{key:value[:,index] for key,value in arrays.items()}, time=times,
                    observation_time=times-SAMPLE_DT, action_time=times-SAMPLE_DT,
                    nominal_motor_positions=action.nominal_positions[index].cpu().numpy(),
                    initial_qpos=initial_qpos[index], initial_qvel=initial_qvel[index],
                    metadata=np.asarray(json.dumps({**metadata,**record})))
                record['sha256'] = digest(target)
                (rejected if reasons else accepted).append(record)
            completed_seeds.append(seed); save_report(False)
            print(f'seed={seed} accepted={sum(row["seed"]==seed for row in accepted)}/{count}',flush=True)
        dataset = MotionDataset([args.output/'dataset.json'])
        (args.output/'dataset-check.json').write_text(json.dumps(dataset.description(),indent=2)+'\n')
        save_report(True)
        from mjlab.utils.os import dump_yaml
        dump_yaml(args.output/'env.yaml',asdict(cfg))
        print(f'Completed: {len(accepted)} accepted, {len(rejected)} rejected, {len({row["pose_row"] for row in accepted})} unique accepted poses',flush=True)
    except BaseException as exc:
        save_report(False,f'{type(exc).__name__}: {exc}')
        raise
    finally:
        env.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',default='4101,4102')
    parser.add_argument('--pose-seed',type=int,default=4100)
    parser.add_argument('--poses-per-seed',type=int,default=32)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    collect(parser.parse_args())
