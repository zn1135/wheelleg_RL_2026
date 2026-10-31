#!/usr/bin/env python3
"""Record frozen walk sensor packets for independent velocity-compensation trials."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from scripts.agent.split import diagnose_walk_v8_regression as shared
from scripts.agent.split.diagnose_walk_goal_feedback import actor_digest
from scripts.agent.split.diagnose_walk_v8_regression import Recorder, digest, sample_physics, write


NUM_PER_CASE, SECONDS, HEIGHT = 16, 30., .26
SCHEDULE_SEEDS = {'discovery': 44, 'validation': 45, 'goal': 43}
FIXED_KEYS = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
WHEEL_OBSERVATION_IDS = (15, 18)


def make_cases(schedule):
    accelerations = (.8, 1.2) if schedule == 'validation' else (.5, 2.)
    cases = []
    for acceleration in accelerations:
        for speed in (0., .5, -.5):
            first = len(cases) * NUM_PER_CASE
            if schedule == 'validation':
                phases = ((0., 3., 0.), (3., 12., speed), (12., 18., 0.),
                          (18., 25., -speed), (25., SECONDS, 0.))
            else:
                phases = ((0., 2., 0.), (2., 16., speed), (16., SECONDS, 0.))
            cases.append(dict(name=f'v{speed:g}-a{acceleration:g}', speed=speed,
                acceleration=acceleration, env_ids=list(range(first, first+NUM_PER_CASE)),
                segments=[dict(start=start, end=end, speed=target, acceleration=acceleration)
                          for start, end, target in phases]))
    return cases


@torch.inference_mode()
def run(args, cases, registry):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from rsl_rl.models import MLPModel

    cfg, train = registry.get_cfgs('chuanliantui_walk', play=True)
    cfg.auto_reset, cfg.seed = False, args.seed
    cfg.metrics['walk_diagnosis'] = MetricsTermCfg(
        func=sample_physics, per_substep=True, reduce='last')
    count = len(cases) * NUM_PER_CASE
    wrapped, _ = registry.make_env('chuanliantui_walk', env_cfg=cfg,
        train_cfg=train, num_envs=count, device=args.device)
    raw = wrapped.unwrapped
    try:
        assert raw.physics_dt == .002 and raw.step_dt == .01
        built, _ = registry.make_alg_runner(wrapped, 'chuanliantui_walk',
            train_cfg=train, device=args.device)
        built.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True,
            map_location=args.device)
        actor = built.alg.actor
        actor.eval()
        assert not actor.obs_normalization
        before = {key:value.detach().cpu().clone() for key,value in actor.state_dict().items()}
        before_sha = actor_digest(before)
        state = raw.command_manager.get_term('standup')
        state.speed_curriculum_frozen = True
        raw.reset(seed=args.seed)
        pd = raw.action_manager.get_term('pd')
        fixed = dict(qpos=raw.sim.data.qpos.cpu().numpy().copy(),
            qvel=raw.sim.data.qvel.cpu().numpy().copy(),
            nominal=pd._nominal_pos.cpu().numpy().copy(), kp=pd.kp_scale.cpu().numpy().copy(),
            kd=pd.kd_scale.cpu().numpy().copy(), strength=pd.motor_strength.cpu().numpy().copy(),
            mass=raw.sim.mj_model.body_mass.copy(), friction=raw.sim.mj_model.geom_friction.copy())
        reference_evidence = None
        if args.reference_npz is not None:
            with np.load(args.reference_npz, allow_pickle=False) as reference:
                equality = {key:np.array_equal(fixed[key], reference['fixed_'+key])
                            for key in FIXED_KEYS}
                assert all(equality.values()), 'Initial/physical arrays differ from reference'
                assert np.array_equal(fixed['qpos'], reference['initial_qpos'])
                assert np.array_equal(fixed['qvel'], reference['initial_qvel'])
            reference_evidence = dict(path=str(args.reference_npz),
                sha256=digest(args.reference_npz), fixed_arrays_bitwise_equal=equality,
                initial_qpos_qvel_bitwise_equal=True)
            metadata = args.reference_npz.with_suffix('.json')
            if metadata.exists():
                reference_meta = json.loads(metadata.read_text())
                assert reference_meta['checkpoint_sha256'] == args.checkpoint_sha256
                if 'actor_state_sha256' in reference_meta:
                    assert reference_meta['actor_state_sha256'] == before_sha
                reference_evidence['metadata_sha256'] = digest(metadata)

        sensor_velocity, candidate_source = None, None
        if args.feedback == 'compensated':
            # encoder采集不导入候选；闭环候选只接收当前传感25槽和固定标称角。
            from scripts.agent.split import walk_sensor_velocity
            sensor_velocity = walk_sensor_velocity.SensorWheelVelocity(raw.sim.mj_model)
            candidate_path = Path(walk_sensor_velocity.__file__).resolve()
            candidate_source = dict(path=str(candidate_path), sha256=digest(candidate_path),
                formula_spec=sensor_velocity.formula_spec)
            shutil.copy2(candidate_path, args.output/'candidate-source.py')

        accelerations = torch.empty(count, device=args.device)
        for case in cases:
            accelerations[case['env_ids']] = case['acceleration']

        def commands(time):
            state.tracking_episode.fill_(True)
            state.zero_hold.zero_()
            state.tracking_velocity.zero_()
            for case in cases:
                for segment in case['segments']:
                    if segment['start'] <= time < segment['end']:
                        state.tracking_velocity[case['env_ids']] = segment['speed']
                        break
            state.tracking_yaw_rate.zero_()
            state.tracking_interval_steps.fill_(1000000)
            state.tracking_acceleration.copy_(accelerations)
            state.episode_height.fill_(HEIGHT)
            state._apply_height(None)

        commands(0.)
        raw.observation_manager.reset()
        raw.obs_buf = raw.observation_manager.compute(update_history=True)
        obs = raw.get_observations()
        recorder = Recorder(raw, round(SECONDS/raw.physics_dt))
        raw._walk_diagnosis = recorder
        policy_steps = round(SECONDS/raw.step_dt)
        poses = np.empty((policy_steps+1, count, fixed['qpos'].shape[-1]), dtype=fixed['qpos'].dtype)
        poses[0] = fixed['qpos']
        pose_live = np.ones((policy_steps+1, count), dtype=bool)
        observation = np.empty((policy_steps, count, 25), dtype=np.float32)
        encoder_m_s = np.empty((policy_steps, count, 3), dtype=np.float32)
        target_com_m_s = np.empty_like(encoder_m_s)
        used_vx_m_s = np.empty((policy_steps, count), dtype=np.float32)
        policy_valid = np.empty((policy_steps, count), dtype=bool)
        ends = np.full(count, SECONDS)
        scene = args.output/(args.feedback+'-scene.mjb')
        mujoco.mj_saveModel(raw.sim.mj_model, str(scene), None)
        with patch.object(pd, 'apply_actions', recorder.apply):
            for tick in range(policy_steps):
                commands(tick*raw.step_dt)
                base = MLPModel.get_latent(actor, obs)
                assert base.shape == (count, 25)
                packet = base.detach().cpu().numpy().copy()
                # 精确保存Actor实际收到的当前帧，包含全部三轴陀螺；不由真姿态重建。
                observation[tick] = packet
                policy_valid[tick] = recorder.live.cpu().numpy()
                velocity = actor.encoder(obs['history'])
                recorder.encoder = velocity/2.
                recorder.true_before = raw.scene['robot'].data.root_com_lin_vel_b.clone()
                encoder_m_s[tick] = recorder.encoder.cpu().numpy()
                target_com_m_s[tick] = recorder.true_before.cpu().numpy()
                policy_velocity = velocity.clone()
                if sensor_velocity is not None:
                    compensated_vx = np.asarray(sensor_velocity.predict(packet, fixed['nominal']))
                    assert compensated_vx.shape == (count,) and np.isfinite(compensated_vx).all()
                    policy_velocity[:, 0] = torch.as_tensor(compensated_vx,
                        dtype=policy_velocity.dtype, device=args.device)*2.
                used_vx_m_s[tick] = (policy_velocity[:, 0]/2.).cpu().numpy()
                recorder.policy_velocity = policy_velocity/2.
                recorder.policy_wheel_velocity = base[:, WHEEL_OBSERVATION_IDS]/.05
                # 两种反馈共用同一Actor路径，vy/vz始终保留原encoder。
                action = actor.mlp(torch.cat((base, policy_velocity), -1))
                if train.clip_actions is not None:
                    action = action.clamp(-train.clip_actions, train.clip_actions)
                recorder.action = action
                obs, _, terminated, timeout, _ = raw.step(action)
                done = terminated | timeout
                first = done & recorder.live
                ends[first.cpu().numpy()] = (tick+1)*raw.step_dt
                recorder.live &= ~done
                poses[tick+1] = raw.sim.data.qpos.cpu().numpy()
                pose_live[tick+1] = recorder.live.cpu().numpy()
                if done.any() and tick+1 < policy_steps:
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs = raw.get_observations()
                if (tick+1) % 500 == 0:
                    print(args.schedule, args.feedback, (tick+1)*raw.step_dt,
                        'seconds, live', int(recorder.live.sum()), flush=True)
        assert recorder.count == round(SECONDS/raw.physics_dt)
        assert all(torch.equal(before[key], value.cpu()) for key,value in actor.state_dict().items())
        after_sha = actor_digest(actor.state_dict())
        assert after_sha == before_sha
        values = recorder.trace.cpu().numpy()
        assert all(np.isfinite(array).all() for array in
                   (values, poses, observation, encoder_m_s, target_com_m_s, used_vx_m_s))
        assert np.array_equal(policy_valid, values[::5, :, recorder.layout['valid'][0]] > .5)
        assert np.array_equal(encoder_m_s, values[::5, :, slice(*recorder.layout['encoder_velocity'])])
        assert np.array_equal(target_com_m_s, values[::5, :, slice(*recorder.layout['true_velocity_before'])])
        assert np.array_equal(used_vx_m_s, values[::5, :, recorder.layout['policy_velocity'][0]])
        if candidate_source is not None:
            assert digest(Path(candidate_source['path'])) == candidate_source['sha256']
        metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_iteration=2999, label=args.feedback, feedback_mode=args.feedback,
            velocity_source=args.feedback, actual_profile='chuanliantui-walk-physical25-v8',
            common_environment_profile='chuanliantui-walk-physical25-v8',
            parameter_updates=0, actor_unchanged=True, encoder_unchanged=True,
            actor_state_sha256=before_sha, actor_state_sha256_after=after_sha,
            schedule=args.schedule, seed=args.seed, seconds=SECONDS, height=HEIGHT,
            warmup=3. if args.schedule == 'validation' else 2.,
            stop_at=25. if args.schedule == 'validation' else 16.,
            dt=raw.physics_dt, policy_dt=raw.step_dt,
            weight_N=float(raw.sim.mj_model.body_mass.sum()*9.81), cases=cases,
            layout=recorder.layout, joint_names=recorder.joint_names, episode_end=ends.tolist(),
            survived=int(recorder.live.sum()), episode_done=(~recorder.live).cpu().tolist(),
            fixed_arrays=list(fixed), reference=reference_evidence,
            scene_mjb=str(scene.resolve()), scene_sha256=digest(scene),
            qpos_key='qpos', qpos_time_key='qpos_time_s', qpos_live_key='qpos_live',
            policy_observation_key='policy_observation', policy_time_key='policy_time_s',
            policy_valid_key='policy_valid', encoder_key='encoder_m_s',
            target_com_key='target_com_m_s', used_vx_key='used_vx_m_s',
            policy_observation_layout=dict(angular_velocity_b_scaled=[0, 3],
                gravity_b=[3, 6], command_scaled=[6, 9], motor_q_relative_nominal=[9, 13],
                motor_dq_scaled=[13, 19], previous_action=[19, 25]),
            policy_observation_scales=dict(angular_velocity=.25, motor_dq=.05),
            policy_observation_source='Exact unnormalized MLPModel.get_latent(actor,obs), current actor group; never reconstructed from privileged qpos/passive joints',
            target_com_scope='Pre-action root body inertial COM velocity in root link frame, privileged scoring only; not whole-robot mass-weighted COM',
            candidate_source=candidate_source,
            compensation_scope='Only vx replaced from current base25 sensor packet and fixed PD nominal, multiplied by2; encoder vy/vz unchanged; no extra filter',
            timing='policy_time_s[k]=k*.01 before action; encoder/COM/base25/used-vx latched together. qpos[k] is that policy boundary. values[5*k:5*k+5] share latched inputs, physical fields follow each 2ms integration; derived fields may lag <=2ms. policy_valid is first-episode live at action start.',
            scope='Frozen nominal collection or gated feedback trial; no training and no task-acceptance verdict')
        np.savez_compressed(args.output/(args.feedback+'.npz'), values=values,
            initial_qpos=fixed['qpos'], initial_qvel=fixed['qvel'],
            qpos=poses, qpos_time_s=np.arange(policy_steps+1)*raw.step_dt, qpos_live=pose_live,
            policy_observation=observation, policy_time_s=np.arange(policy_steps)*raw.step_dt,
            policy_valid=policy_valid, encoder_m_s=encoder_m_s,
            target_com_m_s=target_com_m_s, used_vx_m_s=used_vx_m_s,
            **{'fixed_'+key:value for key,value in fixed.items()})
        write(args.output/(args.feedback+'.json'), metadata)
        return metadata
    finally:
        raw.close()


def main(args):
    assert args.seed == SCHEDULE_SEEDS[args.schedule], 'Use the preregistered seed for this schedule'
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    args.training_artifact = args.training_artifact.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.reference_npz is None and args.schedule == 'goal':
        args.reference_npz = root/'artifacts/walk25-goal-wheel-vx-probe-20261030-r1/encoder.npz'
    if args.reference_npz is not None:
        args.reference_npz = args.reference_npz.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    launch = json.loads((args.training_artifact/'launch.json').read_text())
    source_hashes = json.loads((args.training_artifact/'source-sha256.json').read_text())
    assets = json.loads((args.training_artifact/'physical-files-sha256.json').read_text())
    assert len(source_hashes) == 90 and len(assets) == 40
    runtime = args.output/'frozen-runtime'
    source = runtime/'mjlab_training/src/ct_mjlab'
    shutil.copytree(Path(launch['log_dir'])/'snapshot/ct_mjlab', source,
        ignore=shutil.ignore_patterns('__pycache__'))
    assert all(digest(source/name) == sha for name,sha in source_hashes.items())
    for name,sha in assets.items():
        assert digest(root/name) == sha, name
        destination = runtime/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, destination)
        assert digest(destination) == sha, name
    args.checkpoint_sha256 = digest(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert saved['iter'] == 2999
    assert saved['infos']['experiment_profile_contract'] == 'chuanliantui-walk-physical25-v8'
    del saved
    recorder_file = Path(shared.__file__).resolve()
    recorder_sha = digest(recorder_file)
    shutil.copy2(recorder_file, args.output/'recorder-source.py')
    shutil.copy2(__file__, args.output/'source.py')
    sys.path.insert(0, str(source.parent))
    from ct_mjlab.envs import task_registry
    cases = make_cases(args.schedule)
    metadata = run(args, cases, task_registry)
    assert digest(args.checkpoint) == args.checkpoint_sha256
    assert digest(recorder_file) == recorder_sha
    assert all(digest(source/name) == sha for name,sha in source_hashes.items())
    assert all(digest(runtime/name) == sha for name,sha in assets.items())
    write(args.output/'result.json', dict(verified=True, parameter_updates=0,
        source_files=len(source_hashes), resource_files=len(assets), resources_copied=True,
        recorder_source_sha256=recorder_sha, schedule=args.schedule, seed=args.seed,
        feedback=args.feedback, reference=metadata['reference'], runs=[metadata],
        scope='Single frozen collection integrity only; no behavioral acceptance claim'))
    print(json.dumps(dict(verified=True, parameter_updates=0, schedule=args.schedule,
        feedback=args.feedback, survived=metadata['survived']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--schedule', choices=tuple(SCHEDULE_SEEDS), required=True)
    parser.add_argument('--feedback', choices=('encoder', 'compensated'), required=True)
    parser.add_argument('--reference-npz', type=Path,
        help='Compare all eight initial/physical arrays; goal defaults to the old goal encoder trace')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Collection cannot train')):
        main(parser.parse_args())
