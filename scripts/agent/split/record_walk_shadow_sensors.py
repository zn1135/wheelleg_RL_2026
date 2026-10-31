#!/usr/bin/env python3
"""Record native 500Hz sensor primitives; shadow estimates never control the robot."""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from scripts.agent.split import diagnose_walk_v8_regression as shared
from scripts.agent.split import diagnose_walk_velocity_compensation as schedule_source
from scripts.agent.split.diagnose_walk_goal_feedback import actor_digest
from scripts.agent.split.diagnose_walk_v8_regression import Recorder, digest, sample_physics, write


SCHEDULE_SEEDS = {'study': 46, 'validation': 47}
NUM_PER_CASE, SECONDS, HEIGHT = 16, 30., .26


class SensorTape:
    """Instrument one observation instance without advancing its native delay twice."""

    def __init__(self, observer, live, steps):
        self.observer, self.live, self.count = observer, live, 0
        self.original_raw = observer.raw_sensors
        self.original_push = observer.delay.push
        self.original_sample = observer.sample_substep
        self.raw = torch.empty((steps+1, len(live), 18), device=live.device)
        self.valid = torch.empty((steps+1, len(live)), dtype=torch.bool, device=live.device)
        self.head = np.empty(steps+1, dtype=np.int64)
        self.raw_calls, self.instrumentation_raw_calls = 0, 0
        self.push_calls, self.sample_calls = 0, 0
        self.push_value_residual = torch.zeros((), device=live.device)

    def raw_source(self):
        self.raw_calls += 1
        return self.original_raw()

    def push(self, sample):
        self.push_calls += 1
        return self.original_push(sample)

    def record_raw_read(self):
        self.instrumentation_raw_calls += 1
        return self.observer.raw_sensors()

    def initial(self):
        self.raw[0].copy_(self.record_raw_read())
        self.valid[0].copy_(self.live)
        self.head[0] = self.observer.delay.head

    def sample_substep(self):
        before_pushes, before_head = self.push_calls, self.observer.delay.head
        self.original_sample()
        self.sample_calls += 1
        assert self.push_calls == before_pushes+1, 'Native sampling must push exactly once'
        assert self.observer.delay.head == (before_head+1) % len(self.observer.delay.history)
        # 原生sample_substep已完成；这里只再次读取primitive，不push、不加噪声。
        sample = self.record_raw_read()
        self.count += 1
        self.raw[self.count].copy_(sample)
        self.valid[self.count].copy_(self.live)
        self.head[self.count] = self.observer.delay.head
        difference = (sample-self.observer.delay.history[self.observer.delay.head]).abs().max()
        self.push_value_residual = torch.maximum(self.push_value_residual, difference)


@torch.inference_mode()
def run(args, cases, registry):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from rsl_rl.models import MLPModel
    from ct_mjlab.envs.chuanliantui_split.observations import get_state

    cfg, train = registry.get_cfgs('chuanliantui_walk', play=True)
    cfg.auto_reset, cfg.seed = False, args.seed
    cfg.metrics['walk_diagnosis'] = MetricsTermCfg(
        func=sample_physics, per_substep=True, reduce='last')
    count = len(cases)*NUM_PER_CASE
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
        observer = get_state(raw)
        assert not observer.cfg.noise, 'This records the native nominal encoder baseline'
        assert not observer.cfg.delay and not bool(observer.delay.lags.any()), \
            'Native control must have zero sensor delay in every group'
        native_history_buffer = raw.observation_manager._group_obs_term_history_buffer['history']['proprioception']
        runtime_sources = {}
        for name, component in (('ObservationManager', raw.observation_manager),
                                ('CircularBuffer', native_history_buffer)):
            path = Path(inspect.getfile(type(component))).resolve()
            copy_name = f'external-{name}.py'
            shutil.copy2(path, args.output/copy_name)
            runtime_sources[name] = dict(path=str(path), sha256=digest(path), copied_source=copy_name)
        recorder = Recorder(raw, round(SECONDS/raw.physics_dt))
        raw._walk_diagnosis = recorder
        policy_steps, physics_steps = round(SECONDS/raw.step_dt), round(SECONDS/raw.physics_dt)
        tape = SensorTape(observer, recorder.live, physics_steps)
        poses = np.empty((policy_steps+1, count, fixed['qpos'].shape[-1]), dtype=fixed['qpos'].dtype)
        poses[0] = fixed['qpos']
        pose_live = np.ones((policy_steps+1, count), dtype=bool)
        observation = np.empty((policy_steps, count, 25), dtype=np.float32)
        history = np.empty((policy_steps, count, 125), dtype=np.float32)
        raw_before = np.empty((policy_steps, count, 18), dtype=np.float32)
        native_read = np.empty_like(raw_before)
        source_head = np.empty(policy_steps, dtype=np.int64)
        source_lags = np.empty((policy_steps, count, 4), dtype=np.int64)
        source_indices = np.empty_like(source_lags)
        encoder_m_s = np.empty((policy_steps, count, 3), dtype=np.float32)
        target_com_m_s = np.empty_like(encoder_m_s)
        native_action = np.empty((policy_steps, count, 6), dtype=np.float32)
        policy_valid = np.empty((policy_steps, count), dtype=bool)
        reset_after = np.empty_like(policy_valid)
        ends = np.full(count, SECONDS)
        boundary_raw_residual, native_packet_residual, history_residual = 0., 0., 0.
        scene = args.output/'encoder-scene.mjb'
        mujoco.mj_saveModel(raw.sim.mj_model, str(scene), None)
        with patch.object(observer, 'raw_sensors', tape.raw_source), \
                patch.object(observer.delay, 'push', tape.push), \
                patch.object(observer, 'sample_substep', tape.sample_substep), \
                patch.object(pd, 'apply_actions', recorder.apply):
            tape.initial()
            for tick in range(policy_steps):
                assert tape.count == 5*tick
                commands(tick*raw.step_dt)
                base = MLPModel.get_latent(actor, obs)
                assert base.shape == (count, 25) and obs['history'].shape == (count, 125)
                live = recorder.live
                current_raw = tape.record_raw_read()
                delayed = observer.delay.read()
                if live.any():
                    boundary_raw_residual = max(boundary_raw_residual,
                        float((current_raw[live]-tape.raw[tape.count, live]).abs().max()))
                # commands与上一动作保持当前原生25槽，不随影子传感延迟移动。
                reconstructed = torch.cat((delayed[:, :3]*.25, delayed[:, 3:6],
                    base[:, 6:9], delayed[:, 6:10],
                    delayed[:, 12:18][:, (0, 1, 4, 2, 3, 5)]*.05, pd.raw_action), -1).clamp(-100., 100.)
                native_packet_residual = max(native_packet_residual, float((base-reconstructed).abs().max()))
                history_residual = max(history_residual, float((obs['history'][:, -25:]-base).abs().max()))
                if tick == 0:
                    assert torch.equal(obs['history'], base.repeat(1, 5))
                assert torch.equal(base[:, -6:], pd.raw_action)
                observation[tick], history[tick] = base.cpu().numpy(), obs['history'].cpu().numpy()
                raw_before[tick], native_read[tick] = current_raw.cpu().numpy(), delayed.cpu().numpy()
                source_head[tick] = observer.delay.head
                source_lags[tick] = observer.delay.lags.cpu().numpy()
                source_indices[tick] = np.maximum(tape.count-source_lags[tick], 0)
                policy_valid[tick] = live.cpu().numpy()
                velocity = actor.encoder(obs['history'])
                recorder.encoder = velocity/2.
                recorder.true_before = raw.scene['robot'].data.root_com_lin_vel_b.clone()
                encoder_m_s[tick] = recorder.encoder.cpu().numpy()
                target_com_m_s[tick] = recorder.true_before.cpu().numpy()
                recorder.policy_velocity = velocity/2.
                recorder.policy_wheel_velocity = base[:, (15, 18)]/.05
                # 始终原encoder控制；没有影子估速、补偿、滤波或融合参与动作。
                action = actor.mlp(torch.cat((base, velocity), -1))
                if train.clip_actions is not None:
                    action = action.clamp(-train.clip_actions, train.clip_actions)
                native_action[tick] = action.cpu().numpy()
                recorder.action = action
                obs, _, terminated, timeout, _ = raw.step(action)
                done = terminated | timeout
                first = done & recorder.live
                ends[first.cpu().numpy()] = (tick+1)*raw.step_dt
                recorder.live &= ~done
                reset_after[tick] = done.cpu().numpy()
                poses[tick+1] = raw.sim.data.qpos.cpu().numpy()
                pose_live[tick+1] = recorder.live.cpu().numpy()
                if done.any() and tick+1 < policy_steps:
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs = raw.get_observations()
                if (tick+1) % 500 == 0:
                    print(args.schedule, 'native encoder', (tick+1)*raw.step_dt,
                        'seconds, live', int(recorder.live.sum()), flush=True)
        assert recorder.count == tape.count == tape.sample_calls == tape.push_calls == physics_steps
        assert tape.instrumentation_raw_calls == physics_steps+1+policy_steps
        assert boundary_raw_residual == native_packet_residual == history_residual == 0.
        assert float(tape.push_value_residual) == 0.
        assert all(torch.equal(before[key], value.cpu()) for key,value in actor.state_dict().items())
        after_sha = actor_digest(actor.state_dict())
        assert after_sha == before_sha
        values = recorder.trace.cpu().numpy()
        raw_samples, raw_valid = tape.raw.cpu().numpy(), tape.valid.cpu().numpy()
        assert all(np.isfinite(array).all() for array in (values, poses, observation, history,
            raw_before, native_read, raw_samples, encoder_m_s, target_com_m_s, native_action))
        assert np.array_equal(policy_valid, values[::5, :, recorder.layout['valid'][0]] > .5)
        assert np.array_equal(encoder_m_s, values[::5, :, slice(*recorder.layout['encoder_velocity'])])
        assert np.array_equal(target_com_m_s, values[::5, :, slice(*recorder.layout['true_velocity_before'])])
        np.savez_compressed(args.output/'encoder.npz', values=values,
            initial_qpos=fixed['qpos'], initial_qvel=fixed['qvel'],
            qpos=poses, qpos_time_s=np.arange(policy_steps+1)*raw.step_dt, qpos_live=pose_live,
            raw_sensors=raw_samples, raw_sensor_time_s=np.arange(physics_steps+1)*raw.physics_dt,
            raw_sensor_valid=raw_valid, raw_sensor_head=tape.head,
            raw_sensors_before_policy=raw_before, native_delayed_sensors_before_policy=native_read,
            read_source_head=source_head, read_source_lags=source_lags, read_source_raw_index=source_indices,
            policy_observation=observation, native_history=history,
            policy_previous_action=observation[:, :, -6:], policy_command_scaled=observation[:, :, 6:9],
            native_action=native_action, policy_time_s=np.arange(policy_steps)*raw.step_dt,
            policy_valid=policy_valid, reset_after_policy=reset_after,
            encoder_m_s=encoder_m_s, target_com_m_s=target_com_m_s, used_vx_m_s=encoder_m_s[:, :, 0],
            **{'fixed_'+key:value for key,value in fixed.items()})
        metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_iteration=2999, label='encoder', feedback_mode='encoder', velocity_source='encoder',
            actual_profile='chuanliantui-walk-physical25-v8', parameter_updates=0,
            actor_unchanged=True, encoder_unchanged=True, actor_state_sha256=before_sha,
            actor_state_sha256_after=after_sha, schedule=args.schedule, seed=args.seed, seconds=SECONDS,
            height=HEIGHT, warmup=2. if args.schedule == 'study' else 3.,
            dt=raw.physics_dt, policy_dt=raw.step_dt, cases=cases, layout=recorder.layout,
            joint_names=recorder.joint_names, weight_N=float(raw.sim.mj_model.body_mass.sum()*9.81),
            episode_end=ends.tolist(), episode_done=(~recorder.live).cpu().tolist(), survived=int(recorder.live.sum()),
            fixed_arrays=list(fixed), scene_mjb=str(scene.resolve()), scene_sha256=digest(scene),
            native_observation_cfg=dict(noise=observer.cfg.noise, delay=observer.cfg.delay,
                delay_choices_physics_samples=observer.lag_choices.cpu().tolist(),
                noise_bounds_primitive=observer.noise_bounds.cpu().tolist()),
            runtime_dependency_sources=runtime_sources,
            raw_sensor_source='Actual observation_state.raw_sensors() after native sample_substep; raw qpos/qvel primitive API, never pose differentiation',
            raw_sensor_layout=dict(gyro_body_rad_s=[0, 3], projected_gravity=[3, 6],
                centered_motor_q_rad=[6, 12], motor_dq_rad_s=[12, 18]),
            primitive_motor_order=['lf0', 'lf00', 'rf0', 'rf00', 'lfwheel', 'rfwheel'],
            primitive_wheel_position_slots='raw[:,10:12] remain native zero placeholders',
            native_history_key='native_history', native_history_order='oldest to newest, five25 frames; reset backfills first frame',
            raw_sample_index='S[0]=t0 after reset and initial observations; S[i]=post-step t=i*.002 for i>=1. Live policy k at t=k*.01 reads physical S[5*k].',
            delay_read_alignment='read_source_head/lags plus read_source_raw_index[k,env,group]=max(5*k-lag,0) for first episodes; native_delayed_sensors_before_policy is the exact read. Initial native history uses repeated S0.',
            source_call_counts=dict(sample_substep=tape.sample_calls, native_delay_push=tape.push_calls,
                raw_sensors_all=tape.raw_calls, recording_raw_sensors=tape.instrumentation_raw_calls,
                native_raw_sensors=tape.raw_calls-tape.instrumentation_raw_calls,
                recorded_initial=1, recorded_post_substep=tape.count, recorded_pre_policy=policy_steps),
            exact_alignment_residuals=dict(post_substep_raw_vs_native_pushed=float(tape.push_value_residual),
                live_policy_raw_vs_S5k=boundary_raw_residual, native_base25_vs_delay_read=native_packet_residual,
                history_last25_vs_base25=history_residual),
            reset_scope='First-episode policy_valid/raw_sensor_valid exclude later episodes; reset_after_policy records native local resets. Do not concatenate raw frames across resets or assume invalid S5k equals the new reset state.',
            target_com_scope='Pre-action root body inertial COM in root link frame, privileged scoring only, derived fields can lag <=2ms.',
            shadow_scope='Recording only; no shadow computation or estimate enters native history, encoder or action.',
            timing='Raw S is sampled after each native observation sample_substep; policy base25/history/encoder/COM are captured together before action. Current command and previous action stay in their original native25 slots.',
            scope='Frozen native nominal sensor collection only; no training, shadow control or behavior-acceptance verdict')
        write(args.output/'encoder.json', metadata)
        return metadata
    finally:
        raw.close()


def main(args):
    assert args.seed == SCHEDULE_SEEDS[args.schedule]
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    args.training_artifact, args.checkpoint, args.output = (
        path.resolve() for path in (args.training_artifact, args.checkpoint, args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    launch = json.loads((args.training_artifact/'launch.json').read_text())
    hashes = json.loads((args.training_artifact/'source-sha256.json').read_text())
    assets = json.loads((args.training_artifact/'physical-files-sha256.json').read_text())
    assert len(hashes) == 90 and len(assets) == 40
    runtime = args.output/'frozen-runtime'
    source = runtime/'mjlab_training/src/ct_mjlab'
    shutil.copytree(Path(launch['log_dir'])/'snapshot/ct_mjlab', source,
        ignore=shutil.ignore_patterns('__pycache__'))
    assert all(digest(source/name) == sha for name,sha in hashes.items())
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
    collector_sha = digest(Path(__file__).resolve())
    helper_sources = [Path(shared.__file__).resolve(), Path(schedule_source.__file__).resolve(),
                      Path(inspect.getfile(actor_digest)).resolve()]
    helper_hashes = {str(path):digest(path) for path in helper_sources}
    shutil.copy2(helper_sources[0], args.output/'recorder-source.py')
    shutil.copy2(helper_sources[1], args.output/'schedule-source.py')
    shutil.copy2(helper_sources[2], args.output/'actor-digest-source.py')
    shutil.copy2(__file__, args.output/'source.py')
    sys.path.insert(0, str(source.parent))
    from ct_mjlab.envs import task_registry
    cases = schedule_source.make_cases('discovery' if args.schedule == 'study' else 'validation')
    metadata = run(args, cases, task_registry)
    assert digest(Path(__file__).resolve()) == collector_sha
    assert digest(args.checkpoint) == args.checkpoint_sha256
    assert all(digest(Path(path)) == sha for path,sha in helper_hashes.items())
    assert all(digest(source/name) == sha for name,sha in hashes.items())
    assert all(digest(runtime/name) == sha for name,sha in assets.items())
    assert all(digest(Path(item['path'])) == item['sha256']
               for item in metadata['runtime_dependency_sources'].values())
    write(args.output/'result.json', dict(verified=True, parameter_updates=0,
        source_files=len(hashes), resource_files=len(assets), resources_copied=True,
        collector_source_sha256=collector_sha, helper_source_sha256=helper_hashes,
        runtime_dependency_sources=metadata['runtime_dependency_sources'],
        schedule=args.schedule, seed=args.seed,
        source_call_counts=metadata['source_call_counts'], runs=[metadata],
        scope='Native sensor collection integrity only; no shadow control or task-acceptance claim'))
    print(json.dumps(dict(verified=True, parameter_updates=0, schedule=args.schedule,
        seed=args.seed, survived=metadata['survived'], source_call_counts=metadata['source_call_counts']),
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--schedule', choices=tuple(SCHEDULE_SEEDS), required=True)
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Shadow collection cannot train')):
        main(parser.parse_args())
