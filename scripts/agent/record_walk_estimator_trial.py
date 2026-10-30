#!/usr/bin/env python3
"""Frozen parent/candidate estimator trial, with native500Hz sensors and own feedback."""

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

import diagnose_walk_v8_regression as shared
from diagnose_walk_goal_feedback import actor_digest
from diagnose_walk_v8_regression import Recorder, digest, sample_physics, write


SCHEDULE_SEEDS = {'final48': 48, 'final49': 49, 'regression43': 43}
NUM_PER_CASE, SECONDS, HEIGHT = 16, 30., .26
PARENT_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
MODEL_FIELDS = ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'geom_friction',
                'geom_pos', 'geom_quat', 'geom_size', 'jnt_stiffness', 'dof_damping')


def tree_equal(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def verify_candidate(parent_path, candidate_path):
    assert digest(parent_path) == PARENT_SHA
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
    assert parent.keys() == candidate.keys(), 'Paired model parent fields differ'
    assert parent['iter'] == 2999
    assert parent['actor_state_dict'].keys() == candidate['actor_state_dict'].keys()
    for key in parent:
        if key != 'actor_state_dict':
            assert tree_equal(parent[key], candidate[key]), f'Changed parent field: {key}'
    for key, value in parent['actor_state_dict'].items():
        if not key.startswith('encoder.'):
            assert tree_equal(value, candidate['actor_state_dict'][key]), f'Changed controller: {key}'
    encoder = {key.removeprefix('encoder.'):value for key,value in candidate['actor_state_dict'].items()
               if key.startswith('encoder.')}
    assert set(encoder) == {'0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias'}
    assert all(bool(torch.isfinite(value).all()) for value in encoder.values())
    return encoder, dict(parent_sha256=PARENT_SHA, candidate_sha256=digest(candidate_path),
        parent_fields_except_encoder_bitwise_equal=True, candidate_encoder_sha256=actor_digest(encoder),
        parent_encoder_sha256=actor_digest({key.removeprefix('encoder.'):value for key,value in
            parent['actor_state_dict'].items() if key.startswith('encoder.')}))


def make_cases(schedule):
    accelerations = {'final48':(.6, 1.4), 'final49':(.75, 1.5), 'regression43':(.5, 2.)}[schedule]
    cases = []
    for acceleration in accelerations:
        for speed in (0., .5, -.5):
            first = len(cases)*NUM_PER_CASE
            if schedule == 'final48':
                phases = ((0., 2.5, 0.), (2.5, 13., speed), (13., 19., 0.),
                          (19., 25., -speed), (25., 30., 0.))
            elif schedule == 'final49':
                phases = ((0., 4., 0.), (4., 11., speed), (11., 17., 0.),
                          (17., 24., -speed), (24., 30., 0.))
            else:
                phases = ((0., 2., 0.), (2., 16., speed), (16., 30., 0.))
            cases.append(dict(name=f'v{speed:g}-a{acceleration:g}', speed=speed,
                acceleration=acceleration, env_ids=list(range(first, first+NUM_PER_CASE)),
                segments=[dict(start=start, end=end, speed=target, acceleration=acceleration)
                          for start,end,target in phases]))
    return cases


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
        self.physics_qpos = torch.empty((steps+1, *observer.env.sim.data.qpos.shape), device=live.device)

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
        self.physics_qpos[0].copy_(self.observer.env.sim.data.qpos)

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
        self.physics_qpos[self.count].copy_(self.observer.env.sim.data.qpos)
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
        built.load(str(args.parent), load_cfg={'actor': True}, strict=True,
            map_location=args.device)
        actor = built.alg.actor
        parent_control = {key:value for key,value in actor.state_dict().items() if not key.startswith('encoder.')}
        assert actor_digest(parent_control) == args.controller_sha256
        if args.feedback == 'candidate':
            actor.encoder.load_state_dict(args.candidate_encoder, strict=True)
        actor.eval()
        actor.requires_grad_(False)
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
        for name in MODEL_FIELDS:
            fixed[name] = np.asarray(getattr(raw.sim.mj_model, name)).copy()
            fixed['actual_model_'+name] = getattr(raw.sim.model, name).cpu().numpy().copy()
        fixed['position_gains'] = pd.position_gains.cpu().numpy().copy()
        fixed['velocity_gains'] = pd.velocity_gains.cpu().numpy().copy()
        fixed['env_origins'] = raw.scene.env_origins.cpu().numpy().copy()
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
                                ('CircularBuffer', native_history_buffer), ('Simulator', raw.sim),
                                ('Environment', raw), ('MLPModel', actor)):
            path = Path(inspect.getfile(type(component))).resolve()
            copy_name = f'external-{name}.py'
            shutil.copy2(path, args.output/copy_name)
            runtime_sources[name] = dict(path=str(path), sha256=digest(path), copied_source=copy_name)
        engine_identity = dict(mujoco=mujoco.__version__, torch=torch.__version__, numpy=np.__version__,
            runtime_source_sha256={name:item['sha256'] for name,item in runtime_sources.items()},
            model_timestep=float(raw.sim.mj_model.opt.timestep),
            solver=int(raw.sim.mj_model.opt.solver), iterations=int(raw.sim.mj_model.opt.iterations),
            ls_iterations=int(raw.sim.mj_model.opt.ls_iterations),
            integrator=int(raw.sim.mj_model.opt.integrator), cone=int(raw.sim.mj_model.opt.cone))
        protocol_identity = dict(seed=args.seed, schedule=args.schedule, cases=cases,
            seconds=SECONDS, height_m=HEIGHT, physics_dt_s=.002, policy_dt_s=.01,
            external_yaw_request=0., native_heading_and_parking=True, noise=False, delay=False,
            collector_source_sha256=digest(__file__), parent_sha256=PARENT_SHA,
            frozen_source_manifest_sha256=args.source_manifest_sha256,
            physical_manifest_sha256=args.physical_manifest_sha256)
        baseline_evidence = None
        if args.baseline is not None:
            baseline_meta_path = args.baseline/'encoder.json'
            baseline_trace_path = args.baseline/'encoder.npz'
            baseline_meta = json.loads(baseline_meta_path.read_text())
            assert json.loads((args.baseline/'result.json').read_text())['verified']
            assert baseline_meta['feedback_branch'] == 'parent'
            assert baseline_meta['protocol_identity'] == protocol_identity
            assert baseline_meta['engine_identity'] == engine_identity
            assert baseline_meta['controller_sha256'] == args.controller_sha256
            assert baseline_meta['candidate_checkpoint_sha256'] == args.candidate_sha256
            with np.load(baseline_trace_path, allow_pickle=False) as saved:
                assert baseline_meta['fixed_arrays'] == list(fixed)
                equality = {key:(fixed[key].dtype == saved['fixed_'+key].dtype
                    and fixed[key].shape == saved['fixed_'+key].shape
                    and np.array_equal(fixed[key], saved['fixed_'+key])) for key in fixed}
                assert all(equality.values()), 'Initial/actual physical arrays differ before first action'
            baseline_evidence = dict(path=str(args.baseline), metadata_sha256=digest(baseline_meta_path),
                trace_sha256=digest(baseline_trace_path), all_fixed_arrays_bitwise_equal=equality,
                protocol_engine_and_controller_equal=True, checked_before_first_action=True)
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
        termination_names = list(raw.termination_manager.active_terms)
        termination_flags = np.empty((policy_steps, count, len(termination_names)), dtype=bool)
        first_termination = [None]*count
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
                # 所选encoder实际驱动冻结控制MLP；历史/上一动作来自本分支自身。
                action = actor.mlp(torch.cat((base, velocity), -1))
                if train.clip_actions is not None:
                    action = action.clamp(-train.clip_actions, train.clip_actions)
                native_action[tick] = action.cpu().numpy()
                recorder.action = action
                obs, _, terminated, timeout, _ = raw.step(action)
                done = terminated | timeout
                first = done & recorder.live
                terms = torch.stack([raw.termination_manager.get_term(name) for name in termination_names], -1)
                termination_flags[tick] = terms.cpu().numpy()
                for env in first.nonzero().flatten().cpu().tolist():
                    first_termination[env] = dict(time_s=(tick+1)*raw.step_dt,
                        terms=[name for index,name in enumerate(termination_names) if bool(terms[env, index])])
                ends[first.cpu().numpy()] = (tick+1)*raw.step_dt
                recorder.live &= ~done
                reset_after[tick] = done.cpu().numpy()
                poses[tick+1] = raw.sim.data.qpos.cpu().numpy()
                pose_live[tick+1] = recorder.live.cpu().numpy()
                if done.any() and tick+1 < policy_steps:
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs = raw.get_observations()
                if (tick+1) % 500 == 0:
                    print(args.schedule, args.feedback, 'encoder', (tick+1)*raw.step_dt,
                        'seconds, live', int(recorder.live.sum()), flush=True)
        assert recorder.count == tape.count == tape.sample_calls == tape.push_calls == physics_steps
        assert tape.instrumentation_raw_calls == physics_steps+1+policy_steps
        assert boundary_raw_residual == native_packet_residual == history_residual == 0.
        assert float(tape.push_value_residual) == 0.
        assert all(torch.equal(before[key], value.cpu()) for key,value in actor.state_dict().items())
        after_sha = actor_digest(actor.state_dict())
        assert after_sha == before_sha
        assert actor_digest({key:value for key,value in actor.state_dict().items()
            if not key.startswith('encoder.')}) == args.controller_sha256
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
            physics_qpos=tape.physics_qpos.cpu().numpy(),
            physics_qpos_time_s=np.arange(physics_steps+1)*raw.physics_dt,
            physics_qpos_live=raw_valid,
            raw_sensors=raw_samples, raw_sensor_time_s=np.arange(physics_steps+1)*raw.physics_dt,
            raw_sensor_valid=raw_valid, raw_sensor_head=tape.head,
            raw_sensors_before_policy=raw_before, native_delayed_sensors_before_policy=native_read,
            read_source_head=source_head, read_source_lags=source_lags, read_source_raw_index=source_indices,
            policy_observation=observation, native_history=history,
            policy_previous_action=observation[:, :, -6:], policy_command_scaled=observation[:, :, 6:9],
            native_action=native_action, policy_time_s=np.arange(policy_steps)*raw.step_dt,
            policy_valid=policy_valid, reset_after_policy=reset_after,
            termination_flags=termination_flags, termination_names=np.asarray(termination_names),
            encoder_m_s=encoder_m_s, target_com_m_s=target_com_m_s, used_vx_m_s=encoder_m_s[:, :, 0],
            **{'fixed_'+key:value for key,value in fixed.items()})
        used_checkpoint = args.parent if args.feedback == 'parent' else args.candidate
        used_sha = PARENT_SHA if args.feedback == 'parent' else args.candidate_sha256
        feedback_mode = 'encoder' if args.feedback == 'parent' else 'candidate'
        metadata = dict(checkpoint=str(used_checkpoint), checkpoint_sha256=used_sha,
            parent_checkpoint=str(args.parent), parent_checkpoint_sha256=PARENT_SHA,
            candidate_checkpoint=str(args.candidate), candidate_checkpoint_sha256=args.candidate_sha256,
            candidate_receipt_path=str(args.candidate_receipt), candidate_receipt_sha256=args.candidate_receipt_sha256,
            candidate_verification=args.candidate_evidence, feedback_branch=args.feedback,
            controller_unchanged=True, controller_sha256=args.controller_sha256,
            baseline_pairing=baseline_evidence, protocol_identity=protocol_identity,
            engine_identity=engine_identity, window_lock=None if args.window_lock is None else
                dict(path=str(args.window_lock), sha256=args.window_lock_sha256),
            checkpoint_iteration=2999, label=feedback_mode, feedback_mode=feedback_mode, velocity_source=feedback_mode,
            actual_profile='chuanliantui-walk-physical25-v8', parameter_updates=0,
            actor_unchanged=True, encoder_unchanged=True, actor_state_sha256=before_sha,
            actor_state_sha256_after=after_sha, schedule=args.schedule, seed=args.seed, seconds=SECONDS,
            height=HEIGHT, warmup=cases[0]['segments'][0]['end'],
            dt=raw.physics_dt, policy_dt=raw.step_dt, cases=cases, layout=recorder.layout,
            joint_names=recorder.joint_names, weight_N=float(raw.sim.mj_model.body_mass.sum()*9.81),
            episode_end=ends.tolist(), episode_done=(~recorder.live).cpu().tolist(), survived=int(recorder.live.sum()),
            first_termination=first_termination, termination_names=termination_names,
            root_qpos_adr=raw.scene['robot'].indexing.free_joint_q_adr.cpu().tolist(),
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
            target_com_scope='Pre-action root body inertial COM in root link frame after native policy-boundary forward; supervision/scoring only, never an estimator input and never shifted.',
            shadow_scope='No shadow replacement: selected original or candidate encoder output actually drives the frozen control MLP.',
            timing='Raw S is sampled after each native observation sample_substep; policy base25/history/encoder/COM are captured together before action. Current command and previous action stay in their original native25 slots.',
            scope='Frozen controller/encoder trial collection only; no training or behavior-acceptance verdict; candidate uses its own closed-loop observations/actions/history.')
        write(args.output/'encoder.json', metadata)
        return metadata
    finally:
        raw.close()


def main(args):
    args.seed = SCHEDULE_SEEDS[args.schedule]
    assert (args.feedback == 'candidate') == (args.baseline is not None), \
        'Candidate requires its corresponding parent baseline; parent must not replay a baseline'
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    args.training_artifact, args.parent, args.candidate, args.output = (
        path.resolve() for path in (args.training_artifact, args.parent, args.candidate, args.output))
    if args.baseline is not None:
        args.baseline = args.baseline.resolve()
    if args.window_lock is not None:
        args.window_lock = args.window_lock.resolve()
        args.window_lock_sha256 = digest(args.window_lock)
    args.candidate_encoder, args.candidate_evidence = verify_candidate(args.parent, args.candidate)
    args.candidate_sha256 = args.candidate_evidence['candidate_sha256']
    args.candidate_receipt = args.candidate.with_name('checkpoint-1000.json')
    candidate_receipt = json.loads(args.candidate_receipt.read_text())
    assert candidate_receipt['update'] == 1000 and candidate_receipt['sha256'] == args.candidate_sha256
    assert Path(candidate_receipt['path']).resolve() == args.candidate
    args.candidate_receipt_sha256 = digest(args.candidate_receipt)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    args.controller_sha256 = actor_digest({key:value for key,value in parent['actor_state_dict'].items()
        if not key.startswith('encoder.')})
    assert parent['infos']['experiment_profile_contract'] == 'chuanliantui-walk-physical25-v8'
    del parent
    args.output.mkdir(parents=True, exist_ok=False)
    launch = json.loads((args.training_artifact/'launch.json').read_text())
    hashes = json.loads((args.training_artifact/'source-sha256.json').read_text())
    assets = json.loads((args.training_artifact/'physical-files-sha256.json').read_text())
    args.source_manifest_sha256 = digest(args.training_artifact/'source-sha256.json')
    args.physical_manifest_sha256 = digest(args.training_artifact/'physical-files-sha256.json')
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
    collector_sha = digest(Path(__file__).resolve())
    helper_sources = [Path(shared.__file__).resolve(), Path(inspect.getfile(actor_digest)).resolve()]
    helper_hashes = {str(path):digest(path) for path in helper_sources}
    shutil.copy2(helper_sources[0], args.output/'recorder-source.py')
    shutil.copy2(helper_sources[1], args.output/'actor-digest-source.py')
    shutil.copy2(__file__, args.output/'source.py')
    sys.path.insert(0, str(source.parent))
    from ct_mjlab.envs import task_registry
    cases = make_cases(args.schedule)
    metadata = run(args, cases, task_registry)
    assert digest(Path(__file__).resolve()) == collector_sha
    assert digest(args.parent) == PARENT_SHA and digest(args.candidate) == args.candidate_sha256
    assert digest(args.candidate_receipt) == args.candidate_receipt_sha256
    if args.window_lock is not None:
        assert digest(args.window_lock) == args.window_lock_sha256
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
        feedback_branch=args.feedback, parent_checkpoint_sha256=PARENT_SHA,
        candidate_checkpoint_sha256=args.candidate_sha256, candidate_verification=args.candidate_evidence,
        source_call_counts=metadata['source_call_counts'], runs=[metadata],
        scope='Frozen estimator-in-loop collection integrity only; no task-acceptance claim'))
    print(json.dumps(dict(verified=True, parameter_updates=0, schedule=args.schedule,
        seed=args.seed, survived=metadata['survived'], source_call_counts=metadata['source_call_counts']),
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--feedback', choices=('parent', 'candidate'), required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--window-lock', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--schedule', choices=tuple(SCHEDULE_SEEDS), required=True)
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Frozen collection cannot train')):
        main(parser.parse_args())
