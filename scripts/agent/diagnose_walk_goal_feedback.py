#!/usr/bin/env python3
"""Frozen 30s stop/travel/brake comparison of encoder and sensor-wheel vx."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

import diagnose_walk_v8_regression as shared
from diagnose_walk_v8_regression import Recorder, digest, sample_physics, write


SEED, NUM_PER_CASE = 43, 16
SECONDS, WARMUP, STOP_AT, HEIGHT = 30., 2., 16., .26
WHEEL_RADIUS, WHEEL_OBSERVATION_SCALE = .06, .05
WHEEL_OBSERVATION_IDS = (15, 18)


def actor_digest(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(key.encode())
        value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def make_cases():
    cases = []
    for acceleration in (.5, 2.):
        for speed in (0., .5, -.5):
            first = len(cases) * NUM_PER_CASE
            cases.append(dict(name=f'v{speed:g}-a{acceleration:g}', speed=speed,
                acceleration=acceleration, env_ids=list(range(first, first+NUM_PER_CASE))))
    return cases


@torch.inference_mode()
def run(args, source, cases, reference, registry):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from rsl_rl.models import MLPModel

    cfg, train = registry.get_cfgs('chuanliantui_walk', play=True)
    cfg.auto_reset, cfg.seed = False, SEED
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
        # 原生严格加载角色/配方及课程，只加载Actor，不恢复或执行优化器。
        built.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True,
            map_location=args.device)
        actor = built.alg.actor
        actor.eval()
        assert not actor.obs_normalization
        before = {key:value.detach().cpu().clone() for key,value in actor.state_dict().items()}
        before_sha = actor_digest(before)
        state = raw.command_manager.get_term('standup')
        state.speed_curriculum_frozen = True
        raw.reset(seed=SEED)
        pd = raw.action_manager.get_term('pd')
        fixed = dict(qpos=raw.sim.data.qpos.cpu().numpy().copy(),
            qvel=raw.sim.data.qvel.cpu().numpy().copy(),
            nominal=pd._nominal_pos.cpu().numpy().copy(), kp=pd.kp_scale.cpu().numpy().copy(),
            kd=pd.kd_scale.cpu().numpy().copy(), strength=pd.motor_strength.cpu().numpy().copy(),
            mass=raw.sim.mj_model.body_mass.copy(), friction=raw.sim.mj_model.geom_friction.copy())
        if reference is not None:
            assert all(np.array_equal(fixed[key], reference[key]) for key in fixed), \
                'Initial/physical arrays differ between feedback sources'
        targets = torch.empty(count, device=args.device)
        accelerations = torch.empty_like(targets)
        for case in cases:
            targets[case['env_ids']] = case['speed']
            accelerations[case['env_ids']] = case['acceleration']

        def commands(time):
            state.tracking_episode.fill_(True)
            state.zero_hold.zero_()
            requested = targets if WARMUP <= time < STOP_AT else torch.zeros_like(targets)
            state.tracking_velocity.copy_(requested)
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
        policy_live = np.ones((policy_steps+1, count), dtype=bool)
        ends = np.full(count, SECONDS)
        scene = args.output/(source+'-scene.mjb')
        mujoco.mj_saveModel(raw.sim.mj_model, str(scene), None)
        with patch.object(pd, 'apply_actions', recorder.apply):
            for tick in range(policy_steps):
                commands(tick*raw.step_dt)
                base = MLPModel.get_latent(actor, obs)
                assert base.shape == (count, 25)
                velocity = actor.encoder(obs['history'])
                recorder.encoder = velocity/2.
                recorder.true_before = raw.scene['robot'].data.root_com_lin_vel_b.clone()
                wheel_velocity = base[:, WHEEL_OBSERVATION_IDS]/WHEEL_OBSERVATION_SCALE
                policy_velocity = velocity.clone()
                if source == 'wheel-vx':
                    # 仅替换前向速度反馈；轮速来自Actor同拍传感槽，不取机身真值。
                    wheel_vx = .5*WHEEL_RADIUS*(wheel_velocity[:, 0]-wheel_velocity[:, 1])
                    policy_velocity[:, 0] = wheel_vx*2.
                recorder.policy_velocity = policy_velocity/2.
                recorder.policy_wheel_velocity = wheel_velocity
                # 两支采用完全相同的Actor调用路径，差异只有上面的vx值。
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
                policy_live[tick+1] = recorder.live.cpu().numpy()
                if done.any() and tick+1 < policy_steps:
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs = raw.get_observations()
                if (tick+1) % 500 == 0:
                    print(source, (tick+1)*raw.step_dt, 'seconds, live',
                        int(recorder.live.sum()), flush=True)
        assert recorder.count == round(SECONDS/raw.physics_dt)
        assert all(torch.equal(before[key], value.cpu()) for key,value in actor.state_dict().items())
        after_sha = actor_digest(actor.state_dict())
        assert after_sha == before_sha
        values = recorder.trace.cpu().numpy()
        assert np.isfinite(values).all() and np.isfinite(poses).all()
        required = ('roll_pitch', 'angular_velocity_xy', 'root_linear_velocity_w', 'joint_dq',
                    'policy_velocity', 'policy_wheel_velocity')
        assert all(key in recorder.layout for key in required), 'Recorder extension is missing'
        result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_iteration=2999, label=source, feedback_mode=source, velocity_source=source,
            parameter_updates=0, actor_unchanged=True, encoder_unchanged=True,
            actor_state_sha256=before_sha, actor_state_sha256_after=after_sha,
            seed=SEED, seconds=SECONDS, warmup=WARMUP, stop_at=STOP_AT, height=HEIGHT,
            dt=raw.physics_dt, policy_dt=raw.step_dt,
            weight_N=float(raw.sim.mj_model.body_mass.sum()*9.81), cases=cases,
            layout=recorder.layout, joint_names=recorder.joint_names, episode_end=ends.tolist(),
            survived=int(recorder.live.sum()), episode_done=(~recorder.live).cpu().tolist(),
            initial_identical_to_other_source=reference is not None,
            fixed_arrays=list(fixed), scene_mjb=str(scene.resolve()), scene_sha256=digest(scene),
            qpos_key='qpos', qpos_time_key='qpos_time_s', qpos_live_key='qpos_live',
            wheel_radius_m=WHEEL_RADIUS, wheel_vx_coefficient_m=.5*WHEEL_RADIUS,
            wheel_feedback=dict(radius_m=WHEEL_RADIUS, wheel_radius_m=WHEEL_RADIUS,
                wheel_vx_coefficient_m=.5*WHEEL_RADIUS, forward_signs=[1, -1],
                actor_indices=list(WHEEL_OBSERVATION_IDS), observation_scale=WHEEL_OBSERVATION_SCALE,
                formula_m_s='.5*.06*(actor[15]/.05-actor[18]/.05)',
                replacement='vx only, multiplied by 2 for actor latent; vy/vz retain encoder'),
            schedule='0<=t<2: zero; 2<=t<16: case target; 16<=t<=30: zero; native command ramp/parking/yaw/down retained',
            timing='Policy encoder/COM/wheel feedback latched before action; PD target before each 2ms integration; physical fields sampled after substep, derived fields may lag <=2ms',
            scope='Frozen nominal stop/travel/brake feedback diagnosis only; raw data, no fixed-target final2s summary and no task-acceptance claim')
        np.savez_compressed(args.output/(source+'.npz'), values=values,
            initial_qpos=fixed['qpos'], initial_qvel=fixed['qvel'],
            qpos=poses, qpos_time_s=np.arange(policy_steps+1)*raw.step_dt, qpos_live=policy_live,
            **{'fixed_'+key:value for key,value in fixed.items()})
        write(args.output/(source+'.json'), result)
        return result, fixed
    finally:
        raw.close()


def main(args):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    args.training_artifact = args.training_artifact.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
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
    # 复制并核对资源，避免运行中的源码/物理变动改变本次冻结环境。
    for name, sha in assets.items():
        assert digest(root/name) == sha, name
        destination = runtime/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, destination)
        assert digest(destination) == sha, name
    args.checkpoint_sha256 = digest(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert saved['iter'] == 2999, 'This probe is registered for checkpoint iteration 2999'
    del saved
    recorder_file = Path(shared.__file__).resolve()
    recorder_sha = digest(recorder_file)
    shutil.copy2(recorder_file, args.output/'recorder-source.py')
    shutil.copy2(__file__, args.output/'source.py')
    sys.path.insert(0, str(source.parent))
    from ct_mjlab.envs import task_registry
    cases = make_cases()
    reports, reference = [], None
    for label in ('encoder', 'wheel-vx'):
        report, fixed = run(args, label, cases, reference, task_registry)
        if reference is None:
            reference = fixed
        reports.append(report)
    assert len({report['actor_state_sha256'] for report in reports}) == 1
    assert digest(args.checkpoint) == args.checkpoint_sha256
    assert digest(recorder_file) == recorder_sha
    assert all(digest(source/name) == sha for name,sha in source_hashes.items())
    assert all(digest(runtime/name) == sha for name,sha in assets.items())
    write(args.output/'result.json', dict(verified=True, parameter_updates=0,
        source_files=len(source_hashes), resource_files=len(assets), resources_copied=True,
        initial_and_physical_states_equal=True, same_checkpoint_and_actor_sha=True,
        recorder_source_sha256=recorder_sha, wheel_radius_m=WHEEL_RADIUS,
        wheel_vx_coefficient_m=.5*WHEEL_RADIUS, seed=SEED, seconds=SECONDS, warmup=WARMUP,
        stop_at=STOP_AT, height=HEIGHT, cases=cases, runs=reports,
        scope='Data-collection integrity only; no behavior acceptance verdict'))
    print(json.dumps(dict(verified=True, parameter_updates=0,
        runs=[dict(label=report['label'], survived=report['survived']) for report in reports]),
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Diagnosis cannot train')):
        main(parser.parse_args())
