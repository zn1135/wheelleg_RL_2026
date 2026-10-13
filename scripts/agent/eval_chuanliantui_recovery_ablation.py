"""Evaluate four ablation actors with equal training budgets and paired resets.

Use the original Linux mjlab release/runtime. This does not update policies.
Default: latest checkpoint index shared by all four groups, 32 normal and 32
ground starts, seed 42, fixed zero velocity and 0.30 m height, 20 seconds.
Physical failure is recorded using the common baseline 20-step criterion;
rollouts continue after contact/tilt to give every ground start a full recovery
opportunity. Numerical failures are reset and permanently excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from ct_mjlab.experiments.recovery_ablation import (
    GROUPS, experiment_manifest, make_env_cfg, tensor_digest,
)
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training import huananhu_mdp
from ct_mjlab.reference_training import huananhu_observations
from ct_mjlab.reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ct_mjlab.reference_training.pd_action import CONTROL_CONTRACT


def canonical(value):
    return json.loads(json.dumps(value))


def failure_probe(env):
    failed = huananhu_mdp.failure(env)
    state = env.command_manager.get_term('standup')
    state.eval_common_failure = failed & ~state.numerical_failure
    return state.numerical_failure


def load_checked_actor(path, group, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    infos = saved.get('infos') or {}
    expected_cfg = make_env_cfg(group)
    # Scene construction synchronizes the terrain's initially unset population
    # and spacing with the scene. Check that exact normalization, preserving
    # strict comparisons of every other recorded source/configuration field.
    expected_cfg.scene.terrain.num_envs = expected_cfg.scene.num_envs
    expected_cfg.scene.terrain.env_spacing = expected_cfg.scene.env_spacing
    expected = experiment_manifest(group, expected_cfg)
    # Training launched this exact hash-pinned module with python -m, so its
    # experimental functions were serialized under __main__. Imported replay
    # uses their package names. Normalize only these known function fields.
    experiment_prefix = 'ct_mjlab.experiments.recovery_ablation.'
    for term in expected['environment']['rewards'].values():
        if term['func'].startswith(experiment_prefix):
            term['func'] = '__main__.' + term['func'].removeprefix(experiment_prefix)
    failure_term = expected['environment']['terminations']['failure']
    if failure_term['func'].startswith(experiment_prefix):
        failure_term['func'] = '__main__.' + failure_term['func'].removeprefix(experiment_prefix)
    if canonical(infos.get('recovery_ablation')) != canonical(expected):
        raise ValueError(f'{group}: source or experiment configuration mismatch')
    if (infos.get('control_contract') != CONTROL_CONTRACT or
            infos.get('observation_contract') != OBSERVATION_CONTRACT):
        raise ValueError(f'{group}: control or observation contract mismatch')
    actor = actor_model()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError(f'{group}: nonfinite actor parameters')
    metadata = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    iteration_index=saved['iter'], completed_updates=infos['huananhu_training_iteration'])
    return actor.to(device).eval(), metadata


def rollout(actor, pose, args, out, group, initial_states):
    torch.manual_seed(args.seed)
    cfg = make_env_cfg('baseline', args.num_envs, args.seed, play=True)
    cfg.auto_reset = False
    cfg.episode_length_s = args.seconds
    command = cfg.commands['standup']
    command.ground_start_probability = float(pose == 'ground')
    command.zero_velocity_commands = True
    command.height_range = (args.height, args.height)
    command.resampling_time_range = (1e6, 1e6)
    cfg.terminations['failure'] = TerminationTermCfg(func=failure_probe)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
    raw = env.unwrapped
    state = raw.command_manager.get_term('standup')
    robot = raw.scene['robot']
    if pose not in initial_states:
        initial_states[pose] = {name: value.detach().cpu().clone() for name, value in (
            ('qpos', raw.sim.data.qpos), ('qvel', raw.sim.data.qvel),
            ('command', state.command), ('origins', raw.scene.env_origins))}
        np.savez_compressed(out / f'initial-state-{pose}.npz',
                            **{k: v.numpy() for k, v in initial_states[pose].items()})
    initial = initial_states[pose]
    if not torch.equal(raw.scene.env_origins.cpu(), initial['origins']):
        raise ValueError('Terrain origins differ between paired environments')
    # GPU quaternion construction may vary in the last float32 bit between
    # repeated environments. Explicitly restore identical physical inputs.
    raw.scene.reset(torch.arange(args.num_envs, device=raw.device))
    raw.sim.data.qpos.copy_(initial['qpos'].to(raw.device))
    raw.sim.data.qvel.copy_(initial['qvel'].to(raw.device))
    state.sampled.copy_(initial['command'].to(raw.device))
    state.command.copy_(state.sampled)
    state.ground_command.zero_()
    raw.sim.forward()
    raw.sim.sense()
    huananhu_observations.reset_observations(raw, torch.arange(args.num_envs, device=raw.device))
    obs = env.get_observations()
    initial_hash = tensor_digest([(k, initial[k]) for k in ('qpos', 'qvel', 'command')])
    restore_delta = max(float((raw.sim.data.qpos.cpu() - initial['qpos']).abs().max()),
                        float((raw.sim.data.qvel.cpu() - initial['qvel']).abs().max()))
    if restore_delta > 1e-7:
        raise ValueError(f'Initial-state restore mismatch: {restore_delta}')
    n, device, dt = args.num_envs, raw.device, raw.step_dt
    steps = round(args.seconds / dt)
    active = torch.ones(n, dtype=torch.bool, device=device)
    numeric = torch.zeros_like(active)
    recovered = torch.zeros_like(active)
    hold = torch.zeros(n, device=device)
    first_recover = torch.full((n,), -1., device=device)
    first_failure = first_recover.clone()
    tail_count = torch.zeros(n, device=device)
    tail_height = tail_count.clone()
    tail_error = tail_count.clone()
    tail_speed = tail_count.clone()
    tail_strict = tail_count.clone()
    tail_base_contact = tail_count.clone()
    tail_leg_contact = tail_count.clone()
    tail_inverted = tail_count.clone()
    trace = []
    frame_qpos = [raw.sim.data.qpos[0].cpu().numpy().copy()]
    frame_times = [0.]
    try:
        for i in range(steps):
            with torch.inference_mode():
                actions = actor(obs)
                actions = torch.where(active[:, None], actions, 0.)
                obs, rewards, done, extra = env.step(actions)
            data = robot.data
            gravity = data.projected_gravity_b
            roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
            pitch = torch.asin(gravity[:, 0].clamp(-1., 1.)).abs()
            base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
            wheel = raw.scene['wheel_contact'].data.force_history.norm(dim=-1).amax(-1)
            leg = raw.scene['undesired_contact'].data.force.norm(dim=-1).amax(-1)
            height = data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            error = (height - args.height).abs()
            speed = data.root_link_lin_vel_b[:, :2].norm(dim=-1)
            healthy = active & ~state.numerical_failure
            strict = (healthy & (base <= 1.) & (roll <= torch.pi / 12)
                      & (pitch <= torch.pi / 12) & (error <= .03) & (wheel > 5.).all(-1))
            hold = torch.where(strict, hold + dt, 0.)
            new_recover = (hold >= .5 - 1e-6) & ~recovered
            first_recover[new_recover] = (i + 1) * dt
            recovered |= new_recover
            new_failure = state.eval_common_failure & active & (first_failure < 0)
            first_failure[new_failure] = (i + 1) * dt
            numeric |= state.numerical_failure & active
            if i >= steps - round(5. / dt):
                tail_count += healthy
                tail_height += torch.where(healthy, height, 0.)
                tail_error += torch.where(healthy, error, 0.)
                tail_speed += torch.where(healthy, speed, 0.)
                tail_strict += strict
                tail_base_contact += healthy & (base > 1.)
                tail_leg_contact += healthy & (leg > 1.) & (base <= 1.)
                tail_inverted += healthy & (gravity[:, 2] > 0.)
            if (i + 1) % 5 == 0:
                values = height[healthy]
                trace.append(dict(time_s=(i + 1) * dt, alive_n=int(healthy.sum()),
                                  mean_height_m=float(values.mean()) if values.numel() else None,
                                  strict_n=int(strict.sum()), recovered_n=int(recovered.sum())))
                qpos = raw.sim.data.qpos[0].cpu().numpy().copy()
                if not bool(healthy[0]):
                    qpos = frame_qpos[-1].copy()
                frame_qpos.append(qpos)
                frame_times.append((i + 1) * dt)
            active &= ~state.numerical_failure
            if i + 1 == steps or not bool(active.any()):
                break
            if bool(done.any()):
                raw.reset(env_ids=torch.nonzero(done).flatten())
                obs = env.get_observations()
        full_tail = tail_count == round(5. / dt)
        tail = full_tail & active
        def mean(total):
            return float((total[tail] / tail_count[tail]).mean()) if bool(tail.any()) else None
        sustained = tail & (tail_strict / tail_count.clamp_min(1) >= .95)
        summary = dict(num_envs=n, initial_state_sha256=initial_hash,
                       max_initial_restore_delta=restore_delta, steps_executed=i + 1,
                       strict_recovered=int(recovered.sum()), sustained_last5s=int(sustained.sum()),
                       common_physical_survived20s=int(((first_failure < 0) & active).sum()) if i + 1 == steps else 0,
                       numerical_failures=int(numeric.sum()), tail_env_count=int(tail.sum()),
                       mean_last5s_height_m=mean(tail_height), mean_last5s_abs_height_error_m=mean(tail_error),
                       mean_last5s_speed_m_s=mean(tail_speed), base_contact_fraction=mean(tail_base_contact),
                       leg_contact_without_base_fraction=mean(tail_leg_contact), inverted_fraction=mean(tail_inverted),
                       first_strict_recovery_s=[None if v < 0 else v for v in first_recover.cpu().tolist()],
                       first_common_physical_failure_s=[None if v < 0 else v for v in first_failure.cpu().tolist()],
                       trajectory=trace, representative_sample_index=0,
                       representative_numeric_failure=bool(numeric[0]))
        np.savez_compressed(out / f'{group}-{pose}-qpos.npz',
                            qpos=np.array(frame_qpos), time_s=np.array(frame_times),
                            initial_state_sha256=initial_hash)
        if not (out / 'scene.mjb').exists():
            mujoco.mj_saveModel(raw.sim.mj_model, str(out / 'scene.mjb'), None)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--height', type=float, default=.30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.seconds < 20 or args.num_envs < 1:
        parser.error('Need at least 20 seconds and one environment')
    available = {g: {int(p.stem.split('_')[-1]) for p in (args.campaign / g).glob('model_[0-9]*.pt')}
                 for g in GROUPS}
    common = set.intersection(*available.values())
    iteration = max(common) if args.iteration is None else args.iteration
    if iteration not in common:
        raise ValueError(f'Checkpoint {iteration} is not available in all four groups')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), iteration_index=iteration,
                  seed=args.seed, command=dict(vx=0., yaw=0., height=args.height),
                  seconds=args.seconds, nominal_physics=True, deterministic_actor=True,
                  noise_and_delay=False, learning=False, groups={},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  strict_definition='height +/-3cm, roll/pitch +/-15deg, both wheels >5N, base <=1N, continuous 0.5s',
                  physical_failure_definition='baseline contact>1N or tilt>40deg for20 steps; recorded, does not end recovery opportunity',
                  sustained_definition='at least95% strict standing during full final5s',
                  tail_scope='full final5s, finite survivors including physically failed samples')
    paired = {}
    initial_states = {}
    for group in GROUPS:
        path = args.campaign / group / f'model_{iteration}.pt'
        actor, metadata = load_checked_actor(path, group, args.device)
        results = dict(model=metadata, poses={})
        for pose in ('normal', 'ground'):
            summary = rollout(actor, pose, args, args.output, group, initial_states)
            digest = summary['initial_state_sha256']
            if pose in paired and paired[pose] != digest:
                raise ValueError(f'{group}/{pose}: initial states differ across groups')
            paired[pose] = digest
            results['poses'][pose] = summary
            print('EVALUATION_RESULT ' + json.dumps(dict(group=group, pose=pose,
                  strict_recovered=summary['strict_recovered'], sustained_last5s=summary['sustained_last5s'],
                  height_error_m=summary['mean_last5s_abs_height_error_m'],
                  numerical_failures=summary['numerical_failures'])), flush=True)
        report['groups'][group] = results
        (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
        del actor
        torch.cuda.empty_cache()
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['paired_initial_states_verified'] = True
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('EVALUATION_COMPLETE ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
