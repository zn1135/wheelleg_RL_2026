"""Paired, full-horizon load-bearing replay for the recovery factor campaign."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from ct_mjlab.experiments import recovery_factors as factors
from ct_mjlab.experiments import huananhu_v14_height as source
from ct_mjlab.experiments import huananhu_v14_recovery as recovery
from ct_mjlab.policy import actor_model
from ct_mjlab.wheel_balance_assessment import STANDARD, components
from ct_mjlab.reference_training import huananhu_mdp, huananhu_observations


def numerical_probe(env, *, case):
    (huananhu_mdp.failure if case == 'baseline' else recovery.failure)(env)
    return env.command_manager.get_term('standup').numerical_failure


def load_actor(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    tag = saved['infos']['experiment_profile_contract']
    if tag == source.INTERFACE:
        infos, case = source.validate_checkpoint(saved), 'baseline'
    elif tag == recovery.INTERFACE:
        infos, case = recovery.validate_checkpoint(saved), 'recovery_reference'
    else:
        case = saved['infos']['interface']['case']
        infos = factors.validate_checkpoint(saved, case)
    actor = actor_model().to(device).eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError('Nonfinite actor')
    return actor, infos, case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-ground', type=Path, required=True)
    p.add_argument('--initial-normal', type=Path)
    p.add_argument('--height', type=float, default=.30)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.num_envs != 64 or not .27 <= args.height <= .34:
        p.error('Use the canonical64 states and target height0.27-0.34m')
    actor, infos, case = load_actor(args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        case=case, completed_updates=infos['completed_updates'], learning=False,
        criteria=factors.PARAMETERS, command=dict(height=args.height, vx=0., yaw=0.),
        num_envs=64, seconds=20., noise_and_delay=False, nominal_physics=True,
        evaluation_standard=STANDARD, poses={})
    for pose in ('normal', 'ground'):
        torch.manual_seed(args.seed)
        cfg = factors.make_env_cfg(case, 64, args.seed, play=True)
        cfg.auto_reset = False
        cfg.commands['standup'].ground_start_probability = float(pose == 'ground')
        cfg.commands['standup'].zero_velocity_commands = True
        cfg.commands['standup'].height_range = (args.height, args.height)
        cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
        cfg.terminations['failure'] = TerminationTermCfg(func=numerical_probe, params={'case': case})
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
        try:
            raw = env.unwrapped
            state = raw.command_manager.get_term('standup')
            if case != 'baseline':
                state.curriculum.load(infos['recovery_curriculum'])
            path = args.initial_ground if pose == 'ground' else args.initial_normal
            if path:
                with np.load(path) as saved:
                    initial = {k: torch.tensor(saved[k], device=raw.device) for k in ('qpos', 'qvel', 'origins')}
            else:
                initial = {k: v.clone() for k, v in dict(qpos=raw.sim.data.qpos,
                    qvel=raw.sim.data.qvel, origins=raw.scene.env_origins).items()}
            if not torch.equal(raw.scene.env_origins, initial['origins']):
                raise ValueError('Canonical scene origins differ')
            ids = torch.arange(64, device=raw.device)
            raw.scene.reset(ids)
            raw.sim.data.qpos.copy_(initial['qpos'])
            raw.sim.data.qvel.copy_(initial['qvel'])
            state.sampled[:, :2] = state.command[:, :2] = 0.
            state.sampled[:, 2] = state.command[:, 2] = args.height
            state.ground_command.zero_()
            state.ground_start.fill_(pose == 'ground')
            if state.control_mode is not None:
                state.control_mode.reset(ids, state.ground_start)
            raw.sim.forward()
            raw.sim.sense()
            huananhu_observations.reset_observations(raw, ids)
            raw.observation_manager.reset(ids)
            raw.obs_buf = raw.observation_manager.compute(update_history=True)
            obs = env.get_observations()
            assert torch.equal(raw.sim.data.qpos, initial['qpos']) and torch.equal(raw.sim.data.qvel, initial['qvel'])
            digest = hashlib.sha256(initial['qpos'].cpu().numpy().tobytes()
                                    + initial['qvel'].cpu().numpy().tobytes()).hexdigest()
            np.savez_compressed(args.output / f'initial-{pose}.npz',
                                **{k: v.cpu().numpy() for k, v in initial.items()})
            if not (args.output / 'scene.mjb').exists():
                mujoco.mj_saveModel(raw.sim.mj_model, str(args.output / 'scene.mjb'), None)
            active = torch.ones(64, dtype=torch.bool, device=raw.device)
            hold = torch.zeros(64, 3, device=raw.device)
            achieved = torch.zeros_like(hold, dtype=torch.bool)
            tail = torch.zeros_like(hold)
            height = torch.zeros(64, device=raw.device)
            wheels = torch.zeros(64, 2, device=raw.device)
            other = torch.zeros_like(height)
            qpos = [raw.sim.data.qpos[0].cpu().numpy().copy()]
            trace = []
            component_tail = {}
            for step in range(1000):
                with torch.no_grad():
                    obs, _, _, _ = env.step(torch.where(active[:, None], actor(obs), 0.))
                active &= ~state.numerical_failure
                values = factors.goals(raw)
                robot = raw.scene['robot'].data
                checks = components(gravity=robot.projected_gravity_b,
                    wheel_support=factors.wheel_force(raw),
                    wheel_contact=raw.scene['wheel_ground_probe'].data.found > 0,
                    nonwheel_support=factors.other_force(raw), robot_weight=factors.weight(raw),
                    base_contact=raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
                    linear_velocity=robot.root_link_lin_vel_b, angular_velocity=robot.root_link_ang_vel_b,
                    height=robot.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2],
                    target_height=state.command[:, 2], valid=active, parameters=factors.PARAMETERS)
                if not torch.equal(checks['wheel_balance'], values['wheel_balance'] & active):
                    raise RuntimeError('Replay diagnostics differ from the frozen physical predicate')
                flags = torch.stack([values[s] for s in factors.STAGES], -1) & active[:, None]
                hold = torch.where(flags, hold + raw.step_dt, 0.)
                achieved |= hold >= factors.PARAMETERS['success_hold_s']
                h = raw.scene['robot'].data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                if step >= 750:
                    for name, check in checks.items():
                        if name not in component_tail:
                            component_tail[name] = torch.zeros(64, device=raw.device)
                        component_tail[name] += check & active
                    tail += flags
                    height += h
                    wheels += factors.wheel_force(raw)
                    other += factors.other_force(raw)
                if (step + 1) % 5 == 0:
                    qpos.append(raw.sim.data.qpos[0].cpu().numpy().copy())
                    trace.append(dict(time_s=(step + 1) * raw.step_dt,
                        balanced=int(flags[:, 1].sum()), mean_height=float(h[active].mean()) if active.any() else None,
                        sample0_wheel_n=factors.wheel_force(raw)[0].cpu().tolist(),
                        sample0_other_n=float(factors.other_force(raw)[0])))
            sustained = (tail >= 237.5) & active[:, None]
            result = dict(initial_state_sha256=digest, initial_states_restored=True,
                successes={s: int((achieved[:, i] & active).sum()) for i, s in enumerate(factors.STAGES)},
                sustained_last5s={s: int(sustained[:, i].sum()) for i, s in enumerate(factors.STAGES)},
                balanced_samples=torch.nonzero(sustained[:, 1]).flatten().cpu().tolist(),
                numerical_failures=int((~active).sum()),
                mean_last5s_height_m=float((height[active] / 250).mean()) if active.any() else None,
                mean_last5s_abs_height_error_m=float((height[active] / 250 - args.height).abs().mean()) if active.any() else None,
                mean_last5s_wheel_n=(wheels[active] / 250).mean(0).cpu().tolist() if active.any() else None,
                mean_last5s_nonwheel_n=float((other[active] / 250).mean()) if active.any() else None,
                sample0=dict(height_m=float(height[0] / 250), wheel_n=(wheels[0] / 250).cpu().tolist(),
                             nonwheel_n=float(other[0] / 250)), trajectory=trace)
            result['balance_diagnostics'] = dict(
                temporal_rule='simultaneous predicates on at least 95% of the final 5 seconds',
                component_last5s_fraction={k: (v / 250).cpu().tolist() for k, v in component_tail.items()},
                wheel_load_bearing_samples=torch.nonzero(
                    (component_tail['wheel_load_bearing'] >= 237.5) & active).flatten().cpu().tolist(),
                samples=[dict(sample=i, balanced=bool(sustained[i, 1]),
                    failed_checks=[k for k, v in component_tail.items()
                        if k not in ('target_height', 'height_tracking', 'wheel_load_bearing', 'wheel_balance')
                        and float(v[i]) < 237.5],
                    joint_balance_failed=not bool(sustained[i, 1]),
                    height_m=float(height[i] / 250), wheel_n=(wheels[i] / 250).cpu().tolist(),
                    nonwheel_n=float(other[i] / 250)) for i in range(64)])
            report['poses'][pose] = result
            np.savez_compressed(args.output / f'{case}-{pose}-qpos.npz',
                                qpos=np.array(qpos), time_s=np.arange(len(qpos)) * .1)
            (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
            print('FACTOR_EVALUATION ' + json.dumps({pose: {k: v for k, v in result.items() if k != 'trajectory'}}), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
