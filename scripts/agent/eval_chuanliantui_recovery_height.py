"""Measure righting, two-wheel balance and commanded height independently; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import huananhu_v14_height, huananhu_v14_stand, huananhu_v14_recovery
from ct_mjlab.experiments.recovery_goals import STAGES, PARAMETERS, predicates
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training import huananhu_mdp


def numerical_only(env):
    # Historical actors keep historical observations/modes. Only physical termination
    # is bypassed for the independent, full-horizon recovery assessment.
    huananhu_mdp.failure(env)
    return env.command_manager.get_term('standup').numerical_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--height', type=float, default=.30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.num_envs < 1 or not .27 <= args.height <= .34:
        parser.error('Positive population and target height within0.27-0.34m required')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    interface = saved['infos']['experiment_profile_contract']
    profiles = {p.INTERFACE: p for p in (huananhu_v14_height, huananhu_v14_stand, huananhu_v14_recovery)}
    profile = profiles[interface]
    infos = profile.validate_checkpoint(saved)
    actor = actor_model().to(args.device).eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError('Nonfinite actor weights')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        profile=interface, completed_updates=infos['completed_updates'], learning=False,
        num_envs_per_pose=args.num_envs, seconds=20, target_height_m=args.height, criteria=PARAMETERS,
        deterministic_actor=True, nominal_physics=True, noise_and_delay=False, assistance=False, poses={})
    for pose in ('normal', 'ground'):
        torch.manual_seed(args.seed)
        cfg = profile.make_env_cfg(args.num_envs, args.seed, play=True)
        cfg.auto_reset = False
        cfg.commands['standup'].ground_start_probability = float(pose == 'ground')
        cfg.commands['standup'].zero_velocity_commands = True
        cfg.commands['standup'].height_range = (args.height, args.height)
        cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
        if profile is not huananhu_v14_recovery:
            cfg.terminations['failure'] = TerminationTermCfg(func=numerical_only)
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
        try:
            raw = env.unwrapped
            command = raw.command_manager.get_term('standup')
            if profile is huananhu_v14_recovery:
                command.curriculum.load(infos['recovery_curriculum'])
            obs = env.get_observations()
            active = torch.ones(args.num_envs, device=args.device, dtype=torch.bool)
            achieved = torch.zeros(args.num_envs, 3, device=args.device, dtype=torch.bool)
            hold = torch.zeros(args.num_envs, 3, device=args.device)
            tail = torch.zeros_like(hold)
            for step in range(1000):
                with torch.inference_mode():
                    obs, _, _, _ = env.step(torch.where(active[:, None], actor(obs), 0.))
                active &= ~command.numerical_failure
                data = raw.scene['robot'].data
                values = predicates(gravity=data.projected_gravity_b,
                    wheel_support=(-raw.scene['wheel_contact'].data.force[..., 2]).clamp_min(0.),
                    base_contact=raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
                    linear_velocity=data.root_link_lin_vel_b, angular_velocity=data.root_link_ang_vel_b,
                    height=data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2],
                    target_height=command.command[:, 2], valid=active)
                flags = torch.stack([values[s] for s in STAGES], -1)
                hold = torch.where(flags, hold + raw.step_dt, 0.)
                achieved |= hold >= PARAMETERS['success_hold_s']
                if step >= 750:
                    tail += flags
            valid_achieved = achieved & active[:, None]
            sustained = (tail >= .95 * 250) & active[:, None]
            report['poses'][pose] = dict(
                successes={s: int(valid_achieved[:, i].sum()) for i, s in enumerate(STAGES)},
                sustained_last5s={s: int(sustained[:, i].sum()) for i, s in enumerate(STAGES)},
                numerical_failures=int((~active).sum()))
            print('RECOVERY_GOALS ' + json.dumps({pose: report['poses'][pose]}), flush=True)
            (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
        finally:
            env.close()


if __name__ == '__main__':
    main()
