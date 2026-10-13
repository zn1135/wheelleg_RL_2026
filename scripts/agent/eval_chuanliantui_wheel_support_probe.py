"""Observe the existing recovery actor with ground-only support probes; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import huananhu_v14_recovery as previous
from ct_mjlab.experiments import huananhu_v14_wheel_support as support
from ct_mjlab.experiments.wheel_support_goals import STAGES, PARAMETERS
from ct_mjlab.policy import actor_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    infos = previous.validate_checkpoint(saved)
    actor = actor_model().to('cuda:0').eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        completed_updates=infos['completed_updates'], actor_modes_rewards='unchanged recovery v1',
        learning=False, seed=42, num_envs_per_pose=64, seconds=20., target_height_m=.30,
        criteria=PARAMETERS, nominal_physics=True, poses={})
    for pose in ('normal', 'ground'):
        torch.manual_seed(42)
        cfg = previous.make_env_cfg(64, 42, play=True)
        cfg.scene.sensors = support.make_env_cfg(64, 42, play=True).scene.sensors
        cfg.auto_reset = False
        cfg.commands['standup'].ground_start_probability = float(pose == 'ground')
        cfg.commands['standup'].height_range = (.30, .30)
        cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'), clip_actions=None)
        try:
            raw = env.unwrapped
            command = raw.command_manager.get_term('standup')
            command.curriculum.load(infos['recovery_curriculum'])
            obs = env.get_observations()
            hold = torch.zeros(64, 3, device=env.device)
            success = torch.zeros_like(hold, dtype=torch.bool)
            tail = torch.zeros_like(hold)
            wheels = torch.zeros(64, 2, device=env.device)
            other = torch.zeros(64, device=env.device)
            height = torch.zeros_like(other)
            for step in range(1000):
                with torch.inference_mode():
                    obs, _, _, _ = env.step(actor(obs))
                flags = torch.stack([support.goals(raw)[s] for s in STAGES], -1)
                hold = torch.where(flags, hold + raw.step_dt, 0.)
                success |= hold >= PARAMETERS['success_hold_s']
                if command.numerical_failure.any():
                    raise RuntimeError('Probe numerical failure; do not include invalid survivors')
                if step >= 750:
                    tail += flags
                    wheels += support.wheel_support_force(raw)
                    other += support.nonwheel_support_force(raw)
                    height += raw.scene['robot'].data.root_link_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            weight = support.robot_weight(raw)
            result = dict(successes={s: int(success[:, i].sum()) for i, s in enumerate(STAGES)},
                sustained_last5s={s: int((tail[:, i] >= 237.5).sum()) for i, s in enumerate(STAGES)},
                numerical_failures=0, robot_weight_n=float(weight.mean()),
                mean_last5s_wheel_support_n=(wheels / 250).mean(0).tolist(),
                mean_last5s_wheel_weight_fraction=float((wheels.sum(-1) / 250 / weight).mean()),
                mean_last5s_nonwheel_support_n=float((other / 250).mean()),
                mean_last5s_height_m=float((height / 250).mean()),
                representative_sample0=dict(wheel_support_n=(wheels[0] / 250).tolist(),
                    nonwheel_support_n=float(other[0] / 250), height_m=float(height[0] / 250)))
            report['poses'][pose] = result
            (args.output / 'support-probe.json').write_text(json.dumps(report, indent=2) + '\n')
            print('SUPPORT_PROBE ' + json.dumps({pose: result}), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
