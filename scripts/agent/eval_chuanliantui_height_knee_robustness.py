"""Final-only256-env training-randomization zero-speed checks, without learning."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import height_knee as h
from ct_mjlab.policy import actor_model


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    case = saved['infos']['interface']['case']
    infos = h.validate_checkpoint(saved, case)
    assert infos['completed_updates'] == 3500
    nominal = json.loads(args.evaluation.read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert nominal['checkpoint_sha256'] == digest and nominal['exit_code'] == 0
    assert max(v['sustained_last5s']['height_tracking'] for v in nominal['poses'].values()) >= 32
    args.output.mkdir(parents=True, exist_ok=False)
    actor = actor_model().to(args.device).eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    report = dict(started=now(), learning=False, checkpoint_sha256=digest, completed_updates=3500,
        case=case, num_envs=256, seconds=20., seed=42, heights=[.27, .30, .34],
        domain_randomization='all original A train startup/reset/interval events, noise and delays enabled',
        commands='fixed height, zero vx/yaw including ground override; original .2s torque pause retained',
        fall_definition='ever original delayed contact/40deg failure or numerical failure, no reset/replacement',
        trigger='at least one canonical pose has >=32/64 final height-tracking successes at0.30m',
        records=[], exit_code=None)
    try:
        for command in report['heights']:
            torch.manual_seed(42)
            cfg = h.make_env_cfg(case, 256, 42, play=False)
            cfg.auto_reset = False
            cfg.commands['standup'].zero_velocity_commands = True
            cfg.commands['standup'].height_range = (command, command)
            cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
            wrapped = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
            env = wrapped.unwrapped
            try:
                state = env.command_manager.get_term('standup')
                state.training_iteration = 3500
                first_fall = torch.full((256,), -1, device=args.device, dtype=torch.long)
                numeric = torch.zeros(256, device=args.device, dtype=torch.bool)
                ground = state.ground_start.clone()
                height_sum = torch.zeros(256, device=args.device)
                tail_sum = torch.zeros_like(height_sum)
                within = torch.zeros_like(height_sum)
                tail_within = torch.zeros_like(height_sum)
                obs = wrapped.get_observations()
                with torch.inference_mode():
                    for step in range(1000):
                        obs, _, _, _ = wrapped.step(torch.where(~numeric[:, None], actor(obs), 0.))
                        assert (state.command[:, :2] == 0.).all()
                        failure = env.reset_terminated | state.numerical_failure
                        first_fall[(first_fall < 0) & failure] = step+1
                        numeric |= state.numerical_failure
                        height = env.scene['robot'].data.root_link_pos_w[:, 2]-env.scene.env_origins[:, 2]
                        finite_height = torch.nan_to_num(height, nan=0., posinf=0., neginf=0.)
                        height_sum += finite_height
                        good = ((height-command).abs() <= .03) & ~numeric
                        within += good
                        if step >= 750:
                            tail_sum += finite_height
                            tail_within += good
                falls = first_fall > 0
                row = dict(command_height_m=command, num_envs=256, initial_ground_count=int(ground.sum()),
                    falls_total=int(falls.sum()), falls_before5s=int((falls & (first_fall < 250)).sum()),
                    falls_at_or_after5s=int((first_fall >= 250).sum()),
                    numerical_failures=int(numeric.sum()), first_fall_time_s=(first_fall.float()*.02).cpu().tolist(),
                    mean_height_all20s_m=float(height_sum.mean()/1000), mean_height_last5s_m=float(tail_sum.mean()/250),
                    within3cm_time_fraction_all20s=float(within.mean()/1000),
                    within3cm_time_fraction_last5s=float(tail_within.mean()/250),
                    per_env_mean_height_last5s_m=(tail_sum/250).cpu().tolist(),
                    per_env_within3cm_fraction_last5s=(tail_within/250).cpu().tolist(),
                    fallen_samples=falls.nonzero().flatten().cpu().tolist())
                report['records'].append(row)
                (args.output/'robustness.json').write_text(json.dumps(report, indent=2)+'\n')
                print('HEIGHT_KNEE_ROBUSTNESS '+json.dumps(row), flush=True)
            finally:
                wrapped.close()
            torch.cuda.empty_cache()
        report['exit_code'] = 0
    except Exception:
        report['exit_code'] = 1
        raise
    finally:
        report['ended'] = now()
        (args.output/'robustness.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
