"""Replay r15 or original V14 actors without assistance.

Shares the established 20s rollout and strict-standing metrics with the recovery
ablation evaluator. Runs nominal physics with separate normal/ground batches;
records physical failures without truncating the ground recovery opportunity.
No policy updates, source edits or training-log writes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import eval_chuanliantui_recovery_ablation as replay
from ct_mjlab.experiments import getup_curriculum as getup
from ct_mjlab.policy import actor_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--console', type=Path)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.num_envs < 1 or args.seconds != 20.:
        parser.error('Use a positive population and exactly20 seconds for these metrics')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    interface = (((saved.get('infos') or {}).get('interface')) or {}).get('interface')
    original_v14 = interface == 'chuanliantui-mjlab-scut-v14-flat-v1'
    height_v14 = interface == 'chuanliantui-mjlab-scut-v14-height-v1'
    stand_v14 = interface == 'chuanliantui-mjlab-scut-v14-stand-v1'
    if stand_v14:
        from ct_mjlab.experiments import huananhu_v14_stand as profile
        infos = profile.validate_checkpoint(saved)
        make_cfg = profile.make_env_cfg
    elif height_v14:
        from ct_mjlab.experiments import huananhu_v14_height as profile
        infos = profile.validate_checkpoint(saved)
        make_cfg = profile.make_env_cfg
    elif original_v14:
        from ct_mjlab.experiments import huananhu_v14 as profile
        infos = profile.validate_checkpoint(saved)
        make_cfg = profile.make_env_cfg
    elif interface == 'chuanliantui-mjlab-getup-curriculum-r15-orientation-weight-v1':
        from ct_mjlab.experiments import getup_orientation as profile
        infos = profile.validate_checkpoint(saved)
        make_cfg = profile.make_env_cfg
    elif interface in ('chuanliantui-mjlab-getup-curriculum-r15-half-assistance-v1',
                       'chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1'):
        from functools import partial
        from ct_mjlab.experiments import getup_half_assistance as profile
        strong = interface == profile.INTERFACES[1]
        infos = profile.validate_checkpoint(saved, strong)
        make_cfg = partial(profile.make_env_cfg, strong_orientation=strong)
    else:
        infos = getup.validate_checkpoint(saved)
        make_cfg = getup.make_env_cfg
    flat_v14 = original_v14 or height_v14 or stand_v14
    stage = None if flat_v14 else infos['getup_curriculum']['stage']
    args.height = .30 if flat_v14 else getup.HEIGHTS[stage]
    group = ('huananhu_v14_stand' if stand_v14 else 'huananhu_v14_height' if height_v14
             else 'huananhu_v14' if original_v14 else 'getup_r15')
    actor = actor_model().to(args.device).eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError('Nonfinite actor parameters')

    def make_play_cfg(_group, num_envs, seed, play):
        assert play
        cfg = (make_cfg(num_envs, seed, play=True) if flat_v14 else
               make_cfg(num_envs, seed, play=True, stage=stage))
        assert not getattr(cfg.actions['pd'], 'assistance_enabled', False)
        assert not getattr(cfg.commands['standup'], 'curriculum_enabled', False)
        return cfg

    # Only replace the environment factory. Preserve the common recorder,
    # reset pairing, contact probes and numerical-failure exclusion unchanged.
    replay.make_env_cfg = make_play_cfg
    args.output.mkdir(parents=True, exist_ok=False)
    events = EventAccumulator(str(args.checkpoint.parent), size_guidance={'scalars': 0})
    events.Reload()
    metrics = {}
    for tag in events.Tags()['scalars']:
        values = events.Scalars(tag)
        if values and any(word in tag.lower() for word in
                          ('getup_', 'pull_up_', 'numerical_failure', 'height_error', 'mean_reward')):
            metrics[tag] = dict(step=values[-1].step, value=values[-1].value)
    report = dict(
        started_at_server_utc=datetime.now(timezone.utc).isoformat(),
        profile=interface, stage='flat_v14' if flat_v14 else getup.STAGES[stage],
        curriculum=infos.get('getup_curriculum'),
        completed_updates=infos['completed_updates'], checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_rollout_sha256=hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        num_envs_per_pose=args.num_envs, seconds=args.seconds, seed=args.seed,
        command=dict(vx=0., yaw=0., height=args.height), assistance_enabled=False,
        nominal_physics=True, noise_and_delay=False, deterministic_actor=True, learning=False,
        strict_definition='height +/-3cm, roll/pitch +/-15deg, both wheels >5N, base <=1N, continuous0.5s',
        sustained_definition='at least95% strict standing during full final5s',
        physical_failure_definition='baseline contact>1N or tilt>40deg for20 steps; recorded, does not end recovery opportunity',
        training_metrics=metrics, poses={})
    paired = {}
    for pose in ('normal', 'ground'):
        result = replay.rollout(actor, pose, args, args.output, group, paired)
        report['poses'][pose] = result
        (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
        print('GETUP_EVALUATION ' + json.dumps(dict(pose=pose,
              strict_recovered=result['strict_recovered'], sustained_last5s=result['sustained_last5s'],
              height_error_m=result['mean_last5s_abs_height_error_m'],
              base_contact_fraction=result['base_contact_fraction'],
              numerical_failures=result['numerical_failures'])), flush=True)
    report['completed_at_server_utc'] = datetime.now(timezone.utc).isoformat()
    if args.console:
        log = args.console.read_text()
        report['training_console_tail'] = log.split('Learning iteration')[-1]
        exit_file = args.console.parent / '.exit-code'
        report['training_exit_code'] = int(exit_file.read_text()) if exit_file.exists() else None
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GETUP_EVALUATION_COMPLETE ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
