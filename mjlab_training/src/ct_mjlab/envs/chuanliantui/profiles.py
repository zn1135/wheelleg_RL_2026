"""CLI profile routing; recipe implementations and checkpoint contracts stay frozen.

This module chooses an entry point only. The selected experiment remains responsible
for validating the complete checkpoint before constructing an environment or run.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Profile:
    name: str
    module: str
    interface: str
    playback_error: str
    migration_error: str
    migration_flag: str | None = None
    rejected_flags: tuple[str, ...] = ()
    checkpoint_commands: tuple[str, ...] = ('play',)


_HISTORICAL_FLAGS = (
    'resume_contact_force_fix', 'resume_pose_bias_fix', 'resume_orientation_weight',
    'resume_wide_height_gate', 'resume_disable_knee_limit',
)

# Order preserves historical migration validation precedence.
V14_PROFILES = (
    Profile(
        'huananhu-v14-wheel-support', 'huananhu_v14_wheel_support',
        'chuanliantui-mjlab-v14-wheel-support-v1',
        'Wheel support profile currently supports native mjlab play',
        'Wheel support profile accepts only --resume-v14-wheel-support or its own checkpoint',
        'resume_v14_wheel_support',
        ('resume_v14_recovery_fix', 'resume_v14_height_fix', 'resume_v14_stand_fix') + _HISTORICAL_FLAGS,
        ('play', 'eval', 'export'),
    ),
    Profile(
        'huananhu-v14-recovery', 'huananhu_v14_recovery',
        'chuanliantui-mjlab-v14-recovery-height-v1',
        'Recovery height profile currently supports native mjlab play; use its independent evaluator for reports',
        'Recovery profile accepts only --resume-v14-recovery-fix or its own checkpoint',
        'resume_v14_recovery_fix',
        ('resume_v14_height_fix', 'resume_v14_stand_fix') + _HISTORICAL_FLAGS,
        ('play', 'eval', 'export'),
    ),
    Profile(
        'huananhu-v14-stand', 'huananhu_v14_stand',
        'chuanliantui-mjlab-scut-v14-stand-v1',
        'V14 standing playback requires mjlab',
        'V14 standing accepts only --resume-v14-stand-fix or its own checkpoint',
        'resume_v14_stand_fix', ('resume_v14_height_fix',) + _HISTORICAL_FLAGS,
    ),
    Profile(
        'huananhu-v14-height', 'huananhu_v14_height',
        'chuanliantui-mjlab-scut-v14-height-v1',
        'V14 height playback requires mjlab',
        'V14 height profile accepts only --resume-v14-height-fix or its own checkpoint',
        'resume_v14_height_fix', _HISTORICAL_FLAGS,
    ),
    Profile(
        'huananhu-v14', 'huananhu_v14',
        'chuanliantui-mjlab-scut-v14-flat-v1',
        'Original V14 playback requires mjlab; CPU evaluation is not implemented',
        'Original V14 accepts only its own complete checkpoint, without migration flags',
        rejected_flags=_HISTORICAL_FLAGS,
    ),
)

TRAINING_PROFILES = tuple(profile.name for profile in reversed(V14_PROFILES)) + (
    'getup-r15', 'getup-r15-orientation', 'recover-r14',
)
_HALF_ASSISTANCE_INTERFACES = (
    'chuanliantui-mjlab-getup-curriculum-r15-half-assistance-v1',
    'chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1',
)
_R15_INTERFACES = (
    'chuanliantui-mjlab-getup-curriculum-r15',
    'chuanliantui-mjlab-getup-curriculum-r15-assisted-v1',
    'chuanliantui-mjlab-getup-curriculum-r15-assisted-v2',
    'chuanliantui-mjlab-getup-curriculum-r15-assisted-v3',
)


def _experiment(module):
    return f'ct_mjlab.experiments.{module}'


def resolve_profile(args, parser):
    """Return an experiment module, or None for recover/legacy commands.

    Cache only interface metadata for this invocation; never retain checkpoint
    tensors globally or reuse stale metadata across separate CLI calls.
    """
    interfaces = {}

    def interface(path):
        if path not in interfaces:
            infos = torch.load(path, map_location='cpu', weights_only=False).get('infos') or {}
            interfaces[path] = (infos.get('interface') or {}).get('interface')
        return interfaces[path]

    training = args.command in ('train', 'check')
    checkpoint = getattr(args, 'checkpoint', None)
    for profile in V14_PROFILES:
        if profile.migration_flag and getattr(args, profile.migration_flag, False):
            if args.training_profile != profile.name or not args.resume:
                flag = profile.migration_flag.replace('_', '-')
                parser.error(f'--{flag} requires {profile.name} and --resume')
        selected = training and args.training_profile == profile.name
        if args.command in profile.checkpoint_commands and checkpoint:
            selected = interface(checkpoint) == profile.interface
            if selected and (args.command != 'play' or args.backend != 'mjlab' or args.report):
                parser.error(profile.playback_error)
        if selected:
            if any(getattr(args, flag, False) for flag in profile.rejected_flags):
                parser.error(profile.migration_error)
            return _experiment(profile.module)

    # Fresh r15 runs use half assistance; resumed runs retain their own schedule.
    half_assistance = False
    if training and args.training_profile in ('getup-r15', 'getup-r15-orientation'):
        resume = getattr(args, 'resume', None)
        half_assistance = not resume or interface(resume) in _HALF_ASSISTANCE_INTERFACES
    if args.command == 'play' and checkpoint:
        half_assistance = interface(checkpoint) in _HALF_ASSISTANCE_INTERFACES
        if half_assistance and (args.backend != 'mjlab' or args.report):
            parser.error('Half-assistance playback requires mjlab; CPU evaluation is not implemented')
    if half_assistance:
        return _experiment('getup_half_assistance')

    if getattr(args, 'resume_orientation_weight', False) and (
            args.training_profile != 'getup-r15-orientation' or not args.resume):
        parser.error('--resume-orientation-weight requires getup-r15-orientation and --resume')
    if training and args.training_profile == 'getup-r15-orientation':
        if any(getattr(args, flag, False) for flag in ('resume_contact_force_fix',
                'resume_pose_bias_fix', 'resume_wide_height_gate', 'resume_disable_knee_limit')):
            parser.error('Orientation profile accepts only its own resume or explicit r15-v3 migration')
        return _experiment('getup_orientation')
    for flag in ('resume_contact_force_fix', 'resume_pose_bias_fix'):
        if getattr(args, flag, False) and (args.training_profile != 'getup-r15' or not args.resume):
            parser.error(f'--{flag.replace("_", "-")} requires getup-r15 and --resume')
    if training and args.training_profile == 'getup-r15':
        if getattr(args, 'resume_wide_height_gate', False) or getattr(args, 'resume_disable_knee_limit', False):
            parser.error('r15 requires its own checkpoint; historical migration flags cannot be used')
        return _experiment('getup_curriculum')
    if args.command == 'play' and checkpoint:
        if interface(checkpoint) == 'chuanliantui-mjlab-getup-curriculum-r15-orientation-weight-v1':
            if args.backend != 'mjlab' or args.report:
                parser.error('Orientation playback uses --backend mjlab; CPU evaluation is not implemented')
            return _experiment('getup_orientation')
        if interface(checkpoint) in _R15_INTERFACES:
            if args.backend != 'mjlab' or args.report:
                parser.error('r15 playback currently uses --backend mjlab; CPU evaluation is not implemented')
            return _experiment('getup_curriculum')
    return None
