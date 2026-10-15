#!/usr/bin/env python3
"""Check mjlab entrypoints, routing and config parity without training.

Use the mjlab Python 3.12 environment. Optional --baseline-cli compares routing
with the pre-refactor monolithic cli.py. --physics exercises the public registry
with finite zero-action steps and runner construction; neither option learns.
"""
import argparse
from contextlib import ExitStack, redirect_stderr
from importlib import import_module
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import chuanliantui
from ct_mjlab.envs.chuanliantui.chuanliantui_config import RECIPES
from ct_mjlab.envs.chuanliantui.profiles import TRAINING_PROFILES, resolve_profile
from ct_mjlab.reference_training.entry import serializable
from ct_mjlab.utils.helpers import build_parser
from ct_mjlab.utils.task_registry import TaskRegistry


PREFIX = 'ct_mjlab.experiments.'
FRESH = {
    'huananhu-v14': 'huananhu_v14',
    'huananhu-v14-height': 'huananhu_v14_height',
    'huananhu-v14-stand': 'huananhu_v14_stand',
    'huananhu-v14-recovery': 'huananhu_v14_recovery',
    'huananhu-v14-wheel-support': 'huananhu_v14_wheel_support',
    'getup-r15': 'getup_half_assistance',
    'getup-r15-orientation': 'getup_half_assistance',
    'recover-r14': None,
}
HISTORICAL = {
    'chuanliantui-mjlab-scut-v14-flat-v1': 'huananhu_v14',
    'chuanliantui-mjlab-scut-v14-height-v1': 'huananhu_v14_height',
    'chuanliantui-mjlab-scut-v14-stand-v1': 'huananhu_v14_stand',
    'chuanliantui-mjlab-v14-recovery-height-v1': 'huananhu_v14_recovery',
    'chuanliantui-mjlab-v14-wheel-support-v1': 'huananhu_v14_wheel_support',
    'chuanliantui-mjlab-getup-curriculum-r15-half-assistance-v1': 'getup_half_assistance',
    'chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1': 'getup_half_assistance',
    'chuanliantui-mjlab-getup-curriculum-r15-orientation-weight-v1': 'getup_orientation',
    **{f'chuanliantui-mjlab-getup-curriculum-r15{suffix}': 'getup_curriculum'
       for suffix in ('', '-assisted-v1', '-assisted-v2', '-assisted-v3')},
    'chuanliantui-mjlab-huananhu-v14-flat-standup-r5': None,
    **{f'chuanliantui-mjlab-wheelleg-recover-r{version}': None for version in range(6, 15)},
}
FLAGS = (
    '--resume-v14-wheel-support', '--resume-v14-recovery-fix', '--resume-v14-stand-fix',
    '--resume-v14-height-fix', '--resume-contact-force-fix', '--resume-pose-bias-fix',
    '--resume-orientation-weight', '--resume-wide-height-gate', '--resume-disable-knee-limit',
)


def outcome(call):
    errors = io.StringIO()
    try:
        with redirect_stderr(errors):
            return ('route', call())
    except SystemExit as exc:
        return ('error', exc.code, errors.getvalue().rsplit(': error: ', 1)[-1].strip())


def route(argv, interface=None):
    parser = build_parser()
    with patch('torch.load', return_value={'infos': {'interface': {'interface': interface}}}) as load:
        result = outcome(lambda: resolve_profile(parser.parse_args(argv), parser))
        assert load.call_count <= 1, (argv, load.call_count)
    return result


def check_routes():
    for profile, module in FRESH.items():
        for command in ('train', 'check'):
            assert route([command, '--training-profile', profile]) == (
                'route', PREFIX + module if module else None)
    for interface, module in HISTORICAL.items():
        assert route(['play', '--checkpoint', 'fixture.pt'], interface) == (
            'route', PREFIX + module if module else None)
        if module:
            assert route(['play', '--checkpoint', 'fixture.pt', '--backend', 'cpu'], interface)[0] == 'error'
    # Metadata is local to a call: reusing the same filename must not retain its old route.
    assert route(['play', '--checkpoint', 'fixture.pt'], None) == ('route', None)
    for flag in FLAGS[:4]:
        assert route(['train', flag])[0] == 'error'
    for command in ('train', 'play', 'eval', 'export', 'check'):
        entry = import_module(f'ct_mjlab.scripts.{command}')
        args = ['--output', 'fixture.onnx', '--checkpoint', 'fixture.pt'] if command == 'export' else []
        with patch.object(task_registry, 'run') as run:
            entry.main(args)
            name, parsed, _ = run.call_args.args
            assert name == 'chuanliantui' and parsed.command == command
    assert task_registry.names() == ('chuanliantui',)
    assert tuple(RECIPES) == TRAINING_PROFILES
    return dict(fresh_profiles=len(FRESH), historical_interfaces=len(HISTORICAL), entrypoints=5)


def check_configs():
    results = []
    for profile, module_name in FRESH.items():
        cfg, ppo = task_registry.get_cfgs('chuanliantui', profile=profile, num_envs=2, seed=7)
        module = import_module(PREFIX + module_name) if module_name else import_module(
            'ct_mjlab.reference_training.entry')
        if profile.startswith('getup-'):
            expected = module.make_env_cfg(2, 7, strong_orientation=profile.endswith('orientation'))
            expected_ppo = module.base.runner_cfg(7)
            label = 'getup_r15_half_assistance' + ('_orientation' if profile.endswith('orientation') else '')
            expected_ppo.update(run_name=label, experiment_name='chuanliantui_' + label)
        elif profile == 'recover-r14':
            expected = module.make_huananhu_env_cfg()
            expected.scene.num_envs, expected.seed = 2, 7
            expected_ppo = module.runner_cfg(7)
        else:
            expected, expected_ppo = module.make_env_cfg(2, 7), module.runner_cfg(7)
        assert serializable(cfg) == serializable(expected), profile
        assert ppo == expected_ppo, profile
        assert isinstance(chuanliantui.runner_class(profile), type), profile
        # Config edits in one caller cannot leak into a later environment/run.
        cfg.scene.num_envs = 99
        ppo['actor']['hidden_dims'] = (1,)
        again, again_ppo = task_registry.get_cfgs('chuanliantui', profile=profile, num_envs=2, seed=7)
        assert serializable(again) == serializable(expected) and again_ppo == expected_ppo
        results.append(profile)
    return results


def check_runner_binding():
    """Reject task/profile mistakes before config factories or runners execute."""
    registry = TaskRegistry()
    cfg = SimpleNamespace(seed=7)
    external_env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg, device='cpu'))
    env_factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(unwrapped=external_env.unwrapped))
    cfg_factory = Mock(return_value=(cfg, {'seed': 7}))
    runner_type = Mock()
    runner_factory = Mock(return_value=runner_type)
    for name, default in (('robot', 'default-recipe'), ('other', 'other-recipe')):
        registry.register(name, run=Mock(), get_cfgs=cfg_factory, make_env=env_factory,
                          runner_class=runner_factory, default_profile=default)

    for name, requested, expected in (
            ('robot', None, 'default-recipe'), ('other', None, 'other-recipe'),
            ('robot', 'non-default', 'non-default')):
        env, actual_cfg = registry.make_env(name, profile=requested)
        assert actual_cfg is cfg
        env_factory.assert_called_with(profile=expected)
        for explicit in (None, expected):
            registry.make_alg_runner(env, name, profile=explicit)
            cfg_factory.assert_called_with(profile=expected, seed=7)
            runner_factory.assert_called_with(expected)
            runner_type.assert_called_with(env, {'seed': 7}, None, device='cpu')

    def rejected(call, message):
        before = (cfg_factory.call_count, runner_factory.call_count, runner_type.call_count)
        try:
            call()
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f'Expected rejection: {message}')
        assert before == (cfg_factory.call_count, runner_factory.call_count, runner_type.call_count)

    # Last environment belongs to robot/non-default. Supplying a custom PPO
    # config must not bypass the task/profile checks.
    for train_cfg in (None, {'seed': 9}):
        rejected(lambda: registry.make_alg_runner(env, 'robot', profile='default-recipe',
                 train_cfg=train_cfg), 'Environment profile mismatch')
        rejected(lambda: registry.make_alg_runner(env, 'other', profile='non-default',
                 train_cfg=train_cfg), 'Environment task mismatch')
        rejected(lambda: registry.make_alg_runner(external_env, 'robot', train_cfg=train_cfg),
                 'requires an explicit profile')
    registry.make_alg_runner(external_env, 'robot', profile='external-recipe')
    runner_factory.assert_called_with('external-recipe')
    custom = {'seed': 9}
    registry.make_alg_runner(env, 'robot', train_cfg=custom)
    runner_factory.assert_called_with('non-default')
    runner_type.assert_called_with(env, custom, None, device='cpu')
    registry.get_cfgs('other')
    cfg_factory.assert_called_with(profile='other-recipe')
    return dict(default_and_nondefault_inherited=True, explicit_match_accepted=True,
                task_and_profile_mismatch_rejected_before_construction=True,
                external_profile_required=True, explicit_external_profile_accepted=True,
                custom_ppo_cannot_bypass_binding=True)


def check_baseline(path):
    """Execute only old argument parsing/routing; replace all experiment run calls."""
    source = path.read_text()
    marker = "    if args.command=='train' and args.resume_disable_knee_limit"
    if marker not in source:
        raise ValueError('--baseline-cli must be the pre-refactor monolithic cli.py')
    namespace = {'__name__': 'ct_mjlab._structure_baseline', '__package__': 'ct_mjlab'}
    exec(compile(source[:source.index(marker)], str(path), 'exec'), namespace)
    parser_namespace = dict(namespace)
    parser_source = source[:source.index('    support_migration =')] + '    return args\n'
    exec(compile(parser_source, str(path), 'exec'), parser_namespace)
    count = 0
    selected = []

    def capture(module):
        return lambda args: selected.append(PREFIX + module)

    def old():
        selected.clear()
        namespace['main']()
        return selected[0] if selected else None

    def cases():
        for profile in FRESH:
            yield ['check', '--training-profile', profile], None
            for resume in (None, *HISTORICAL):
                for flags in ((), *((flag,) for flag in FLAGS)):
                    argv = ['train', '--training-profile', profile, *flags]
                    if resume:
                        argv += ['--resume', 'fixture.pt']
                    yield argv, resume
            for flags in itertools.combinations(FLAGS, 2):
                yield ['train', '--training-profile', profile, '--resume', 'fixture.pt', *flags], None
        for interface in (None, *HISTORICAL):
            for command in ('play', 'eval', 'export'):
                options = ([], ['--backend', 'cpu'], ['--report', 'fixture.json']) if command == 'play' else ([],)
                for extra in options:
                    argv = [command, '--checkpoint', 'fixture.pt', *extra]
                    if command == 'export':
                        argv += ['--output', 'fixture.onnx']
                    yield argv, interface

    with ExitStack() as stack:
        for module in set(HISTORICAL.values()) - {None}:
            stack.enter_context(patch.object(import_module(PREFIX + module), 'run', capture(module)))
        for argv, interface in cases():
            saved = {'infos': {'interface': {'interface': interface}}}
            with patch.object(sys, 'argv', ['ct-mjlab', *argv]), patch('torch.load', return_value=saved):
                expected = outcome(old)
                old_args = parser_namespace['main']()
            assert vars(old_args) == vars(build_parser().parse_args(argv)), argv
            actual = route(argv, interface)
            assert actual == expected, (argv, interface, expected, actual)
            count += 1
    return count


def check_provenance():
    from ct_mjlab.utils import provenance
    # Remote-release fixture verifies recursive inclusion of the new layout,
    # excluded caches and unchanged version metadata, without writing root logs/.
    with tempfile.TemporaryDirectory(prefix='mjlab-structure-') as temp:
        root = Path(temp)
        for name in ('mjlab_training/src/ct_mjlab/scripts/train.py', 'scripts/agent/check.py',
                     'mjlab_training/.venv/skip.py', 'mjlab_training/src/__pycache__/skip.pyc'):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('fixture')
        (root / 'source-version.txt').write_text('提交：' + 'a' * 40 + '\n快照：fixture\n')
        output = root / 'new-run'
        output.mkdir()
        with patch.object(provenance, 'ROOT', root):
            provenance.snapshot_source(output)
        record = json.loads((output / 'source-state.json').read_text())
        assert record['source_kind'] == 'remote-release' and record['head'] == 'a' * 40
        assert set(record['changed_files_sha256']) == {
            'mjlab_training/src/ct_mjlab/scripts/train.py', 'scripts/agent/check.py'}
        assert not (output / 'source.diff').exists()
    return True


def check_physics(device, steps):
    results = []
    for profile in ('huananhu-v14', 'recover-r14', 'getup-r15'):
        env, cfg = task_registry.make_env('chuanliantui', profile=profile, num_envs=2,
                                          seed=42, device=device, play=True)
        try:
            expected, _ = task_registry.get_cfgs('chuanliantui', profile=profile, num_envs=2, play=True)
            # mjlab Scene._add_terrain copies these two scene fields into terrain
            # during construction; compare the effective config after that step.
            expected.scene.terrain.num_envs = expected.scene.num_envs
            expected.scene.terrain.env_spacing = expected.scene.env_spacing
            assert serializable(cfg) == serializable(expected), profile
            runner, _ = task_registry.make_alg_runner(env, 'chuanliantui')
            # recover-r14 creates its runner class inside a factory on each call;
            # compare its defining module/name rather than distinct class objects.
            expected_type = chuanliantui.runner_class(profile)
            assert (type(runner).__module__, type(runner).__qualname__) == (
                expected_type.__module__, expected_type.__qualname__), profile
            with torch.inference_mode():
                for _ in range(steps):
                    obs, rewards, _, _ = env.step(torch.zeros(2, 6, device=device))
                    assert obs['actor'].shape == (2, 35) and obs['critic'].shape == (2, 78)
                    assert all(torch.isfinite(value).all() for value in obs.values())
                    assert torch.isfinite(rewards).all()
            del runner
            results.append(profile)
        finally:
            env.close()
    return dict(profiles=results, steps=steps, device=device, learning=False,
                runner_profile_inherited=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-cli', type=Path)
    parser.add_argument('--physics', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    result = dict(scope='structure/config/routing/finite tensors; not behavior acceptance',
                  routing=check_routes(), configs=check_configs(), provenance=check_provenance(),
                  runner_binding=check_runner_binding())
    if args.baseline_cli:
        result['baseline_routing_cases'] = check_baseline(args.baseline_cli)
    if args.physics:
        result['physics'] = check_physics(args.device, args.steps)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
