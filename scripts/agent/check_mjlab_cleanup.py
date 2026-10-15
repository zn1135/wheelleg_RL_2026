#!/usr/bin/env python3
"""Exercise recover command cleanup with substitute environments/runners/viewers.

By default no GPU environment, GUI, PPO learning or historical log writes. Real
command branches run with substitute resources and temporary output paths.
--real-check additionally repeats finite GPU checks in the same process.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import mujoco.viewer

from ct_mjlab import policy, registry, training_viewer
from ct_mjlab.envs.chuanliantui import commands
from ct_mjlab.utils.helpers import build_parser
from ct_mjlab.utils import provenance


def check_case(command, failure=None, render=None):
    events = []
    error = KeyboardInterrupt('injected interrupt') if failure == 'interrupt' else RuntimeError(
        f'injected {failure}')
    num_envs = 2 if render == 'grid' else 1
    obs = {'actor': torch.zeros(num_envs, 35), 'critic': torch.zeros(num_envs, 78)}
    env = Mock(num_envs=num_envs, obs=obs)
    env.unwrapped = SimpleNamespace(cfg={}, sim=SimpleNamespace(mj_model=object(), mj_data=object()))
    env.step.return_value = (obs, torch.zeros(num_envs), None, {})
    env.close.side_effect = lambda: events.append('env.close')
    viewer = Mock()
    viewer.is_running.return_value = render == 'grid'

    def close_viewer():
        events.append('viewer.close')
        if failure == 'viewer_close':
            raise error

    viewer.close.side_effect = close_viewer
    runner_init = Mock()
    runner_load = Mock()
    runner_learn = Mock()

    class Runner:
        def __init__(self, env, *args, **kwargs):
            self.env = env
            runner_init()

        def load(self, *args, **kwargs):
            runner_load()

        def learn(self, *args, **kwargs):
            runner_learn()

    argv = [command, '--device', 'cpu', '--num-envs', str(num_envs)]
    if command in ('train', 'check'):
        argv += ['--training-profile', 'recover-r14']
    if command == 'train':
        argv += ['--iterations', '1']
    elif command == 'check':
        argv += ['--steps', '1']
    else:
        argv += ['--seconds', '0.02']
    if render:
        argv += ['--render']
    if failure == 'load':
        argv += ['--resume', 'fixture.pt']
    if failure == 'actor':
        argv += ['--checkpoint', 'fixture.pt']
    if failure == 'label':
        argv += ['--run-name', 'invalid/name']
    if failure in ('step', 'interrupt'):
        env.step.side_effect = error
    if failure == 'shape':
        obs['actor'] = torch.zeros(num_envs, 1)
    for name, target in (('runner', runner_init), ('load', runner_load), ('learn', runner_learn),
                         ('open', viewer.open), ('draw', viewer.draw)):
        if failure == name:
            target.side_effect = error

    parser = build_parser()
    args = parser.parse_args(argv)
    viewer_acquired = bool(render) and command == 'play' and failure not in ('actor', 'launch')
    with tempfile.TemporaryDirectory(prefix='mjlab-cleanup-') as temp, ExitStack() as patches:
        patches.enter_context(patch.object(commands, 'ROOT', Path(temp)))
        patches.enter_context(patch.object(commands, 'manifest', return_value={}))
        patches.enter_context(patch.object(commands, 'runner_cfg', return_value={}))
        patches.enter_context(patch.object(provenance, 'snapshot_source',
                                          side_effect=error if failure == 'snapshot' else None))
        patches.enter_context(patch.object(registry, 'make_task',
                                          side_effect=error if failure == 'create' else None,
                                          return_value=env))
        for name in ('checkpoint_is_legacy', 'checkpoint_uses_constant_modes',
                     'checkpoint_policy_joint_order', 'checkpoint_knee_limit_weight',
                     'checkpoint_reset_config', 'checkpoint_height_gate', 'checkpoint_command_config'):
            patches.enter_context(patch.object(policy, name, return_value=None))
        patches.enter_context(patch.object(policy, 'runner_class', return_value=Runner))
        patches.enter_context(patch.object(policy, 'load_actor', side_effect=error))
        patches.enter_context(patch.object(training_viewer, 'NativeTrainingViewer', return_value=viewer))
        patches.enter_context(patch.object(mujoco.viewer, 'launch_passive',
                                          side_effect=error if failure == 'launch' else None,
                                          return_value=viewer))
        # Exercise the actual viewer runner wrapper without a window or learning.
        patches.enter_context(patch.object(training_viewer, 'learn_with_viewer',
            side_effect=lambda runner, learn, options, *a, **kw: learn(*a, **kw)))
        raised = None
        try:
            with redirect_stdout(io.StringIO()):
                commands.run(args, parser)
        except BaseException as exc:
            raised = exc
        if failure == 'shape':
            assert isinstance(raised, AssertionError), raised
        elif failure == 'label':
            assert isinstance(raised, ValueError) and str(raised) == 'run-name must be a label', raised
        elif failure:
            assert raised is error, (command, failure, render, raised)
        else:
            assert raised is None, raised
        expected = [] if failure == 'create' else ['env.close']
        if viewer_acquired:
            expected.insert(0, 'viewer.close')
        assert events == expected, (command, failure, render, events, expected)
    return dict(command=command, failure=failure, render=render, cleanup=events)


def check_viewer_wrapper_compatibility():
    error = KeyboardInterrupt('wrapper interrupt')

    class Runner:
        def __init__(self):
            self.env = Mock()

        def learn(self):
            pass

    for options, expected in (({}, 1), ({'close_env_on_error': False}, 0)):
        runner = training_viewer.viewer_runner_class(Runner, None, **options)()
        with patch.object(training_viewer, 'learn_with_viewer', side_effect=error):
            try:
                runner.learn()
            except KeyboardInterrupt as exc:
                assert exc is error
            else:
                raise AssertionError('Expected interrupt')
        assert runner.env.close.call_count == expected
    return True


def check_real_environment():
    from ct_mjlab.cli import main as cli_main
    create = registry.make_task
    calls = []
    acquired = []

    def tracked_create(*args, **kwargs):
        env = create(*args, **kwargs)
        env.close = Mock(wraps=env.close)
        env.unwrapped.recorder_manager.close = Mock(wraps=env.unwrapped.recorder_manager.close)
        acquired.append(env)
        return env

    for _ in range(2):
        with patch.object(registry, 'make_task', side_effect=tracked_create), redirect_stdout(io.StringIO()):
            cli_main(['check', '--training-profile', 'recover-r14', '--num-envs', '2', '--steps', '8'])
        assert len(acquired) == 1
        env = acquired.pop()
        assert env.close.call_count == 1
        assert env.unwrapped.recorder_manager.close.call_count == 1
        calls.append(env.close.call_count)
        del env
    return dict(runs=2, num_envs=2, steps_per_run=8, env_close_calls=calls,
                recorder_close_calls=[1, 1], learning=False, gui=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--real-check', action='store_true')
    args = parser.parse_args()
    cases = []
    for command, failures in (
            ('check', (None, 'step', 'interrupt', 'shape', 'create')),
            ('train', (None, 'snapshot', 'runner', 'load', 'learn', 'label', 'create')),
            ('play', (None, 'step', 'interrupt', 'actor', 'create'))):
        for failure in failures:
            cases.append(check_case(command, failure))
    for failure in (None, 'open', 'draw', 'step', 'interrupt', 'viewer_close'):
        cases.append(check_case('play', failure, render='grid'))
    for failure in (None, 'launch', 'step', 'interrupt', 'viewer_close'):
        cases.append(check_case('play', failure, render='single'))
    for failure in (None, 'runner', 'load', 'learn'):
        cases.append(check_case('train', failure, render='grid'))
    result = dict(scope='command lifecycle with substitutes; no training/GPU/GUI',
                  count=len(cases), cases=cases,
                  viewer_wrapper_legacy_cleanup=check_viewer_wrapper_compatibility())
    if args.real_check:
        result['real_check'] = check_real_environment()
        result['scope'] = 'substitute lifecycle cases plus finite GPU check close hooks; no training/GUI'
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(count=len(cases), passed=True, viewer_wrapper_legacy_cleanup=True,
                          real_check=result.get('real_check'))))


if __name__ == '__main__':
    main()
