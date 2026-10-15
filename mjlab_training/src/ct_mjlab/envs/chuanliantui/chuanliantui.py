"""Task environment/runner factories and version-aware command execution."""
from importlib import import_module

from .chuanliantui_config import recipe
from .profiles import resolve_profile


def make_env(*, profile='huananhu-v14', num_envs=4096, seed=42, device='cuda:0', play=False):
    definition, module = recipe(profile)
    env_type = getattr(module, definition.env_class)
    options = dict(num_envs=num_envs, seed=seed, device=device)
    if profile == 'recover-r14':
        options.update(randomize=not play)
    else:
        options.update(play=play)
    if profile in ('getup-r15', 'getup-r15-orientation'):
        options['strong_orientation'] = profile == 'getup-r15-orientation'
    return env_type(**options)


def runner_class(profile='huananhu-v14'):
    definition, module = recipe(profile)
    runner = getattr(module, definition.runner_class)
    return runner() if profile == 'recover-r14' else runner


def run(args, parser):
    module = resolve_profile(args, parser)
    if module is not None:
        return import_module(module).run(args)
    from .commands import run as run_recover
    return run_recover(args, parser)
