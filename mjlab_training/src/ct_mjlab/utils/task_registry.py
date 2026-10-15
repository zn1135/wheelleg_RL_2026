"""Task factories in the same role as Isaac's utils/task_registry.py.

Factories return fresh configs. Registering a task does not import a simulator,
allocate a GPU environment, create a log directory or start learning.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskDefinition:
    run: Callable
    get_cfgs: Callable
    make_env: Callable
    runner_class: Callable
    default_profile: str


class TaskRegistry:
    def __init__(self):
        self._tasks = {}

    def register(self, name, *, run, get_cfgs, make_env, runner_class, default_profile):
        if name in self._tasks:
            raise ValueError(f'Task already registered: {name}')
        self._tasks[name] = TaskDefinition(run, get_cfgs, make_env, runner_class, default_profile)

    def names(self):
        return tuple(self._tasks)

    def _task(self, name):
        try:
            return self._tasks[name]
        except KeyError:
            raise ValueError(f'Task not registered: {name}') from None

    def get_cfgs(self, name, *, profile=None, **kwargs):
        """Return independent environment/PPO configs for a fresh profile."""
        task = self._task(name)
        return task.get_cfgs(profile=task.default_profile if profile is None else profile, **kwargs)

    def make_env(self, name, *, profile=None, **kwargs):
        """Return (environment, actual environment config), without learning."""
        task = self._task(name)
        profile = task.default_profile if profile is None else profile
        env = task.make_env(profile=profile, **kwargs)
        # Wrapper-only metadata: never add this to serialized environment configs
        # or historical checkpoint contracts.
        env._ct_mjlab_task_binding = (name, profile)
        return env, env.unwrapped.cfg

    def make_alg_runner(self, env, name, *, profile=None, train_cfg=None, log_dir=None):
        """Build a fresh runner; checkpoint migrations belong to run()/CLI.

        Registered environments retain their task/profile; explicit mismatches
        are rejected. External environments require an explicit profile.
        No implicit log creation or learning. Pass the config returned by get_cfgs
        to customize PPO, and an already created log_dir when logging is wanted.
        """
        task = self._task(name)
        binding = getattr(env, '_ct_mjlab_task_binding', None)
        if binding is not None:
            env_name, env_profile = binding
            if name != env_name:
                raise ValueError(f'Environment task mismatch: expected {env_name!r}, got {name!r}')
            if profile is not None and profile != env_profile:
                raise ValueError(f'Environment profile mismatch: expected {env_profile!r}, got {profile!r}')
            profile = env_profile
        elif profile is None:
            raise ValueError('An environment created outside the task registry requires an explicit profile')
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name, profile=profile, seed=env.unwrapped.cfg.seed)
        runner_type = task.runner_class(profile)
        runner = runner_type(env, train_cfg, log_dir, device=env.unwrapped.device)
        return runner, train_cfg

    def run(self, name, args, parser):
        """Execute the task's version-aware CLI workflow."""
        return self._task(name).run(args, parser)


task_registry = TaskRegistry()
