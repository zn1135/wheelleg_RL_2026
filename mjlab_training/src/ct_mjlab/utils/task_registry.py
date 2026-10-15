"""Task registry in the same role as wheel_legged_gym/utils/task_registry.py.

Each registered task provides fresh env/PPO config factories and a runner class.
Registering a task does not import a simulator, allocate a GPU environment,
create a log directory or start learning.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[4]
LOG_ROOT = ROOT / 'logs' / 'mjlab'


@dataclass(frozen=True)
class TaskDefinition:
    env_cfg: Callable  # (play: bool) -> ManagerBasedRlEnvCfg
    train_cfg: Callable  # () -> RslRlOnPolicyRunnerCfg
    runner_class: type


class TaskRegistry:
    def __init__(self):
        self._tasks = {}

    def register(self, name, *, env_cfg, train_cfg, runner_class):
        if name in self._tasks:
            raise ValueError(f'Task already registered: {name}')
        self._tasks[name] = TaskDefinition(env_cfg, train_cfg, runner_class)

    def names(self):
        return tuple(self._tasks)

    def _task(self, name):
        try:
            return self._tasks[name]
        except KeyError:
            raise ValueError(f'Task not registered: {name}; choose from {self.names()}') from None

    def get_cfgs(self, name, *, play=False):
        """Return independent (env_cfg, train_cfg) for the task."""
        task = self._task(name)
        return task.env_cfg(play=play), task.train_cfg()

    def make_env(self, name, *, env_cfg=None, train_cfg=None, play=False, num_envs=None,
                 seed=None, device='cuda:0', render_mode=None):
        """Create the wrapped vector environment; returns (env, env_cfg)."""
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends

        default_env_cfg, default_train_cfg = self.get_cfgs(name, play=play)
        env_cfg = default_env_cfg if env_cfg is None else env_cfg
        train_cfg = default_train_cfg if train_cfg is None else train_cfg
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
        if seed is not None:
            env_cfg.seed = seed
        configure_torch_backends()
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        return RslRlVecEnvWrapper(env, clip_actions=train_cfg.clip_actions), env_cfg

    def make_alg_runner(self, env, name, *, train_cfg=None, log_dir=None, resume=None,
                        run_name='', device='cuda:0'):
        """Create the PPO runner; log_dir='default' uses logs/mjlab/<experiment>/<time>."""
        train_cfg = self.get_cfgs(name)[1] if train_cfg is None else train_cfg
        # Validate before creating log_dir so a bad call leaves no empty run directory.
        if resume is not None and not Path(resume).is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume}')
        if run_name:
            if '/' in run_name or '\\' in run_name:
                raise ValueError('run_name must be a label, not a path')
            train_cfg.run_name = run_name
        if log_dir == 'default':
            stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_dir = LOG_ROOT / train_cfg.experiment_name / (stamp + (f'_{run_name}' if run_name else ''))
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=False)
        runner = self._task(name).runner_class(
            env, asdict(train_cfg), None if log_dir is None else str(log_dir), device)
        if resume is not None:
            print(f'[INFO] Resuming from {resume}', flush=True)
            runner.load(str(resume))
        return runner, train_cfg


task_registry = TaskRegistry()
