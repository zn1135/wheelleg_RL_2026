"""Register mjlab environments, independently of wheel_legged_gym/Isaac."""
from ..utils.task_registry import task_registry
from .chuanliantui.chuanliantui import make_env, run, runner_class
from .chuanliantui.chuanliantui_config import get_cfgs

task_registry.register(
    'chuanliantui', run=run, get_cfgs=get_cfgs,
    make_env=make_env, runner_class=runner_class, default_profile='huananhu-v14',
)

__all__ = ['task_registry']
