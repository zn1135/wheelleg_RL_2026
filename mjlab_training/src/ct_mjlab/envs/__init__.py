"""Register mjlab tasks, independently of wheel_legged_gym/Isaac."""
from ..utils.task_registry import task_registry
from .chuanliantui.multimode_env_cfg import make_multimode_env_cfg, make_multimode_ppo_cfg
from .chuanliantui.multimode_runner import MultimodeOnPolicyRunner

task_registry.register(
    'chuanliantui', env_cfg=make_multimode_env_cfg, train_cfg=make_multimode_ppo_cfg,
    runner_class=MultimodeOnPolicyRunner,
)

__all__ = ['task_registry']
