"""Register mjlab tasks, independently of wheel_legged_gym/Isaac."""
from ..utils.task_registry import task_registry
from .chuanliantui.multimode_env_cfg import make_multimode_env_cfg, make_multimode_ppo_cfg
from .chuanliantui.multimode_runner import MultimodeOnPolicyRunner
from .chuanliantui_stand.env_cfg import make_stand_env_cfg, make_stand_ppo_cfg
from .chuanliantui_stand.runner import StandOnPolicyRunner
from .chuanliantui_stand_v2.env_cfg import make_stand_v2_env_cfg, make_stand_v2_ppo_cfg
from .chuanliantui_stand_v2.runner import StandV2OnPolicyRunner
from .chuanliantui_split.env_cfg import (
    make_getup_env_cfg, make_getup_ppo_cfg, make_walk_env_cfg, make_walk_ppo_cfg)
from .chuanliantui_split.runner import SplitOnPolicyRunner

task_registry.register(
    'chuanliantui', env_cfg=make_multimode_env_cfg, train_cfg=make_multimode_ppo_cfg,
    runner_class=MultimodeOnPolicyRunner,
)
task_registry.register(
    'chuanliantui_stand', env_cfg=make_stand_env_cfg, train_cfg=make_stand_ppo_cfg,
    runner_class=StandOnPolicyRunner,
)
task_registry.register(
    'chuanliantui_stand_v2', env_cfg=make_stand_v2_env_cfg, train_cfg=make_stand_v2_ppo_cfg,
    runner_class=StandV2OnPolicyRunner,
)

task_registry.register('chuanliantui_getup', env_cfg=make_getup_env_cfg,
    train_cfg=make_getup_ppo_cfg, runner_class=SplitOnPolicyRunner)
task_registry.register('chuanliantui_walk', env_cfg=make_walk_env_cfg,
    train_cfg=make_walk_ppo_cfg, runner_class=SplitOnPolicyRunner)

__all__ = ['task_registry']
