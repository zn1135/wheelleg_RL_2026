"""仅注册保留的算法环境；旧独立训练task已撤销。"""
from ..utils.task_registry import task_registry
from .chuanliantui.multimode_env_cfg import make_multimode_env_cfg, make_multimode_ppo_cfg
from .chuanliantui.multimode_runner import MultimodeOnPolicyRunner
from .chuanliantui_split.env_cfg import make_walk_env_cfg, make_walk_ppo_cfg
from .chuanliantui_split.runner import SplitOnPolicyRunner

task_registry.register('smp', env_cfg=make_multimode_env_cfg,
    train_cfg=make_multimode_ppo_cfg, runner_class=MultimodeOnPolicyRunner)
task_registry.register('supermimic', env_cfg=make_walk_env_cfg,
    train_cfg=make_walk_ppo_cfg, runner_class=SplitOnPolicyRunner)
from .fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO

task_registry.register('fudan_ppo_walk', env_cfg=lambda play=False: FudanWalkCfg(),
    train_cfg=FudanWalkCfgPPO,
    task_class='ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',
    runner_class='ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.FudanRunner')
__all__ = ['task_registry']
