"""2027 wheel-legged robot training tasks."""
from ct_mjlab.utils.task_registry import task_registry
from .config import TASK, WheellegCfg, WheellegCfgPPO

task_registry.register(
    TASK, env_cfg=lambda play=False: WheellegCfg(), train_cfg=WheellegCfgPPO,
    task_class="ct_mjlab.envs.2027_wheelleg.env.WheellegEnv",
    runner_class="ct_mjlab.envs.2027_wheelleg.runner.WheellegRunner",
)
