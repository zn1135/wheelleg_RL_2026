"""Fresh-run config factories; historical checkpoint contracts live in their original modules."""
from dataclasses import dataclass
from importlib import import_module

from .profiles import TRAINING_PROFILES


@dataclass(frozen=True)
class Recipe:
    module: str
    env_class: str
    runner_class: str


RECIPES = {
    'huananhu-v14': Recipe('huananhu_v14', 'V14VecEnv', 'V14Runner'),
    'huananhu-v14-height': Recipe('huananhu_v14_height', 'HeightVecEnv', 'HeightRunner'),
    'huananhu-v14-stand': Recipe('huananhu_v14_stand', 'StandVecEnv', 'StandRunner'),
    'huananhu-v14-recovery': Recipe('huananhu_v14_recovery', 'RecoveryVecEnv', 'RecoveryRunner'),
    'huananhu-v14-wheel-support': Recipe('huananhu_v14_wheel_support', 'RecoveryVecEnv', 'RecoveryRunner'),
    'getup-r15': Recipe('getup_half_assistance', 'HalfAssistanceVecEnv', 'HalfAssistanceRunner'),
    'getup-r15-orientation': Recipe('getup_half_assistance', 'HalfAssistanceVecEnv', 'HalfAssistanceRunner'),
    'recover-r14': Recipe('ct_mjlab.reference_training.entry', 'RecoverVecEnv', 'runner_class'),
}


def recipe(profile):
    try:
        definition = RECIPES[profile]
    except KeyError:
        raise ValueError(f'Unknown training profile: {profile}; choose from {TRAINING_PROFILES}') from None
    module = definition.module
    if not module.startswith('ct_mjlab.'):
        module = f'ct_mjlab.experiments.{module}'
    return definition, import_module(module)


def get_cfgs(*, profile='huananhu-v14', num_envs=4096, seed=42, play=False):
    """Return fresh (mjlab env cfg, RSL-RL PPO dict), with stage zero for r15.

    Use the CLI for historical resume/playback: it restores the saved stage and
    applies each version's full contract validation before creating an environment.
    """
    _, module = recipe(profile)
    if profile == 'recover-r14':
        env_cfg = module.make_huananhu_env_cfg(play=play)
        env_cfg.scene.num_envs, env_cfg.seed = num_envs, seed
        train_cfg = module.runner_cfg(seed)
    elif profile in ('getup-r15', 'getup-r15-orientation'):
        strong = profile == 'getup-r15-orientation'
        env_cfg = module.make_env_cfg(num_envs, seed, play, strong_orientation=strong)
        train_cfg = module.base.runner_cfg(seed)
        label = 'getup_r15_half_assistance' + ('_orientation' if strong else '')
        train_cfg.update(run_name=label, experiment_name='chuanliantui_' + label)
    else:
        env_cfg = module.make_env_cfg(num_envs, seed, play)
        train_cfg = module.runner_cfg(seed)
    return env_cfg, train_cfg
