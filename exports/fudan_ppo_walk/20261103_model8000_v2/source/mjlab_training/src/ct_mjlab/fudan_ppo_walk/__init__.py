"""旧包导入兼容；生产环境和训练器已迁移到envs及rsl_rl。"""
def make_env_cfg(*, play=False):
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import make_env_cfg
    return make_env_cfg(play=play)

def make_train_cfg():
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfgPPO
    return FudanWalkCfgPPO()

def make_runner(*args, **kwargs):
    from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import FudanRunner
    return FudanRunner(*args, **kwargs)
