"""Frozen reference hyperparameters and the distinct physical-motor contract."""
from dataclasses import dataclass
REFERENCE = '8204e853dfd2ed06d85a322e1a998c3d20a3be2c'
CONTRACT = 'fudan-ppo-walk-physical25-history125-critic144-v1'
RESET_CONTRACT = 'fudan-physical-pd-centred-warmup-noisy5-clipped-history-v3'
RESET_V2_CONTRACT = 'fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2'
LEGACY_RESET_CONTRACT = 'fudan-trial01-fixed026-noisy5-clipped-history-v1'
TASK = 'fudan_ppo_walk'
PHYSICS_DT, POLICY_DT = (0.002, 0.01)
MOTORS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
LEGS = (0, 1, 3, 4)
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO, section_dict
WEIGHTS = FudanWalkCfg().rewards.scales
PPO_CONFIG = dict(num_learning_epochs=5, num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.01, learning_rate=0.001, extra_learning_rate=0.001, max_grad_norm=1.0, use_clipped_value_loss=True, schedule='adaptive', desired_kl=0.005)

# v1模型采用的固定网络接口；用于拒绝配置漂移，不能按当前配置动态生成。
NETWORK_CONFIG_V1 = dict(encoder_hidden_dims=[128, 64], actor_hidden_dims=[128, 64, 32],
                         critic_hidden_dims=[256, 128, 64], activation='elu', init_noise_std=.5)


def validate_network_contract(train_cfg=None):
    cfg = FudanWalkCfgPPO() if train_cfg is None else train_cfg
    if section_dict(cfg.policy) != NETWORK_CONFIG_V1:
        raise ValueError('Checkpoint v1 network contract differs from the current policy configuration')


def make_network(train_cfg=None):
    from ct_mjlab.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence
    cfg = FudanWalkCfgPPO() if train_cfg is None else train_cfg
    return ActorCriticSequence(25, 144, 6, 125, 3, **section_dict(cfg.policy))

def make_train_cfg():
    return FudanWalkCfgPPO()
TrainConfig = FudanWalkCfgPPO
