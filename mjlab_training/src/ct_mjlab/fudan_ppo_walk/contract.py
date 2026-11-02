"""Frozen reference hyperparameters and the distinct physical-motor contract."""
from dataclasses import dataclass

REFERENCE = '8204e853dfd2ed06d85a322e1a998c3d20a3be2c'
CONTRACT = 'fudan-ppo-walk-physical25-history125-critic144-v1'
RESET_CONTRACT = 'fudan-physical-pd-centred-warmup-noisy5-clipped-history-v3'
RESET_V2_CONTRACT = 'fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2'
LEGACY_RESET_CONTRACT = 'fudan-trial01-fixed026-noisy5-clipped-history-v1'
TASK = 'fudan_ppo_walk'
PHYSICS_DT, POLICY_DT = .002, .01
MOTORS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
LEGS = (0, 1, 3, 4)
WEIGHTS = dict(tracking_lin_vel=1., tracking_lin_vel_enhance=1.,
    tracking_ang_vel=1., tracking_ang_vel_enhance=1., base_height=1., nominal_state=-1.,
    lin_vel_z=-1., ang_vel_xy=-.2, orientation=-100., dof_vel=-5e-5, dof_acc=-2.5e-7,
    torques=-.0001, action_rate=-.01, action_smooth=-.01, collision=-1., dof_pos_limits=-1.)
PPO_CONFIG = dict(num_learning_epochs=5, num_mini_batches=4, clip_param=.2, gamma=.99,
    lam=.95, value_loss_coef=1., entropy_coef=.01, learning_rate=.001,
    extra_learning_rate=.001, max_grad_norm=1., use_clipped_value_loss=True,
    schedule='adaptive', desired_kl=.005)


def make_network():
    from .network import ActorCriticSequence
    return ActorCriticSequence(25, 144, 6, 125, 3, encoder_hidden_dims=[128,64],
        actor_hidden_dims=[128,64,32], critic_hidden_dims=[256,128,64],
        activation='elu', init_noise_std=.5)


@dataclass
class TrainConfig:
    seed: int = 1
    experiment_name: str = TASK
    clip_actions: float = 100.
    num_steps_per_env: int = 48
    save_interval: int = 100
    run_name: str = ''


def make_train_cfg():
    return TrainConfig()
