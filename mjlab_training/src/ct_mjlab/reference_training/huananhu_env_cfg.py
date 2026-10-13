"""Continuous Huananhu recover line on the current chuanliantui hardware."""

from mjlab.envs import mdp as env_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import GeomCfg

from . import huananhu_mdp, huananhu_observations, huananhu_rewards
from .huananhu_action import HuananhuPdActionCfg
from .huananhu_randomization import make_huananhu_events
from .model import POLICY_JOINTS, create_robot_cfg, standing_motor_positions
from .v14_observations import V14ObservationCfg
from .velocity_env_cfg import make_chuanliantui_velocity_env_cfg, make_chuanliantui_velocity_ppo_cfg

RESET_CONFIG_FIELDS = ('training_stage', 'ground_start_probability',
                       'ground_randomization_probability', 'ground_tilt_max_rad',
                       'ground_yaw_max_rad', 'ground_height_offset_range')
COMMAND_CONFIG_FIELDS = ('zero_velocity_commands', 'linear_velocity_range',
                         'yaw_velocity_range', 'heading_probability',
                         'heading_control_stiffness', 'special_modes_enabled')


def make_huananhu_env_cfg(*, play=False, recover=True, dynamic_modes=True,
                         policy_joint_order=POLICY_JOINTS, knee_limit_weight=None,
                         reset_config=None, height_gate=None, command_config=None):
    height_gate = tuple(huananhu_rewards.HEIGHT_GATE if height_gate is None else height_gate)
    if len(height_gate) != 2 or not 0. < height_gate[0] < height_gate[1]:
        raise ValueError('Height gate requires 0 < full < off')
    cfg = make_chuanliantui_velocity_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 4096
    cfg.scene.entities['robot'] = create_robot_cfg(base_height=.38)
    cfg.scene.entities['robot'].geoms = (GeomCfg(geom_names_expr=('.*',), priority=1),)
    cfg.scene.terrain.geoms = (GeomCfg(geom_names_expr=('^terrain$',), friction=(1., .005, .0001)),)
    cfg.scene.sensors = tuple(
        ContactSensorCfg(name=name, primary=ContactMatch(mode='body', pattern=pattern, entity='robot'),
                         fields=('found', 'force'), reduce='netforce', history_length=8)
        for name, pattern in (
            ('wheel_contact', '^(lfwheel|rfwheel)$'),
            ('base_contact', '^base_link$'),
            ('undesired_contact', '^(base_link|[lr]f(0|1|00|01|02|03))$'),
        )
    )
    observation_cfg = V14ObservationCfg(noise=not play, delay=not play)
    cfg.observations = {
        name: ObservationGroupCfg(
            terms={'proprioception': ObservationTermCfg(func=func, params={'cfg': observation_cfg})},
            concatenate_terms=True, enable_corruption=False, nan_policy='error')
        for name, func in (('actor', huananhu_observations.actor_observation),
                           ('critic', huananhu_observations.critic_observation))
    }
    cfg.actions = {'pd': HuananhuPdActionCfg(
        entity_name='robot', policy_joint_order=policy_joint_order,
        leg_position_scale=.5, wheel_stiffness=.1, wheel_damping=0.,
        nominal_leg_positions=standing_motor_positions(), motor_randomization=not play,
        randomize_delays=not play, spring_force_bias_range=(0., 0.) if play else (-50., 50.),
        standup_landing_gate=False, standup_pushes=False,
    )}
    cfg.commands = {'standup': huananhu_mdp.HuananhuCommandCfg()}
    if recover:
        # The existing v12 task is the continuous recover line. Other tasks
        # explicitly request the historical mixed base before replacing commands.
        cfg.commands['standup'] = huananhu_mdp.HuananhuCommandCfg(
            training_stage='recover_mixed_reset_r10', ground_start_probability=.5,
            zero_velocity_commands=False, special_modes_enabled=False,
            strict_recovery_metrics=True,
            ground_randomization_probability=0., ground_tilt_max_rad=0.,
            ground_yaw_max_rad=0., ground_height_offset_range=(0., 0.))
    if reset_config is not None:
        if set(reset_config) != set(RESET_CONFIG_FIELDS):
            raise ValueError('Reset configuration must contain the complete reset profile')
        for name, value in reset_config.items():
            setattr(cfg.commands['standup'], name, value)
    cfg.commands['standup'].dynamic_modes=dynamic_modes
    if command_config is not None:
        if set(command_config) != set(COMMAND_CONFIG_FIELDS):
            raise ValueError('Command configuration must contain the complete velocity profile')
        for name, value in command_config.items():
            setattr(cfg.commands['standup'], name, value)
    cfg.events = make_huananhu_events(play=play)
    cfg.events['reset_pose'] = EventTermCfg(func=huananhu_mdp.reset_pose, mode='reset')
    cfg.events['reset_observations'] = EventTermCfg(func=huananhu_observations.reset_observations, mode='reset')
    cfg.metrics = {'observation_delay_ms': MetricsTermCfg(
        func=huananhu_observations.sample_observations, per_substep=True, reduce='last')}
    cfg.curriculum = {}
    cfg.rewards = {name: RewardTermCfg(func=huananhu_rewards.reward, weight=weight,
                                    params={'name': name, 'height_gate': height_gate})
                   for name, weight in huananhu_rewards.REWARD_WEIGHTS.items()}
    if knee_limit_weight is not None:
        cfg.rewards['knee_limit'].weight = knee_limit_weight
    cfg.terminations = {
        'failure': TerminationTermCfg(func=huananhu_mdp.failure),
        'time_out': TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    }
    # Preserve the validated closed-chain integrator step, match source control dt.
    cfg.sim.mujoco.timestep = .002
    cfg.decimation = 10
    cfg.episode_length_s = 20.
    return cfg


def make_huananhu_ppo_cfg():
    cfg = make_chuanliantui_velocity_ppo_cfg()
    cfg.actor = RslRlModelCfg(
        hidden_dims=(256, 128, 64), activation='elu', obs_normalization=False,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'})
    cfg.critic = RslRlModelCfg(hidden_dims=(256, 128, 64), activation='elu', obs_normalization=False)
    cfg.algorithm.learning_rate = 1e-4
    cfg.algorithm.value_loss_coef = 4.
    cfg.algorithm.entropy_coef = .005
    cfg.algorithm.num_learning_epochs = 5
    cfg.algorithm.num_mini_batches = 4
    cfg.algorithm.desired_kl = .01
    cfg.algorithm.gamma = .99
    cfg.algorithm.lam = .95
    cfg.num_steps_per_env = 24
    cfg.max_iterations = 20000
    cfg.experiment_name = 'chuanliantui_huananhu_flat_velocity_heading_r14'
    cfg.clip_actions = None
    return cfg
