"""Independent Huananhu V14 Flat profile on the current chuanliantui hardware."""

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
from .model import create_robot_cfg, standing_motor_positions
from .v14_observations import V14ObservationCfg
from .velocity_env_cfg import make_chuanliantui_velocity_env_cfg, make_chuanliantui_velocity_ppo_cfg


def make_huananhu_env_cfg(*, play=False):
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
        entity_name='robot', leg_position_scale=.5, wheel_stiffness=.1, wheel_damping=0.,
        nominal_leg_positions=standing_motor_positions(), motor_randomization=not play,
        randomize_delays=not play, spring_force_bias_range=(0., 0.) if play else (-50., 50.),
        standup_landing_gate=False, standup_pushes=False,
    )}
    cfg.commands = {'standup': huananhu_mdp.HuananhuCommandCfg()}
    cfg.events = make_huananhu_events(play=play)
    cfg.events['reset_pose'] = EventTermCfg(func=huananhu_mdp.reset_pose, mode='reset')
    cfg.events['reset_observations'] = EventTermCfg(func=huananhu_observations.reset_observations, mode='reset')
    cfg.metrics = {'observation_delay_ms': MetricsTermCfg(
        func=huananhu_observations.sample_observations, per_substep=True, reduce='last')}
    cfg.curriculum = {}
    cfg.rewards = {name: RewardTermCfg(func=huananhu_rewards.reward, weight=weight, params={'name': name})
                   for name, weight in huananhu_rewards.REWARD_WEIGHTS.items()}
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
    cfg.experiment_name = 'chuanliantui_huananhu_flat_sourcexml_spring90mm_v11init_velocity_reciprocal_v12'
    cfg.clip_actions = None
    return cfg
