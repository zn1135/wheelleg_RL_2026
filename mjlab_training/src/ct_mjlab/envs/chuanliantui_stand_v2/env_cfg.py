"""Isaac history estimator and get-up recipe on the mjlab physical robot."""

from dataclasses import fields

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..chuanliantui import standup_randomization
from ..chuanliantui.v14_observations import V14ObservationCfg
from ..chuanliantui_stand.env_cfg import make_stand_env_cfg, make_stand_ppo_cfg
from . import diagnostics, mdp, observations, rewards
from .action import StandV2PdActionCfg


def make_stand_v2_env_cfg(*, play=False):
    cfg = make_stand_env_cfg(play=play)
    original = cfg.actions["pd"]
    cfg.actions["pd"] = StandV2PdActionCfg(**{
        item.name: getattr(original, item.name)
        for item in fields(StandV2PdActionCfg) if item.name != "class_type"})
    action = cfg.actions["pd"]
    action.motor_randomization = action.randomize_delays = not play
    action.action_delay_s = None
    action.action_delay_choices_s = (0.,)
    action.nominal_leg_offset_range = (0., 0.)
    # Isaac has no spring randomization; keep mjlab's nominal spring curve/damping.
    action.spring_force_bias_range = (0., 0.)
    action.standup_pushes = not play
    cfg.commands["standup"] = mdp.StandV2CommandCfg()
    for name in ("stand_disturbance", "huananhu_domain_randomization", "huananhu_reset_disturbances"):
        cfg.events.pop(name, None)
    if not play:
        cfg.events["isaac_domain_randomization"] = EventTermCfg(
            func=standup_randomization.randomize_at_startup, mode="startup")
        cfg.events["isaac_reset_pushes"] = EventTermCfg(
            func=standup_randomization.reset_pushes, mode="reset")
    obs_cfg = V14ObservationCfg(noise=not play, delay=not play)
    for name, func in (("actor", observations.actor_observation), ("critic", observations.critic_observation)):
        cfg.observations[name].terms["proprioception"].func = func
        cfg.observations[name].terms["proprioception"].params = {"cfg": obs_cfg}
    cfg.events["reset_observations"].func = observations.reset_observations
    cfg.events["reset_observations"].params = {"cfg": obs_cfg}
    cfg.metrics["observation_delay_ms"].func = observations.sample_observations
    # MetricsManager reports true episode means, including shorter failed episodes.
    cfg.metrics["base_contact_force_N"] = MetricsTermCfg(func=mdp.base_contact_force)
    cfg.metrics["base_contact_fraction"] = MetricsTermCfg(func=diagnostics.base_contact_fraction)
    cfg.metrics["normal_base_contact_fraction"] = MetricsTermCfg(func=diagnostics.normal_base_contact_fraction)
    for side, name in enumerate(("left", "right")):
        cfg.metrics[f"{name}_wheel_support_N"] = MetricsTermCfg(
            func=diagnostics.wheel_support_force, params={"side": side})
    cfg.metrics["wheel_support_weight_ratio"] = MetricsTermCfg(func=diagnostics.wheel_support_weight_ratio)
    cfg.metrics["normal_wheel_support_fraction"] = MetricsTermCfg(func=diagnostics.normal_wheel_support_fraction)
    cfg.observations["history"] = ObservationGroupCfg(terms={"proprioception": ObservationTermCfg(
        func=observations.actor_observation, params={"cfg": obs_cfg},
        history_length=observations.HISTORY_LENGTH)}, concatenate_terms=True, enable_corruption=False)
    cfg.observations["velocity_target"] = ObservationGroupCfg(terms={"velocity": ObservationTermCfg(
        func=observations.velocity_target)}, concatenate_terms=True, enable_corruption=False)
    cfg.rewards = {name: RewardTermCfg(func=rewards.reward, weight=weight, params={"name": name})
                   for name, weight in rewards.WEIGHTS.items() if name != "tracking_lin_vel_tight"}
    cfg.terminations = {"failure": TerminationTermCfg(func=mdp.failure),
                        "time_out": TerminationTermCfg(func=mdp.timeout, time_out=True)}
    return cfg


def make_stand_v2_ppo_cfg():
    cfg = make_stand_ppo_cfg()
    cfg.experiment_name = "chuanliantui_stand_v2"
    cfg.actor.class_name = "ct_mjlab.envs.chuanliantui_stand_v2.network:HistoryActor"
    cfg.actor.distribution_cfg["init_std"] = .5
    cfg.algorithm.class_name = "ct_mjlab.envs.chuanliantui_stand_v2.network:HistoryPPO"
    return cfg
