"""Standing-only training profile with the existing 500/100 Hz control path."""

import re
from dataclasses import fields, replace

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..chuanliantui import huananhu_rewards
from ..chuanliantui.multimode_env_cfg import make_multimode_env_cfg, make_multimode_ppo_cfg
from . import mdp, rewards, observations, randomization
from .action import StandPdActionCfg


def make_stand_env_cfg(*, play=False):
    cfg = make_multimode_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 2048
    cfg.sim.mujoco.timestep = .002
    cfg.decimation = 5
    original_action = cfg.actions["pd"]
    cfg.actions["pd"] = StandPdActionCfg(**{
        item.name: getattr(original_action, item.name)
        for item in fields(original_action) if item.name != "class_type"})
    cfg.actions["pd"].action_delay_s = None
    cfg.actions["pd"].action_delay_choices_s = ()
    cfg.actions["pd"].randomize_delays = True
    cfg.actions["pd"].randomization_scale = 1. if play else .25
    if not play:
        cfg.actions["pd"].nominal_leg_offset_range = tuple(
            value * .25 for value in cfg.actions["pd"].nominal_leg_offset_range)
        cfg.events["huananhu_domain_randomization"] = EventTermCfg(
            func=randomization.randomize_at_startup, mode="startup", params={"scale": .25})
    observation_cfg = observations.StandObservationCfg(noise=not play, delay=not play)
    for name, func in (("actor", observations.actor_observation), ("critic", observations.critic_observation)):
        cfg.observations[name].terms["proprioception"].func = func
        cfg.observations[name].terms["proprioception"].params = {"cfg": observation_cfg}
    cfg.events["reset_observations"].func = observations.reset_observations
    cfg.metrics["observation_delay_ms"].func = observations.sample_observations
    cfg.episode_length_s = 40.
    pose = mdp.standing_pose()
    initial = cfg.scene.entities["robot"].init_state
    initial.pos = (0., 0., pose["height"] + mdp.RESET_CLEARANCE_M)
    initial.joint_pos = {f"^{re.escape(name)}$": value for name, value in pose["joint_positions"].items()}
    # Reset posture changes; the PD/observation nominal motor angles do not.
    cfg.commands["standup"] = mdp.StandCommandCfg(disturbances=not play)
    cfg.events["reset_pose"] = EventTermCfg(func=mdp.reset_pose, mode="reset")
    cfg.events.pop("huananhu_velocity_push", None)
    cfg.events.pop("huananhu_local_wrench", None)
    if not play:
        cfg.events["stand_disturbance"] = EventTermCfg(func=mdp.disturb_once, mode="step")
    cfg.rewards = {
        name: RewardTermCfg(func=huananhu_rewards.reward, weight=weight, params={"name": name})
        for name, weight in huananhu_rewards.REWARD_WEIGHTS.items()
    }
    for name in ("flat_orientation_y_v", "flat_orientation_y_exp",
                 "flat_orientation_x_v", "flat_orientation_x_exp"):
        cfg.rewards[name] = RewardTermCfg(func=rewards.orientation_outside_recovery,
            weight=huananhu_rewards.REWARD_WEIGHTS[name], params={"name": name})
    # Legacy static costs also stay off while getting up or recovering.
    for name in ("stand_still_lin_vel", "no_fork", "no_fork_square",
                 "leg_joint_vel", "wheel_vel"):
        cfg.rewards[name] = RewardTermCfg(func=rewards.hold_cost_outside_recovery,
            weight=huananhu_rewards.REWARD_WEIGHTS[name], params={"name": name})
    cfg.rewards["stand_low_velocity"] = RewardTermCfg(func=rewards.low_velocity_hold, weight=3.)
    cfg.rewards["stand_quiet_legs"] = RewardTermCfg(func=rewards.quiet_legs, weight=3.)
    cfg.rewards["stand_leg_speed_cost"] = RewardTermCfg(func=rewards.quiet_leg_speed_cost, weight=-20.)
    cfg.rewards["stand_body_speed_margin_cost"] = RewardTermCfg(func=rewards.body_speed_margin_cost, weight=-40.)
    cfg.rewards["stand_posture"] = RewardTermCfg(func=rewards.standing_posture, weight=2.)
    cfg.rewards["stand_quiet_wheels"] = RewardTermCfg(func=rewards.quiet_wheels, weight=1.)
    cfg.rewards["stand_recovery_upright"] = RewardTermCfg(func=rewards.recovery_upright, weight=2.)
    cfg.rewards["stand_recovery_wheel_height"] = RewardTermCfg(func=rewards.recovery_wheel_height, weight=4.)
    cfg.metrics["physical_leg_speed_squared"] = MetricsTermCfg(
        func=rewards.sample_leg_velocity, per_substep=True)
    return cfg


def configure_evaluation(cfg):
    """Fixed acceptance retains full original noise and physical randomization."""
    cfg.actions["pd"].randomization_scale = 1.
    if "huananhu_domain_randomization" in cfg.events:
        cfg.events["huananhu_domain_randomization"].params["scale"] = 1.
        cfg.actions["pd"].nominal_leg_offset_range = (
            make_multimode_env_cfg(play=False).actions["pd"].nominal_leg_offset_range)
    for group in cfg.observations.values():
        term = group.terms["proprioception"]
        term.params["cfg"] = replace(term.params["cfg"], noise_scale=1.)
    return cfg


def make_stand_ppo_cfg():
    cfg = make_multimode_ppo_cfg()
    cfg.experiment_name = "chuanliantui_stand_v1"
    cfg.num_steps_per_env = mdp.STEPS_PER_ITERATION
    return cfg
