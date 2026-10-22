"""Flat-ground Huananhu task on the validated CAD robot.

2026-10-17: the phase goal moved from mixed-terrain stand-up/recovery to
linear-velocity and yaw-rate tracking, so the generated flat/slope/step/stair
terrain is replaced by the infinite ground plane and the command envelope is
narrowed to a deployable band (see huananhu_mdp). Mode and flag channels, the
ground-start/recover machinery and the 39/82 contract are unchanged. On a plane
every env is terrain family 0 (flat), so the slope/step/stair triggers stay
dormant; `multimode_terrains.py` is kept for restoring the mixed-terrain recipe.
"""

import json
import math

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg, TerrainHeightSensorCfg

from . import huananhu_randomization, multimode_mdp, multimode_observations, multimode_rewards
from .huananhu_env_cfg import make_huananhu_env_cfg, make_huananhu_ppo_cfg
from .v14_observations import V14ObservationCfg
from .model import MODEL_XML


# Explicit play/check knockdowns only; training does not register this event.
KNOCKDOWN_ANGULAR_SPEED_RANGE = (18., 28.)


def make_multimode_env_cfg(*, play=False):
    cfg = make_huananhu_env_cfg(play=play)
    timing = json.loads(MODEL_XML.with_name("h7_timing.json").read_text())
    # Training broadens action waiting; playback retains the measured one tick.
    cfg.actions["pd"].action_delay_s = timing["simulation"]["action_delay_s"] if play else None
    cfg.actions["pd"].action_delay_choices_s = (.006, .008, .010)
    cfg.actions["pd"].motor_response_time_constant_s = timing["simulation"]["motor_response_time_constant_s"]
    cfg.actions["pd"].randomize_delays = not play
    cfg.actions["pd"].nominal_leg_offset_range = (0., 0.) if play else (-math.radians(5), math.radians(5))
    # Flat phase: the ground is the infinite plane, so the generated terrain is gone and
    # every env is terrain family 0. Travel is bounded only by
    # multimode_mdp.OUT_OF_PATCH_RADIUS_M, which ends a runaway episode; it does not need
    # to separate environments (each env is its own physics world), so env_spacing keeps
    # the task's own value instead of growing with the radius.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors += (
        RayCastSensorCfg(
            name="terrain_scan", frame=ObjRef(type="body", name="base_link", entity="robot"),
            pattern=GridPatternCfg(size=(1.2, 0.), resolution=.2),
            ray_alignment="yaw", max_distance=2., include_geom_groups=(0,),
            debug_vis=False),
        TerrainHeightSensorCfg(
            name="wheel_height", frame=(
                ObjRef(type="body", name="lfwheel", entity="robot"),
                ObjRef(type="body", name="rfwheel", entity="robot")),
            pattern=GridPatternCfg(size=(0., 0.), resolution=.1),
            ray_alignment="world", max_distance=2., include_geom_groups=(0,),
            reduction="min", debug_vis=False),
    )
    observation_cfg = V14ObservationCfg(noise=not play, delay=not play)
    cfg.observations = {
        name: ObservationGroupCfg(
            terms={"proprioception": ObservationTermCfg(func=func, params={"cfg": observation_cfg})},
            concatenate_terms=True, enable_corruption=False, nan_policy="error")
        for name, func in (("actor", multimode_observations.actor_observation),
                           ("critic", multimode_observations.critic_observation))
    }
    standup_cfg = multimode_mdp.MultimodeCommandCfg()
    cfg.commands["standup"] = standup_cfg
    cfg.events["reset_pose"] = EventTermCfg(
        func=multimode_mdp.reset_pose, mode="reset",
        params={"ground_probability": standup_cfg.ground_start_probability})
    if not play:
        # Keep ordinary training disturbances; recovery handles naturally occurring falls.
        cfg.events["huananhu_velocity_push"] = EventTermCfg(
            func=huananhu_randomization.push_root_velocity, mode="interval",
            interval_range_s=(5., 10.), params={"linear_range": (-.6, .6)})
        cfg.events["huananhu_local_wrench"] = EventTermCfg(
            func=huananhu_randomization.sample_local_wrench, mode="interval",
            interval_range_s=(5., 10.),
            params={"force_range": (-20., 20.), "torque_range": (-2., 2.)})
    cfg.events["reset_observations"] = EventTermCfg(func=multimode_observations.reset_observations, mode="reset")
    cfg.metrics["observation_delay_ms"] = MetricsTermCfg(
        func=multimode_observations.sample_observations, per_substep=True, reduce="last")
    cfg.terminations["failure"] = TerminationTermCfg(func=multimode_mdp.failure)
    for name, weight in multimode_rewards.WEIGHTS.items():
        cfg.rewards[name] = RewardTermCfg(
            func=multimode_rewards.reward, weight=weight, params={"name": name})
    return cfg


def make_multimode_ppo_cfg():
    cfg = make_huananhu_ppo_cfg()
    cfg.experiment_name = "chuanliantui_huananhu_multimode_v1"
    return cfg
