"""Independent mixed-terrain Huananhu task on the validated CAD robot."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg, TerrainHeightSensorCfg
from mjlab.terrains import BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg, TerrainGeneratorCfg

from . import multimode_mdp, multimode_observations, multimode_rewards
from .huananhu_env_cfg import make_huananhu_env_cfg, make_huananhu_ppo_cfg
from .v14_observations import V14ObservationCfg
from .multimode_terrains import BoxSlopeTerrainCfg


def make_multimode_env_cfg(*, play=False):
    cfg = make_huananhu_env_cfg(play=play)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42, curriculum=True, size=(6., 6.), num_rows=3, num_cols=4,
        difficulty_range=(0., 1.),
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=.50),
            "slope": BoxSlopeTerrainCfg(proportion=.20),
            "step": BoxPyramidStairsTerrainCfg(
                proportion=.15, step_height_range=(.04, .08), step_width=2.,
                platform_width=1., border_width=.5),
            "stair": BoxPyramidStairsTerrainCfg(
                proportion=.15, step_height_range=(.04, .06), step_width=.45,
                platform_width=1., border_width=.5),
        },
    )
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
    cfg.commands["standup"] = multimode_mdp.MultimodeCommandCfg()
    cfg.events["reset_pose"] = EventTermCfg(func=multimode_mdp.reset_pose, mode="reset")
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
