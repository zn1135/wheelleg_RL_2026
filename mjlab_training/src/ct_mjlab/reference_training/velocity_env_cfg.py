"""First trainable flat-ground velocity task for chuanliantui."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import velocity_mdp
from .model import ACTUATED_JOINTS, create_robot_cfg
from .pd_action import ChuanliantuiPdActionCfg


# Policy coordinates are the four physical leg motors.
_LEG_JOINTS = ("lf0", "lf00", "rf0", "rf00")
_ACTUATED_JOINTS = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)
_LEG_JOINTS_CFG = SceneEntityCfg("robot", joint_names=_LEG_JOINTS, preserve_order=True)


def make_chuanliantui_velocity_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build a conservative flat-ground body-frame velocity tracking task."""
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(func=env_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=env_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=env_mdp.projected_gravity),
        "leg_pos": ObservationTermCfg(
            func=env_mdp.joint_pos_rel, params={"asset_cfg": _LEG_JOINTS_CFG}
        ),
        "joint_vel": ObservationTermCfg(
            func=env_mdp.joint_vel_rel, params={"asset_cfg": _ACTUATED_JOINTS}
        ),
        "command": ObservationTermCfg(
            func=env_mdp.generated_commands, params={"command_name": "velocity"}
        ),
        "last_action": ObservationTermCfg(func=env_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="error",
        ),
        "critic": ObservationGroupCfg(
            terms={
                **actor_terms,
                "root_height": ObservationTermCfg(func=velocity_mdp.root_height),
            },
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="error",
        ),
    }
    commands: dict[str, CommandTermCfg] = {
        "velocity": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 5.0),
            rel_standing_envs=0.15,
            rel_forward_envs=0.30,
            debug_vis=play,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.8, 0.8),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.8, 0.8),
            ),
        )
    }
    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=velocity_mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "velocity", "std": 0.45},
        ),
        "track_yaw_velocity": RewardTermCfg(
            func=velocity_mdp.track_yaw_velocity,
            weight=0.75,
            params={"command_name": "velocity", "std": 0.5},
        ),
        "upright": RewardTermCfg(
            func=velocity_mdp.upright, weight=1.0, params={"std": 0.35}
        ),
        "nominal_height": RewardTermCfg(
            func=velocity_mdp.nominal_height,
            weight=0.5,
            params={"target": 0.3376, "std": 0.05},
        ),
        "torque_l2": RewardTermCfg(
            func=velocity_mdp.native_torque_l2,
            weight=-2.0e-4,
            params={"asset_cfg": _ACTUATED_JOINTS},
        ),
        "action_rate": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-0.01),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=env_mdp.bad_orientation, params={"limit_angle": 0.85}
        ),
        "too_low": TerminationTermCfg(
            func=env_mdp.root_height_below_minimum, params={"minimum_height": 0.19}
        ),
        "nan": TerminationTermCfg(func=env_mdp.nan_detection),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": create_robot_cfg()},
            num_envs=1 if play else 256,
            env_spacing=3.0,
        ),
        observations=observations,
        actions={"pd": ChuanliantuiPdActionCfg(
            entity_name="robot", spring_force_bias_range=(0.0, 0.0) if play else (-50.0, 50.0)
        )},
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=2.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=2_000,
            mujoco=MujocoCfg(
                timestep=0.005,
                integrator="implicitfast",
                iterations=100,
                ls_iterations=20,
                tolerance=1e-10,
                cone="elliptic",
                impratio=10.0,
            ),
        ),
        decimation=4,
        episode_length_s=12.0,
    )


def make_chuanliantui_velocity_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(128, 64), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=3,
            num_mini_batches=2,
            learning_rate=5.0e-4,
            entropy_coef=0.01,
            desired_kl=0.01,
        ),
        experiment_name="chuanliantui_velocity_flat_direct_pd",
        logger="tensorboard",
        upload_model=False,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=5_000,
        clip_actions=None,
    )
