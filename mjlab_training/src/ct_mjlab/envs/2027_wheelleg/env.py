"""2027 closed-chain robot on the shared PPO walking observation/reward loop."""
from dataclasses import replace
import hashlib

import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import (
    FudanEnv, PhysicalAction, make_env_cfg as make_walk_env_cfg,
)
from .config import TASK, WheellegCfg
from .model import MODEL_XML, SPRING_FORCE_N, create_robot_cfg, load_spec
from .reset_geometry import ResetGeometry, nominal_reset_pose


def wheel_profile(profile):
    if profile != "mechanical":
        raise ValueError("2027 wheels use direct wheel-axis torque")
    return dict(version="2027-wheel-unit-torque-v1", model_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),
                wheel_order=["lfwheel", "rfwheel"], motor_reduction=15.7,
                actuator_gain=1., extra_output_gain=1., speed_shaping=False)


class PhysicalAction2027(PhysicalAction):
    def rebuild_reset_poses(self):
        solver = ResetGeometry()
        poses = [solver.solve(row) for row in self.nominal_positions.detach().cpu().numpy()]
        names = self._entity.joint_names
        self.reset_joint_positions = torch.tensor(
            [[pose["joint_positions"][name] for name in names] for pose in poses],
            dtype=self._nominal_pos.dtype, device=self.device)
        self.reset_root_height = torch.tensor([pose["height"] for pose in poses],
                                             dtype=self._nominal_pos.dtype, device=self.device)

    def apply_actions(self):
        super().apply_actions()
        # The only supplied spring measurement is its nominal 400 N thrust.
        self._env.sim.data.ctrl[:, self._spring_ids] = SPRING_FORCE_N


def make_env_cfg(*, play=False, disturbances=None, task_cfg=None):
    task = WheellegCfg() if task_cfg is None else task_cfg
    cfg = make_walk_env_cfg(play=play, disturbances=disturbances, task_cfg=task)
    previous = cfg.scene.entities["robot"]
    robot = create_robot_cfg()
    robot.geoms = previous.geoms
    pose = nominal_reset_pose()
    robot.init_state.pos = (0., 0., pose["height"])
    robot.init_state.joint_pos = {f"^{name}$": value for name, value in pose["joint_positions"].items()}
    cfg.scene.entities["robot"] = robot
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.mujoco.solver = "newton"
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.tolerance = 1e-10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 10.
    cfg.actions["pd"] = replace(cfg.actions["pd"], class_type=PhysicalAction2027)
    return cfg


class WheellegEnv(FudanEnv):
    task_name = TASK
    physics_cfg_factory = staticmethod(make_env_cfg)
    wheel_profile_factory = staticmethod(wheel_profile)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Continuous hip shafts have no positional soft-stop penalty.
        self.front_limits[[0, 2], 0] = -torch.inf
        self.front_limits[[0, 2], 1] = torch.inf
