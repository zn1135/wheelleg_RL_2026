"""2027 robot geometry, free-base asset and closed reset poses."""
from functools import lru_cache
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ROOT = Path(__file__).resolve().parents[5]
MODEL_XML = ROOT / "resources/robots/2027_wheelleg/mjlab/wheelleg_2027.xml"
ACTUATED_JOINTS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
SPRING_FORCE_N = 400.0


def load_spec():
    return mujoco.MjSpec.from_file(str(MODEL_XML))


def load_model():
    return load_spec().compile()


def closed_chain_residual(model, data):
    mujoco.mj_kinematics(model, data)
    pairs = [
        (f"{side}_{a}", f"{side}_{b}")
        for side in ("left", "right")
        for a, b in (("rear_pin1", "front_pin1"), ("rear_pin2", "front_pin2"),
                     ("spring_tip", "gas_spring_lower"))
    ]
    return max(float(np.linalg.norm(data.site(a).xpos - data.site(b).xpos)) for a, b in pairs)


def solve_springs(model, data):
    mujoco.mj_kinematics(model, data)
    for side, parent in (("left", "lf0"), ("right", "rf0")):
        delta = data.body(parent).xmat.reshape(3, 3).T @ (
            data.site(f"{side}_gas_spring_lower").xpos - data.site(f"{side}_gas_spring_upper").xpos)
        quat = model.body(f"{side}_spring1").quat
        angle = 2 * np.arctan2(quat[2], quat[0])
        length = model.body(f"{side}_spring2").pos[2]
        data.qpos[model.joint(f"{side}_spring1_joint").qposadr[0]] = np.arctan2(delta[0], delta[2]) - angle
        data.qpos[model.joint(f"{side}_spring2_joint").qposadr[0]] = np.hypot(delta[0], delta[2]) - length
    mujoco.mj_kinematics(model, data)


def _leg_constraints(model, data, side):
    return np.concatenate([
        (data.site(f"{side}_rear_pin{i}").xpos - data.site(f"{side}_front_pin{i}").xpos)[[0, 2]]
        for i in (1, 2)
    ])


@lru_cache(maxsize=1)
def standing_pose():
    """Solve both leg assemblies at a 0.26 m chassis height from this robot's CAD."""
    model = load_model()
    data = mujoco.MjData(model)
    wheel_radius = 0.060
    base_height = 0.26
    body_x = float(model.body("base_link").ipos[0])
    for side, prefix, sign in (("left", "lf", 1), ("right", "rf", -1)):
        names = [prefix + s for s in ("0", "00", "1", "01", "02", "03")]
        ids = [int(model.joint(name).qposadr[0]) for name in names]
        lower = [-np.pi, -np.pi, *([0.033] if sign == 1 else [-0.96]), -np.pi, -np.pi, -np.pi]
        upper = [np.pi, np.pi, *([0.96] if sign == 1 else [-0.033]), np.pi, np.pi, np.pi]
        initial = np.array([-.5, .3, .5, -.5, .1, -.5]) * sign

        def residual(values):
            data.qpos[ids] = values
            mujoco.mj_kinematics(model, data)
            wheel = data.body(prefix + "wheel").xpos
            return np.r_[_leg_constraints(model, data, side),
                         wheel[0] - body_x, wheel[2] - (wheel_radius - base_height)]

        result = least_squares(residual, initial, bounds=(lower, upper),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
        if not result.success or np.abs(residual(result.x)).max() > 1e-8:
            raise ValueError(f"2027 {side} standing geometry cannot close: {result.message}")
    solve_springs(model, data)
    assert closed_chain_residual(model, data) < 1e-8
    return {model.joint(i).name: float(data.qpos[model.jnt_qposadr[i]])
            for i in range(1, model.njnt)}


def standing_motor_positions():
    pose = standing_pose()
    return tuple(pose[name] for name in ("lf0", "lf00", "rf0", "rf00"))


def create_robot_cfg():
    return EntityCfg(
        spec_fn=load_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=ACTUATED_JOINTS, command_field="effort"),)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0., 0., .262),
            joint_pos={f"^{re.escape(name)}$": value for name, value in standing_pose().items()},
            joint_vel={".*": 0.},
        ),
    )
