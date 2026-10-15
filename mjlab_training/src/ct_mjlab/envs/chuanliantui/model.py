"""Model loading and closed-chain initialization for chuanliantui.

The vendored MJCF has four leg-loop and two spring-loop constraints. All passive
coordinates, including spring orientation and extension, are initialized to a
closed pose. Six torque motors are the public interface; two internal slider
motors apply the compression-dependent gas-spring forces.
"""

from __future__ import annotations

from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .gas_spring import SPRING_FORCE_LIMIT, gas_spring_force

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
MODEL_XML = _PROJECT_ROOT / "resources" / "robots" / "chuanliantui_new_1" / "mjlab" / "chuanliantui.xml"

# Logical sides follow the body frame: left is +Y, right is -Y.
# Mesh asset names retain their original CAD provenance, opposite these names.
ACTUATED_JOINTS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
GAS_SPRING_ACTUATORS = ("left_gas_spring_motor", "right_gas_spring_motor")
GAS_SPRING_JOINTS = ("left_spring2_joint", "right_spring2_joint")
_EXPECTED_JOINTS = (
    "floating_base",
    "lf0",
    "lf1",
    "lfwheel",
    "left_spring1_joint",
    "left_spring2_joint",
    "rf0",
    "rf1",
    "rfwheel",
    "right_spring1_joint",
    "right_spring2_joint",
    "lf00",
    "lf01",
    "lf02",
    "lf03",
    "rf00",
    "rf01",
    "rf02",
    "rf03",
)
_EXPECTED_ACTUATORS = (
    "lf0_motor",
    "lf00_motor",
    "lfwheel_motor",
    "rf0_motor",
    "rf00_motor",
    "rfwheel_motor",
    *GAS_SPRING_ACTUATORS,
)
_EXPECTED_CTRL_RANGES = np.array(
    ((-40.0, 40.0), (-40.0, 40.0), (-3.9, 3.9),
     (-40.0, 40.0), (-40.0, 40.0), (-3.9, 3.9),
     (0.0, SPRING_FORCE_LIMIT), (0.0, SPRING_FORCE_LIMIT)),
    dtype=np.float64,
)


def load_spec() -> mujoco.MjSpec:
    """Load a fresh MJCF specification without sharing mutable state."""
    if not MODEL_XML.is_file():
        raise FileNotFoundError(f"Vendored chuanliantui model is missing: {MODEL_XML}")
    return mujoco.MjSpec.from_file(str(MODEL_XML))


def load_model() -> mujoco.MjModel:
    """Compile a fresh MuJoCo model with the environment's physics settings."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    return _configure_physics(model)


def load_view_model() -> mujoco.MjModel:
    """Compile a fixed-base robot with a viewer-only floor for inspection.

    The robot MJCF deliberately contains no terrain: mjlab supplies it through
    ``SceneCfg``.  The standalone native viewer needs its own floor and a
    fixed base, so both are applied to a temporary specification instead of
    changing the free-base robot asset.
    """
    spec = load_spec()
    floating_base = next(joint for joint in spec.joints if joint.name == "floating_base")
    spec.delete(floating_base)
    floor = spec.worldbody.add_geom(
        name="viewer_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(20.0, 20.0, 0.1),
        friction=(0.5, 0.005, 0.0001),
    )
    floor.rgba = (0.12, 0.16, 0.20, 1.0)
    return _configure_physics(spec.compile())


def _configure_physics(model: mujoco.MjModel) -> mujoco.MjModel:
    """Apply the shared direct-MuJoCo physics settings to a compiled model."""
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.tolerance = 1e-10
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10.0
    return model


def validate_model_contract(model: mujoco.MjModel, *, fixed_base: bool = False) -> None:
    """Reject accidental MJCF changes that break the physical/control contract."""
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    )
    expected_joints = _EXPECTED_JOINTS[1:] if fixed_base else _EXPECTED_JOINTS
    if names != expected_joints:
        raise RuntimeError(f"Unexpected joint contract: {names}")
    if model.neq != 6 or model.ntendon != 0:
        raise RuntimeError(f"Expected six connect constraints and no tendons, got {model.neq}/{model.ntendon}")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    if actuator_names != _EXPECTED_ACTUATORS:
        raise RuntimeError(f"Unexpected actuator contract: {actuator_names}")
    if not np.allclose(model.actuator_ctrlrange, _EXPECTED_CTRL_RANGES, atol=1e-12):
        raise RuntimeError(
            "Unexpected actuator control ranges: "
            f"{model.actuator_ctrlrange.tolist()}"
        )
    for actuator_id, joint_name in zip(gas_spring_actuator_ids(model), GAS_SPRING_JOINTS, strict=True):
        joint_id = model.joint(joint_name).id
        if (model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE
                or model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[actuator_id, 0] != joint_id):
            raise RuntimeError(f"Gas spring actuator must drive slider {joint_name}")
        if (not model.actuator_ctrllimited[actuator_id]
                or not model.actuator_forcelimited[actuator_id]
                or not np.allclose(model.actuator_forcerange[actuator_id], (0.0, SPRING_FORCE_LIMIT))
                or not np.allclose(model.actuator_gear[actuator_id], (1., 0., 0., 0., 0., 0.))):
            raise RuntimeError(f"Gas spring {joint_name} must enforce 0–{SPRING_FORCE_LIMIT:g} N thrust")


def gas_spring_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    """Return the two internal gas-spring actuators in a stable left/right order."""
    ids = []
    for expected_name in GAS_SPRING_ACTUATORS:
        matching_ids = [
            actuator_id
            for actuator_id in range(model.nu)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
            and name.rsplit("/", maxsplit=1)[-1] == expected_name
        ]
        if len(matching_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one gas-spring actuator named '{expected_name}', "
                f"found {matching_ids}"
            )
        ids.append(matching_ids[0])
    return np.asarray(ids, dtype=np.int32)


def apply_gas_spring_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Apply nominal (unrandomized) spring effort in a native MuJoCo simulation."""
    actuator_ids = gas_spring_actuator_ids(model)
    joint_ids = model.actuator_trnid[actuator_ids, 0]
    force = gas_spring_force(data.qpos[model.jnt_qposadr[joint_ids]], data.qvel[model.jnt_dofadr[joint_ids]])
    data.ctrl[actuator_ids] = force
    return force


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"Model is missing joint '{name}'")
    return int(model.jnt_qposadr[joint_id])


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Model is missing closed-chain site '{name}'")
    return site_id


_SIDES = (
    (
        "left",
        "lf0",
        "lf1",
        ("lf00", "lf01", "lf02", "lf03"),
        (
            ("left_rear_pin1", "left_front_pin1"),
            ("left_rear_pin2", "left_front_pin2"),
        ),
        (0.06, -0.10),
    ),
    (
        "right",
        "rf0",
        "rf1",
        ("rf00", "rf01", "rf02", "rf03"),
        (
            ("right_rear_pin1", "right_front_pin1"),
            ("right_rear_pin2", "right_front_pin2"),
        ),
        (-0.06, 0.10),
    ),
)


def closed_chain_residual(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return the largest distance between any paired closed-chain sites."""
    residual = 0.0
    for _, _, _, _, pairs, _ in _SIDES:
        for rear_name, front_name in pairs:
            delta = data.site_xpos[_site_id(model, rear_name)] - data.site_xpos[
                _site_id(model, front_name)
            ]
            residual = max(residual, float(np.linalg.norm(delta)))
    for side in ("left", "right"):
        delta = data.site_xpos[_site_id(model, f"{side}_spring_tip")] - data.site_xpos[
            _site_id(model, f"{side}_gas_spring_lower")
        ]
        residual = max(residual, float(np.linalg.norm(delta)))
    return residual


def solve_standing_joint_positions(
    model: mujoco.MjModel | None = None,
    *,
    front_joint_targets: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Solve a closed pose, returning the model's current joint coordinates.

    ``front_joint_targets`` and the optimizer variables are CAD-relative angles
    (q - qpos0). This preserves the specified physical micro-squat/standup pose
    when a motor's angular zero is redefined with MJCF ref.
    The rear-chain joints are solved independently per side so all four physical
    pin pairs coincide before the first simulation step.
    """
    model = model or load_model()
    validate_model_contract(model)
    joint_positions = {
        name: float(model.qpos0[_joint_qpos_address(model, name)])
        for name in _EXPECTED_JOINTS[1:]
    }

    for side_name, front_hip, front_knee, passive, pairs, default_targets in _SIDES:
        targets = (
            front_joint_targets[side_name]
            if front_joint_targets is not None
            else default_targets
        )
        data = mujoco.MjData(model)
        hip_address = _joint_qpos_address(model, front_hip)
        knee_address = _joint_qpos_address(model, front_knee)
        passive_addresses = [_joint_qpos_address(model, name) for name in passive]
        front_addresses = [hip_address, knee_address]
        data.qpos[front_addresses] = model.qpos0[front_addresses] + targets
        passive_reference = model.qpos0[passive_addresses]

        def residual(values: np.ndarray) -> np.ndarray:
            data.qpos[passive_addresses] = passive_reference + values
            mujoco.mj_forward(model, data)
            return np.concatenate(
                [
                    data.site_xpos[_site_id(model, rear)]
                    - data.site_xpos[_site_id(model, front)]
                    for rear, front in pairs
                ]
            )

        result = least_squares(
            residual,
            x0=np.zeros(len(passive)),
            bounds=(-np.pi, np.pi),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=2_000,
        )
        error = float(np.max(np.abs(residual(result.x))))
        if not result.success or error > 1e-8:
            raise RuntimeError(
                f"Failed to solve {side_name} closed-chain stance: "
                f"success={result.success}, residual={error:.3g}, message={result.message}"
            )
        joint_positions[front_hip], joint_positions[front_knee] = data.qpos[front_addresses]
        joint_positions.update(dict(zip(passive, passive_reference + result.x, strict=True)))

    verification_data = mujoco.MjData(model)
    for name, value in joint_positions.items():
        verification_data.qpos[_joint_qpos_address(model, name)] = value
    mujoco.mj_forward(model, verification_data)
    # Each spring has a fixed transverse eye offset and extends in the leg's XZ
    # plane. Solve its hinge and slide analytically after the main linkage IK.
    for side, parent_name in (("left", "lf0"), ("right", "rf0")):
        upper = verification_data.site(f"{side}_gas_spring_upper").xpos
        lower = verification_data.site(f"{side}_gas_spring_lower").xpos
        delta = verification_data.body(parent_name).xmat.reshape(3, 3).T @ (lower - upper)
        reference_quat = model.body(f"{side}_spring1").quat
        reference_angle = 2 * np.arctan2(reference_quat[2], reference_quat[0])
        reference_length = model.body(f"{side}_spring2").pos[2]
        values = (np.arctan2(delta[0], delta[2]) - reference_angle,
                  np.hypot(delta[0], delta[2]) - reference_length)
        for suffix, value in zip(("spring1_joint", "spring2_joint"), values, strict=True):
            name = f"{side}_{suffix}"
            joint_positions[name] = float(value)
            verification_data.qpos[_joint_qpos_address(model, name)] = value
    mujoco.mj_forward(model, verification_data)
    residual = closed_chain_residual(model, verification_data)
    if residual > 1e-8:
        raise RuntimeError(f"Solved stance violates closed-chain constraint: {residual:.3g} m")
    # mjlab infers the reset tensor dtype from these configuration values;
    # NumPy scalar angles would turn it into float64 instead of simulator float32.
    return {name: float(value) for name, value in joint_positions.items()}


def standing_motor_positions() -> tuple[float, float, float, float]:
    """Micro-squat targets for the four physical leg motors, left then right."""
    positions = solve_standing_joint_positions()
    return tuple(float(positions[name]) for name in ("lf0", "lf00", "rf0", "rf00"))


def create_robot_cfg(
    *,
    base_height: float = 0.3376,
    front_joint_targets: dict[str, tuple[float, float]] | None = None,
) -> EntityCfg:
    """Build the entity while exposing only the six user-facing torque motors."""
    model = load_model()
    return EntityCfg(
        spec_fn=load_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=ACTUATED_JOINTS),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, base_height),
            # mjlab resolves these keys with re.match: an unanchored "rf0"
            # also matches rf00/rf01/... and destroys the solved closed pose.
            joint_pos={
                f"^{re.escape(name)}$": value
                for name, value in solve_standing_joint_positions(
                    model, front_joint_targets=front_joint_targets
                ).items()
            },
            joint_vel={".*": 0.0},
        ),
    )
