#!/usr/bin/env python3
"""Independent CPU MuJoCo sim2sim for multimode 39-D checkpoints.

This is deliberately a new entry point rather than an edit to the V11 35-D
replay.  It loads the versioned multimode asset and any checkpoint that declares
the 39-D multimode contracts, builds the 39-D actor observation, and fixes the
task-only tail to the deployment profile NORMAL / active / non-ground /
non-airborne / non-step:

    mode[28:35]  = [1, 0, 0, 0, 0, 0, 0]
    flags[35:39] = [0, 1, 0, 0]

Observation and PD targets keep the training action term's centres: the actor sees
leg angles relative to the solved closed standing nominal (XML qpos0 plus the
micro-squat targets) and the leg PD target is built around the same nominal, while
raw XML qpos0 stays only the soft-limit clamp centre.  Contact
parameters mirror the training scene's robot-geom priority patch (play friction:
0.5 sliding on robot-floor contacts).

Scope is flat ground.  `--cmd-vx/--cmd-yaw/--cmd-height` hold a constant command,
so one entry point covers both the historical zero-speed 0.30 m static standing
check and the flat velocity/yaw-rate tracking check; the report carries the
tracking errors it measures.  `--teleop` instead hands those three channels to the
keyboard in the viewer, inside the trained command envelope; it is a viewing mode
and therefore refuses `--report`, which measures one fixed command.  It does not
emulate H7, perform learning, or claim a hardware result.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = ROOT / "artifacts" / "multimode-v1-replay-20261014"
CHECKPOINT = IMPORT_ROOT / "models" / "model_3023.pt"
# The historical model_3023 hash, kept for reference. The check is no longer a hard
# pin: --expect-sha256 pins a checkpoint when a caller wants that, otherwise the
# contract metadata below (39-D observation/experiment/control contracts) is what
# guarantees the graph and the actor body are the multimode ones.
CHECKPOINT_SHA256 = "624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a"
# Training and native playback share one robot XML; the pose bank stays versioned.
MODEL_ASSET = ROOT / "resources" / "robots" / "chuanliantui_new_1" / "mjlab" / "chuanliantui.xml"
POSE_BANK = ROOT / "resources" / "robots" / "chuanliantui_new_1" / "multimode_model3023" / "huananhu_pose_bank.npz"

MOTORS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
OBS_ORDER = np.array((0, 1, 3, 4, 2, 5), dtype=np.intp)
LEGS = np.array((0, 1, 3, 4), dtype=np.intp)
WHEELS = np.array((2, 5), dtype=np.intp)
NORMAL_MODE = np.array((1., 0., 0., 0., 0., 0., 0.), dtype=np.float32)
NORMAL_FLAGS = np.array((0., 1., 0., 0.), dtype=np.float32)
GROUND_RECOVER_MODE = np.array((0., 0., 0., 1., 0., 0., 0.), dtype=np.float32)
DT, DECIMATION = .002, 10
WHEEL_RADIUS = .06  # multimode_mdp.WHEEL_RADIUS; clearance/airborne tests
AIRBORNE_TIME_S = .02  # airborne flips after one clear, unloaded control period
RECOVERY_HEIGHT_M = .20  # H7 uses mean L*cos(theta), not world base height.
DEFAULT_COMMAND = (0., 0., .30)
# Tracking tolerances for the motion report. They mirror the single source of truth in
# scripts/agent/assess_multimode_tracking.py; this script must not import the training
# package, so the values are repeated here and a CPU single-robot replay is indicative
# evidence, never the acceptance itself.
TRACKING_SETTLE_S = 2.0
VX_TOL, YAW_TOL, HEIGHT_TOL, TILT_TOL_DEG = .30, .30, .03, 25.
HELP = "Home/Backspace: reset current pose | R: random ground pose | F1: pause | close window: exit"
# --teleop hands the three command channels to the keyboard. The HUD string stays
# ASCII because the MuJoCo font has no arrow glyphs; terminal prints may use them.
# Bounds mirror the trained command envelope in docs/mjlab-chuanliantui.md:
# body +x velocity ±1.5 m/s, yaw rate ±1.5 rad/s, target height 0.27-0.34 m.
TELEOP_HUD = ("Up/Down vx +-0.1 | Left/Right yaw +-0.2 | PgUp/PgDn or ,/. height +-0.02 | "
              "Space stop | Home/Backspace reset | F1 pause")
TELEOP_PRINT = ("↑/↓ 前进速度 ±0.1 m/s | ←/→ 偏航角速度 ±0.2 rad/s | PgUp/PgDn 或 , / . 高度 ±0.02 m | "
                "空格/回车 停车 | Home/Backspace/Insert/Delete/End 复位 | F1 暂停 | "
                "字母键是 viewer 内置快捷键，勿用")
# Reset aliases: many laptop keyboards have no Home key, so the reset action also sits
# on Backspace (always present) plus Insert/Delete/End, and height gets , / . as a
# PgUp/PgDn substitute. Codes are GLFW key numbers.
TELEOP_RESET_KEYS = (268, 259, 260, 261, 269)   # Home, Backspace, Insert, Delete, End
TELEOP_HEIGHT_UP_KEYS = (266, 46)               # PgUp, .
TELEOP_HEIGHT_DOWN_KEYS = (267, 44)             # PgDn, ,
TELEOP_VX_LIMIT, TELEOP_YAW_LIMIT = 1.5, 1.5
TELEOP_HEIGHT_MIN, TELEOP_HEIGHT_MAX = .27, .34
TELEOP_DVX, TELEOP_DYAW, TELEOP_DHEIGHT = .1, .2, .02
LEFT_JOINTS = ("lf0", "lf1", "lfwheel", "left_spring1_joint", "left_spring2_joint",
               "lf00", "lf01", "lf02", "lf03")
RIGHT_JOINTS = ("rf0", "rf1", "rfwheel", "right_spring1_joint", "right_spring2_joint",
                "rf00", "rf01", "rf02", "rf03")
POSE_JOINTS = LEFT_JOINTS + RIGHT_JOINTS
SPRING_ACTUATORS = ("left_gas_spring_motor", "right_gas_spring_motor")
SPRING_INITIAL_COMPRESSION, SPRING_DAMPING, SPRING_FORCE_LIMIT = .01, 50., 150.
ONNX_ACTION_ATOL = 1e-4


def gas_spring_force(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    compression = np.clip(SPRING_INITIAL_COMPRESSION - position, 0., None)
    elastic = np.clip(compression * (SPRING_FORCE_LIMIT / .09), 0., SPRING_FORCE_LIMIT)
    return np.clip(elastic - SPRING_DAMPING * velocity, 0., SPRING_FORCE_LIMIT)


def gas_spring_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray([model.actuator(name).id for name in SPRING_ACTUATORS], dtype=np.intp)


def closed_chain_residual(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    pairs = (("left_rear_pin1", "left_front_pin1"), ("left_rear_pin2", "left_front_pin2"),
             ("right_rear_pin1", "right_front_pin1"), ("right_rear_pin2", "right_front_pin2"),
             ("left_spring_tip", "left_gas_spring_lower"), ("right_spring_tip", "right_gas_spring_lower"))
    return max(float(np.linalg.norm(data.site(a).xpos - data.site(b).xpos)) for a, b in pairs)


def solve_standing_joint_positions(model: mujoco.MjModel) -> dict[str, float]:
    """Solve the historical model's closed micro-squat pose from its versioned XML."""
    positions = {name: float(model.qpos0[model.joint(name).qposadr[0]]) for name in POSE_JOINTS}
    sides = (("lf0", "lf1", ("lf00", "lf01", "lf02", "lf03"),
              (("left_rear_pin1", "left_front_pin1"), ("left_rear_pin2", "left_front_pin2")), (.06, -.10)),
             ("rf0", "rf1", ("rf00", "rf01", "rf02", "rf03"),
              (("right_rear_pin1", "right_front_pin1"), ("right_rear_pin2", "right_front_pin2")), (-.06, .10)))
    for hip, knee, passive, pairs, target in sides:
        data = mujoco.MjData(model)
        front = [model.joint(name).qposadr[0] for name in (hip, knee)]
        passive_q = [model.joint(name).qposadr[0] for name in passive]
        reference = model.qpos0[passive_q].copy()
        data.qpos[front] = model.qpos0[front] + target
        def residual(values):
            data.qpos[passive_q] = reference + values
            mujoco.mj_forward(model, data)
            return np.concatenate([data.site(rear).xpos - data.site(front_name).xpos
                                   for rear, front_name in pairs])
        result = least_squares(residual, x0=np.zeros(4), bounds=(-np.pi, np.pi),
                               xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=2_000)
        if not result.success or np.max(np.abs(residual(result.x))) > 1e-8:
            raise RuntimeError(f"Unable to solve historical closed-chain stance for {hip}")
        positions[hip], positions[knee] = data.qpos[front]
        positions.update(zip(passive, reference + result.x, strict=True))
    data = mujoco.MjData(model)
    for name, value in positions.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    for side, parent in (("left", "lf0"), ("right", "rf0")):
        delta = data.body(parent).xmat.reshape(3, 3).T @ (
            data.site(f"{side}_gas_spring_lower").xpos - data.site(f"{side}_gas_spring_upper").xpos)
        quat = model.body(f"{side}_spring1").quat
        positions[f"{side}_spring1_joint"] = float(np.arctan2(delta[0], delta[2]) - 2 * np.arctan2(quat[2], quat[0]))
        positions[f"{side}_spring2_joint"] = float(np.hypot(delta[0], delta[2]) - model.body(f"{side}_spring2").pos[2])
        for suffix in ("spring1_joint", "spring2_joint"):
            data.qpos[model.joint(f"{side}_{suffix}").qposadr[0]] = positions[f"{side}_{suffix}"]
    mujoco.mj_forward(model, data)
    if closed_chain_residual(model, data) > 1e-8:
        raise RuntimeError("Solved stance violates closed-chain constraint")
    return positions


def load_pose_bank() -> dict[str, dict[str, np.ndarray]]:
    if not POSE_BANK.is_file():
        raise FileNotFoundError(f"Missing multimode pose bank: {POSE_BANK}")
    with np.load(POSE_BANK, allow_pickle=False) as data:
        return {mode: {side: data[f"{mode}_{side}_pos"].copy() for side in ("left", "right")}
                for mode in ("normal", "positive", "negative")}


class Actor(torch.nn.Module):
    """The deterministic actor body, excluding training-only distribution std."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(39, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 6),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_actor(path: Path, expect_sha256: str | None = None) -> Actor:
    if expect_sha256 is not None and sha256(path) != expect_sha256:
        raise ValueError("Checkpoint does not match --expect-sha256")
    if expect_sha256 is None and sha256(path) == CHECKPOINT_SHA256:
        pass  # historical model_3023
    saved = torch.load(path, map_location="cpu", weights_only=False)
    info = saved.get("infos", {})
    required = {
        "observation_contract": "chuanliantui-huananhu-multimode-39-82-v1",
        "experiment_profile_contract": "chuanliantui-huananhu-multimode-39-82-v1",
        "control_contract": "chuanliantui-direct-motor-pd-physical-spring-0to150n-preload10mm-xzero-reset-leftplusy-rawaction-sourcexml-spring90mm-v10",
    }
    if any(info.get(key) != value for key, value in required.items()):
        raise ValueError("model_3023 checkpoint metadata does not match the multimode contract")
    if tuple(saved["actor_state_dict"]["mlp.0.weight"].shape) != (256, 39):
        raise ValueError("model_3023 actor does not have the required 39-D input")
    actor = Actor().eval()
    actor.load_state_dict({key: value for key, value in saved["actor_state_dict"].items()
                           if key.startswith("mlp.")}, strict=True)
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Non-finite policy parameter")
    return actor


def build_model(model_path: Path = MODEL_ASSET):
    """Compile the versioned multimode XML with an in-memory flat floor."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Multimode XML missing: {model_path}")
    spec = mujoco.MjSpec.from_file(str(model_path))
    floor = spec.worldbody.add_geom(
        name="sim2sim_flat_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(20., 20., .1), friction=(1., .005, .0001), priority=0,
    )
    floor.rgba = (.16, .20, .25, 1.)
    model = spec.compile()
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    # Contact parameters mirror the training scene: huananhu_env_cfg patches every
    # robot geom with GeomCfg(geom_names_expr=('.*',), priority=1), so robot-floor
    # contacts take the robot's own friction (0.5 sliding). With both sides left at
    # priority 0, native MuJoCo mixes equal priorities elementwise (max) and would
    # use the floor's 1.0, i.e. twice the training sliding friction (2026-10-17
    # audit finding; regression-asserted in check_multimode_cpu_sim2sim.py).
    floor_geom_id = model.geom("sim2sim_flat_floor").id
    for geom_id in range(model.ngeom):
        if geom_id != floor_geom_id:
            model.geom_priority[geom_id] = 1
    model.opt.timestep = DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.tolerance = 1e-10
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10.
    # Constraint line-search budget: the training MujocoCfg (velocity_env_cfg.py) lowers
    # it from the mjlab default 50 to 20 and mjlab writes that value into the compiled
    # model (sim.py:138); keep the two reference implementations equal.
    model.opt.ls_iterations = 20
    return model


class Playback:
    def __init__(self, initial_pose: str = "normal", command=DEFAULT_COMMAND, *, model_path: Path = MODEL_ASSET,
                 wheel_friction_compensation=None):
        self.model = build_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.wheel_friction_compensation = wheel_friction_compensation
        self._gas_spring_force = gas_spring_force
        self._closed_chain_residual = closed_chain_residual
        self._solve_standing_joint_positions = solve_standing_joint_positions
        self.q = np.array([self.model.joint(name).qposadr[0] for name in MOTORS], dtype=np.intp)
        self.v = np.array([self.model.joint(name).dofadr[0] for name in MOTORS], dtype=np.intp)
        self.motor_ids = np.array([self.model.actuator(name + "_motor").id for name in MOTORS], dtype=np.intp)
        self.spring_ids = gas_spring_actuator_ids(self.model)
        spring_joints = self.model.actuator_trnid[self.spring_ids, 0]
        self.spring_q = self.model.jnt_qposadr[spring_joints]
        self.spring_v = self.model.jnt_dofadr[spring_joints]
        self.floor_geom = self.model.geom("sim2sim_flat_floor").id
        self.base_body = self.model.body("base_link").id
        self.wheel_bodies = tuple(self.model.body(name).id for name in ("lfwheel", "rfwheel"))
        self.hip_joints = tuple(self.model.joint(name).id for name in ("lf0", "rf0"))
        self.reference = self.model.qpos0[self.q].astype(np.float32)
        # Observation and PD-target centre: the training action term keeps the four leg
        # motors centred on the solved closed standing pose (nominal_leg_positions =
        # model.standing_motor_positions(), i.e. XML qpos0 plus the micro-squat targets),
        # and the actor observation subtracts that same nominal
        # (v14_observations.raw_sensors). Raw XML qpos0 stays only the soft-limit clamp
        # centre (pd_action._reference_pos). Using qpos0 for both, as this entry first
        # did, feeds the actor a constant ~0.06 rad leg-angle bias and holds a wrong
        # stance; the two centres must stay split exactly like the training side.
        self.stance = self._solve_standing_joint_positions(self.model)
        self.nominal = np.array([self.stance[name] for name in MOTORS], dtype=np.float32)
        self.command = np.asarray(command, dtype=np.float32)
        if self.command.shape != (3,):
            raise ValueError(f"Command must have three entries, got {self.command.shape}")
        self.steps = 0
        self.last_action = np.zeros(6, dtype=np.float32)
        self.initial_pose = initial_pose
        self.ground_start = False
        self.recovered = False
        self.airborne_time = 0.
        self.stable_time = 0.
        self._pose_bank = None
        self.rng = np.random.default_rng()
        self.reset(initial_pose)

    def reset(self, pose: str | None = None):
        pose = self.initial_pose if pose is None else pose
        if pose not in ("normal", "ground"):
            raise ValueError(f"Unknown initial pose: {pose}")
        mujoco.mj_resetData(self.model, self.data)
        if pose == "normal":
            # The historical normal reset height is 0.38 m.  Use its exact
            # closed-chain solver for every passive coordinate rather than setting
            # a six-motor approximation.
            self.data.qpos[:7] = (0., 0., .38, 1., 0., 0., 0.)
            for name, value in self.stance.items():
                self.data.qpos[self.model.joint(name).qposadr[0]] = value
        else:
            # Independently sample each leg's historical ground-reset branch:
            # positive/negative at the historical 0.6/0.4 probability.  The viewer
            # deliberately keeps initial speeds at zero so each reset is readable.
            if self._pose_bank is None:
                self._pose_bank = load_pose_bank()
            self.data.qpos[:7] = (0., 0., .25, 1., 0., 0., 0.)
            for side, names in (("left", LEFT_JOINTS), ("right", RIGHT_JOINTS)):
                mode = "positive" if self.rng.random() < .6 else "negative"
                rows = self._pose_bank[mode][side]
                joint_pose = rows[self.rng.integers(len(rows))]
                for name, value in zip(names, joint_pose, strict=True):
                    self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.steps = 0
        self.last_action.fill(0.)
        self.ground_start = pose == "ground"
        self.recovered = False
        self.airborne_time = 0.
        self.stable_time = 0.
        mujoco.mj_forward(self.model, self.data)
        residual = self._closed_chain_residual(self.model, self.data)
        # Pose-bank rows are stored float32; reloading them into MuJoCo can add
        # a few 1e-8 m of closure error after their double-precision generation.
        if residual > 1e-7:
            raise RuntimeError(f"{pose} reset violates historical closed chain: {residual:.3e} m")

    def mode_and_flags(self) -> tuple[np.ndarray, np.ndarray]:
        """Match the historical ground-start recovery gate at the control rate."""
        # Ground-start activation at 0.20 s, computed in float32 exactly like
        # MultimodeCommand.policy_active (episode_length_buf * step_dt >= .2): the float32
        # product leaves 10 * .02 just below .2, so the flag flips on the 11th completed
        # step, not the 10th. The torque hold in controls() stays the integer 10-step
        # gate (`steps < 10`), matching huananhu_action._ground_hold.
        policy_active = bool(not self.ground_start) or bool(
            np.float32(self.steps) * np.float32(DT * DECIMATION) >= np.float32(.2))
        mode = GROUND_RECOVER_MODE if self.ground_start and not self.recovered else NORMAL_MODE
        # The airborne flag (slot 37) follows multimode_mdp.MultimodeCommand.advance
        # and is not a fixed 0 (2026-10-17 second audit); step() maintains
        # airborne_time. step_active stays 0 on this flat-only task.
        return mode, np.array((self.ground_start, policy_active,
                               self.airborne_time >= AIRBORNE_TIME_S, 0.), dtype=np.float32)

    def observation(self) -> np.ndarray:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array((0., 0., -1.))
        q = self.data.qpos[self.q].astype(np.float32) - self.nominal
        q[WHEELS] = 0.
        obs = np.concatenate((
            (np.clip(self.command[0], -100., 100.), 0., np.clip(self.command[1], -100., 100.),
             np.clip(self.command[2], 0., 1.) * 5.),  # v14_observations.basic clamps vx/yaw to +-100
            np.clip(self.data.qvel[3:6], -100., 100.) * .5,
            np.clip(gravity, -100., 100.),
            np.clip(q[OBS_ORDER], -100., 100.),
            np.clip(self.data.qvel[self.v][OBS_ORDER], -200., 200.) * .1,
            np.clip(self.last_action[OBS_ORDER], -100., 100.),
            *self.mode_and_flags(),
        )).astype(np.float32)
        if obs.shape != (39,) or not np.isfinite(obs).all():
            raise FloatingPointError("Non-finite or malformed 39-D observation")
        return obs

    def controls(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q, velocity = self.data.qpos[self.q], self.data.qvel[self.v]
        target = np.clip(self.nominal[LEGS] + .5 * action[LEGS],
                         self.reference[LEGS] - 3.14, self.reference[LEGS] + 3.14)
        torque = np.zeros(6, dtype=np.float64)
        torque[LEGS] = 60. * (target - q[LEGS]) - 2. * velocity[LEGS]
        target_velocity = np.clip(10. * action[WHEELS], -150., 150.)
        # Wheel velocity loop with the gains huananhu_env_cfg configures: wheel_stiffness=.1,
        # wheel_damping=0. The 0.12/0.02 pair belongs to the older chuanliantui task.
        torque[WHEELS] = .1 * (target_velocity - velocity[WHEELS])
        if self.wheel_friction_compensation is not None:
            # Experimental wheel loss compensation, smooth through zero speed.
            torque[WHEELS] += self.wheel_friction_compensation * np.tanh(velocity[WHEELS] / .5)
        torque = np.clip(torque, (-40., -40., -3.9, -40., -40., -3.9), (40., 40., 3.9, 40., 40., 3.9))
        if self.ground_start and self.steps < 10:
            torque.fill(0.)
        spring = self._gas_spring_force(self.data.qpos[self.spring_q], self.data.qvel[self.spring_v])
        return torque, np.asarray(spring, dtype=np.float64)

    def step(self, controller) -> tuple[np.ndarray, np.ndarray]:
        obs = self.observation()
        action = controller(obs)
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite actor output")
        self.last_action = action.astype(np.float32, copy=True)
        torque = None
        wheel_forces = np.zeros(2)
        force_buffer = np.zeros(6)
        for _ in range(DECIMATION):
            torque, spring = self.controls(action)
            self.data.ctrl[self.motor_ids] = torque
            self.data.ctrl[self.spring_ids] = spring
            mujoco.mj_step(self.model, self.data)
            substep_forces = np.zeros(2)
            for contact_id in range(self.data.ncon):
                contact = self.data.contact[contact_id]
                bodies = (self.model.geom_bodyid[contact.geom1],
                          self.model.geom_bodyid[contact.geom2])
                for index, body in enumerate(self.wheel_bodies):
                    if body in bodies:
                        mujoco.mj_contactForce(self.model, self.data, contact_id,
                                               force_buffer)
                        substep_forces[index] += float(np.linalg.norm(force_buffer[:3]))
            wheel_forces = np.maximum(wheel_forces, substep_forces)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        # Training's airborne flag (multimode_mdp.MultimodeCommand.advance): both
        # wheel centers clear of the flat floor by more than WHEEL_RADIUS + 2 cm
        # with the peak wheel contact force over the control step below 5 N, held
        # for at least one control period before the flag flips. The 2026-10-17
        # second audit found the entry had pinned this flag to 0.
        clear = all(self.data.xpos[body][2] > WHEEL_RADIUS + .02
                    for body in self.wheel_bodies)
        self.airborne_time = (self.airborne_time + DT * DECIMATION
                              if clear and bool((wheel_forces < 5.).all()) else 0.)
        row = self.metrics()
        stable = (not row["base_contact"] and max(map(abs, row["roll_pitch_deg"])) < 40.
                  and row["hip_wheel_height_m"] > RECOVERY_HEIGHT_M)
        self.stable_time = self.stable_time + DT * DECIMATION if stable else 0.
        self.recovered |= self.ground_start and self.stable_time >= .5
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite MuJoCo state")
        if max(self.data.warning.number):
            raise FloatingPointError("MuJoCo reported a simulation warning")
        return action, torque

    def hip_wheel_height(self) -> float:
        """Mean leg-plane L*cos(theta), in the body frame as in H7's estimate.

        Use each hip pivot and wheel axle, excluding the lateral axle offset.
        World base height, wheel radius and ground contact do not enter this value.
        """
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        heights = []
        for hip, wheel in zip(self.hip_joints, self.wheel_bodies, strict=True):
            leg = rotation.T @ (self.data.xpos[wheel] - self.data.xanchor[hip])
            length = np.hypot(leg[0], leg[2])
            angle = np.arctan2(leg[0], -leg[2])  # Leg angle relative to body-down.
            heights.append(length * np.cos(angle))
        return float(np.mean(heights))

    def body_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Body-frame linear and angular velocity of base_link.

        mjOBJ_XBODY with flg_local=1 gives the body frame; mjOBJ_BODY would return the
        inertial-principal frame, whose axes are permuted once the base inertia
        eigenvalues are sorted descending. That trap is recorded in docs/ai/sim2sim.md
        and must not be reintroduced here.
        """
        result = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 self.base_body, result, 1)
        # MuJoCo orders this result rotational-first: a pure translation (qvel=(1.5,0,0,0,0,0))
        # returns [0,0,0, 1.5,0,0]. sim2sim/mj_sim2sim.py indexes it the same way. Reading
        # result[:3] as linear would report the roll rate as vx and the vertical speed as the
        # yaw rate, which is exactly the bug this comment exists to prevent.
        return result[3:], result[:3]

    def metrics(self, action: np.ndarray | None = None, torque: np.ndarray | None = None) -> dict:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array((0., 0., -1.))
        base_contact, wheel_contact = False, [False, False]
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            bodies = (self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2])
            if self.base_body in bodies:
                base_contact = True
            for index, body in enumerate(self.wheel_bodies):
                wheel_contact[index] |= body in bodies and self.floor_geom in (contact.geom1, contact.geom2)
        linear, angular = self.body_velocity()
        return {
            "time_s": float(self.data.time), "height_m": float(self.data.qpos[2]),
            "hip_wheel_height_m": self.hip_wheel_height(),
            "x_m": float(self.data.qpos[0]), "y_m": float(self.data.qpos[1]),
            "vx_m_s": float(linear[0]), "vy_m_s": float(linear[1]),
            "yaw_rate_rad_s": float(angular[2]),
            "roll_pitch_deg": [float(np.degrees(np.arctan2(-gravity[1], -gravity[2]))),
                               float(np.degrees(np.arcsin(np.clip(gravity[0], -1., 1.))))],
            "closed_chain_residual_m": float(self._closed_chain_residual(self.model, self.data)),
            "base_contact": base_contact, "both_wheels_contact": bool(all(wheel_contact)),
            "action": None if action is None else action.tolist(),
            "torque_nm": None if torque is None else torque.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--model", type=Path, default=MODEL_ASSET,
                        help="MuJoCo XML; use a separate real2sim candidate for response comparisons")
    parser.add_argument("--wheel-friction-compensation", type=Path,
                        help="experimental: read the two dry-friction values from real2sim candidate-parameters.json")
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--initial-pose", choices=("normal", "ground"), default="normal",
                        help="historical normal reset or a deterministic legal ground-start pose")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--onnx", type=Path,
                        help="fixed batch=1 ONNX controller produced by the export tool")
    parser.add_argument("--pytorch-controller", action="store_true",
                        help="use the checkpoint actor for control while retaining the same CPU physics")
    parser.add_argument("--cmd-vx", type=float, default=DEFAULT_COMMAND[0],
                        help="constant body +x velocity command [m/s]")
    parser.add_argument("--cmd-yaw", type=float, default=DEFAULT_COMMAND[1],
                        help="constant yaw-rate command [rad/s]")
    parser.add_argument("--cmd-height", type=float, default=DEFAULT_COMMAND[2],
                        help="constant target height [m]")
    parser.add_argument("--expect-sha256", default=None,
                        help="pin the checkpoint hash; omit to accept any checkpoint declaring the "
                             "39-D multimode contracts")
    parser.add_argument("--teleop", action="store_true",
                        help="drive vx/yaw-rate/height from the keyboard in the viewer; needs "
                             "--render and cannot be combined with --report")
    args = parser.parse_args()
    if args.seconds < 0 or not np.isfinite(args.seconds) or (args.seconds == 0 and not args.render):
        parser.error("--seconds must be positive, or 0 together with --render")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    if args.pytorch_controller and args.onnx is not None:
        parser.error("--pytorch-controller cannot be combined with --onnx")
    if not args.pytorch_controller and args.onnx is None:
        parser.error("provide --onnx, or use --pytorch-controller for a reference-only replay")
    if args.teleop and args.report:
        parser.error("--teleop cannot be combined with --report: the report measures one fixed command")
    if args.teleop and not args.render:
        parser.error("--teleop needs --render: keys only reach the process through the viewer")
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint, args.expect_sha256)
    friction = None
    if args.wheel_friction_compensation:
        parameters = json.loads(args.wheel_friction_compensation.read_text())
        values = np.asarray(parameters['additional_parameters'], dtype=float)
        if values.shape != (2, 3) or not np.isfinite(values).all() or np.any(values < 0):
            parser.error("wheel compensation requires finite nonnegative parameters with shape [2,3]")
        if list(parameters['wheel_order']) != ['lfwheel', 'rfwheel']:
            parser.error("wheel compensation must use [lfwheel, rfwheel] order")
        friction = values[:, 2]
    env = Playback(args.initial_pose, (args.cmd_vx, args.cmd_yaw, args.cmd_height), model_path=args.model,
                   wheel_friction_compensation=friction)
    onnx_path, evaluator, action_diff = None, None, [0.0]
    if not args.pytorch_controller:
        import onnx
        from onnx.reference import ReferenceEvaluator
        onnx_path = args.onnx.resolve()
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        input_tensor, output_tensor = model.graph.input[0].type.tensor_type, model.graph.output[0].type.tensor_type
        input_shape = [int(d.dim_value) for d in input_tensor.shape.dim]
        output_shape = [int(d.dim_value) for d in output_tensor.shape.dim]
        if (model.graph.input[0].name, input_tensor.elem_type, input_shape) != ("obs", onnx.TensorProto.FLOAT, [1, 39]):
            raise ValueError("ONNX controller must expose obs: float32[1,39]")
        if (model.graph.output[0].name, output_tensor.elem_type, output_shape) != ("actions", onnx.TensorProto.FLOAT, [1, 6]):
            raise ValueError("ONNX controller must expose actions: float32[1,6]")
        evaluator = ReferenceEvaluator(model)

    def controller(obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            torch_action = actor(torch.from_numpy(obs)[None]).numpy()[0]
        if evaluator is None:
            return torch_action
        onnx_action = evaluator.run(None, {"obs": obs[None]})[0][0]
        action_diff[0] = max(action_diff[0], float(np.abs(torch_action - onnx_action).max()))
        return np.asarray(onnx_action, dtype=np.float32)

    events: deque[int] = deque()
    viewer = None
    paused, completed, rows = False, False, []
    total_steps = 0
    print(f"{args.checkpoint.name} | independent CPU MuJoCo | {args.initial_pose} start"
          f"{' | TELEOP' if args.teleop else ''}", flush=True)
    if args.teleop:
        print("遥操作模式：点击 MuJoCo 窗口获得焦点后按键 —— " + TELEOP_PRINT, flush=True)

    def report_command() -> None:
        print(f"[teleop] vx {env.command[0]:+.1f} m/s  偏航角速度 {env.command[1]:+.1f} rad/s  "
              f"高度 {env.command[2]:.2f} m", flush=True)
    try:
        if args.render:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.append)
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 135., -18.
                viewer.opt.geomgroup[3] = 0
        while args.seconds == 0 or total_steps * DT * DECIMATION < args.seconds:
            started = time.monotonic()
            if viewer and not viewer.is_running():
                break
            while events:
                key = events.popleft()
                if key in TELEOP_RESET_KEYS:  # Home / Backspace / Insert / Delete / End
                    env.reset()
                    if args.teleop:
                        # A reset lands in a standing pose; keeping a nonzero speed
                        # command would drive off immediately, so stop. Height is kept.
                        env.command[0], env.command[1] = 0., 0.
                        report_command()
                elif key in (82, 114):  # R/r
                    env.reset("ground")
                elif key == 290:  # F1
                    paused = not paused
                elif args.teleop and key in (265, 264, 263, 262, 32, 257) + TELEOP_HEIGHT_UP_KEYS + TELEOP_HEIGHT_DOWN_KEYS:
                    if key == 265:    # up
                        env.command[0] = min(env.command[0] + TELEOP_DVX, TELEOP_VX_LIMIT)
                    elif key == 264:  # down
                        env.command[0] = max(env.command[0] - TELEOP_DVX, -TELEOP_VX_LIMIT)
                    elif key == 263:  # left
                        env.command[1] = min(env.command[1] + TELEOP_DYAW, TELEOP_YAW_LIMIT)
                    elif key == 262:  # right
                        env.command[1] = max(env.command[1] - TELEOP_DYAW, -TELEOP_YAW_LIMIT)
                    elif key in TELEOP_HEIGHT_UP_KEYS:    # PgUp / .
                        env.command[2] = min(env.command[2] + TELEOP_DHEIGHT, TELEOP_HEIGHT_MAX)
                    elif key in TELEOP_HEIGHT_DOWN_KEYS:  # PgDn / ,
                        env.command[2] = max(env.command[2] - TELEOP_DHEIGHT, TELEOP_HEIGHT_MIN)
                    else:             # space / enter: stop, height command stays
                        env.command[0], env.command[1] = 0., 0.
                    report_command()
            if not paused:
                if viewer:
                    with viewer.lock():
                        action, torque = env.step(controller)
                else:
                    action, torque = env.step(controller)
                total_steps += 1
                row = env.metrics(action, torque)
                if args.report:
                    rows.append(row)
                if total_steps % 50 == 0:
                    print(json.dumps(row), flush=True)
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = env.data.qpos[:3]
                row = env.metrics()
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    f"{args.checkpoint.name} CPU sim2sim | {'PAUSED' if paused else 'RUNNING'}\n"
                    f"cmd vx {env.command[0]:+.1f} m/s | yaw {env.command[1]:+.1f} rad/s | "
                    f"height {env.command[2]:.3f} m\n"
                    # Measured body-frame velocity from the same mjOBJ_XBODY read the report uses.
                    f"vel vx {row['vx_m_s']:+.2f} m/s | vy {row['vy_m_s']:+.2f} m/s | "
                    f"wz {row['yaw_rate_rad_s']:+.2f} rad/s | "
                    f"|v| {np.hypot(row['vx_m_s'], row['vy_m_s']):.2f} m/s\n"
                    f"base height {row['height_m']:.3f} m | hip-wheel {row['hip_wheel_height_m']:.3f} m\n"
                    f"{TELEOP_HUD if args.teleop else HELP}", "")])
                viewer.sync()
                time.sleep(max(0., DT * DECIMATION - (time.monotonic() - started)))
        completed = args.seconds == 0 or total_steps * DT * DECIMATION >= args.seconds
    finally:
        if viewer:
            viewer.close()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            stable = [row for row in rows if row["time_s"] >= 1.0]
            height_error = [abs(row["height_m"] - float(env.command[2])) for row in stable]
            tilt = [max(abs(value) for value in row["roll_pitch_deg"]) for row in stable]
            acceptance = {
                "samples_after_1s": len(stable),
                "height_error_max_m": max(height_error, default=None),
                "tilt_max_deg": max(tilt, default=None),
                "base_contact_count": sum(row["base_contact"] for row in stable),
                "dual_wheel_contact_count": sum(row["both_wheels_contact"] for row in stable),
                "onnx_vs_torch_action_atol": None if evaluator is None else ONNX_ACTION_ATOL,
                "onnx_vs_torch_action_max_abs_diff": None if evaluator is None else action_diff[0],
                "passed": bool(completed and stable and max(height_error) <= .03 and max(tilt) <= 15.
                               and not any(row["base_contact"] for row in stable)
                               and all(row["both_wheels_contact"] for row in stable)
                               and (evaluator is None or action_diff[0] <= ONNX_ACTION_ATOL)),
            }
            settled = [row for row in rows if row["time_s"] >= TRACKING_SETTLE_S]

            def mean(values):
                return float(np.mean(values)) if values else None

            tracking = {
                "settle_s": TRACKING_SETTLE_S,
                "samples_after_settle": len(settled),
                "command": {"vx_m_s": float(env.command[0]), "yaw_rad_s": float(env.command[1]),
                            "height_m": float(env.command[2])},
                "velocity_definition": "body-frame 6D velocity of base_link via mjOBJ_XBODY (local)",
                "mean_vx_error_m_s": mean([abs(row["vx_m_s"] - env.command[0]) for row in settled]),
                "mean_vx_bias_m_s": mean([row["vx_m_s"] - env.command[0] for row in settled]),
                "mean_yaw_rate_error_rad_s": mean([abs(row["yaw_rate_rad_s"] - env.command[1])
                                                   for row in settled]),
                "mean_yaw_rate_bias_rad_s": mean([row["yaw_rate_rad_s"] - env.command[1]
                                                  for row in settled]),
                "mean_lateral_speed_m_s": mean([abs(row["vy_m_s"]) for row in settled]),
                "mean_height_error_m": mean([abs(row["height_m"] - env.command[2]) for row in settled]),
                "max_tilt_deg": max([max(abs(value) for value in row["roll_pitch_deg"]) for row in settled],
                                    default=None),
                "base_contact_count": sum(row["base_contact"] for row in settled),
                "travel_m": (float(np.hypot(rows[-1]["x_m"] - rows[0]["x_m"],
                                            rows[-1]["y_m"] - rows[0]["y_m"])) if rows else None),
            }
            # Same criteria as the GPU measurement, with no zero-command bypass: a zero
            # command must keep the speed inside the same tolerance rather than skipping the
            # check. A run that did not complete, or an ONNX graph that disagrees with the
            # PyTorch actor, cannot pass.
            tracking["passed"] = bool(
                completed and settled
                and (evaluator is None or action_diff[0] <= ONNX_ACTION_ATOL)
                and tracking["mean_height_error_m"] <= HEIGHT_TOL
                and tracking["max_tilt_deg"] <= TILT_TOL_DEG
                and tracking["base_contact_count"] == 0
                and tracking["mean_vx_error_m_s"] <= VX_TOL
                and tracking["mean_yaw_rate_error_rad_s"] <= YAW_TOL)
            observation_profile = {
                "shape": [39],
                "normal_static": {"mode_28_35": NORMAL_MODE.tolist(), "flags_35_39": NORMAL_FLAGS.tolist()},
            }
            if args.initial_pose == "ground":
                observation_profile["ground_start_prefix"] = {
                    "mode_28_35": GROUND_RECOVER_MODE.tolist(), "flags_35_39": [1., 0., 0., 0.],
                    "policy_activation_s": .2,
                }
            args.report.write_text(json.dumps({
                "backend": "independent-native-mujoco-cpu", "completed": completed,
                "seconds": args.seconds, "physics_dt_s": DT, "control_dt_s": DT * DECIMATION,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                "initial_pose": args.initial_pose,
                "controller": "pytorch" if evaluator is None else "onnx-reference",
                "onnx": None if onnx_path is None else str(onnx_path),
                "onnx_sha256": None if onnx_path is None else sha256(onnx_path),
                "max_onnx_vs_torch_action_diff": None if evaluator is None else action_diff[0],
                "model_asset": str(args.model), "model_asset_sha256": sha256(args.model),
                "wheel_friction_compensation": None if friction is None else {
                    "source": str(args.wheel_friction_compensation),
                    "sha256": sha256(args.wheel_friction_compensation),
                    "coefficient_nm": friction.tolist(), "velocity_smoothing_rad_s": .5,
                    "scope": "experimental controller; not physical calibration or H7 validation"},
                "observation": observation_profile,
                "recovery_height": {
                    "quantity": "body-frame mean leg-plane L*cos(theta)",
                    "threshold_m": RECOVERY_HEIGHT_M,
                    "scope": "ground-start recovery completion only; contact/tilt gates retained",
                },
                "acceptance_after_1s": acceptance,
                "tracking": tracking,
                "trajectory": rows,
            }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
