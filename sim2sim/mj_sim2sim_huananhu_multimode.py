#!/usr/bin/env python3
"""Independent CPU MuJoCo sim2sim for historical multimode model_3023.

This is deliberately a new entry point rather than an edit to the V11 35-D
replay.  It loads the versioned multimode asset and checkpoint, builds the
39-D actor observation, and fixes the task-only tail to the deployment profile
NORMAL / active / non-ground / non-airborne / non-step:

    mode[28:35]  = [1, 0, 0, 0, 0, 0, 0]
    flags[35:39] = [0, 1, 0, 0]

Scope is flat, zero-speed, 0.30 m static standing.  It does not emulate H7,
perform learning, or claim a hardware result.
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
CHECKPOINT_SHA256 = "624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a"
MODEL_ASSET = ROOT / "resources" / "robots" / "chuanliantui_new_1" / "multimode_model3023" / "chuanliantui.xml"
POSE_BANK = MODEL_ASSET.with_name("huananhu_pose_bank.npz")

MOTORS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
OBS_ORDER = np.array((0, 1, 3, 4, 2, 5), dtype=np.intp)
LEGS = np.array((0, 1, 3, 4), dtype=np.intp)
WHEELS = np.array((2, 5), dtype=np.intp)
NORMAL_MODE = np.array((1., 0., 0., 0., 0., 0., 0.), dtype=np.float32)
NORMAL_FLAGS = np.array((0., 1., 0., 0.), dtype=np.float32)
GROUND_RECOVER_MODE = np.array((0., 0., 0., 1., 0., 0., 0.), dtype=np.float32)
DT, DECIMATION = .002, 10
COMMAND = np.array((0., 0., .30), dtype=np.float32)
HELP = "Home: reset current pose | R: random ground pose | F1: pause | close window: exit"
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


def load_actor(path: Path) -> Actor:
    if sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint is not the expected historical model_3023")
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


def build_model():
    """Compile the versioned multimode XML with an in-memory flat floor."""
    if not MODEL_ASSET.is_file():
        raise FileNotFoundError(f"Multimode XML missing: {MODEL_ASSET}")
    spec = mujoco.MjSpec.from_file(str(MODEL_ASSET))
    floor = spec.worldbody.add_geom(
        name="sim2sim_flat_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(20., 20., .1), friction=(1., .005, .0001), priority=0,
    )
    floor.rgba = (.16, .20, .25, 1.)
    model = spec.compile()
    model.opt.timestep = DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.tolerance = 1e-10
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10.
    return model


class Playback:
    def __init__(self, initial_pose: str = "normal"):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
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
        self.reference = self.model.qpos0[self.q].astype(np.float32)
        self.steps = 0
        self.last_action = np.zeros(6, dtype=np.float32)
        self.initial_pose = initial_pose
        self.ground_start = False
        self.recovered = False
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
            for name, value in self._solve_standing_joint_positions(self.model).items():
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
        self.stable_time = 0.
        mujoco.mj_forward(self.model, self.data)
        residual = self._closed_chain_residual(self.model, self.data)
        # Pose-bank rows are stored float32; reloading them into MuJoCo can add
        # a few 1e-8 m of closure error after their double-precision generation.
        if residual > 1e-7:
            raise RuntimeError(f"{pose} reset violates historical closed chain: {residual:.3e} m")

    def mode_and_flags(self) -> tuple[np.ndarray, np.ndarray]:
        """Match the historical ground-start recovery gate at the control rate."""
        policy_active = not self.ground_start or self.steps >= 10  # 0.20 s at 50 Hz
        mode = GROUND_RECOVER_MODE if self.ground_start and not self.recovered else NORMAL_MODE
        return mode, np.array((self.ground_start, policy_active, 0., 0.), dtype=np.float32)

    def observation(self) -> np.ndarray:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array((0., 0., -1.))
        q = self.data.qpos[self.q].astype(np.float32) - self.reference
        q[WHEELS] = 0.
        obs = np.concatenate((
            (COMMAND[0], 0., COMMAND[1], np.clip(COMMAND[2], 0., 1.) * 5.),
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
        target = np.clip(self.reference[LEGS] + .5 * action[LEGS],
                         self.reference[LEGS] - 3.14, self.reference[LEGS] + 3.14)
        torque = np.zeros(6, dtype=np.float64)
        torque[LEGS] = 60. * (target - q[LEGS]) - 2. * velocity[LEGS]
        target_velocity = np.clip(10. * action[WHEELS], -150., 150.)
        torque[WHEELS] = .12 * (target_velocity - velocity[WHEELS]) - .02 * velocity[WHEELS]
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
        for _ in range(DECIMATION):
            torque, spring = self.controls(action)
            self.data.ctrl[self.motor_ids] = torque
            self.data.ctrl[self.spring_ids] = spring
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        row = self.metrics()
        stable = (not row["base_contact"] and max(map(abs, row["roll_pitch_deg"])) < 40.
                  and row["height_m"] > .20)
        self.stable_time = self.stable_time + DT * DECIMATION if stable else 0.
        self.recovered |= self.ground_start and self.stable_time >= .5
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite MuJoCo state")
        if max(self.data.warning.number):
            raise FloatingPointError("MuJoCo reported a simulation warning")
        return action, torque

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
        return {
            "time_s": float(self.data.time), "height_m": float(self.data.qpos[2]),
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
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--initial-pose", choices=("normal", "ground"), default="normal",
                        help="historical normal reset or a deterministic legal ground-start pose")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--onnx", type=Path,
                        help="fixed batch=1 ONNX controller produced by the export tool")
    parser.add_argument("--pytorch-controller", action="store_true",
                        help="use the checkpoint actor for control while retaining the same CPU physics")
    args = parser.parse_args()
    if args.seconds < 0 or not np.isfinite(args.seconds) or (args.seconds == 0 and not args.render):
        parser.error("--seconds must be positive, or 0 together with --render")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    if args.pytorch_controller and args.onnx is not None:
        parser.error("--pytorch-controller cannot be combined with --onnx")
    if not args.pytorch_controller and args.onnx is None:
        parser.error("provide --onnx, or use --pytorch-controller for a reference-only replay")
    torch.set_num_threads(1)
    actor, env = load_actor(args.checkpoint), Playback(args.initial_pose)
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
    print(f"multimode model_3023 | independent CPU MuJoCo | {args.initial_pose} start", flush=True)
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
                if key == 268:  # Home
                    env.reset()
                elif key in (82, 114):  # R/r
                    env.reset("ground")
                elif key == 290:  # F1
                    paused = not paused
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
                    f"model_3023 CPU sim2sim | {'PAUSED' if paused else 'RUNNING'}\\n"
                    f"height {row['height_m']:.3f} / 0.300 m\\n{HELP}", "")])
                viewer.sync()
                time.sleep(max(0., DT * DECIMATION - (time.monotonic() - started)))
        completed = args.seconds == 0 or total_steps * DT * DECIMATION >= args.seconds
    finally:
        if viewer:
            viewer.close()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            stable = [row for row in rows if row["time_s"] >= 1.0]
            height_error = [abs(row["height_m"] - float(COMMAND[2])) for row in stable]
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
                "model_asset": str(MODEL_ASSET), "model_asset_sha256": sha256(MODEL_ASSET),
                "observation": observation_profile,
                "acceptance_after_1s": acceptance,
                "trajectory": rows,
            }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
