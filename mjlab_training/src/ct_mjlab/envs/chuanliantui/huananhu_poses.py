"""Offline, physically closed V14-style reset poses for the CAD robot.

The source leg length is mapped by normalized extension to the CAD linkage.
The source swing angle (and, for ground poses, the height/length ratio) is
preserved.  Expensive MuJoCo closure runs only while building the bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import brentq
import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .model import MODEL_XML, closed_chain_residual, load_model, solve_standing_joint_positions


LEFT_JOINTS = (
    "lf0", "lf1", "lfwheel", "left_spring1_joint", "left_spring2_joint",
    "lf00", "lf01", "lf02", "lf03",
)
RIGHT_JOINTS = (
    "rf0", "rf1", "rfwheel", "right_spring1_joint", "right_spring2_joint",
    "rf00", "rf01", "rf02", "rf03",
)
POSE_JOINT_NAMES = LEFT_JOINTS + RIGHT_JOINTS
MODES = ("normal", "positive", "negative")
NORMAL_ROOT_HEIGHT = 0.38
GROUND_ROOT_HEIGHT = 0.25
_TILT = float(np.arccos(np.cos(0.15) ** 2))
_CLEARANCE = 0.002
_KNEE_INTERIOR_MARGIN = 2e-5
_CACHE_VERSION = 4


@dataclass(frozen=True)
class PoseBank:
    """Arrays have shape (count, 9) per side, in LEFT/RIGHT_JOINTS order.

    ``ground_min_z`` is the lowest collision-mesh vertex for that side at a
    0.25 m, level root.  ``normal_tilt_min_z`` is a conservative lower bound
    at zero root height for any roll/pitch in [-0.15, 0.15] radians.
    """

    positions: dict[str, dict[str, np.ndarray]]
    ground_min_z: dict[str, dict[str, np.ndarray]]
    normal_tilt_min_z: dict[str, dict[str, np.ndarray]]
    base_tilt_min_z: float
    generation_stats: dict[str, dict[str, int]]

    def to_torch(self, device: str | torch.device) -> TorchPoseBank:
        return TorchPoseBank(
            {mode: {side: torch.as_tensor(self.positions[mode][side], device=device)
                    for side in ("left", "right")} for mode in MODES},
            {mode: {side: torch.as_tensor(self.normal_tilt_min_z[mode][side], device=device)
                    for side in ("left", "right")} for mode in MODES},
            self.base_tilt_min_z,
        )


@dataclass(frozen=True)
class TorchPoseBank:
    """GPU-resident pose tables; no MuJoCo or CPU transfer on the reset path."""

    positions: dict[str, dict[str, torch.Tensor]]
    normal_tilt_min_z: dict[str, dict[str, torch.Tensor]]
    base_tilt_min_z: float

    def sample(self, mode: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently sample left/right rows; return (joint_pos, root_z).

        Columns follow POSE_JOINT_NAMES.  Ground signs may differ between
        legs: call ``sample_sides(left_mode, right_mode, count)`` for that case.
        """
        return self.sample_sides(mode, mode, count)

    def sample_sides(
        self, left_mode: str, right_mode: str, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if left_mode not in MODES or right_mode not in MODES:
            raise ValueError(f"Unknown V14 reset mode: {left_mode}/{right_mode}")
        if (left_mode == "normal") != (right_mode == "normal"):
            raise ValueError("Normal/ground mode is selected per robot, not per leg")
        left = self.positions[left_mode]["left"]
        right = self.positions[right_mode]["right"]
        left_id = torch.randint(left.shape[0], (count,), device=left.device)
        right_id = torch.randint(right.shape[0], (count,), device=right.device)
        joint_pos = torch.cat((left[left_id], right[right_id]), dim=1)
        if left_mode == right_mode == "normal":
            left_low = self.normal_tilt_min_z["normal"]["left"][left_id]
            right_low = self.normal_tilt_min_z["normal"]["right"][right_id]
            minimum = torch.minimum(left_low, right_low).clamp(max=self.base_tilt_min_z)
            root_z = torch.maximum(torch.full_like(minimum, NORMAL_ROOT_HEIGHT), _CLEARANCE - minimum)
        else:
            root_z = torch.full((count,), GROUND_ROOT_HEIGHT, device=left.device, dtype=left.dtype)
        return joint_pos, root_z


def _mesh_vertices(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    first = int(model.mesh_vertadr[mesh_id])
    last = first + int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[first:last]
    return vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]


def _side_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    prefix = "lf" if side == "left" else "rf"
    return [
        gid for gid in range(model.ngeom)
        if model.geom_group[gid] == 3
        and (model.body(model.geom_bodyid[gid]).name.startswith(prefix))
    ]


def _minimum_geometry_z(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int], *, tilt: bool
) -> float:
    minimum = float("inf")
    for gid in geom_ids:
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Pose-bank clearance requires mesh collision geometry")
        vertices = _mesh_vertices(model, data, gid)
        z = vertices[:, 2]
        if tilt:
            radius = np.linalg.norm(vertices[:, :2], axis=1)
            z = np.minimum(z, z * np.cos(_TILT)) - radius * np.sin(_TILT)
        minimum = min(minimum, float(np.min(z)))
    return minimum


def _wheel_any_spin_min_z(
    model: mujoco.MjModel, data: mujoco.MjData, wheel_body: int,
    geom_ids: list[int], *, tilt: bool,
) -> float:
    """Lower bound on wheel mesh height for every wheel joint angle.

    The axle is transverse (world Y with a level root).  Spinning a wheel
    rotates each vertex's XZ projection through a circle of radius ``rho``.
    Under root roll/pitch, its lateral offset can lower it by at most
    ``abs(y) * sin(max_tilt)``.  This bound covers continuous spin angles,
    including the random [-2*pi, 2*pi] reset angle.
    """
    center = data.xpos[wheel_body]
    worst_radius = 0.0
    for gid in geom_ids:
        vertices = _mesh_vertices(model, data, gid) - center
        rho = np.linalg.norm(vertices[:, [0, 2]], axis=1)
        if tilt:
            rho = rho + np.abs(vertices[:, 1]) * np.sin(_TILT)
        worst_radius = max(worst_radius, float(np.max(rho)))
    z = float(center[2])
    if tilt:
        z = min(z, z * np.cos(_TILT)) - float(np.linalg.norm(center[:2])) * np.sin(_TILT)
    return z - worst_radius


def _front_kinematics(model: mujoco.MjModel, data: mujoco.MjData, knee: float) -> tuple[float, float]:
    data.qpos[:] = model.qpos0
    data.qpos[model.joint("lf1").qposadr[0]] = knee
    mujoco.mj_forward(model, data)
    delta = data.body("lfwheel").xpos - data.body("lf0").xpos
    return float(np.hypot(delta[0], delta[2])), float(np.arctan2(delta[0], -delta[2]))


def pose_length_range(model: mujoco.MjModel | None = None) -> tuple[float, float]:
    """Derive the reset XZ hip-to-wheel range from the XML knee limits.

    The inner 2e-5 rad margin keeps sampled poses away from the joint stops;
    it does not reinterpret a requested physical length as a mechanical limit.
    Other mechanical limits must admit both ends of this range explicitly.
    """
    if model is None:
        model = load_model()
    data = mujoco.MjData(model)
    joint = model.joint("lf1")
    knee_bounds = tuple(float(value) for value in joint.range)
    if (not joint.limited or not np.isfinite(knee_bounds).all()
            or knee_bounds[1] - knee_bounds[0] <= 2 * _KNEE_INTERIOR_MARGIN):
        raise ValueError("Pose generation requires a finite left knee joint range wider than its reset margin")
    lengths = np.asarray([
        _front_kinematics(model, data, value)[0] for value in np.linspace(*knee_bounds, 65)
    ])
    if not np.all(np.diff(lengths) > 1e-10):
        raise ValueError("Left knee joint range must follow the monotonic normal-assembly branch")
    interior = (knee_bounds[0] + _KNEE_INTERIOR_MARGIN, knee_bounds[1] - _KNEE_INTERIOR_MARGIN)
    result = []
    for knee in interior:
        length, phi = _front_kinematics(model, data, knee)
        endpoint = solve_standing_joint_positions(
            model, front_joint_targets={"left": (phi, knee), "right": (-phi, -knee)}
        )
        # An inherited spring stop must not silently discard short-leg samples.
        for name, value in endpoint.items():
            endpoint_joint = model.joint(name)
            if endpoint_joint.limited and not (
                endpoint_joint.range[0] + 1e-5 <= value <= endpoint_joint.range[1] - 1e-5
            ):
                raise ValueError(
                    f"XML-derived leg length {length:.6f} m violates mechanical joint limit "
                    f"{name}: {value:.9f} outside {tuple(endpoint_joint.range)} with reset margin; "
                    "resolve the mechanical limits instead of filtering out this length"
                )
        result.append(length)
    return result[0], result[1]


def _sample_source(
    mode: str, rng: np.random.Generator, length_range: tuple[float, float]
) -> tuple[float, float]:
    if mode == "normal":
        length = rng.uniform(0.13, 0.32)
        angle = rng.uniform(-np.pi / 2, 0.75 * np.pi)
        fraction = (length - 0.13) / (0.32 - 0.13)
    else:
        length = rng.uniform(0.14, 0.36)
        height = rng.uniform(-0.06, 0.12 if mode == "positive" else 0.0)
        angle = (1 if mode == "positive" else -1) * np.arccos(np.clip(height / length, -1 + 1e-6, 1 - 1e-6))
        fraction = (length - 0.14) / (0.36 - 0.14)
    mapped_length = length_range[0] + fraction * (length_range[1] - length_range[0])
    return float(mapped_length), float(angle)


def _cache_signature() -> str:
    """Include collision assets and sampling semantics, not only XML paths."""
    xml_bytes = MODEL_XML.read_bytes()
    root = ET.fromstring(xml_bytes)
    compiler = root.find("compiler")
    mesh_dir = MODEL_XML.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    meshes = [
        (mesh.get("name"), mesh.get("file"),
         hashlib.sha256((mesh_dir / mesh.get("file")).read_bytes()).hexdigest())
        for mesh in root.findall("asset/mesh[@file]")
    ]
    payload = {
        "version": _CACHE_VERSION,
        "model": hashlib.sha256(xml_bytes).hexdigest(),
        "meshes": meshes,
        "knee_interior_margin": _KNEE_INTERIOR_MARGIN,
        "root_heights": (NORMAL_ROOT_HEIGHT, GROUND_ROOT_HEIGHT),
        "tilt": _TILT,
        "clearance": _CLEARANCE,
        "sampling_algorithm": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "closure_algorithm": hashlib.sha256(inspect.getsource(solve_standing_joint_positions).encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_pose_bank(
    count: int = 1024, *, seed: int = 20261005, cache_path: str | Path | None = None
) -> PoseBank:
    """Build or load per-mode, per-side closed pose tables on the CPU.

    Use a cache path in ``assets/chuanliantui`` when training repeatedly.  A
    missing/stale cache is regenerated; no repository file is written unless a
    cache path is explicitly supplied.  At runtime call ``bank.to_torch(device)``.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    path = Path(cache_path) if cache_path is not None else None
    model_hash = hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()
    cache_signature = _cache_signature()
    if path is not None and path.is_file():
        with np.load(path, allow_pickle=False) as f:
            if (int(f["version"]) == _CACHE_VERSION and int(f["count"]) == count
                    and int(f["seed"]) == seed and str(f.get("cache_signature", "")) == cache_signature):
                return PoseBank(
                    {mode: {side: f[f"{mode}_{side}_pos"] for side in ("left", "right")} for mode in MODES},
                    {mode: {side: f[f"{mode}_{side}_ground"] for side in ("left", "right")} for mode in MODES},
                    {mode: {side: f[f"{mode}_{side}_tilt"] for side in ("left", "right")} for mode in MODES},
                    float(f["base_tilt"]),
                    {mode: {reason: int(f[f"{mode}_{reason}"])
                            for reason in ("attempts", "solver", "branch", "limits", "residual", "clearance")}
                     for mode in MODES},
                )

    model = load_model()
    data = mujoco.MjData(model)
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in POSE_JOINT_NAMES}
    knee_bounds = tuple(float(value) for value in model.joint("lf1").range)
    length_range = pose_length_range(model)

    def front_kinematics(knee: float) -> tuple[float, float]:
        return _front_kinematics(model, data, knee)

    geom_ids = {side: _side_geom_ids(model, side) for side in ("left", "right")}
    wheel_ids = {
        side: [gid for gid in geom_ids[side] if model.geom_bodyid[gid] == model.body("lfwheel" if side == "left" else "rfwheel").id]
        for side in ("left", "right")
    }
    wheel_bodies = {side: model.body("lfwheel" if side == "left" else "rfwheel").id
                    for side in ("left", "right")}
    other_ids = {side: [gid for gid in geom_ids[side] if gid not in wheel_ids[side]]
                 for side in ("left", "right")}
    base_ids = [gid for gid in range(model.ngeom) if model.geom_group[gid] == 3 and model.geom_bodyid[gid] == model.body("base_link").id]
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.0
    mujoco.mj_forward(model, data)
    base_tilt = _minimum_geometry_z(model, data, base_ids, tilt=True)

    positions = {mode: {side: [] for side in ("left", "right")} for mode in MODES}
    ground_z = {mode: {side: [] for side in ("left", "right")} for mode in MODES}
    tilt_z = {mode: {side: [] for side in ("left", "right")} for mode in MODES}
    stats = {mode: {reason: 0 for reason in
                    ("attempts", "solver", "branch", "limits", "residual", "clearance")}
             for mode in MODES}
    rng = np.random.default_rng(seed)
    for mode in MODES:
        attempts = 0
        while len(positions[mode]["left"]) < count:
            attempts += 1
            stats[mode]["attempts"] += 1
            if attempts > count * 20 + 100:
                raise RuntimeError(f"Cannot fill {mode} pose bank: {len(positions[mode]['left'])}/{count}")
            length, angle = _sample_source(mode, rng, length_range)
            knee = brentq(lambda value: front_kinematics(value)[0] - length, *knee_bounds)
            _, phi = front_kinematics(knee)
            front = phi - angle
            try:
                pose = solve_standing_joint_positions(
                    model, front_joint_targets={"left": (front, knee), "right": (-front, -knee)}
                )
                reference = solve_standing_joint_positions(
                    model, front_joint_targets={"left": (phi, knee), "right": (-phi, -knee)}
                )
            except RuntimeError:
                stats[mode]["solver"] += 1
                continue
            # Rotation of a sound planar assembly changes both drive angles
            # together.  A different rear angle indicates an assembly jump.
            left_rear_delta = (pose["lf00"] - reference["lf00"]) + angle
            right_rear_delta = (pose["rf00"] - reference["rf00"]) - angle
            if max(abs(left_rear_delta), abs(right_rear_delta)) > 0.05:
                stats[mode]["branch"] += 1
                continue
            if any(
                model.joint(name).limited
                and not (model.joint(name).range[0] + 1e-5 <= value <= model.joint(name).range[1] - 1e-5)
                for name, value in pose.items()
            ):
                stats[mode]["limits"] += 1
                continue
            data.qpos[:] = model.qpos0
            data.qpos[2] = 0.0
            for name, value in pose.items():
                data.qpos[qpos_adr[name]] = value
            mujoco.mj_forward(model, data)
            if closed_chain_residual(model, data) > 1e-8:
                stats[mode]["residual"] += 1
                continue
            side_ground = {
                side: GROUND_ROOT_HEIGHT + min(
                    _minimum_geometry_z(model, data, other_ids[side], tilt=False),
                    _wheel_any_spin_min_z(model, data, wheel_bodies[side], wheel_ids[side], tilt=False),
                )
                for side in ("left", "right")
            }
            if mode != "normal" and min(side_ground.values()) < -_CLEARANCE:
                stats[mode]["clearance"] += 1
                continue
            for side, names in (("left", LEFT_JOINTS), ("right", RIGHT_JOINTS)):
                positions[mode][side].append([pose[name] for name in names])
                ground_z[mode][side].append(side_ground[side])
                tilt_z[mode][side].append(min(
                    _minimum_geometry_z(model, data, other_ids[side], tilt=True),
                    _wheel_any_spin_min_z(model, data, wheel_bodies[side], wheel_ids[side], tilt=True),
                ))

    bank = PoseBank(
        {mode: {side: np.asarray(positions[mode][side], dtype=np.float32) for side in ("left", "right")} for mode in MODES},
        {mode: {side: np.asarray(ground_z[mode][side], dtype=np.float32) for side in ("left", "right")} for mode in MODES},
        {mode: {side: np.asarray(tilt_z[mode][side], dtype=np.float32) for side in ("left", "right")} for mode in MODES},
        base_tilt,
        stats,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _CACHE_VERSION, "count": count, "seed": seed,
                   "model_hash": model_hash, "cache_signature": cache_signature,
                   "length_range": length_range, "base_tilt": base_tilt}
        for mode in MODES:
            for reason, value in stats[mode].items():
                payload[f"{mode}_{reason}"] = value
        for mode in MODES:
            for side in ("left", "right"):
                payload[f"{mode}_{side}_pos"] = bank.positions[mode][side]
                payload[f"{mode}_{side}_ground"] = bank.ground_min_z[mode][side]
                payload[f"{mode}_{side}_tilt"] = bank.normal_tilt_min_z[mode][side]
        np.savez_compressed(path, **payload)
    return bank


def sample_reset_state(env, env_ids: torch.Tensor | None, ground_bool: torch.Tensor) -> None:
    """Write V14-style reset state for selected envs, using a GPU-resident bank.

    Ground selection is per environment.  Its positive/negative angle branch
    is selected independently for each leg with 0.6/0.4 conditional weights.
    Normal root roll/pitch/yaw are uniform in +/-0.15,+/-0.15,+/-pi;
    ground roots are level at 0.25 m.  Initial passive joint speeds are zero;
    active motor speeds are sampled independently without projecting velocity
    onto the closed-chain tangent space, matching the source reset convention.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    count = len(env_ids)
    if count == 0:
        return
    ground = torch.as_tensor(ground_bool, device=env.device, dtype=torch.bool)
    if ground.ndim == 0:
        ground = ground.expand(count)
    if ground.shape != (count,):
        raise ValueError(f"ground_bool must have shape ({count},), got {tuple(ground.shape)}")

    bank = getattr(env, "_huananhu_pose_bank", None)
    if bank is None:
        cache = MODEL_XML.parent / "huananhu_pose_bank.npz"
        bank = make_pose_bank(cache_path=cache).to_torch(env.device)
        env._huananhu_pose_bank = bank
    normal_pos, normal_z = bank.sample("normal", count)
    positive_pos, _ = bank.sample("positive", count)
    negative_pos, _ = bank.sample("negative", count)
    left_positive = torch.rand(count, device=env.device) < 0.6
    right_positive = torch.rand(count, device=env.device) < 0.6
    left = torch.where(left_positive[:, None], positive_pos[:, :9], negative_pos[:, :9])
    right = torch.where(right_positive[:, None], positive_pos[:, 9:], negative_pos[:, 9:])
    joint_pos = torch.where(ground[:, None], torch.cat((left, right), dim=1), normal_pos)
    root_z = torch.where(ground, torch.full_like(normal_z, GROUND_ROOT_HEIGHT), normal_z)

    robot = env.scene["robot"]
    name_to_col = {name: i for i, name in enumerate(POSE_JOINT_NAMES)}
    joint_names = tuple(robot.joint_names)
    if set(joint_names) != set(POSE_JOINT_NAMES):
        raise RuntimeError(f"Unexpected robot joint names for pose bank: {joint_names}")
    joint_pos = joint_pos[:, [name_to_col[name] for name in joint_names]].clone()
    joint_vel = torch.zeros_like(joint_pos)
    for name in ("lf0", "lf00", "rf0", "rf00"):
        joint_vel[:, joint_names.index(name)] = (torch.rand(count, device=env.device) * 2 - 1) * (np.pi / 2)
    for name in ("lfwheel", "rfwheel"):
        joint_pos[:, joint_names.index(name)] = (torch.rand(count, device=env.device) * 2 - 1) * (2 * np.pi)
        joint_vel[:, joint_names.index(name)] = (torch.rand(count, device=env.device) * 2 - 1) * 50.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += root_z
    roll = (torch.rand(count, device=env.device) * 2 - 1) * 0.15
    pitch = (torch.rand(count, device=env.device) * 2 - 1) * 0.15
    yaw = (torch.rand(count, device=env.device) * 2 - 1) * np.pi
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    identity = torch.zeros_like(quat)
    identity[:, 0] = 1.0
    root_state[:, 3:7] = torch.where(ground[:, None], identity, quat)
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
