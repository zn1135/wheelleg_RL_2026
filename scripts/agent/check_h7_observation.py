#!/usr/bin/env python3
"""对照 H7_clion C 观测、CAD 闭链与训练 build_obs；可复算 obs63 串口采集。"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import isaacgym  # 必须在 torch（由 mj_sim2sim_ct 导入）之前
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2sim import mj_sim2sim_ct as sim
from sim2sim.chuanliantui_closed_adapter import _PlanarLegGeometry, ClosedChainAdapter


class Input(ct.Structure):
    _fields_ = [(name, ct.c_float * count) for name, count in
                (("dm_pos", 4), ("dm_vel", 4), ("wheel_vel", 2), ("quat", 4),
                 ("gyro", 3), ("command", 3), ("last_action", 6))]


class Result(ct.Structure):
    _fields_ = [("obs", ct.c_float * 25), ("joint_pos", ct.c_float * 4),
                ("joint_vel", ct.c_float * 6), ("jac", (ct.c_float * 2) * 2)]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


class Reference:
    def __init__(self, cad_zero_fit=None, bilateral_zero_fit=None):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "sim2sim/chuanliantui.xml"))
        self.geometry = [_PlanarLegGeometry(model, side) for side in ClosedChainAdapter._SIDES]
        self.front_zero = [np.arctan2(-g.front1_offset[1], g.front1_offset[0]) for g in self.geometry]
        self.rear_zero = [np.arctan2(-g.rear1_offset[1], g.rear1_offset[0]) for g in self.geometry]
        # Default candidate is the user-confirmed matching SolidWorks zero pose.
        # --cad-zero-fit additionally verifies the source capture and endpoint.
        right_odd = (cad_zero_fit["provisional_odd_slot_cad_zero_rad"]
                     if cad_zero_fit else 0.8138714045)
        right_even = (cad_zero_fit["provisional_even_slot_cad_zero_rad"]
                      if cad_zero_fit else 2.3227884081)
        self.odd_zero = [right_odd, 0.7904777736]
        self.even_zero = [right_even, 2.3689040152]
        if bilateral_zero_fit:
            sides = bilateral_zero_fit["conditional_cad_angle_zero_by_model_side_rad"]
            self.odd_zero = [sides["lf_physical_right"]["odd_front_dm3"],
                             sides["rf_physical_left"]["odd_front_dm1"]]
            self.even_zero = [sides["lf_physical_right"]["even_rear_dm2"],
                              sides["rf_physical_left"]["even_rear_dm0"]]

    def build(self, data):
        q = np.zeros(6)
        dq = np.zeros(6)
        for side, geometry in enumerate(self.geometry):
            sign = -1 if side == 0 else 1
            source, dof = 2 * (1 - side), 3 * side
            # B: keep each original motor's sign/zero; odd drives CAD front,
            # even drives CAD rear. Absolute pose remains a separate gate.
            front = sign * wrap(data.dm_pos[source + 1] - self.odd_zero[side])
            rear = sign * wrap(data.dm_pos[source] + np.pi - self.even_zero[side])
            knee, jac = geometry.knee_and_jacobian(front, rear)
            q[dof:dof + 2] = (front, knee)
            dq[dof] = sign * data.dm_vel[source + 1]
            dq[dof + 1] = sign * np.dot(jac, np.asarray(data.dm_vel)[[source + 1, source]])
            dq[dof + 2] = sign * data.wheel_vel[side]
        quat = np.asarray(data.quat)
        quat = quat / np.linalg.norm(quat)
        obs = sim.build_obs(quat[[1, 2, 3, 0]], np.asarray(data.gyro), q, dq,
                            np.asarray(data.command), np.asarray(data.last_action))
        return obs, q[[0, 1, 3, 4]], dq


def check_native(library, ref):
    rng = np.random.default_rng(20261001)
    maximum = {"knee_rad": 0.0, "jacobian": 0.0, "obs": 0.0, "joint_pos": 0.0, "joint_vel": 0.0}
    count = 0
    for _ in range(1000):
        data = Input()
        for side, geometry in enumerate(ref.geometry):
            front = float(np.float32(rng.uniform(-3, 3)))
            rear = float(np.float32(front + rng.uniform(-0.4, 0.4)))
            expected_knee, expected_jac = geometry.knee_and_jacobian(front, rear)
            knee, jac = ct.c_float(), (ct.c_float * 2)()
            assert library.Lower_Geometry_Knee(side, front, rear, ct.byref(knee), jac)
            maximum["knee_rad"] = max(maximum["knee_rad"], abs(wrap(knee.value - expected_knee)))
            maximum["jacobian"] = max(maximum["jacobian"], float(np.max(np.abs(np.asarray(jac) - expected_jac))))
            sign = -1 if side == 0 else 1
            source = 2 * (1 - side)
            data.dm_pos[source + 1] = wrap(sign * front + ref.odd_zero[side])
            data.dm_pos[source] = wrap(sign * rear + ref.even_zero[side] - np.pi)
        for name, _ in Input._fields_:
            if name == "dm_pos":
                continue
            target = getattr(data, name)
            scale = 300 if name in ("command", "last_action") else 10
            for i in range(len(target)):
                target[i] = rng.uniform(-scale, scale)
        result = Result()
        assert library.Lower_Observation_Build(ct.byref(data), ct.byref(result))
        expected = ref.build(data)
        for name, actual, target in zip(("obs", "joint_pos", "joint_vel"),
                                        (result.obs, result.joint_pos, result.joint_vel), expected):
            error = float(np.max(np.abs(np.asarray(actual) - target)))
            maximum[name] = max(maximum[name], error)
            np.testing.assert_allclose(actual, target, atol=2e-4, rtol=2e-4)
        for side, geometry in enumerate(ref.geometry):
            front = result.joint_pos[side * 2]
            source = 2 * (1 - side)
            sign = -1 if side == 0 else 1
            rear = sign * wrap(data.dm_pos[source] + np.pi - ref.even_zero[side])
            _, target_jac = geometry.knee_and_jacobian(front, rear)
            np.testing.assert_allclose(result.jac[side], target_jac, atol=2e-4, rtol=2e-4)
        count += 1
    assert maximum["knee_rad"] < 2e-4 and maximum["jacobian"] < 2e-4, maximum
    # 非法输入必须拒绝且清空旧结果，包括 last_action，避免 NaN 泄漏进VOFA。
    for name, _ in Input._fields_:
        for index in range(len(getattr(data, name))):
            bad = Input.from_buffer_copy(data)
            getattr(bad, name)[index] = float("nan")
            result = Result()
            result.obs[0] = 123
            assert not library.Lower_Observation_Build(ct.byref(bad), ct.byref(result))
            assert not any(bytes(result))
    bad = Input.from_buffer_copy(data)
    bad.quat[:] = [0] * 4
    assert not library.Lower_Observation_Build(ct.byref(bad), ct.byref(Result()))
    for side, front, rear in ((2, 0, 0), (0, float("nan"), 0), (1, 0, float("inf"))):
        knee, jac = ct.c_float(123), (ct.c_float * 2)(123, 123)
        assert not library.Lower_Geometry_Knee(side, front, rear, ct.byref(knee), jac)
        assert knee.value == 0 and list(jac) == [0, 0]
    return {"samples": count, "geometry_cases": 2 * count, "max_abs_error": maximum,
            "scope": "同一候选电机映射的纯数值一致性，不代表实机映射已标定"}


def check_source_isolation(library):
    """按实测槽身份独立检查C输出，不依赖Reference的来源映射。"""
    baseline = Input()
    baseline.dm_pos[:] = [-0.7365421, 0.6915373, -1.1178, 0.37107]
    baseline.wheel_vel[:] = [1.25, -2.5]
    baseline.quat[:] = [1, 0, 0, 0]
    baseline.gyro[:] = [0.2, -0.3, 0.4]
    baseline.command[:] = [0.1, -0.2, 0.2]
    baseline.last_action[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    base = Result()
    assert library.Lower_Observation_Build(ct.byref(baseline), ct.byref(base))
    cases = []
    # 源槽 -> 已实测模型侧，位置输出、速度输出、前输入符号；不复用Reference。
    for source, model, pos, vel, sign in (
            (0, "lf", [2, 3], [3, 4], 1), (1, "lf", [2, 3], [3, 4], 1),
            (2, "rf", [0, 1], [0, 1], -1), (3, "rf", [0, 1], [0, 1], -1)):
        data = Input.from_buffer_copy(baseline)
        data.dm_pos[source] += 0.07
        data.dm_vel[source] = 0.4
        actual = Result()
        assert library.Lower_Observation_Build(ct.byref(data), ct.byref(actual))
        allowed_obs = [9 + i for i in pos] + [13 + i for i in vel]
        for name, allowed in (("obs", allowed_obs), ("joint_pos", pos), ("joint_vel", vel)):
            delta = np.asarray(getattr(actual, name)) - np.asarray(getattr(base, name))
            unchanged = [i for i in range(len(delta)) if i not in allowed]
            assert np.all(delta[unchanged] == 0), (source, name, delta)
            assert np.max(np.abs(delta[allowed])) > 1e-4, (source, name, delta)
        # B: odd DM slots drive CAD front long arms and therefore the model hip.
        if source in (1, 3):
            np.testing.assert_allclose(actual.joint_pos[pos[0]] - base.joint_pos[pos[0]],
                                       sign * 0.07, atol=2e-6)
            np.testing.assert_allclose(actual.joint_vel[vel[0]], sign * 0.4, atol=1e-7)
        else:
            assert actual.joint_pos[pos[0]] == base.joint_pos[pos[0]]
            assert actual.joint_vel[vel[0]] == 0
        cases.append({"driver_slot": source, "model_side": model, "passed": True})
    return {"cases": cases, "scope": "单槽扰动只影响实测对应侧；两轮、IMU、命令、last_action保持不变"}


def check_capture(directory, capture_tool, ref):
    raw = directory / "raw.bin"
    decoder = capture_tool.Decoder(63, layout_id=2501)
    rows = decoder.feed(raw.read_bytes())
    if not rows or decoder.framing_errors or decoder.nonfinite_frames:
        raise ValueError("采集为空或存在坏帧")
    max_error = np.zeros(25)
    max_q_error = np.zeros(4)
    max_dq_error = np.zeros(6)
    for row in rows:
        if row[62] != 2501 or row[25] != 1 or row[61] != 0:
            raise ValueError("采集布局或观测有效性不符")
        if row[26] != 63 or (int(row[27]) & 15) != 12:
            raise ValueError("采集不是全在线且失能状态")
        data = Input()
        data.dm_pos[:] = row[33:37]
        data.dm_vel[:] = row[37:41]
        data.wheel_vel[:] = row[41:43]
        data.quat[:] = row[43:47]
        data.gyro[:] = row[47:50]
        data.command[:] = (0, 0, 0.20)
        expected, q, dq = ref.build(data)
        max_error = np.maximum(max_error, np.abs(np.asarray(row[:25]) - expected))
        max_q_error = np.maximum(max_q_error, np.abs(np.asarray(row[50:54]) - q))
        max_dq_error = np.maximum(max_dq_error, np.abs(np.asarray(row[54:60]) - dq))
    assert np.max(max_error) < 2e-4, max_error
    assert np.max(max_q_error) < 2e-4 and np.max(max_dq_error) < 2e-4
    return {"frames": len(rows), "raw_sha256": digest(raw),
            "max_abs_obs_error": max_error.tolist(), "max_abs_joint_pos_error": max_q_error.tolist(),
            "max_abs_joint_vel_error": max_dq_error.tolist(),
            "physical_mapping_verified": all(row[31] == 1 for row in rows),
            "scope": "从串口同拍原始输入经CAD几何和训练build_obs复算，未验证真实姿态/轴向/零点"}


def check_cad_zero_fit(library, fit_data, capture_dir, bilateral_data=None):
    """One confirmed matching CAD pose checks the C angle transform at raw zero."""
    raw_report = json.loads((capture_dir / "report.json").read_text(encoding="utf-8"))
    assert fit_data["raw_capture_sha256"] == raw_report["raw_sha256"]
    position = [raw_report["channels"]["dm{}_pos_rad".format(i)]["mean"]
                for i in range(4)]
    data = Input()
    data.dm_pos[:] = position
    data.quat[:] = [1, 0, 0, 0]
    data.command[:] = [0, 0, 0.2]
    result = Result()
    assert library.Lower_Observation_Build(ct.byref(data), ct.byref(result))
    hip, knee = result.joint_pos[:2]
    xml = mujoco.MjModel.from_xml_path(str(ROOT / "sim2sim/chuanliantui.xml"))
    upper = np.asarray(xml.body_pos[xml.body("rf1").id])[[0, 2]]
    lower = np.asarray(xml.body_pos[xml.body("rfwheel").id])[[0, 2]]

    def rotate(angle, point):
        c, s = np.cos(angle), np.sin(angle)
        return np.array((c * point[0] - s * point[1], s * point[0] + c * point[1]))

    wheel = rotate(hip, upper) + rotate(hip + knee, lower)
    target = np.asarray(fit_data["cad_target_xz_m"])
    error_mm = ((wheel - target) * 1000).tolist()
    assert np.max(np.abs(error_mm)) < 1.0, error_mm
    result_report = {"cad_target_xz_m": target.tolist(), "c_wheel_xz_m": wheel.tolist(),
                     "error_mm": error_mm, "right_side_only": bilateral_data is None,
                     "scope": "同姿态CAD端点与观测零偏数值对照；不证明实体方向/Jacobian"}
    if bilateral_data:
        assert bilateral_data["left_mirror_pose_user_confirmed"]
        np.testing.assert_allclose(position, bilateral_data["old_capture_dm_pos_zero_rad"], atol=1e-7)
        left_hip, left_knee = result.joint_pos[2:4]
        left_upper = np.asarray(xml.body_pos[xml.body("lf1").id])[[0, 2]]
        left_lower = np.asarray(xml.body_pos[xml.body("lfwheel").id])[[0, 2]]
        left_wheel = rotate(-left_hip, left_upper) + rotate(-(left_hip + left_knee), left_lower)
        left_error_mm = ((left_wheel - target) * 1000).tolist()
        assert np.max(np.abs(left_error_mm)) < 1.0, left_error_mm
        result_report["left_mirrored_cad_wheel_xz_m"] = left_wheel.tolist()
        result_report["left_mirrored_error_mm"] = left_error_mm
        result_report["scope"] += "; 左侧同点来自用户确认的镜像姿态，非独立左侧CAD测量"
    return result_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h7-root", type=Path, default=os.environ.get("H7_REPO_PATH"))
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--cad-zero-fit", type=Path,
                        help="同姿态SolidWorks单姿态拟合JSON；校验B候选CAD角零偏")
    parser.add_argument("--raw-zero-capture", type=Path,
                        help="与--cad-zero-fit匹配的125帧失能原始零位采集目录")
    parser.add_argument("--bilateral-zero-fit", type=Path,
                        help="用户确认双侧镜像姿态后生成的四槽CAD零偏JSON")
    parser.add_argument("--out", type=Path, required=True, help="新的JSON结果文件")
    args = parser.parse_args()
    if not args.h7_root or not args.h7_root.is_dir():
        parser.error("先加载.env.local或使用--h7-root")
    if args.out.exists():
        parser.error("结果文件已存在")
    host = load_module("h7_host_tests", args.h7_root / "tools/test_host.py")
    compiler, env = host.host_compiler()
    fit_data = None
    if args.cad_zero_fit:
        if not args.raw_zero_capture:
            parser.error("--cad-zero-fit requires --raw-zero-capture")
        fit_data = json.loads(args.cad_zero_fit.read_text(encoding="utf-8"))
    bilateral_data = None
    if args.bilateral_zero_fit:
        if not fit_data:
            parser.error("--bilateral-zero-fit requires --cad-zero-fit and --raw-zero-capture")
        bilateral_data = json.loads(args.bilateral_zero_fit.read_text(encoding="utf-8"))
        if not bilateral_data.get("left_mirror_pose_user_confirmed"):
            parser.error("bilateral fit has no user-confirmed mirror pose")
    ref = Reference(fit_data, bilateral_data)
    with tempfile.TemporaryDirectory(prefix="h7-obs-check-") as temporary:
        binary = Path(temporary) / "observation.so"
        subprocess.run(compiler + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                                   "-IApp", "App/lower_geometry.c", "App/lower_observation.c",
                                   "-lm", "-o", str(binary)], cwd=args.h7_root, env=env, check=True)
        library = ct.CDLL(str(binary))
        library.Lower_Geometry_Knee.argtypes = [ct.c_uint8, ct.c_float, ct.c_float,
                                               ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)]
        library.Lower_Geometry_Knee.restype = ct.c_uint8
        library.Lower_Observation_Build.argtypes = [ct.POINTER(Input), ct.POINTER(Result)]
        library.Lower_Observation_Build.restype = ct.c_uint8
        report = {"native": check_native(library, ref),
                  "source_isolation": check_source_isolation(library)}
        if fit_data:
            report["cad_zero_fit"] = check_cad_zero_fit(library, fit_data,
                                                         args.raw_zero_capture,
                                                         bilateral_data)
    if args.capture:
        capture_tool = load_module("h7_vofa_capture", args.h7_root / "tools/vofa_capture.py")
        report["capture"] = check_capture(args.capture, capture_tool, ref)
    report["source_sha256"] = {str(path): digest(path) for path in (
        args.h7_root / "App/lower_geometry.c", args.h7_root / "App/lower_observation.c",
        ROOT / "sim2sim/chuanliantui_closed_adapter.py", ROOT / "sim2sim/chuanliantui.xml",
        ROOT / "sim2sim/mj_sim2sim_ct.py", Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
