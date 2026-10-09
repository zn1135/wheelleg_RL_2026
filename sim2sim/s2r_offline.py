#!/usr/bin/env python3
"""离线复算 S2R1 策略并在闭链 MuJoCo 中重放实机电机命令。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODEL_JOINTS = ("rf0", "rf00", "rfwheel", "lf0", "lf00", "lfwheel")
MOTOR_NAMES = ("front_left", "rear_left", "front_right", "rear_right",
               "wheel_left", "wheel_right")
# S2R1 CONTROL 的物理槽顺序是四个 DM，随后两轮；模型交叉顺序在映射里显式指定。
S2R_ACTIVE = 1 << 2
S2R_OUTPUT_ENABLED = 1 << 6
S2R_ONLINE = 1 << 7
S2R_FAULT = 1 << 9
S2R_FALLEN = 1 << 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_session(folder):
    folder = Path(folder)
    meta_paths = sorted(folder.glob("meta_*.json"))
    if not meta_paths:
        raise ValueError("缺少完整 META；请用 H7/tools/s2r_capture.py 解码完整采集")
    metas = [json.loads(path.read_text(encoding="utf-8")) for path in meta_paths]
    if any(other != metas[0] for other in metas[1:]):
        raise ValueError("会话包含多个不同 META 配置；请按配置变更事件分段")
    meta = metas[0]
    if meta.get("protocol_version") != 1:
        raise ValueError("只支持 S2R1 v1")
    frames_path = folder / "frames.jsonl"
    frames = []
    with frames_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            frame = json.loads(line)
            if frame.get("packet_duplicate"):
                continue
            if frame.get("type") in ("POLICY", "HISTORY", "CONTROL", "IMU"):
                frames.append(frame)
    if not frames:
        raise ValueError("会话不含 POLICY、CONTROL 或 IMU 帧")
    identities = {(frame["boot_id"], frame["session_id"]) for frame in frames}
    if len(identities) != 1 or next(iter(identities))[1] == 0:
        raise ValueError("每次只能分析同一启动的一次非零策略会话")
    return meta, frames_path, frames


def finite_vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if np.isfinite(array).all() else None


def policy_inputs(frames):
    histories = {}
    for frame in frames:
        if frame["type"] == "HISTORY":
            payload = frame["payload"]
            value = finite_vector(payload.get("history"), 125)
            if value is not None:
                histories[(payload["history_epoch"], payload["history_seq"])] = value
    for frame in frames:
        if frame["type"] != "POLICY":
            continue
        payload = frame["payload"]
        obs = finite_vector(payload.get("obs"), 25)
        history = finite_vector(frame.get("network_history"), 125)
        if history is None:
            history = histories.get((payload["history_epoch"], payload["history_seq"]))
        if obs is None or history is None or not np.allclose(history[-25:], obs, atol=1e-5):
            continue
        yield frame, obs, history


def compare_policy(meta, frames, checkpoint, output):
    # Isaac Gym 的导入顺序是此仓库 Python 环境的契约。
    import isaacgym  # noqa: F401
    import torch
    from mj_sim2sim_ct import load_policy

    # S2R1 POLICY.action_published 已由固件采样层转回训练空间。
    if meta.get("action_size") not in (None, 6):
        raise ValueError("META 中的动作维度不是 chuanliantui 6 维接口")
    action_clip = float(meta.get("action_clip", 0))
    if not math.isfinite(action_clip) or action_clip < 0:
        raise ValueError("META 中 action_clip 无效")
    policy = load_policy(str(checkpoint), device="cpu")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for frame, obs, history in policy_inputs(frames):
        with torch.inference_mode():
            predicted, latent = policy.act_inference(
                torch.from_numpy(obs.astype(np.float32))[None],
                torch.from_numpy(history.astype(np.float32))[None])
        predicted = predicted[0].numpy()
        if action_clip:
            predicted = np.clip(predicted, -action_clip, action_clip)
        latent = latent[0].numpy()
        actual = finite_vector(frame["payload"].get("action_published"), 6)
        actual_latent = finite_vector(frame["payload"].get("latent"), 3)
        row = {"t_us": int(frame["t_us"]),
               "policy_seq": int(frame["payload"]["policy_seq"]),
               "published_available": actual is not None}
        for i in range(6):
            row[f"host_action_{i}"] = float(predicted[i])
            row[f"h7_action_{i}"] = float(actual[i]) if actual is not None else ""
        for i in range(3):
            row[f"host_latent_{i}"] = float(latent[i])
            row[f"h7_latent_{i}"] = float(actual_latent[i]) if actual_latent is not None else ""
        rows.append(row)
    if not rows:
        output.rmdir()
        raise ValueError("没有可同步的 25+125 维策略输入；需完整 HISTORY 与 POLICY 帧")
    with (output / "policy_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = np.asarray([
        [row[f"host_action_{i}"] - row[f"h7_action_{i}"] for i in range(6)]
        for row in rows if row["published_available"]], dtype=np.float64)
    report = {
        "samples": len(rows), "h7_action_samples": len(differences),
        "action_rmse": np.sqrt(np.mean(differences ** 2, axis=0)).tolist()
        if len(differences) else None,
        "checkpoint_sha256": sha256(checkpoint),
        "note": "动作差异只有在 checkpoint 与板端模型权重一致时才能解释为推理实现差异",
    }
    (output / "policy_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_mapping(path):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if mapping.get("approved") is not True:
        raise ValueError("映射须由现场人员核对并明确写 approved=true")
    records = mapping.get("motor_to_model", {})
    if set(records) != set(MOTOR_NAMES):
        raise ValueError("motor_to_model 必须列出四个 DM 与左右两轮物理槽")
    joints = [records[name].get("joint") for name in MOTOR_NAMES]
    if set(joints) != set(MODEL_JOINTS):
        raise ValueError("六个物理槽必须与模型执行器一一对应")
    order = [joints.index(joint) for joint in MODEL_JOINTS]
    sign = np.asarray([records[MOTOR_NAMES[i]].get("sign") for i in order], dtype=float)
    offset = np.asarray([records[MOTOR_NAMES[i]].get("offset_rad") for i in order], dtype=float)
    if not np.isin(sign, (-1, 1)).all() or not np.isfinite(offset).all():
        raise ValueError("sign 只能为 ±1，offset_rad 必须是有限实数")
    return order, sign, offset


def control_samples(frames, order, sign, offset, max_gap_s):
    samples = []
    for frame in frames:
        if frame["type"] != "CONTROL":
            continue
        flags = int(frame["flags"])
        if (flags & (S2R_ACTIVE | S2R_OUTPUT_ENABLED | S2R_ONLINE)
                != S2R_ACTIVE | S2R_OUTPUT_ENABLED | S2R_ONLINE
                or flags & (S2R_FAULT | S2R_FALLEN)):
            continue
        payload = frame["payload"]
        q = finite_vector(payload.get("q_motor"), 6)
        dq = finite_vector(payload.get("dq_motor"), 6)
        tau = finite_vector(payload.get("tau_motor_request"), 6)
        if q is None or dq is None or tau is None:
            continue
        if int(payload.get("motor_feedback_valid_mask", 0)) & 0x3F != 0x3F:
            continue
        samples.append((int(frame["t_us"]), int(payload["control_seq"]),
                        sign * (q[order] - offset), sign * dq[order], sign * tau[order]))
    # 录制结束后的回传帧时间戳仍是原采样时刻；按采样时刻排序并去重。
    samples.sort(key=lambda item: (item[0], item[1]))
    samples = list({(item[0], item[1]): item for item in samples}.values())
    if len(samples) < 2:
        raise ValueError("至少需要两帧有效 CONTROL；先采集 100 Hz 短窗")
    t = np.asarray([item[0] for item in samples], dtype=np.int64)
    gaps = np.diff(t) * 1e-6
    if np.any(gaps <= 0) or float(gaps.max()) > max_gap_s:
        raise ValueError("CONTROL 时间戳不递增或间隔超过 {:.0f} ms；不能盲目保持旧力矩".format(
            max_gap_s * 1000))
    return samples


def replay_mujoco(frames, mapping, model_path, output, base_height, max_gap_s,
                  gas_spring_force):
    import mujoco
    from chuanliantui_closed_adapter import ClosedChainAdapter

    order, sign, offset = load_mapping(mapping)
    samples = control_samples(frames, order, sign, offset, max_gap_s)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.neq != 4:
        raise ValueError("只接受含 4 个 connect 的 chuanliantui 闭链 MJCF")
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = ClosedChainAdapter(mujoco, model, data)
    actuators = np.asarray([model.actuator(name + "_motor").id for name in MODEL_JOINTS])
    qpos_ids = np.asarray([model.jnt_qposadr[model.joint(name).id]
                           for name in MODEL_JOINTS])
    dof_ids = np.asarray([model.jnt_dofadr[model.joint(name).id]
                          for name in MODEL_JOINTS])
    spring_ids = np.asarray([model.actuator(name).id for name in
                             ("left_gas_spring_motor", "right_gas_spring_motor")])
    first_q = samples[0][2]
    left_knee, _ = adapter.geometry["left"].knee_and_jacobian(first_q[0], first_q[1])
    right_knee, _ = adapter.geometry["right"].knee_and_jacobian(first_q[3], first_q[4])
    adapter.set_virtual_pose(np.asarray([first_q[0], left_knee, 0.0,
                                         first_q[3], right_knee, 0.0]))
    data.qpos[qpos_ids[[2, 5]]] = first_q[[2, 5]]
    if np.max(np.abs(data.qpos[qpos_ids] - first_q)) > 0.02:
        raise ValueError("初始实体角与闭链模型装配分支不符；复核物理映射和零点")
    base_jid = model.joint("floating_base").id
    data.qpos[model.jnt_qposadr[base_jid] + 2] = base_height
    data.qvel[dof_ids] = samples[0][3]
    mujoco.mj_forward(model, data)

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    t0 = samples[0][0]
    sample_times = np.asarray([item[0] for item in samples], dtype=np.int64)
    for t_us, sequence, q_real, _, _ in samples:
        target_time = (t_us - t0) * 1e-6
        while data.time + model.opt.timestep / 2 < target_time:
            cmd_index = max(0, np.searchsorted(
                sample_times, t0 + int(data.time * 1e6), side="right") - 1)
            data.ctrl[:] = 0.0
            data.ctrl[actuators] = samples[cmd_index][4]
            data.ctrl[spring_ids] = gas_spring_force
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise ValueError("MuJoCo 状态发散，结果目录保留已写内容供检查")
        q_model = data.qpos[qpos_ids]
        row = {"t_s": target_time, "control_seq": sequence}
        for i, joint in enumerate(MODEL_JOINTS):
            row[f"{joint}_real_rad"] = float(q_real[i])
            row[f"{joint}_model_rad"] = float(q_model[i])
        rows.append(row)
    with (output / "motor_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    real = np.asarray([[row[f"{joint}_real_rad"] for joint in MODEL_JOINTS] for row in rows])
    predicted = np.asarray([[row[f"{joint}_model_rad"] for joint in MODEL_JOINTS] for row in rows])
    report = {
        "samples": len(rows), "duration_s": rows[-1]["t_s"],
        "motor_angle_rmse_rad": dict(zip(MODEL_JOINTS,
            np.sqrt(np.mean((predicted - real) ** 2, axis=0)).tolist())),
        "mapping_sha256": sha256(mapping), "model_sha256": sha256(model_path),
        "base_height_assumed_m": base_height,
        "gas_spring_force_assumed_n": gas_spring_force,
        "limitations": ["没有世界位置、线速度及接触真值；此结果只核对电机响应",
                        "S2R1 当前不记录精确 CAN 提交与电机内部力矩生效时刻",
                        "轮子没有角度绝对零位时，其角度 RMSE 不宜用于评价动力学"],
    }
    (output / "mujoco_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True,
                        help="H7 s2r_capture.py 导出的 boot_*/session_* 目录")
    parser.add_argument("--out", type=Path, required=True, help="尚不存在的结果目录")
    parser.add_argument("--checkpoint", type=Path, help="完整 chuanliantui model_*.pt，用于上位机复算")
    parser.add_argument("--mapping", type=Path, help="已核对的六电机映射 JSON，用于 MuJoCo 重放")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).with_name("chuanliantui.xml"))
    parser.add_argument("--base-height", type=float, default=0.33)
    parser.add_argument("--gas-spring-force", type=float, default=150.0)
    parser.add_argument("--max-control-gap-ms", type=float, default=20.0)
    args = parser.parse_args()
    if not args.checkpoint and not args.mapping:
        parser.error("至少指定 --checkpoint（复算策略）或 --mapping（MuJoCo 重放）")
    if args.out.exists():
        parser.error("--out 必须是尚不存在的目录，以免覆盖历史数据")
    if not math.isfinite(args.base_height) or args.base_height <= 0:
        parser.error("--base-height 必须为正数")
    if not math.isfinite(args.gas_spring_force) or args.gas_spring_force < 0:
        parser.error("--gas-spring-force 必须非负")
    if not math.isfinite(args.max_control_gap_ms) or args.max_control_gap_ms <= 0:
        parser.error("--max-control-gap-ms 必须为正数")
    meta, frames_path, frames = read_session(args.session)
    args.out.mkdir(parents=True)
    reports = {"input_sha256": {"frames": sha256(frames_path)}, "meta": meta}
    if args.checkpoint:
        reports["policy"] = compare_policy(meta, frames, args.checkpoint,
                                           args.out / "policy")
    if args.mapping:
        reports["mujoco"] = replay_mujoco(
            frames, args.mapping, args.model, args.out / "mujoco", args.base_height,
            args.max_control_gap_ms * 1e-3, args.gas_spring_force)
    (args.out / "summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in reports.items() if key != "meta"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
