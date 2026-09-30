#!/usr/bin/env python3
"""固定机身闭链模型重放 H7 JID1 力矩；拟合延迟、倍率、阻尼、摩擦。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from chuanliantui_closed_adapter import ClosedChainAdapter
from view_gas_spring_ct import build_model


H7_NAMES = ("front_left", "rear_left", "front_right", "rear_right")
MODEL_JOINTS = ("lf0", "lf00", "rf0", "rf00")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mapping(path):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if mapping.get("approved") is not True:
        raise ValueError("mapping 必须由台架作者核对并标记 approved=true")
    records = mapping.get("motor_to_model", {})
    if set(records) != set(H7_NAMES):
        raise ValueError("motor_to_model 必须列出四个物理 DM 槽")
    if {records[name].get("joint") for name in H7_NAMES} != set(MODEL_JOINTS):
        raise ValueError("四个实体关节必须一一对应 lf0/lf00/rf0/rf00")
    order = [H7_NAMES.index(next(name for name in H7_NAMES
                                 if records[name]["joint"] == joint)) for joint in MODEL_JOINTS]
    sign = np.array([records[H7_NAMES[i]]["sign"] for i in order], dtype=float)
    offset = np.array([records[H7_NAMES[i]]["offset_rad"] for i in order], dtype=float)
    if not np.isin(sign, (-1, 1)).all() or not np.isfinite(offset).all():
        raise ValueError("sign 只能是 ±1，offset_rad 必须有限")
    return order, sign, offset


def load_run(directory, order, sign, offset):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "run" or manifest.get("result") != "complete":
        raise ValueError(f"{directory} 不是完整的力矩试验")
    csv_path = directory / "joint_samples.csv"
    if manifest.get("csv_sha256") and manifest["csv_sha256"] != digest(csv_path):
        raise ValueError(f"{directory} CSV 与采集时校验值不一致")
    if (manifest.get("bad_frames", 0)
            or manifest.get("sample_gaps", 0) > max(2, manifest.get("samples", 0) // 100)):
        raise ValueError(f"{directory} 存在 CRC 错误或超过 1% 的 USB 丢帧")
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"].startswith("run_"):
                rows.append(row)
    if len(rows) < 100:
        raise ValueError("有效 run 样本不足 100 帧")
    t_ns = np.array([int(row["t_sample_ns"]) for row in rows], dtype=np.int64)
    cmd_ns = np.array([int(row["t_can_queue_ns"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(t_ns) <= 0) or np.max(np.diff(t_ns)) > 8000000:
        raise ValueError("板端样本时间不单调或间隔超过 8 ms")
    if (int(rows[-1]["dropped_samples"]) - int(rows[0]["dropped_samples"])
            > max(2, len(rows) // 100)
            or int(rows[-1]["bad_frames"]) != int(rows[0]["bad_frames"])):
        raise ValueError("run 期间板端丢样本超过 1% 或收到损坏帧")
    if np.any(cmd_ns <= 0):
        raise ValueError("缺少 CAN 排队时间戳")
    flags = np.array([int(row["flags"]) for row in rows])
    if np.any(flags & 2) or np.any((flags & 8) == 0):
        raise ValueError("run 中出现锁止或 CAN 排队失败")
    t0 = t_ns[0]
    q_h7 = np.array([[float(row[f"q_raw_{name}_rad"]) for name in H7_NAMES]
                     for row in rows])
    tau_h7 = np.array([[float(row[f"tau_cmd_{name}_nm"]) for name in H7_NAMES]
                       for row in rows])
    q = sign * (q_h7[:, order] - offset)
    tau = sign * tau_h7[:, order]
    if not np.isfinite(q).all() or not np.isfinite(tau).all() or not np.any(tau != 0):
        raise ValueError("角度／力矩无效或整段力矩为零")
    leg = np.array([[float(row[f"leg_len_{side}_m"])
                     for side in ("left", "right")] for row in rows])
    angle = np.array([[float(row[f"leg_angle_{side}_rad"])
                       for side in ("left", "right")] for row in rows])
    return {"t": (t_ns - t0) * 1e-9, "cmd_t": (cmd_ns - t0) * 1e-9,
            "q": q, "tau": tau, "leg": leg, "angle": angle,
            "csv": csv_path}


class Replay:
    def __init__(self):
        self.model, self.spring_ids = build_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.adapter = ClosedChainAdapter(mujoco, self.model, self.data)
        self.actuators = np.array([
            self.model.actuator(name + "_motor").id for name in MODEL_JOINTS
        ])
        self.qpos = np.array([self.adapter.qpos[name] for name in MODEL_JOINTS])
        self.dof = np.array([self.adapter.dof[name] for name in MODEL_JOINTS])
        self.base_damping = self.model.dof_damping[self.dof].copy()
        self.base_friction = self.model.dof_frictionloss[self.dof].copy()
        self.hip_body = np.array([self.model.body(name).id for name in ("lf0", "rf0")])
        self.wheel_body = np.array([self.model.body(name).id for name in ("lfwheel", "rfwheel")])

    def initialize(self, q):
        knee_left, _ = self.adapter.geometry["left"].knee_and_jacobian(q[0], q[1])
        knee_right, _ = self.adapter.geometry["right"].knee_and_jacobian(q[2], q[3])
        self.adapter.set_virtual_pose(np.array([q[0], knee_left, 0, q[2], knee_right, 0]))
        if np.max(np.abs(self.data.qpos[self.qpos] - q)) > 0.02:
            raise ValueError("实机关节角与闭链模型装配分支不一致；先复核映射／零点")

    def leg_geometry(self):
        delta = self.data.xpos[self.wheel_body] - self.data.xpos[self.hip_body]
        return np.linalg.norm(delta[:, (0, 2)], axis=1), np.arctan2(delta[:, 0], -delta[:, 2])

    def run(self, run, params):
        delay_s, gain, damping, friction = params
        self.model.dof_damping[self.dof] = self.base_damping * damping
        self.model.dof_frictionloss[self.dof] = self.base_friction * friction
        self.initialize(run["q"][0])
        q_pred = np.empty_like(run["q"])
        leg_pred = np.empty_like(run["leg"])
        angle_pred = np.empty_like(run["angle"])
        for index, t in enumerate(run["t"]):
            while self.data.time + self.model.opt.timestep / 2 < t:
                input_t = self.data.time - delay_s
                cmd_index = np.searchsorted(run["cmd_t"], input_t, side="right") - 1
                self.data.ctrl[:] = 0
                if cmd_index >= 0:
                    self.data.ctrl[self.actuators] = gain * run["tau"][cmd_index]
                self.data.ctrl[self.spring_ids] = 150.0
                mujoco.mj_step(self.model, self.data)
                if not np.isfinite(self.data.qpos).all():
                    raise ValueError("MuJoCo 状态发散")
            q_pred[index] = self.data.qpos[self.qpos]
            leg_pred[index], angle_pred[index] = self.leg_geometry()
        return q_pred, leg_pred, angle_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="拟合采集目录")
    parser.add_argument("--holdout", type=Path, required=True, help="独立验证采集目录")
    parser.add_argument("--mapping", type=Path, required=True, help="作者已核对的关节映射 JSON")
    parser.add_argument("--out", type=Path, required=True, help="新输出目录")
    parser.add_argument("--max-evals", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out 必须是不存在的目录")
    if args.max_evals < 1:
        parser.error("--max-evals 必须为正整数")
    order, sign, offset = load_mapping(args.mapping)
    train = load_run(args.run, order, sign, offset)
    holdout = load_run(args.holdout, order, sign, offset)
    if digest(train["csv"]) == digest(holdout["csv"]):
        raise ValueError("holdout 必须来自独立采集，不能与拟合 run 相同")
    replay = Replay()
    replay.initialize(train["q"][0])
    replay.initialize(holdout["q"][0])
    sampled = slice(None, None, 5)

    def residual(params):
        try:
            q_pred, _, _ = replay.run(train, params)
            return (q_pred[sampled] - train["q"][sampled]).ravel()
        except (RuntimeError, ValueError):
            return np.full(train["q"][sampled].size, 100.0)

    delay_grid = np.arange(11, dtype=float) * 0.002
    coarse = [(float(np.mean(residual((delay, 1.0, 1.0, 1.0)) ** 2)), delay)
              for delay in delay_grid]
    candidates = sorted(coarse)[:3]
    fits = []
    for _, delay in candidates:
        fit = least_squares(lambda x: residual((delay, *x)),
                            x0=(1.0, 1.0, 1.0),
                            bounds=((0.5, 0.2, 0.2), (1.5, 5.0, 5.0)),
                            max_nfev=args.max_evals)
        fits.append((float(np.mean(fit.fun ** 2)), delay, fit))
    fits.sort(key=lambda entry: entry[0])
    best_error, best_delay, fit = fits[0]
    best_params = (best_delay, *fit.x)
    args.out.mkdir(parents=True)
    metrics = {}
    for label, run in (("train", train), ("holdout", holdout)):
        pred_q, pred_leg, pred_angle = replay.run(run, best_params)
        metrics[label] = {
            "motor_q_rmse_rad": np.sqrt(np.mean((pred_q - run["q"]) ** 2, axis=0)).tolist(),
            "leg_length_rmse_m": np.sqrt(np.mean((pred_leg - run["leg"]) ** 2, axis=0)).tolist(),
            "leg_angle_rmse_rad": np.sqrt(np.mean((pred_angle - run["angle"]) ** 2, axis=0)).tolist(),
        }
        with (args.out / f"{label}_comparison.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_s", *(f"q_real_{x}" for x in MODEL_JOINTS),
                             *(f"q_model_{x}" for x in MODEL_JOINTS),
                             "leg_real_left", "leg_real_right",
                             "leg_model_left", "leg_model_right",
                             "angle_real_left", "angle_real_right",
                             "angle_model_left", "angle_model_right"])
            for i, t in enumerate(run["t"]):
                writer.writerow([t, *run["q"][i], *pred_q[i],
                                 *run["leg"][i], *pred_leg[i],
                                 *run["angle"][i], *pred_angle[i]])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
        for i, name in enumerate(MODEL_JOINTS):
            axes[0].plot(run["t"], run["q"][:, i], label=name + " real", alpha=0.7)
            axes[0].plot(run["t"], pred_q[:, i], "--", label=name + " model")
        for i, side in enumerate(("left", "right")):
            axes[1].plot(run["t"], run["leg"][:, i], label=side + " real")
            axes[1].plot(run["t"], pred_leg[:, i], "--", label=side + " model")
            axes[2].plot(run["t"], run["angle"][:, i], label=side + " real")
            axes[2].plot(run["t"], pred_angle[:, i], "--", label=side + " model")
        for ax, unit in zip(axes, ("motor angle [rad]", "leg length [m]", "leg angle [rad]")):
            ax.set_ylabel(unit)
            ax.legend(ncol=4, fontsize=8)
            ax.grid(True)
        axes[-1].set_xlabel("MCU time [s]")
        figure.tight_layout()
        figure.savefig(args.out / f"{label}_comparison.png", dpi=150)
        plt.close(figure)
    report = {
        "fit": dict(zip(("delay_s", "torque_gain", "damping_scale", "friction_scale"),
                        [float(x) for x in best_params])),
        "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
        "continuous_jacobian_rank": int(np.linalg.matrix_rank(fit.jac)),
        "parameters_identifiable": bool(np.linalg.matrix_rank(fit.jac) == 3
                                         and (len(fits) < 2 or fits[1][0] > best_error * 1.01)),
        "delay_grid_scores": [{"delay_s": float(delay), "mean_q_error_sq": float(score)}
                              for score, delay in coarse],
        "model_assumptions": "fixed base; constant gas spring 150 N per side; CAN queue timestamp is not motor effective time; leg-angle conventions require static verification",
        "metrics": metrics,
        "input_sha256": {"train": digest(train["csv"]), "holdout": digest(holdout["csv"]),
                         "mapping": digest(args.mapping),
                         "model": digest(Path(__file__).with_name("chuanliantui.xml"))},
    }
    (args.out / "fit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
