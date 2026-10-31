"""按初始左右腿前后分支统计稳站v2配对评估中的倒置（只读已有评估产物，不跑仿真）。

分支用初始qpos正运动学的机身系髋→轮心x：>0为轮在髋前。倒置为500Hz轨迹中
gravity_z>0（倾角超过90°），只统计首回合、active且有限的样本。
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def branches(scene, qpos):
    model = mujoco.MjModel.from_binary_path(str(scene))
    data = mujoco.MjData(model)
    hips = [model.body(f"robot/{n}").id for n in ("lf0", "rf0")]
    wheels = [model.body(f"robot/{n}").id for n in ("lfwheel", "rfwheel")]
    out = np.empty((len(qpos), 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[model.body("robot/base_link").id].reshape(3, 3)
        for side in range(2):
            out[i, side] = (rotation.T @ (data.xpos[wheels[side]] - data.xpos[hips[side]]))[0]
    return out


def label(x):
    front = x > 0.
    if front[0] == front[1]:
        return "same"
    return "left_front_right_rear" if front[0] else "left_rear_right_front"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = None
    report = {}
    for name in args.labels:
        initial = np.load(args.root / name / "initial.npz")
        if reference is None:
            reference = initial["qpos"]
            x = branches(args.scene, reference)
            groups = np.array([label(row) for row in x])
        elif not np.array_equal(initial["qpos"], reference):
            raise SystemExit(f"{name}: 初始qpos与首个标签不一致")
        trace = np.load(args.root / name / "trajectories.npz")
        fields = json.loads((args.root / name / "result.json").read_text())["physical_fields"]
        p = trace["physical"]
        gz, live = p[..., fields.index("gravity_z")], (p[..., fields.index("active")] > .5) & (p[..., fields.index("finite")] > .5)
        inverted = (gz > 0.) & live
        first = np.where(inverted.any(0), inverted.argmax(0) * .002, np.nan)
        rows = {}
        for group in ("same", "left_front_right_rear", "left_rear_right_front"):
            ids = np.flatnonzero(groups == group)
            hit = ids[~np.isnan(first[ids])]
            rows[group] = dict(n=int(len(ids)), inverted=int(len(hit)), ids=hit.tolist(),
                               first_inversion_s=[round(float(first[i]), 3) for i in hit])
        report[name] = rows
    summary = dict(root=str(args.root), branch_rule="body-frame hip→wheel x>0 means wheel ahead of hip",
                   inversion_rule="gravity_z>0 while active and finite, 500Hz", groups=report,
                   hip_to_wheel_x_m=x.round(4).tolist(), group_of_env=groups.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    for name, rows in report.items():
        print(name, {g: f"{r['inverted']}/{r['n']}" for g, r in rows.items()},
              "first:", [min(r["first_inversion_s"] or [np.nan]) for r in rows.values()],
              [max(r["first_inversion_s"] or [np.nan]) for r in rows.values()])


if __name__ == "__main__":
    main()
