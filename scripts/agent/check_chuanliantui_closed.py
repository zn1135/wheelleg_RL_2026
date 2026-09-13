#!/usr/bin/env python3
"""chuanliantui 真实闭链 MJCF 的静态结构和约束检查。"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "sim2sim/chuanliantui.xml"
SITE_PAIRS = (
    ("right_A1", "right_rear_pin1", "right_front_pin1"),
    ("right_A2", "right_rear_pin2", "right_front_pin2"),
    ("left_A1", "left_rear_pin1", "left_front_pin1"),
    ("left_A2", "left_rear_pin2", "left_front_pin2"),
)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    expected = (21, 20, 6, 16, 15, 4)
    actual = (model.nq, model.nv, model.nu, model.nbody, model.njnt, model.neq)
    print("nq={} nv={} nu={} nbody={} njnt={} neq={}".format(*actual))
    assert actual == expected, "模型结构不符：{} != {}".format(actual, expected)

    site_ids = []
    mujoco.mj_forward(model, data)
    for label, rear_name, front_name in SITE_PAIRS:
        rear = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, rear_name)
        front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, front_name)
        error = float(np.linalg.norm(data.site_xpos[rear] - data.site_xpos[front]))
        print("{} initial_pin_error_m={:.9g}".format(label, error))
        assert error < 1e-9, "{} 初始销轴未闭合".format(label)
        site_ids.append((rear, front))

    max_error = 0.0
    for _ in range(1000):  # 5 s，无控制
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("无控制仿真出现非有限状态")
        max_error = max(
            max_error,
            max(float(np.linalg.norm(data.site_xpos[rear] - data.site_xpos[front])) for rear, front in site_ids),
        )
    print("5s_no_control_max_pin_error_m={:.9g}, final_base_z={:.9g}".format(max_error, data.qpos[2]))
    assert max_error < 1e-3, "无控制约束误差过大"
    print("闭链静态检查通过")


if __name__ == "__main__":
    main()
