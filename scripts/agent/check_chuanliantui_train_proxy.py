#!/usr/bin/env python3
"""静态检查 chuanliantui 串联训练代理 MJCF。"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "sim2sim/chuanliantui_train_proxy.xml"
DOF_NAMES = ("lf0", "lf1", "lfwheel", "rf0", "rf1", "rfwheel")
ACTUATOR_NAMES = tuple("{}_motor".format(name) for name in DOF_NAMES)
TORQUE_LIMITS = np.array((40.0, 40.0, 3.9, 40.0, 40.0, 3.9))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    )
    movable_names = tuple(name for name in joint_names if name != "floating_base")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    print("nq={} nv={} nu={} njnt={} neq={}".format(model.nq, model.nv, model.nu, model.njnt, model.neq))
    print("movable_dofs={}".format(movable_names))
    print("actuators={}".format(actuator_names))
    assert model.neq == 0, "串联训练代理不得包含 equality/connect 约束"
    assert movable_names == DOF_NAMES, "DOF 顺序不符：{}".format(movable_names)
    assert actuator_names == ACTUATOR_NAMES, "执行器顺序不符：{}".format(actuator_names)
    expected_ranges = np.column_stack((-TORQUE_LIMITS, TORQUE_LIMITS))
    assert np.allclose(model.actuator_ctrlrange, expected_ranges), model.actuator_ctrlrange
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(16):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    print("串联训练代理静态检查通过")


if __name__ == "__main__":
    main()
