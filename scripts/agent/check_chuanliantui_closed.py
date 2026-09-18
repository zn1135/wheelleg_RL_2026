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
    expected = (21, 20, 8, 16, 15, 4)
    actual = (model.nq, model.nv, model.nu, model.nbody, model.njnt, model.neq)
    print("nq={} nv={} nu={} nbody={} njnt={} neq={}".format(*actual))
    assert actual == expected, "模型结构不符：{} != {}".format(actual, expected)

    def named(kind, name):
        index = mujoco.mj_name2id(model, kind, name)
        assert index >= 0, "缺少对象：" + name
        return index

    spring_ids = []
    for side, prefix in (("left", "lf"), ("right", "rf")):
        aid = named(mujoco.mjtObj.mjOBJ_ACTUATOR, side + "_gas_spring_motor")
        tid = named(mujoco.mjtObj.mjOBJ_TENDON, side + "_gas_spring_tendon")
        assert model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_TENDON
        assert model.actuator_trnid[aid, 0] == tid
        np.testing.assert_allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
        assert model.actuator_ctrllimited[aid]
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], [0, 150])
        assert model.tendon_num[tid] == 2
        for offset, (end, body) in enumerate((("upper", prefix + "0"), ("lower", prefix + "1"))):
            sid = named(mujoco.mjtObj.mjOBJ_SITE, side + "_gas_spring_" + end)
            assert model.site_bodyid[sid] == named(mujoco.mjtObj.mjOBJ_BODY, body)
            wrap = model.tendon_adr[tid] + offset
            assert model.wrap_type[wrap] == mujoco.mjtWrap.mjWRAP_SITE
            assert model.wrap_objid[wrap] == sid
        rear = named(mujoco.mjtObj.mjOBJ_JOINT, prefix + "00")
        assert not model.jnt_limited[rear], "后输入轴不应有整圈限位"
        spring_ids.append(aid)
    for index, name in enumerate(("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")):
        assert named(mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor") == index
    assert set(spring_ids).isdisjoint(range(6))
    for force in (0.0, 150.0):
        data.ctrl[spring_ids] = force
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force[spring_ids], force)
        np.testing.assert_allclose(data.actuator_force[:6], 0)
    mujoco.mj_resetData(model, data)

    site_ids = []
    mujoco.mj_forward(model, data)
    for label, rear_name, front_name in SITE_PAIRS:
        rear = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, rear_name)
        front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, front_name)
        error = float(np.linalg.norm(data.site_xpos[rear] - data.site_xpos[front]))
        print("{} initial_pin_error_m={:.9g}".format(label, error))
        assert error < 1e-9, "{} 初始销轴未闭合".format(label)
        site_ids.append((rear, front))

    # 同侧前后输入轴一起转动，保持膝部零位；检查跨整圈时几何仍闭合。
    for angle in np.linspace(-2 * np.pi, 2 * np.pi, 65):
        mujoco.mj_resetData(model, data)
        for name in ("lf0", "lf00", "rf0", "rf00"):
            jid = named(mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = angle
        mujoco.mj_forward(model, data)
        error = max(np.linalg.norm(data.site_xpos[r] - data.site_xpos[f]) for r, f in site_ids)
        assert error < 1e-8, "整圈姿态销轴未闭合：{}".format(error)
    print("气弹簧连接、0/150 N 施力及 ±360° 闭合检查通过")
    mujoco.mj_resetData(model, data)

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
