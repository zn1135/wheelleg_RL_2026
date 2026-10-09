#!/usr/bin/env python3
"""静态检查 chuanliantui 串联训练代理 MJCF。"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "sim2sim/chuanliantui_train_proxy.xml"
CLOSED_MODEL_PATH = REPO_ROOT / "sim2sim/chuanliantui.xml"
DOF_NAMES = ("rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel")
ACTUATOR_NAMES = tuple("{}_motor".format(name) for name in DOF_NAMES)
GAS_ACTUATOR_NAMES = ("right_gas_spring_motor", "left_gas_spring_motor")
TORQUE_LIMITS = np.array((40.0, 40.0, 3.9, 40.0, 40.0, 3.9))


def check_gas_springs(model: mujoco.MjModel) -> None:
    proxy, closed = ET.parse(MODEL_PATH).getroot(), ET.parse(CLOSED_MODEL_PATH).getroot()
    assert model.ntendon == 2
    for side, prefix in (("right", "rf"), ("left", "lf")):
        for end, suffix in (("upper", "0"), ("lower", "1")):
            path = ".//body[@name='{}{}']/site[@name='{}_gas_spring_{}']".format(
                prefix, suffix, side, end
            )
            target, source = proxy.find(path), closed.find(path)
            assert target is not None and source is not None, path
            assert target.attrib == source.attrib, "气弹簧端点与闭链不一致: " + path
        for path in (
            "./tendon/spatial[@name='{}_gas_spring_tendon']".format(side),
            "./actuator/motor[@name='{}_gas_spring_motor']".format(side),
        ):
            target, source = proxy.find(path), closed.find(path)
            assert target is not None and source is not None, path
            assert target.attrib == source.attrib, path
            assert [child.attrib for child in target] == [child.attrib for child in source], path

    gas_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                        for name in GAS_ACTUATOR_NAMES])
    assert np.array_equal(gas_ids, [6, 7]), gas_ids
    assert np.all(model.actuator_trntype[gas_ids] == mujoco.mjtTrn.mjTRN_TENDON)
    knee_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in ("rf1", "lf1")])
    knee_qadr, knee_dadr = model.jnt_qposadr[knee_ids], model.jnt_dofadr[knee_ids]
    tendon_ids = model.actuator_trnid[gas_ids, 0]
    data = mujoco.MjData(model)
    data.qpos[2] = 2.0
    for knee in (0.0, 0.3, 0.6):
        data.qpos[knee_qadr] = (knee, -knee)
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        assert np.allclose(data.qfrc_actuator, 0.0), "0 N 必须恢复无气弹簧条件"
        data.ctrl[gas_ids] = 150.0
        mujoco.mj_forward(model, data)
        assert np.allclose(data.actuator_force[gas_ids], 150.0)
        assert np.allclose(data.actuator_force[:6], 0.0), "物理弹簧不得写入策略电机"
        gas_torque = data.actuator_force[gas_ids] @ data.actuator_moment[gas_ids]
        assert np.allclose(data.qfrc_actuator, gas_torque)
        non_knee = np.ones(model.nv, dtype=bool)
        non_knee[knee_dadr] = False
        assert np.allclose(gas_torque[non_knee], 0.0, atol=1e-10)
        # 对伸张力有 tau = F * d(length)/dq；用有限差分独立检查力矩符号/幅值。
        epsilon = 1e-6
        for qadr, dadr, tendon_id in zip(knee_qadr, knee_dadr, tendon_ids):
            original = float(data.qpos[qadr])
            lengths = []
            for offset in (-epsilon, epsilon):
                data.qpos[qadr] = original + offset
                mujoco.mj_forward(model, data)
                lengths.append(float(data.ten_length[tendon_id]))
            data.qpos[qadr] = original
            expected = 150.0 * (lengths[1] - lengths[0]) / (2.0 * epsilon)
            assert np.isclose(gas_torque[dadr], expected, rtol=1e-7, atol=1e-7), (
                gas_torque[dadr], expected
            )
        print("gas_spring knee=+/-{:.2f}: torque_nm={}".format(knee, gas_torque[knee_dadr]))


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
    assert np.isclose(model.opt.timestep, 0.002), model.opt.timestep
    assert (model.nq, model.nv, model.nu) == (13, 12, 8)
    assert model.neq == 0, "串联训练代理不得包含 equality/connect 约束"
    assert movable_names == DOF_NAMES, "DOF 顺序不符：{}".format(movable_names)
    assert actuator_names == ACTUATOR_NAMES + GAS_ACTUATOR_NAMES, "执行器顺序不符：{}".format(actuator_names)
    expected_ranges = np.vstack((np.column_stack((-TORQUE_LIMITS, TORQUE_LIMITS)),
                                 ((0.0, 150.0), (0.0, 150.0))))
    assert np.allclose(model.actuator_ctrlrange, expected_ranges), model.actuator_ctrlrange
    for name in DOF_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        # 仅两膝加强限位约束，其余关节保留 MuJoCo 默认的限位柔度。
        expected_solref = (0.004, 1.0) if name in ("rf1", "lf1") else (0.02, 1.0)
        assert np.allclose(model.jnt_solref[joint_id], expected_solref), (
            "{} 限位 solref 不符：{}，期望 {}".format(
                name, model.jnt_solref[joint_id], expected_solref
            )
        )
    check_gas_springs(model)
    data = mujoco.MjData(model)
    data.ctrl[6:] = 150.0
    mujoco.mj_forward(model, data)
    for _ in range(16):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    print("串联训练代理静态检查通过")


if __name__ == "__main__":
    main()
