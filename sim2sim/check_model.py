#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""imcawl.xml 模型自检：碰撞对 + 轮圆柱位置。改完 XML 后跑一次。

用法: python sim2sim/check_model.py
"""
import os
import sys

import mujoco
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from mj_sim2sim import DEFAULT_DOF_POS, JOINT_NAMES  # noqa: E402

m = mujoco.MjModel.from_xml_path(os.path.join(_THIS_DIR, "imcawl.xml"))
d = mujoco.MjData(m)
print(f"XML 编译 OK  ngeom={m.ngeom}")
qpos_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES]

def contacts(tag, expect_leg_ground=None):
    pairs = set()
    for i in range(d.ncon):
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1])
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2])
        pairs.add((b1, b2))
    print(f"{tag}: {sorted(pairs) if pairs else '无接触'}")
    robot_internal = [p for p in pairs if "world" not in p]
    assert not robot_internal, f"出现自碰撞(应对齐训练 self_collisions 关闭): {robot_internal}"
    return pairs

# 1) 站立位：默认角真实站高≈0.291(轮心相对基座-0.2285+半径0.0625)，取 z=0.28 保证轮子压地
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.28
for i in range(6):
    d.qpos[qpos_adr[i]] = DEFAULT_DOF_POS[i]
mujoco.mj_forward(m, d)
pairs = contacts("站立位接触对(应只有轮-地)")
assert pairs, "站立位应有轮-地接触，但检测到 0 个接触"
assert all("wheel" in p[0] + p[1] for p in pairs), f"站立位出现非轮接触: {pairs}"

# 2) 低位下压：逐步降低基座直到腿部网格触地，验证腿碰撞体生效；全程不得出现自碰撞
leg_contact_z = None
for z in [0.12, 0.10, 0.08, 0.06, 0.04, 0.02]:
    d.qpos[2] = z
    for i in range(6):
        d.qpos[qpos_adr[i]] = DEFAULT_DOF_POS[i]
    mujoco.mj_forward(m, d)
    pairs = contacts(f"z={z:.2f} 接触对")
    if any(("f0" in p[0] + p[1]) or ("f1" in p[0] + p[1]) for p in pairs):
        leg_contact_z = z
        break
assert leg_contact_z is not None, "压到 z=0.02 仍无腿-地接触：腿部碰撞体未生效！"
print(f"腿部碰撞体生效 ✓ (z={leg_contact_z:.2f} 时腿网格触地)")

# 3) 轮碰撞圆柱位置：真实胶面中心在 x=∓0.2354(=0.2479-0.0125)
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.30
for i in range(6):
    d.qpos[qpos_adr[i]] = DEFAULT_DOF_POS[i]
mujoco.mj_forward(m, d)
for n, expect_x in [("l_wheel_link", -0.2354), ("r_wheel_link", +0.2354)]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
        if m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            x = d.geom_xpos[g][0]
            print(f"{n} 碰撞圆柱中心 x={x:+.4f} (期望 {expect_x:+.4f}) z={d.geom_xpos[g][2]:.4f}")
            assert abs(x - expect_x) < 1e-3, f"{n} 圆柱位置偏差过大"

print("全部自检通过 ✓")
