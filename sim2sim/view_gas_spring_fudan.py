#!/usr/bin/env python3
"""复旦闭链腿固定机身对照视图：无策略，实体电机为零。"""

import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


DEFAULT_SOURCE = Path("/home/zn/文档/fudan_rl_wheel_leg-main/mujoco/assert_now/"
                      "infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml")


def build_model(source):
    source = Path(source).resolve()
    root = ET.parse(str(source)).getroot()
    # 保留原项目的网格、惯量、阻尼、限位和求解器，只在内存中固定机身。
    for asset in root.find("asset"):
        if asset.get("file"):
            asset.set("file", str((source.parent / asset.get("file")).resolve()))
    base = root.find("worldbody/body[@name='base_Link_del']")
    base.remove(base.find("freejoint"))
    base.set("pos", "0 0 0.8")
    site_info = {site.get("name"): (body.get("name"), site.get("pos", "0 0 0"))
                 for body in root.iter("body") for site in body.findall("site")}
    # MuJoCo 3.2.2 不支持 connect 的 site1/site2；显式保留双方局部锚点。
    anchors = []
    for index, connect in enumerate(root.find("equality")):
        if connect.tag != "connect" or connect.get("site1") is None:
            continue
        site1 = connect.attrib.pop("site1")
        site2 = connect.attrib.pop("site2")
        body1, point1 = site_info[site1]
        body2, point2 = site_info[site2]
        connect.set("body1", body1)
        connect.set("body2", body2)
        connect.set("anchor", point1)
        anchors.append((index, site1, site2, point1, point2))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for index, _, _, point1, point2 in anchors:
        model.eq_data[index, :3] = np.fromstring(point1, sep=" ")
        model.eq_data[index, 3:6] = np.fromstring(point2, sep=" ")
    spring_ids = []
    for side, prefix, color in (("Left", "left", [0, .9, 1, 1]),
                                ("Right", "right", [1, .35, .05, 1])):
        tid = model.tendon(side + "_loop1_tendon").id
        spring_ids.append(model.actuator(side + "_loop1_motor").id)
        model.tendon_width[tid] = .006
        model.tendon_rgba[tid] = color
        for link in ("lf0", "lf1"):
            sid = model.site(prefix + "_" + link + "_qitanhuang").id
            model.site_rgba[sid] = color
            model.site_size[sid, 0] = .009
            model.site_group[sid] = 5
    return model, np.array(spring_ids, dtype=int), anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gas_spring_force", type=float, default=150,
                        help="每侧力 [N]，默认 150 与当前机器人对照；复旦原脚本为 300")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=10)
    args = parser.parse_args()
    if not 0 <= args.gas_spring_force <= 300:
        parser.error("气弹簧力须在 0~300 N 内")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration 必须为有限正数")
    model, spring_ids, anchors = build_model(args.xml)
    data = mujoco.MjData(model)
    state = {"paused": False, "reset": False, "spring": True}
    motor_ids = np.setdiff1d(np.arange(model.nu), spring_ids)

    def update_ctrl():
        data.ctrl[:] = 0
        data.ctrl[spring_ids] = args.gas_spring_force if state["spring"] else 0

    def step():
        update_ctrl()
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError("仿真出现非有限状态")

    def key_callback(key):
        if key == 32:
            state["paused"] = not state["paused"]
        elif key == 268:
            state["reset"] = True
        elif key == 294:
            state["spring"] = not state["spring"]

    update_ctrl()
    mujoco.mj_forward(model, data)
    print("复旦原腿模型：固定机身，无策略，实体电机为零，每侧气弹簧 {} N。".format(
        args.gas_spring_force), flush=True)
    for name in ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint"):
        joint = model.joint(name)
        print(name, "limited=", bool(joint.limited[0]), "range=", joint.range, flush=True)

    if args.headless:
        for _ in range(int(np.ceil(args.duration / model.opt.timestep))):
            step()
        mujoco.mj_forward(model, data)
        assert np.allclose(data.xpos[model.body("base_Link_del").id], [0, 0, .8])
        assert np.all(data.actuator_force[motor_ids] == 0)
        assert np.allclose(data.actuator_force[spring_ids], args.gas_spring_force)
        for _, site1, site2, _, _ in anchors:
            error = np.linalg.norm(data.site(site1).xpos - data.site(site2).xpos)
            print(site1, "闭链端点误差 [m]:", error, flush=True)
            if error >= .001:
                print("注意：原始 site 未完全重合；保留原坐标供对照，不代表闭合质量通过。", flush=True)
        print("运行检查完成：基座固定、电机零力矩、气弹簧力正确，t=", data.time, flush=True)
        return

    from mujoco import viewer as mj_viewer
    print("FUDAN 窗口：左青右橙；双击腿部后 Ctrl+右键拖动。空格暂停，Home复位，F5开关气弹簧。", flush=True)
    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.opt.sitegroup[:] = 0
            viewer.opt.sitegroup[5] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.cam.lookat[:] = [0, 0, .85]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 100
            viewer.cam.elevation = -10
        while viewer.is_running():
            start = time.perf_counter()
            viewer.sync()
            with viewer.lock():
                if state["reset"]:
                    mujoco.mj_resetData(model, data)
                    state["reset"] = False
                update_ctrl()
                if not state["paused"]:
                    for _ in range(5):
                        step()
                mujoco.mj_forward(model, data)
                scene = viewer.user_scn
                scene.ngeom = 0
                for i, (side, joint, aid) in enumerate(zip(
                        ("L", "R"), ("lf1_Joint", "rf1_Joint"), spring_ids)):
                    force = data.actuator_force[aid]
                    torque = force * data.actuator_moment[aid, model.joint(joint).dofadr[0]]
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                                       np.array([0, 0, 1.15 - .07 * i]), np.eye(3).ravel(),
                                       np.array([1, 1, 1, 1], dtype=np.float32))
                    geom.label = "FUDAN {}: {:+.2f} Nm | {:.0f} N".format(side, torque, force)
                    scene.ngeom += 1
            time.sleep(max(0, 5 * model.opt.timestep - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
