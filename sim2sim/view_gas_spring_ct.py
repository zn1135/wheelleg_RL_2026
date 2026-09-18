#!/usr/bin/env python3
"""固定机身的 chuanliantui 气弹簧机构演示，无策略、无电机驱动。"""

import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from chuanliantui_closed_adapter import ClosedChainAdapter


def build_model():
    """在内存中固定基座；不改变用于 sim2sim 的原始 XML。"""
    source = Path(__file__).resolve().with_name("chuanliantui.xml")
    root = ET.parse(str(source)).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir")).resolve()))
    base = root.find("worldbody/body[@name='base_link']")
    base.remove(base.find("freejoint"))
    base.set("pos", "0 0 0.8")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    spring_ids = []
    for side, color in (("left", [0.0, 0.9, 1.0, 1.0]),
                        ("right", [1.0, 0.35, 0.05, 1.0])):
        tid = model.tendon(side + "_gas_spring_tendon").id
        model.tendon_width[tid] = 0.006
        model.tendon_rgba[tid] = color
        spring_ids.append(model.actuator(side + "_gas_spring_motor").id)
        for end in ("upper", "lower"):
            sid = model.site(side + "_gas_spring_" + end).id
            model.site_rgba[sid] = color
            model.site_size[sid, 0] = 0.009
            model.site_group[sid] = 5
    return model, np.asarray(spring_ids, dtype=int)


def spring_readings(model, data, spring_ids):
    """读取气弹簧自身的直接关节力矩，不含重力、鼠标外力或闭链约束反力。"""
    readings = []
    for side, joint, aid in zip(("L", "R"), ("lf1", "rf1"), spring_ids):
        dof = model.joint(joint).dofadr[0]
        # gear=1 的 tendon motor：moment 为带符号力臂 dl/dq [m/rad]。
        arm = float(data.actuator_moment[aid, dof])
        force = float(data.actuator_force[aid])
        readings.append((side, joint, force, arm, force * arm))
    return readings


def draw_spring_readings(scene, model, data, spring_ids):
    """在机身上方显示两行实时读数；只添加渲染元素。"""
    scene.ngeom = 0
    base_pos = data.xpos[model.body("base_link").id]
    colors = ([0.0, 0.9, 1.0, 1.0], [1.0, 0.35, 0.05, 1.0])
    for i, (side, joint, force, arm, torque) in enumerate(
            spring_readings(model, data, spring_ids)):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
            base_pos + np.array([0, 0, 0.20 - 0.07 * i]),
            np.eye(3).ravel(), np.asarray(colors[i], dtype=np.float32))
        geom.label = "{} {}: {:+.2f} N m | F={:.0f} N | arm={:.1f} mm".format(
            side, joint, torque, force, abs(arm) * 1000)
        scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas_spring_force", type=float, default=150.0,
                        help="每侧气弹簧力 [N]，0~150，默认 150")
    parser.add_argument("--headless", action="store_true", help="不打开窗口，运行有限时长检查")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="无窗口检查的仿真时长 [s]，默认 10")
    args = parser.parse_args()
    if not 0 <= args.gas_spring_force <= 150:
        parser.error("--gas_spring_force 必须在 0~150 N 内")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration 必须是有限正数")

    model, spring_ids = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = ClosedChainAdapter(mujoco, model, data)
    adapter.set_virtual_pose(np.array([-0.06, 0.10, 0, 0.06, -0.10, 0]))
    initial_qpos = data.qpos.copy()
    base_id = model.body("base_link").id
    initial_base_pos = data.xpos[base_id].copy()
    controls = {"paused": False, "reset": False, "spring": True}

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        data.ctrl[spring_ids] = args.gas_spring_force if controls["spring"] else 0
        mujoco.mj_forward(model, data)

    def key_callback(key):
        if key == 32:  # Space
            controls["paused"] = not controls["paused"]
        elif key == 268:  # Home
            controls["reset"] = True
        elif key == 294:  # F5
            controls["spring"] = not controls["spring"]
            print("气弹簧：{}".format("开启" if controls["spring"] else "关闭"), flush=True)

    def step():
        data.ctrl[:] = 0  # 实体电机始终无驱动，仅气弹簧施力。
        data.ctrl[spring_ids] = args.gas_spring_force if controls["spring"] else 0
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError("仿真出现非有限状态")

    reset()
    print("固定机身高度 0.8 m；无策略、无电机驱动；每侧气弹簧 {} N。".format(
        args.gas_spring_force), flush=True)
    if args.headless:
        for _ in range(int(np.ceil(args.duration / model.opt.timestep))):
            step()
        mujoco.mj_forward(model, data)
        assert np.allclose(data.xpos[base_id], initial_base_pos, atol=1e-12)
        assert np.all(data.ctrl[adapter.actuator_ids] == 0)
        assert np.allclose(data.actuator_force[spring_ids], args.gas_spring_force)
        print("检查完成：t={:.2f}s，基座固定，电机为零，气弹簧力={} N，长度={} m".format(
            data.time, data.actuator_force[spring_ids], data.ten_length), flush=True)
        for side, joint, force, arm, torque in spring_readings(model, data, spring_ids):
            print("{} {}: 力={:.1f} N，力臂={:.2f} mm，力矩={:+.3f} N·m".format(
                side, joint, force, abs(arm) * 1000, torque), flush=True)
        return

    from mujoco import viewer as mj_viewer

    print("左侧青色，右侧橙色；双击选中腿部，Ctrl+右键拖动施力，Ctrl+左键拖动施加转矩。\n"
          "机身上方实时显示气弹簧对 lf1/rf1 的直接力矩、轴向力 F、有效力臂 arm。\n"
          "空格：暂停/继续；Home：复位；F5：开关气弹簧；关闭窗口退出。", flush=True)
    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[:] = 0
            viewer.opt.sitegroup[5] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.cam.lookat[:] = [0, 0, 0.70]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 100
            viewer.cam.elevation = -10
        while viewer.is_running():
            frame_start = time.perf_counter()
            # sync 接收鼠标施力，并在释放鼠标后清除对应扰动力。
            viewer.sync()
            with viewer.lock():
                if controls["reset"]:
                    reset()
                    controls["reset"] = False
                data.ctrl[:] = 0
                data.ctrl[spring_ids] = args.gas_spring_force if controls["spring"] else 0
                if not controls["paused"]:
                    for _ in range(4):
                        step()
                mujoco.mj_forward(model, data)
                draw_spring_readings(viewer.user_scn, model, data, spring_ids)
            time.sleep(max(0, 4 * model.opt.timestep - (time.perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
