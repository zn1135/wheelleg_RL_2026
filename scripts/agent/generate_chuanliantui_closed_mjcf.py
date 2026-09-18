#!/usr/bin/env python3
"""从 chuanliantui_new_1 原始 URDF 生成真实闭链静态 MJCF。

训练仍使用 6-DOF 串联近似资产；本文件只生成 MuJoCo 的真实机构模型，
不接入串联策略的状态或力矩映射。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "sim2sim/chuanliantui.xml"

# 两条真实支链。imu 和气弹簧/销轴标记 link 仅用于推导 site，不能成为独立 DOF。
PHYSICAL_LINKS = {
    "base_link", "rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel",
    "rf00", "rf01", "rf02", "rf03", "lf00", "lf01", "lf02", "lf03",
}
ROOT_CHILDREN = ("rf0", "lf0", "rf00", "lf00")
WHEELS = {"rfwheel", "lfwheel"}
CONTINUOUS_JOINTS = WHEELS | {"lf00", "rf00"}
GAS_SPRING_SITES = (
    ("rf001", "rf0", "right_gas_spring_upper"),
    ("rf002", "rf1", "right_gas_spring_lower"),
    ("lf001", "lf0", "left_gas_spring_upper"),
    ("lf002", "lf1", "left_gas_spring_lower"),
)
MOTOR_LIMITS = {
    "rf0": 40.0, "rf00": 40.0, "rfwheel": 3.9,
    "lf0": 40.0, "lf00": 40.0, "lfwheel": 3.9,
}

# marker link 的 URDF origin 是真实销轴在 base_link 零位坐标系中的位置。
# (marker, 后支链 body1, 前支链 body2, connect name)
CLOSURES = (
    ("rf04", "rf02", "rf0", "rf_loop1"),
    ("rf05", "rf03", "rf1", "rf_loop2"),
    ("lf04", "lf02", "lf0", "lf_loop1"),
    ("lf05", "lf03", "lf1", "lf_loop2"),
)


def numbers(text: str):
    return [float(x) for x in text.split()]


def fmt(values):
    return " ".join("{:.12g}".format(v) for v in values)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def mat_vec(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]


def mat_mul(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat_transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def add(left, right):
    return [left[i] + right[i] for i in range(3)]


def subtract(left, right):
    return [left[i] - right[i] for i in range(3)]


def indent(node, level=0):
    whitespace = "\n" + "  " * level
    if len(node):
        if not node.text or not node.text.strip():
            node.text = whitespace + "  "
        for child in node:
            indent(child, level + 1)
        if not node[-1].tail or not node[-1].tail.strip():
            node[-1].tail = whitespace
    if level and (not node.tail or not node.tail.strip()):
        node.tail = whitespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = ET.parse(args.urdf).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    child_joint = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}

    missing = PHYSICAL_LINKS - set(links)
    if missing:
        raise ValueError("URDF 缺少物理 link: {}".format(sorted(missing)))
    for marker, rear, front, _ in CLOSURES:
        if marker not in joints or rear not in child_joint or front not in child_joint:
            raise ValueError("URDF 缺少闭链标记或 body: {}".format(marker))

    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    poses = {"base_link": ([0.0, 0.0, 0.0], identity)}

    def pose(link_name):
        if link_name in poses:
            return poses[link_name]
        joint = child_joint[link_name]
        parent_name = joint.find("parent").attrib["link"]
        parent_pos, parent_rot = pose(parent_name)
        origin = joint.find("origin")
        local_pos = numbers(origin.attrib["xyz"])
        local_rot = rpy_matrix(numbers(origin.attrib.get("rpy", "0 0 0")))
        world_pos = add(parent_pos, mat_vec(parent_rot, local_pos))
        world_rot = mat_mul(parent_rot, local_rot)
        poses[link_name] = (world_pos, world_rot)
        return poses[link_name]

    sites = []
    connects = []
    for marker, rear, front, connect_name in CLOSURES:
        pin_world, _ = pose(marker)
        rear_pos, rear_rot = pose(rear)
        front_pos, front_rot = pose(front)
        rear_local = mat_vec(mat_transpose(rear_rot), subtract(pin_world, rear_pos))
        front_local = mat_vec(mat_transpose(front_rot), subtract(pin_world, front_pos))
        side = "right" if marker.startswith("r") else "left"
        suffix = "1" if marker.endswith("04") else "2"
        sites.extend(((rear, "{}_rear_pin{}".format(side, suffix), rear_local, "1 0 0 1"),
                      (front, "{}_front_pin{}".format(side, suffix), front_local, "1 1 0 1")))
        connects.append((connect_name, rear, front, rear_local))

    for marker, body_name, site_name in GAS_SPRING_SITES:
        marker_pos, _ = pose(marker)
        body_pos, body_rot = pose(body_name)
        local = mat_vec(mat_transpose(body_rot), subtract(marker_pos, body_pos))
        sites.append((body_name, site_name, local, "1 0 0 1"))

    mj = ET.Element("mujoco", {"model": "chuanliantui_closed"})
    mj.append(ET.Comment(
        "由 chuanliantui_new_1/urdf/chuanliantui.urdf 生成；真实闭链静态模型。"
        " 含气弹簧恒定伸张推力执行器；不含 IMU 或串联策略映射。"
    ))
    ET.SubElement(mj, "compiler", {
        "angle": "radian",
        "meshdir": "../resources/robots/chuanliantui_new_1/meshes",
        "autolimits": "true",
    })
    ET.SubElement(mj, "option", {
        "timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "implicitfast",
        "solver": "Newton", "iterations": "100", "tolerance": "1e-10",
        "cone": "elliptic", "impratio": "10",
    })
    defaults = ET.SubElement(mj, "default")
    ET.SubElement(defaults, "joint", {"damping": "0", "armature": "0", "frictionloss": "0"})
    ET.SubElement(defaults, "geom", {
        "contype": "1", "conaffinity": "1", "condim": "3", "friction": "0.5 0.005 0.0001",
    })
    visual = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2", "density": "0"})
    collision = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", {"type": "mesh", "group": "3", "contype": "1", "conaffinity": "0"})
    ET.SubElement(defaults, "motor", {"ctrllimited": "true"})

    assets = ET.SubElement(mj, "asset")
    for link_name in sorted(PHYSICAL_LINKS):
        mesh_file = "base_link_simple.STL" if link_name == "base_link" else "{}.STL".format(link_name)
        ET.SubElement(assets, "mesh", {"name": link_name, "file": mesh_file})
    ET.SubElement(assets, "material", {"name": "metal", "rgba": "0.75 0.75 0.75 1"})
    ET.SubElement(assets, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.2 0.3 0.4",
        "rgb2": "0.1 0.15 0.2", "width": "512", "height": "512",
    })
    ET.SubElement(assets, "material", {"name": "grid", "texture": "grid", "texrepeat": "4 4", "reflectance": "0.1"})

    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(worldbody, "light", {"directional": "true", "pos": "0 0 4", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "20 20 0.1", "material": "grid",
        "condim": "3", "friction": "0.5 0.005 0.0001",
    })

    children = {}
    for name in PHYSICAL_LINKS - {"base_link"}:
        joint = child_joint[name]
        parent = joint.find("parent").attrib["link"]
        if parent in PHYSICAL_LINKS:
            children.setdefault(parent, []).append(name)

    site_map = {}
    for body, name, local, rgba in sites:
        site_map.setdefault(body, []).append((name, local, rgba))

    def add_inertial(body, link_name):
        inertial = links[link_name].find("inertial")
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        ET.SubElement(body, "inertial", {
            "pos": origin.attrib["xyz"],
            "mass": inertial.find("mass").attrib["value"],
            "fullinertia": fmt([
                float(inertia.attrib["ixx"]), float(inertia.attrib["iyy"]), float(inertia.attrib["izz"]),
                float(inertia.attrib["ixy"]), float(inertia.attrib["ixz"]), float(inertia.attrib["iyz"]),
            ]),
        })

    def add_body(parent, link_name):
        joint = child_joint[link_name]
        origin = joint.find("origin")
        body = ET.SubElement(parent, "body", {"name": link_name, "pos": origin.attrib["xyz"]})
        joint_attrs = {"name": joint.attrib["name"], "type": "hinge", "axis": joint.find("axis").attrib["xyz"]}
        if link_name in CONTINUOUS_JOINTS:
            joint_attrs["limited"] = "false"
        else:
            limit = joint.find("limit")
            joint_attrs["range"] = "{} {}".format(limit.attrib["lower"], limit.attrib["upper"])
        ET.SubElement(body, "joint", joint_attrs)
        add_inertial(body, link_name)
        ET.SubElement(body, "geom", {"class": "visual", "mesh": link_name, "material": "metal"})
        ET.SubElement(body, "geom", {"class": "collision", "mesh": link_name})
        for site_name, local, rgba in site_map.get(link_name, []):
            ET.SubElement(body, "site", {"name": site_name, "pos": fmt(local), "size": "0.004", "rgba": rgba})
        for child in sorted(children.get(link_name, [])):
            add_body(body, child)

    base = ET.SubElement(worldbody, "body", {"name": "base_link", "pos": "0 0 0.35"})
    ET.SubElement(base, "freejoint", {"name": "floating_base"})
    add_inertial(base, "base_link")
    ET.SubElement(base, "geom", {"class": "visual", "mesh": "base_link", "material": "metal"})
    ET.SubElement(base, "geom", {"class": "collision", "mesh": "base_link"})
    for child in ROOT_CHILDREN:
        add_body(base, child)

    equality = ET.SubElement(mj, "equality")
    for name, rear, front, anchor in connects:
        ET.SubElement(equality, "connect", {
            "name": name, "body1": rear, "body2": front, "anchor": fmt(anchor),
            "solref": "0.01 1", "solimp": "0.95 0.99 0.001 0.5 2",
        })

    tendons = ET.SubElement(mj, "tendon")
    tendons.append(ET.Comment("正执行器力沿 site 连线推开两端，使气弹簧伸长。"))
    for side in ("left", "right"):
        tendon = ET.SubElement(tendons, "spatial", {
            "name": side + "_gas_spring_tendon", "width": "0.001",
        })
        for end in ("upper", "lower"):
            ET.SubElement(tendon, "site", {"site": side + "_gas_spring_" + end})

    actuators = ET.SubElement(mj, "actuator")
    for joint_name in ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel"):
        limit = MOTOR_LIMITS[joint_name]
        ET.SubElement(actuators, "motor", {
            "name": "{}_motor".format(joint_name), "joint": joint_name, "gear": "1",
            "ctrlrange": "-{} {}".format(limit, limit),
        })

    actuators.append(ET.Comment("每侧恒定伸张推力由控制端写入，默认 150 N，0 N 关闭。"))
    for side in ("left", "right"):
        ET.SubElement(actuators, "motor", {
            "name": side + "_gas_spring_motor", "tendon": side + "_gas_spring_tendon",
            "gear": "1", "ctrlrange": "0 150",
        })

    indent(mj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mj).write(args.output, encoding="utf-8", xml_declaration=True)
    print("generated", args.output)
    for name, rear, front, anchor in connects:
        print("{}: {} -> {} anchor={}".format(name, rear, front, fmt(anchor)))


if __name__ == "__main__":
    main()
