#!/usr/bin/env python3
"""由 Isaac Gym 的 chuanliantui_train.urdf 生成串联训练代理 MJCF。

该模型用于 Isaac 串联训练资产的一致性回放，不含真实机构的 connect 闭链约束，
不得作为真机或真实闭链机构的动力学验证模型。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "sim2sim/chuanliantui_train_proxy.xml"
GAS_SPRING_SOURCE = REPO_ROOT / "sim2sim/chuanliantui.xml"

DOF_NAMES = ("rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel")
WHEEL_JOINTS = {"rfwheel", "lfwheel"}
TORQUE_LIMITS = {
    "rf0": 40.0,
    "rf1": 40.0,
    "rfwheel": 3.9,
    "lf0": 40.0,
    "lf1": 40.0,
    "lfwheel": 3.9,
}


def _indent(node: ET.Element, level: int = 0) -> None:
    whitespace = "\n" + "  " * level
    if len(node):
        if not node.text or not node.text.strip():
            node.text = whitespace + "  "
        for child in node:
            _indent(child, level + 1)
        if not node[-1].tail or not node[-1].tail.strip():
            node[-1].tail = whitespace
    if level and (not node.tail or not node.tail.strip()):
        node.tail = whitespace


def _mesh_name(link: ET.Element) -> str:
    if link.attrib["name"] == "base_link":
        # 原始 CAD 是超出 MuJoCo STL 面数上限的 ASCII 网格；仓库中已有
        # 保持外形的简化版本，真实闭链 XML 也使用同一文件。
        return "base_link_simple.STL"
    mesh = link.find("./visual/geometry/mesh")
    if mesh is None:
        mesh = link.find("./collision/geometry/mesh")
    if mesh is None:
        raise ValueError("link '{}' 缺少 mesh".format(link.attrib["name"]))
    return Path(mesh.attrib["filename"]).name


def _add_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError("link '{}' 缺少 inertial".format(link.attrib["name"]))
    origin = inertial.find("origin")
    inertia = inertial.find("inertia")
    mass = inertial.find("mass")
    if origin is None or inertia is None or mass is None:
        raise ValueError("link '{}' inertial 不完整".format(link.attrib["name"]))
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": origin.attrib["xyz"],
            "mass": mass.attrib["value"],
            "fullinertia": " ".join(
                inertia.attrib[name]
                for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
        },
    )


def _add_gas_springs(mj: ET.Element, actuators: ET.Element) -> None:
    """复制闭链前支路上的物理弹簧；六个策略电机保留在执行器列表前六位。"""
    source = ET.parse(GAS_SPRING_SOURCE).getroot()
    tendons = ET.Element("tendon")
    mj.insert(list(mj).index(actuators), tendons)
    for side, prefix in (("right", "rf"), ("left", "lf")):
        for end, suffix in (("upper", "0"), ("lower", "1")):
            body_name = prefix + suffix
            path = ".//body[@name='{}']".format(body_name)
            original, target = source.find(path), mj.find(path)
            if original is None or target is None:
                raise ValueError("气弹簧端点缺少 body: " + body_name)
            # 两个模型的前支路使用同一个局部坐标系，才能直接复制 CAD 端点。
            for attr, default in (("pos", "0 0 0"), ("quat", "1 0 0 0")):
                expected = tuple(map(float, original.get(attr, default).split()))
                actual = tuple(map(float, target.get(attr, default).split()))
                if len(actual) != len(expected) or any(
                    not math.isclose(a, b, rel_tol=0., abs_tol=1e-10)
                    for a, b in zip(actual, expected)
                ):
                    raise ValueError("气弹簧端点局部系不一致: {} {}".format(body_name, attr))
            name = side + "_gas_spring_" + end
            site = original.find("./site[@name='{}']".format(name))
            if site is None:
                raise ValueError("闭链 XML 缺少气弹簧 site: " + name)
            target.append(deepcopy(site))
        tendon_name = side + "_gas_spring_tendon"
        tendon = source.find("./tendon/spatial[@name='{}']".format(tendon_name))
        motor = source.find("./actuator/motor[@name='{}_gas_spring_motor']".format(side))
        if tendon is None or motor is None or motor.get("tendon") != tendon_name:
            raise ValueError("闭链 XML 气弹簧 tendon/motor 不完整: " + side)
        tendons.append(deepcopy(tendon))
        actuators.append(deepcopy(motor))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = ET.parse(args.urdf).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    child_joint = {
        joint.find("child").attrib["link"]: joint for joint in root.findall("joint")
    }
    movable = tuple(name for name, joint in joints.items() if joint.attrib["type"] == "revolute")
    if set(movable) != set(DOF_NAMES):
        raise ValueError("训练 URDF 的可动 DOF 不匹配: {}".format(movable))
    if "base_link" not in links:
        raise ValueError("训练 URDF 缺少 base_link")

    children: dict[str, list[str]] = {}
    for child_name, joint in child_joint.items():
        parent_name = joint.find("parent").attrib["link"]
        children.setdefault(parent_name, []).append(child_name)

    def ordered_children(parent_name: str) -> list[str]:
        names = children.get(parent_name, [])
        preferred = {
            "base_link": ("rf0", "lf0"),
            "rf0": ("rf1", "rf00"),
            "rf1": ("rfwheel",),
            "lf0": ("lf1", "lf00"),
            "lf1": ("lfwheel",),
        }.get(parent_name, ())
        order = {name: index for index, name in enumerate(preferred)}
        return sorted(names, key=lambda name: (order.get(name, len(order)), name))

    mj = ET.Element("mujoco", {"model": "chuanliantui_train_proxy"})
    mj.append(
        ET.Comment(
            "由 chuanliantui_train.urdf 生成；仅用于 Isaac 串联训练代理一致性回放，"
            "不含真实闭链 connect 约束，不可用于真实机构验证；"
            "气弹簧端点和传动复制自 chuanliantui.xml，由回放脚本每侧施加 150 N。"
        )
    )
    ET.SubElement(
        mj,
        "compiler",
        {
            "angle": "radian",
            "meshdir": "../resources/robots/chuanliantui_new_1/meshes",
            "autolimits": "true",
        },
    )
    ET.SubElement(
        mj,
        "option",
        {
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": "100",
            "tolerance": "1e-10",
            "cone": "elliptic",
            "impratio": "10",
        },
    )
    defaults = ET.SubElement(mj, "default")
    ET.SubElement(defaults, "joint", {"damping": "0", "armature": "0", "frictionloss": "0"})
    ET.SubElement(
        defaults,
        "geom",
        {"contype": "1", "conaffinity": "1", "condim": "3", "friction": "0.5 0.005 0.0001"},
    )
    visual = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(
        visual,
        "geom",
        {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2", "density": "0"},
    )
    collision = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", {"type": "mesh", "group": "3", "contype": "1", "conaffinity": "0"})
    ET.SubElement(defaults, "motor", {"ctrllimited": "true"})

    assets = ET.SubElement(mj, "asset")
    for link_name in sorted(links):
        ET.SubElement(assets, "mesh", {"name": link_name, "file": _mesh_name(links[link_name])})
    ET.SubElement(assets, "material", {"name": "metal", "rgba": "0.75 0.75 0.75 1"})
    ET.SubElement(assets, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.15 0.2", "width": "512", "height": "512"})
    ET.SubElement(assets, "material", {"name": "grid", "texture": "grid", "texrepeat": "4 4", "reflectance": "0.1"})

    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(worldbody, "light", {"directional": "true", "pos": "0 0 4", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "20 20 0.1", "material": "grid", "condim": "3", "friction": "0.5 0.005 0.0001"})

    def add_body(parent: ET.Element, link_name: str, origin: ET.Element | None = None) -> None:
        attrs = {"name": link_name}
        if origin is not None:
            attrs["pos"] = origin.attrib["xyz"]
        body = ET.SubElement(parent, "body", attrs)
        if link_name == "base_link":
            ET.SubElement(body, "freejoint", {"name": "floating_base"})
        else:
            joint = child_joint[link_name]
            if joint.attrib["type"] == "revolute":
                joint_attrs = {"name": joint.attrib["name"], "type": "hinge", "axis": joint.find("axis").attrib["xyz"]}
                if joint.attrib["name"] in WHEEL_JOINTS:
                    joint_attrs.update(
                        limited="false", damping="0.002", frictionloss="0.001"
                    )
                else:
                    joint_attrs.update(damping="0.03", frictionloss="0.015")
                    if joint.attrib["name"] in ("rf1", "lf1"):
                        # 对齐 Isaac 的膝限位柔度，避免站立载荷下过度越过硬行程。
                        joint_attrs["solreflimit"] = "0.004 1"
                    limit = joint.find("limit")
                    if limit is None:
                        raise ValueError("关节 '{}' 缺少 limit".format(joint.attrib["name"]))
                    joint_attrs["range"] = "{} {}".format(limit.attrib["lower"], limit.attrib["upper"])
                ET.SubElement(body, "joint", joint_attrs)
        _add_inertial(body, links[link_name])
        ET.SubElement(body, "geom", {"class": "visual", "mesh": link_name, "material": "metal"})
        ET.SubElement(body, "geom", {"class": "collision", "mesh": link_name})
        for child_name in ordered_children(link_name):
            child_origin = child_joint[child_name].find("origin")
            if child_origin is None:
                raise ValueError("关节 '{}' 缺少 origin".format(child_joint[child_name].attrib["name"]))
            add_body(body, child_name, child_origin)

    add_body(worldbody, "base_link")
    actuators = ET.SubElement(mj, "actuator")
    for joint_name in DOF_NAMES:
        limit = TORQUE_LIMITS[joint_name]
        ET.SubElement(
            actuators,
            "motor",
            {"name": "{}_motor".format(joint_name), "joint": joint_name, "gear": "1", "ctrlrange": "-{} {}".format(limit, limit)},
        )
    _add_gas_springs(mj, actuators)

    _indent(mj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mj).write(args.output, encoding="utf-8", xml_declaration=True)
    print("generated", args.output)
    print("movable_dofs={}".format(DOF_NAMES))


if __name__ == "__main__":
    main()
