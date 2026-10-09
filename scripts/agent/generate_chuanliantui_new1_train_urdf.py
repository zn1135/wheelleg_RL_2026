#!/usr/bin/env python3
"""由 chuanliantui_new_1 真实源 URDF 生成 6-DOF 串联训练派生资产。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui.urdf"
OUTPUT = REPO_ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf"

MAIN_JOINTS = ("rf0", "rf1", "rfwheel", "lf0", "lf1", "lfwheel")
REAR_JOINTS = ("lf00", "lf01", "lf02", "lf03", "rf00", "rf01", "rf02", "rf03")
MARKER_JOINTS = ("imu-link", "lf001", "lf002", "rf001", "rf002", "lf04", "lf05", "rf04", "rf05")
REPARENT = {"lf00": "lf0", "rf00": "rf0"}
# 旧串联训练资产的主链执行器契约；清理其比较目录后仍需保持策略接口一致。
TRAIN_LIMITS = {
    "rf0": {"lower": "-100", "upper": "100", "effort": "40", "velocity": "16.8"},
    "rf1": {"lower": "-0.12", "upper": "0.77", "effort": "40", "velocity": "16.8"},
    "rfwheel": {"effort": "3.9", "velocity": "58.4"},
    "lf0": {"lower": "-100", "upper": "100", "effort": "40", "velocity": "16.8"},
    "lf1": {"lower": "-0.77", "upper": "0.12", "effort": "40", "velocity": "16.8"},
    "lfwheel": {"effort": "3.9", "velocity": "58.4"},
}


def numbers(text):
    return np.fromstring(text, sep=" ")


def fmt(values):
    return " ".join("{:.15g}".format(float(value)) for value in values)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def matrix_to_rpy(matrix):
    pitch = np.arcsin(-np.clip(matrix[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) < 1e-12:
        return np.array((0.0, pitch, np.arctan2(-matrix[0, 1], matrix[1, 1])))
    return np.array(
        (
            np.arctan2(matrix[2, 1], matrix[2, 2]),
            pitch,
            np.arctan2(matrix[1, 0], matrix[0, 0]),
        )
    )


def transform(origin):
    result = np.eye(4)
    result[:3, :3] = rpy_matrix(numbers(origin.attrib.get("rpy", "0 0 0")))
    result[:3, 3] = numbers(origin.attrib["xyz"])
    return result


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
    root = ET.parse(SOURCE).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    child_joint = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}

    poses = {"base_link": np.eye(4)}

    def pose(link):
        if link in poses:
            return poses[link]
        joint = child_joint[link]
        parent = joint.find("parent").attrib["link"]
        poses[link] = pose(parent) @ transform(joint.find("origin"))
        return poses[link]

    # 训练主链沿用已固定的串联训练 limit 契约。
    for name in MAIN_JOINTS:
        joint = joints[name]
        joint.attrib["type"] = "revolute"
        limit = joint.find("limit")
        if limit is not None:
            joint.remove(limit)
        joint.append(ET.Element("limit", TRAIN_LIMITS[name]))

    # 后支链固定到同侧 f0；用零位世界位姿反算每个新的 fixed origin。
    for name in REAR_JOINTS:
        joint = joints[name]
        child = joint.find("child").attrib["link"]
        old_parent = joint.find("parent").attrib["link"]
        new_parent = REPARENT.get(name, old_parent)
        local = np.linalg.inv(pose(new_parent)) @ pose(child)
        joint.find("parent").attrib["link"] = new_parent
        origin = joint.find("origin")
        origin.attrib["xyz"] = fmt(local[:3, 3])
        origin.attrib["rpy"] = fmt(matrix_to_rpy(local[:3, :3]))
        joint.attrib["type"] = "fixed"
        for tag in ("axis", "limit"):
            child_tag = joint.find(tag)
            if child_tag is not None:
                joint.remove(child_tag)

    # IMU、气弹簧与销轴 marker 只是定位物，固定在 base_link 且不保留无效 limit/axis。
    for name in MARKER_JOINTS:
        joint = joints[name]
        joint.attrib["type"] = "fixed"
        for tag in ("axis", "limit"):
            child_tag = joint.find(tag)
            if child_tag is not None:
                joint.remove(child_tag)

    # Isaac Gym 不解析 ROS package:// URI；训练资产使用与旧派生 URDF 相同的相对 mesh 路径。
    package_prefix = "package://chuanliantui/meshes/"
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith(package_prefix):
            mesh.attrib["filename"] = "../meshes/" + filename[len(package_prefix):]

    indent(root)
    ET.ElementTree(root).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print("generated", OUTPUT)


if __name__ == "__main__":
    main()
