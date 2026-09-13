#!/usr/bin/env python3
"""核对真实 chuanliantui_new_1 与其 6-DOF 训练派生 URDF。"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui.urdf"
TRAIN = REPO_ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf"
MAIN_JOINTS = ("lf0", "lf1", "lfwheel", "rf0", "rf1", "rfwheel")
REAR_JOINTS = ("rf00", "rf01", "rf02", "rf03", "lf00", "lf01", "lf02", "lf03")
MARKER_JOINTS = ("imu-link", "rf001", "rf002", "lf001", "lf002", "rf04", "rf05", "lf04", "lf05")
TRAIN_LIMITS = {
    "lf0": {"lower": "-100", "upper": "100", "effort": "40", "velocity": "16.8"},
    "lf1": {"lower": "-0.12", "upper": "0.77", "effort": "40", "velocity": "16.8"},
    "lfwheel": {"effort": "3.9", "velocity": "58.4"},
    "rf0": {"lower": "-100", "upper": "100", "effort": "40", "velocity": "16.8"},
    "rf1": {"lower": "-0.77", "upper": "0.12", "effort": "40", "velocity": "16.8"},
    "rfwheel": {"effort": "3.9", "velocity": "58.4"},
}


def numbers(text):
    return np.fromstring(text, sep=" ")


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


def origin_transform(joint):
    origin = joint.find("origin")
    result = np.eye(4)
    result[:3, :3] = rpy_matrix(numbers(origin.attrib.get("rpy", "0 0 0")))
    result[:3, 3] = numbers(origin.attrib["xyz"])
    return result


def zero_poses(root):
    child_joint = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
    poses = {"base_link": np.eye(4)}

    def pose(link):
        if link in poses:
            return poses[link]
        joint = child_joint[link]
        parent = joint.find("parent").attrib["link"]
        poses[link] = pose(parent) @ origin_transform(joint)
        return poses[link]

    for link in child_joint:
        pose(link)
    return poses


def main():
    source = ET.parse(SOURCE).getroot()
    train = ET.parse(TRAIN).getroot()
    source_links = {link.attrib["name"] for link in source.findall("link")}
    train_links = {link.attrib["name"] for link in train.findall("link")}
    source_joints = {joint.attrib["name"]: joint for joint in source.findall("joint")}
    train_joints = {joint.attrib["name"]: joint for joint in train.findall("joint")}

    assert source_links == train_links
    assert set(source_joints) == set(train_joints)
    movable = tuple(name for name, joint in train_joints.items() if joint.attrib["type"] == "revolute")
    fixed = tuple(name for name, joint in train_joints.items() if joint.attrib["type"] == "fixed")
    print("movable=", movable)
    print("fixed_count=", len(fixed))
    assert set(movable) == set(MAIN_JOINTS)
    assert len(fixed) == 17
    assert all(train_joints[name].attrib["type"] == "fixed" for name in REAR_JOINTS + MARKER_JOINTS)
    assert train_joints["rf00"].find("parent").attrib["link"] == "rf0"
    assert train_joints["lf00"].find("parent").attrib["link"] == "lf0"

    for name in MAIN_JOINTS:
        actual = train_joints[name].find("limit").attrib
        expected = TRAIN_LIMITS[name]
        assert actual == expected, "{} limit 不一致: {} != {}".format(name, actual, expected)

    source_poses = zero_poses(source)
    train_poses = zero_poses(train)
    max_position = 0.0
    max_rotation = 0.0
    for link in source_links:
        max_position = max(max_position, float(np.linalg.norm(source_poses[link][:3, 3] - train_poses[link][:3, 3])))
        max_rotation = max(max_rotation, float(np.linalg.norm(source_poses[link][:3, :3] - train_poses[link][:3, :3])))
    print("zero_pose_max_position_m={:.9g}, rotation_frobenius={:.9g}".format(max_position, max_rotation))
    assert max_position < 1e-12 and max_rotation < 1e-12

    mesh_missing = []
    for mesh in train.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        prefix = "../meshes/"
        if not filename.startswith(prefix) or not (TRAIN.parent / filename).resolve().is_file():
            mesh_missing.append(filename)
    assert not mesh_missing, mesh_missing
    print("训练派生 URDF 静态检查通过")


if __name__ == "__main__":
    main()
