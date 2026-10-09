"""气弹簧资产读取检查：MuJoCo tendon、任意轴/RPY、端点变更及左右改名。

只做有限状态下的数值对照，不创建训练 run，不替代策略行为验收。
"""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import isaacgym  # 必须先于 torch
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf"
SOURCE = TRAIN.with_name("chuanliantui.urdf")
# 单独加载数学模块，不初始化旧 Isaac 任务注册表。
spec = importlib.util.spec_from_file_location(
    "gas_geometry", ROOT / "wheel_legged_gym/utils/chuanliantui_gas_spring.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GasSpringGeometry = module.GasSpringGeometry
DEFS = (("rf0", "rf1", "rf001", "rf002"), ("lf0", "lf1", "lf001", "lf002"))


def save(root, directory, name):
    path = Path(directory) / name
    ET.ElementTree(root).write(path, encoding="utf-8")
    return path


def oracle(root):
    """从 URDF 构建只有实体髋/膝和端点的 MuJoCo 模型，独立检查三维变换。"""
    joints = {j.get("name"): j for j in root.findall("joint")}
    child = {j.find("child").get("link"): j for j in joints.values()}
    frames = {"base_link": (np.zeros(3), np.eye(3))}

    def transform(joint):
        node = joint.find("origin")
        xyz = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        matrix = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        return xyz, matrix

    def pose(name):
        if name not in frames:
            joint = child[name]
            p, r = pose(joint.find("parent").get("link"))
            xyz, matrix = transform(joint)
            frames[name] = (p + r @ xyz, r @ matrix)
        return frames[name]

    def fmt(values):
        return " ".join(format(float(v), ".17g") for v in values)

    mj = ET.Element("mujoco")
    ET.SubElement(mj, "compiler", angle="radian")
    world = ET.SubElement(mj, "worldbody")
    tendons = ET.SubElement(mj, "tendon")
    actuators = ET.SubElement(mj, "actuator")
    for i, (hip, knee, upper_marker, lower_marker) in enumerate(DEFS):
        parent = world
        for name in (hip, knee):
            xyz, matrix = transform(joints[name])
            quat = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
            body_name = joints[name].find("child").get("link")
            parent = ET.SubElement(parent, "body", name=body_name, pos=fmt(xyz), quat=fmt(quat))
            ET.SubElement(parent, "joint", name=name, type="hinge",
                          axis=joints[name].find("axis").get("xyz"))
            ET.SubElement(parent, "geom", type="sphere", size="0.01", mass="1",
                          contype="0", conaffinity="0")
            marker = upper_marker if name == hip else lower_marker
            position, rotation = pose(body_name)
            point = rotation.T @ (pose(marker)[0] - position)
            ET.SubElement(parent, "site", name="site_" + marker, pos=fmt(point))
        tendon = ET.SubElement(tendons, "spatial", name="gas_" + str(i))
        ET.SubElement(tendon, "site", site="site_" + upper_marker)
        ET.SubElement(tendon, "site", site="site_" + lower_marker)
        ET.SubElement(actuators, "motor", tendon="gas_" + str(i), gear="1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(mj, encoding="unicode"))
    return model, mujoco.MjData(model)


def check_against_mujoco(path):
    root = ET.parse(path).getroot()
    geometry = GasSpringGeometry(path, dtype=torch.float64)
    model, data = oracle(root)
    hip_ids = np.array([model.joint(d[0]).id for d in DEFS])
    knee_ids = np.array([model.joint(d[1]).id for d in DEFS])
    hip_q, knee_q = model.jnt_qposadr[hip_ids], model.jnt_qposadr[knee_ids]
    hip_v, knee_v = model.jnt_dofadr[hip_ids], model.jnt_dofadr[knee_ids]
    max_error = 0.
    for value in np.linspace(-3.3, 3.3, 19):
        hips, knees = np.array([value, -.73 * value]), np.array([.31 * value, -.17 + .2 * value])
        data.qpos[hip_q], data.qpos[knee_q] = hips, knees
        data.ctrl[:] = [150., 75.]
        mujoco.mj_forward(model, data)
        length, arm = geometry.lengths_and_moment_arms(torch.tensor(knees))
        torque = geometry.knee_torques(torch.tensor(knees), torch.tensor([150., 75.], dtype=torch.float64))
        axes = geometry.knee_axes_in_base(torch.tensor(hips)).numpy()
        np.testing.assert_allclose(axes, data.xaxis[knee_ids], rtol=0, atol=1e-12)
        np.testing.assert_allclose(length.numpy(), data.ten_length, rtol=0, atol=1e-12)
        np.testing.assert_allclose(torque.numpy(), data.qfrc_actuator[knee_v], rtol=0, atol=1e-10)
        np.testing.assert_allclose(data.qfrc_actuator[hip_v], 0., rtol=0, atol=1e-10)
        max_error = max(max_error, float(np.max(np.abs(torque.numpy() - data.qfrc_actuator[knee_v]))))
        for i, qadr in enumerate(knee_q):
            center, eps = data.qpos[qadr], 1e-6
            data.qpos[qadr] = center + eps
            mujoco.mj_forward(model, data)
            plus = data.ten_length[i]
            data.qpos[qadr] = center - eps
            mujoco.mj_forward(model, data)
            minus = data.ten_length[i]
            data.qpos[qadr] = center
            np.testing.assert_allclose(arm[i].item(), (plus - minus) / (2 * eps), rtol=0, atol=1e-10)
    return max_error


def main():
    torch.set_num_threads(1)
    max_error = max(check_against_mujoco(path) for path in (SOURCE, TRAIN))
    baseline = GasSpringGeometry(TRAIN, dtype=torch.float64)
    q = torch.tensor([[.1, -.2], [.6, -.7]], dtype=torch.float64)
    with TemporaryDirectory(prefix="ct-gas-geometry-") as directory:
        # 真正修改定位件坐标：旧的硬编码端点实现无法通过这组检查。
        root = ET.parse(TRAIN).getroot()
        marker = root.find("joint[@name='rf001']/origin")
        point = np.fromstring(marker.get("xyz"), sep=" ")
        point[0] += .012
        marker.set("xyz", " ".join(map(str, point)))
        moved = save(root, directory, "moved.urdf")
        max_error = max(max_error, check_against_mujoco(moved))
        moved_geo = GasSpringGeometry(moved, dtype=torch.float64)
        assert torch.max(torch.abs(moved_geo.knee_torques(q, 150.)[:, 0]
                                   - baseline.knee_torques(q, 150.)[:, 0])) > 1e-3
        torch.testing.assert_close(moved_geo.knee_torques(q, 150.)[:, 1],
                                   baseline.knee_torques(q, 150.)[:, 1], rtol=0, atol=0)

        # origin RPY、非平行斜轴及非单位输入轴，检查轴向与关节坐标系。
        root = ET.parse(TRAIN).getroot()
        for name, rpy, axis in (("rf0", ".2 -.3 .1", "1 2 -3"),
                                ("rf1", "-.4 .2 .3", "-2 1 .5"),
                                ("lf0", "-.1 .4 -.2", ".3 -1 2"),
                                ("lf1", ".1 -.2 -.3", "2 -3 1")):
            joint = root.find("joint[@name='" + name + "']")
            joint.find("origin").set("rpy", rpy)
            joint.find("axis").set("xyz", axis)
        rotated = save(root, directory, "rotated.urdf")
        max_error = max(max_error, check_against_mujoco(rotated))

        # 双向左右改名后，实体不动；相应交换输入/输出即可得到同一物理结果。
        root = ET.parse(TRAIN).getroot()
        for node in root.iter():
            for attr in ("name", "link"):
                value = node.get(attr, "")
                if value.startswith("lf"):
                    node.set(attr, "rf" + value[2:])
                elif value.startswith("rf"):
                    node.set(attr, "lf" + value[2:])
        renamed = GasSpringGeometry(save(root, directory, "renamed.urdf"), dtype=torch.float64)
        torch.testing.assert_close(renamed.knee_torques(q.flip(-1), 150.),
                                   baseline.knee_torques(q, 150.).flip(-1), rtol=0, atol=1e-12)

        # joint 和 child link 不必同名。
        root.find("joint[@name='rf0']").set("name", "right_front_motor")
        custom_defs = (("right_front_motor", "rf1", "rf001", "rf002"), DEFS[1])
        custom = GasSpringGeometry(save(root, directory, "custom.urdf"),
                                   dtype=torch.float64, spring_joints=custom_defs)
        torch.testing.assert_close(custom.knee_torques(q, 150.),
                                   renamed.knee_torques(q, 150.), rtol=0, atol=0)

        # 错误资产必须报错，不能把无效轴或零长度当成零补偿。
        for kind in ("missing_marker", "zero_axis", "coincident"):
            root = ET.parse(TRAIN).getroot()
            if kind == "missing_marker":
                root.remove(root.find("link[@name='rf001']"))
            elif kind == "zero_axis":
                root.find("joint[@name='rf1']/axis").set("xyz", "0 0 0")
            else:
                # 上、下端定位件位于同一零位坐标，存在端点重合。
                root.find("joint[@name='rf001']/origin").set(
                    "xyz", root.find("joint[@name='rf002']/origin").get("xyz"))
            try:
                GasSpringGeometry(save(root, directory, kind + ".urdf"))
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid geometry accepted: " + kind)

    forces = torch.tensor([[0., 150.], [75., 0.]], dtype=torch.float64)
    _, arms = baseline.lengths_and_moment_arms(q)
    torch.testing.assert_close(baseline.knee_torques(q, forces), forces * arms, rtol=0, atol=0)
    torch.testing.assert_close(baseline.knee_torques(q, 0.), torch.zeros_like(q), rtol=0, atol=0)
    # 批量 GPU 数值路径与 CPU 对照，不启动物理仿真或训练。
    if torch.cuda.is_available():
        gpu = GasSpringGeometry(TRAIN, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(gpu.knee_torques(q.cuda(), forces.cuda()).cpu(),
                                   baseline.knee_torques(q, forces), rtol=0, atol=1e-10)
    print("PASS: asset endpoints, arbitrary axes/RPY, naming, force batches, degenerate rejection")
    print("MuJoCo tendon max torque error: {:.3g} Nm; length finite differences passed".format(max_error))
    print("Numeric checks only; no training or policy behavior acceptance.")


if __name__ == "__main__":
    main()
