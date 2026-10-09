"""由 CAD URDF 推导气弹簧端点、长度、力臂及实体膝关节力矩。

定位件在 URDF 零位装配中给出安装点；运行时安装点分别随上、下连杆运动。
这里只计算物理气弹簧的广义力，不向电机叠加软件补偿。
"""

from xml.etree import ElementTree as ET

import numpy as np
import torch


def _vector(text, name):
    values = np.asarray([float(v) for v in text.split()], dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Invalid URDF vector: " + name)
    return values


def _origin(joint):
    result = np.eye(4)
    origin = joint.find("origin")
    if origin is None:
        return result
    result[:3, 3] = _vector(origin.get("xyz", "0 0 0"), joint.get("name"))
    roll, pitch, yaw = _vector(origin.get("rpy", "0 0 0"), joint.get("name"))
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                             np.sin(pitch), np.cos(yaw), np.sin(yaw))
    result[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return result


def _axis(joint):
    if joint.get("type") not in ("revolute", "continuous"):
        raise ValueError("Gas spring requires a rotational joint: " + joint.get("name"))
    axis = joint.find("axis")
    result = _vector(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0",
                     joint.get("name"))
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError("Gas spring joint axis is zero: " + joint.get("name"))
    return result / length


def _rotate(vector, axis, angle):
    """Rodrigues 旋转，支持 [..., 弹簧数, 3] 广播；axis 为单位向量。"""
    c, s = torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)
    vector, axis = torch.broadcast_tensors(vector, axis)
    return (c * vector + s * torch.cross(axis, vector, dim=-1)
            + (1 - c) * (vector * axis).sum(dim=-1, keepdim=True) * axis)


class GasSpringGeometry:
    """两个气弹簧的实体关节几何；输出按 spring_joints 的顺序排列。

    每项是 (髋joint名, 膝joint名, 上端定位link名, 下端定位link名)。名称仅
    用于查找，不决定左右轴向。髋必须连接 base_link，膝必须连接髋的子link。
    支持任意单位关节轴及 origin RPY；不从名字猜测正负，不重建虚拟关节。
    """

    def __init__(self, urdf_path, device="cpu", dtype=torch.float32,
                 spring_joints=(("rf0", "rf1", "rf001", "rf002"),
                                ("lf0", "lf1", "lf001", "lf002"))):
        root = ET.parse(urdf_path).getroot()
        elements = root.findall("joint")
        joints = {j.get("name"): j for j in elements}
        links = {link.get("name") for link in root.findall("link")}
        if len(joints) != len(elements):
            raise ValueError("Duplicate URDF joint name")
        child_joints = {j.find("child").get("link"): j for j in elements}
        if len(child_joints) != len(elements):
            raise ValueError("Multiple URDF parents for one link")
        poses, visiting = {"base_link": np.eye(4)}, set()

        def pose(link):
            if link not in links:
                raise ValueError("Missing gas spring URDF link: " + link)
            if link in poses:
                return poses[link]
            if link in visiting or link not in child_joints:
                raise ValueError("Invalid URDF chain at gas spring link: " + link)
            visiting.add(link)
            joint = child_joints[link]
            poses[link] = pose(joint.find("parent").get("link")) @ _origin(joint)
            visiting.remove(link)
            return poses[link]

        if len(spring_joints) != 2:
            raise ValueError("Expected two gas spring joint/marker definitions")
        upper, lower, origins, axes, hip_axes, base_axes, levers = [], [], [], [], [], [], []
        upper_bodies, lower_bodies = [], []
        for hip, knee, upper_marker, lower_marker in spring_joints:
            if hip not in joints or knee not in joints:
                raise ValueError("Missing gas spring hip/knee joint: {} / {}".format(hip, knee))
            hip_joint, knee_joint = joints[hip], joints[knee]
            upper_body = hip_joint.find("child").get("link")
            lower_body = knee_joint.find("child").get("link")
            if (hip_joint.find("parent").get("link") != "base_link"
                    or knee_joint.find("parent").get("link") != upper_body):
                raise ValueError("Gas spring requires base -> hip -> knee topology: " + knee)
            hip_pose, knee_pose = pose(upper_body), pose(lower_body)
            upper_point = hip_pose[:3, :3].T @ (pose(upper_marker)[:3, 3] - hip_pose[:3, 3])
            lower_point = knee_pose[:3, :3].T @ (pose(lower_marker)[:3, 3] - knee_pose[:3, 3])
            knee_origin = _origin(knee_joint)
            knee_axis = _axis(knee_joint)
            axis_parent = knee_origin[:3, :3] @ knee_axis
            # 整个旋转圆周都不能让两个端点重合，避免运行时隐藏奇异力臂。
            delta = knee_origin[:3, 3] - upper_point
            lever = knee_origin[:3, :3] @ lower_point
            parallel = np.dot(delta + lever, axis_parent)
            delta_radial = delta - np.dot(delta, axis_parent) * axis_parent
            lever_radial = lever - np.dot(lever, axis_parent) * axis_parent
            min_length = np.hypot(parallel, np.linalg.norm(delta_radial) - np.linalg.norm(lever_radial))
            if min_length < 1e-8:
                raise ValueError("Gas spring endpoints can coincide: " + knee)
            upper.append(upper_point)
            lower.append(lower_point)
            origins.append(knee_origin[:3, 3])
            levers.append(lever)
            axes.append(axis_parent)
            hip_axes.append(hip_pose[:3, :3] @ _axis(hip_joint))
            base_axes.append(hip_pose[:3, :3] @ axis_parent)
            upper_bodies.append(upper_body)
            lower_bodies.append(lower_body)

        def tensor(values):
            return torch.as_tensor(np.asarray(values), device=device, dtype=dtype)

        self.upper, self.lower = tensor(upper), tensor(lower)
        self.knee_origin, self.zero_levers = tensor(origins), tensor(levers)
        # axes 已转到上连杆坐标系，可直接用于旋转和力矩投影。
        self.axes = tensor(axes)
        self.lever_cross = torch.cross(self.axes, self.zero_levers, dim=-1)
        self.lever_parallel = (self.zero_levers * self.axes).sum(dim=-1, keepdim=True) * self.axes
        self.hip_axes_base, self.zero_knee_axes_base = tensor(hip_axes), tensor(base_axes)
        self.coaxial = bool(np.max(np.abs(np.cross(hip_axes, base_axes))) < 1e-12)
        self.upper_bodies, self.lower_bodies = tuple(upper_bodies), tuple(lower_bodies)

    def lengths_and_moment_arms(self, knee_angles):
        """返回 [..., 2] 长度[m]及带符号 dl/dq[m/rad]；角度是实体膝 q。"""
        if knee_angles.shape[-1:] != (2,):
            raise ValueError("Gas spring knee angles must end in dimension 2")
        c, s = torch.cos(knee_angles).unsqueeze(-1), torch.sin(knee_angles).unsqueeze(-1)
        lever = c * self.zero_levers + s * self.lever_cross + (1 - c) * self.lever_parallel
        delta = self.knee_origin + lever - self.upper
        length = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / length.unsqueeze(-1)
        arm = (torch.cross(lever, direction, dim=-1) * self.axes).sum(dim=-1)
        return length, arm

    def knee_torques(self, knee_angles, force_n):
        """物理推力 F 产生 F*dl/dq；force_n 支持标量或可广播的两侧力值。

        正力沿端点连线向外伸张；两端都随同一髋旋转，串联髋广义力为零。
        力值不经过电机限矩、倍率或动作延迟。
        """
        _, arm = self.lengths_and_moment_arms(knee_angles)
        # 标量直接参与算子，避免每个物理子步向 GPU 上传一个力值 tensor。
        if not isinstance(force_n, (float, int)):
            force_n = torch.as_tensor(force_n, device=arm.device, dtype=arm.dtype)
        return force_n * arm

    def knee_axes_in_base(self, hip_angles):
        """当前膝轴在 base_link 坐标系的方向，包含髋角与安装 RPY。"""
        if hip_angles.shape[-1:] != (2,):
            raise ValueError("Gas spring hip angles must end in dimension 2")
        # 当前 CAD 髋/膝轴共轴；髋转角不改变膝轴，不必重复计算三角函数。
        if self.coaxial:
            return self.zero_knee_axes_base.expand(hip_angles.shape + (3,))
        return _rotate(self.zero_knee_axes_base, self.hip_axes_base, hip_angles)
