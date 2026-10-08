"""CAD 气弹簧在串联主链上的等效膝力矩；输入为物理关节角。

端点由 CAD lf001/lf002、rf001/rf002 换算到 f0/f1 局部系，
与 sim2sim/chuanliantui.xml 的 gas_spring sites 一致。
不经过 H7 电机 Jacobian，不作为电机输出或软件补偿。
"""

from xml.etree import ElementTree as ET

import torch


UPPER_POINTS = (
    (0.025606057708123, 0.003500000000049, -0.037105298931257),
    (0.0256060577081279, -0.003500000000051, -0.0371052989312678),
)
LOWER_POINTS = (
    (0.019692562545793, -0.010000000000277, -0.044464367480479),
    (0.019692562545789, 0.010000000000237, -0.044464367480282),
)


class GasSpringGeometry:
    """只支持当前 CAD 的共面 y 轴主链；资产改变时显式报错。"""

    def __init__(self, urdf_path, device="cpu", dtype=torch.float32):
        joints = {j.attrib["name"]: j for j in ET.parse(urdf_path).getroot().findall("joint")}
        origins, axes = [], []
        for hip, knee in (("lf0", "lf1"), ("rf0", "rf1")):
            for name in (hip, knee):
                joint = joints[name]
                rpy = [float(v) for v in joint.find("origin").get("rpy", "0 0 0").split()]
                axis = [float(v) for v in joint.find("axis").attrib["xyz"].split()]
                if any(abs(v) > 1e-10 for v in rpy) or axis != [0., -1. if name[0] == "l" else 1., 0.]:
                    raise ValueError("Gas spring requires the current CAD y-axis main chain: " + name)
            origins.append([float(v) for v in joints[knee].find("origin").attrib["xyz"].split()])
            axes.append(axis)
        self.upper = torch.tensor(UPPER_POINTS, device=device, dtype=dtype)
        self.lower = torch.tensor(LOWER_POINTS, device=device, dtype=dtype)
        self.knee_origin = torch.tensor(origins, device=device, dtype=dtype)
        self.axes = torch.tensor(axes, device=device, dtype=dtype)

    def knee_torques(self, knee_angles, force_n):
        """返回 [..., 2] 左/右膝力矩；正力沿端点连线向外伸张。

        所有计算在大腿局部系内完成。气弹簧长度不随整体髋角改变，
        因此对串联髋关节的广义力矩为零。
        """
        angle = knee_angles * self.axes[:, 1]
        c, s = torch.cos(angle), torch.sin(angle)
        lever = torch.stack((
            c * self.lower[:, 0] + s * self.lower[:, 2],
            self.lower[:, 1].expand_as(angle),
            -s * self.lower[:, 0] + c * self.lower[:, 2],
        ), dim=-1)
        delta = self.knee_origin + lever - self.upper
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        force = float(force_n) * delta / length.clamp_min(1e-9)
        return (torch.cross(lever, force, dim=-1) * self.axes).sum(dim=-1)
