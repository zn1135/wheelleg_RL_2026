"""保留 CAD 闭链机构，在其主动轴状态上运行 H7 的五杆解算和 RL 控制。

算法/常量对应 H7 的 leg_solver.c、rl_torque.c、machine_config.c 大机器配置。
H7 第一逻辑腿对应模型 lf，第二腿对应 rf；CAD 已按物理轮命名，不再交换轮。
H7 平面角采用 (x, -z)：q_H7 = CAD 连杆名义方向角 + axis_y * q_CAD。
后轴注册只是仿真坐标对齐，可显式覆盖，不是实测编码器零点。这里不叠加
驱动层 dm_zero/dm_sign，也不使用另一部署仓库的偏置。物理气弹簧由 MJ 主循环施力。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from .chuanliantui_closed_adapter import ClosedChainAdapter
except ImportError:  # 兼容 sim2sim 脚本直接执行。
    from chuanliantui_closed_adapter import ClosedChainAdapter

H7_RL_SIGN = np.array([-1., -1., -1., 1., 1., 1.])
H7_RL_ZERO = np.array([2.476872, 3.086386, 2.476872, 3.086386])
DEFAULT_DOF_POS = np.array([-.06, .10, 0., .06, -.10, 0.])
TORQUE_LIMITS = np.array([40., 40., 3.9, 40., 40., 3.9])
LEG_INDICES = np.array([0, 1, 3, 4])
WHEEL_INDICES = np.array([2, 5])


def _wrap(angle):
    """H7 的闭区间 [-pi, pi] 约定，保留正向恰为 pi 的边界。"""
    value = np.asarray(angle, dtype=np.float64)
    wrapped = (value + np.pi) % (2.0 * np.pi) - np.pi
    return np.where((wrapped == -np.pi) & (value > 0), np.pi, wrapped)


@dataclass(frozen=True)
class H7LegSolution:
    valid: bool = False
    thigh_angle: float = 0.0
    virtual_shank_angle: float = 0.0
    d_virtual_shank_angle: float = 0.0
    # d(vshank_H7)/d(front_H7, rear_H7)，与 C 数组 [rear, front] 交换列。
    jacobian: np.ndarray = field(default_factory=lambda: np.zeros(2))
    virtual_leg_length: float = 0.0


def solve_h7_leg(front, rear, front_vel=0.0, rear_vel=0.0):
    """复刻 H7 大机器五杆几何与虚拟膝速度；无效输入返回有限的零结果。

    固定 lu=.21、lg=.25，与 H7 大机器表一致。Python/libm 数值对照不代表
    MCU 上 CMSIS 查表三角函数的位级一致性。此函数不使用 CAD 被动关节状态。
    """
    values = np.asarray([front, rear, front_vel, rear_vel], dtype=np.float64)
    if not np.isfinite(values).all():
        return H7LegSolution()
    front, rear, front_vel, rear_vel = values
    lu, lg = .21, .25
    xa, ya = lu * np.cos(front), lu * np.sin(front)
    xb, yb = lu * np.cos(rear), lu * np.sin(rear)
    dx, dy = xb - xa, yb - ya
    lg_dx2, lg_dy2 = 2.0 * lg * dx, 2.0 * lg * dy
    ab_sq = dx * dx + dy * dy
    disc = lg_dx2 * lg_dx2 + lg_dy2 * lg_dy2 - ab_sq * ab_sq
    if disc < -1e-6:
        return H7LegSolution()
    phi_a = 2.0 * np.arctan2(lg_dy2 + np.sqrt(max(disc, 0.0)), lg_dx2 + ab_sq)
    xp, yp = xa + lg * np.cos(phi_a), ya + lg * np.sin(phi_a)
    phi_b = np.arctan2(yp - yb, xp - xb)
    length = np.hypot(xp, yp)
    sin_ab = np.sin(phi_a - phi_b)
    if length < 1e-3 or abs(sin_ab) < 1e-6:
        return H7LegSolution()
    jac_rear = lu * np.sin(rear - phi_b) / (lg * sin_ab)
    jac_front = -lu * np.sin(front - phi_b) / (lg * sin_ab) - 1.0
    jacobian = np.array([jac_front, jac_rear])
    with np.errstate(over="ignore", invalid="ignore"):
        velocity = jac_front * front_vel + jac_rear * rear_vel
    if not np.isfinite([*jacobian, velocity, length]).all():
        return H7LegSolution()
    return H7LegSolution(
        valid=True,
        thigh_angle=float(_wrap(front)),
        virtual_shank_angle=float(_wrap(phi_a - front - np.pi / 2.0)),
        d_virtual_shank_angle=float(velocity),
        jacobian=jacobian,
        virtual_leg_length=float(length),
    )


class H7ClosedChainAdapter(ClosedChainAdapter):
    """H7 逻辑状态/控制与 CAD 主动轴之间的适配；初态仍用父类 CAD 闭合求解。"""

    def __init__(self, mujoco, model, data, rear_zero=None, wheel_vel_limit=20.0):
        super().__init__(mujoco, model, data)
        if wheel_vel_limit is not None and (not np.isfinite(wheel_vel_limit) or wheel_vel_limit <= 0):
            raise ValueError("wheel_vel_limit must be positive or None")
        self.wheel_vel_limit = wheel_vel_limit
        names = ("rf0", "rf00", "rfwheel", "lf0", "lf00", "lfwheel")
        joint_ids = np.array([model.joint(name).id for name in names])
        axes = model.jnt_axis[joint_ids]
        if not np.allclose(axes[:, [0, 2]], 0) or not np.allclose(np.abs(axes[:, 1]), 1):
            raise ValueError("H7 adapter requires CAD active joints on the local y axis")
        self.motor_axis_y = axes[:, 1].copy()
        self.motor_qpos = model.jnt_qposadr[joint_ids].copy()
        self.motor_dof = model.jnt_dofadr[joint_ids].copy()
        self.front_zero = np.array([
            np.arctan2(-self.geometry[side].front1_offset[1],
                       self.geometry[side].front1_offset[0])
            for side in ("right", "left")
        ])
        if not np.allclose(self.front_zero, H7_RL_ZERO[[0, 2]], rtol=0, atol=1e-5):
            raise ValueError("CAD front directions do not match H7 rl.zero: {}".format(self.front_zero))
        nominal_rear = np.array([
            np.arctan2(-self.geometry[side].rear1_offset[1],
                       self.geometry[side].rear1_offset[0])
            for side in ("right", "left")
        ])
        self.rear_zero = nominal_rear if rear_zero is None else np.asarray(rear_zero, dtype=float).copy()
        if self.rear_zero.shape != (2,) or not np.isfinite(self.rear_zero).all():
            raise ValueError("rear_zero must contain two finite CAD-to-H7 registration angles")
        self.valid = False
        self.last_jacobians = np.zeros((2, 2))
        self.leg_solutions = (H7LegSolution(), H7LegSolution())
        self._firmware_pos = np.zeros(6)
        self._firmware_vel = np.zeros(6)

    def read_policy_state(self):
        """主动前/后轴解算为训练 q/dq；轮绝对角为0，轮速直接读取当前主动轴。

        last_jacobians 的两行为 lf/rf，两列为 H7 前/后电机角对虚拟膝的导数。
        任一腿或轮速无效时整组返回零，valid=False，不把坏状态送入策略。
        """
        self.valid = False
        self.last_jacobians[:] = 0.0
        self._firmware_pos[:] = 0.0
        self._firmware_vel[:] = 0.0
        position = np.zeros(6)
        velocity = np.zeros(6)
        solutions = [H7LegSolution(), H7LegSolution()]
        for side in range(2):
            i = side * 3
            front = self.front_zero[side] + self.motor_axis_y[i] * self.data.qpos[self.motor_qpos[i]]
            rear = self.rear_zero[side] + self.motor_axis_y[i + 1] * self.data.qpos[self.motor_qpos[i + 1]]
            vf, vb, vw = self.motor_axis_y[i:i + 3] * self.data.qvel[self.motor_dof[i:i + 3]]
            solution = solve_h7_leg(front, rear, vf, vb)
            solutions[side] = solution
            if not solution.valid or not np.isfinite(vw):
                self.leg_solutions = tuple(solutions)
                return np.zeros(6), np.zeros(6)
            position[i:i + 2] = [solution.thigh_angle, solution.virtual_shank_angle]
            velocity[i:i + 3] = [vf, solution.d_virtual_shank_angle, vw]
        self.leg_solutions = tuple(solutions)
        self.last_jacobians[:] = [solution.jacobian for solution in solutions]
        self._firmware_pos[:] = position
        self._firmware_vel[:] = velocity
        train_pos = np.zeros(6)
        train_pos[LEG_INDICES] = H7_RL_SIGN[LEG_INDICES] * _wrap(position[LEG_INDICES] - H7_RL_ZERO)
        self.valid = True
        return train_pos, H7_RL_SIGN * velocity

    def compute_control(self, action):
        """训练动作→H7 PD/虚拟力矩限幅→H7 Jacobian→实体限幅→CAD 广义力矩。"""
        self.read_policy_state()
        try:
            action = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError):
            return np.zeros(6), np.zeros(6)
        if not self.valid or action.shape != (6,) or not np.isfinite(action).all():
            return np.zeros(6), np.zeros(6)
        action_fw = H7_RL_SIGN * np.clip(action, -100.0, 100.0)
        target = H7_RL_ZERO + H7_RL_SIGN[LEG_INDICES] * DEFAULT_DOF_POS[LEG_INDICES]
        target += .5 * action_fw[LEG_INDICES]
        virtual_fw = np.zeros(6)
        virtual_fw[LEG_INDICES] = (
            10.0 * _wrap(target - self._firmware_pos[LEG_INDICES])
            - self._firmware_vel[LEG_INDICES]
        )
        wheel_target = 10.0 * action_fw[WHEEL_INDICES]
        if self.wheel_vel_limit is not None:
            wheel_target = np.clip(wheel_target, -self.wheel_vel_limit, self.wheel_vel_limit)
        virtual_fw[WHEEL_INDICES] = .1 * (wheel_target - self._firmware_vel[WHEEL_INDICES])
        virtual_fw = np.clip(virtual_fw, -TORQUE_LIMITS, TORQUE_LIMITS)
        motor_fw = np.zeros(6)
        for side in range(2):
            i = 3 * side
            jac_front, jac_rear = self.last_jacobians[side]
            # 当前 H7 gas_comp_sign=[0,0]：不另加软件弹簧补偿。
            motor_fw[i] = virtual_fw[i] + virtual_fw[i + 1] * jac_front
            motor_fw[i + 1] = virtual_fw[i + 1] * jac_rear
            motor_fw[i + 2] = virtual_fw[i + 2]
        if not np.isfinite(motor_fw).all():
            return np.zeros(6), np.zeros(6)
        motor_fw = np.clip(motor_fw, -TORQUE_LIMITS, TORQUE_LIMITS)
        return H7_RL_SIGN * virtual_fw, self.motor_axis_y * motor_fw
