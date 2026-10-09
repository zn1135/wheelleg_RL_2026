"""chuanliantui 串联策略与真实闭链 MuJoCo 模型之间的适配。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _xz(value):
    return np.asarray(value, dtype=np.float64)[[0, 2]]


def _rotate(angle, value):
    c, s = np.cos(angle), np.sin(angle)
    return np.array((c * value[0] - s * value[1], s * value[0] + c * value[1]))


def _perpendicular(value):
    return np.array((-value[1], value[0]))


def _angle(value):
    return np.arctan2(value[1], value[0])


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class _PlanarLegGeometry:
    """从本机 MJCF 尺寸解两个电机角到被动膝角，不读取被动关节状态。"""

    def __init__(self, model, side):
        self.name, _, _, _, front, _, rear_chain, pins, _ = side
        rear, rear1, rear2, rear3 = rear_chain
        body_names = (front, side[5]) + rear_chain
        for name in body_names:
            if not np.allclose(model.body_quat[model.body(name).id], (1, 0, 0, 0), atol=1e-8):
                raise RuntimeError("{} 几何反算要求 body 局部坐标轴未旋转".format(self.name))
            if np.linalg.norm(model.jnt_pos[model.joint(name).id]) > 1e-8:
                raise RuntimeError("{} 几何反算要求关节位于 body 原点".format(self.name))

        def body_pos(name):
            return _xz(model.body_pos[model.body(name).id])

        def site_pos(name):
            return _xz(model.site_pos[model.site(name).id])

        self.front_origin = body_pos(front)
        self.front1_offset = body_pos(side[5])
        self.front_pin1 = site_pos(pins[0][1])
        self.front_pin2 = site_pos(pins[1][1])
        self.rear_origin = body_pos(rear)
        self.rear1_offset = body_pos(rear1)
        self.rear2_offset = body_pos(rear2)
        self.rear3_offset = body_pos(rear3)
        self.rear_pin1 = site_pos(pins[0][0])
        self.rear_pin2 = site_pos(pins[1][0])

        axes = [model.jnt_axis[model.joint(name).id] for name in body_names]
        if any(abs(axis[0]) > 1e-8 or abs(axis[2]) > 1e-8
               or abs(abs(axis[1]) - 1.0) > 1e-8 for axis in axes):
            raise RuntimeError("{} 几何反算要求各关节绕局部 y 轴转动".format(self.name))
        if any(axis[1] * axes[0][1] < 0 for axis in axes[1:]):
            raise RuntimeError("{} 关节轴方向不一致".format(self.name))
        # x-z 平面正角方向与 MuJoCo 绕 y 轴的正向相反。
        self.sign = -float(axes[0][1])

        # 零位 CAD 装配定义闭链的交叉/非交叉分支；运行中保持同一装配分支。
        first_origin = self.rear_origin + self.rear1_offset
        first_target = self.front_origin + self.front_pin1
        first_elbow = first_origin + self.rear2_offset
        self.first_branch = self._branch(first_target - first_origin,
                                         first_elbow - first_origin)
        second_origin = self.front_origin + self.front1_offset
        second_target = (self.rear_origin + self.rear1_offset +
                         self.rear2_offset + self.rear3_offset)
        second_pin = second_origin + self.front_pin2
        self.second_branch = self._branch(second_target - second_origin,
                                          second_pin - second_origin)
        self._cached_motor_angles = None
        self._cached_result = None

    def _branch(self, baseline, point):
        cross = np.linalg.det(np.stack((baseline, point)))
        if abs(cross) < 1e-10:
            raise RuntimeError("{} 零位几何位于奇异分支".format(self.name))
        return 1.0 if cross > 0.0 else -1.0

    def _circle_point(self, first, second, first_radius, second_radius, branch):
        delta = second - first
        distance = np.linalg.norm(delta)
        if distance < 1e-9 or distance > first_radius + second_radius + 1e-4 or \
                distance < abs(first_radius - second_radius) - 1e-4:
            raise RuntimeError("{} 电机角超出闭链几何可达范围".format(self.name))
        along = (first_radius ** 2 - second_radius ** 2 + distance ** 2) / (2.0 * distance)
        height_sq = first_radius ** 2 - along ** 2
        if height_sq < -1e-6:
            raise RuntimeError("{} 闭链圆交点不存在".format(self.name))
        direction = delta / distance
        return first + along * direction + branch * np.sqrt(max(height_sq, 0.0)) * \
            _perpendicular(direction)

    def knee_and_jacobian(self, front_motor, rear_motor):
        """返回膝角及 [d膝角/d前电机角, d膝角/d后电机角]。"""
        motor_angles = (float(front_motor), float(rear_motor))
        if motor_angles == self._cached_motor_angles:
            return self._cached_result
        front_angle = self.sign * front_motor
        rear_angle = self.sign * rear_motor

        front_pin1 = self.front_origin + _rotate(front_angle, self.front_pin1)
        rear1_origin = self.rear_origin + _rotate(rear_angle, self.rear1_offset)
        rear2_origin = self._circle_point(
            rear1_origin, front_pin1, np.linalg.norm(self.rear2_offset),
            np.linalg.norm(self.rear_pin1), self.first_branch)
        rear2_angle = _angle(front_pin1 - rear2_origin) - _angle(self.rear_pin1)

        front1_origin = self.front_origin + _rotate(front_angle, self.front1_offset)
        rear3_origin = rear2_origin + _rotate(rear2_angle, self.rear3_offset)
        front_pin2 = self._circle_point(
            front1_origin, rear3_origin, np.linalg.norm(self.front_pin2),
            np.linalg.norm(self.rear_pin2), self.second_branch)
        front1_angle = _angle(front_pin2 - front1_origin) - _angle(self.front_pin2)
        knee = _wrap(self.sign * front1_angle - front_motor)

        # 两个闭环销各提供 x/z 两个方程；对几何约束求导得到被动膝角的
        # 电机 Jacobian。这个映射不依赖 MuJoCo 的被动 qpos 或 efc_J。
        rear_pin1 = front_pin1
        rear_pin2 = front_pin2
        passive = np.zeros((4, 4), dtype=np.float64)  # [rear1,rear2,front1,rear3]
        motors = np.zeros((4, 2), dtype=np.float64)   # [front0,rear0]
        sign = self.sign
        motors[:2, 0] = -sign * _perpendicular(front_pin1 - self.front_origin)
        motors[:2, 1] = sign * _perpendicular(rear_pin1 - self.rear_origin)
        passive[:2, 0] = sign * _perpendicular(rear_pin1 - rear1_origin)
        passive[:2, 1] = sign * _perpendicular(rear_pin1 - rear2_origin)
        motors[2:, 0] = -sign * _perpendicular(front_pin2 - self.front_origin)
        motors[2:, 1] = sign * _perpendicular(rear_pin2 - self.rear_origin)
        passive[2:, 0] = sign * _perpendicular(rear_pin2 - rear1_origin)
        passive[2:, 1] = sign * _perpendicular(rear_pin2 - rear2_origin)
        passive[2:, 2] = -sign * _perpendicular(front_pin2 - front1_origin)
        passive[2:, 3] = sign * _perpendicular(rear_pin2 - rear3_origin)
        condition = np.linalg.cond(passive)
        if not np.isfinite(condition) or condition > 1e6:
            raise RuntimeError("{} 闭链几何 Jacobian 奇异：cond={:.3g}".format(
                self.name, condition))
        knee_jacobian = np.linalg.solve(passive, -motors)[2]
        self._cached_motor_angles = motor_angles
        self._cached_result = (float(knee), knee_jacobian)
        return self._cached_result


class ClosedChainAdapter:
    """将虚拟 [f0, f1, wheel]×左右关节量映射到真实 [f0, f00, wheel]×左右电机。"""

    _SIDES = (
        # 虚拟关节索引, 真实主动/被动 joint, connect site 对, actuator 顺序索引
        ("right", 0, 1, 2, "rf0", "rf1", ("rf00", "rf01", "rf02", "rf03"),
         (("right_rear_pin1", "right_front_pin1"), ("right_rear_pin2", "right_front_pin2")), (0, 1, 2)),
        ("left", 3, 4, 5, "lf0", "lf1", ("lf00", "lf01", "lf02", "lf03"),
         (("left_rear_pin1", "left_front_pin1"), ("left_rear_pin2", "left_front_pin2")), (3, 4, 5)),
    )
    def __init__(self, mujoco, model, data):
        self.mujoco = mujoco
        self.model = model
        self.data = data
        if model.neq != 4:
            raise RuntimeError("闭链模型必须含 4 个 connect，当前 neq={}".format(model.neq))
        self.qpos = {}
        self.dof = {}
        self.site = {}
        for side in self._SIDES:
            for name in (side[4], side[5]) + side[6]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise RuntimeError("MJCF 缺少关节: {}".format(name))
                self.qpos[name] = model.jnt_qposadr[jid]
                self.dof[name] = model.jnt_dofadr[jid]
            for rear, front in side[7]:
                for name in (rear, front):
                    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                    if sid < 0:
                        raise RuntimeError("MJCF 缺少闭链 site: {}".format(name))
                    self.site[name] = sid
        self.actuator_ids = np.array([
            self._actuator("rf0_motor"), self._actuator("rf00_motor"), self._actuator("rfwheel_motor"),
            self._actuator("lf0_motor"), self._actuator("lf00_motor"), self._actuator("lfwheel_motor"),
        ], dtype=np.int32)
        self.geometry = {
            side[0]: _PlanarLegGeometry(model, side) for side in self._SIDES
        }
        equality_rows = np.where(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
        if len(equality_rows) != 12:
            raise RuntimeError("4 个 connect 应产生 12 条 equality 行，当前为 {}".format(len(equality_rows)))
        self.eq_ids = {}
        for side in self._SIDES:
            _, _, _, _, f0_name, f1_name, chain, _, _ = side
            expected_pairs = ((chain[2], f0_name), (chain[3], f1_name))
            ids = []
            for body1_name, body2_name in expected_pairs:
                body1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
                body2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
                matches = [
                    eq_id for eq_id in range(model.neq)
                    if model.eq_obj1id[eq_id] == body1 and model.eq_obj2id[eq_id] == body2
                ]
                if len(matches) != 1:
                    raise RuntimeError("无法唯一定位 {} 的 connect: {} -> {}".format(side[0], body1_name, body2_name))
                ids.append(matches[0])
            self.eq_ids[side[0]] = tuple(ids)

    def _actuator(self, name):
        aid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError("MJCF 缺少实体执行器: {}".format(name))
        return aid

    def _pin_residual(self, side):
        return np.concatenate([
            self.data.site_xpos[self.site[rear]] - self.data.site_xpos[self.site[front]]
            for rear, front in side[7]
        ])

    def set_virtual_pose(self, virtual_dof):
        """以训练端虚拟 f0/f1 目标角求闭链一致的所有被动关节初值。"""
        solutions = []
        for side in self._SIDES:
            _, f0_index, f1_index, _, f0_name, f1_name, chain, _, _ = side
            target_f0 = float(virtual_dof[f0_index])
            target_f1 = float(virtual_dof[f1_index])

            def residual(x):
                self.mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[self.qpos[f0_name]] = target_f0
                self.data.qpos[self.qpos[f1_name]] = target_f1
                for name, value in zip(chain, x):
                    self.data.qpos[self.qpos[name]] = value
                self.mujoco.mj_forward(self.model, self.data)
                return self._pin_residual(side)

            result = least_squares(
                residual, np.zeros(4), bounds=(-np.pi, np.pi), xtol=1e-13,
                ftol=1e-13, gtol=1e-13, max_nfev=2000,
            )
            if not result.success or np.max(np.abs(residual(result.x))) > 1e-8:
                raise RuntimeError("{} 闭链初值求解失败: {}".format(side[0], result.message))
            solutions.append((side, target_f0, target_f1, result.x))

        self.mujoco.mj_resetData(self.model, self.data)
        for side, target_f0, target_f1, values in solutions:
            _, _, _, _, f0_name, f1_name, chain, _, _ = side
            self.data.qpos[self.qpos[f0_name]] = target_f0
            self.data.qpos[self.qpos[f1_name]] = target_f1
            for name, value in zip(chain, values):
                self.data.qpos[self.qpos[name]] = value
        self.mujoco.mj_forward(self.model, self.data)
        error = max(np.max(np.abs(self._pin_residual(side))) for side in self._SIDES)
        if error > 1e-8:
            raise RuntimeError("闭链初值残差过大: {:.3g} m".format(error))
        for side in self._SIDES:
            name, _, _, _, front, knee_name, chain, _, _ = side
            predicted, _ = self.geometry[name].knee_and_jacobian(
                self.data.qpos[self.qpos[front]], self.data.qpos[self.qpos[chain[0]]])
            actual = self.data.qpos[self.qpos[knee_name]]
            if abs(_wrap(predicted - actual)) > 1e-5:
                raise RuntimeError("{} 初始姿态与零位装配分支不一致".format(name))

    def read_virtual_leg_state(self):
        """仅由前/后电机的位置和速度反算左右虚拟膝状态。"""
        state = {}
        for side in self._SIDES:
            name, _, _, _, front, _, chain, _, _ = side
            rear = chain[0]
            knee, jacobian = self.geometry[name].knee_and_jacobian(
                self.data.qpos[self.qpos[front]], self.data.qpos[self.qpos[rear]])
            motor_vel = np.array((self.data.qvel[self.dof[front]],
                                  self.data.qvel[self.dof[rear]]))
            state[name] = (knee, float(jacobian @ motor_vel))
        return state

    def map_virtual_torques(self, virtual_torque):
        """用本机机构几何的 Jacobian 映射虚拟力矩到两个实体电机。"""
        controls = np.zeros(6, dtype=np.float64)
        for side in self._SIDES:
            name, f0_index, f1_index, wheel_index, f0_name, _, chain, _, actuator_indices = side
            _, knee_jacobian = self.geometry[name].knee_and_jacobian(
                self.data.qpos[self.qpos[f0_name]], self.data.qpos[self.qpos[chain[0]]])
            leg_motor_torque = np.array((
                virtual_torque[f0_index] + knee_jacobian[0] * virtual_torque[f1_index],
                knee_jacobian[1] * virtual_torque[f1_index],
            ))
            hip_index, rear_index, wheel_actuator = actuator_indices
            controls[hip_index] = leg_motor_torque[0]
            controls[rear_index] = leg_motor_torque[1]
            controls[wheel_actuator] = virtual_torque[wheel_index]
        limits = np.array((40.0, 40.0, 3.9, 40.0, 40.0, 3.9))
        return np.clip(controls, -limits, limits)
