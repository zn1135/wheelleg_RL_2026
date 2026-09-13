"""chuanliantui 串联策略与真实闭链 MuJoCo 模型之间的适配。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


class ClosedChainAdapter:
    """将虚拟 [f0, f1, wheel]×左右关节量映射到真实 [f0, f00, wheel]×左右电机。"""

    _SIDES = (
        # 虚拟关节索引, 真实主动/被动 joint, connect site 对, actuator 顺序索引
        ("left", 0, 1, 2, "lf0", "lf1", ("lf00", "lf01", "lf02", "lf03"),
         (("left_rear_pin1", "left_front_pin1"), ("left_rear_pin2", "left_front_pin2")), (0, 1, 2)),
        ("right", 3, 4, 5, "rf0", "rf1", ("rf00", "rf01", "rf02", "rf03"),
         (("right_rear_pin1", "right_front_pin1"), ("right_rear_pin2", "right_front_pin2")), (3, 4, 5)),
    )
    _MAX_JACOBIAN_CONDITION = 1e6

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
            self._actuator("lf0_motor"), self._actuator("lf00_motor"), self._actuator("lfwheel_motor"),
            self._actuator("rf0_motor"), self._actuator("rf00_motor"), self._actuator("rfwheel_motor"),
        ], dtype=np.int32)
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

    def _equality_rows(self, side):
        rows = np.where(
            (self.data.efc_type == self.mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            & np.isin(self.data.efc_id, self.eq_ids[side[0]])
        )[0]
        if len(rows) != 6:
            raise RuntimeError("{} 应有 6 条 equality 行，当前为 {}".format(side[0], len(rows)))
        return rows

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

    def map_virtual_torques(self, virtual_torque):
        """用 equality 速度 Jacobian 计算 tau_motor = J_virtual^T tau_virtual。"""
        self.mujoco.mj_forward(self.model, self.data)
        jacobian = self.data.efc_J.reshape(self.data.nefc, self.model.nv)
        controls = np.zeros(6, dtype=np.float64)
        for side in self._SIDES:
            _, f0_index, f1_index, wheel_index, f0_name, f1_name, chain, _, actuator_indices = side
            rows = self._equality_rows(side)
            active = [self.dof[f0_name], self.dof[chain[0]]]
            passive = [self.dof[f1_name]] + [self.dof[name] for name in chain[1:]]
            passive_jacobian = jacobian[rows][:, passive]
            active_jacobian = jacobian[rows][:, active]
            passive_response, _, rank, singular_values = np.linalg.lstsq(
                passive_jacobian, -active_jacobian, rcond=None
            )
            condition = np.inf if singular_values[-1] == 0 else singular_values[0] / singular_values[-1]
            residual = np.linalg.norm(passive_jacobian @ passive_response + active_jacobian)
            tolerance = 1e-8 * max(1.0, np.linalg.norm(active_jacobian))
            if rank != len(passive) or condition > self._MAX_JACOBIAN_CONDITION or residual > tolerance:
                raise RuntimeError(
                    "{} 闭链 Jacobian 不可用: rank={}, cond={:.3g}, residual={:.3g}".format(
                        side[0], rank, condition, residual
                    )
                )
            virtual_jacobian = np.array(((1.0, 0.0), passive_response[0]), dtype=np.float64)
            leg_motor_torque = virtual_jacobian.T @ virtual_torque[[f0_index, f1_index]]
            hip_index, rear_index, wheel_actuator = actuator_indices
            controls[hip_index] = leg_motor_torque[0]
            controls[rear_index] = leg_motor_torque[1]
            controls[wheel_actuator] = virtual_torque[wheel_index]
        limits = np.array((40.0, 40.0, 3.9, 40.0, 40.0, 3.9))
        return np.clip(controls, -limits, limits)
