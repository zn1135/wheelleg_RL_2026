#!/usr/bin/env python3
"""CPU 对照实际 H7 C 的五杆解算与 RL 力矩链；不连接硬件或运行策略。

显式指定 --h7-repo；真实 C 源码保持只读，host 头替身与共享库生成于 /tmp。
LEG_TRIG_LIBM=1 使用 libm，不代表 MCU/CMSIS 查表实现的位级一致性。
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

STUBS = {
    "dm.h": """#ifndef HOST_DM_H
#define HOST_DM_H
#include <stdint.h>
enum { DM_MOTOR_LEG_F_LFT, DM_MOTOR_LEG_B_LFT, DM_MOTOR_LEG_F_RGT, DM_MOTOR_LEG_B_RGT, DM_MOTOR_NUM };
#endif
""",
    "dji.h": """#ifndef HOST_DJI_H
#define HOST_DJI_H
#define DJI_MOTOR_WHEEL_LFT 0u
#define DJI_MOTOR_WHEEL_RGT 1u
#define DJI_MOTOR_NUM 2u
#endif
""",
    "ai_platform.h": "typedef void *ai_handle; typedef struct { int unused; } ai_buffer;\n",
    "robot_control.h": '#include "machine_config.h"\n#define CTRL_DT MACHINE_CTRL_DT\n',
}

WRAPPER = r'''
#include <math.h>
#include <string.h>
#include "leg_solver.h"
#include "rl_torque.h"
#include "machine_config.h"
static int solve(const float *q, const float *v, leg_state_t *leg) {
    Leg_Init(leg);
    leg->config.lu = machine->leg_lu;
    leg->config.lg = machine->leg_lg;
    leg->config.configured = 1;
    leg->input.hip_f = q[0]; leg->input.hip_b = q[1];
    leg->input.d_hip_f = v[0]; leg->input.d_hip_b = v[1];
    return Leg_Solve(leg);
}
int bridge_solver(const float *input, float *output) {
    leg_state_t leg;
    memset(output, 0, 6 * sizeof(float));
    if (!solve(input, input + 2, &leg)) return 0;
    output[0] = leg.output.thigh_angle;
    output[1] = leg.output.virtual_shank_angle;
    output[2] = leg.output.d_virtual_shank_angle;
    output[3] = leg.output.vshank_jac[1];
    output[4] = leg.output.vshank_jac[0];
    output[5] = leg.output.virtual_leg_length;
    return 1;
}
int bridge_control(const float *angles, const float *speeds,
                   const float *wheels, const float *action, float *output) {
    leg_state_t legs[2]; rl_torque_param_t param; rl_torque_state_t state;
    torque_output_t torque; float action_fw[6];
    memset(output, 0, 24 * sizeof(float));
    if (!solve(angles, speeds, &legs[0]) || !solve(angles + 2, speeds + 2, &legs[1])) return 0;
    for (int i = 0; i < 6; ++i) {
        if (!isfinite(action[i])) return 0;
        action_fw[i] = clampf(action[i], -100.0f, 100.0f) * machine->rl.sign[i];
    }
    RL_Torque_Param_Init(&param, RL_MODEL_STANDUP);
    RL_Torque_State_Init(&state, &param);
    if (!RL_Torque_Compute(&legs[0], &legs[1], &param, wheels, action_fw, &state, &torque)) return 0;
    for (int side = 0; side < 2; ++side) {
        int i = 3 * side;
        output[i] = machine->rl.sign[i] * Angle_Wrap_180(legs[side].output.thigh_angle - machine->rl.zero[2 * side]);
        output[i + 1] = machine->rl.sign[i + 1] * Angle_Wrap_180(legs[side].output.virtual_shank_angle - machine->rl.zero[2 * side + 1]);
        output[6 + i] = machine->rl.sign[i] * speeds[2 * side];
        output[7 + i] = machine->rl.sign[i + 1] * legs[side].output.d_virtual_shank_angle;
        output[8 + i] = machine->rl.sign[i + 2] * wheels[side];
        output[18 + i] = torque.dm[2 * side];
        output[19 + i] = torque.dm[2 * side + 1];
        output[20 + i] = torque.dji[side];
    }
    for (int i = 0; i < 6; ++i) output[12 + i] = machine->rl.sign[i] * state.virtual_torque[i];
    return 1;
}
void bridge_config(float *output) {
    for (int i = 0; i < 6; ++i) output[i] = machine->rl.sign[i];
    for (int i = 0; i < 4; ++i) output[6 + i] = machine->rl.zero[i];
    output[10] = machine->leg_lu; output[11] = machine->leg_lg;
    output[12] = machine->dm_trq_clamp; output[13] = machine->dji_trq_clamp;
    output[14] = machine->gas_comp_sign[0]; output[15] = machine->gas_comp_sign[1];
}
'''


def compile_reference(h7_repo, directory):
    algorithm = h7_repo / "imcalib/Algorithm"
    library = h7_repo / "imcalib/user-lib"
    sources = [algorithm / "leg_solver.c", algorithm / "rl_torque.c",
               library / "pid.c", library / "machine_config.c"]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, content in STUBS.items():
        (directory / name).write_text(content)
    wrapper = directory / "bridge.c"
    wrapper.write_text(WRAPPER)
    output = directory / "h7_reference.so"
    command = shlex.split(os.environ.get("CC", "cc")) + [
        "-shared", "-fPIC", "-O2", "-std=c99", "-DLEG_TRIG_LIBM=1",
        "-I" + str(directory), "-I" + str(algorithm), "-I" + str(library),
        str(wrapper), *map(str, sources), "-lm", "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    reference = ctypes.CDLL(str(output))
    array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    reference.bridge_solver.argtypes = [array, array]
    reference.bridge_solver.restype = ctypes.c_int
    reference.bridge_control.argtypes = [array] * 5
    reference.bridge_control.restype = ctypes.c_int
    reference.bridge_config.argtypes = [array]
    reference.bridge_config.restype = None
    return reference


def check_solver(reference, module):
    rng = np.random.default_rng(1907)
    inputs = [[2.54, .70, 0., 0.], [2.9, 1.1, -2., 4.],
              [-3.2, .8, 7., -1.], [8.7, -4.4, -3., 8.]]
    for _ in range(96):
        front, rear = rng.uniform(-5.0, 5.0, 2)
        if abs(np.sin((front - rear) / 2)) > .03:
            inputs.append([front, rear, *rng.uniform(-10, 10, 2)])
    for values in inputs:
        values = np.asarray(values, dtype=np.float32)
        expected = np.zeros(6, dtype=np.float32)
        valid = bool(reference.bridge_solver(values, expected))
        result = module.solve_h7_leg(*values)
        assert result.valid == valid, values
        if valid:
            actual = np.array([result.thigh_angle, result.virtual_shank_angle,
                               result.d_virtual_shank_angle, *result.jacobian,
                               result.virtual_leg_length])
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=8e-5)
    for values in ([0., 0., 0., 0.], [np.nan, 1., 0., 0.],
                   [2., np.inf, 0., 0.], [2., 1., np.nan, 0.]):
        expected = np.zeros(6, dtype=np.float32)
        valid = reference.bridge_solver(np.asarray(values, dtype=np.float32), expected)
        result = module.solve_h7_leg(*values)
        assert not valid and not result.valid
        assert np.isfinite(result.jacobian).all()
    print("PASS: H7 leg_solver.c geometry/velocity/Jacobian and invalid inputs (libm)")


def check_adapter(reference, module):
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "sim2sim/chuanliantui.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = module.H7ClosedChainAdapter(mujoco, model, data)
    # 父类仍只用于 CAD 闭合初态；运行时解算随后独立使用 H7 主动轴算法。
    initial_pose = np.array([-.06, .10, 0., .06, -.10, 0.])
    adapter.set_virtual_pose(initial_pose)
    initial_q, initial_dq = adapter.read_policy_state()
    assert adapter.valid
    np.testing.assert_allclose(initial_q, initial_pose, atol=2e-6)
    np.testing.assert_array_equal(initial_dq, np.zeros(6))
    joint_names = ["lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel"]
    jids = [model.joint(name).id for name in joint_names]
    qadr, vadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    axes = model.jnt_axis[jids, 1]
    front_zero = [np.arctan2(-model.body_pos[model.body(n).id, 2],
                            model.body_pos[model.body(n).id, 0]) for n in ("lf1", "rf1")]
    rear_zero = [np.arctan2(-model.body_pos[model.body(n).id, 2],
                           model.body_pos[model.body(n).id, 0]) for n in ("lf01", "rf01")]
    config = np.zeros(16, dtype=np.float32)
    reference.bridge_config(config)
    np.testing.assert_allclose(front_zero, config[[6, 8]], atol=1e-6)
    assert np.array_equal(config[14:], [0., 0.]), "gas compensation must remain disabled"
    rng = np.random.default_rng(401)
    actions = [np.zeros(6), [1., -2., 3., -.5, 1.2, -4.],
               [200., -200., 200., -150., 150., -200.]]
    actions += list(rng.uniform(-12, 12, (32, 6)))
    for action in actions:
        data.qpos[qadr] = rng.uniform(-.3, .3, 6)
        data.qvel[vadr] = rng.uniform(-25, 25, 6)
        h7q = np.array([front_zero[0] + axes[0] * data.qpos[qadr[0]],
                       rear_zero[0] + axes[1] * data.qpos[qadr[1]],
                       front_zero[1] + axes[3] * data.qpos[qadr[3]],
                       rear_zero[1] + axes[4] * data.qpos[qadr[4]]], dtype=np.float32)
        h7v = (axes[[0, 1, 3, 4]] * data.qvel[vadr[[0, 1, 3, 4]]]).astype(np.float32)
        wheel = (axes[[2, 5]] * data.qvel[vadr[[2, 5]]]).astype(np.float32)
        expected = np.zeros(24, dtype=np.float32)
        assert reference.bridge_control(h7q, h7v, wheel, np.asarray(action, dtype=np.float32), expected)
        q, dq = adapter.read_policy_state()
        tau, motor = adapter.compute_control(action)
        assert adapter.valid and adapter.last_jacobians.shape == (2, 2)
        np.testing.assert_allclose(q, expected[:6], atol=2e-5)
        np.testing.assert_allclose(dq, expected[6:12], rtol=2e-5, atol=8e-5)
        np.testing.assert_allclose(tau, expected[12:18], rtol=2e-5, atol=5e-4)
        np.testing.assert_allclose(motor, axes * expected[18:24], rtol=2e-5, atol=5e-4)
        assert np.all(np.abs(motor) <= np.array([40, 40, 3.9, 40, 40, 3.9]))

    # 被动膝和支链状态故意污染，主动轴不变；H7 读数和控制必须保持不变。
    q_before, dq_before = adapter.read_policy_state()
    torque_before = adapter.compute_control(np.ones(6))
    for name in ("lf1", "rf1", "lf01", "lf02", "lf03", "rf01", "rf02", "rf03"):
        joint = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint]] = np.nan
        data.qvel[model.jnt_dofadr[joint]] = np.nan
    q, dq = adapter.read_policy_state()
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(dq, dq_before)
    for actual, expected in zip(adapter.compute_control(np.ones(6)), torque_before):
        np.testing.assert_array_equal(actual, expected)
    for invalid in ([np.nan] * 6, [np.inf] * 6, [1., 2.]):
        assert all(np.array_equal(x, np.zeros(6)) for x in adapter.compute_control(invalid))
    for index in range(6):
        original = data.qvel[vadr[index]]
        data.qvel[vadr[index]] = np.nan
        assert all(np.array_equal(x, np.zeros(6)) for x in adapter.compute_control(np.ones(6)))
        assert not adapter.valid
        data.qvel[vadr[index]] = original

    # 后轴注册覆盖必须进入求解；它是CAD名义坐标参数，不是实测编码器零点。
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    override = np.asarray(rear_zero) + [.05, -.03]
    adapter = module.H7ClosedChainAdapter(mujoco, model, data, rear_zero=override)
    q, _ = adapter.read_policy_state()
    for i, slot in enumerate((1, 4)):
        result = module.solve_h7_leg(front_zero[i], override[i])
        expected = config[slot] * ((result.virtual_shank_angle - config[7 + 2 * i] + np.pi) % (2 * np.pi) - np.pi)
        np.testing.assert_allclose(q[slot], expected, atol=1e-6)
    print("PASS: actual H7 rl_torque/pid/config C vs adapter, signs/wheel limits/wrap/saturation")
    print("PASS: active-shaft-only state, rear registration override and fail-closed outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h7-repo", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ct-h7-c-check-", dir="/tmp") as directory:
        reference = compile_reference(args.h7_repo.resolve(), Path(directory))
        expected = np.zeros(6, dtype=np.float32)
        assert reference.bridge_solver(np.array([2.54, .70, 0., 0.], dtype=np.float32), expected)
        print("C reference compiled from", args.h7_repo.resolve())
        assert importlib.util.find_spec("sim2sim.chuanliantui_h7_adapter") is not None, "H7 control adapter is not implemented"
        module = importlib.import_module("sim2sim.chuanliantui_h7_adapter")
        check_solver(reference, module)
        check_adapter(reference, module)
    print("CPU C/Python numerical comparison only; no firmware bitwise or real-robot validation.")


if __name__ == "__main__":
    main()
