#!/usr/bin/env python3
"""在电脑上对照 H7 纯 C 虚拟 PD 与 B 闭链力矩映射；不打开硬件。"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import isaacgym  # 必须先于 sim 间接导入的 torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2sim import mj_sim2sim_ct as sim


Float6 = ct.c_float * 6
Float4 = ct.c_float * 4
Jacobian = (ct.c_float * 2) * 2
Flags6 = ct.c_uint8 * 6


def expected(action, pos, vel, jac, limits):
    virtual = sim.compute_torques(np.asarray(action, dtype=np.float64),
                                  np.asarray((pos[0], pos[1], 0.0,
                                              pos[2], pos[3], 0.0), dtype=np.float64),
                                  np.asarray(vel, dtype=np.float64))
    # B: DM0/2 CAD rear; DM1/3 CAD front.
    mapped = np.array((jac[1][1] * virtual[4],
                       virtual[3] + jac[1][0] * virtual[4],
                       -jac[0][1] * virtual[1],
                       -(virtual[0] + jac[0][0] * virtual[1]),
                       virtual[5], -virtual[2]))
    clipped = np.abs(mapped) > limits
    return virtual, np.clip(mapped, -limits, limits), clipped.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h7-root", type=Path, default=os.environ.get("H7_REPO_PATH"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.h7_root or not args.h7_root.is_dir():
        parser.error("请配置 H7_REPO_PATH 或 --h7-root")
    if args.out.exists():
        parser.error("结果文件已存在")
    import importlib.util
    spec = importlib.util.spec_from_file_location("host_tests", args.h7_root / "tools/test_host.py")
    host_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host_tests)
    compiler, env = host_tests.host_compiler()
    with tempfile.TemporaryDirectory(prefix="policy-control-check-") as temporary:
        shared = Path(temporary) / "policy_control.so"
        command = compiler + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-shared",
                              "-fPIC", "-IApp", "App/lower_policy_control.c", "-lm",
                              "-o", str(shared)]
        subprocess.run(command, cwd=args.h7_root, env=env, check=True)
        library = ct.CDLL(str(shared))
        compute = library.Lower_Policy_Control_Compute
        compute.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                            ct.POINTER(ct.c_float), ct.POINTER(ct.c_float * 2),
                            ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                            ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8)]
        compute.restype = ct.c_bool
        rng = np.random.default_rng(20261001)
        maximum = {"virtual_tau": 0.0, "motor_tau": 0.0}
        for index in range(1000):
            action = np.asarray(rng.uniform(-120, 120, 6), dtype=np.float32)
            pos = np.asarray(rng.uniform(-2, 2, 4), dtype=np.float32)
            vel = np.asarray(rng.uniform(-10, 10, 6), dtype=np.float32)
            jac = np.asarray(rng.uniform(-5, 5, (2, 2)), dtype=np.float32)
            limits = np.asarray((rng.uniform(1, 40), rng.uniform(1, 40),
                                 rng.uniform(1, 40), rng.uniform(1, 40),
                                 rng.uniform(0.1, 3.9), rng.uniform(0.1, 3.9)),
                                dtype=np.float32)
            virtual, motor, saturated = Float6(), Float6(), Flags6()
            ok = compute(Float6(*action), Float4(*pos), Float6(*vel),
                         Jacobian(*(tuple(row) for row in jac)), Float6(*limits),
                         virtual, motor, saturated)
            if not ok:
                raise AssertionError("C rejected finite case {}".format(index))
            reference = expected(action, pos, vel, jac, limits)
            for name, actual, target in (("virtual_tau", virtual, reference[0]),
                                         ("motor_tau", motor, reference[1])):
                error = float(np.max(np.abs(np.asarray(actual) - target)))
                maximum[name] = max(maximum[name], error)
                np.testing.assert_allclose(actual, target, atol=2e-4, rtol=2e-4)
            np.testing.assert_array_equal(saturated, reference[2])
    source = args.h7_root / "App/lower_policy_control.c"
    report = {"samples": 1000, "max_abs_error": maximum,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": "离线数值对照；B 的绝对零点、极性和整机输出仍待实测"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
