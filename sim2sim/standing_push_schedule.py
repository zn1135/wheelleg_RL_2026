"""共享站立推扰计划：世界系牛顿，固定 2 ms 子步，脉冲区间左闭右开。

JSON 最小格式为 {"version": 1, "events": [
    {"t_start_s": 5.0, "duration_s": 0.1, "force_world_n": [10., 0., 0.]}
] }。可选 physics_dt_s 必须为 0.002；可选 duration_s 限定计划总时长。
load(path) / generate() 返回计划对象，force_at(t_s) 返回本子步施加的三维力。
外部计划可使用任意有限非零力；默认生成器限定水平 5–15 N。
只描述开环推扰，不把推后恢复运动状态称为原地稳站。
"""

import argparse
from copy import deepcopy
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np


PHYSICS_DT_S = 0.002
EARLIEST_PUSH_S = 5.0
ALIGNMENT_TOLERANCE_S = 1e-8


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return value


def _step(value, name):
    value = _number(value, name)
    if value < 0:
        raise ValueError(name + " must be nonnegative")
    if value / PHYSICS_DT_S > 2 ** 53 - 1:
        raise ValueError(name + " exceeds exactly representable physics steps")
    step = round(value / PHYSICS_DT_S)
    if abs(value - step * PHYSICS_DT_S) > ALIGNMENT_TOLERANCE_S:
        raise ValueError(name + " must align to the 0.002 s physics step")
    return step


def validate(plan):
    """检查并返回独立的规范 dict；不排序、不合并、不静默截短事件。"""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    version = plan.get("version")
    if isinstance(version, bool) or not isinstance(version, Integral) or version != 1:
        raise ValueError("plan version must be integer 1")
    physics_dt = _number(plan.get("physics_dt_s", PHYSICS_DT_S), "physics_dt_s")
    if not math.isclose(physics_dt, PHYSICS_DT_S, rel_tol=0., abs_tol=1e-12):
        raise ValueError("physics_dt_s must be 0.002")
    result = deepcopy(plan)
    result["version"] = 1
    result["physics_dt_s"] = PHYSICS_DT_S
    total_steps = None
    if "duration_s" in plan:
        total_steps = _step(plan["duration_s"], "duration_s")
        if total_steps <= 0:
            raise ValueError("duration_s must be positive")
        result["duration_s"] = total_steps * PHYSICS_DT_S
    events = plan.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    normalized = []
    previous_end = 0
    for index, event in enumerate(events):
        label = "events[{}]".format(index)
        if not isinstance(event, dict) or not all(
            key in event for key in ("t_start_s", "duration_s", "force_world_n")
        ):
            raise ValueError(label + " requires t_start_s, duration_s, force_world_n")
        start = _step(event["t_start_s"], label + ".t_start_s")
        length = _step(event["duration_s"], label + ".duration_s")
        if start < round(EARLIEST_PUSH_S / PHYSICS_DT_S):
            raise ValueError(label + " must start at or after 5 s")
        if length <= 0:
            raise ValueError(label + ".duration_s must be positive")
        if start < previous_end:
            raise ValueError(label + " is unsorted or overlaps the previous event")
        if total_steps is not None and start + length > total_steps:
            raise ValueError(label + " extends beyond plan duration_s")
        force = event["force_world_n"]
        if (not isinstance(force, (list, tuple, np.ndarray))
                or np.ndim(force) != 1 or len(force) != 3):
            raise ValueError(label + ".force_world_n must contain three numbers")
        force = [_number(value, label + ".force_world_n") for value in force]
        if not any(value != 0. for value in force):
            raise ValueError(label + ".force_world_n must be nonzero; use a gap instead")
        normalized.append({"t_start_s": start * PHYSICS_DT_S,
                           "duration_s": length * PHYSICS_DT_S,
                           "force_world_n": force})
        previous_end = start + length
    result["events"] = normalized
    return result


class StandingPushSchedule:
    def __init__(self, plan):
        self._plan = validate(plan)
        self._starts = [_step(e["t_start_s"], "t_start_s") for e in self._plan["events"]]
        self._ends = [start + _step(e["duration_s"], "duration_s")
                      for start, e in zip(self._starts, self._plan["events"])]
        self._forces = [np.asarray(e["force_world_n"], dtype=np.float64)
                        for e in self._plan["events"]]

    def to_dict(self):
        """完整计划的独立副本，可直接 JSON 序列化写入评估 metadata。"""
        return deepcopy(self._plan)

    def force_at(self, t_s):
        """查询物理子步起点；容许浮点累积误差，不接受真正偏离网格的时间。"""
        step = _step(t_s, "t_s")
        index = int(np.searchsorted(self._starts, step, side="right")) - 1
        if index >= 0 and step < self._ends[index]:
            return self._forces[index].copy()
        return np.zeros(3, dtype=np.float64)


def load(path):
    """读取一个现有 JSON 计划，不依赖 Torch、Isaac 或仿真状态。"""
    with Path(path).open(encoding="utf-8") as stream:
        return StandingPushSchedule(json.load(stream))


def force_at(plan, t_s):
    """函数式查询入口；反复查询建议复用 load/generate 返回的计划对象。"""
    if not isinstance(plan, StandingPushSchedule):
        plan = StandingPushSchedule(plan)
    return plan.force_at(t_s)


def generate(duration_s=30., seed=42, first_push_s=5., gap_s_range=(3., 5.),
             push_duration_s=.1, force_n_range=(5., 15.)):
    """默认首推在 5 s，后续从上一脉冲结束后留出 3–5 s 随机空档。

    空档均匀采样整数物理步；幅值/水平角均匀采样。尾部不截短脉冲。
    共享生成后的 JSON 才是两个评估端的推扰契约，不各自重新抽样。
    """
    total = _step(duration_s, "duration_s")
    start = _step(first_push_s, "first_push_s")
    length = _step(push_duration_s, "push_duration_s")
    if total <= 0 or length <= 0 or start < round(EARLIEST_PUSH_S / PHYSICS_DT_S):
        raise ValueError("positive duration and first_push_s >= 5 s are required")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, bounds in (("gap_s_range", gap_s_range), ("force_n_range", force_n_range)):
        if (not isinstance(bounds, (list, tuple, np.ndarray))
                or np.ndim(bounds) != 1 or len(bounds) != 2):
            raise ValueError(name + " requires two bounds")
    gap_low, gap_high = [_step(v, "gap_s_range") for v in gap_s_range]
    low, high = [_number(v, "force_n_range") for v in force_n_range]
    if gap_low > gap_high or not 0. < low <= high:
        raise ValueError("ranges must be ordered and force magnitude must be positive")
    rng = np.random.default_rng(int(seed))
    events = []
    while start + length <= total:
        magnitude = float(rng.uniform(low, high))
        angle = float(rng.uniform(-math.pi, math.pi))
        events.append({"t_start_s": start * PHYSICS_DT_S,
                       "duration_s": length * PHYSICS_DT_S,
                       "force_world_n": [magnitude * math.cos(angle), magnitude * math.sin(angle), 0.]})
        start += length + int(rng.integers(gap_low, gap_high + 1))
    return StandingPushSchedule({
        "version": 1, "physics_dt_s": PHYSICS_DT_S,
        "duration_s": total * PHYSICS_DT_S, "seed": int(seed), "events": events,
        "generation": {"first_push_s": _step(first_push_s, "first_push_s") * PHYSICS_DT_S,
                       "gap_s_range": [gap_low * PHYSICS_DT_S, gap_high * PHYSICS_DT_S],
                       "push_duration_s": length * PHYSICS_DT_S, "force_n_range": [low, high]},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新 JSON 文件；不覆盖已有文件")
    parser.add_argument("--duration_s", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plan = generate(duration_s=args.duration_s, seed=args.seed).to_dict()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("generated {}: {} events".format(args.output, len(plan["events"])))


if __name__ == "__main__":
    main()
