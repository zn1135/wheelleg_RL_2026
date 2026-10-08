"""CPU 推扰计划契约检查；无仿真，不代表抗扰行为验收。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2sim.standing_push_schedule import (
    PHYSICS_DT_S, StandingPushSchedule, force_at, generate, load, validate,
)


def rejected(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid input was accepted: {} {}".format(args, kwargs))


def main():
    plan = {"version": 1, "physics_dt_s": .002, "duration_s": 30., "events": [
        {"t_start_s": 5., "duration_s": .1, "force_world_n": [10., -2., 0.]},
        {"t_start_s": 8.1, "duration_s": .1, "force_world_n": [-8., 3., 0.]},
    ]}
    schedule = StandingPushSchedule(plan)
    for event in plan["events"]:
        start = round(event["t_start_s"] / PHYSICS_DT_S)
        duration = round(event["duration_s"] / PHYSICS_DT_S)
        np.testing.assert_array_equal(schedule.force_at((start - 1) * PHYSICS_DT_S), [0., 0., 0.])
        for step in range(start, start + duration):
            np.testing.assert_array_equal(schedule.force_at(step * PHYSICS_DT_S), event["force_world_n"])
        np.testing.assert_array_equal(schedule.force_at((start + duration) * PHYSICS_DT_S), [0., 0., 0.])
        for delta in (-1e-10, 0., 1e-10):
            np.testing.assert_array_equal(schedule.force_at(start * PHYSICS_DT_S + delta), event["force_world_n"])
            np.testing.assert_array_equal(schedule.force_at((start + duration) * PHYSICS_DT_S + delta), [0., 0., 0.])
    all_forces = np.array([schedule.force_at(step * PHYSICS_DT_S) for step in range(15000)])
    assert np.count_nonzero(np.linalg.norm(all_forces, axis=1)) == 100
    np.testing.assert_allclose(all_forces.sum(axis=0) * PHYSICS_DT_S, [.2, .1, 0.], atol=1e-12)
    np.testing.assert_array_equal(schedule.force_at(29.), [0., 0., 0.])
    np.testing.assert_array_equal(schedule.force_at(5.), [10., -2., 0.])  # 无状态，允许逆序查询。
    returned = schedule.force_at(5.)
    returned[:] = 999.
    np.testing.assert_array_equal(force_at(plan, 5.), schedule.force_at(5.))
    assert StandingPushSchedule({"version": 1, "events": []}).force_at(5.).sum() == 0.
    adjacent = deepcopy(plan)
    adjacent["events"][1]["t_start_s"] = 5.1
    np.testing.assert_array_equal(StandingPushSchedule(adjacent).force_at(5.1), [-8., 3., 0.])

    generated = generate().to_dict()
    assert generated == generate().to_dict()
    assert generated != generate(seed=43).to_dict()
    numpy_parameters = generate(seed=np.int64(42), gap_s_range=np.array([3, 5]),
                                force_n_range=np.array([5, 15])).to_dict()
    assert json.loads(json.dumps(numpy_parameters)) == generated
    assert generated["events"][0]["t_start_s"] == 5.
    for index, event in enumerate(generated["events"]):
        assert 5. <= np.linalg.norm(event["force_world_n"]) <= 15.
        assert event["force_world_n"][2] == 0.
        assert event["duration_s"] == .1
        assert event["t_start_s"] + event["duration_s"] <= 30.
        if index:
            previous = generated["events"][index - 1]
            gap = event["t_start_s"] - previous["t_start_s"] - previous["duration_s"]
            assert 3. - 1e-12 <= gap <= 5. + 1e-12
    with tempfile.TemporaryDirectory(prefix="standing-push-check-") as directory:
        path = Path(directory) / "plan.json"
        path.write_text(json.dumps(generated), encoding="utf-8")
        loaded = load(path)
        original = StandingPushSchedule(generated)
        assert loaded.to_dict() == generated
        for step in range(15000):
            np.testing.assert_array_equal(loaded.force_at(step * PHYSICS_DT_S),
                                          original.force_at(step * PHYSICS_DT_S))

    for key, value in (("version", True), ("version", 2), ("physics_dt_s", .01),
                       ("duration_s", 0.), ("duration_s", float("nan")), ("events", None)):
        invalid = deepcopy(plan)
        invalid[key] = value
        rejected(validate, invalid)
    for key, value in (("t_start_s", 4.998), ("t_start_s", 5.001), ("t_start_s", True),
                       ("duration_s", 0.), ("duration_s", -.1), ("duration_s", .001),
                       ("duration_s", float("inf")), ("force_world_n", [0., 0., 0.]),
                       ("force_world_n", [1., 2.]), ("force_world_n", [True, 0., 0.]),
                       ("force_world_n", [float("nan"), 0., 0.])):
        invalid = deepcopy(plan)
        invalid["events"][0][key] = value
        rejected(validate, invalid)
    invalid = deepcopy(plan)
    invalid["events"][1]["t_start_s"] = 5.098
    rejected(validate, invalid)
    invalid = deepcopy(plan)
    invalid["events"].reverse()
    rejected(validate, invalid)
    invalid = deepcopy(plan)
    invalid["events"][1]["t_start_s"] = 29.95
    rejected(validate, invalid)
    for time in (-.002, .001, float("nan"), float("inf"), True):
        rejected(schedule.force_at, time)
    rejected(generate, seed=True)
    rejected(generate, force_n_range=(15., 5.))
    rejected(generate, gap_s_range=(5., 3.))
    rejected(generate, gap_s_range=None)
    print("PASS: 2 ms 边界/浮点误差、正负力、50步脉冲、空档/相邻/重叠、序列化及 seed 可复现性")
    print("default 30 s seed=42: {} events".format(len(generated["events"])))


if __name__ == "__main__":
    main()
