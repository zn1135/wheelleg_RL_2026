"""CPU 检查 H7 闭链回放的十拍预热和无效观测恢复，不启动仿真。"""

import isaacgym  # 必须先于 torch
import numpy as np

from sim2sim import mj_sim2sim_ct as sim


def main():
    history_type = getattr(sim, "H7PolicyHistory", None)
    assert history_type is not None, "闭链回放尚未接入 H7 十拍预热及失效恢复"
    state = history_type()
    for step in range(10):
        obs = np.full(25, step + 1.0)
        history, active = state.update(obs, True)
        assert not active, "H7 的前十个有效策略步只能运行零动作 PD"
        np.testing.assert_equal(history[-25:], obs)
        if step == 0:
            np.testing.assert_equal(history, np.ones(125))
    history, active = state.update(np.full(25, 11.0), True)
    assert active
    np.testing.assert_equal(history.reshape(5, 25)[:, 0], [7, 8, 9, 10, 11])
    history, active = state.update(np.full(25, 999.0), False)
    assert not active
    np.testing.assert_equal(history, np.zeros(125))
    assert not state.ready, "无效观测必须关闭 PD 出力资格"
    for _ in range(10):
        history, active = state.update(np.full(25, 4.0), True)
        assert not active
        np.testing.assert_equal(history, np.full(125, 4.0))
    assert state.update(np.zeros(25), True)[1]
    history, active = state.update(np.full(25, np.nan), True)
    assert not active and np.isfinite(history).all()
    state.reset()
    history, active = state.update(np.full(25, 2.0), True)
    assert not active
    assert state.ready, "有效预热必须允许零动作 PD，与无效观测区分"
    np.testing.assert_equal(history, np.full(125, 2.0))
    print("PASS: H7 十拍预热、FIFO 顺序、无效观测清理和重置后重新预热")


if __name__ == "__main__":
    main()
