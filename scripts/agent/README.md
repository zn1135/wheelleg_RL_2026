# Agent 自动化脚本

本目录放供 AI Agent 调用的自动化脚本——尤其是**可自动判读**的检查，用来弥补本仓库没有单元测试框架的缺口。

约定：

- 用仓库指定的解释器（见 [AGENTS.md](../../AGENTS.md)），不要假设 `python` 在 PATH 里就是对的。
- 退出码要有意义：0 = 通过，非 0 = 失败，这样 Agent 不必解析人类可读输出。
- 新增脚本后在 [docs/ai/build-test.md](../../docs/ai/build-test.md) 里登记。

现成的人工判读工具在 `sim2sim/`（`check_model.py` / `eval_isaac.py` / `mj_sim2sim.py --selfcheck`），本目录目前为空。
