# mjlab 目录与统一任务入口重构检查

执行日期：2026-10-15（Asia/Shanghai）。分支：`26_wheelleg`；基线提交：
`109da6d636ce7b55bc58264da534f3745c0156d4`，在当前工作区修改，未提交。

## 范围

按 Isaac 训练包的职责增加 `ct_mjlab/envs/`、`scripts/`、`utils/`：
统一注册环境、提供环境/PPO 配置工厂、拆分命令行参数与任务分流、提供五个独立脚本。
原 `scripts/mjlab.sh` 和 `ct-mjlab` 继续通过兼容入口调用相同任务。

历史 `experiments/`、`reference_training/` 保留原实现及路径，35 个 Python 文件的
SHA-256 与重构前一致。不改变配方、观测、动作、控制时序、根初态、物理资产和部署接口；
未修改 H7 代码。本次不产生新模型，不需要重训来适配目录变化。

开始时已有 `.gitignore`、`AGENTS.md`、`CONTRIBUTING.md`、`docs/ai/project-log.md`
未提交修改，以及 `MJDATA.TXT`、`MUJOCO_LOG.TXT` 暂存删除；这些已有改动原样保留。
本次源码改动为 `cli.py` 与新目录，另新增结构检查脚本并更新目录/验证说明。

## 检查结果

全部在仓库根使用 `mjlab_training/.venv/bin/python` 或 `scripts/mjlab.sh` 执行。
最终各项退出码均为 0：

| 检查 | 结果与边界 |
|---|---|
| `check_mjlab_structure.py --baseline-cli /tmp/mjlab-structure-d02KRz/cli_before.py --out /tmp/mjlab-structure-d02KRz/structure.json`（设置 `PYTHONPATH=mjlab_training/src`） | 8 个新训练配方、22 个历史接口、5 个独立脚本转发；2251 组新旧参数/分流/报错对照一致，分流期间最多读取一次 checkpoint |
| 同脚本 `--physics --steps 8 --out /tmp/mjlab-structure-d02KRz/physics.json` | V14、recover-r14、r15 各 2 环境，cuda:0，runner 构造及各 8 步零动作检查；Actor35/Critic78、奖励有限；没有调用 learn |
| `scripts/mjlab.sh check --training-profile recover-r14 --num-envs 2 --steps 8` | 旧入口贯通 recover 配方，有限物理/张量检查完成 |
| `python -m ct_mjlab.scripts.check --num-envs 2 --steps 8`（本项目 mjlab Python） | 新入口贯通默认 V14，`finite=true`，`numerical=0` |
| `python -m ct_mjlab.scripts.eval --seconds 0.04 --report /tmp/mjlab-structure-d02KRz/cpu-eval.json`（本项目 mjlab Python） | 无 checkpoint、零动作，仅验证 CPU 评估入口及报告落盘；报告 `passed=false`，不计行为通过 |
| 兼容模块 SHA-256 核对、`git diff --check` | 历史实现未变化，差异无空白错误 |

配置检查逐个对照原模块工厂，检查新调用不复用前次可变配置。源码快照检查使用临时
远端发布夹具，验证新子目录纳入快照且排除虚拟环境、缓存，不写根 `logs/`。

GPU 检查脚本初版把实例化后配置与原始配置直接比较，因 mjlab 构造 Scene 时会
把 `scene.num_envs/env_spacing` 写入 `terrain` 而失败；核对安装版 `Scene._add_terrain`
后，只对齐这两个初始化字段再比较，重跑通过。生产配置未为此修改。

旧入口基线文件 SHA-256：
`b91229b0f4b12ec348d3dd103ab20da350ac06acdf65040d7ec7a81cea0213a1`。
可从上述基线提交提取 `mjlab_training/src/ct_mjlab/cli.py`，重跑对照。
临时详细输出位于 `/tmp/mjlab-structure-d02KRz/`，可能随系统清理；常规可复执行命令见
[构建与测试](../ai/build-test.md#mjlab-结构与入口检查)。

## 收尾：环境与 runner 配方绑定

注册任务时显式指定 `default_profile`；`make_env` 在返回的包装环境上记录任务与配方，
不改动环境配置或 checkpoint 合同。`make_alg_runner` 省略配方时沿用该记录，显式指定
不同任务或配方则在读取 PPO 配置、构造 runner 前拒绝。外部创建的环境必须显式传配方，
自定义 PPO 配置也不能跳过上述检查。配置工厂与历史续训/回放的职责已补充到 README。

最终执行下列命令，退出码 0：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  scripts/agent/check_mjlab_structure.py \
  --baseline-cli /tmp/mjlab-structure-d02KRz/cli_before.py \
  --physics --steps 8 --out /tmp/mjlab-structure-d02KRz/final-structure.json
```

- 2251 组新旧入口对照、8 配方配置一致性、22 历史接口分流继续通过。
- 默认/非默认配方自动沿用、显式相同配方、自定义 PPO 配置保留均通过。
- 错误任务/配方及未声明配方的外部环境均在配置工厂和 runner 执行前拒绝；外部环境显式配方可用。
- V14、recover-r14、r15 各 2 个 GPU 环境省略 runner 配方参数时，仍选中各自的 runner，
  各完成 8 步零动作有限张量检查，没有调用学习。
- 历史实现 35 个文件的哈希再次核对一致，`git diff --check` 通过；原暂存删除项保持不变。

收尾检查初次使用 `isinstance` 比较 r14 runner，因其工厂每次动态创建新类而失败；
改为比较定义模块和限定类名后重跑通过。此修正仅针对检查脚本，未修改历史 runner。

本次没有启动训练/TensorBoard，没有使用历史模型完成长时间回放、续训或 ONNX 实际导出，
没有执行双轮平衡行为验收或真机验证；上述结果仅证明本次结构改动的配置、分流和有限执行路径。

## 后续修复：recover 命令资源释放

用户要求修复独立复查记录 E1 后，`envs/chuanliantui/commands.py` 使用 ExitStack
统一管理成功创建的环境和 viewer：train/check/play 的正常返回、运行异常、KeyboardInterrupt
均执行清理。grid viewer 在 open/draw 前登记关闭，passive viewer 在取得句柄后登记关闭；
先关闭 viewer，再关闭环境，即使 viewer.close 抛出异常也会继续调用 env.close。

`training_viewer.viewer_runner_class` 增加仅关键字参数 `close_env_on_error=True`，
旧调用保持原行为；recover 显式传 False，把环境释放交给外层管理，避免异常时重复关闭。
历史配方和管理器的 35 个文件未修改，训练/控制参数及日志路径不变。

新增正式检查 `scripts/agent/check_mjlab_cleanup.py`，命令见
[构建与测试](../ai/build-test.md#mjlab-结构与入口检查)。结果：

- 默认 32 组替身资源场景通过，覆盖 train/check/play 正常返回、初始化/运行失败、
  中断、viewer.close 失败和渲染训练包装器的旧默认清理行为；不实际学习或打开窗口。
- 加 `--real-check` 后，同进程两次真实 recover-r14 GPU check，各 2 环境/8 步，
  每次 env.close 和 recorder.close 均恰好调用一次，退出码 0。
- 结构检查与 2251 组基线分流对照重跑通过，`git diff --check` 与兼容模块哈希核对通过。
- 独立 Reviewer 复核清理顺序与兼容性，并独立重跑 32 组检查，未发现新增缺陷。

本轮输出：`/tmp/mjlab-structure-d02KRz/cleanup-gpu.json`、`cleanup-structure.json`；
独立检查：`/tmp/mjlab-cleanup-independent-review.json`。未实际训练、运行 GUI 或验证
驱动显存缓存清空；这些检查不能证明完整训练或真机行为通过。
