# 2026-11-01 辅助脚本原始源码归档

此归档保存整理前的333个Python辅助脚本及相关源码，共3048个文件。`source.tar.gz` 保存原始相对路径和文件内容；`source-sha256.json` 为逐文件校验清单，`dependencies.json` 记录脚本导入和显式引用关系。原工作区有未提交改动，因此不能仅用 Git 提交号代替本快照。

26个固定旧实验工具列入归档（其中4个原入口因另一任务正在使用而暂留）；1个重复检查入口合并。所有原始脚本（包括合并前版本）仍在压缩包中。具体处置查[迁移表](../../agent/MIGRATION.md)。

## 恢复到隔离目录

从当前仓库根目录执行，目标必须是仓库之外尚不存在的目录：

```bash
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.restore_script_archive \
  --output /tmp/wheelleg-script-review
```

工具核对全部源码哈希后恢复原目录结构，再链接本仓 `logs/`、`artifacts/`、`resources/` 和 mjlab 虚拟环境。它不运行旧脚本、不复制模型、不写入原实验产物；这些链接仍指向原数据，复核命令的输出必须选新的独立目录。

恢复目录内使用归档前的原命令。固定实验脚本依赖当时的配方、模型和数据；数据与模型继续使用原脚本的哈希检查，不允许为通过而改写校验值。涉及更早版本生产源码的实验，按原记录使用其既有源码快照，不能把本次迁移前的源码当成所有历史版本。

例如，只复核已保存的SuperMimic报告，不运行模型、物理或训练：

```bash
cd /tmp/wheelleg-script-review
mjlab_training/.venv/bin/python scripts/agent/report_supermimic_v13_count_budget.py --selfcheck
```

成功判据：退出码0、`all_checks_passed=true`，且报告明确 `model_forwards=0`、`physics_steps=0`、`optimizer_updates=0`。此检查已在本次隔离恢复中执行；不代表其他25个归档工具都重新运行过，也不代表A700行为通过。

## 复核边界

- 已检查归档的源码依赖存在，26个归档脚本中直接写出的 `logs/`、`artifacts/` 路径在整理时均存在。命令参数提供的数据路径仍须按原实验记录选择。
- 模型、数据、机器人资产和第三方运行环境不打包到源码压缩包。归档不是跨机器完整环境镜像，外部资产仍需原校验或原实验快照确认。
- Isaac必须使用已安装Isaac Gym的Python 3.8环境，不用mjlab环境重跑。固件源码虽保全，但本次不编译、不烧录。
- 源码压缩包保留旧实现是为了历史复核；它不属于日常维护入口，也不计作活跃代码减少。
- 依赖缺失、配方不匹配或校验失败时停止该项复核并记录缺口，不跳过验证、不启动追加训练。

验证记录见[脚本整理验收](../../../docs/diagnostics/20261101-script-cleanup.md)。
