# 常用命令

从仓库根目录执行。只维护⑥SuperMimic、⑦BC／BC初始化PPO、⑧SMP先验／SMP辅助PPO；模型采纳状态见[项目实验记录](docs/ai/project-log.md)。

| 操作 | 入口 |
|---|---|
| SuperMimic训练／回放／导出 | `PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m ct_mjlab.supermimic`，见[SuperMimic说明](docs/mjlab-chuanliantui-supermimic.md) |
| BC训练 | `scripts/mjlab.sh train-bc`，见[BC说明](docs/mjlab-behavior-cloning.md) |
| SMP先验训练 | `scripts/mjlab.sh train-prior`，见[SMP说明](docs/mjlab-smp-prior.md) |
| BC初始化／SMP辅助PPO | `scripts/mjlab.sh train --task smp --smp-prior <prior>`；须明确训练预算，见[保留线命令](docs/commands/mjlab.md) |
| 检查与行为验收 | [构建与验证](docs/ai/build-test.md)、[统一运动标准](docs/interfaces/chuanliantui-motion-acceptance-v1.md) |
| 示范采集／传统控制对照 | `scripts/mjlab.sh teacher`、`scripts/mjlab.sh collect`，见[教师说明](docs/mjlab-lqr-vmc-teacher.md) |
| 实采／标定 | [实采命令](docs/commands/real2sim.md) |
| MuJoCo sim2sim | 原`sim2sim/`实现及参数保持，见[sim2sim约定](docs/ai/sim2sim.md) |
| Isaac历史环境 | [Isaac说明](docs/commands/isaac.md)，不混用mjlab Python |

旧task `chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`、`chuanliantui_getup`、`chuanliantui_walk`不再注册。当前注册名只有`smp`和`supermimic`，分别承载⑦⑧及⑥的必要环境；SuperMimic使用自己的算法入口，不通过普通PPO训练。

每次训练命令同时给出对应TensorBoard命令；仅改初态或复位分布时先做20环境可视预检。未给预算不启动训练。源码清理与资产例外见[清理记录](docs/maintenance/20261101-retain-678-review.md)。
