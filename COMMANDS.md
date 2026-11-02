# 常用命令

从仓库根目录执行。维护⑥SuperMimic、⑦BC／BC初始化PPO、⑧SMP先验／SMP辅助PPO，以及新增的复旦PPO行走适配；模型采纳状态见[项目实验记录](docs/ai/project-log.md)。

| 操作 | 入口 |
|---|---|
| SuperMimic训练／回放／导出 | `PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m ct_mjlab.supermimic`，见[SuperMimic说明](docs/mjlab-chuanliantui-supermimic.md) |
| BC训练 | `scripts/mjlab.sh train-bc`，见[BC说明](docs/mjlab-behavior-cloning.md) |
| SMP先验训练 | `scripts/mjlab.sh train-prior`，见[SMP说明](docs/mjlab-smp-prior.md) |
| BC初始化／SMP辅助PPO | `scripts/mjlab.sh train --task smp --smp-prior <prior>`；须明确训练预算，见[保留线命令](docs/commands/mjlab.md) |
| 统一GPU验收 | `scripts/mjlab.sh accept --task <fudan_ppo_walk/supermimic/bc/smp> --recording <GPU记录> --output <新目录>`，见[统一标准](docs/interfaces/20261102-unified-gpu-acceptance.md) |
| 检查与行为验收 | [构建与验证](docs/ai/build-test.md)、[统一运动标准](docs/interfaces/chuanliantui-motion-acceptance-v1.md) |
| 示范采集／传统控制对照 | `scripts/mjlab.sh teacher`、`scripts/mjlab.sh collect`，见[教师说明](docs/mjlab-lqr-vmc-teacher.md) |
| 实采／标定 | [实采命令](docs/commands/real2sim.md) |
| 最终sim2sim验证 | 各task使用自己的GPU采集入口，再用统一accept评分；原生CPU入口只作历史复现，见[统一标准](docs/interfaces/20261102-unified-gpu-acceptance.md) |
| Isaac历史环境 | [Isaac说明](docs/commands/isaac.md)，不混用mjlab Python |

旧task `chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`、`chuanliantui_getup`、`chuanliantui_walk`不再注册。当前注册名为`smp`、`supermimic`、`fudan_ppo_walk`；前两者分别承载⑦⑧及⑥，新增复旦PPO为独立配方；SuperMimic使用自己的算法入口，不通过普通PPO训练。

每次训练命令同时给出对应TensorBoard命令；仅改初态或复位分布时先做20环境可视预检。未给预算不启动训练。源码清理与资产例外见[清理记录](docs/maintenance/20261101-retain-678-review.md)。

## 复旦 PPO 行走适配

最终sim2sim验证仅用GPU，CPU物理入口已删除。`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.evaluate --checkpoint <model.pt> --output <新GPU目录> --deadline <本次截止Unix秒>`；完整987例，`--diagnostic`另列诊断。评分/视频和边界见[GPU验证约定](docs/interfaces/20261102-fudan-gpu-validation.md)。

独立任务 `fudan_ppo_walk`，使用 `scripts/mjlab.sh fudan-ppo-walk {train,export}`；20环境可视预检、期限参数、GPU987样本矩阵与TensorBoard命令见[冻结规格](docs/mjlab-fudan-ppo-walk.md)。普通SMP训练入口及五个旧任务禁用规则保持。

复旦初始化修正版本为reset/history v2；新20环境可视预检尚未执行，旧trial01预检不能放行。命令和新预算建议见[修正交付](artifacts/fudan-reset-v2-20261101/README.md)。
