# 辅助脚本导航

仅维护⑥SuperMimic、⑦BC、⑧SMP。旧multimode／stand／stand_v2／split专项脚本随独立task清理，共用函数提取至shared，不保留旧路径包装。

| 分类 | 内容 |
|---|---|
| [supermimic](supermimic/) | 世界模型／策略检查、预测与行为评分、诊断及报告 |
| [imitation](imitation/) | LQR教师、示范采集、BC、SMP及BC初始化PPO |
| [motion](motion/) | 统一运动标准与保存的SuperMimic轨迹复算 |
| [shared](shared/) | 保留线所需的渲染、物理采集与静态几何函数 |
| [real2sim](real2sim/) | 实采、标定、CPU镜像与部署侧核对 |
| [isaac](isaac/) | Isaac历史工具，本次未清理该训练环境 |
| [maintenance](maintenance/) | 保留线检查、迁移审计与源码保全 |

入口统一为对应环境的`python -m scripts.agent.<分类>.<工具>`，从仓库根目录运行。mjlab使用`mjlab_training/.venv/bin/python`并设置`PYTHONPATH=mjlab_training/src`；不使用系统Python或混用Isaac环境。不支持安全`--help`的脚本只做静态检查。

[命令导航](../../COMMANDS.md) · [原路径迁移表](MIGRATION.md) · [清理范围及验证](../../docs/maintenance/20261101-retain-678-review.md) · [批准保留的经验](../../docs/ai/training-lessons.md)
