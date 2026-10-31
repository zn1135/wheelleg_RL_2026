# 辅助脚本导航

日常操作先看[常用命令](../../COMMANDS.md)；查旧文件看[逐文件迁移表](MIGRATION.md)。辅助工具用于专项测量和验证，不替代正式训练入口。

所有 Python 工具从仓库根目录用 `python -m scripts.agent.<分类>.<工具>` 运行；Python 必须替换为对应环境。

运动评估合并采用[统一运动标准 v1](../../docs/interfaces/chuanliantui-motion-acceptance-v1.md)，起立暂不纳入。multimode、stand_v2 tracking／速度／转向及SuperMimic的判定已接入共同实现；旧报告须由保存轨迹重新计算。统一离线入口为`python -m scripts.agent.motion.assess --measurement <目录> --output <新结果.json>`，`--selfcheck`检查四项判据，`--audit-tracking <目录>`保留原stand_v2数据审计。

| 分类 | 内容 | 迁移保留文件数 |
|---|---|---|
| [motion](motion/) | 统一运动判据、保存轨迹复算与离线审计（本轮合并新增，替代两个独立检查） | 1 |
| [multimode](multimode/) | 多模式任务 | 22 |
| [stand](stand/) | 独立稳站 | 11 |
| [stand_v2](stand_v2/) | 稳站v2 | 41 |
| [split](split/) | 起立／行走双策略 | 46 |
| [supermimic](supermimic/) | SuperMimic | 22 |
| [imitation](imitation/) | LQR示范／行为克隆／SMP | 101 |
| [isaac](isaac/) | Isaac旧任务与资产工具 | 27 |
| [real2sim](real2sim/) | CPU回放／实采／标定 | 37 |
| [maintenance](maintenance/) | 维护与归档复核 | 2 |

## 运行与新增约定

- mjlab：`export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"`，使用 `mjlab_training/.venv/bin/python`。
- Isaac：加载本机 `.env.local`，使用 `$WHEELLEGGED_PYTHON`；先导入 isaacgym 再导入 torch，不混装依赖。
- 参数未知时先看源码或文档；没有 argparse 的检查可能忽略 `--help` 并直接执行。
- 新脚本放所属分类，正式检查登记到[构建与验证](../../docs/ai/build-test.md)。优先扩展已有同用途工具，固定某次实验的报告放实验归档。
- 新输出必须使用新目录，不能覆盖 logs、模型或既有实验结果。

## 本次整理

迁移前333个Python脚本；26个固定实验工具列入归档，1个重复入口合并，306个原有Python文件按用途保留。额外3个shell／C／JSON辅助文件随调用方迁移。新共享模块和维护检查另计，见验收报告。

保留未能证明废弃的工具；保留不表示全部已在当前模型上通过。归档恢复需要原模型、数据和匹配环境，参见[复核说明](../archive/20261101-script-cleanup/README.md)。

## 运行中保护例外

另一个任务正在执行SuperMimic续训与验收，7个原字节入口暂留于本目录（不是包装脚本）。其中4个拟归档、3个已有分类副本；不要在该链路结束前删除或覆盖。具体清单和后续清理条件见[runtime-protection.json](runtime-protection.json)。因此当前尚未完全清空旧入口。
