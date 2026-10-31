# 常用命令

所有命令从仓库根目录执行。模型采纳状态只查[项目实验记录](docs/ai/project-log.md)；下面按操作找入口，不把某次实验模型当作默认推荐。

| 要做什么 | 入口与成功判据 |
|---|---|
| 训练／续训 | 下方任务表选择配方；日志出现新 run 和 checkpoint。每次先明确预算，配套打开该 run 的 TensorBoard。 |
| 看效果 | mjlab 用 `scripts/mjlab.sh play`；独立 CPU 回放查对应任务说明。窗口显示正常仅代表回放启动。 |
| 验收 | [正式检查导航](docs/ai/build-test.md)；运动采用[统一运动标准](docs/interfaces/chuanliantui-motion-acceptance-v1.md)，脚本接入状态见标准说明。 |
| 导出 | `scripts/mjlab.sh export`；任务和 checkpoint 必须配套，检查 ONNX 数值报告。 |
| 实采／标定 | [Jetson连接、录制、回放](docs/commands/real2sim.md)；校准专项工具在 [real2sim分类](scripts/agent/real2sim/)。 |
| 历史复核 | [旧脚本迁移表](scripts/agent/MIGRATION.md)、[归档恢复说明](scripts/archive/20261101-script-cleanup/README.md)。 |

## 选择任务与环境

| 任务 | 训练、回放、导出命令及正式验收 |
|---|---|
| 起立／行走 `chuanliantui_getup`、`chuanliantui_walk` | [双策略说明](docs/interfaces/20261030-chuanliantui-two-policy.md)；专项检查查迁移表的 split 分类 |
| SuperMimic | [任务与配方说明](docs/mjlab-chuanliantui-supermimic.md)；算法入口保持原有 `ct_mjlab.supermimic` |
| multimode `chuanliantui` | [基础操作示例](docs/commands/mjlab.md)、[任务说明](docs/mjlab-chuanliantui.md) |
| 独立稳站 `chuanliantui_stand` | [稳站任务](docs/mjlab-chuanliantui-stand.md) |
| 稳站v2 `chuanliantui_stand_v2` | [稳站v2](docs/mjlab-chuanliantui-stand-v2.md) |
| LQR示范／行为克隆／SMP | [teacher](docs/mjlab-lqr-vmc-teacher.md)、[行为克隆](docs/mjlab-behavior-cloning.md)、[SMP](docs/mjlab-smp-prior.md) |
| Isaac 历史任务 | [独立环境、训练、回放与sim2sim](docs/commands/isaac.md)；禁止使用 mjlab 环境执行 |

mjlab 正式入口 `scripts/mjlab.sh` 自动选择仓库虚拟环境。辅助检查使用：

```bash
export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_layout
```

查具体工具：打开[分类导航](scripts/agent/README.md)，再查[逐文件迁移表](scripts/agent/MIGRATION.md)。旧 `scripts/agent/文件.py` 路径已按迁移表处理，不再保留包装脚本；另一个训练任务仍在使用的7个原文件暂留，见[保护清单](scripts/agent/runtime-protection.json)；没有参数解析的旧检查不能用 `--help` 试探。

## 训练前后

训练预算沿用本次明确约定；文档里的轮数只作示例。只改初始位置或复位分布时，先运行对应任务的20环境可视预检，确认初态无穿地、弹飞或意外分布，再使用该任务说明中的训练命令。

每份训练命令必须同时指定 TensorBoard 的实际日志目录。训练入口打印 `Training output` 后，将其填入：

```bash
ct_log_dir="logs/填入本次实际训练子目录"
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$ct_log_dir" --port 8080
```

成功判据：页面出现本次 run 的训练曲线。Isaac 使用其自身环境的 TensorBoard，详见 Isaac 命令页。有限数值检查、形状检查和地面起立统计均不能单独代表行为验收。

## 历史与部署入口

- [历史 PPO400 回放示例](docs/commands/history.md)：保留原命令含义，不代表当前采纳模型。
- [历史专项检查登记](docs/ai/script-check-reference.md)：按迁移表寻找新入口或归档副本。
- [部署工程选择](deployment/README.md)：UART／USB 固件编译烧录继续使用各自工程说明。
