# 构建与验证

SuperMimic V13 r2已按实际更新数完成：从A700完整恢复后恰新增1500次actor Adam到A2200，无用户时长上限，不因摔倒或平衡丢失早停；精确封存A1200/A1700，训练结束后才串行评估。正式训练/完成审计、三次固定seed43/96例评估及pipeline均退出0；累计W9056/E/A/C各2200，新增W6000/E/A/C各1500、新399360/累计642048转移，学习循环4509.199秒/训练进程4522.847秒。A1200/A1700均存活0/96、平衡0/96，阶段128/384与112/384；A2200存活96/96、平衡仍0/96、384/384完整，数值异常均0。A2200前进vx −.493～−.492/MAE .992～.993m/s，后退vx −.684～−.683/MAE .183～.184m/s；停车水平mean .553～.589/peak .795～.850m/s、高度.303～.305m。末1秒每轮/总承重持续条件均96例失败，机身离地/姿态/角速度各0例失败；不能写成整段无接触。仅存活和部分跟踪恢复，平衡未恢复，不推断更多更新必然有效或无效。[r2实际命令、退出与结果](../../artifacts/supermimic-v13-continuation-20261101-r2/README.md)。101生产源/算法/物理不改；固定入口及Recorder兼容副本恢复另存属于目录迁移维护，不作算法证据。

按任务选择正式入口；检查输出必须注明模型、配方和覆盖范围。旧实验的详细命令保留在[历史专项检查登记](script-check-reference.md)，脚本旧路径统一查[迁移表](../../scripts/agent/MIGRATION.md)。

## 环境与入口

- mjlab 使用 `scripts/mjlab.sh` 或 `mjlab_training/.venv/bin/python`。直接运行辅助模块前设置 `PYTHONPATH=mjlab_training/src`。
- Isaac 使用 `.env.local` 的 `WHEELLEGGED_PYTHON`；先导入 isaacgym 再导入 torch，不安装新依赖或混用 mjlab 环境，见[旧任务规则](isaac-legacy.md)。
- 仓库没有单元测试框架或 CI。下表是正式程序检查／行为测量入口，不统一称作单测。

| 范围 | 程序检查模块（前缀均为 `scripts.agent.`） | 行为验收与命令 |
|---|---|---|
| multimode | `multimode.check_mjlab_multimode`、`multimode.check_multimode_final_balance` | [任务说明](../mjlab-chuanliantui.md)；双轮平衡和跟踪分别测量 |
| 稳站 | `stand.check_chuanliantui_stand` | [稳站说明](../mjlab-chuanliantui-stand.md) |
| 稳站v2 | `stand_v2.check_chuanliantui_stand_v2` | [稳站v2说明](../mjlab-chuanliantui-stand-v2.md) |
| 起立／行走 | `split.check_chuanliantui_two_policy`、`split.check_walk_resume` | [双策略接口](../interfaces/20261030-chuanliantui-two-policy.md)；按迁移表使用对应专项验收 |
| SuperMimic | `supermimic.check_chuanliantui_supermimic`、`supermimic.check_supermimic_alternating` | [配方与验证](../mjlab-chuanliantui-supermimic.md)；旧配方检查不能代替当前算法检查 |
| SuperMimic预测误差 | `supermimic.record_supermimic_physical_prediction`、`supermimic.score_supermimic_prediction_details` | 冻结策略采集24环境/seed55诊断，再离线测10/80/240ms运动、双腿几何、观测与奖励误差；参数零更新，不代替96例行为验收。历史对照输入由 `supermimic.score_supermimic_prediction` 生成 |
| SuperMimic预测进步对照 | `supermimic.compare_supermimic_prediction_progress` | A700/A1808/A2200在相同当前及历史窗口比较；整体误差统一用A2200归一化尺度，避免尺度变化造成虚假改善；零训练、零新仿真 |
| LQR／BC／SMP | `imitation.check_lqr_vmc`、`imitation.check_bc`、`imitation.check_smp` | [teacher](../mjlab-lqr-vmc-teacher.md)、[BC](../mjlab-behavior-cloning.md)、[SMP](../mjlab-smp-prior.md) |
| 实采／CPU镜像 | `real2sim.check_real2sim_huananhu`，其他按 [real2sim目录](../../scripts/agent/real2sim/) | [实采命令](../commands/real2sim.md)、[sim2sim约定](sim2sim.md) |
| Isaac | [isaac目录](../../scripts/agent/isaac/)及原 `sim2sim/` 入口 | [训练→回放→sim2sim](../commands/isaac.md)；保留旧环境和接口限制 |

模型、参数和必需输入查表中任务文档；不要只凭模块名运行。`check` 有限数值检查、`--selfcheck` 形状检查和地面起立统计都不能单独代表行为通过。串联腿正式验收遵循[双轮平衡标准](../interfaces/chuanliantui-wheel-balance-acceptance-v2.md)，原成功样本任一失效即判退化。

2026-11-01确认[统一运动标准 v1](../interfaces/chuanliantui-motion-acceptance-v1.md)：前后速度平均绝对误差≤0.10 m/s、转向平均绝对误差≤0.10 rad/s、停车稳定阶段水平速度峰值≤0.05 m/s、稳定巡航及停车阶段俯仰绝对值峰值≤5°。起立暂不纳入本次统一。multimode、stand_v2速度／转向／tracking和SuperMimic的运动判定已接入共同实现；旧[平地跟踪](../interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)的0.30门槛不能代表新标准通过；刹停时限等完整协议仍待统一。

## 脚本整理的正式检查

```bash
export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_layout
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_merges
```

第一项核对原始源码压缩包、333个Python脚本及辅助资产的迁移覆盖、模块导入和脚本路径；第二项用历史测量和保存的预测张量验证合并函数结果与原实现相同，并检查浮点计数边界。均不训练，不运行物理仿真，不能证明机器人行为改善。完整结果与未验证项见[本次验收记录](../diagnostics/20261101-script-cleanup.md)。

运动统一入口（只读保存轨迹，不启动策略或物理）：

```bash
mjlab_training/.venv/bin/python -m scripts.agent.motion.assess --selfcheck
mjlab_training/.venv/bin/python -m scripts.agent.motion.assess --measurement <已有测量目录> --output <新结果.json>
mjlab_training/.venv/bin/python -m scripts.agent.motion.assess --audit-tracking <原stand_v2测量目录> --initial-reference <原初态.npz> --output <新审计.json>
```

前两项检查统一判据／复算四项指标；最后一项保留原stand_v2源码、模型、命令送达、初态和500Hz平衡数据审计，不把旧诊断改称新标准通过。支持保存的SuperMimic和stand_v2 tracking／speed／yaw格式；缺少匹配源码、轨迹哈希或必要信号时拒绝或标记缺项。稳定窗口沿用原测量协议，完整运动协议尚未定稿。

新检查放对应分类并在本页登记。仅为某次实验生成的汇总放实验归档，不持续向日常入口增加同用途副本。
