# 统一运动评估实现计划

> **For agentic workers:** Use superpowers:executing-plans; execute in the current workspace as requested.

**Goal:** 合并运动指标与判定实现，提供保存轨迹的统一复算入口，减少独立检查脚本。

**Architecture:** `scripts.agent.motion.assess`负责四项阈值、逐阶段判定、已有轨迹适配与离线检查。保留任务采集器及额外诊断，删除由本入口完整替代的检查文件。

**Tech Stack:** 仓库mjlab Python、NumPy、argparse；不增加依赖。

**Spec:** `docs/interfaces/chuanliantui-motion-acceptance-v1.md`

## Global Constraints

- 速度平均绝对误差≤0.10 m/s、转向≤0.10 rad/s、停车水平速度峰值≤0.05 m/s、稳定俯仰峰值≤5°。
- 不统计起立；稳定窗口沿用记录协议并写明，不新增刹停时间与瞬态门槛。
- 缺失信号、空窗口、提前终止不通过；原失败样本保留分母。
- 保存原始源码与SHA；不修改保护中的旧入口、模型、日志、算法或H7；不训练、提交、推送。

## Review Focus

- 等号与浮点归约；不让0.30旧门槛或未确认0.05跟踪门槛继续活跃。
- 横向漂移不能被前向零速掩盖；平均误差正负不能抵消。
- 稳态窗口不能按实际跟踪是否好来挑选；缺少偏航等信号须报告缺项。
- 早终止、NaN、门控未就绪须影响完整性，不能只统计幸存数据。
- 历史数据的源校验、命令送达及初态审计不能因合并而丢失。

## Task 1: 保存源码与先写检查

- [x] 本轮前源码、依赖清单、SHA保存至`scripts/archive/20261101-motion-normalization/`。
- [x] 在现有`maintenance/check_script_merges.py`增加motion-only检查，先运行观察缺失实现失败。

## Task 2: 合并判据及离线入口

- [x] 新增`motion/assess.py`：`metrics_verdict`与`phase_verdict`；统一四项指标、缺项及完整性。
- [x] 统一入口支持`--selfcheck`、stand_v2旧轨迹完整性审计及已有SuperMimic/stand_v2数据复算；写新输出、保留输入哈希。
- [x] 边界、反向误差抵消、横向速度、数据缺失和早终止检查通过。

## Task 3: 接入采集器并删除被替代入口

- [x] multimode、stand_v2 tracking/speed/yaw、SuperMimic使用共用判据；各自诊断单列，不以起立判据替代运动结论。
- [x] 将multimode旧算术检查和stand_v2独立数据审计合入统一入口，删除两文件，不留包装。
- [x] 同步迁移表、当前命令及正式检查导航，标明旧数值与统一标准重评区别。

## Task 4: 验证与交付

- [x] 保存轨迹逐项独立重算；数值不因共享实现改变，阈值改变导致的结论变化单独记录。
- [x] 正式源码布局、导入、参数检查及最终独立复查通过。
- [x] 报告实际脚本数变化、代码行变化与未覆盖范围。

## 执行记录

用户已明确要求执行；本计划直接执行，不新增审批、工作树或Git操作。

执行完成：21组边界／异常检查、27/27入口参数检查、160个历史阶段及64例tracking复算通过；独立复查3项P2均补先失败后通过的检查并修复。本任务净少1脚本，负责代码行增加309；另一个任务新增1脚本，当前工作区总数仍320。完整限制见docs/diagnostics/20261101-motion-normalization.md。
