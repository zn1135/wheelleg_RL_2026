# 接口记录：复旦8000轮停车+2°参考

| 项目 | 内容 |
|---|---|
| 日期、作者、复查 | 2026-11-03；Codex实现，用户明确“用这个”；H7负责人待确认 |
| 机器人、任务 | 本仓实体串联闭链机器人，`fudan_ppo_walk`，GPU仿真 |
| 旧接口 | `fudan-ppo-walk-physical25-history125-critic144-v1`，reset/history v2；[原接口](20261101-fudan-ppo-walk.md) |
| 新推理配方 | `fudan-selected-parking-pitch-reference-v1`；张量合同不变 |
| 仓库与版本 | `zn1135/wheelleg_RL_2026`，`26_wheelleg`，HEAD `c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9`，有未提交改动；[执行源快照](../../artifacts/fudan_ppo_walk/fudan-selected-8000-pitch2-20261103/source-sha256.json) |
| 权威实现 | `FudanWalkCfg.inference`、`ct_mjlab.fudan_ppo_walk.selected_policy`、`scripts.agent.fudan_ppo_walk.evaluate` |
| 部署仓 | 本次未交付H7，代码未改，负责人适配/时序/实机均待确认 |

## 输入、时序与兼容性

| 项目 | 旧定义 | 当前选用方案 |
|---|---|---|
| 模型 | 调用方显式指定 | 默认固定8000-v2，SHA256 `39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9` |
| Actor/历史 | 25/125，重力槽3:6；其余顺序、单位和缩放按原v2 | 同形状；图外将重力向量绕Y轴转动参考角，当前帧和5个历史帧按各自时间处理 |
| 参考 | 0 | 停车前进及转向命令都为零时+2°；行走/转向不加偏置 |
| 激活 | 无 | 原地从t=2s开始；驾驶后从零速段开始，均以0.5s斜坡加入 |
| 历史 | 原环境维护v2 FIFO | 原缓存不改，仅给Actor/Encoder的副本变换；上拍动作仍为实际执行动作 |
| 测量和评分 | 实际物理状态 | 实际物理状态，不减2°，统一5°姿态门槛保持 |
| 关节/动作/PD/XML | 原25/125/6模型及物理控制 | 无改动；物理/PD 2ms，策略10ms |
| 积分补偿 | 默认关闭 | 关闭；开关变化时选用配方拒绝继续加载 |

实际变换：`gx'=cos(b)gx+sin(b)gz`，`gy'=gy`，`gz'=-sin(b)gx+cos(b)gz`。不施加外力固定姿态，真实pitch由策略控制结果决定。

当前入口按冻结评估协议的时间计划执行，支持原47工况的末端停车/连续原地保持；未验证“停车后再次驶出”，该类新协议直接拒绝。复位以新轨迹时钟重新开始参考，不复用上一回合时间。它不是通用遥控停车控制器。

未指定`--checkpoint`时，GPU评估读取该选用方案并把完整配方写入`protocol.json.selected_policy`。显式指定`--checkpoint`仍使用原始输入，保持历史复现。训练入口和普通ONNX导出不自动加入此偏置；已有8000 ONNX本身也不含+2°。

## 验证与使用范围

- [26项无物理检查](../../artifacts/fudan_ppo_walk/fudan-selected-8000-pitch2-20261103/checks.json)：与已执行探针的输入计算逐值一致，覆盖47工况的激活计划、移动/转向不变、默认/显式CLI、模型哈希及积分条件拒绝。无物理/网络前向/优化器更新。
- [已执行三高度九停车](../../artifacts/fudan_ppo_walk/fudan-pitch-reference-small-20261103/README.md)：8000+2°停车9/9，低高度腿部运动增加；用户明确选用。新默认入口未另跑物理，原45行走/转向不退化及正式987矩阵仍未验证。
- [模型选用记录](../model-deliveries/20261103-fudan-8000-pitch2-selection.md)。本次只选用GPU仿真方案；H7同步、数值/实时性和真机验证未执行，负责人待确认。

回退：显式指定原8000检查点即可恢复原输入；历史PT、ONNX、日志与原始轨迹保持。
