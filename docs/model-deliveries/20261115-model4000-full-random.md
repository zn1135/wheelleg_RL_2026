# 模型归档：4000轮全幅随机化，平地退化候选

2026-11-15，用户明确要求随本次Git提交一并归档4000 PT。交付人Codex，接收人为用户；独立复查及H7负责人核对待确认。机器人chuanliantui_new_1实体闭链轮腿，任务fudan_ppo_walk。本包仅实验留档，4000未被采纳，不替换用户实机3000基线或任何部署默认。

## 模型与接口

| 项目 | 实际值 |
| --- | --- |
| 完整PT | [20261115_model4000_pd200_p20d1_full_random_ground04to12_force3_push.pt](../../artifacts/fudan_ppo_walk/models/20261115_model4000_pd200_p20d1_full_random_ground04to12_force3_push.pt) |
| 大小 | 1,443,465字节 |
| SHA256 | `1513279bcc519814af3f1aea266967a937f845244002c7c03e83037fa16878b6` |
| 原文件 | `logs/mjlab/fudan_ppo_walk/20261115_pd200_full_random_force3_push_resume3018_to4000_01/model_final.pt`，原位保留、逐字节复制 |
| 网络/历史合同 | fudan-ppo-walk-physical25-history125-critic144-v1 / fudan-physical-pd-centred-warmup-noisy5-clipped-history-v3 |
| 输入输出 | Actor当前25/历史125→6动作，含encoder；Critic144；顺序、坐标、缩放与原3000一致 |
| 保存内容 | 完整网络、两Adam、课程、累计计数和CPU/CUDA RNG；4000轮、双Adam各80000，partial_update=false |
| 采样计数 | 786616320，其中184320为3018中断时弃用采样；完整更新对应786432000 |
| 控制/物理 | 腿20/1、轮.1/0，机械轮；积分1ms、PD5ms/200Hz、策略10ms/100Hz |
| 物理资产 | XML SHA `ed817cf8b0d41bc3625021c97deef33549f0c8aef7ec4e98a49d6d6a9ac2bb11`；其他参数SHA见清单 |
| 训练规模 | seed1，4096×48/history v3；3000→3018已完成18轮，本次3018→4000补982轮，合计新增1000轮 |
| 恢复 | 权重、两Adam、计数、课程、CPU/CUDA RNG逐值恢复；物理回合/历史/rollout重新初始化，弃用采样未进入梯度 |
| 3018源PT SHA | `d266c5d4091a99d6e5f92c79f689d6185ddd332fde38ddf31f97b7c9cf3fa2c2` |

配方：参数与观测噪声恢复原全幅；真实轮地滑动系数.4～1.2，机身各轴±3N持续外力/矩0，7秒/.3m/s单积分步短推扰；动作/传感各10～20ms，原17奖励含高度1/.001、原气弹簧、默认角/动作尺度/限矩保持。详见[接口记录](../interfaces/20261115-fudan-ground-friction-continuation.md)。

## 代码与来源

训练仓`https://github.com/zn1135/wheelleg_RL_2026.git`，分支26_wheelleg，运行基准HEAD `39587f355207979c4e464892954b819b2b18bc42`，训练时工作区有改动。归档提交以本记录所属Git提交为准，不能仅用训练前HEAD代表实际代码。

本次暂存树中的149项训练Python源码及物理资源，与run保存的运行快照SHA逐文件全部相同；共享文件只暂存本次摩擦实现，台阶任务的其他片段继续留在工作区。运行时全幅配方通过启动器覆盖，精确有效/PPO配置、源SHA、预算与恢复记录嵌入[来源与验证清单](data/20261115-model4000-full-random-manifest.json)，清单SHA `73b99976d88d7f7697bd0e5ded3611c90ca2c8fa65754d89a1c127770bb355ef`。训练源码在本次提交直接保存，无需依赖后来台阶开发代码。

原结束审计因并行新增台阶任务造成工作区源码门禁失败，旧failed状态原样保留；预算/学习/保存快照复核另行通过。GPU效果检查执行哈希验证过的训练快照，不把工作区源码门禁写成通过。脏工作区差异/新增启动器快照仍在[原协调目录](../../artifacts/fudan_ppo_walk/20261115-pd200-resume3018-4000-01/)，未改历史logs。模型检查所需的真实日志、轨迹和视频仍保留本机；清单嵌入指标与SHA，不将其都打包进Git。

环境版本仅在本次归档环境读取：Python 3.12.3、PyTorch 2.14.0、CUDA 13.0、MuJoCo 3.11.0、MuJoCo Warp 3.11.0、mjlab 1.6.0、Warp 1.17.0；训练时依赖未独立锁定，不能把归档读取版本当成独立训练版本证明。

已完成训练入口为`resume_pd200_remaining run --output artifacts/fudan_ppo_walk/20261115-pd200-resume3018-4000-01`；完整执行参数与实际训练日志目录在清单。训练已结束，不重复启动。仅追溯TensorBoard：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261115_pd200_full_random_force3_push_resume3018_to4000_01 --port 6020
```

## 分阶段验证

| 阶段 | 状态与证据 |
| --- | --- |
| 训练完成 | 通过原预算：982新增轮、双Adam各19640更新、982个奖励点完整；继承弃用采样单列 |
| 物理/接口 | 20GPU预检7020积分/40次单步推扰；实际接触系数误差0、外力叠加公式误差0；仅对应有限配方检查 |
| 同初态配对 | 47个标称工况、1128项初态/物理/控制/延迟数组一致，统一accept cohort匹配 |
| 平地运动行为 | **失败、退化**：3000的47/47→4000的40/47，丢低中高度零指令2、前后停车4、原地变高1 |
| 腿摆/漂移 | .26m末1秒腿摆3.527→4.813°，30秒漂移.084→.405m；.30m漂移改善但腿摆.109→3.844° |
| 摩擦/外扰子集 | 两模型各27例，均2/27、同两成功，无新增；全部完整有限，非完整随机化分布验收 |
| 视频 | 3段正常1倍/50fps左右对照，完整解码与模型/轨迹/视频SHA通过，用户最终视频验收待确认 |
| 完整987/起立/真实硬件 | 未执行，不从GPU子集迁移结论 |
| ONNX/板端/固件 | 不适用本次PT归档；未新导出、未修改或写入固件，H7核对待确认 |
| Isaac | 不适用，独立mjlab任务 |

[完整效果报告及视频](../../artifacts/fudan_ppo_walk/20261115-random4000-effect-01/RESULTS.md)。旧模型与原有证据保留。允许用途为实验复现与离线对照；下一步优先保留3000，从仅扩大摩擦的单项对照开始，要求47原成功全部保留及停车不恶化，再逐项增加外扰。本次未启动该新训练。

## 2026-11-15 后续 ONNX 导出与实机记录

交付对象为闭链轮腿复旦 PPO 行走策略的 4000 轮实验模型。训练基线为 `26_wheelleg` 的 `cf5accb82f54a8607f58207a649ba11c2288077d`，checkpoint、配置快照与训练验收见上文。H7 独立部署仓维护分支为 `main`，本次整理时工作区基于 `fb64e2ca4367eaeb90151dd9fde26af3b46b29f6` 并含 Jetson 实控和限幅改动。实测时烧录二进制身份与部署负责人独立复查待确认。

- 导出使用 Windows 项目环境的 PyTorch 2.15.0+cpu 和 ONNX 1.23.2。ONNX 输入为 float32 `obs[1,25]` 和 `obs_history[1,125]`，输出为 `actions[1,6]`。外部 Jetson 走 JRL2 契约 `3000201`，策略 100 Hz、H7 PD 200 Hz。ONNX SHA256 为 `3be34e5244eb729df25ad51ea570ed03c86fa3691d02e3051f506d371311aefa`，与现场引擎元数据一致。[导出包](../../exports/fudan_ppo_walk/20261115_model4000_pd200_full_random_v3/) · [256 组导出检查](../../exports/fudan_ppo_walk/20261115_model4000_pd200_full_random_v3/export-check.json)
- H7 实测策略前后命令达到 ±2.5 m/s，连续直行稳态轮速折算中位数分别为 +2.21 和 −1.84 m/s。32,443 次动作首次用于 PD 的延迟 P99 为 4.961 ms。末段零指令期出现明显姿态异常并退出 RL，整机稳定性仍待验证。原始日志归独立部署仓 `rl_model_logs/25obs-6action-pd2000-4000rounds`。
- 交付范围为实验复现和部署对照。GPU 平地行为退化结论见上文，导出数值、实机采样和时序数据各自记录。
