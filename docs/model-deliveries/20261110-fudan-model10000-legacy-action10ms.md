# 复旦10000轮PT：原物理与动作延迟10ms

2026-11-10按用户要求准备提交10000轮完整PT，附运行来源、校验清单和源码补丁。交付人Codex；接收人及H7独立复查人待确认。仅保存模型，不更改默认选用权重、不导出ONNX、不修改H7或烧录。当前没有10000轮行为验收结论。

## 模型与训练条件

| 项目 | 内容 |
|---|---|
| 机器人/任务 | 本仓闭链轮腿模型，`fudan_ppo_walk`；共享XML与标定限位保持 |
| 完整PT | `artifacts/fudan_ppo_walk/models/20261110_model10000_legacy_action10ms.pt`，1441417字节 |
| PT SHA256 | `2d3ff255b1996a39a3dc1d0d36e51cf68c241117c6563dc9acb852309de30434` |
| 训练预算 | 从9000新增1000轮，到10000自动停止；4096环境×48步/轮、seed1 |
| 实际时间 | 2026-11-10 20:29:40至21:42:10，Asia/Shanghai；训练主循环4311.22秒 |
| 更新与日志 | 新增196608000转移、两Adam各20000次更新；累计1449984000转移、两Adam各200000次更新；16项奖励日志完整 |
| 物理条件 | `legacy-linear-spring-v1`，附加拟合负载关闭；原线性气弹簧/轮模型、闭链与标定限位保持 |
| 延迟/时序 | 动作10ms、传感2ms，策略10ms/物理与PD2ms；主动推扰关闭，无新增持续外力/外矩 |
| 原随机化 | Kp/Kd/力矩0.95～1.05；质量/惯量0.9～1.1、附加质量−1～2kg、COM±20mm、摩擦采样0.6～1.4、零位±0.03rad及原噪声 |
| 训练源 | `logs/mjlab/fudan_ppo_walk/20261104_resume8000_to9000_2ms/model_final.pt`，SHA256 `4ff0c038f26b7aa9d9d9919c97e59cd4e8124f1ddee5eaec1d2944436f8a37a3` |
| 恢复范围 | 网络、两Adam与RNG恢复审计通过；物理、回合和历史现场重新初始化，并非逐轨迹续接 |
| 合同 | `fudan-ppo-walk-physical25-history125-critic144-v1`；v2历史`fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2` |
| 张量 | Actor当前25维＋历史125维→6动作，Critic144维；完整PT含encoder/actor/critic及优化器状态，需配套Python加载，非独立ONNX图 |

该10000权重**不包含**之后准备的动作6～14ms、电机倍率±15%、扩大物理/噪声随机化、短推扰以及每回合±10N/±10N·m持续外载。当前磁盘默认配置已变化，不能用其反推本模型训练条件。

## 可重建来源

训练仓`https://github.com/zn1135/wheelleg_RL_2026.git`，分支`26_wheelleg`，基准HEAD `12d650623681d9740fbb3f684ac2b3318970480b`，运行时工作区有修改；HEAD单独不代表训练版本。部署仓/负责人版本未核对，本次无部署改动或关联PR。

[来源清单](data/20261110-model10000-manifest.json)保存142个实际训练Python源码及资产SHA、源模型、预算、恢复审计摘要和训练条件。其SHA256为`6af5d8eb0e4473b8cfd8b3781adcc829b9b86549fc07e53a4ec35a0e9b2cc48f`。

[训练运行时补丁](data/20261110-model10000-source.patch)仅包含`pd_action.py`与`fudan_ppo_walk_config.py`两处运行时差异；SHA256为`1ba1b6ba1c603a21b28ebc9dbc941c3844de13f35d22348b42c0fa806855aced`。该补丁使用零上下文；在隔离目录以`git apply --unidiff-zero`对基准原文件应用并逐字节核对通过，另外140个训练源码与基准相符。补丁作为交付数据，不覆盖当前下一阶段配置；完整日志快照仍原位保留。

资产XML SHA256 `505c9b9bb2b7a20e15cf9d2cd4ca664332db402687e34185d8a57de173dedbe8`。本次交付读取环境的PyTorch/MuJoCo/Warp版本见清单；训练依赖未另行完整冻结，不能仅据仓库SHA称外部依赖全部可重现。Isaac Gym不适用，本次训练为MuJoCo Warp GPU/RTX4060 Laptop。

## 命令与验证

已完成的命令，原日志与检查点不覆盖：

```bash
scripts/mjlab.sh train --task fudan_ppo_walk --num-envs 4096 --iterations 1000 --resume logs/mjlab/fudan_ppo_walk/20261104_resume8000_to9000_2ms/model_final.pt --preserve-history-v2 --precheck artifacts/fudan_ppo_walk/20261110-legacy-action10ms-resume9000-to10000-01/precheck/report.json --log-dir logs/mjlab/fudan_ppo_walk/20261110_legacy_action10ms_resume9000_to10000_01
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261110_legacy_action10ms_resume9000_to10000_01 --port 6006
```

训练退出0，完成审计退出0。`model_10000.pt`与`model_final.pt`完整学习载荷相同、SHA相同，交付复制逐字节核对通过；未执行额外物理或策略前向。原始证据为`artifacts/fudan_ppo_walk/20261110-legacy-action10ms-resume9000-to10000-01/`内`process-result.json`、`resume-audit.json`及`initial-state-check.json`，完整训练快照/日志在上述run。

| 阶段 | 状态 |
|---|---|
| 同条件20环境GPU初态/有限零动作预检 | 通过；不表示策略平衡/运动能力 |
| 训练轮数、恢复、计数、16项日志及参数有限性审计 | 通过 |
| 10000模型MuJoCo站立/行走/起立与完整987行为验收 | 未执行 |
| ONNX导出/数值比较、板端数值/时序、真机行为 | 未执行 |
| Isaac回放 | 不适用，本次不涉及Isaac |

允许作为已完成训练的检查点用于后续人工评估/另行授权续训；不声明部署通过。此前8000/9000或其他拟合物理模型的行为成绩不能移用于本模型。下一步优先按明确的物理/延迟/外载条件对10000权重做GPU回放，H7负责人独立复查待确认。
