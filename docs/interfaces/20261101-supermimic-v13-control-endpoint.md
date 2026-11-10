# V13：每次只预测一个控制步终态

## 确认范围

2026-11-01用户要求将“预测5个物理子步”改回单个控制步，并明确选择：观测和奖励直接从终态计算，不重建2ms传感延迟队列。

本轮基于`26_wheelleg`共享工作区，延续129维航向无关、无动作历史输入。实现修改在`supermimic/motion.py`、`motion_adapter.py`、`learner.py`和`networks.MotionCritic`；其他工作区改动保留，无Git提交推送。

## 定义

| 项目 | 前一版 | 本版 |
|---|---|---|
| 神经网络输入 | 129 | 129，含义保持 |
| 神经网络输出 | 5×46=230 | 46 |
| 每次预测 | 2/4/6/8/10ms五个状态 | 10ms一个终态 |
| 增量归一化标签 | 每个2ms相邻增量 | 本控制步起点→10ms终点增量 |
| 世界模型监督 | 五个物理子步 | 仅10ms终点 |
| 返回path | `[...,5,64]` | `[...,1,64]`；只有终态，不复制、不插值 |
| 想象观测 | 由2ms队列按延迟抽样 | 终态传感量＋噪声，直接生成25维帧 |
| 想象奖励 | 部分几何取8ms、关节取10ms | 所有几何/速度/关节取10ms终态 |
| 想象力矩估算 | 按子步延迟选当前/旧动作 | 终态状态＋当前动作的解析PD估计 |
| 价值网络输入 | 324 | 162（47运动＋18噪声＋1年龄＋75参数＋21上下文） |

47维真实运动状态、46维局部增量的物理含义不变。254维状态作为真实采集兼容容器保留，想象状态中的162维2ms传感队列槽置零；观测和价值网络均不读取这些槽，不将零值冒充真实传感记录。策略自己的125维五帧历史仍按100Hz逐帧更新。

## 真实与想象的边界

真实仿真仍为500Hz物理/PD、100Hz策略，采集每个策略步仍保存5个真实子步和原始回合边界。动作顺序、缩放、XML、执行器、真实延迟、失败判断和H7未改。训练标签取自动复位前的原回合终态，不取复位后的状态。

用户选择的“直接终态”只作用于世界模型想象，不会自动取消真实传感/动作延迟。真实环境的部分奖励特征仍采样于最后物理子步积分前，想象奖励改为终态；这种差异是明确记录的建模选择，尚未验证对训练稳定性或真机表现的影响。

原五子步误差和238ms特征终点等历史评分定义不适用于本版的新总体指标；后续预测评估须以控制终态和实际时长重新对齐，不直接与旧数字混算。

## 合同与兼容

新算法合同：`chuanliantui-supermimic-control-endpoint-heading-invariant-bptt-v13-v3`。
新世界损失配方：`control-endpoint-squared-delta-std-heading-invariant-v4`。
新辅助配方：`control-endpoint-pd-soft-termination-v3`。

旧230输出世界模型、子步增量归一化及324维价值网络不允许完整续训；不能裁剪权重并宣称等价恢复。旧actor/估速器接口仍25/125/6，可单独继承。新世界模型与价值网络尚未训练，需要新预算。没有新部署模型交付，H7同步不适用本次内部模型改动，真机未验收。

## 验证

```bash
PYTHONPATH=mjlab_training/src CUDA_VISIBLE_DEVICES='' mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_control_endpoint --output artifacts/<新终态检查目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_no_history --integration --output artifacts/<新集成检查目录>
PYTHONPATH=mjlab_training/src CUDA_VISIBLE_DEVICES='' mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_heading_inputs --output artifacts/<新坐标检查目录>
```

[终态检查](../../artifacts/supermimic/supermimic-v13-control-endpoint-check-20261101-r1/report.json)：单增量终态精确一致；无中间补值；中间4帧改为NaN不改变损失/归一化；终态改动会改变损失；每条转移只更新1个增量样本；终态观测/奖励、队列置零且不被critic读取、世界和策略想象反向传播有限。

[集成检查](../../artifacts/supermimic/supermimic-v13-control-endpoint-integration-20261101-r1/report.json)与[坐标检查](../../artifacts/supermimic/supermimic-v13-control-endpoint-heading-20261101-r1/report.json)另列。所有检查禁止Adam.step，不代表训练收敛、行为改善或板端通过。
