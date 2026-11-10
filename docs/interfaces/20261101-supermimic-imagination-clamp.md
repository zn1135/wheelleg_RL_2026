# SuperMimic：只在策略想象中夹紧预测状态

2026-11-01用户指定实施“想象状态夹紧”。在当前129→46控制步终态模型上加入，无新训练，不修改真实物理、监督标签、动作限幅、学习率或H7。

代码：`motion_adapter.MotionStateAdapter.sanitize_endpoint`定义数值边界；`motion.WorldModel.step(imagination=True)`在生成终态后调用；`Learner.imagination`显式启用。世界模型拟合和普通预测默认`imagination=False`。

## 范围

| 量 | 边界 | 来源与含义 |
|---|---|---|
| 根高度 | 0.02～2m | 参考仓库sanitize的宽松数值边界 |
| 根线速度 | 机体系每分量±20m/s | 使用参考数值；在机体系夹紧后转回世界系，保持整体yaw对称性 |
| 根角速度 | 机体系每分量±40rad/s | 使用参考数值；本机MuJoCo广义角速度已是机体系 |
| 有限位的关节位置 | 编译XML的jnt_range | 仅jnt_limited为真的关节，按实际qpos地址映射 |
| 无限位的关节位置 | 不额外夹紧 | 不套用另一个USD机器人限位 |
| 左右轮速度 | ±150rad/s | 现有PD轮目标上限，作为想象数值保护；不是实测机械速度上限 |
| 其他关节速度 | 不额外夹紧 | 当前XML没有可核实的速度限值；未把旧简化URDF或参考机器人的17/60/300直接移植 |

机身速度的坐标选择与参考实现的世界系逐分量夹紧不同，目的是维持此前确认的航向无关输入/预测等变性。范围内的世界系速度直接保留原值，避免仅旋转往返造成不必要浮点变化。

夹紧发生在观测、奖励、软存活、价值评估和下一拍之前，保证这些量看到同一个受约束状态。没有插值、没有修改真实执行器、没有改真实回合失败条件。

NaN/Inf保持非有限，由原数值/失败检查处理，不转换成看似正常的边界值。范围内梯度保持，超出硬边界的对应梯度为0。逐关节夹紧不等于闭链约束求解，也不保证接触、承重或平衡正确。

## 合同与兼容性

新合同：`chuanliantui-supermimic-control-endpoint-imagination-clamp-bptt-v13-v4`。
新交替配方：`control-endpoint-world4-real-encoder-actor-bptt-critic-clamp-v3`，其中记录完整`imagination_state_clamp`。

网络仍129维输入、46维输出、162维价值输入；边界缓冲随模型保存，XML哈希仍校验。同形状旧配方不能静默完整续训，旧actor单独加载保持。旧日志与模型保留，新版本尚未训练，没有部署交付或H7同步改动。

## 验证

```bash
PYTHONPATH=mjlab_training/src CUDA_VISIBLE_DEVICES='' mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_imagination_clamp --output artifacts/<新的夹紧检查目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_no_history --integration --output artifacts/<新的集成检查目录>
```

[CPU检查](../../artifacts/supermimic/supermimic-v13-imagination-clamp-check-20261101-r1/report.json)：范围内逐字节不变；高度/速度/有限关节限位；未知限位保留；范围内梯度恒等、越界梯度零；非有限不隐藏；平面变换一致；默认预测不夹紧；拟合/归一化不调用sanitize；回放张量不变；每个想象步启用；夹紧后观测/奖励/存活同源。

[集成与保存恢复检查](../../artifacts/supermimic/supermimic-v13-imagination-clamp-integration-20261101-r1/report.json)另列。全部优化器更新0；尚未验证预测误差、真实平衡/运动、跨种子或真机效果。
