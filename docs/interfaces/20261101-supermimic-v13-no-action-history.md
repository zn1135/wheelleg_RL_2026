# V13世界模型移除历史动作输入

## 范围与来源

- 日期：2026-11-01；按用户“删除V13的历史输入”执行，仅世界模型的学习输入。
- 机器人/任务：串联轮腿，SuperMimic；真实观测、执行器和物理配方保持。
- 代码基线：`26_wheelleg`，提交`9f5eacd`之后的共享工作区；本次只改`supermimic/motion.py`、`learner.py`及对应检查/文档，保留其他对话已有改动。
- 旧合同：`chuanliantui-supermimic-motion-only-alternating-bptt-v13`。
- 新合同：`chuanliantui-supermimic-motion-only-alternating-bptt-v13-no-action-history-v1`。
- 实现：`motion.WorldModel.inputs/observe/step`；保存恢复：`learner.WORLD_LOSS_RECIPE/ALGORITHM_CONTRACT/ACTOR_TRANSFER_CONTRACTS`。

## 输入差异（索引从0开始）

| 字段 | 旧模型 | 新模型 |
|---|---|---|
| 运动状态 | 0:47 | 0:47 |
| 物理参数/实际延迟 | 47:122 | 47:122 |
| 当前动作 | 122:128 | 122:128 |
| 上一拍动作 | 128:134 | 删除 |
| 上上拍动作 | 134:140 | 删除 |
| 回合年龄，秒 | 140:141 | 128:129 |
| 首层与输入归一化宽度 | 141 | 129 |
| 输出/运动增量 | 5×46 | 5×46，保持 |

控制策略仍为25维当前帧、125维五帧历史、六动作原物理顺序。动作缩放、坐标系、100Hz策略/500Hz物理、真实执行延迟、初态和XML未变。H7无同步代码改动；没有新模型交付或真机通过结论。

历史仍记录在replay，并传给解析延迟PD奖励估计。它不再影响预测运动路径，但在真实参数存在延迟时，可以影响奖励里的估算力矩；没有将这两个职责混为一项输入删除。历史完整物理模型`networks.WorldModel`保持封存语义，当前Learner使用`motion.WorldModel`。

## 兼容性

- 141维旧世界模型及其输入归一化不能直接恢复到129维网络；没有静默删权重列或伪装完整续训。
- `--resume`拒绝旧合同；即使伪造新合同但仍带旧首层，形状检查也会在任何网络/优化器变更前拒绝。
- 旧V13完整actor可单独初始化/推理，数值逐字节保持。新的世界模型、价值网络和优化器按actor-only初始化语义冷启动，不称作原检查点完整续接。
- 原`--init-world`限定的V12来源也因配方/形状不兼容被拒绝。新世界模型训练需另行授权预算。
- 删除历史并不消除真实动作延迟；非零延迟时可能丢失必要状态信息，本轮没有修改延迟去掩盖该问题。

## 验证

```bash
PYTHONPATH=mjlab_training/src CUDA_VISIBLE_DEVICES='' mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_no_history --output artifacts/<新的CPU检查目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_no_history --integration --output artifacts/<新的GPU检查目录>
```

CPU检查已通过：[r1报告](../../artifacts/supermimic/supermimic-v13-no-history-check-20261101-r1/report.json)。覆盖非零网络头的历史扰动不变、历史梯度缺失/当前动作梯度保留、归一化不读历史、世界拟合损失不依赖历史、解析延迟PD历史保留、旧完整包及假新合同错形状拒绝、新包保存恢复和旧actor逐字节一致。优化器更新0。

GPU有限采集检查结果另见[r2报告](../../artifacts/supermimic/supermimic-v13-no-history-check-20261101-r2/report.json)；未训练新权重，未做完整行为、独立种子或真机验收。H7负责人核对不适用本次内部世界模型改动，新策略交付时仍需共同核对。
